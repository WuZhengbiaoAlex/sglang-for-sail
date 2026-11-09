"""CPU regressions for MLA DFlash loading, context KV and DP-local candidates.

The accelerator attention constructor is the only model boundary replaced by
a toy implementation. Projection/loading math and runtime scoping stay real.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models import dflash
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn
from transformers import PretrainedConfig

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(*, mla=True, q_lora_rank=3):
    return PretrainedConfig(
        architectures=["DFlash2DraftModel"],
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=5,
        num_target_layers=93,
        num_attention_heads=2,
        vocab_size=16,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        rms_norm_eps=1e-6,
        max_position_embeddings=1048576,
        rope_theta=50000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        is_causal=True,
        layer_types=["sliding_attention"] * 5,
        sliding_window=1024,
        block_size=5,
        dflash_config={
            "attention_mode": "mla" if mla else "gqa",
            "target_layer_ids": [7, 23, 51, 67, 83, 91],
            "selector_rank": 2,
            "selector_top_k": 2,
            "conv_kernel_size": 2,
            "conv_group_size": 2,
        },
    )


def _load(param, weight):
    with torch.no_grad():
        param.copy_(weight)


class _PairLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


def _toy_mla_init(self, **kwargs):
    nn.Module.__init__(self)
    for key in (
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ):
        setattr(self, key, kwargs[key])
    hidden = kwargs["hidden_size"]
    kv_rows = self.kv_lora_rank + self.qk_rope_head_dim
    if self.q_lora_rank is None:
        self.kv_a_proj_with_mqa = _PairLinear(hidden, kv_rows, bias=False)
    else:
        self.fused_qkv_a_proj_with_mqa = nn.Linear(
            hidden, self.q_lora_rank + kv_rows, bias=False
        )
    self.kv_b_proj = nn.Linear(
        self.kv_lora_rank,
        kwargs["num_heads"] * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
    )
    self.kv_b_proj.weight.weight_loader = _load
    self.kv_a_layernorm = nn.Identity()
    self.rotary_emb = lambda positions, q, k: (q, k + positions[:, None, None])
    self.w_kc = self.w_vc = None
    for name in ("attn_mqa", "attn_mha", "attn_mqa_for_dcp_decode"):
        setattr(self, name, SimpleNamespace())
    # The real base normalizes RoPE in place: the checkpoint config must survive.
    if kwargs["rope_scaling"]:
        kwargs["rope_scaling"]["rope_type"] = "deepseek_yarn"


class TestDFlashMLA(CustomTestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            dflash.DeepseekV2AttentionMLA, "__init__", _toy_mla_init
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_causality_window_and_rope_config_are_preserved(self):
        config = _config()
        attention = dflash.DFlashMLAAttention(config, layer_id=0)
        for name in ("attn_mqa", "attn_mha", "attn_mqa_for_dcp_decode"):
            layer = getattr(attention, name)
            self.assertEqual(layer.attn_type, AttentionType.DECODER)
            self.assertEqual(layer.sliding_window_size, 1023)
        self.assertEqual(config.rope_scaling["rope_type"], "yarn")
        config.is_causal = False
        self.assertEqual(
            dflash.DFlashMLAAttention(config, 0).attn_mqa.attn_type,
            AttentionType.ENCODER_ONLY,
        )

    def test_reject_quantized_draft_weights(self):
        with self.assertRaisesRegex(ValueError, "unquantized"):
            dflash.DFlashMLAAttention(_config(), 0, quant_config=object())

    def test_checkpoint_names_load_split_a_and_absorbed_kv(self):
        attention = dflash.DFlashMLAAttention(_config(), 0)
        model = dflash.DFlashDraftModel.__new__(dflash.DFlashDraftModel)
        nn.Module.__init__(model)
        layer = nn.Module()
        layer.self_attn = attention
        model.layers = nn.ModuleList([layer])
        model.projector_type = None
        model.is_mla_draft = True
        q = torch.arange(12, dtype=torch.float32).view(3, 4)
        kv = torch.arange(16, dtype=torch.float32).view(4, 4) + 20
        b = torch.arange(20, dtype=torch.float32).view(10, 2) + 40
        with torch.no_grad():
            model.load_weights(
                [
                    ("model.layers.0.self_attn.q_a_proj.weight", q),
                    ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", kv),
                    ("model.layers.0.self_attn.kv_b_proj.weight", b),
                ]
            )
        torch.testing.assert_close(
            attention.fused_qkv_a_proj_with_mqa.weight, torch.cat((q, kv))
        )
        torch.testing.assert_close(attention.w_kc, b.view(2, 5, 2)[:, :2])
        torch.testing.assert_close(
            attention.w_vc, b.view(2, 5, 2)[:, 2:].transpose(1, 2)
        )
        old_k, old_v = attention.w_kc.data_ptr(), attention.w_vc.data_ptr()
        with torch.no_grad():
            attention.kv_b_proj.weight.weight_loader(attention.kv_b_proj.weight, b + 1)
        self.assertEqual(attention.w_kc.data_ptr(), old_k)
        self.assertEqual(attention.w_vc.data_ptr(), old_v)
        torch.testing.assert_close(attention.w_kc, (b + 1).view(2, 5, 2)[:, :2])
        with self.assertRaisesRegex(ValueError, "shard 0 expects"):
            attention._load_fused_a_proj_shard(
                attention.fused_qkv_a_proj_with_mqa.weight, kv, 0
            )
        with torch.no_grad(), self.assertRaisesRegex(ValueError, r"shard=1"):
            model.load_weights(
                [
                    ("layers.0.self_attn.q_a_proj.weight", q),
                    ("layers.0.self_attn.kv_b_proj.weight", b),
                ]
            )
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "kv_b_proj.weight"):
            model.load_weights(
                [
                    ("layers.0.self_attn.q_a_proj.weight", q),
                    ("layers.0.self_attn.kv_a_proj_with_mqa.weight", kv),
                ]
            )

    def test_context_projection_uses_kv_rows_with_and_without_q_lora(self):
        hidden = torch.arange(8, dtype=torch.float32).view(2, 4)
        positions = torch.tensor([9, 10])
        weight = torch.arange(16, dtype=torch.float32).view(4, 4) / 16
        expected = hidden @ weight.T
        for q_rank in (None, 3):
            with self.subTest(q_lora_rank=q_rank), torch.no_grad():
                attention = dflash.DFlashMLAAttention(_config(q_lora_rank=q_rank), 0)
                if q_rank is None:
                    attention.kv_a_proj_with_mqa.weight.copy_(weight)
                else:
                    attention.fused_qkv_a_proj_with_mqa.weight[:q_rank].fill_(
                        float("nan")
                    )
                    attention.fused_qkv_a_proj_with_mqa.weight[q_rank:].copy_(weight)
                latent, rope = attention.project_context_kv(positions, hidden)
                torch.testing.assert_close(latent, expected[:, None, :2])
                torch.testing.assert_close(
                    rope, expected[:, None, 2:] + positions[:, None, None]
                )

    def test_forward_restores_enclosing_attention_inputs_on_success_and_error(self):
        attention = dflash.DFlashMLAAttention(_config(), 0)
        hidden = torch.ones(2, 4)
        sentinel = object()

        def forward(_self, **kwargs):
            self.assertIs(get_forward().attn_inputs.hidden_states_local, hidden)
            return kwargs["hidden_states"] + 2

        with get_forward().scoped(attn_inputs=sentinel):
            with mock.patch.object(dflash.DeepseekV2AttentionMLA, "forward", forward):
                actual = attention(torch.arange(2), hidden, SimpleNamespace())
            torch.testing.assert_close(actual, hidden + 2)
            self.assertIs(get_forward().attn_inputs, sentinel)
            with mock.patch.object(
                dflash.DeepseekV2AttentionMLA,
                "forward",
                side_effect=RuntimeError("forward failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    attention(torch.arange(2), hidden, SimpleNamespace())
            self.assertIs(get_forward().attn_inputs, sentinel)

    def test_five_layer_block5_uses_mla_and_six_context_features(self):
        # No accelerator linear/norm construction; keep model layout and selector real.
        with mock.patch.object(
            dflash, "RMSNorm", side_effect=lambda *a, **kw: nn.Identity()
        ), mock.patch.object(
            dflash, "DFlashMLP", side_effect=lambda *a, **kw: nn.Identity()
        ):
            model = dflash.DFlash2DraftModel(_config())
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.block_size, 5)
        self.assertEqual(model.num_context_features, 6)
        self.assertEqual(model.fc.in_features, 24)
        self.assertFalse(model.supports_fused_context_kv)
        for layer in model.layers:
            self.assertIsInstance(layer.self_attn, dflash.DFlashMLAAttention)
            self.assertEqual(layer.attention_conv.block_size, 5)
            self.assertEqual(layer.mlp_conv.block_size, 5)

    def test_gqa_decoder_selection_is_unchanged(self):
        with mock.patch.object(
            dflash, "RMSNorm", side_effect=lambda *a, **kw: nn.Identity()
        ), mock.patch.object(
            dflash, "DFlashMLP", side_effect=lambda *a, **kw: nn.Identity()
        ), mock.patch.object(
            dflash.DFlashAttention,
            "__init__",
            lambda self, **kw: nn.Module.__init__(self),
        ):
            layer = dflash.DFlashDecoderLayer(_config(mla=False), 0)
        self.assertIsInstance(layer.self_attn, dflash.DFlashAttention)
        self.assertNotIsInstance(layer.self_attn, dflash.DFlashMLAAttention)


class TestDFlashCandidateDP(CustomTestCase):
    def test_dp_head_candidates_and_partition_use_only_attention_tp(self):
        hidden = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 1.0], [-2.0, 4.0]])
        head = SimpleNamespace(
            weight=weight[2:],
            use_attn_tp_group=True,
            shard_indices=SimpleNamespace(num_org_elements=2, org_vocab_start_index=2),
        )
        peer_logits = hidden @ weight[:2].T
        peer_vals, peer_ids = torch.topk(peer_logits, 2, dim=-1)

        def gather(x, dim):
            if not x.is_floating_point():
                peer = peer_ids
            elif x.shape[-1] == 1:
                peer = torch.logsumexp(peer_logits, -1, keepdim=True)
            else:
                peer = peer_vals
            return torch.cat((peer, x), dim=dim)

        def local_topk_lse(logits, k):
            vals, ids = torch.topk(logits, k, dim=-1)
            return vals, ids, torch.logsumexp(logits, dim=-1)

        forbidden = mock.Mock(side_effect=AssertionError("cross-DP gather"))
        with get_parallel().override(
            tp_group=SimpleNamespace(world_size=8, all_gather=forbidden),
            attn_tp_group=SimpleNamespace(world_size=2, all_gather=gather),
        ), mock.patch.object(dflash, "lilicorr_topk_lse", local_topk_lse):
            ids, vals, lse = dflash.candidate_topk(hidden, head, 2, with_partition=True)
        logits = hidden @ weight.T
        expected_vals, expected_ids = torch.topk(logits, 2, dim=-1)
        torch.testing.assert_close(ids, expected_ids)
        torch.testing.assert_close(vals, expected_vals)
        torch.testing.assert_close(lse, torch.logsumexp(logits, -1))
        forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
