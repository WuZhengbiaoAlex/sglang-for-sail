"""DFlash2 MLA configuration and accepted-prefix cache contracts on CPU."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3Config

from sglang.srt.arg_groups import speculative_hook
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import reset_context
from sglang.srt.speculative.dflash_utils import (
    dflash_draft_cell_size_per_token,
    get_dflash_layer_attention_params,
    is_dflash_mla_draft_config,
    validate_dflash_mla_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**overrides):
    fields = dict(
        architectures=["DFlash2DraftModel"],
        hidden_size=7168,
        num_hidden_layers=5,
        num_attention_heads=96,
        num_key_value_heads=96,
        head_dim=64,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        is_causal=True,
        layer_types=["sliding_attention"] * 5,
        sliding_window=1024,
        use_sliding_window=True,
        max_window_layers=5,
        dflash_config={
            "attention_mode": "mla",
            "block_size": 5,
            "target_layer_ids": [7, 23, 51, 67, 83, 91],
            "selector_rank": 256,
            "selector_top_k": 32,
        },
    )
    fields.update(overrides)
    return Qwen3Config(**fields)


class TestDFlashMLARuntime(CustomTestCase):
    def tearDown(self):
        reset_context()
        super().tearDown()

    def test_public_mla_geometry_and_exact_cache_cost(self):
        cfg = object.__new__(ModelConfig)
        cfg.hf_config = cfg.hf_text_config = _config()
        self.assertTrue(cfg.hf_config.use_sliding_window)
        self.assertEqual(cfg.hf_config.sliding_window, 1024)
        cfg._derive_model_shapes()
        self.assertTrue(cfg.is_dflash_mla_draft)
        self.assertEqual(cfg.attention_arch, AttentionArch.MLA)
        self.assertEqual(cfg.swa_v_head_dim, 128)
        # Latent+RoPE KV is replicated, not divided by the draft attention TP.
        for tp_size in (1, 8, 32):
            self.assertEqual(
                dflash_draft_cell_size_per_token(
                    draft_model_config=cfg,
                    draft_num_layers=5,
                    draft_kv_cache_dtype=torch.bfloat16,
                    tp_size=tp_size,
                ),
                (512 + 64) * 5 * 2,
            )
        self.assertEqual(
            get_dflash_layer_attention_params(cfg.hf_config, 0),
            (1023, AttentionType.DECODER),
        )

    def test_legacy_gqa_and_invalid_mixed_layouts(self):
        self.assertFalse(is_dflash_mla_draft_config(_config(dflash_config={})))
        with self.assertRaisesRegex(ValueError, "mix mla"):
            is_dflash_mla_draft_config(
                _config(dflash_config={"attention_modes": ["mla"] * 4 + ["gqa"]})
            )
        with self.assertRaisesRegex(ValueError, "attention_bias"):
            validate_dflash_mla_draft_config(_config(attention_bias=True))
        with self.assertRaisesRegex(ValueError, "kv_lora_rank"):
            validate_dflash_mla_draft_config(_config(kv_lora_rank=None))
        gqa = SimpleNamespace(
            attention_arch=AttentionArch.MHA,
            head_dim=64,
            v_head_dim=64,
            get_num_kv_heads=lambda tp: 96 // tp,
        )
        self.assertEqual(
            dflash_draft_cell_size_per_token(
                draft_model_config=gqa,
                draft_num_layers=5,
                draft_kv_cache_dtype=torch.bfloat16,
                tp_size=8,
            ),
            12 * 128 * 5 * 2,
        )

    def test_private_nextn_rewrite_is_preserved(self):
        cfg = object.__new__(ModelConfig)
        cfg.is_draft_model = True
        cfg.speculative_algorithm = "EAGLE"
        cfg.hf_config = SimpleNamespace(
            architectures=["KimiK3ForConditionalGeneration"]
        )
        cfg.hf_text_config = SimpleNamespace(
            num_nextn_predict_layers=1, num_hidden_layers=93
        )
        cfg._config_draft_model()
        self.assertEqual(cfg.hf_config.architectures, ["KimiK31ForCausalLMNextN"])
        self.assertEqual(cfg.hf_config.num_hidden_layers, 1)
        self.assertEqual(cfg.hf_config.mtp_checkpoint_layer, 93)
        self.assertIsNone(cfg.hf_config.linear_attn_config)

    def test_prefix_commit_does_not_overwrite_rejected_slots(self):
        # Only replace the device scatter boundary; exercise production masking
        # against a real tensor store, including a request accepting zero rows.
        store = torch.full((20, 3), -1.0)

        def scatter(layer, loc, nope, rope):
            rows = torch.cat((nope, rope), dim=-1).reshape(-1, 3)
            for index, row in zip(loc.tolist(), rows):
                store[index] = row

        pool = SimpleNamespace(set_mla_kv_buffer=scatter)
        loc = torch.tensor([[3, 4, 5], [8, 9, 10], [12, 13, 14]])
        lens = torch.tensor([1, 3, 0], dtype=torch.int32)
        nope = torch.arange(18, dtype=torch.float32).view(9, 1, 2)
        rope = torch.arange(9, dtype=torch.float32).view(9, 1, 1) + 100
        MLATokenToKVPool.set_mla_kv_buffer_prefix_valid(
            pool, None, loc, lens, nope, rope
        )
        expected = torch.cat((nope, rope), dim=-1).reshape(9, 3)
        for slot, row in ((3, 0), (8, 3), (9, 4), (10, 5)):
            torch.testing.assert_close(store[slot], expected[row])
        for slot in (1, 2, 4, 5, 6, 7, 11, 12, 13, 14, 15):
            torch.testing.assert_close(store[slot], torch.full((3,), -1.0))
        with self.assertRaisesRegex(ValueError, "batch size"):
            MLATokenToKVPool.set_mla_kv_buffer_prefix_valid(
                pool, None, loc, lens[:2], nope, rope
            )

    def test_ppu_inherited_dsa_resolves_independent_triton_draft(self):
        cfg = SimpleNamespace(speculative_draft_attention_backend=None)
        resolved = {}
        with (
            mock.patch.object(speculative_hook, "resolving_view", return_value=cfg),
            mock.patch.object(speculative_hook, "resolved_view", return_value=cfg),
            mock.patch.object(
                speculative_hook, "attention_backends_of", return_value=("dsa", "dsa")
            ),
            mock.patch.object(
                speculative_hook.current_platform, "is_out_of_tree", return_value=False
            ),
            mock.patch("sglang.srt.utils.is_ppu", return_value=True),
            mock.patch.object(
                speculative_hook,
                "declare_resolution",
                side_effect=lambda args, source, **kw: resolved.update(kw),
            ),
        ):
            speculative_hook._resolve_dflash_draft_attention_backend(object())
        self.assertEqual(resolved, {"speculative_draft_attention_backend": "triton"})


if __name__ == "__main__":
    unittest.main()
