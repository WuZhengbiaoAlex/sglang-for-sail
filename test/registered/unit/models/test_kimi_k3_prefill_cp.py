"""CPU regressions for Kimi hybrid CP token/head layout and PP bank transport."""

import unittest
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs import kimi_linear
from sglang.srt.layers.attn_residual import AttnResidual
from sglang.srt.layers.cp import interleave as cp_interleave
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKimiK3PrefillCP(CustomTestCase):
    def parallel(self, **changes):
        args = dict(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=8,
            attn_cp_rank=3,
            tp_size=8,
            tp_rank=3,
            attn_dp_size=1,
            attn_dcp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
        )
        args.update(changes)
        return NS(**args)

    def test_head_shards_and_state_pool_match_tp_without_rank_during_planning(self):
        cfg = kimi_linear.KimiLinearConfig(
            linear_attn_config={
                "num_heads": 96,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": [1],
                "full_attn_layers": list(range(2, 33)),
            },
        )
        parallel = self.parallel()
        with patch.object(kimi_linear, "get_parallel", return_value=parallel):
            self.assertEqual(kimi_linear.kimi_linear_head_parallel(), (8, 3))
            self.assertEqual(cfg.mamba2_cache_params.shape.num_k_heads_per_tp, 12)
            # ModelConfig can ask for state sizes before process ranks exist.
            del parallel.tp_rank
            self.assertEqual(cfg.mamba2_cache_params.shape.num_k_heads_per_tp, 12)
        with patch.object(
            kimi_linear,
            "get_parallel",
            return_value=self.parallel(
                enable_prefill_cp=False, attn_tp_size=2, attn_tp_rank=1
            ),
        ):
            self.assertEqual(kimi_linear.kimi_linear_head_parallel(), (2, 1))
            self.assertEqual(cfg.mamba2_cache_params.shape.num_k_heads_per_tp, 48)
        for override in (
            {"cp_strategy": "zigzag"},
            {"attn_cp_size": 4},
            {"attn_dp_size": 2},
            {"attn_dcp_size": 2},
        ):
            with patch.object(
                kimi_linear, "get_parallel", return_value=self.parallel(**override)
            ):
                with self.assertRaisesRegex(ValueError, "CP=TP"):
                    kimi_linear.kimi_linear_head_parallel()

    @contextmanager
    def cp_inputs(self, rank=3, tokens=16):
        # Exercise the real interleave shard and context-manager restoration.
        strategy = InterleaveCPStrategy(8)
        meta = strategy.build_metadata(
            num_tokens=tokens, seqs_len=[tokens], extend_seqs_len=[tokens]
        )
        batch = NS(attn_cp_metadata=meta, spec_info=None)
        with (
            patch.object(cp_utils, "is_cp_active", return_value=True),
            patch.object(cp_utils, "get_cp_strategy", return_value=strategy),
            patch(
                "sglang.srt.layers.cp.base.get_parallel",
                return_value=self.parallel(attn_cp_rank=rank),
            ),
            patch.object(
                cp_utils, "cp_interleave_input_ids", side_effect=lambda x, b: x[rank::8]
            ),
        ):
            yield batch

    def test_pp_stage_without_embedding_only_shards_ids_and_restores_batch(self):
        ids = torch.arange(16)
        saved = torch.tensor([-7])
        with self.cp_inputs() as batch:
            batch.input_ids_global = saved
            batch.spec_info = NS(hidden_states=torch.ones(16, 2))
            with cp_utils.cp_shard_model_inputs(None, ids, batch, ids) as (
                emb,
                pos,
                local_ids,
            ):
                self.assertIsNone(emb)
                torch.testing.assert_close(pos, ids[3::8])
                torch.testing.assert_close(local_ids, ids[3::8])
                self.assertEqual(batch.spec_info.hidden_states.shape[0], 16)
            self.assertIs(batch.input_ids_global, saved)
            with self.assertRaisesRegex(RuntimeError, "body"):
                with cp_utils.cp_shard_model_inputs(None, ids, batch, ids):
                    raise RuntimeError("body")
            self.assertIs(batch.input_ids_global, saved)

    def test_cp_dense_reduces_aligned_full_rows_then_adds_local_prefix(self):
        full = torch.arange(32, dtype=torch.float32).view(16, 2)
        local = full[3::8]
        prefix = torch.full_like(local, 13)
        model = NS(
            _dp_attention=True,
            _dense_attn_tp=False,
            gate_up_proj=Mock(side_effect=lambda x: (x * 2, None)),
            act_fn=lambda x: x + 1,
            down_proj=Mock(side_effect=lambda x: (x * 3, None)),
        )
        model.forward = MethodType(kimi_k3.KimiK3MLP.forward, model)
        with (
            patch.object(kimi_k3, "is_cp_active", return_value=True),
            patch.object(
                kimi_k3, "cp_gather_full_sequence_states", return_value=full
            ) as gather,
            patch.object(
                kimi_k3, "cp_shard_hidden_states", side_effect=lambda x, b: x[3::8]
            ),
        ):
            out = model.forward(local, prefix_sum=prefix, forward_batch=NS())
        torch.testing.assert_close(out, ((full * 2 + 1) * 3)[3::8] + prefix)
        self.assertIs(model.gate_up_proj.call_args.args[0], full)
        self.assertIs(gather.call_args.args[0], local)
        with patch.object(kimi_k3, "is_cp_active", return_value=False):
            model._dp_attention = False
            out = model.forward(local, forward_batch=NS())
        torch.testing.assert_close(out, (local * 2 + 1) * 3)

    def test_kda_sees_full_causal_sequence_and_preserves_state_between_chunks(self):
        class Recurrence:
            def __init__(self):
                self.state = torch.zeros(1)
                self.seen = []

            def __call__(self, batch, mixed_qkv, a, b):
                self.seen.append(mixed_qkv.clone())
                out = mixed_qkv.cumsum(0) + self.state
                self.state = out[-1].clone()
                return out.view(1, -1, 1, 1)

        recurrence = Recurrence()
        owner = NS(
            _cp_tp_group=True,
            _kda_hip_fused_decode_ready=False,
            _kda_fused_decode_ready=False,
            do_fuse_qkvbfg=False,
            use_full_rank_gate=False,
            head_dim=1,
            all_reduce_fusion=False,
            attn=recurrence,
            o_norm=lambda x, gate: x,
            o_proj=lambda x, skip_all_reduce=False: (
                x / 8 if skip_all_reduce else x,
                None,
            ),
        )
        owner.forward_qkvbfg = lambda x: (
            x,
            torch.zeros_like(x),
            torch.zeros_like(x),
            torch.ones_like(x),
        )
        batch = NS(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=InterleaveCPStrategy(8).build_metadata(16, [16], [16]),
        )
        chunks = [
            torch.arange(1, 17, dtype=torch.float32).view(-1, 1),
            torch.arange(17, 33, dtype=torch.float32).view(-1, 1),
        ]
        outputs = []
        for full in chunks:
            with (
                patch.object(kimi_k3, "is_cp_active", return_value=True),
                patch.object(
                    kimi_k3, "cp_gather_full_sequence_states", return_value=full
                ),
                patch.object(kimi_k3, "get_parallel", return_value=self.parallel()),
                patch.object(
                    cp_interleave, "get_parallel", return_value=self.parallel()
                ),
                patch.object(
                    kimi_k3,
                    "attn_cp_reduce_scatter_tensor",
                    side_effect=lambda out, x: out.copy_(x.view(8, 2, 1)[3] * 8),
                ),
            ):
                outputs.append(
                    kimi_k3.KimiK3DeltaAttention.forward(
                        owner, full[3::8], torch.arange(2), batch, None
                    )
                )
        expected = torch.cat(chunks).cumsum(0)
        torch.testing.assert_close(outputs[0], expected[:16][3::8])
        torch.testing.assert_close(outputs[1], expected[16:][3::8])
        self.assertEqual([len(x) for x in recurrence.seen], [16, 16])

    @staticmethod
    def rank_major_reference(sequence, rank_rows, cp_size=8):
        # Independent token-by-token oracle, including physical alignment beyond
        # ceil(tokens / CP). Padding must not leak into recurrence or residuals.
        out = sequence.new_zeros((rank_rows * cp_size, *sequence.shape[1:]))
        for token in range(sequence.shape[0]):
            out[(token % cp_size) * rank_rows + token // cp_size] = sequence[token]
        return out

    def padded_metadata(self, tokens, rank_rows):
        metadata = InterleaveCPStrategy(8).build_metadata(tokens, [tokens], [tokens])
        metadata.per_rank_logical_token = metadata.per_rank_actual_token.copy()
        metadata.per_rank_actual_token = [rank_rows] * 8
        metadata.max_rank_len = [rank_rows] * 8
        return metadata

    def test_interleave_reorder_round_trip_preserves_padding_and_kv_path(self):
        strategy = InterleaveCPStrategy(8)
        for tokens in (0, 8, 9, 15, 17, 64, 32769):
            # Production pads each physical rank length to a CP multiple.
            rank_rows = ((tokens + 63) // 64) * 8
            sequence = torch.arange(tokens * 3, dtype=torch.float32).view(tokens, 3)
            expected = self.rank_major_reference(sequence, rank_rows)
            metadata = self.padded_metadata(tokens, rank_rows)
            batch = NS(forward_mode=ForwardMode.EXTEND, attn_cp_metadata=metadata)
            for rank in (0, 7):
                parallel = self.parallel(attn_cp_rank=rank, tp_rank=rank)
                parallel.attn_cp_group = object()
                local = expected.view(8, rank_rows, 3)[rank].clone()
                logical = metadata.per_rank_logical_token[rank]
                local[logical:] = -999  # No poisoned physical padding may leak.

                def all_gather(out, input_):
                    torch.testing.assert_close(input_[:logical], local[:logical])
                    self.assertEqual(torch.count_nonzero(input_[logical:]).item(), 0)
                    out.copy_(expected)

                with (
                    self.subTest(tokens=tokens, rank=rank),
                    patch.object(cp_interleave, "get_parallel", return_value=parallel),
                    patch(
                        "sglang.srt.layers.cp.base.get_parallel", return_value=parallel
                    ),
                    patch.object(
                        cp_interleave,
                        "use_symmetric_memory",
                        return_value=nullcontext(),
                    ),
                    patch.object(
                        cp_interleave, "is_allocation_symmetric", return_value=False
                    ),
                    patch.object(
                        cp_interleave,
                        "attn_cp_all_gather_into_tensor",
                        side_effect=all_gather,
                    ),
                ):
                    reordered = cp_interleave.cp_sequence_to_interleave_order(
                        sequence, batch, rank_rows * 8
                    )
                    torch.testing.assert_close(reordered, expected)
                    torch.testing.assert_close(
                        strategy.gather_hidden_states(local, batch), sequence
                    )
                    # DSA KV gather retains its original path and values.
                    torch.testing.assert_close(
                        strategy.gather_kv_cache(local, batch), sequence
                    )

    def test_interleave_tail_gather_index_and_noncp_identity(self):
        sequence = torch.arange(18, dtype=torch.float32).view(9, 2)
        metadata = self.padded_metadata(9, 8)
        metadata.gather_index = torch.tensor([7, 3, 17, 2, 31, 15, 60, 4, 45])
        batch = NS(forward_mode=ForwardMode.EXTEND, attn_cp_metadata=metadata)
        reordered = cp_interleave.cp_sequence_to_interleave_order(sequence, batch, 64)
        expected = torch.zeros(64, 2)
        for token, slot in enumerate(metadata.gather_index.tolist()):
            expected[slot] = sequence[token]
        torch.testing.assert_close(reordered, expected)
        torch.testing.assert_close(reordered[metadata.gather_index], sequence)
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE):
            batch.forward_mode = mode
            self.assertIs(
                cp_interleave.cp_sequence_to_interleave_order(sequence, batch, 64),
                sequence,
            )

    def test_cp_kda_reduce_scatter_matches_sum_then_shard_and_keeps_norm_order(self):
        weight = torch.tensor([[1.0, 2.0, -1.0], [3.0, -1.0, 4.0]])
        for tokens in (8, 9, 17, 65):
            rank_rows = ((tokens + 63) // 64) * 8
            full = torch.arange(tokens * 2, dtype=torch.float32).view(tokens, 2)
            gate = full + 7
            metadata = self.padded_metadata(tokens, rank_rows)
            batch = NS(forward_mode=ForwardMode.EXTEND, attn_cp_metadata=metadata)
            projected = (full + gate) @ weight
            # Every TP rank contributes a distinct partial, so omitting the
            # reduction or selecting a shard before summing cannot pass.
            sum_reference = projected * sum(range(1, 9))
            rank_major_sum = self.rank_major_reference(sum_reference, rank_rows)
            for rank in range(8):
                parallel = self.parallel(attn_cp_rank=rank, tp_rank=rank)
                local = self.rank_major_reference(full, rank_rows).view(
                    8, rank_rows, 2
                )[rank]

                def norm(x, g):
                    torch.testing.assert_close(g[:, 0], gate)
                    torch.testing.assert_close(x[0, :, 0], full)
                    return x + g

                def projection(x, *, skip_all_reduce=False):
                    self.assertTrue(skip_all_reduce)
                    torch.testing.assert_close(
                        x, self.rank_major_reference(full + gate, rank_rows)
                    )
                    return (x @ weight) * (rank + 1), None

                def reduce_scatter(output, partial):
                    torch.testing.assert_close(
                        partial,
                        self.rank_major_reference(projected * (rank + 1), rank_rows),
                    )
                    output.copy_(rank_major_sum.view(8, rank_rows, 3)[rank])

                owner = NS(
                    _cp_tp_group=True,
                    _kda_hip_fused_decode_ready=False,
                    _kda_fused_decode_ready=False,
                    do_fuse_qkvbfg=False,
                    use_full_rank_gate=False,
                    head_dim=2,
                    all_reduce_fusion=False,
                    forward_qkvbfg=lambda x: (x, x[:, :1] * 0, x * 0, gate),
                    attn=lambda batch, mixed_qkv, a, b: mixed_qkv[None, :, None, :],
                    o_norm=norm,
                    o_proj=Mock(side_effect=projection),
                )
                owner.o_proj.reduce_results = True
                with (
                    self.subTest(tokens=tokens, rank=rank),
                    patch.object(kimi_k3, "is_cp_active", return_value=True),
                    patch.object(kimi_k3, "get_parallel", return_value=parallel),
                    patch.object(cp_interleave, "get_parallel", return_value=parallel),
                    patch.object(
                        kimi_k3, "cp_gather_full_sequence_states", return_value=full
                    ),
                    patch.object(
                        kimi_k3,
                        "attn_cp_reduce_scatter_tensor",
                        side_effect=reduce_scatter,
                    ),
                ):
                    output = kimi_k3.KimiK3DeltaAttention.forward(
                        owner, local, torch.arange(rank_rows), batch, None
                    )
                torch.testing.assert_close(
                    output, rank_major_sum.view(8, rank_rows, 3)[rank]
                )
                self.assertTrue(owner.o_proj.reduce_results)
                self.assertEqual(output.shape, (rank_rows, 3))

    def test_cp_inactive_short_extend_and_decode_keep_projection_default_reduce(self):
        values = torch.tensor([[2.0, 3.0]])
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            owner = NS(
                _cp_tp_group=True,
                _kda_hip_fused_decode_ready=False,
                _kda_fused_decode_ready=False,
                do_fuse_qkvbfg=False,
                use_full_rank_gate=False,
                head_dim=2,
                all_reduce_fusion=False,
                forward_qkvbfg=lambda x: (x, x[:, :1] * 0, x * 0, x),
                attn=lambda batch, mixed_qkv, a, b: mixed_qkv[None, :, None, :],
                o_norm=lambda x, g: x,
                o_proj=Mock(side_effect=lambda x: (x * 8, None)),
            )
            owner.o_proj.reduce_results = True
            with (
                patch.object(kimi_k3, "is_cp_active", return_value=False),
                patch.object(
                    kimi_k3,
                    "cp_gather_full_sequence_states",
                    side_effect=AssertionError("inactive gather"),
                ),
                patch.object(
                    kimi_k3,
                    "attn_cp_reduce_scatter_tensor",
                    side_effect=AssertionError("inactive scatter"),
                ),
            ):
                output = kimi_k3.KimiK3DeltaAttention.forward(
                    owner, values, torch.tensor([0]), NS(forward_mode=mode), None
                )
            torch.testing.assert_close(output, values * 8)
            self.assertEqual(owner.o_proj.call_args.kwargs, {})
            self.assertTrue(owner.o_proj.reduce_results)

    def test_pp_keeps_full_bank_local_and_only_last_stage_gathers(self):
        ids = torch.arange(16)
        embeds = torch.arange(32, dtype=torch.float32).view(16, 2)
        local = embeds[3::8]
        bank = torch.arange(16, dtype=torch.float32).view(2, 4, 2)
        proxy = PPProxyTensors({"hidden_states": local, "residual": bank})
        body = Mock(return_value=proxy)
        body.embed_tokens = Mock(return_value=embeds)
        owner = NS(
            pp_group=NS(is_first_rank=True, is_last_rank=False),
            model=body,
            capture_aux_hidden_states=False,
        )
        with self.cp_inputs() as batch:
            with patch.object(
                kimi_k3, "kimi_linear_uses_cp_tp_group", return_value=True
            ):
                got = kimi_k3.KimiK3LinearForCausalLM._forward_cp(
                    owner, ids, ids, batch, None, None
                )
            self.assertIs(got, proxy)
            torch.testing.assert_close(body.call_args.args[3], local)
            self.assertIs(got["residual"], bank)
            # The next PP stage inherits every frozen bank row, not only the
            # current residual stream. It expands capacity for later layers.
            inherited = AttnResidual(local, 8, block_residual=bank)
            self.assertEqual(inherited.num_valid_blocks, 4)
            torch.testing.assert_close(inherited.block_residual[:, :4], bank)
            owner.pp_group = NS(is_first_rank=False, is_last_rank=True)
            body.reset_mock()
            body.return_value = local + 1
            body.embed_tokens = Mock(side_effect=AssertionError("PP1 must not embed"))
            with (
                patch.object(
                    kimi_k3, "kimi_linear_uses_cp_tp_group", return_value=True
                ),
                patch.object(kimi_k3.torch.cuda, "current_stream", return_value=None),
                patch.object(
                    kimi_k3, "cp_gather_after_forward", return_value=embeds + 1
                ) as gather,
            ):
                got = kimi_k3.KimiK3LinearForCausalLM._forward_cp(
                    owner, None, ids, batch, None, proxy
                )
            self.assertIsNone(body.call_args.args[3])
            self.assertIs(body.call_args.args[4], proxy)
            self.assertIs(gather.call_args.args[0], body.return_value)
            torch.testing.assert_close(got, embeds + 1)

    def test_runner_uses_model_boundary_before_generic_embedding_or_logits(self):
        model = NS(
            handles_prefill_cp=True,
            forward=Mock(return_value="logits"),
            get_input_embeddings=Mock(side_effect=AssertionError("generic CP path")),
        )
        owner = NS(model_runner=NS(model=model))
        batch = NS(input_ids=torch.arange(16), positions=torch.arange(16))
        proxy = object()
        result = EagerRunner._execute_extend_cp(
            owner, batch, {"pp_proxy_tensors": proxy}
        )
        self.assertEqual(result, "logits")
        model.forward.assert_called_once_with(
            batch.input_ids, batch.positions, batch, pp_proxy_tensors=proxy
        )
        self.assertTrue(kimi_k3.KimiK3LinearForCausalLM.handles_prefill_cp)
        self.assertTrue(kimi_k3.KimiK3ForConditionalGeneration.handles_prefill_cp)


if __name__ == "__main__":
    unittest.main()
