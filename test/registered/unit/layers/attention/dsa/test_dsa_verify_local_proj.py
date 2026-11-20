"""Contracts for moving replicated Kimi VERIFY Q and head-weight projections after sharding."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.indexer_request_sharding import (
    IndexerRequestShardMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_shard(rank, actual):
    ctx = torch.zeros(32, 5, dtype=torch.int32)
    ctx[:actual] = 61 + torch.arange(actual)[:, None] * 71 + torch.arange(5)
    return IndexerRequestShardMetadata(
        32,
        5,
        rank,
        8,
        ctx[rank * 4 : rank * 4 + 4],
        torch.empty(0),
        torch.empty(0),
    )


class TestDSAVerifyLocalProj(unittest.TestCase):
    def test_verify_projects_owned_q_and_head_weights_but_writes_full_k(self):
        for rank, actual, enabled in (
            (0, 32, True),
            (3, 25, True),
            (7, 25, True),
            (3, 32, False),
        ):
            with self.subTest(rank=rank, actual=actual, enabled=enabled):
                # Include DP tail padding beyond the 32-request metadata bucket.
                rows = 165
                x = (torch.arange(rows) + 1000).float().view(-1, 1)
                q_lora = x + 1000
                positions = torch.arange(rows)
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    attn_cp_metadata=None,
                    seq_lens=torch.ones(32),
                )
                shard = make_shard(rank, actual)
                metadata = SimpleNamespace(indexer_request_shard=shard)
                idx = object.__new__(Indexer)
                torch.nn.Module.__init__(idx)
                idx.use_verify_local_proj = enabled
                idx.use_fp8 = True
                idx.alt_stream = None
                idx.dsa_enable_prefill_cp = idx.use_dsa_indexer_fusion = False
                idx.use_fp4 = idx.bf16_indexer = idx.use_int8 = False
                idx.wq_b, idx.weights_proj = Mock(set_lora=False), Mock(set_lora=False)
                idx._should_skip_logits_computation = Mock(return_value=False)
                idx._aiter_fused_fp8_active = Mock(return_value=False)
                idx._get_q_k_bf16 = Mock(
                    side_effect=lambda q, x, *args, **kwargs: (q, x, None)
                )
                idx._get_logits_head_gate = Mock(side_effect=lambda x, scale: x * scale)
                idx._store_index_k_cache = Mock()
                idx._get_topk_paged = Mock(return_value=torch.empty(rows, 1))
                backend = SimpleNamespace(get_indexer_metadata=lambda *args: metadata)
                with (
                    patch.object(dsa_indexer, "_is_cuda", True),
                    patch.object(dsa_indexer, "_is_hip", False),
                    patch.object(dsa_indexer, "_is_npu", False),
                    patch.object(dsa_indexer, "get_attn_backend", return_value=backend),
                    patch.object(
                        dsa_indexer,
                        "_is_in_piecewise_or_breakable_cuda_graph",
                        return_value=False,
                    ),
                    patch.object(
                        dsa_indexer, "is_graph_dsa_split_op_surface", return_value=False
                    ),
                    patch.object(
                        dsa_indexer,
                        "maybe_capture_indexer_topk",
                        side_effect=lambda _, value: value,
                    ),
                    patch(
                        "sglang.kernels.ops.attention.dsa.triton_kernel.act_quant",
                        side_effect=lambda q, *_: (q, torch.ones_like(q)),
                    ),
                ):
                    idx.block_size, idx.scale_fmt = 128, "ue8m0"
                    result = idx.forward_cuda(x, q_lora, positions, batch, 0)

                owned = (
                    slice(shard.query_start, shard.query_end)
                    if enabled
                    else slice(None)
                )
                q_in, k_in, pos_in, _ = idx._get_q_k_bf16.call_args.args
                torch.testing.assert_close(q_in, q_lora[owned])
                self.assertIs(k_in, x)
                self.assertIs(pos_in, positions)
                self.assertIs(idx._store_index_k_cache.call_args.kwargs["key"], x)
                torch.testing.assert_close(
                    idx._get_logits_head_gate.call_args.args[0], x[owned]
                )
                _, _, q_out, weights, _ = idx._get_topk_paged.call_args.args
                torch.testing.assert_close(q_out, q_lora[owned])
                torch.testing.assert_close(weights, x[owned])
                self.assertEqual(
                    idx._get_topk_paged.call_args.kwargs.get("full_query_rows"),
                    rows if enabled else None,
                )
                self.assertEqual(result.shape[0], rows)

    def test_local_logits_inputs_keep_per_rank_pages_and_full_padded_output(self):
        rows, slots_per_page = 165, 64
        q = torch.arange(rows).reshape(-1, 1, 1)
        weights = torch.arange(rows).reshape(-1, 1) + 100
        logical = torch.arange(160, dtype=torch.int32).remainder(127).reshape(-1, 1)
        logical = torch.cat([logical, logical.new_full((160, 1), -1)], dim=1)
        actual = 25
        logical[actual * 5 :] = -1
        idx = SimpleNamespace(index_topk=2, _mask_init_and_local_tokens=Mock())
        for rank in (0, 3, 7):
            shard = make_shard(rank, actual)
            owned = slice(shard.query_start, shard.query_end)
            table = torch.arange(320, dtype=torch.int32).reshape(160, 2) + rank * 320

            def gather(out, local):
                torch.testing.assert_close(local, logical[owned])
                out.copy_(logical)

            group = SimpleNamespace(
                world_size=8, rank_in_group=rank, all_gather_into_tensor=gather
            )

            def native(
                fn, q_in, cache, w_in, lens, page_rows, schedule, maxlen, **kwargs
            ):
                torch.testing.assert_close(q_in, q[owned])
                torch.testing.assert_close(w_in, weights[owned])
                torch.testing.assert_close(page_rows, table[owned])
                self.assertIs(lens, shard.ctx_lens_2d)
                return torch.empty(20, 128)

            expected = torch.full((rows, 2), -1, dtype=torch.int32)
            active = torch.arange(actual * 5)
            offsets = logical[active, 0]
            expected[active, 0] = (
                table[active, offsets // slots_per_page] * slots_per_page
                + offsets % slots_per_page
            )
            with (
                patch.object(
                    dsa_indexer,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_tp_group=group),
                ),
                patch.object(
                    dsa_indexer, "deepgemm_paged_mqa_logits_native", side_effect=native
                ),
                patch.object(
                    dsa_indexer,
                    "_topk_transform_v2_logical",
                    return_value=logical[owned],
                ),
            ):
                for presharded in (False, True):
                    with self.subTest(rank=rank, presharded=presharded):
                        actual_slots = Indexer._get_topk_paged_tp_sharded(
                            idx,
                            q[owned] if presharded else q,
                            torch.empty(0),
                            weights[owned] if presharded else weights,
                            table,
                            shard,
                            paged_mqa_logits=Mock(),
                            max_seq_len=128,
                            full_query_rows=rows if presharded else None,
                        )
                        torch.testing.assert_close(actual_slots, expected)

    def test_no_local_projection_for_unsharded_mode_cp_lora_or_tuple_input(self):
        x = torch.empty(160, 4)
        shard = make_shard(2, 29)
        meta = SimpleNamespace(indexer_request_shard=shard)
        idx = SimpleNamespace(
            use_verify_local_proj=True,
            wq_b=SimpleNamespace(),
            weights_proj=SimpleNamespace(),
        )
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, attn_cp_metadata=None
        )
        get = lambda m, h: Indexer._get_verify_projection_shard(idx, fb, m, h)
        self.assertIs(get(meta, x), shard)
        self.assertIsNone(get(None, x))
        self.assertIsNone(get(SimpleNamespace(), x))
        self.assertIsNone(get(meta, (x, x)))
        for name in ("wq_b", "weights_proj"):
            with (
                self.subTest(name=name),
                patch.object(getattr(idx, name), "set_lora", True, create=True),
            ):
                self.assertIsNone(get(meta, x))
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.EXTEND,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode), patch.object(fb, "forward_mode", mode):
                self.assertIsNone(get(meta, x))
        with patch.object(fb, "attn_cp_metadata", object()):
            self.assertIsNone(get(meta, x))


if __name__ == "__main__":
    unittest.main()
