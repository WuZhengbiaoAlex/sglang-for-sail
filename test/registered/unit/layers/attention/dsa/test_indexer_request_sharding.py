"""Request ownership and logical/physical TopK contracts, without a server."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    _topk_transform_v2_logical,
)
from sglang.srt.layers.attention.dsa.indexer_request_sharding import (
    IndexerRequestShardMetadata,
    map_logical_topk_to_physical,
)
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIndexerRequestSharding(CustomTestCase):
    @staticmethod
    def backend():
        backend = object.__new__(DeepseekSparseAttnBackend)
        for name, value in dict(
            indexer_request_sharding_enabled=True,
            ppu_fp8_indexer=True,
            indexer_num_heads=32,
            indexer_head_dim=128,
            real_page_size=64,
            dsa_index_kpool=1,
            dsa_index_topk=2048,
            use_fused_topk=True,
            hisparse_coordinator=None,
            speculative_num_draft_tokens=5,
            dsa_topk_backend=SimpleNamespace(should_use_topk_v2=lambda: True),
            req_to_token=torch.empty((1, 131072), dtype=torch.int32),
        ).items():
            setattr(backend, name, value)
        return backend

    @staticmethod
    def shard(batch, rank, ctx):
        local_batch = batch // 8
        return IndexerRequestShardMetadata(
            full_batch_size=batch,
            next_n=5,
            tp_rank=rank,
            tp_size=8,
            ctx_lens_2d=ctx[rank * local_batch : (rank + 1) * local_batch],
            schedule_metadata=torch.empty((8, 2), dtype=torch.int32),
            topk_v2_plan=torch.empty((local_batch * 5 + 1, 2), dtype=torch.int32),
        )

    def test_whole_requests_and_distinct_causal_rows_survive_rank_concatenation(self):
        for batch in (16, 32, 64):
            with self.subTest(batch=batch):
                # Unique request/query labels expose splitting or reordering the
                # five causal rows, unlike a constant tensor fixture.
                labels = torch.arange(batch)[:, None] * 100 + torch.arange(5)
                ctx = (labels + 4096).to(torch.int32)
                restored = []
                for rank in range(8):
                    shard = self.shard(batch, rank, ctx)
                    self.assertEqual(shard.local_batch_size, batch // 8)
                    self.assertEqual(shard.query_start, rank * (batch // 8) * 5)
                    self.assertEqual(shard.query_end, (rank + 1) * (batch // 8) * 5)
                    rows = labels.flatten()[shard.query_start : shard.query_end]
                    torch.testing.assert_close(
                        rows.reshape(-1, 5) % 100,
                        torch.arange(5).expand(batch // 8, -1),
                    )
                    torch.testing.assert_close(
                        shard.ctx_lens_2d.flatten(), (rows + 4096).to(torch.int32)
                    )
                    restored.append(rows)
                torch.testing.assert_close(torch.cat(restored), labels.flatten())

    def test_capture_bucket_ownership_does_not_shift_when_actual_batch_shrinks(self):
        for bucket, actual in (
            (16, 1),
            (16, 7),
            (16, 8),
            (16, 9),
            (32, 25),
            (32, 31),
            (32, 32),
            (64, 33),
        ):
            with self.subTest(bucket=bucket, actual=actual):
                ctx = torch.zeros((bucket, 5), dtype=torch.int32)
                ctx[:actual] = (
                    1000 + torch.arange(actual)[:, None] * 64 + torch.arange(5)
                ).to(torch.int32)
                local = [self.shard(bucket, rank, ctx) for rank in range(8)]
                restored = torch.cat([part.ctx_lens_2d for part in local])
                torch.testing.assert_close(restored, ctx)
                self.assertEqual(
                    sum(int((part.ctx_lens_2d > 0).sum()) for part in local), actual * 5
                )
                self.assertTrue(
                    torch.equal(restored[actual:], torch.zeros_like(restored[actual:]))
                )
                self.assertEqual(
                    [part.query_start for part in local],
                    [rank * (bucket // 8) * 5 for rank in range(8)],
                )

    def test_real_padding_producer_uses_request_count_not_length_one(self):
        expanded = torch.arange(1, 6, dtype=torch.int32).repeat(32)
        original = expanded.clone()
        for actual in (0, 1, 7, 8, 9, 25, 29, 31, 32):
            pieces = [
                DeepseekSparseAttnBackend._indexer_shard_context_lengths(
                    expanded, 32, 5, rank, 8, 32 - actual
                )
                for rank in range(8)
            ]
            expected = expanded.reshape(32, 5).clone()
            expected[actual:] = 0
            torch.testing.assert_close(torch.cat(pieces), expected)
        torch.testing.assert_close(expanded, original)
        with self.assertRaises(ValueError):
            DeepseekSparseAttnBackend._indexer_shard_context_lengths(
                expanded, 32, 5, 0, 8, 33
            )

    def test_static_gate_falls_back_uniformly_for_unsupported_geometry(self):
        parallel = SimpleNamespace(
            attn_tp_size=8, attn_tp_rank=0, pp_size=1, attn_cp_size=1, attn_dcp_size=1
        )
        execution = SimpleNamespace(
            kernel=SimpleNamespace(dsa_paged_mqa_logits_backend="deepgemm")
        )
        module = "sglang.srt.layers.attention.dsa_backend"
        with (
            patch(module + "._is_ppu", True),
            patch(module + ".get_parallel", return_value=parallel),
            patch(module + ".get_exec", return_value=execution),
        ):
            for batch in (16, 32, 64):
                self.assertTrue(
                    self.backend()._use_indexer_request_sharding(
                        ForwardMode.TARGET_VERIFY, batch
                    )
                )
            for batch in (0, 1, 7, 8, 9, 25, 31, 33):
                self.assertIsNone(
                    self.backend()._build_indexer_request_shard(
                        ForwardMode.TARGET_VERIFY,
                        batch,
                        torch.empty(batch * 5, dtype=torch.int32),
                    )
                )
            for name, value in (
                ("indexer_request_sharding_enabled", False),
                ("ppu_fp8_indexer", False),
                ("indexer_num_heads", 12),
                ("indexer_head_dim", 64),
                ("real_page_size", 128),
                ("dsa_index_kpool", 2),
                ("dsa_index_topk", 4096),
                ("use_fused_topk", False),
                ("hisparse_coordinator", object()),
            ):
                obj = self.backend()
                setattr(obj, name, value)
                self.assertFalse(
                    obj._use_indexer_request_sharding(ForwardMode.TARGET_VERIFY, 32),
                    name,
                )
            self.assertFalse(
                self.backend()._use_indexer_request_sharding(ForwardMode.DECODE, 32)
            )
            self.assertFalse(
                self.backend()._use_indexer_request_sharding(ForwardMode.EXTEND, 32)
            )
            for name, value in (
                ("attn_tp_size", 4),
                ("pp_size", 2),
                ("attn_cp_size", 2),
                ("attn_dcp_size", 2),
            ):
                old = getattr(parallel, name)
                setattr(parallel, name, value)
                self.assertFalse(
                    self.backend()._use_indexer_request_sharding(
                        ForwardMode.TARGET_VERIFY, 32
                    ),
                    name,
                )
                setattr(parallel, name, old)
            execution.kernel.dsa_paged_mqa_logits_backend = "cutedsl"
            self.assertFalse(
                self.backend()._use_indexer_request_sharding(
                    ForwardMode.TARGET_VERIFY, 32
                )
            )

    def test_capture_refresh_preserves_local_pointers_and_full_metadata(self):
        backend = self.backend()
        parallel = SimpleNamespace(
            attn_tp_size=8, attn_tp_rank=7, pp_size=1, attn_cp_size=1, attn_dcp_size=1
        )
        execution = SimpleNamespace(
            kernel=SimpleNamespace(dsa_paged_mqa_logits_backend="deepgemm")
        )

        def schedule(ctx):
            # An opaque native schedule surrogate that changes capacity at
            # maximum context. Tests the real producer's reserve/copy behavior.
            values = torch.tensor(
                [[int(ctx.max()), int(ctx.sum())], [ctx.shape[0], ctx.shape[1]]],
                dtype=torch.int32,
            )
            return values.repeat(2, 1) if int(ctx.max()) > 65536 else values

        def plan(lengths):
            result = torch.zeros((len(lengths) + 1, 2), dtype=torch.int32)
            result[1:, 0] = lengths.cumsum(0).int()
            result[1:, 1] = lengths
            return result

        backend._build_paged_mqa_schedule_metadata = schedule
        backend._build_topk_v2_plan = plan
        expanded = (torch.arange(160) + 1000).int()
        original = expanded.clone()
        module = "sglang.srt.layers.attention.dsa_backend"
        with (
            patch(module + "._is_ppu", True),
            patch(module + ".get_parallel", return_value=parallel),
            patch(module + ".get_exec", return_value=execution),
        ):
            shard = backend._build_indexer_request_shard(
                ForwardMode.TARGET_VERIFY, 32, expanded, in_capture=True, num_padding=3
            )
        metadata = SimpleNamespace(
            indexer_request_shard=shard, dsa_seqlens_expanded=expanded
        )
        tracked = (shard.ctx_lens_2d, shard.schedule_metadata, shard.topk_v2_plan)
        pointers = [t.data_ptr() for t in tracked]
        self.assertEqual(shard.schedule_metadata.shape, (4, 2))
        for padding in (3, 31, 0):
            backend._refresh_indexer_request_shard(metadata, num_padding=padding)
            expected = expanded.reshape(32, 5)[28:32].clone()
            expected[max(0, 32 - padding - 28) :] = 0
            torch.testing.assert_close(shard.ctx_lens_2d, expected)
            torch.testing.assert_close(shard.topk_v2_plan, plan(expected.flatten()))
            torch.testing.assert_close(shard.schedule_metadata[:2], schedule(expected))
            self.assertTrue(bool((shard.schedule_metadata[2:] == 0).all()))
            self.assertEqual(pointers, [t.data_ptr() for t in tracked])
        torch.testing.assert_close(expanded, original)

    def test_production_method_uses_attention_group_and_restores_hidden_padding(self):
        rank, batch, width, topk = 3, 16, 128, 3
        ctx = torch.full((batch, 5), width, dtype=torch.int32)
        shard = self.shard(batch, rank, ctx)
        q = torch.arange((batch * 5 + 4) * 32 * 128).reshape(-1, 32, 128).float()
        weights = torch.ones((q.shape[0], 32))
        pages = torch.arange(batch * 5 * 2, dtype=torch.int32).reshape(-1, 2)
        logical = torch.tensor([0, 64, -1], dtype=torch.int32).repeat(batch * 5, 1)
        indexer = object.__new__(Indexer)
        indexer.index_topk = topk
        indexer.num_init_tokens = indexer.num_local_tokens = 0
        indexer._get_paged_mqa_logits_dispatch = object()
        selected_dispatch = object()

        def logits_fn(
            dispatch,
            local_q,
            kv,
            local_weights,
            local_ctx,
            local_pages,
            schedule,
            max_len,
            *,
            q_offset,
            B,
            next_n,
        ):
            self.assertIs(dispatch, selected_dispatch)
            torch.testing.assert_close(local_q, q[shard.query_start : shard.query_end])
            torch.testing.assert_close(
                local_pages, pages[shard.query_start : shard.query_end]
            )
            self.assertEqual((q_offset, B, next_n), (10, 2, 5))
            return torch.zeros((q_offset, width))

        def gather(out, local):
            torch.testing.assert_close(
                local, logical[shard.query_start : shard.query_end]
            )
            out.copy_(logical)

        group = SimpleNamespace(
            world_size=8, rank_in_group=rank, all_gather_into_tensor=gather
        )
        trap = SimpleNamespace(
            all_gather_into_tensor=lambda *_: self.fail("world/EP collective used")
        )
        parallel = SimpleNamespace(attn_tp_group=group, tp_group=trap, ep_group=trap)
        module = "sglang.srt.layers.attention.dsa.dsa_indexer"
        with (
            patch(module + ".get_parallel", return_value=parallel),
            patch(module + ".deepgemm_paged_mqa_logits_native", side_effect=logits_fn),
            patch(
                module + "._topk_transform_v2_logical",
                return_value=logical[shard.query_start : shard.query_end].clone(),
            ),
        ):
            out = indexer._get_topk_paged_tp_sharded(
                q,
                torch.empty(0),
                weights,
                pages,
                shard,
                paged_mqa_logits=selected_dispatch,
                max_seq_len=width,
            )
        expected = torch.stack(
            (
                pages[:, 0] * 64,
                pages[:, 1] * 64,
                torch.full((batch * 5,), -1, dtype=torch.int32),
            ),
            dim=1,
        )
        torch.testing.assert_close(out[: batch * 5], expected)
        self.assertEqual(tuple(out.shape), (batch * 5 + 4, topk))
        self.assertTrue(bool((out[batch * 5 :] == -1).all()))

    def test_verify_shard_retains_selected_bf16_logits_callable(self):
        """The outer dispatcher must not drop its dtype choice at the TP branch."""
        batch, next_n = 32, 5
        rows = batch * next_n
        ctx = torch.full((batch, next_n), 128, dtype=torch.int32)
        shard = self.shard(batch, 0, ctx)
        pages = torch.zeros((rows, 2), dtype=torch.int32)
        metadata = SimpleNamespace(
            get_page_table_64=lambda: pages,
            get_seqlens_expanded=lambda: ctx.flatten(),
            get_seqlens_int32=lambda: ctx[:, 0],
            get_dsa_extend_len_cpu=lambda: [1] * rows,
            uses_paged_topk_v2=lambda n, k: n == rows and k == 3,
            paged_mqa_schedule_metadata=torch.empty((8, 2), dtype=torch.int32),
            paged_mqa_ctx_lens_2d=ctx,
            indexer_request_shard=shard,
            force_unfused_topk=False,
        )
        indexer = object.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.use_fp4 = indexer.bf16_indexer = indexer.use_int8 = False
        indexer.use_fp8 = True
        indexer.index_topk = 3
        indexer.paged_mqa_logits_backend = SimpleNamespace(is_cutedsl=lambda: False)
        indexer._get_index_k_read_buffer = lambda *_: torch.zeros((1, 64 * 132))
        indexer._get_paged_mqa_logits_dispatch = lambda *a, **kw: torch.zeros(
            (20, 128), dtype=torch.float32
        )

        def native(*args, logits_dtype=torch.float32, **kwargs):
            return torch.zeros((20, 128), dtype=logits_dtype)

        def sharded(
            q,
            kv,
            weights,
            actual_pages,
            actual_shard,
            *,
            paged_mqa_logits,
            max_seq_len,
            page_size,
        ):
            self.assertIs(actual_shard, shard)
            self.assertIs(actual_pages, pages)
            scores = paged_mqa_logits(q, kv, weights)
            # Surface the callable's output dtype in the branch result, rather
            # than merely checking that a mock received some callable.
            return torch.full(
                (rows, 3), int(scores.dtype == torch.bfloat16), dtype=torch.int32
            )

        module = "sglang.srt.layers.attention.dsa.dsa_indexer"
        with (
            patch(module + "._is_cuda", True),
            patch(module + "._is_hip", False),
            patch(module + "._is_xpu", False),
            patch(module + ".is_ppu", return_value=True),
            patch(
                module + ".get_token_to_kv_pool",
                return_value=SimpleNamespace(page_size=64),
            ),
            patch(
                module + ".deep_gemm",
                SimpleNamespace(fp8_paged_mqa_logits=native),
                create=True,
            ),
            patch.object(indexer, "_get_topk_paged_tp_sharded", side_effect=sharded),
        ):
            for enabled in (False, True):
                indexer.use_bf16_paged_logits = enabled
                out = indexer._get_topk_paged(
                    SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
                    0,
                    torch.zeros((rows, 32, 128)),
                    torch.ones((rows, 32, 1)),
                    metadata,
                )
                torch.testing.assert_close(
                    out, torch.full((rows, 3), int(enabled), dtype=torch.int32)
                )

    def test_ppu_precomputed_hook_passes_padding_and_restores_metadata_on_error(self):
        from sglang.srt.hardware_backend.ppu.attention.ppu_dsa_hooks import (
            _ppu_dsa_init_forward_metadata_replay_cuda_graph_from_precomputed as hook,
        )

        saved = object()
        metadata = SimpleNamespace(flashmla_metadata=saved)
        backend = object()
        for padding in (None, 31):

            def original(actual_self, batch, actual_metadata, mode, *, num_padding):
                self.assertIs(actual_self, backend)
                self.assertIs(actual_metadata, metadata)
                self.assertIsNone(actual_metadata.flashmla_metadata)
                self.assertEqual(num_padding, padding)
                return "forwarded"

            self.assertEqual(
                hook(
                    original,
                    backend,
                    32,
                    metadata,
                    ForwardMode.TARGET_VERIFY,
                    num_padding=padding,
                ),
                "forwarded",
            )
            self.assertIs(metadata.flashmla_metadata, saved)

        def fail(*args, **kwargs):
            raise RuntimeError("original replay failed")

        with self.assertRaisesRegex(RuntimeError, "original replay failed"):
            hook(fail, backend, 32, metadata, ForwardMode.TARGET_VERIFY, num_padding=31)
        self.assertIs(metadata.flashmla_metadata, saved)

    def test_page_mapping_uses_each_receivers_physical_table_and_preserves_minus_one(
        self,
    ):
        logical = torch.tensor(
            [[0, 63, 64, 127, 128, -1], [128, 64, -1, 0, 63, 127]], dtype=torch.int32
        )
        tables = [
            torch.tensor([[9, 3, 7], [4, 8, 1]], dtype=torch.int32),
            torch.tensor([[2, 7, 6], [9, 0, 3]], dtype=torch.int32),
        ]
        original = logical.clone()
        outputs = []
        for table in tables:
            output = map_logical_topk_to_physical(logical, table, page_size=64)
            expected = torch.tensor(
                [
                    [
                        int(table[row, p // 64]) * 64 + p % 64 if p >= 0 else -1
                        for p in values
                    ]
                    for row, values in enumerate(logical.tolist())
                ],
                dtype=torch.int32,
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assertEqual(output.dtype, torch.int32)
            self.assertTrue(
                torch.equal(
                    output[logical == -1], torch.full_like(output[logical == -1], -1)
                )
            )
            outputs.append(output)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        torch.testing.assert_close(logical, original)

    def test_all_padding_and_strided_page_rows_are_safe(self):
        # Unowned requests must not turn -1 into the last physical page.
        table = torch.arange(24, dtype=torch.int32).reshape(4, 6)[:, ::2]
        logical = torch.full((4, 7), -1, dtype=torch.int32)
        output = map_logical_topk_to_physical(logical, table, page_size=64)
        torch.testing.assert_close(output, logical, rtol=0, atol=0)
        empty = map_logical_topk_to_physical(logical[:0], table[:0], page_size=64)
        self.assertEqual(tuple(empty.shape), (0, 7))

    def test_logical_topk_requests_native_logical_mode_and_retains_padding(self):
        logits = torch.tensor(
            [[10.0, 50.0, 30.0, 20.0], [99.0, 98.0, 97.0, 96.0], [5.0, 8.0, 6.0, 7.0]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([2, 0, 4], dtype=torch.int32)
        plan = torch.zeros((4, 2), dtype=torch.int32)

        def select(
            scores, lens, page_tables, out, page_size, metadata, out_raw_indices=None
        ):
            self.assertIsNone(page_tables)
            self.assertIsNone(out_raw_indices)
            self.assertIs(metadata, plan)
            self.assertEqual(page_size, 64)
            out.fill_(-1)
            for row, length in enumerate(lens.tolist()):
                take = min(length, out.shape[1])
                if take:
                    out[row, :take] = scores[row, :length].topk(take).indices.int()

        with patch(
            "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2",
            side_effect=select,
        ):
            actual = _topk_transform_v2_logical(logits, lengths, 3, plan)
        expected = torch.tensor(
            [[1, 0, -1], [-1, -1, -1], [1, 3, 2]], dtype=torch.int32
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_logical_topk_rejects_global_or_stale_shape_plan(self):
        # B32's full plan [161,2] must not serve its local20-query shard.
        scores = torch.zeros((20, 128), dtype=torch.float32)
        lengths = torch.full((20,), 128, dtype=torch.int32)
        for shape in ((161, 2), (20, 2), (21, 1)):
            with self.subTest(plan_shape=shape), self.assertRaises(AssertionError):
                _topk_transform_v2_logical(
                    scores, lengths, 16, torch.zeros(shape, dtype=torch.int32)
                )

    def test_bf16_logical_topk_observes_refreshed_lengths_without_rebinding(self):
        # This checks the graph metadata contract on CPU, not GPU graph capture:
        # the same length/plan tensors are updated in place between invocations.
        scores = torch.tensor(
            [[0, 5, 2, 7, 4, 1, 6, 3], [7, 6, 5, 4, 3, 2, 1, 0]],
            dtype=torch.bfloat16,
        )
        lengths = torch.tensor([4, 0], dtype=torch.int32)
        plan = torch.zeros((3, 2), dtype=torch.int32)
        pointers = lengths.data_ptr(), plan.data_ptr()

        def select(logits, lens, pages, out, page_size):
            self.assertEqual(logits.dtype, torch.bfloat16)
            self.assertIs(lens, lengths)
            self.assertIsNone(pages)
            self.assertEqual(page_size, 64)
            out.fill_(-1)
            for row, length in enumerate(lens.tolist()):
                take = min(length, out.shape[1])
                if take:
                    out[row, :take] = (
                        logits[row, :length].float().topk(take).indices.int()
                    )

        with (
            patch(
                "sglang.srt.layers.attention.dsa.dsa_topk_backend.is_ppu",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2",
                side_effect=AssertionError("BF16 must not enter FP32 kernel"),
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2_bf16",
                side_effect=select,
            ),
        ):
            first = _topk_transform_v2_logical(scores, lengths, 3, plan)
            lengths.copy_(torch.tensor([0, 2], dtype=torch.int32))
            plan[1:, 1].copy_(lengths)
            second = _topk_transform_v2_logical(scores, lengths, 3, plan)
        self.assertEqual(pointers, (lengths.data_ptr(), plan.data_ptr()))
        torch.testing.assert_close(
            first, torch.tensor([[3, 1, 2], [-1, -1, -1]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            second, torch.tensor([[-1, -1, -1], [0, 1, -1]], dtype=torch.int32)
        )

    def test_bf16_logical_topk_requires_ppu_and_sixteen_byte_row_alignment(self):
        lengths = torch.tensor([4, 4], dtype=torch.int32)
        plan = torch.zeros((3, 2), dtype=torch.int32)
        module = "sglang.srt.layers.attention.dsa.dsa_topk_backend"
        for ppu, width in ((False, 8), (True, 4)):
            with self.subTest(ppu=ppu, width=width), patch(
                module + ".is_ppu", return_value=ppu
            ), self.assertRaises(AssertionError):
                _topk_transform_v2_logical(
                    torch.zeros((2, width), dtype=torch.bfloat16), lengths, 3, plan
                )

    def test_logical_topk_rejects_invalid_abi_without_dispatch(self):
        scores = torch.zeros((2, 8), dtype=torch.float32)
        lengths = torch.tensor([4, 8], dtype=torch.int32)
        plan = torch.zeros((3, 2), dtype=torch.int32)
        cases = [
            (scores.half(), lengths, 2, plan, 64),
            (scores[:, ::2], lengths, 2, plan, 64),
            (torch.zeros((2, 9)), lengths, 2, plan, 64),
            (scores, lengths.long(), 2, plan, 64),
            (scores, lengths[:1], 2, plan, 64),
            (scores, lengths, 2, plan.long(), 64),
            (scores, lengths, 0, plan, 64),
            (scores, lengths, 2049, plan, 64),
            (scores, lengths, 2, plan, 0),
            (scores, lengths, 2, plan, 63),
        ]
        with patch(
            "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2",
            side_effect=AssertionError("native must not be reached"),
        ) as native:
            for args in cases:
                with self.subTest(
                    shape=tuple(args[0].shape), topk=args[2], page=args[4]
                ):
                    with self.assertRaises(AssertionError):
                        _topk_transform_v2_logical(*args)
            native.assert_not_called()

    def test_zero_query_abi_returns_empty_without_native_dispatch(self):
        with patch(
            "sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2"
        ) as native:
            out = _topk_transform_v2_logical(
                torch.empty((0, 128), dtype=torch.float32),
                torch.empty((0,), dtype=torch.int32),
                2048,
                torch.zeros((1, 2), dtype=torch.int32),
            )
        self.assertEqual(tuple(out.shape), (0, 2048))
        self.assertEqual(out.dtype, torch.int32)
        native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
