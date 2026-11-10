import unittest
from contextlib import contextmanager
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _restore_trtllm_decode_dp_padding,
    _trim_trtllm_decode_dp_padding,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSABackendDPPadding(CustomTestCase):
    def test_occupied_uneven_decode_with_idle_rank_selects_max_len(self):
        with (
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
            patch(
                "sglang.srt.layers.dp_attention.dp_gather_width",
                return_value=4,
            ),
        ):
            mode = DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=False,
                global_num_tokens=[8, 7, 6, 0],
            )
        self.assertEqual(mode, DpPaddingMode.MAX_LEN)

    def test_trim_and_restore_active_eager_decode_padding(self):
        q_all = torch.arange(8 * 2 * 3).view(8, 2, 3)
        topk_indices = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)

        real_q, real_topk, num_padding_rows = _trim_trtllm_decode_dp_padding(
            q_all,
            topk_indices,
            real_batch_size=6,
        )

        self.assertTrue(torch.equal(real_q, q_all[:6]))
        self.assertTrue(torch.equal(real_topk, topk_indices[:6]))
        self.assertEqual(num_padding_rows, 2)

        real_output = torch.ones((6, 1, 2, 5), dtype=torch.bfloat16)
        output = _restore_trtllm_decode_dp_padding(real_output, num_padding_rows)
        self.assertEqual(output.shape, (8, 1, 2, 5))
        self.assertTrue(torch.equal(output[:6], real_output))
        self.assertTrue(torch.all(output[6:] == 0))

    def test_no_padding_preserves_existing_tensors(self):
        q_all = torch.empty((2, 2, 3))
        topk_indices = torch.empty((2, 8), dtype=torch.int32)

        real_q, real_topk, num_padding_rows = _trim_trtllm_decode_dp_padding(
            q_all,
            topk_indices,
            real_batch_size=2,
        )

        self.assertIs(real_q, q_all)
        self.assertIs(real_topk, topk_indices)
        self.assertEqual(num_padding_rows, 0)
        self.assertIs(
            _restore_trtllm_decode_dp_padding(real_q, num_padding_rows),
            real_q,
        )

    def test_rejects_metadata_larger_than_physical_batch(self):
        with self.assertRaisesRegex(
            AssertionError, "metadata batch size \\(3\\) exceeds q batch size \\(2\\)"
        ):
            _trim_trtllm_decode_dp_padding(
                torch.empty((2, 2, 3)),
                torch.empty((2, 8), dtype=torch.int32),
                real_batch_size=3,
            )

    def test_trtllm_decode_runs_real_rows_then_restores_physical_batch(self):
        metadata = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([8, 12], dtype=torch.int32),
            page_table_1=torch.zeros((2, 12), dtype=torch.int32),
            max_seq_len_k=12,
        )
        backend = SimpleNamespace(
            forward_metadata=metadata,
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _layer_id: torch.zeros((24, 3))
            ),
            real_page_size=1,
            kv_cache_dim=3,
            use_fused_topk=False,
            qk_nope_head_dim=2,
            kv_lora_rank=2,
            qk_rope_head_dim=1,
            workspace_buffer=None,
            dsa_index_topk=2,
            _multi_ctas_kv_counter_buffer=None,
            device="cpu",
            num_q_heads=2,
        )
        backend._pad_topk_indices = MethodType(
            DeepseekSparseAttnBackend._pad_topk_indices, backend
        )
        backend._pad_trtllm_sparse_page_table = MethodType(
            DeepseekSparseAttnBackend._pad_trtllm_sparse_page_table, backend
        )
        backend._multi_ctas_kv_counter_for = MethodType(
            DeepseekSparseAttnBackend._multi_ctas_kv_counter_for, backend
        )

        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=2,
            head_dim=3,
            k_scale_float=None,
            scaling=1.0,
        )
        forward_batch = SimpleNamespace()
        q = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
        topk_indices = torch.arange(4 * 2, dtype=torch.int32).view(4, 2)

        flashinfer = ModuleType("flashinfer")
        flashinfer_decode = ModuleType("flashinfer.decode")
        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)
            return torch.ones((2, 1, 2, 2), dtype=torch.bfloat16)

        flashinfer_decode.trtllm_batch_decode_with_kv_cache_mla = fake_decode
        flashinfer.decode = flashinfer_decode

        def fake_transform(*, page_table, topk_indices, page_size):
            self.assertEqual(page_table.shape[0], 2)
            self.assertEqual(topk_indices.shape[0], 2)
            self.assertEqual(page_size, 1)
            return torch.zeros((2, 2), dtype=torch.int32)

        with (
            patch.dict(
                "sys.modules",
                {
                    "flashinfer": flashinfer,
                    "flashinfer.decode": flashinfer_decode,
                },
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.transform_index_page_table_decode",
                side_effect=fake_transform,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.attention.dsa_backend.grow_multi_ctas_kv_counter_buffer_if_needed",
                return_value=None,
            ),
        ):
            output = DeepseekSparseAttnBackend._forward_trtllm(
                backend,
                q=q,
                k=torch.empty((4, 1, 2)),
                v=torch.empty((4, 1, 2)),
                layer=layer,
                forward_batch=forward_batch,
                seq_lens=metadata.cache_seqlens_int32,
                save_kv_cache=False,
                topk_indices=topk_indices,
            )

        self.assertEqual(captured["query"].shape, (2, 1, 2, 3))
        self.assertEqual(output.shape, (4, 1, 2, 2))
        self.assertTrue(torch.all(output[:2] == 1))
        self.assertTrue(torch.all(output[2:] == 0))


class TestDSAEmptyDPMetadata(CustomTestCase):
    def make_backend(self):
        # Keep metadata construction real; replace only accelerator planners
        # and hardware queries. No model or KV pool allocation is needed.
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.device = "cpu"
        backend.real_page_size = 64
        backend.speculative_num_draft_tokens = 5
        backend.dsa_index_topk = 4
        backend.dsa_index_kpool = 1
        backend.dsa_prefill_impl = "flashmla_sparse_q8"
        backend.dsa_decode_impl = "flashmla_sparse_q8"
        backend.dsa_kv_cache_store_fp8 = True
        backend.enable_auto_select_prefill_impl = False
        backend.supports_mha_one_shot = False
        backend.use_mha = True
        backend._is_in_breakable_cuda_graph = lambda: False
        backend._is_in_tc_piecewise_cuda_graph = lambda: False
        backend._get_device_sm = lambda: 80
        backend._is_blackwell = lambda: False
        backend.token_to_kv_pool = SimpleNamespace(
            dtype=torch.bfloat16, size=128, page_size=64
        )
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(3 * 16, dtype=torch.int32).view(3, 16)
        )
        backend._arange_buf = torch.arange(32, dtype=torch.int32)
        backend.forward_metadata = object()  # A preceding nonempty iteration.
        backend._build_topk_v2_plan = Mock(return_value=torch.tensor([[7, 8]]))
        backend._build_paged_mqa_schedule_metadata = Mock(
            return_value=torch.tensor([[9, 10]])
        )
        backend._compute_flashmla_metadata = Mock()
        return backend

    def make_batch(self, mode, seq_lens, extend_lens=None, *, host_lens=True):
        lengths = torch.tensor(seq_lens, dtype=torch.int64)
        extend_lens = extend_lens if extend_lens is not None else []
        return SimpleNamespace(
            batch_size=len(seq_lens),
            forward_mode=mode,
            seq_lens=lengths,
            seq_lens_cpu=lengths.clone() if host_lens else None,
            seq_lens_sum=sum(seq_lens),
            req_pool_indices=torch.arange(len(seq_lens), dtype=torch.int64),
            extend_seq_lens_cpu=extend_lens,
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
            extend_prefix_lens_cpu=[
                kv - q for kv, q in zip(seq_lens, extend_lens, strict=False)
            ],
            extend_num_tokens=sum(extend_lens),
            global_num_tokens_cpu=None,
        )

    def test_empty_idle_extend_and_verify_skip_all_native_planning(self):
        for mode in (
            ForwardMode.IDLE,
            ForwardMode.EXTEND,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            for host_lens in (True, False):
                with self.subTest(mode=mode, host_lens=host_lens):
                    backend = self.make_backend()
                    batch = self.make_batch(mode, [], host_lens=host_lens)
                    # Other DP groups are busy; physical token padding does
                    # not create logical attention requests on this rank.
                    batch.global_num_tokens_cpu = [5, 0, 0, 0]
                    backend.init_forward_metadata(batch)
                    self.assertIsNone(backend.forward_metadata)
                    self.assertFalse(backend.use_mha)
                    self.assertEqual(backend.dsa_prefill_impl, "flashmla_sparse_q8")
                    backend._build_topk_v2_plan.assert_not_called()
                    backend._build_paged_mqa_schedule_metadata.assert_not_called()
                    backend._compute_flashmla_metadata.assert_not_called()

    def test_nonempty_decode_rebuilds_after_idle_with_or_without_host_lengths(self):
        for host_lens in (True, False):
            with self.subTest(host_lens=host_lens):
                backend = self.make_backend()
                batch = self.make_batch(ForwardMode.DECODE, [6, 9], host_lens=host_lens)
                with (
                    get_parallel().override(attn_cp_size=1),
                    patch(
                        "sglang.srt.layers.attention.dsa_backend.is_cuda",
                        return_value=True,
                    ),
                ):
                    backend.init_forward_metadata(batch)
                    previous = backend.forward_metadata
                    backend.init_forward_metadata(self.make_batch(ForwardMode.IDLE, []))
                    backend.init_forward_metadata(batch)
                metadata = backend.forward_metadata
                self.assertIsNot(metadata, previous)
                self.assertEqual(metadata.max_seq_len_k, 9)
                self.assertEqual(metadata.max_seq_len_q, 1)
                self.assertEqual(metadata.cache_seqlens_int32.tolist(), [6, 9])
                self.assertEqual(metadata.cu_seqlens_k.tolist(), [0, 6, 15])
                self.assertEqual(metadata.cu_seqlens_q.tolist(), [0, 1, 2])
                self.assertEqual(metadata.dsa_cache_seqlens_int32.tolist(), [4, 4])
                self.assertEqual(metadata.page_table_1.shape, (2, 9))
                self.assertEqual(metadata.paged_mqa_ctx_lens_2d.tolist(), [[6], [9]])
                self.assertIs(
                    metadata.topk_v2_plan, backend._build_topk_v2_plan.return_value
                )
                self.assertEqual(backend._build_topk_v2_plan.call_count, 2)
                self.assertEqual(
                    backend._build_paged_mqa_schedule_metadata.call_count, 2
                )
                self.assertEqual(
                    backend._build_topk_v2_plan.call_args.args[0].tolist(), [6, 9]
                )

    def test_nonempty_q8_extend_preserves_ragged_prefix_metadata(self):
        backend = self.make_backend()
        batch = self.make_batch(ForwardMode.EXTEND, [6, 9], [2, 3])
        with (
            get_parallel().override(attn_cp_size=1),
            patch("sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsa_backend.can_dsa_prefill_cp_interleave",
                return_value=False,
            ),
        ):
            backend.init_forward_metadata(batch)
        metadata = backend.forward_metadata
        self.assertFalse(backend.use_mha)
        self.assertEqual(backend.dsa_prefill_impl, "flashmla_sparse_q8")
        self.assertEqual(metadata.max_seq_len_q, 3)
        self.assertEqual(metadata.max_seq_len_k, 9)
        self.assertEqual(metadata.cu_seqlens_q.tolist(), [0, 2, 5])
        self.assertEqual(metadata.dsa_seqlens_expanded.tolist(), [5, 6, 7, 8, 9])
        self.assertEqual(metadata.topk_indices_offset.tolist(), [0, 0, 6, 6, 6])
        self.assertEqual(
            metadata.page_table_1_flattened.tolist(),
            list(range(6)) + list(range(16, 25)),
        )
        self.assertEqual(metadata.indexer_k_start_end[0].tolist(), [0, 0, 6, 6, 6])
        self.assertEqual(metadata.indexer_k_start_end[1].tolist(), [5, 6, 13, 14, 15])
        self.assertEqual(metadata.token_to_batch_idx.tolist(), [0, 0, 1, 1, 1])
        self.assertIs(metadata.topk_v2_plan, backend._build_topk_v2_plan.return_value)
        self.assertEqual(
            backend._build_topk_v2_plan.call_args.args[0].tolist(), [5, 6, 7, 8, 9]
        )
        backend._build_paged_mqa_schedule_metadata.assert_not_called()


class TestPPUGroupedVerifyMetadata(CustomTestCase):
    """Exercise real metadata orchestration with CPU substitutes for GPU kernels."""

    module = "sglang.srt.layers.attention.dsa_backend"

    @staticmethod
    def expand_lengths(extend_lens, cache_lens, total_len, max_extend_len):
        return torch.cat(
            [
                torch.arange(int(end - count + 1), int(end + 1), dtype=torch.int32)
                for count, end in zip(extend_lens, cache_lens, strict=True)
            ]
        ).clamp_min(0)

    @staticmethod
    def topk_plan(lengths):
        return torch.cat((lengths, lengths.new_zeros(1))).view(-1, 1)

    def make_backend(self, bs=1):
        backend = TestDSAEmptyDPMetadata.make_backend(self)
        backend.ppu_fp8_indexer = True
        backend.indexer_num_heads = 32
        backend.indexer_head_dim = 128
        backend.num_q_heads = 12
        backend.use_fused_topk = False
        backend.hisparse_coordinator = None
        backend.dsa_drop_wide_page_table = False
        backend.experimental_kpool_metadata_fusion = False
        backend.req_to_token = torch.arange(bs * 128, dtype=torch.int32).view(bs, 128)
        backend.req_to_token_pool.req_to_token = backend.req_to_token
        backend._build_paged_mqa_schedule_metadata = MethodType(
            DeepseekSparseAttnBackend._build_paged_mqa_schedule_metadata, backend
        )
        backend._build_topk_v2_plan = Mock(side_effect=self.topk_plan)
        backend.decode_cuda_graph_metadata = {
            "page_table": torch.zeros((bs * 5, 128), dtype=torch.int32),
            "cu_seqlens_q": torch.arange(bs * 5 + 1, dtype=torch.int32),
        }
        return backend

    @contextmanager
    def cpu_planners(self, *, ppu=True, sm100=False):
        planner = Mock(side_effect=lambda ctx, *args, **kwargs: ctx.clone())
        fake_dg = SimpleNamespace(
            get_paged_mqa_logits_metadata=planner, get_num_sms=lambda: 39
        )
        with (
            get_parallel().override(attn_cp_size=1),
            patch(f"{self.module}._is_ppu", ppu),
            patch(f"{self.module}._is_hip", False),
            patch(f"{self.module}.is_cuda", return_value=True),
            patch(
                f"{self.module}.get_platform",
                return_value=SimpleNamespace(is_sm100=sm100),
            ),
            patch(f"{self.module}.deep_gemm", fake_dg, create=True),
            patch(
                f"{self.module}.seqlens_expand_triton", side_effect=self.expand_lengths
            ),
            patch(
                "sglang.kernels.ops.attention.dsv4.topk.plan_topk_v2",
                side_effect=self.topk_plan,
            ),
        ):
            yield planner

    def test_grouped_builder_and_eager_keep_each_query_causal(self):
        for bs in (1, 32):
            with self.subTest(bs=bs), self.cpu_planners() as planner:
                backend = self.make_backend(bs)
                seq = (torch.arange(bs) % 5 + 10).tolist()
                batch = TestDSAEmptyDPMetadata.make_batch(
                    self, ForwardMode.TARGET_VERIFY, seq
                )
                backend.init_forward_metadata(batch)
                metadata = backend.forward_metadata
                expected = torch.tensor(seq, dtype=torch.int32)[:, None] + torch.arange(
                    1, 6, dtype=torch.int32
                )
                torch.testing.assert_close(metadata.paged_mqa_ctx_lens_2d, expected)
                torch.testing.assert_close(
                    metadata.dsa_seqlens_expanded, expected.flatten()
                )
                self.assertEqual(
                    planner.call_args.kwargs["metadata_extra"], (5, 32, 128, 1)
                )
                self.assertEqual(metadata.page_table_1.shape[0], bs * 5)
                self.assertEqual(metadata.real_page_table.shape[0], bs * 5)
                self.assertEqual(metadata.topk_v2_plan.shape[0], bs * 5 + 1)
                torch.testing.assert_close(
                    metadata.topk_v2_plan[:-1, 0], expected.flatten()
                )

    def test_grouping_guards_and_non_fp8_metadata_remain_split(self):
        for field, value in (
            ("ppu_fp8_indexer", False),  # BF16, INT8 and FP4 all resolve false.
            ("indexer_num_heads", 16),
            ("indexer_head_dim", 64),
            ("real_page_size", 128),
            ("dsa_index_kpool", 2),
            ("speculative_num_draft_tokens", 1),
            ("speculative_num_draft_tokens", 7),
        ):
            with self.subTest(field=field), self.cpu_planners(sm100=True):
                backend = self.make_backend()
                setattr(backend, field, value)
                width = backend.speculative_num_draft_tokens
                expanded = torch.arange(11, 11 + width, dtype=torch.int32)
                ctx = backend._build_paged_mqa_schedule_2d_ctx_lens(
                    ForwardMode.TARGET_VERIFY, expanded[-1:], expanded, 1
                )
                self.assertEqual(ctx.shape, (width, 1))
                torch.testing.assert_close(ctx[:, 0], expanded)
        for bf16, elem in ((False, 1), (True, 2)):
            with self.subTest(
                bf16=bf16
            ), self.cpu_planners(), envs.SGLANG_SAIL_BF16_INDEXER.override(bf16):
                backend = self.make_backend()
                backend.ppu_fp8_indexer = False
                self.assertEqual(
                    backend._paged_mqa_schedule_metadata_extra(
                        torch.ones((5, 1), dtype=torch.int32)
                    ),
                    (1, 12, 128, elem),
                )

    def test_other_forward_modes_and_sm100_keep_existing_layout(self):
        backend = self.make_backend()
        cache = torch.tensor([15], dtype=torch.int32)
        expanded = torch.arange(11, 16, dtype=torch.int32)
        with self.cpu_planners():
            for mode in (
                ForwardMode.DECODE,
                ForwardMode.IDLE,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                ctx = backend._build_paged_mqa_schedule_2d_ctx_lens(
                    mode, cache, expanded, 1
                )
                self.assertEqual(ctx.shape[1], 1)
            for width in (2, 6):
                backend.speculative_num_draft_tokens = width
                ctx = backend._build_paged_mqa_schedule_2d_ctx_lens(
                    ForwardMode.TARGET_VERIFY, cache, expanded.new_ones(width), 1
                )
                self.assertEqual(ctx.shape, (1, width))
        backend.speculative_num_draft_tokens = 5
        with self.cpu_planners(ppu=False, sm100=True):
            ctx = backend._build_paged_mqa_schedule_2d_ctx_lens(
                ForwardMode.TARGET_VERIFY, cache, expanded, 1
            )
            torch.testing.assert_close(ctx, torch.full((1, 5), 15, dtype=torch.int32))

    def capture(self, backend, seq):
        bs = seq.numel()
        backend._build_forward_metadata_cuda_graph(
            bs, bs * 5, torch.arange(bs), seq, seq, ForwardMode.TARGET_VERIFY, None
        )
        metadata = backend.forward_metadata
        self.assertEqual(metadata.paged_mqa_ctx_lens_2d.shape, (bs, 5))
        self.assertTrue(torch.all(metadata.paged_mqa_ctx_lens_2d == 128))
        return metadata

    @staticmethod
    def pointers(metadata):
        return tuple(
            getattr(metadata, name).data_ptr()
            for name in (
                "paged_mqa_ctx_lens_2d",
                "paged_mqa_schedule_metadata",
                "dsa_seqlens_expanded",
                "page_table_1",
                "real_page_table",
                "topk_v2_plan",
            )
        )

    def assert_replay(self, metadata, seq, pointers):
        expected = seq.to(torch.int32)[:, None] + torch.arange(1, 6, dtype=torch.int32)
        self.assertEqual(self.pointers(metadata), pointers)
        torch.testing.assert_close(metadata.paged_mqa_ctx_lens_2d, expected)
        torch.testing.assert_close(metadata.dsa_seqlens_expanded, expected.flatten())
        torch.testing.assert_close(metadata.topk_v2_plan[:-1, 0], expected.flatten())
        self.assertEqual(metadata.real_page_table.shape[0], seq.numel() * 5)

    def test_capture_and_fused_or_ordinary_replay_preserve_addresses(self):
        for fused in (False, True):
            with self.subTest(fused=fused), self.cpu_planners() as planner:
                backend = self.make_backend(2)
                seq = torch.tensor([10, 20], dtype=torch.int32)
                metadata = self.capture(backend, seq)
                pointers = self.pointers(metadata)

                def fake_fused(**kw):
                    # The GPU kernel only produces the original per-query
                    # metadata; production refresh must fill the grouped ctx.
                    self.assertIsNone(kw["paged_mqa_ctx_lens_2d"])
                    values = kw["seq_lens"][:, None] + torch.arange(
                        1, 6, dtype=torch.int32
                    )
                    kw["cache_seqlens"].copy_(values[:, -1])
                    kw["seqlens_expanded"].copy_(values.flatten())
                    kw["dsa_cache_seqlens"].copy_(values.flatten().clamp(max=4))
                    pages = kw["req_to_token"][
                        kw["req_pool_indices"]
                    ].repeat_interleave(5, 0)
                    kw["page_table_1"].copy_(pages)
                    kw["real_page_table"].copy_(
                        backend._transform_table_1_to_real(pages)
                    )

                backend._fused_verify_metadata = Mock(side_effect=fake_fused)
                for delta in (3, 7):
                    replay_seq = seq + delta
                    # Skip only the fused-kernel eligibility check for the
                    # ordinary fallback. The later schedule refresh stays CUDA.
                    with patch(f"{self.module}.is_cuda", side_effect=[fused, True]):
                        backend._apply_cuda_graph_metadata(
                            2,
                            torch.arange(2),
                            replay_seq,
                            replay_seq,
                            ForwardMode.TARGET_VERIFY,
                            None,
                        )
                    self.assert_replay(metadata, replay_seq, pointers)
                    self.assertEqual(
                        planner.call_args.kwargs["metadata_extra"], (5, 32, 128, 1)
                    )
                self.assertEqual(
                    backend._fused_verify_metadata.call_count, 2 if fused else 0
                )

    def test_precomputed_replay_refreshes_grouped_schedule_in_place(self):
        with self.cpu_planners() as planner:
            backend = self.make_backend(2)
            seq = torch.tensor([10, 20], dtype=torch.int32)
            metadata = self.capture(backend, seq)
            pointers = self.pointers(metadata)
            for delta in (2, 9):
                replay_seq = seq + delta
                expanded = (
                    replay_seq[:, None] + torch.arange(1, 6, dtype=torch.int32)
                ).flatten()
                dsa_lens = expanded.clamp(max=4)
                pages = backend.req_to_token.repeat_interleave(5, 0)
                precomputed = SimpleNamespace(
                    cache_seqlens=replay_seq + 5,
                    cu_seqlens_k=torch.cat(
                        (seq.new_zeros(1), (replay_seq + 5).cumsum(0))
                    ),
                    page_indices=pages,
                    max_seqlen_k=128,
                    max_len=128,
                    dsa_cache_seqlens=dsa_lens,
                    seqlens_expanded=expanded,
                    dsa_cu_seqlens_k=torch.cat((seq.new_zeros(1), dsa_lens.cumsum(0))),
                    seqlens_expanded_size=10,
                    flashmla_metadata=None,
                    real_page_table=backend._transform_table_1_to_real(pages),
                )
                # Use the real individual-copy fallback, without trying to
                # launch the CUDA fused-copy kernel on CPU tensors.
                with patch(
                    "sglang.srt.layers.attention.dsa.dsa_metadata_manager._is_hip", True
                ):
                    backend.init_forward_metadata_replay_cuda_graph_from_precomputed(
                        2, precomputed, ForwardMode.TARGET_VERIFY
                    )
                self.assert_replay(metadata, replay_seq, pointers)
                self.assertEqual(
                    planner.call_args.kwargs["metadata_extra"], (5, 32, 128, 1)
                )

    def test_indexer_missing_schedule_uses_grouped_width_and_actual_heads(self):
        from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

        module = "sglang.srt.layers.attention.dsa.dsa_indexer"
        for bs in (1, 32):
            with self.subTest(bs=bs):
                rows, width = bs * 5, 128
                ctx = torch.arange(11, 11 + rows, dtype=torch.int32).view(bs, 5)
                pages = (
                    torch.arange(bs * 2, dtype=torch.int32)
                    .view(bs, 2)
                    .repeat_interleave(5, 0)
                )
                metadata = SimpleNamespace(
                    get_page_table_64=lambda: pages,
                    get_seqlens_expanded=lambda: ctx.flatten(),
                    get_seqlens_int32=lambda: ctx[:, -1],
                    get_dsa_extend_len_cpu=lambda: [5] * bs,
                    paged_mqa_schedule_metadata=None,
                    paged_mqa_ctx_lens_2d=ctx,
                    topk_transform=Mock(
                        return_value=torch.zeros((rows, 4), dtype=torch.int32)
                    ),
                )
                native = Mock(return_value=torch.zeros((rows, width)))
                indexer = SimpleNamespace(
                    use_fp4=False,
                    bf16_indexer=False,
                    sm_count=39,
                    index_topk=4,
                    paged_mqa_logits_backend=SimpleNamespace(
                        is_cutedsl=lambda: False, is_aiter=lambda: False
                    ),
                    _get_index_k_read_buffer=lambda *args: torch.empty(
                        (bs * 2, 64 * 132), dtype=torch.uint8
                    ),
                    _get_paged_mqa_logits_dispatch=native,
                    _mask_init_and_local_tokens=Mock(),
                )
                planner = Mock(return_value=torch.zeros((bs, 5), dtype=torch.int32))
                fake_dg = SimpleNamespace(get_paged_mqa_logits_metadata=planner)
                with (
                    patch(f"{module}._is_cuda", True),
                    patch(f"{module}._is_hip", False),
                    patch(f"{module}._is_xpu", False),
                    patch(f"{module}.is_ppu", return_value=True),
                    patch(f"{module}.deep_gemm", fake_dg, create=True),
                    patch(
                        f"{module}.get_token_to_kv_pool",
                        return_value=SimpleNamespace(page_size=64),
                    ),
                ):
                    Indexer._get_topk_paged(
                        indexer,
                        SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
                        0,
                        torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn),
                        torch.ones((rows, 32, 1)),
                        metadata,
                    )
                self.assertEqual(
                    planner.call_args.kwargs["metadata_extra"], (5, 32, 128, 1)
                )
                args = native.call_args.args
                self.assertEqual(args[0].shape, (bs, 5, 32, 128))
                self.assertEqual(args[2].shape, (rows, 32))
                torch.testing.assert_close(args[3], ctx)
                torch.testing.assert_close(args[4], pages[::5])
                self.assertEqual(args[4].data_ptr(), pages.data_ptr())
                self.assertEqual(
                    metadata.topk_transform.call_args.args[0].shape, (rows, width)
                )


if __name__ == "__main__":
    unittest.main()
