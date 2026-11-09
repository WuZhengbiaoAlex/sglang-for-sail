import unittest
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

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


if __name__ == "__main__":
    unittest.main()
