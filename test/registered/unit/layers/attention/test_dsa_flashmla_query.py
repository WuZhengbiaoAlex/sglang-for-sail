import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSAFlashMLAQuery(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=0,
            tp_q_head_num=2,
            v_head_dim=6,
            head_dim=9,
            scaling=1.0,
        )
        # Stride both heads and features, so reinterpreting Q or dropping
        # the RoPE slice would corrupt the values handed to attention.
        self.q = torch.arange(3 * 4 * 18, dtype=torch.bfloat16).view(3, 4, 18)[
            :, ::2, 1::2
        ]
        self.topk = torch.tensor([[0, -1], [0, 1], [1, 2]], dtype=torch.int32)
        self.cache = torch.empty((3, 1, 9), dtype=torch.uint8)
        self.backend = dsa_backend.DeepseekSparseAttnBackend.__new__(
            dsa_backend.DeepseekSparseAttnBackend
        )
        self.backend.dsa_decode_impl = self.backend.dsa_prefill_impl = (
            "flashmla_sparse_q8"
        )
        self.backend.dsa_index_kpool = 1
        self.backend.dsa_kv_cache_store_fp8 = True
        self.backend.use_mha = False
        self.backend.use_fused_topk = True
        self.backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        self.backend.hisparse_coordinator = None
        self.backend.token_to_kv_pool = Mock()
        self.backend.token_to_kv_pool.get_key_buffer.return_value = self.cache
        self.backend.forward_metadata = SimpleNamespace(
            page_table_1_flattened=torch.arange(3, dtype=torch.int32),
        )

    def forward(self, q, q_rope, mode):
        batch = SimpleNamespace(forward_mode=mode, extend_prefix_lens_cpu=[1])
        return self.backend.forward_extend(
            q, None, None, self.layer, batch, q_rope=q_rope, topk_indices=self.topk
        )

    def test_paged_verify_packs_only_split_q(self):
        cases = (
            (self.q[..., :6], self.q[..., 6:], 1),
            (self.q.contiguous(), None, 0),
            (self.q, None, 0),
        )
        for q, q_rope, calls in cases:
            with (
                self.subTest(split=q_rope is not None, stride=q.stride()),
                patch.object(dsa_backend, "_is_ppu", True),
                patch.object(
                    dsa_backend,
                    "concat_mla_absorb_q_general",
                    wraps=dsa_backend.concat_mla_absorb_q_general,
                ) as concat,
                patch.object(self.backend, "_forward_ppu_flashmla_sparse_q8") as sparse,
            ):
                self.forward(q, q_rope, ForwardMode.TARGET_VERIFY)
                sparse.assert_called_once()
                packed = sparse.call_args.kwargs["q_all"]
                torch.testing.assert_close(packed, self.q, atol=0, rtol=0)
                self.assertTrue(packed.is_contiguous())
                if q_rope is None and q.is_contiguous():
                    self.assertEqual(packed.data_ptr(), q.data_ptr())
                self.assertEqual(concat.call_count, calls)

    def test_packed_q_alignment(self):
        # BF16 offsets of eight and one elements give an aligned and an
        # unaligned view respectively, although both remain contiguous.
        for offset, compiling, expect_reuse in (
            (8, False, True),
            (1, False, False),
            (8, True, False),
        ):
            storage = torch.empty(self.q.numel() + offset, dtype=self.q.dtype)
            q = storage[offset:].view_as(self.q)
            q.copy_(self.q)
            self.assertTrue(q.is_contiguous())
            self.assertEqual(q.data_ptr() % 16, (offset * q.element_size()) % 16)
            with (
                self.subTest(offset=offset, compiling=compiling),
                patch.object(dsa_backend, "_is_ppu", True),
                patch.object(torch.compiler, "is_compiling", return_value=compiling),
                patch.object(
                    dsa_backend,
                    "concat_mla_absorb_q_general",
                    wraps=dsa_backend.concat_mla_absorb_q_general,
                ) as concat,
                patch.object(self.backend, "_forward_ppu_flashmla_sparse_q8") as sparse,
            ):
                self.forward(q, None, ForwardMode.TARGET_VERIFY)
                concat.assert_not_called()
                sparse.assert_called_once()
                packed = sparse.call_args.kwargs["q_all"]
                self.assertEqual(packed.data_ptr() % 16, 0)
                self.assertEqual(packed.data_ptr() == q.data_ptr(), expect_reuse)
                torch.testing.assert_close(packed, self.q, atol=0, rtol=0)

    def test_ragged_ppu_packs_split_q_for_attention(self):
        with (
            patch.object(dsa_backend, "_is_ppu", True),
            patch.object(dsa_backend, "dequantize_k_cache_paged") as dequant,
            patch.object(
                dsa_backend,
                "concat_mla_absorb_q_general",
                wraps=dsa_backend.concat_mla_absorb_q_general,
            ) as concat,
            patch.object(self.backend, "_forward_ppu_flashmla_sparse_q8") as sparse,
        ):
            self.forward(self.q[..., :6], self.q[..., 6:], ForwardMode.EXTEND)
        dequant.assert_called_once_with(
            self.cache, self.backend.forward_metadata.page_table_1_flattened
        )
        concat.assert_called_once()
        torch.testing.assert_close(
            sparse.call_args.kwargs["q_all"], self.q, atol=0, rtol=0
        )

    def test_ragged_q8kv8_consumes_components_directly(self):
        for split in (True, False):
            with (
                self.subTest(split=split),
                patch.object(dsa_backend, "_is_ppu", False),
                patch.object(
                    dsa_backend,
                    "concat_mla_absorb_q_general",
                    wraps=dsa_backend.concat_mla_absorb_q_general,
                ) as concat,
                patch.object(self.backend, "_forward_flashmla_sparse_q8kv8") as sparse,
            ):
                q, q_rope = (
                    (self.q[..., :6], self.q[..., 6:]) if split else (self.q, None)
                )
                self.forward(q, q_rope, ForwardMode.EXTEND)
                concat.assert_not_called()
                sparse.assert_called_once()
                args = sparse.call_args.kwargs
                torch.testing.assert_close(
                    args["q_nope"], self.q[..., :6], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    args["q_rope"], self.q[..., 6:], atol=0, rtol=0
                )
                self.assertIs(args["paged_kv_cache"], self.cache)


if __name__ == "__main__":
    unittest.main()
