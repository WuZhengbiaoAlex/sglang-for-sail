"""A sole request can share its page mapping across all prefill queries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend, TopkTransformMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTopKPageBroadcast(CustomTestCase):
    def _run(self, single_request, enabled):
        logits = torch.arange(24).reshape(3, 8).to(torch.bfloat16)
        lengths = torch.tensor([4, 6, 8], dtype=torch.int32)
        table = torch.arange(8, dtype=torch.int32)[None] * 7
        if not single_request:
            table = torch.cat([table, table + 100])
        batches = torch.tensor([0, 0, 0] if single_request else [0, 1, 1])
        metadata = SimpleNamespace(page_table_1=table, token_to_batch_idx=batches)
        seen = []

        def topk(scores, starts, ends, pages, out, page_size, raw):
            seen.append(pages)
            for row in range(3):
                indices = torch.topk(scores[row, :int(ends[row])].float(), 2).indices
                out[row] = pages[row, indices]

        # Platform probe only: exercise the PPU-specific stride contract on CPU.
        with (
            patch('sglang.srt.layers.attention.dsa.dsa_topk_backend.is_ppu', return_value=True),
            patch('sglang.kernels.ops.attention.dsv4.top_k_per_row_prefill_bf16', side_effect=topk),
            envs.SGLANG_PPU_DSA_TOPK_BROADCAST_PAGE_TABLE.override(enabled),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_DSA_USE_DSV4_BF16_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(False),
        ):
            output = DSATopKBackend.SGL_KERNEL.topk_transform(
                logits, lengths, 2, TopkTransformMethod.PAGED, metadata,
                cu_seqlens_q_topk=torch.arange(4, dtype=torch.int32),
                row_starts=torch.zeros(3, dtype=torch.int32), batch_idx_list=batches)
        expected = torch.stack([table[batches[r], torch.tensor([int(lengths[r])-1, int(lengths[r])-2])]
                                for r in range(3)])
        torch.testing.assert_close(output, expected)
        if single_request and enabled:
            self.assertEqual(seen[0].stride(0), 0)
            self.assertEqual(seen[0].untyped_storage().data_ptr(), table.untyped_storage().data_ptr())
        else:
            self.assertNotEqual(seen[0].stride(0), 0)

    def test_single_request_uses_zero_stride_view(self):
        self._run(True, True)

    def test_disabled_uses_materialized_page_rows(self):
        self._run(True, False)

    def test_multiple_requests_keep_distinct_page_maps(self):
        self._run(False, True)


if __name__ == '__main__':
    unittest.main()
