"""Host-side partition invariants for Kimi K3.1 indexer query sharding."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer, _is_kimi_k31_indexer_config

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestIndexerTPSplit(CustomTestCase):
    def test_normalized_text_config_is_recognized(self):
        self.assertTrue(_is_kimi_k31_indexer_config(SimpleNamespace(model_type='kimi_k31')))
        self.assertTrue(_is_kimi_k31_indexer_config(SimpleNamespace(
            model_type='kimi_linear', architectures=['KimiK31LinearForCausalLM'])))
        self.assertFalse(_is_kimi_k31_indexer_config(SimpleNamespace(
            model_type='kimi_linear', architectures=['KimiK3LinearForCausalLM'])))
        self.assertFalse(_is_kimi_k31_indexer_config(None))

    def test_cached_32k_partition_is_complete_and_balanced(self):
        bounds = Indexer._indexer_tp_split_bounds(torch.tensor([32768]), [3328], 3328, 16)
        self.assertEqual(bounds[0], 0)
        self.assertEqual(bounds[-1], 3328)
        self.assertTrue(all(a < b for a, b in zip(bounds, bounds[1:])))
        costs = torch.arange(29441, 32769)
        loads = [int(costs[a:b].sum()) for a, b in zip(bounds, bounds[1:])]
        self.assertLess(max(loads) / min(loads), 1.01)
        # Gather concatenates contiguous shards back to the original query order.
        rows = torch.arange(3328 * 3).view(3328, 3)
        restored = torch.cat([rows[a:b] for a, b in zip(bounds, bounds[1:])])
        torch.testing.assert_close(restored, rows)

    def test_mixed_prefixes_zero_length_and_nondivisible_rows(self):
        for seq, ext, q, tp in [([9, 150, 40], [0, 111, 19], 130, 16),
                                ([4096], [4096], 4096, 16),
                                ([16], [16], 16, 16)]:
            with self.subTest(ext=ext):
                bounds = Indexer._indexer_tp_split_bounds(torch.tensor(seq), ext, q, tp)
                self.assertEqual(len(bounds), tp + 1)
                self.assertEqual((bounds[0], bounds[-1]), (0, q))
                self.assertTrue(all(a < b for a, b in zip(bounds, bounds[1:])))

    def test_metadata_mismatch_uses_equal_row_partition(self):
        bounds = Indexer._indexer_tp_split_bounds(torch.tensor([100]), [7], 37, 4)
        self.assertEqual(bounds, [0, 9, 18, 27, 37])

    def test_invalid_partition_rejected(self):
        for q, tp in [(3, 4), (4, 0)]:
            with self.subTest(q=q, tp=tp), self.assertRaises(ValueError):
                Indexer._indexer_tp_split_bounds(torch.tensor([q]), [q], q, tp)

    def test_disabled_instance_does_not_read_runtime_or_enter_collective(self):
        indexer = object.__new__(Indexer)
        indexer.tp_split_min_rows = None
        self.assertIsNone(indexer._indexer_tp_split(None, 3328))


if __name__ == "__main__":
    unittest.main()
