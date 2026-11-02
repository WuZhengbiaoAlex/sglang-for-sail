"""Hybrid DSA indexer storage must match the selected writer format."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestHybridDSAIndexerFormat(CustomTestCase):
    def test_fp4_and_fp8_page_stride(self):
        with (
            patch("sglang.srt.mem_cache.memory_pool.is_ppu", return_value=True),
            patch.object(DSATokenToKVPool, "use_bf16_indexer", False),
        ):
            for fp4, bytes_per_token in ((False, 132), (True, 68)):
                with self.subTest(fp4=fp4):
                    pool = HybridLinearKVPool(
                        size=128,
                        dtype=torch.bfloat16,
                        page_size=64,
                        head_num=1,
                        head_dim=576,
                        full_attention_layer_ids=[3],
                        device="cpu",
                        mamba_pool=None,
                        use_mla=True,
                        kv_lora_rank=512,
                        qk_rope_head_dim=64,
                        use_dsa=True,
                        index_head_dim=128,
                        kv_cache_dim=576,
                        use_fp4_indexer=fp4,
                    )
                    buffer = pool.get_index_k_with_scale_buffer(3)
                    self.assertEqual(pool.full_kv_pool.use_fp4_indexer, fp4)
                    self.assertEqual(buffer.shape[-1], 64 * bytes_per_token)
                    self.assertEqual(buffer.stride(0), 64 * bytes_per_token)


if __name__ == "__main__":
    unittest.main()
