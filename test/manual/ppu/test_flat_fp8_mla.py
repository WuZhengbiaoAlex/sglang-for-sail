"""PPU raw E4M3 cache sizing, incremental writes, and native DSA graph replay."""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_ppu
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available() and is_ppu(), "requires PPU")
class TestPPUFlatFP8MLA(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="flashmla_sparse_q8",
            dsa_decode_backend="flashmla_sparse_q8",
        )
        publish(self.args, role="tokenizer")

    def test_pool_and_decode_graph(self):
        import flash_mla
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        torch.manual_seed(298)
        for rope_dim, absent_rope in ((0, True), (0, False), (64, False)):
            with self.subTest(rope_dim=rope_dim, absent_rope=absent_rope):
                dim, rows = 512 + rope_dim, 1024
                pool = DSATokenToKVPool(
                    size=rows,
                    page_size=64,
                    kv_lora_rank=512,
                    dtype=torch.float8_e4m3fn,
                    qk_rope_head_dim=rope_dim,
                    layer_num=1,
                    device="cuda",
                    index_head_dim=128,
                    enable_memory_saver=False,
                    kv_cache_dim=dim,
                )
                self.assertFalse(pool.dsa_kv_cache_store_fp8)
                cache = pool.get_key_buffer(0)
                self.assertEqual(cache.shape, (rows + 64, 1, dim))
                self.assertEqual(cache.dtype, torch.float8_e4m3fn)
                layer = types.SimpleNamespace(layer_id=0)
                loc = torch.randperm(rows - 1, device="cuda")[:128] + 1
                loc[0] = 0  # Reserved padding writes must be skipped.
                k = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.3
                rope = (
                    None
                    if absent_rope
                    else torch.randn(
                        128, 1, rope_dim, device="cuda", dtype=torch.bfloat16
                    )
                    * 0.3
                )
                q = torch.randn(3, 12, dim, device="cuda", dtype=torch.bfloat16) * 0.3
                scale = (128 + rope_dim) ** -0.5
                ids = (
                    loc[1:65]
                    .to(torch.int32)[None, None, :]
                    .expand(3, 1, 64)
                    .contiguous()
                )
                ids[0, 0, 5] = -1
                backend = object.__new__(DeepseekSparseAttnBackend)
                backend._ppu_q8_prefill = flash_mla.flash_mla_sparse_quantized_fwd
                seen_ptrs = []

                def decode(**kwargs):
                    seen_ptrs.append(kwargs["kv"].data_ptr())
                    return flash_mla.flash_mla_sparse_quantized_decode_fwd(**kwargs)

                backend._ppu_q8_decode = decode

                def call():
                    pool.set_mla_kv_buffer(layer, loc, k, rope)
                    return backend._forward_ppu_flashmla_sparse_q8(
                        q, cache, 512, ids[:, 0], scale, decode=True
                    )

                for _ in range(3):
                    call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = call()
                k.mul_(-2)
                if rope is not None:
                    rope.mul_(-2)
                q.mul_(0.5)
                graph.replay()
                expected_cache = torch.zeros(
                    cache.shape, device="cuda", dtype=torch.uint8
                )
                new_rows = k if rope_dim == 0 else torch.cat((k, rope), -1)
                expected_cache[loc[1:]] = (
                    new_rows[1:].to(torch.float8_e4m3fn).view(torch.uint8)
                )
                torch.testing.assert_close(
                    cache.view(torch.uint8), expected_cache, rtol=0, atol=0
                )
                expected, _ = flash_mla.flash_mla_sparse_quantized_decode_fwd(
                    q, expected_cache.view(torch.float8_e4m3fn), ids, scale
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertTrue(
                    all(pointer == cache.data_ptr() for pointer in seen_ptrs)
                )

    def test_layout_estimate(self):
        from sglang.srt.mem_cache.kv_cache_configurator import (
            calculate_mla_kv_cache_dim,
        )
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        for rope in (0, 64):
            cfg = types.SimpleNamespace(
                hf_config=None, kv_lora_rank=512, qk_rope_head_dim=rope
            )
            runner = types.SimpleNamespace(
                model_config=cfg,
                kv_cache_dtype=torch.float8_e4m3fn,
                server_args=self.args,
            )
            with patch(
                "sglang.srt.mem_cache.kv_cache_configurator.is_deepseek_dsa",
                return_value=True,
            ):
                self.assertEqual(
                    calculate_mla_kv_cache_dim(
                        model_config=cfg, kv_cache_dtype=torch.float8_e4m3fn
                    ),
                    512 + rope,
                )
            with patch(
                "sglang.srt.model_executor.model_runner_kv_cache_mixin.is_deepseek_dsa",
                return_value=True,
            ):
                self.assertEqual(
                    ModelRunnerKVCacheMixin.calculate_mla_kv_cache_dim(runner),
                    512 + rope,
                )


if __name__ == "__main__":
    unittest.main()
