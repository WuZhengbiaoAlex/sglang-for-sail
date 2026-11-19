import unittest

import torch
from torch import nn

from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.attention.fla.kda_gate_norm_fp8 import kda_gate_norm_fp8
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.srt.layers.quantization.ppu_dense_w8a8_fp8 import (
    PPUDenseW8A8FP8LinearMethod,
    quantize_dense_weight_to_w8a8_fp8,
)
from sglang.srt.utils import is_ppu
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=20, suite="stage-b-test-1-gpu-ppu")


@unittest.skipUnless(is_ppu() and torch.cuda.is_available(), "requires PPU")
class TestKDAGateNormFP8(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.rng = torch.Generator(device="cuda").manual_seed(97)
        self.norm = FusedRMSNormGated(
            128, eps=1e-6, activation="sigmoid", dtype=torch.bfloat16, device="cuda"
        )
        self.norm.weight.data.copy_(self.rand(128))

    def rand(self, *shape):
        return torch.randn(
            shape, generator=self.rng, dtype=torch.bfloat16, device="cuda"
        )

    def reference(self, x, gate):
        normalized = self.norm(x.clone(), gate).reshape(x.shape[0], 1536)
        return normalized, sglang_per_token_quant_fp8(normalized)

    def assert_pair_equal(self, actual, expected):
        q, scale = actual
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(tuple(scale.shape), (q.shape[0], 1))
        self.assertTrue(q.is_contiguous() and scale.is_contiguous())
        self.assertTrue(torch.equal(q.view(torch.uint8), expected[0].view(torch.uint8)))
        self.assertTrue(
            torch.equal(scale.view(torch.int32), expected[1].view(torch.int32))
        )

    @torch.inference_mode()
    def test_gate_layouts_and_quantization(self):
        for rows, strided in ((1, False), (5, True), (160, True), (641, False)):
            with self.subTest(rows=rows, strided=strided):
                x = self.rand(rows, 12, 128)
                gate = (
                    self.rand(rows, 4, 12, 128)[:, 3]
                    if strided
                    else self.rand(*x.shape)
                )
                if rows > 1:
                    x[0].zero_()
                    x[1].mul_(1e-12)
                    gate[-1, 0].fill_(-80)
                    gate[-1, 1].fill_(80)
                _, reference = self.reference(x, gate)
                result = kda_gate_norm_fp8(x, gate, self.norm.weight, self.norm.eps)
                self.assert_pair_equal(result, reference)

    @torch.inference_mode()
    def test_graph_replay_and_projection(self):
        x = self.rand(160, 12, 128)
        gate = self.rand(160, 4, 12, 128)[:, 3]
        method = PPUDenseW8A8FP8LinearMethod("test.kda.o_proj")
        proj = nn.Module()
        weight, scale = quantize_dense_weight_to_w8a8_fp8(self.rand(128, 1536))
        proj.weight = nn.Parameter(weight, requires_grad=False)
        proj.register_buffer("_ppu_w8a8_fp8_weight_scale", scale)
        proj._ppu_w8a8_fp8_enabled = True

        def fused():
            return kda_gate_norm_fp8(x, gate, self.norm.weight, self.norm.eps)

        fused()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = fused()
        for _ in range(2):
            x.copy_(self.rand(*x.shape))
            gate.copy_(self.rand(*gate.shape))
            graph.replay()
            normed, reference = self.reference(x, gate)
            self.assert_pair_equal(result, reference)
            self.assertTrue(
                torch.equal(method.apply(proj, normed), method.apply(proj, result))
            )


if __name__ == "__main__":
    unittest.main()
