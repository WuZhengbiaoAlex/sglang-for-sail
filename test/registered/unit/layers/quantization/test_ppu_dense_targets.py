"""CPU contracts for selective PPU dense GEMM targets."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import (
    PPUDenseGemmMode,
    get_ppu_dense_gemm_mode,
)
from sglang.srt.layers.quantization.ppu_dense import (
    create_ppu_dense_linear_method,
    should_use_ppu_dense_w8a8,
)
from sglang.srt.layers.quantization.ppu_dense_w8a8_fp8 import (
    PPUDenseW8A8FP8LinearMethod,
    quantize_dense_weight_to_w8a8_fp8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MODE_ENV = "SGLANG_PPU_DENSE_GEMM_MODE"
_SHARED_GATE = "language_model.model.layers.1.mlp.shared_experts.gate_up_proj"
_SHARED_DOWN = "language_model.model.layers.1.mlp.shared_experts.down_proj"
_DENSE_GATE = "language_model.model.layers.0.mlp.gate_up_proj"
_DENSE_DOWN = "language_model.model.layers.0.mlp.down_proj"
_ATTENTION_QKV = "language_model.model.layers.1.self_attn.fused_qkvg_proj"
_ATTENTION_QKVGBFA = (
    "language_model.model.layers.1.self_attn.fused_qkvgbfa_proj"
)
_MLA_QKV = (
    "language_model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa"
)
_ATTENTION_AUX = "language_model.model.layers.1.self_attn.f_a_proj"
_ATTENTION_OUT = "language_model.model.layers.1.self_attn.o_proj"
_Q_B = "language_model.model.layers.3.self_attn.q_b_proj"
_KV_B = "language_model.model.layers.3.self_attn.kv_b_proj"
_G_PROJ = "language_model.model.layers.3.self_attn.g_proj"
_F_B = "language_model.model.layers.1.self_attn.f_b_proj"
_MOE_ROUTER = "language_model.model.layers.1.mlp.gate"
_LATENT_MOE_UP = "language_model.model.layers.1.mlp.routed_expert_up_proj"


class TestPPUDenseGemmMode(CustomTestCase):
    def mode_env(self, value=None):
        env = {key: val for key, val in os.environ.items() if key != _MODE_ENV}
        if value is not None:
            env[_MODE_ENV] = value
        return patch.dict(os.environ, env, clear=True)

    def test_mode_defaults_to_bf16_and_accepts_w8a8(self):
        with self.mode_env():
            self.assertIs(get_ppu_dense_gemm_mode(), PPUDenseGemmMode.BF16)
        with self.mode_env(" W8A8 "):
            self.assertIs(get_ppu_dense_gemm_mode(), PPUDenseGemmMode.W8A8)

    def test_fixed_w8a8_scope(self):
        selected = (
            _SHARED_GATE,
            _SHARED_DOWN,
            _DENSE_GATE,
            _DENSE_DOWN,
            _ATTENTION_QKV,
            _MLA_QKV,
            _ATTENTION_QKVGBFA,
            _ATTENTION_OUT,
            _LATENT_MOE_UP,
        )
        fixed_bf16 = (
            _Q_B,
            _KV_B,
            _G_PROJ,
            _F_B,
            _ATTENTION_AUX,
            _MOE_ROUTER,
            "language_model.lm_head",
            "language_model.model.layers.1.self_attn.dspark_markov_w2",
        )
        self.assertTrue(all(should_use_ppu_dense_w8a8(x) for x in selected))
        self.assertTrue(all(not should_use_ppu_dense_w8a8(x) for x in fixed_bf16))

    def test_invalid_mode_fails_fast(self):
        with self.mode_env("fp8_block128"):
            with self.assertRaises(ValueError):
                get_ppu_dense_gemm_mode()

    def test_factory_has_one_switch_and_preserves_fixed_bf16(self):
        with patch(
            "sglang.srt.layers.quantization.ppu_dense.is_ppu",
            return_value=True,
        ):
            with self.mode_env("bf16"):
                self.assertIsNone(create_ppu_dense_linear_method(_DENSE_GATE))
            with self.mode_env("w8a8"):
                self.assertIsInstance(
                    create_ppu_dense_linear_method(_DENSE_GATE),
                    PPUDenseW8A8FP8LinearMethod,
                )
                self.assertIsNone(create_ppu_dense_linear_method(_Q_B))

    def test_fp8_w8a8_weight_quantization_contract(self):
        weight = torch.tensor(
            [[-2.0, -0.5, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.bfloat16,
        )
        quantized, scale = quantize_dense_weight_to_w8a8_fp8(weight)
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(scale.shape, (2, 1))
        self.assertTrue(torch.equal(quantized[1].float(), torch.zeros(4)))
        dequantized = quantized.float() * scale
        self.assertTrue(torch.allclose(dequantized, weight.float(), atol=0.02))

    def test_fp8_w8a8_post_load_releases_bf16_parameter(self):
        layer = torch.nn.Linear(16, 64, bias=False, dtype=torch.bfloat16)
        original_bytes = layer.weight.numel() * layer.weight.element_size()
        method = PPUDenseW8A8FP8LinearMethod("test.weight")
        with patch(
            "sglang.srt.layers.quantization.ppu_dense_w8a8_fp8.is_ppu",
            return_value=True,
        ):
            method.process_weights_after_loading(layer)

        self.assertTrue(layer._ppu_w8a8_fp8_enabled)
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            layer.weight.numel() * layer.weight.element_size(), original_bytes // 2
        )
        self.assertEqual(layer._ppu_w8a8_fp8_weight_scale.shape, (64, 1))

        x = torch.randn(3, 16, dtype=torch.float32)
        expected = torch.nn.functional.linear(
            x,
            layer.weight.float() * layer._ppu_w8a8_fp8_weight_scale,
        )
        self.assertTrue(torch.equal(method.apply(layer, x), expected))

    def test_fp8_w8a8_bf16_apply_uses_dedicated_per_token_quantizer(self):
        layer = torch.nn.Module()
        layer.weight = torch.zeros((64, 16), dtype=torch.float8_e4m3fn)
        layer._ppu_w8a8_fp8_weight_scale = torch.ones((64, 1))
        layer._ppu_w8a8_fp8_enabled = True
        method = PPUDenseW8A8FP8LinearMethod("test.weight")
        x = torch.randn((3, 16), dtype=torch.bfloat16)
        quantized = torch.zeros_like(x, dtype=torch.float8_e4m3fn)
        activation_scale = torch.ones((3, 1))

        def write_zero_output(_a, _as, _b, _bs, output):
            output.zero_()

        module = "sglang.srt.layers.quantization.ppu_dense_w8a8_fp8"
        with (
            patch(
                f"{module}.sglang_per_token_quant_fp8",
                return_value=(quantized, activation_scale),
            ) as per_token_quant,
            patch(
                f"{module}.gemm_nt_f8f8bf16", side_effect=write_zero_output
            ) as gemm,
        ):
            output = method.apply(layer, x)

        per_token_quant.assert_called_once()
        self.assertTrue(torch.equal(per_token_quant.call_args.args[0], x))
        gemm.assert_called_once()
        self.assertIs(gemm.call_args.args[0], quantized)
        self.assertIs(gemm.call_args.args[1], activation_scale)
        self.assertEqual(output.shape, (3, 64))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_fp8_w8a8_prequantized_apply_skips_quantizer(self):
        layer = torch.nn.Module()
        layer.weight = torch.zeros((64, 16), dtype=torch.float8_e4m3fn)
        layer._ppu_w8a8_fp8_weight_scale = torch.ones((64, 1))
        layer._ppu_w8a8_fp8_enabled = True
        method = PPUDenseW8A8FP8LinearMethod("test.weight")
        quantized = torch.zeros((3, 16), dtype=torch.float8_e4m3fn)
        activation_scale = torch.ones((3, 1), dtype=torch.float32)

        def write_zero_output(_a, _as, _b, _bs, output):
            output.zero_()

        module = "sglang.srt.layers.quantization.ppu_dense_w8a8_fp8"
        with (
            patch(f"{module}.sglang_per_token_quant_fp8") as quant,
            patch(
                f"{module}.gemm_nt_f8f8bf16", side_effect=write_zero_output
            ) as gemm,
        ):
            output = method.apply(layer, (quantized, activation_scale))

        quant.assert_not_called()
        gemm.assert_called_once()
        self.assertIs(gemm.call_args.args[0], quantized)
        self.assertIs(gemm.call_args.args[1], activation_scale)
        self.assertEqual(output.shape, (3, 64))

    def test_fp8_w8a8_prequantized_apply_validates_contract(self):
        layer = torch.nn.Module()
        layer.weight = torch.zeros((64, 16), dtype=torch.float8_e4m3fn)
        layer._ppu_w8a8_fp8_weight_scale = torch.ones((64, 1))
        layer._ppu_w8a8_fp8_enabled = True
        method = PPUDenseW8A8FP8LinearMethod("test.weight")

        with self.assertRaisesRegex(TypeError, "E4M3 data and FP32 scales"):
            method.apply(
                layer,
                (
                    torch.zeros((3, 16), dtype=torch.bfloat16),
                    torch.ones((3, 1), dtype=torch.float32),
                ),
            )

        quantized = torch.zeros((16, 3), dtype=torch.float8_e4m3fn).t()
        activation_scale = torch.ones((3, 1), dtype=torch.float32)
        self.assertFalse(quantized.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous data"):
            method.apply(layer, (quantized, activation_scale))

        quantized = torch.zeros((3, 16), dtype=torch.float8_e4m3fn)
        activation_scale = torch.empty_strided(
            (3, 1), (2, 1), dtype=torch.float32
        )
        self.assertFalse(activation_scale.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous scales"):
            method.apply(layer, (quantized, activation_scale))

    def test_fp8_w8a8_empty_bf16_apply_skips_device_kernels(self):
        layer = torch.nn.Module()
        layer.weight = torch.zeros((64, 16), dtype=torch.float8_e4m3fn)
        layer._ppu_w8a8_fp8_weight_scale = torch.ones((64, 1))
        layer._ppu_w8a8_fp8_enabled = True
        method = PPUDenseW8A8FP8LinearMethod("test.weight")
        x = torch.empty((0, 16), dtype=torch.bfloat16)

        module = "sglang.srt.layers.quantization.ppu_dense_w8a8_fp8"
        with (
            patch(f"{module}.sglang_per_token_quant_fp8") as quant,
            patch(f"{module}.gemm_nt_f8f8bf16") as gemm,
        ):
            output = method.apply(layer, x)

        quant.assert_not_called()
        gemm.assert_not_called()
        self.assertEqual(output.shape, (0, 64))
        self.assertEqual(output.dtype, torch.bfloat16)

if __name__ == "__main__":
    unittest.main()
