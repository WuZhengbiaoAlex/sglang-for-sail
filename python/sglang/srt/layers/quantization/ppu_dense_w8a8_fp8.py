"""Experimental online FP8-E4M3 W8A8 path for PPU dense linears."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.srt.layers.quantization.fp8_utils import gemm_nt_f8f8bf16
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_ppu, logger


def _store_nonpersistent_buffer(layer: nn.Module, name: str, value: torch.Tensor):
    if name in layer._buffers:
        layer._buffers[name] = value
    else:
        layer.register_buffer(name, value, persistent=False)


@torch.no_grad()
def quantize_dense_weight_to_w8a8_fp8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel E4M3 quantization for a [N, K] weight."""
    if weight.ndim != 2:
        raise ValueError(f"dense FP8 W8A8 requires a rank-2 weight, got {weight.shape}")
    if weight.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"dense FP8 W8A8 requires BF16/FP16 weight, got {weight.dtype}"
        )

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    weight_fp32 = weight.float()
    scale = weight_fp32.abs().amax(dim=1, keepdim=True).clamp_min(1e-10) / fp8_max
    quantized = (
        (weight_fp32 / scale)
        .clamp(-fp8_max, fp8_max)
        .to(fp8_dtype)
        .contiguous()
    )
    return quantized, scale.contiguous()


class PPUDenseW8A8FP8LinearMethod(UnquantizedLinearMethod):
    """Load BF16 normally, then replace it with per-channel E4M3 weights."""

    accepts_prequantized_input = True

    def __init__(self, prefix: str):
        self.prefix = prefix
        self._runtime_hit_logged = False

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if not is_ppu():
            raise RuntimeError("PPU dense FP8 W8A8 was selected on a non-PPU device")

        n, k = layer.weight.shape
        if n < 64 or k % 16 != 0:
            logger.warning(
                "PPU dense FP8 W8A8 falling back to BF16 for %s: "
                "incompatible shape N=%d K=%d",
                self.prefix,
                n,
                k,
            )
            return

        weight, weight_scale = quantize_dense_weight_to_w8a8_fp8(layer.weight)
        layer.weight = nn.Parameter(weight, requires_grad=False)
        _store_nonpersistent_buffer(layer, "_ppu_w8a8_fp8_weight_scale", weight_scale)
        layer._ppu_w8a8_fp8_enabled = True
        logger.info(
            "PPU dense FP8 W8A8 materialized prefix=%s N=%d K=%d",
            self.prefix,
            n,
            k,
        )
        logger.info_once(
            "Enabled PPU dense FP8-E4M3 W8A8; weights use per-output-channel "
            "scales and activations use dynamic "
            "per-token scales; original BF16 weights are released."
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enabled = getattr(layer, "_ppu_w8a8_fp8_enabled", False)
        weight_scale = getattr(layer, "_ppu_w8a8_fp8_weight_scale", None)
        prequantized = isinstance(x, tuple)
        if prequantized:
            if not enabled:
                raise RuntimeError("prequantized FP8 input requires PPU W8A8 weights")
            if len(x) != 2:
                raise ValueError("prequantized FP8 input must be an (input, scale) pair")
            x_q, x_scale = x
            if x_q.dtype != torch.float8_e4m3fn or x_scale.dtype != torch.float32:
                raise TypeError(
                    "prequantized FP8 input requires E4M3 data and FP32 scales"
                )
            if (
                x_q.ndim != 2
                or x_scale.shape != (x_q.shape[0], 1)
                or not x_q.is_contiguous()
                or not x_scale.is_contiguous()
            ):
                raise ValueError(
                    "prequantized FP8 input requires contiguous data [M, K] "
                    "and contiguous scales [M, 1]"
                )
            input_shape = x_q.shape
            x_2d = x_q
            device = x_q.device
        else:
            if not enabled:
                return F.linear(x, layer.weight, bias)

            weight = layer.weight
            if x.dtype != torch.bfloat16:
                dequantized = (weight.float() * weight_scale).to(x.dtype)
                return F.linear(x, dequantized, bias)

            input_shape = x.shape
            x_2d = x.reshape(-1, input_shape[-1]).contiguous()
            device = x.device

        weight = layer.weight
        if x_2d.shape[0] == 0:
            # DP-attention sends empty synchronization batches to idle DP
            # ranks.  DeepGEMM cannot launch a zero-grid GEMM, so preserve the
            # LinearBase empty-tensor contract without touching the device
            # quantization or GEMM kernels.
            return torch.empty(
                (*input_shape[:-1], weight.shape[0]),
                dtype=torch.bfloat16,
                device=device,
            )

        if not self._runtime_hit_logged:
            logger.info(
                "PPU dense FP8 W8A8 runtime hit prefix=%s weight_shape=%s",
                self.prefix,
                tuple(weight.shape),
            )
            self._runtime_hit_logged = True

        if not prequantized:
            x_q, x_scale = sglang_per_token_quant_fp8(x_2d)
        out = torch.empty(
            (x_2d.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=device,
        )
        gemm_nt_f8f8bf16(x_q, x_scale, weight, weight_scale, out)
        if bias is not None:
            out.add_(bias.to(out.dtype))
        return out.view(*input_shape[:-1], weight.shape[0])
