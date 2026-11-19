"""PPU KDA sigmoid-gated RMSNorm and per-token E4M3 quantization."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _kda_gate_norm_fp8_kernel(
    X,
    G,
    W,
    Q,
    SCALE,
    EPS: gl.constexpr,
    X_ROW: gl.constexpr,
    X_HEAD: gl.constexpr,
    X_DIM: gl.constexpr,
    G_ROW: gl.constexpr,
    G_HEAD: gl.constexpr,
    G_DIM: gl.constexpr,
):
    row = gl.program_id(0).to(gl.int64)
    # Match the FP32 reduction order of layer_norm_gated_fwd_kernel on PPU.
    # Eight consecutive values per lane contribute to each 128-wide head.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
    head = gl.arange(0, 16, layout=gl.constexpr(gl.SliceLayout(1, layout)))
    dim = gl.arange(0, 128, layout=gl.constexpr(gl.SliceLayout(0, layout)))
    valid = head[:, None] < 12
    x = gl.load(
        X + row * X_ROW + head[:, None] * X_HEAD + dim[None, :] * X_DIM,
        mask=valid,
        other=0,
    ).to(gl.float32)
    gate = gl.load(
        G + row * G_ROW + head[:, None] * G_HEAD + dim[None, :] * G_DIM,
        mask=valid,
        other=0,
    ).to(gl.float32)
    weight = gl.load(W + dim).to(gl.float32)
    rstd = 1.0 / gl.sqrt(gl.sum(x * x, axis=1) / 128 + EPS)
    sigmoid_gate = 1.0 / (1.0 + gl.exp(-gate))
    normalized = x * rstd[:, None] * weight[None, :] * sigmoid_gate

    # The separate norm stores BF16 before native FP8 quantization reads it.
    value = normalized.to(gl.bfloat16).to(gl.float32)
    amax = gl.max(gl.max(gl.abs(value), 1), 0)
    # Native per-token quantization uses round-to-nearest FP32 division.
    scale = gl.div_rn(amax, 448.0)
    inv_scale = gl.where(scale == 0.0, 0.0, gl.div_rn(1.0, scale))
    gl.store(
        Q + row * 1536 + head[:, None] * 128 + dim[None, :],
        value * inv_scale,
        mask=valid,
    )
    gl.store(SCALE + row, scale)


def kda_gate_norm_fp8(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [tokens, 12, 128] KDA outputs with one scale per token."""
    if x.ndim != 3 or x.shape[1:] != (12, 128) or gate.shape != x.shape:
        raise ValueError(
            "Expected KDA activations and gates of shape [tokens, 12, 128]"
        )
    if weight.shape != (128,) or not weight.is_contiguous():
        raise ValueError("Expected a contiguous 128-element KDA norm weight")
    if x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise TypeError("Expected BF16 KDA activations and gates")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("Expected a BF16 or FP32 KDA norm weight")
    if not x.is_cuda or gate.device != x.device or weight.device != x.device:
        raise ValueError("KDA activations, gates and weight must share a PPU device")

    rows = x.shape[0]
    q = torch.empty((rows, 1536), device=x.device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
    if rows:
        _kda_gate_norm_fp8_kernel[(rows,)](
            x, gate, weight, q, scale, eps, *x.stride(), *gate.stride(), num_warps=4
        )
    return q, scale
