"""PPU model-owned dense GEMM precision helpers.

Kimi-K3 selection is limited to BF16 or channel-wise FP8 W8A8, with fixed
BF16 exclusions for accuracy- or shape-sensitive projections.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.environ import PPUDenseGemmMode, get_ppu_dense_gemm_mode
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_ppu

_DENSE_W8A8_EXCLUDED_PREFIXES = (
    "lm_head",
    "vision_tower",
    "mm_projector",
)
# This allowlist is the production W8A8 boundary. Standalone q_b_proj,
# kv_b_proj, g_proj, f_a_proj, f_b_proj, the router/TopK GEMM, and
# DSpark-owned linears remain BF16 regardless of SGLANG_PPU_DENSE_GEMM_MODE.
# The always-merged full-rank KDA fused_qkvgbfa_proj is one selected GEMM and
# follows the global mode as a unit.
_DENSE_W8A8_SUFFIXES = (
    ".gate_up_proj",
    ".down_proj",
    ".routed_expert_up_proj",
    ".qkv_proj",
    ".fused_qkvg_proj",
    ".fused_qkvgbfa_proj",
    ".fused_qkvbfg_a_proj",
    ".fused_qkv_a_proj_with_mqa",
    ".q_proj",
    ".q_a_proj",
    ".kv_a_proj_with_mqa",
    ".o_proj",
)


def should_use_ppu_dense_w8a8(prefix: str) -> bool:
    """Select the fixed Kimi-K3 W8A8 scope and preserve BF16 auxiliaries."""
    if not prefix or any(
        fragment in prefix for fragment in _DENSE_W8A8_EXCLUDED_PREFIXES
    ):
        return False
    return prefix.endswith(_DENSE_W8A8_SUFFIXES)


def create_ppu_dense_linear_method(
    prefix: str,
) -> Optional[UnquantizedLinearMethod]:
    """Build the selected PPU dense method for a BF16 checkpoint linear.

    W8A8 is the channel-wise FP8 implementation qualified by the current K3
    accuracy/performance run. A few model-owned BF16 projections (notably
    latent-MoE up) call this helper directly after constructing an unquantized
    parameter; fixed BF16 projections are rejected by the suffix allowlist.
    """
    if (
        not is_ppu()
        or get_ppu_dense_gemm_mode() is not PPUDenseGemmMode.W8A8
        or not should_use_ppu_dense_w8a8(prefix)
    ):
        return None

    from sglang.srt.layers.quantization.ppu_dense_w8a8_fp8 import (
        PPUDenseW8A8FP8LinearMethod,
    )

    return PPUDenseW8A8FP8LinearMethod(prefix)
