"""PPU1.5 W4A4 MegaMoE adapter; shared experts remain outside this path."""

import inspect
import logging
from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_expert_recv_stats_kwargs
from sglang.srt.runtime_context import get_buffer, get_parallel


@lru_cache(maxsize=16)
def _supports_logical_expert_count(kernel):
    try:
        supported = "num_logical_experts" in inspect.signature(kernel).parameters
    except (TypeError, ValueError):
        supported = False
    if not supported:
        logging.getLogger(__name__).warning(
            "DeepGEMM lacks logical-expert workload tuning; incremental EPLB "
            "will retain capacity-based MegaMoE kernel selection."
        )
    return supported


def _get_mega_moe_kwargs(experts, kernel):
    kwargs = get_expert_recv_stats_kwargs(experts)
    if kwargs and _supports_logical_expert_count(kernel):
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        # The counter is bound only during incremental target decode.
        kwargs["num_logical_experts"] = (
            get_global_expert_location_metadata().num_logical_experts
        )
    return kwargs


def prepare_weights(experts, *, activation):
    """Transform raw checkpoint UE8M0 pairs once, before ordinary preprocessing."""
    import deep_gemm

    if getattr(experts, "_mega_moe_weights_built", False):
        return
    activation = "swiglu" if activation == "silu" else activation
    l1, l2 = deep_gemm.transform_weights_for_mega_moe(
        (experts.w13_weight.data, experts.w13_weight_scale.data),
        (experts.w2_weight.data, experts.w2_weight_scale.data),
        activation=activation,
    )
    # K3 sends every routed batch through MegaMoE; do not retain a duplicate
    # checkpoint layout across all layers.
    experts.w13_weight.data, experts.w13_weight_scale.data = l1
    experts.w2_weight.data, experts.w2_weight_scale.data = l2
    # Parameter .data views do not retain transform bases after arena rebinding.
    experts.mega_l1_weights = (experts.w13_weight.data, experts.w13_weight_scale.data)
    experts.mega_l2_weights = (experts.w2_weight.data, experts.w2_weight_scale.data)
    experts._mega_moe_weights_built = True
    experts._mxfp4_backend = "deep_gemm"


def forward(
    experts,
    hidden_states,
    topk_ids,
    topk_weights,
    *,
    hidden_size,
    intermediate_size,
    top_k,
    activation="swiglu",
    activation_clamp=None,
    situ_beta=4.0,
    situ_linear_beta=25.0,
):
    import deep_gemm

    if not getattr(experts, "_mega_moe_weights_built", False):
        raise RuntimeError("PPU MegaMoE weights have not been prepared.")
    num_tokens = hidden_states.shape[0]
    activation = "swiglu" if activation == "silu" else activation
    capacity = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    if num_tokens > capacity:
        raise ValueError(
            f"PPU MegaMoE local tokens {num_tokens} exceed capacity {capacity}."
        )
    group = get_parallel().moe_ep_group.device_group
    key = (
        id(group),
        hidden_states.device,
        experts.num_experts,
        capacity,
        top_k,
        hidden_size,
        intermediate_size,
        activation,
    )
    buffers = get_buffer("ppu_mega_moe.workspaces", dict)
    if key not in buffers:
        buffers[key] = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts=experts.num_experts,
            num_max_tokens_per_rank=capacity,
            num_topk=top_k,
            hidden=hidden_size,
            intermediate_hidden=intermediate_size,
            mma_type="fp4xfp4",
            activation=activation,
        )
    buf = buffers[key]
    # The public PPU API stages int64 routes and applies routing weights once
    # in final combine. It owns all scratch/counter recycling, including graphs.
    deep_gemm.mega_moe_pre_dispatch(
        hidden_states,
        topk_ids.to(torch.int64).contiguous(),
        topk_weights.to(torch.float32).contiguous(),
        buf,
    )
    y = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    deep_gemm.fp4_fp4_mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        **_get_mega_moe_kwargs(experts, deep_gemm.fp4_fp4_mega_moe),
        recipe=(1, 1, 32),
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=True,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return y
