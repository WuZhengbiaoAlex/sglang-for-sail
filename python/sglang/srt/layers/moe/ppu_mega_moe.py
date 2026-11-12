"""PPU1.5 W4A4 MegaMoE adapter; shared experts remain outside this path."""

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import get_expert_recv_stats_kwargs
from sglang.srt.runtime_context import get_buffer, get_forward, get_parallel


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
    if envs.SGLANG_OPT_ULTRA_EP_PPU.get():
        from sglang.srt.layers.moe.ppu_mega_moe_ultra_ep import register_weights

        del l1, l2
        register_weights(experts, activation=activation)
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
    num_experts = experts.num_experts
    routed_ids = topk_ids.to(torch.int64).contiguous()
    weight_event = None
    gemm_kwargs = get_expert_recv_stats_kwargs(experts)
    ultra_ep = getattr(experts, "_ppu_mega_ultra_ep", None)
    if ultra_ep is not None and get_forward().is_extend_in_batch:
        from sglang.srt.layers.moe.ppu_mega_moe_ultra_ep import prepare_prefill

        # UltraEP can reroute IDs in place. Preserve the model's logical IDs.
        if routed_ids is topk_ids:
            routed_ids = routed_ids.clone()
        routed_ids, weight_event = prepare_prefill(experts, routed_ids)
        num_experts = ultra_ep.num_global_physical_experts
        replica = ultra_ep.replica_weight_pool()
        gemm_kwargs = {
            "l1_weights_replicas": (replica.w1, replica.w1_scale),
            "l2_weights_replicas": (replica.w2, replica.w2_scale),
        }
    key = (
        id(group),
        hidden_states.device,
        num_experts,
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
            num_experts=num_experts,
            num_max_tokens_per_rank=capacity,
            num_topk=top_k,
            hidden=hidden_size,
            intermediate_hidden=intermediate_size,
            mma_type="mxf4xmxf4",
            activation=activation,
        )
    buf = buffers[key]
    # The public PPU API stages int64 routes and applies routing weights once
    # in final combine. It owns all scratch/counter recycling, including graphs.
    deep_gemm.mega_moe_pre_dispatch(
        hidden_states,
        routed_ids,
        topk_weights.to(torch.float32).contiguous(),
        buf.x,
        buf.x_sf,
        buf.topk_idx,
        buf.topk_weights,
        num_tokens,
        group_size=32,
        mma_type="mxf4xmxf4",
    )
    y = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device=hidden_states.device
    )
    if weight_event is not None:
        weight_event.current_stream_wait()
    deep_gemm.fp4_fp4_mega_moe(
        y,
        experts.mega_l1_weights,
        experts.mega_l2_weights,
        buf,
        **gemm_kwargs,
        recipe=(1, 1, 32),
        activation=activation,
        activation_clamp=activation_clamp,
        fast_math=True,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return y
