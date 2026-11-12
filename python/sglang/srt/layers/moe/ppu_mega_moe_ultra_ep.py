"""Online UltraEP placement for the direct PPU SiTU MegaMoE prefill path."""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_buffer, get_parallel, get_server_args

logger = logging.getLogger(__name__)


def register_weights(experts, *, activation):
    """Put each layer's masters in symmetric memory and share one replica pool."""
    from ultra_ep import ppu

    args = get_server_args()
    if (
        activation != "situ"
        or args.enable_eplb
        or args.enable_eplb_async
        or args.ep_dispatch_algorithm is not None
        or args.ep_num_redundant_experts
        or args.init_expert_location != "trivial"
        or args.expert_distribution_recorder_mode is not None
        or args.enable_two_batch_overlap
        or args.speculative_algorithm is not None
        or experts.num_fused_shared_experts
    ):
        raise ValueError(
            "PPU UltraEP prefill requires SiTU MegaMoE with ordinary contiguous "
            "expert ownership, separate shared experts and no speculative draft"
        )

    group = get_parallel().moe_ep_group.device_group
    local_masters = int(experts.num_local_experts)
    replicas = envs.SGLANG_OPT_ULTRA_EP_PPU_REDUNDANT_PER_RANK.get()
    if replicas <= 0 or group.size() <= 1:
        raise ValueError(
            "PPU UltraEP requires positive replicas and multiple EP ranks"
        )
    if local_masters * group.size() != experts.num_experts:
        raise ValueError(
            "PPU UltraEP requires the same master expert count on every rank"
        )

    managers = get_buffer("ppu_mega_moe.ultra_ep_managers", dict)
    manager = managers.get(id(group))
    if manager is None:
        manager = ppu.Manager(
            group=group,
            num_local_master_experts=local_masters,
            num_local_redundant_experts=replicas,
        )
        managers[id(group)] = manager
        if group.rank() == 0:
            logger.info(
                "PPU UltraEP enabled for MegaMoE prefill: "
                "EP%d, %d masters and %d replicas per rank",
                group.size(),
                local_masters,
                replicas,
            )
    elif (
        manager.num_local_master_experts != local_masters
        or manager.num_local_redundant_experts != replicas
    ):
        raise ValueError("PPU UltraEP expert geometry must match across layers")

    torch.cuda.empty_cache()
    alloc = ppu.allocate_symmetric_expert_tensors(
        group, (*experts.mega_l1_weights, *experts.mega_l2_weights), 0
    )
    w1, s1, w2, s2 = alloc.tensors
    experts.w13_weight.data, experts.w13_weight_scale.data = w1, s1
    experts.w2_weight.data, experts.w2_weight_scale.data = w2, s2
    experts.mega_l1_weights = (w1, s1)
    experts.mega_l2_weights = (w2, s2)
    manager.register_physical_weight_pool(
        experts.layer_id,
        ppu.PhysicalWeightPool(
            w1=w1,
            w1_scale=s1,
            w1_peer_base_ptrs=alloc.peer_base_ptrs,
            w2=w2,
            w2_scale=s2,
            w2_peer_base_ptrs=alloc.peer_base_ptrs,
            keepalive=(alloc.backing,),
        ),
    )

    replica = manager.replica_weight_pool()
    if replica is None:
        replica_alloc = ppu.allocate_local_expert_tensors(alloc.tensors, replicas)
        r1, rs1, r2, rs2 = replica_alloc.tensors
        manager.register_replica_weight_pool(
            ppu.ReplicaWeightPool(
                w1=r1,
                w1_scale=rs1,
                w2=r2,
                w2_scale=rs2,
                keepalive=(replica_alloc.backing,),
            )
        )
    else:
        replica_tensors = (replica.w1, replica.w1_scale, replica.w2, replica.w2_scale)
        for weight, template in zip(
            replica_tensors, alloc.tensors
        ):
            if (
                weight.shape[1:] != template.shape[1:]
                or weight.stride() != template.stride()
                or weight.dtype != template.dtype
                or weight.device != template.device
            ):
                raise ValueError(
                    "PPU UltraEP layers must share the replica weight layout"
                )
    experts._ppu_mega_ultra_ep = manager


def prepare_prefill(experts, logical_ids):
    """Solve placement and start the replica pull before dispatch preparation."""
    manager = experts._ppu_mega_ultra_ep
    layer_id = experts.layer_id
    manager.update_placement(layer_id, logical_ids)
    event = manager.weight_sync(layer_id, async_finish=True)
    routed_ids = manager.reroute(layer_id, logical_ids)
    if manager.rank == 0 and not getattr(manager, "_sglang_prefill_routed", False):
        logger.info(
            "PPU UltraEP routed its first MegaMoE prefill batch: "
            "%d local tokens, %d physical experts",
            routed_ids.shape[0],
            manager.num_global_physical_experts,
        )
        manager._sglang_prefill_routed = True
    return routed_ids, event
