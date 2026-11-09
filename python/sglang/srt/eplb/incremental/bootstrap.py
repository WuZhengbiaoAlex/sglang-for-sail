# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model-runner initialization for IncrementalEPLB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
import torch
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    get_global_expert_location_metadata,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_parallel,
    get_schedule,
    get_spec,
    remote_instance_transfer_engine_enabled,
)

if TYPE_CHECKING:
    from sglang.srt.eplb.incremental.decode_bindings import (
        IncrementalEPLBDecodeBindings,
    )
    from sglang.srt.eplb.incremental.mover import ExpertWeightMover
    from sglang.srt.eplb.incremental.planner import IncrementalPlanner
    from sglang.srt.eplb.incremental.puller import ExpertPuller

logger = logging.getLogger(__name__)


class IncrementalEPLBComponents(msgspec.Struct, frozen=True):
    planner: IncrementalPlanner
    mover: ExpertWeightMover
    puller: ExpertPuller
    decode_bindings: IncrementalEPLBDecodeBindings


def rebuild_initial_placement(*, model_config, meta, moe_ep_rank):
    """Initialize contiguous expert homes and two spare slots per rank."""
    moe_config = get_exec().moe
    algorithm = moe_config.ep_dispatch_algorithm
    if algorithm != "static":
        raise ValueError(
            f"IncrementalEPLB requires --ep-dispatch-algorithm static; got {algorithm}"
        )
    if get_lora().enable_lora:
        raise ValueError(
            "IncrementalEPLB does not support LoRA because physical expert replicas "
            "do not have corresponding adapter slots"
        )
    if get_spec().speculative_algorithm not in (None, "DSPARK", "DFLASH"):
        raise ValueError(
            "IncrementalEPLB supports DSPARK, DFLASH, or non-speculative decode"
        )
    if get_disagg().enable_pdmux:
        raise ValueError(
            "IncrementalEPLB does not support PDMux because concurrent prefill and "
            "decode streams do not share a placement boundary"
        )
    if get_disagg().disaggregation_mode == "prefill":
        raise ValueError(
            "IncrementalEPLB is decode-only and cannot run on a prefill server"
        )
    if moe_config.moe_a2a_backend not in ("deepep", "megamoe"):
        raise ValueError("IncrementalEPLB requires --moe-a2a-backend deepep or megamoe")
    if moe_config.moe_a2a_backend == "deepep" and moe_config.deepep_mode not in (
        "auto",
        "low_latency",
    ):
        raise ValueError("IncrementalEPLB requires --deepep-mode auto or low_latency")
    if not moe_config.disable_shared_experts_fusion:
        raise ValueError("IncrementalEPLB requires --disable-shared-experts-fusion")
    if moe_config.expert_distribution_recorder_mode is not None:
        raise ValueError(
            "IncrementalEPLB cannot be combined with expert distribution recording"
        )
    if get_exec().graph.cuda_graph_config.decode.backend == Backend.DISABLED:
        raise RuntimeError("IncrementalEPLB requires decode CUDA graphs")
    if get_schedule().num_continuous_decode_steps != 1:
        raise RuntimeError("IncrementalEPLB requires --num-continuous-decode-steps 1")
    if moe_config.elastic_ep_backend is not None:
        raise RuntimeError("IncrementalEPLB is incompatible with elastic EP")
    if remote_instance_transfer_engine_enabled():
        raise RuntimeError(
            "IncrementalEPLB is incompatible with remote-instance weight transfer"
        )
    if meta is None:
        raise RuntimeError("IncrementalEPLB setup failed: no expert metadata")
    P = int(meta.num_physical_experts)
    E = int(meta.num_logical_experts)
    R = int(meta.ep_size)
    if P <= E:
        raise ValueError(
            f"IncrementalEPLB needs redundant experts ({P} physical, {E} logical)"
        )
    if P % R or E % R:
        raise ValueError(
            f"IncrementalEPLB needs physical ({P}) and logical ({E}) expert counts "
            f"divisible by EP size {R}"
        )

    npe, epr = P // R, E // R
    spares = npe - epr
    if spares != 2:
        raise ValueError(
            "IncrementalEPLB needs exactly two spare slots per "
            f"rank at initialization; got {spares}"
        )

    row = torch.empty(P, dtype=torch.int64)
    for rank in range(R):
        base = rank * npe
        row[base : base + epr] = torch.arange(
            rank * epr, (rank + 1) * epr, dtype=torch.int64
        )
        for i in range(epr, npe):
            row[base + i] = rank * epr + (i - epr) % epr

    rebuilt = ExpertLocationMetadata.init_by_mapping(
        model_config=model_config,
        physical_to_logical_map=row.unsqueeze(0).repeat(meta.num_layers, 1),
        moe_ep_rank=moe_ep_rank,
    )
    if rebuilt is None:
        raise RuntimeError("IncrementalEPLB failed to build expert metadata")
    if rebuilt.logical_to_rank_dispatch_physical_map is None:
        raise RuntimeError("IncrementalEPLB failed to build static routing metadata")
    # Initial prefill routes exclude spare slots; unified publication updates
    # this map to the primary owner when home slots are reused.
    home_map = rebuilt.logical_to_rank_dispatch_physical_map
    logical = torch.arange(E, dtype=home_map.dtype, device=home_map.device)
    home_map.copy_((logical // epr * npe + logical % epr).expand_as(home_map))
    logger.info(
        "IncrementalEPLB: initial placement has %d homes + two spare slots per rank",
        epr,
    )
    return rebuilt


def init_incremental_eplb(
    *,
    tp_size,
    moe_ep_rank,
    device,
    gpu_id,
    model,
    transport=None,
    transport_factory=None,
) -> IncrementalEPLBComponents:
    """Create the planner, symmetric weight arenas, and copy controller."""
    from sglang.srt.eplb.incremental import planner as planner_module

    meta = get_global_expert_location_metadata()
    if meta is None:
        raise RuntimeError("IncrementalEPLB expert metadata is missing")
    if transport is not None and transport_factory is not None:
        raise ValueError("Pass transport or transport_factory, not both")
    peer_memory = transport_factory is None and (
        transport is None or transport.requires_peer_memory
    )
    if not planner_module.is_available(peer_memory=peer_memory):
        raise RuntimeError(
            "IncrementalEPLB setup failed: plan extension unavailable: "
            f"{planner_module.extension_error(peer_memory=peer_memory)}"
        )

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    ep_group = get_moe_ep_group().device_group
    nnodes = get_parallel().nnodes
    if nnodes < 1 or tp_size % nnodes:
        raise ValueError(
            f"IncrementalEPLB requires TP size {tp_size} to divide evenly across "
            f"{nnodes} node(s)"
        )
    gpus_per_node = tp_size // nnodes
    ranks_per_node = min(meta.ep_size, gpus_per_node)
    if meta.ep_size % ranks_per_node:
        raise ValueError(
            f"IncrementalEPLB EP size {meta.ep_size} is incompatible with "
            f"{gpus_per_node} GPU rank(s) per node"
        )
    # Construct the external engine only after extension and layout validation.
    if transport_factory is not None:
        transport = transport_factory()
    try:
        coordination = None
        if transport is not None and not transport.requires_peer_memory:
            from sglang.srt.eplb.incremental.transport import HostCollectiveExchange

            coordination = HostCollectiveExchange(transport.cpu_group)
        planner = planner_module.IncrementalPlanner(
            num_layers=meta.num_layers,
            num_logical_experts=meta.num_logical_experts,
            num_physical_experts=meta.num_physical_experts,
            ep_size=meta.ep_size,
            device=torch.device(device, gpu_id),
            ep_group=ep_group,
            ep_rank=moe_ep_rank,
            ranks_per_node=ranks_per_node,
            coordination=coordination,
        )
        if planner.resident_p2l.shape != meta.physical_to_logical_map.shape:
            raise RuntimeError(
                f"IncrementalEPLB placement shape {tuple(meta.physical_to_logical_map.shape)} "
                f"does not match planner shape {tuple(planner.resident_p2l.shape)}"
            )
        for placement in (
            planner.resident_p2l,
            planner.planned_p2l,
            planner.next_p2l,
        ):
            placement.copy_(meta.physical_to_logical_map)

        mover, puller, n_moe = _init_weight_pull(
            model=model,
            moe_ep_rank=moe_ep_rank,
            planner=planner,
            meta=meta,
            transport=transport,
        )
        from sglang.srt.eplb.incremental.decode_bindings import (
            IncrementalEPLBDecodeBindings,
        )

        decode_bindings = IncrementalEPLBDecodeBindings.from_planner(meta, planner)
        debug_directory = envs.SGLANG_INCREMENTAL_EPLB_DEBUG_DIR.get()
        if debug_directory:
            from dataclasses import replace

            from sglang.srt.eplb.incremental.balance_debug import BalanceDebug

            balance_debug = BalanceDebug(
                Path(debug_directory) / "balance",
                planner,
                planner.owners.flatten(1),
            )
            balance_debug.install(model)
            decode_bindings = replace(decode_bindings, balance_debug=balance_debug)
        logger.info(
            "Incremental EPLB: home-slot reuse, split %.3fx / merge %.3fx mean rank load, "
            "2 copies/layer/round; prefill uses primary owners without splitting or counting",
            float(planner.split_target),
            float(planner.merge_target),
        )
        logger.info(
            "IncrementalEPLB ready: EP%d rank %d, %d MoE layers, %d initial homes + "
            "two spare slots/rank, %d ranks/node",
            meta.ep_size,
            moe_ep_rank,
            n_moe,
            planner.npe - planner.spares_total,
            planner.rpn,
        )
        return IncrementalEPLBComponents(
            planner=planner, mover=mover, puller=puller, decode_bindings=decode_bindings
        )
    except Exception:
        if transport is not None:
            try:
                transport.close()
            except Exception:
                logger.exception("Incremental EPLB initialization cleanup failed")
        raise


def _init_weight_pull(*, model, moe_ep_rank, planner, meta, transport=None):
    from sglang.srt.eplb.incremental.mover import (
        ExpertWeightMover,
        collect_layer_weights,
        collect_weight_aliases,
        validate_mega_moe_experts,
    )
    from sglang.srt.eplb.incremental.puller import build_expert_puller

    dispatch_map = meta.logical_to_rank_dispatch_physical_map
    if dispatch_map is None:
        raise RuntimeError(
            "IncrementalEPLB failed to initialize its physical routing table"
        )
    # Capture keeps separate decode and prefill routing pointers.
    prefill_map = dispatch_map
    dispatch_map = dispatch_map.clone()
    planner.attach_dispatch_tensor(dispatch_map)
    planner.attach_prefill_dispatch(prefill_map)
    planner.build_dispatch()

    if get_exec().moe.moe_a2a_backend == "megamoe":
        validate_mega_moe_experts(model)

    num_local_physical_experts = meta.num_physical_experts // meta.ep_size
    layer_weights = collect_layer_weights(
        model,
        meta.num_layers,
        num_local_physical_experts=num_local_physical_experts,
    )
    moe_layers = [i for i, weights in enumerate(layer_weights) if weights]
    if not moe_layers:
        raise RuntimeError("IncrementalEPLB found no MoE expert weights")

    # Validate the first routing table before capture or weight movement.
    torch.cuda.current_stream().synchronize()
    layers = torch.tensor(moe_layers, dtype=torch.int64, device=dispatch_map.device)
    routes = dispatch_map.index_select(0, layers).long()
    if bool(((routes < 0) | (routes >= planner.P)).any()):
        raise RuntimeError("IncrementalEPLB built an out-of-range route")
    placement = planner.resident_p2l.index_select(0, layers).long()
    logical = torch.arange(
        planner.E, dtype=torch.int64, device=routes.device
    ).expand_as(routes)
    wrong = placement.gather(1, routes) != logical
    fill_start = planner.npe - planner.spares_total
    unsafe = routes.remainder(planner.npe) >= fill_start
    if bool(wrong.any()) or bool(unsafe.any()):
        raise RuntimeError(
            "IncrementalEPLB initial routing failed validation: "
            f"{int(wrong.sum())} wrong-expert and "
            f"{int(unsafe.sum())} initial spare-slot routes"
        )

    mover = ExpertWeightMover(
        layer_weights,
        num_local_physical_experts=num_local_physical_experts,
        ep_group=planner.ep_group,
        ep_rank=moe_ep_rank,
        ep_size=meta.ep_size,
        num_spare_slots=planner.spares_total,
        transport=transport,
        weight_aliases=collect_weight_aliases(model, layer_weights),
    )
    mover.setup()
    puller = build_expert_puller(
        planner,
        mover,
    )

    tables = mover.address_tables()
    logger.info(
        "IncrementalEPLB weight pull: %d regions/expert, %.1f MiB/expert, "
        "%.2f GiB symmetric arena",
        tables.max_regions,
        max(tables.bytes_per_expert) / 2**20,
        mover.arena_bytes / 2**30,
    )
    return mover, puller, len(moe_layers)
