# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Weight transport for incremental EPLB.

Mooncake registration and one-sided reads follow PR #45282 (776f60a2).
Destinations are planner-reserved execution slots rather than a staging layer.
"""

from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta


class PeerCopyTransport:
    requires_peer_memory = True

    def submit(self, sources, destinations, lengths, source_ranks, *, stream):
        from . import copy_engine

        copy_engine.issue_batch(sources, destinations, lengths, stream=stream)

    def close(self):
        pass


class MooncakeTransport:
    requires_peer_memory = False

    def __init__(self, engine, cpu_group, *, memory_pool=None):
        self.engine = engine
        self.cpu_group = cpu_group
        self.sessions = []
        self._registered = []
        self._arenas = []
        self._memory_pool = memory_pool

    @staticmethod
    def _create_memory_pool(gpu_id):
        import os

        from sglang.srt.environ import envs

        if envs.MOONCAKE_PROTOCOL.get() != "nvlink":
            return None
        # TE's topology discovery may prefer an HCA despite protocol="nvlink".
        conflicts = (
            "MC_FORCE_HCA",
            "MC_FORCE_TCP",
            "MC_INTRANODE_NVLINK",
            "MC_INTRA_NVLINK",
        )
        if os.environ.get("MC_FORCE_MNNVL") != "1" or any(
            name in os.environ for name in conflicts
        ):
            raise ValueError(
                "Incremental EPLB nvlink requires MC_FORCE_MNNVL=1 and no "
                "MC_FORCE_HCA/MC_FORCE_TCP/MC_INTRANODE_NVLINK/MC_INTRA_NVLINK overrides"
            )
        from sglang.srt.disaggregation.mooncake.utils import (
            init_mooncake_custom_mem_pool,
        )

        enabled, pool, _ = init_mooncake_custom_mem_pool(
            f"cuda:{gpu_id}", custom_mem_pool_type="NVLINK"
        )
        if not enabled or pool is None:
            raise RuntimeError("Mooncake NVLink expert memory pool is unavailable")
        return pool

    @classmethod
    def create(cls, *, gpu_id):
        from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
            init_mooncake_transfer_engine,
        )
        from sglang.srt.distributed.parallel_state import get_moe_ep_group
        from sglang.srt.runtime_context import get_disagg, get_exec
        from sglang.srt.utils.network import get_local_ip_auto

        memory_pool = cls._create_memory_pool(gpu_id)
        engine = init_mooncake_transfer_engine(
            hostname=get_local_ip_auto(),
            gpu_id=gpu_id,
            ib_device=get_disagg().disaggregation_ib_device
            or get_exec().moe.mooncake_ib_device,
        )
        import torch.distributed as dist

        # Keep worker collectives off the inference thread's CPU communicator.
        # EP groups are disjoint, so only their members need to join creation.
        cpu_group = dist.new_group(
            ranks=list(get_moe_ep_group().ranks),
            backend="gloo",
            use_local_synchronization=True,
            timeout=timedelta(seconds=60),
        )
        return cls(engine, cpu_group, memory_pool=memory_pool)

    def allocate_arena(self, size, device):
        import torch

        context = (
            torch.cuda.use_mem_pool(self._memory_pool, device=device)
            if self._memory_pool is not None
            else nullcontext()
        )
        with context:
            return torch.empty(size, dtype=torch.uint8, device=device)

    def _registration_regions(self, arenas):
        if self._memory_pool is None:
            return [(t.data_ptr(), t.numel() * t.element_size(), t) for t in arenas]
        # NVLink exports backing allocations, not caching-allocator subblocks.
        # Register each allocation once and deregister that same base address.
        segments = self._memory_pool.snapshot()
        regions = {}
        for tensor in arenas:
            ptr = tensor.data_ptr()
            end = ptr + tensor.numel() * tensor.element_size()
            covering = [
                s
                for s in segments
                if s["address"] <= ptr and end <= s["address"] + s["total_size"]
            ]
            if len(covering) != 1:
                raise RuntimeError(
                    "Mooncake arena must belong to one NVLink pool allocation"
                )
            segment = covering[0]
            base, size = segment["address"], segment["total_size"]
            if base not in regions:
                regions[base] = (base, size, [])
            regions[base][2].append(tensor)
        return list(regions.values())

    def prepare(self, mover):
        import torch
        import torch.distributed as dist

        # Finish initial weight copies before exposing these addresses to peers.
        torch.cuda.synchronize()
        local = {layer: tensor.data_ptr() for layer, tensor in mover._symm.items()}
        for ptr, size, storage in self._registration_regions(mover._symm.values()):
            ret = self.engine.engine.register_memory(ptr, size)
            if ret != 0:
                raise RuntimeError(f"Mooncake expert registration failed: {ret}")
            self._registered.append(ptr)
            self._arenas.append(storage)
        gathered = [None] * dist.get_world_size(self.cpu_group)
        dist.all_gather_object(
            gathered, (self.engine.get_session_id(), local), group=self.cpu_group
        )
        if len(gathered) != mover.ep_size:
            raise ValueError("Mooncake address exchange must use the EP group")
        self.sessions = [session for session, _ in gathered]
        for rank, (_, addresses) in enumerate(gathered):
            if addresses.keys() != local.keys():
                raise ValueError("MoE layers differ across Mooncake EP peers")
            for layer, ptr in addresses.items():
                mover._peer_base[rank, layer] = ptr

    def submit(self, sources, destinations, lengths, source_ranks, *, stream):
        if not (len(sources) == len(destinations) == len(lengths) == len(source_ranks)):
            raise ValueError("Mismatched Mooncake transfer descriptors")
        batches = defaultdict(lambda: ([], [], []))
        for src, dst, size, rank in zip(sources, destinations, lengths, source_ranks):
            if not 0 <= rank < len(self.sessions) or min(src, dst, size) <= 0:
                raise ValueError("Invalid Mooncake transfer descriptor")
            local, remote, sizes = batches[rank]
            local.append(dst)
            remote.append(src)
            sizes.append(size)
        # Runs on the pull worker. Sync-read completion precedes publication;
        # a CUDA event alone cannot represent an outstanding NIC transfer.
        for rank in sorted(batches):
            local, remote, sizes = batches[rank]
            ret = self.engine.engine.batch_transfer_sync_read(
                self.sessions[rank], local, remote, sizes
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake expert read from EP{rank} failed: {ret}")

    def close(self):
        """Attempt all registrations; retain storage whose deregistration failed."""
        errors, registered, arenas = [], [], []
        for ptr, arena in zip(self._registered, self._arenas):
            try:
                ret = self.engine.engine.unregister_memory(ptr)
                if ret != 0:
                    raise RuntimeError(f"Mooncake expert deregistration failed: {ret}")
            except Exception as exc:
                errors.append(exc)
                registered.append(ptr)
                arenas.append(arena)
        self._registered, self._arenas = registered, arenas
        if self.cpu_group is not None:
            import torch.distributed as dist

            try:
                dist.destroy_process_group(self.cpu_group)
                self.cpu_group = None
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def synchronize_shutdown(self):
        import torch.distributed as dist

        dist.monitored_barrier(
            group=self.cpu_group, timeout=timedelta(seconds=10), wait_all_ranks=True
        )


class HostCollectiveExchange:
    """Portable count and generation exchange, called only on the plan worker."""

    def __init__(self, cpu_group):
        self.cpu_group = cpu_group

    def gather(self, planner):
        import torch
        import torch.distributed as dist

        # Include the boundary in the existing collective, including no-copy
        # rounds. This catches different generations without another collective.
        generation = int(planner.published_count) + 1
        local = torch.empty(planner.phys_delta.numel() + 1, dtype=torch.int64)
        local[0] = generation
        local[1:].copy_(planner.phys_delta.flatten().cpu())
        peers = [torch.empty_like(local) for _ in range(planner.R)]
        dist.all_gather(peers, local, group=self.cpu_group)
        generations = [int(peer[0]) for peer in peers]
        if any(value != generation for value in generations):
            raise RuntimeError(
                f"Incremental EPLB count generations diverged: expected {generation}, "
                f"EP peers reported {generations}"
            )
        counts = torch.stack([peer[1:] for peer in peers])
        planner.gathered.copy_(counts.reshape_as(planner.gathered))

    def synchronize(self, generation, stream):
        import torch
        import torch.distributed as dist

        stream.synchronize()
        value = torch.tensor([generation], dtype=torch.int64)
        peers = [
            torch.empty_like(value) for _ in range(dist.get_world_size(self.cpu_group))
        ]
        dist.all_gather(peers, value, group=self.cpu_group)
        if any(int(peer.item()) != generation for peer in peers):
            raise RuntimeError("Incremental EPLB copy generations diverged")
