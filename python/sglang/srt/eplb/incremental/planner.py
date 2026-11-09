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
"""IncrementalEPLB load collection, placement, and routing publication."""

from __future__ import annotations

import logging
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from . import config as policy
from .cpu_planner import CpuPlanner

logger = logging.getLogger(__name__)

_EXT = {}
_EXT_ERR = {}


def _load_ext(*, peer_memory=True):
    if peer_memory in _EXT or peer_memory in _EXT_ERR:
        return _EXT.get(peer_memory)
    try:
        from torch.utils.cpp_extension import load

        directory = Path(__file__).parent / "csrc"
        sources = [str(directory / "load_counts.cu")]
        flags = ["-O3"]
        if peer_memory:
            sources.append(str(directory / "peer_gather.cu"))
            flags.append("-DSGLANG_INCREMENTAL_PEER_MEMORY")
        _EXT[peer_memory] = load(
            name="sglang_incremental_load_peer"
            if peer_memory
            else "sglang_incremental_load",
            sources=sources,
            extra_cuda_cflags=flags,
            verbose=False,
        )
        logger.info("Incremental EPLB load-collection extension ready")
    except Exception as e:  # noqa: BLE001
        _EXT_ERR[peer_memory] = str(e)
        logger.warning("Incremental EPLB load-collection extension unavailable: %s", e)
    return _EXT.get(peer_memory)


def is_available(*, peer_memory=True) -> bool:
    return _load_ext(peer_memory=peer_memory) is not None


def extension_error(*, peer_memory=True) -> Optional[str]:
    return _EXT_ERR.get(peer_memory)


@lru_cache(None)
def load_runtime_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_incremental_runtime",
        sources=[str(Path(__file__).parent / "csrc" / "runtime.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class IncrementalPlanner:
    """Decode placement with home-slot reuse and load-relative split/merge."""

    def __init__(
        self,
        num_layers: int,
        num_logical_experts: int,
        num_physical_experts: int,
        ep_size: int,
        device: torch.device,
        ep_group=None,
        ep_rank: int = 0,
        ranks_per_node: int = 0,
        coordination=None,
        split_target=None,
        merge_target=None,
    ):
        if min(num_layers, num_logical_experts, num_physical_experts, ep_size) <= 0:
            raise ValueError("IncrementalEPLB dimensions must be positive")
        if (
            num_logical_experts % ep_size
            or num_physical_experts % ep_size
            or num_physical_experts <= num_logical_experts
        ):
            raise ValueError(
                "logical and physical experts must divide EP size, with "
                "physical > logical"
            )
        if not 0 <= ep_rank < ep_size:
            raise ValueError(f"EP rank {ep_rank} is outside EP size {ep_size}")
        if device.type != "cuda":
            raise ValueError(f"IncrementalEPLB requires a CUDA device, got {device}")
        self.L = num_layers
        self.E = num_logical_experts
        self.P = num_physical_experts
        self.R = ep_size
        self.coordination = coordination
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.rpn = ranks_per_node or ep_size
        if self.rpn <= 0 or ep_size % self.rpn:
            raise ValueError(f"ranks_per_node {self.rpn} must divide EP size {ep_size}")

        self.npe = num_physical_experts // ep_size
        self.spares_total = self.npe - num_logical_experts // ep_size
        if self.spares_total != 2:
            raise ValueError(
                "IncrementalEPLB needs exactly two spare slots per "
                "rank at initialization; got "
                f"{self.spares_total} "
                f"({num_physical_experts} physical - {num_logical_experts} "
                f"logical experts over EP{ep_size})"
            )
        z = dict(device=device, dtype=torch.int32)
        # DeepEP accumulates local physical-slot receive counts here.
        self.phys = torch.zeros((self.L, self.npe), **z)
        self.phys_snapshot = torch.zeros_like(self.phys)
        self.phys_delta = torch.zeros((self.L, self.npe), **z)
        # Int32 subtraction preserves deltas across counter wraparound.
        self.phys_read = torch.zeros((self.L, self.npe), **z)
        self.gathered = torch.zeros((self.R, self.L, self.npe), **z)
        self.load_step = torch.zeros(
            (self.L, self.E), device=device, dtype=torch.float32
        )
        self._peer_gather_ring_size = 4
        self._peer_gather_header_ints = 32
        self._pg_step = 1
        if self.R > 1 and self.coordination is None:
            self._setup_peer_gather()
        # Resident, in-flight, and scratch placements have independent storage.
        self.resident_p2l = torch.zeros((self.L, self.P), **z)
        self.count_p2l_snapshot = torch.zeros_like(self.resident_p2l)
        self.planned_p2l = torch.zeros((self.L, self.P), **z)
        self.next_p2l = torch.zeros((self.L, self.P), **z)
        self.dispatch = torch.zeros((self.L, self.E), **z)
        self._dispatch_out = None
        self._staged_dispatch = None
        # Published by the single CPU worker after copy records are complete.
        self.publication = (0, False)

        self.plan_stream = torch.cuda.Stream(device=device)
        self._ev_snapshot_ready = torch.cuda.Event()
        self._ev_adoption_ready = torch.cuda.Event()
        self._ev_activation_complete = torch.cuda.Event()
        self._counts_baselined = False
        self._adoption_staged = False
        self._activation_enqueued = False

        if split_target is None:
            split_target = policy.SPLIT_TARGET
        if merge_target is None:
            merge_target = policy.MERGE_TARGET
        ratio = Fraction(str(split_target))
        if not 0 < ratio <= 1 or ratio.denominator > 1_000_000:
            raise ValueError(
                "split_target must be in (0, 1] with at most six decimal places"
            )
        self.split_target = ratio
        self.merge_target = Fraction(str(merge_target))
        if (
            not 0 < self.merge_target < ratio
            or self.merge_target.denominator > 1_000_000
        ):
            raise ValueError("merge_target must be positive and below split_target")
        self.extension = load_runtime_extension()
        self._cpu_planner = CpuPlanner(threads=policy.CPU_PLANNER_THREADS)
        self.owners = torch.full(
            (self.L, self.E, self.R), -1, dtype=torch.int32, device=self.phys.device
        )
        logical = torch.arange(self.E, device=self.phys.device)
        epr = self.E // self.R
        self.owners[:, :, 0] = logical // epr * self.npe + logical % epr
        self.planned_owners = self.owners.clone()
        self.next_owners = self.owners.clone()
        self.pull_width = self.npe
        self.pull_list = torch.empty((self.L, self.npe, 4), dtype=torch.int32)
        self.pull_experts = torch.empty((self.L, self.npe), dtype=torch.int32)
        self.prefill_dispatch = None

    @property
    def published_count(self):
        return self.publication[0]

    def _setup_peer_gather(self) -> None:
        """Allocate the symmetric count ring and resolve peer addresses."""
        try:
            import torch.distributed._symmetric_memory as symm_mem

            if self.ep_group is None:
                raise RuntimeError("EP process group is missing")
            plane = int(self.L) * int(self.npe)
            slot_stride = ((self._peer_gather_header_ints + plane + 31) // 32) * 32
            count_ints = self._peer_gather_ring_size * slot_stride
            completion_ints = self._peer_gather_ring_size * 2
            total_ints = count_ints + completion_ints
            buf = symm_mem.empty(
                total_ints * 4, dtype=torch.uint8, device=self.load_step.device
            )
            buf.zero_()
            torch.cuda.current_stream(device=self.load_step.device).synchronize()
            handle = symm_mem.rendezvous(buf, group=self.ep_group)
            ptrs = []
            for r in range(self.R):
                pb = handle.get_buffer(
                    r, (total_ints * 4,), torch.uint8, storage_offset=0
                )
                ptrs.append(int(pb.data_ptr()))
            self._pg_buf = buf
            self._pg_handle = handle
            self._pg_region = buf.view(torch.int32)
            self._pg_peer_ptrs = torch.tensor(
                ptrs, dtype=torch.int64, device=self.load_step.device
            )
            self._pg_plane = plane
            self._pg_completion_offset = count_ints
            logger.info(
                "IncrementalEPLB: peer-pull counts gather armed -- %d ranks, %d-slot "
                "ring of %d ints (%.1f KiB/rank)",
                self.R,
                self._peer_gather_ring_size,
                plane,
                total_ints * 4 / 1024.0,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "IncrementalEPLB requires symmetric-memory peer-pull counts "
                f"collection, but setup failed ({type(e).__name__}: {e})"
            ) from e

    def collect_load(self) -> None:
        """Collect counts using the placement that produced their snapshot."""
        ext = _load_ext(peer_memory=self.R > 1 and self.coordination is None)
        if ext is None:
            raise RuntimeError("load-collection kernel unavailable")

        if self.R > 1 and self.coordination is not None:
            self.coordination.gather(self)
        elif self.R > 1:
            generation = self._pg_step
            ext.peer_gather(
                self._pg_peer_ptrs,
                self.gathered,
                self._pg_plane,
                self.R,
                generation,
            )
            self._pg_step = generation + 1
        else:
            self.gathered[0].copy_(self.phys_delta)
        ext.phys_to_logical(
            self.gathered,
            self.count_p2l_snapshot,
            self.load_step,
            self.R,
        )

    def _synchronize_copies(self, generation: int) -> None:
        """Wait until every rank has completed the copied generation."""
        if generation <= 0:
            raise ValueError(f"invalid IncrementalEPLB copy generation {generation}")
        if self.R == 1:
            return
        if self.coordination is not None:
            self.coordination.synchronize(generation, self.plan_stream)
            return
        ext = _load_ext()
        if ext is None:
            raise RuntimeError("plan kernel unavailable")
        ext.sync_copy_generation(
            self._pg_peer_ptrs,
            self._pg_region,
            self._pg_completion_offset,
            self.R,
            generation,
        )

    def attach_dispatch_tensor(self, t: torch.Tensor) -> None:
        """Attach SGLang's fixed-address logical-to-physical routing table."""
        if tuple(t.shape) != (self.L, self.E):
            raise ValueError(
                f"dispatch tensor shape {tuple(t.shape)} != ({self.L}, {self.E})"
            )
        if t.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"unsupported dispatch dtype {t.dtype}")
        if t.device != self.dispatch.device:
            raise ValueError(
                f"dispatch tensor on {t.device}, planner on {self.dispatch.device}"
            )
        if not t.is_contiguous():
            raise ValueError("dispatch tensor must be contiguous")
        self._dispatch_out = t
        self._staged_dispatch = torch.empty_like(t)

    def prepare_step(self) -> None:
        """Baseline counts and snapshot the preceding placement window."""
        main = torch.cuda.current_stream()
        if not self._counts_baselined:
            # Exclude warmup and capture from the cumulative count baseline.
            self.phys_read.copy_(self.phys)
            self._counts_baselined = True

        # begin_step's _finish_pending observes the previous publication only
        # after the planner has finished reading both snapshot buffers.
        self.phys_snapshot.copy_(self.phys)
        self.count_p2l_snapshot.copy_(self.resident_p2l)
        torch.sub(self.phys_snapshot, self.phys_read, out=self.phys_delta)
        self.phys_read.copy_(self.phys_snapshot)
        if self.R > 1 and self.coordination is None:
            # Publish before replay: a persistent MoE kernel can otherwise
            # occupy the device while a peer collector waits for these counts.
            ext = _load_ext()
            if ext is None:
                raise RuntimeError("load-collection kernel unavailable")
            ext.publish_counts(self._pg_region, self.phys_delta, self._pg_step)
        self._ev_snapshot_ready.record(main)
        self.plan_stream.wait_event(self._ev_snapshot_ready)

    def launch_plan(self) -> None:
        """Overlap the next plan with replay after routing activation."""
        side = self.plan_stream
        if self._activation_enqueued:
            side.wait_event(self._ev_activation_complete)
            self._activation_enqueued = False

        with torch.cuda.stream(side):
            self._plan_and_publish()

    def attach_prefill_dispatch(self, tensor):
        if (
            tensor.shape != self.dispatch.shape
            or tensor.data_ptr() == self._dispatch_out.data_ptr()
        ):
            raise ValueError(
                "Prefill requires a separate fixed-address primary-owner map"
            )
        self.prefill_dispatch = tensor

    def build_dispatch(self):
        out = self._dispatch_out if self._dispatch_out is not None else self.dispatch
        out.copy_(self.owners[:, :, 0])
        return out

    def assert_copy_safe(self):
        """Publication follows this boundary's snapshot of all prior main-stream reads.

        Destinations exclude every current owner, so the current graph can run
        concurrently. The snapshot event protects retired owners from the prior
        graph or prefill. Do not wait for the current graph: that defeats overlap.
        """
        if not self._ev_snapshot_ready.query():
            raise RuntimeError(
                "Incremental EPLB publication preceded its read-completion fence"
            )

    def plan(self):
        self.collect_load()
        return self._cpu_planner.plan(
            self.load_step.to(torch.int64),
            self.planned_p2l,
            self.planned_owners,
            self.R,
            policy.MAX_COPIES_PER_LAYER_PER_STEP,
            policy.MAX_OWNER_MOVES_PER_LAYER_PER_STEP,
            self.rpn,
            self.split_target.numerator,
            self.split_target.denominator,
            self.merge_target.numerator,
            self.merge_target.denominator,
            policy.EXCHANGE_GAIN_DIVISOR,
        )

    def _plan_and_publish(self):
        has_work = self.plan()
        if has_work:
            self._cpu_planner.stage(
                self.next_p2l,
                self.next_owners,
                self.pull_list,
                self.pull_experts,
                self.ep_rank,
            )
        # CPU records are ready; device uploads remain ordered before commit
        # and stage_adoption on plan_stream. The worker consumes this tuple.
        self.publication = (self.published_count + 1, has_work)

    def commit(self):
        with torch.cuda.stream(self.plan_stream):
            self.planned_p2l.copy_(self.next_p2l)
            self.planned_owners.copy_(self.next_owners)

    def stage_adoption(self, generation, copy_event):
        if self._adoption_staged:
            raise RuntimeError("Previous unified adoption is still pending")
        self.plan_stream.wait_event(copy_event)
        with torch.cuda.stream(self.plan_stream):
            self._synchronize_copies(generation)
            self._staged_dispatch.copy_(self.planned_owners[:, :, 0])
            self._ev_adoption_ready.record(self.plan_stream)
        self._adoption_staged = True

    def activate_staged(self):
        if not self._adoption_staged or self.prefill_dispatch is None:
            raise RuntimeError("Unified adoption is not ready for publication")
        main = torch.cuda.current_stream()
        late = not self._ev_adoption_ready.query()
        main.wait_event(self._ev_adoption_ready)
        self.resident_p2l.copy_(self.planned_p2l)
        self.owners.copy_(self.planned_owners)
        self._dispatch_out.copy_(self._staged_dispatch)
        # Main-stream order drains preceding decode/prefill reads first. The
        # next prefill uses only the primary; copies protect every current owner.
        self.prefill_dispatch.copy_(self._staged_dispatch)
        self._ev_activation_complete.record(main)
        self._activation_enqueued = True
        self._adoption_staged = False
        return late
