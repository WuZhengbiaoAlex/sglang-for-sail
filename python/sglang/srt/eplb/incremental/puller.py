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
"""Issue IncrementalEPLB weight pulls outside the decode CUDA graph."""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import List, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb.incremental.transport import PeerCopyTransport

logger = logging.getLogger(__name__)

_LOG_EVERY = 20


class CopyBatch(NamedTuple):
    sources: list[int]
    destinations: list[int]
    lengths: list[int]
    source_ranks: list[int]
    experts: int
    nbytes: int


class ExpertPuller:
    """Copies and stages each rebalance plan while the current graph runs."""

    def __init__(
        self,
        planner,
        address_tables,
        *,
        stream: Optional[torch.cuda.Stream] = None,
        transport=None,
    ):
        self.transport = PeerCopyTransport() if transport is None else transport
        self.planner = planner
        self.tables = address_tables
        self.pull_width = int(planner.pull_width)
        num_local = int(address_tables.num_local)
        num_homes = int(address_tables.num_homes)
        if num_local != planner.npe or num_homes != planner.E // planner.R:
            raise ValueError("Arena geometry disagrees with the placement")

        self._last_seq = planner.published_count
        self._pull = planner.pull_list.numpy()

        self._layers = [int(layer) for layer in address_tables.moe_layers]

        self._stream = stream or torch.cuda.Stream()
        self._event = torch.cuda.Event()
        self._inflight = False
        self._boundary_active = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="incremental-eplb-pull"
        )
        self._pending: Optional[Future] = None
        self._closed = False

        self._inflight_generation: Optional[int] = None
        self._active_generation = None
        self._nstep = 0
        self._since_log = 0
        self.stats = dict(
            decode_steps=0,
            boundaries=0,
            batches=0,
            adopted=0,
            adoption_late=0,
            submission_late=0,
            submission_wait_seconds=0.0,
            empty=0,
            experts=0,
            copies=0,
            bytes=0,
        )
        self._debug = None
        self._debug_round = None
        debug_dir = envs.SGLANG_INCREMENTAL_EPLB_DEBUG_DIR.get()
        if debug_dir:
            from sglang.srt.eplb.incremental.transfer_debug import TransferDebug

            self._debug = TransferDebug(
                Path(debug_dir) / "transfers",
                rank=planner.ep_rank,
                ranks_per_node=planner.rpn,
                event_factory=torch.cuda.Event,
            )

    def _log_boundary(self) -> None:
        if not _LOG_EVERY:
            return
        self._since_log += 1
        if self._since_log >= _LOG_EVERY:
            self._since_log = 0
            logger.info("IncrementalEPLB %s", self.summary())

    def _activate_inflight(self) -> None:
        """Activate routing that was staged during the preceding graph."""
        if not self._inflight:
            return
        if self._inflight_generation is None:
            raise RuntimeError("in-flight copy batch has no publication generation")
        late = self.planner.activate_staged()
        self._active_generation = self._inflight_generation
        self._inflight_generation = None
        self._inflight = False
        self.stats["adopted"] += 1
        if not late:
            return
        self.stats["adoption_late"] += 1
        if self.stats["adoption_late"] == 1:
            logger.warning(
                "IncrementalEPLB: staged routing was not ready at its adoption "
                "boundary; inspect planning and transfer timings"
            )

    def _consume_publication(self):
        """Consume CPU records produced by this worker's completed plan call."""
        generation, has_work = self.planner.publication
        if generation != self._last_seq + 1 or type(has_work) is not bool:
            raise RuntimeError(
                f"Invalid IncrementalEPLB publication {self.planner.publication}; "
                f"expected generation {self._last_seq + 1}"
            )
        self._last_seq = generation
        return generation, has_work

    def _expand_batch(self, moves=None) -> CopyBatch:
        """Expand expert pulls into raw region copies, validating every record."""
        pull = self._pull
        peer_base = self.tables.peer_base
        slot_offset = self.tables.slot_offset
        region_bytes = self.tables.region_bytes
        bytes_per_expert = self.tables.bytes_per_expert
        local_base = self.tables.local_base
        lo = 0
        hi = self.tables.num_local

        source_ranks = []
        src_out: List[int] = []
        dst_out: List[int] = []
        nb_out: List[int] = []
        add_src = src_out.append
        add_dst = dst_out.append
        add_nb = nb_out.append
        experts = 0
        n_bytes = 0

        for layer in self._layers:
            peers = peer_base[layer]
            lbase = local_base[layer]
            off = slot_offset[layer]
            rbytes = region_bytes[layer]
            bpe = bytes_per_expert[layer]
            row = pull[layer].tolist()
            for i in range(self.pull_width):
                src_rank, src_slot, dst_slot, record_layer = row[i]
                if src_rank == -1:
                    continue
                if src_rank < -1:
                    raise RuntimeError(
                        "pull list could not find a stable source for layer "
                        f"{layer}, destination slot {dst_slot}"
                    )
                if record_layer != layer:
                    raise RuntimeError(
                        f"pull list row {layer} names layer {record_layer}"
                    )
                if src_rank >= self.tables.ep_size:
                    raise RuntimeError(
                        f"pull list names source rank {src_rank}, outside EP"
                        f"{self.tables.ep_size}"
                    )
                if not (0 <= src_slot < self.tables.num_local):
                    raise RuntimeError(
                        f"pull list names source slot {src_slot}, outside "
                        f"[0, {self.tables.num_local})"
                    )
                if not (lo <= dst_slot < hi):
                    raise RuntimeError(
                        f"pull list names destination slot {dst_slot}, outside "
                        f"writable slots [{lo}, {hi})"
                    )
                sbase = peers[src_rank]
                for a, b, nbytes in zip(off[src_slot], off[dst_slot], rbytes):
                    source_ranks.append(src_rank)
                    add_src(sbase + a)
                    add_dst(lbase + b)
                    add_nb(nbytes)
                experts += 1
                n_bytes += bpe
                if moves is not None:
                    moves.append(
                        self._debug.move(
                            layer,
                            src_rank,
                            src_slot,
                            dst_slot,
                            bpe,
                            expert=int(self.planner.pull_experts[layer, i]),
                        )
                    )
        return CopyBatch(src_out, dst_out, nb_out, source_ranks, experts, n_bytes)

    def begin_step(self, *, batch_size=None, original_batch_size=None) -> None:
        """Snapshot loads and adopt the previous plan before decode."""
        if self._closed:
            raise RuntimeError("IncrementalEPLB puller is closed")
        if self._boundary_active:
            raise RuntimeError("previous boundary did not finish after replay")
        self._nstep += 1
        self.stats["decode_steps"] += 1
        # Finish host submission before reusing publication buffers or
        # appending the next snapshot wait to the planner stream.
        wait_start = time.perf_counter() if self._debug else None
        submissions_late = self.stats["submission_late"]
        self._finish_pending()
        self._debug_round = None
        if self._debug:
            self._debug_round = self._debug.new_round(self._nstep)
            self._debug_round.data.update(
                host_boundary_time_ns=time.time_ns(),
                forward_batch_size=batch_size,
                original_batch_size=original_batch_size,
                prior_generation=self._inflight_generation,
                prior_submission_join_host_ms=(time.perf_counter() - wait_start) * 1000,
                prior_submission_late=self.stats["submission_late"] > submissions_late,
            )
            self._debug_round.mark("boundary_start", torch.cuda.current_stream())
        self.planner.prepare_step()

        self.stats["boundaries"] += 1
        self._log_boundary()
        adoptions_late = self.stats["adoption_late"]
        self._activate_inflight()
        if self._debug_round is not None:
            self._debug_round.data["prior_adoption_late"] = (
                self.stats["adoption_late"] > adoptions_late
            )
            self._debug_round.mark("boundary_end", torch.cuda.current_stream())
            self._debug_round.mark("target_start", torch.cuda.current_stream())
        self._boundary_active = True

    def after_replay(self) -> None:
        """Queue planning and publication after the decode graph is enqueued."""
        if not self._boundary_active:
            return
        if self._pending is not None:
            raise RuntimeError("previous IncrementalEPLB submission is still pending")

        if self._debug_round is not None:
            self._debug_round.mark("target_end", torch.cuda.current_stream())
            self._debug_round.data["worker_queued_monotonic_ns"] = time.monotonic_ns()
        self._pending = self._executor.submit(self._submit_on_device)
        self._boundary_active = False

    def _finish_pending(self) -> None:
        pending = self._pending
        if pending is None:
            return
        late = not pending.done()
        start = time.perf_counter()
        # Retain a failed future: no later boundary may silently resume after
        # an incomplete generation. result() propagates the worker exception.
        pending.result()
        if late:
            self.stats["submission_late"] += 1
            self.stats["submission_wait_seconds"] += time.perf_counter() - start
        self._pending = None

    def close(self) -> None:
        """Attempt every cleanup even when the worker or a peer has failed."""
        if self._closed:
            return
        self._closed = True
        errors = []

        def attempt(operation):
            try:
                operation()
            except Exception as exc:
                errors.append(exc)
                logger.exception("Incremental EPLB cleanup failed")

        attempt(self._finish_pending)
        attempt(lambda: self._executor.shutdown(wait=True))

        def drain_streams():
            with torch.cuda.device(self._stream.device):
                attempt(self._stream.synchronize)
                attempt(self.planner.plan_stream.synchronize)

        attempt(drain_streams)
        if not self.transport.requires_peer_memory:
            # Incoming reads must finish too. A failed peer must not leave
            # graceful shutdown waiting for the default Gloo timeout.
            attempt(self.transport.synchronize_shutdown)
        attempt(self.transport.close)
        if self._debug:

            def close_debug():
                with torch.cuda.device(self._stream.device):
                    self._debug.close()

            attempt(close_debug)
        if errors:
            raise errors[0]

    def _submit_on_device(self) -> None:
        # CUDA device/stream selection is thread-local. All GPU operations
        # below use the existing planner or copy stream, never a worker default.
        with torch.cuda.device(self._stream.device), torch.inference_mode():
            if self._debug_round is not None:
                self._debug_round.mark("plan_start", self.planner.plan_stream)
            self.planner.launch_plan()
            if self._debug_round is not None:
                self._debug_round.mark("plan_end", self.planner.plan_stream)
            self._submit_publication()

    def _submit_publication(self) -> None:
        """Consume one publication and enqueue copies plus staged adoption."""
        trace = self._debug_round if self._debug else None
        if trace:
            trace.data["worker_queue_host_ms"] = (
                time.monotonic_ns() - trace.data["worker_queued_monotonic_ns"]
            ) / 1e6
            publication_start = time.perf_counter()
        generation, global_work = self._consume_publication()
        if trace:
            trace.data.update(
                generation=generation,
                slot_selection="per_layer_rank",
                global_work=global_work,
                publication_host_ms=(time.perf_counter() - publication_start) * 1000,
                local_experts=0,
                local_region_copies=0,
                local_bytes=0,
            )
        if not global_work:
            self.stats["empty"] += 1
            if trace:
                self._debug.submitted(trace)
            return
        if trace:
            expand_start = time.perf_counter()
            batch = self._expand_batch(moves=trace.data["moves"])
            trace.data["expand_host_ms"] = (time.perf_counter() - expand_start) * 1000
        else:
            batch = self._expand_batch()

        self.planner.assert_copy_safe()
        self.planner.commit()
        if trace:
            copy_start = time.perf_counter()
            if batch.sources:
                trace.mark("copy_start", self._stream)
        if batch.sources:
            self.transport.submit(
                batch.sources,
                batch.destinations,
                batch.lengths,
                batch.source_ranks,
                stream=self._stream,
            )
        if trace:
            trace.data["copy_submit_host_ms"] = (
                time.perf_counter() - copy_start
            ) * 1000
            if batch.sources:
                trace.mark("copy_end", self._stream)
        self._event.record(self._stream)
        self.planner.stage_adoption(generation, self._event)
        if trace:
            trace.mark("adoption_ready", self.planner.plan_stream)
        self._inflight = True
        self._inflight_generation = generation
        self.stats["batches"] += 1
        self.stats["experts"] += batch.experts
        self.stats["copies"] += len(batch.sources)
        self.stats["bytes"] += batch.nbytes
        if trace:
            trace.data.update(
                local_experts=batch.experts,
                local_region_copies=len(batch.sources),
                local_bytes=batch.nbytes,
            )
            self._debug.submitted(trace)

    def summary(self) -> str:
        """Return one-line control-loop statistics."""
        s = self.stats
        boundaries = max(1, s["boundaries"])
        decode_steps = max(1, s["decode_steps"])
        return (
            f"pull: {s['decode_steps']} decode steps, {s['boundaries']} plans, "
            f"{s['batches']} copy batches, "
            f"{s['adoption_late']} staged adoptions late, "
            f"{s['submission_late']} host submissions late "
            f"({s['submission_wait_seconds'] * 1000:.3f} ms boundary wait), "
            f"{s['empty']} empty, "
            f"{s['adopted']} placements adopted, "
            f"{s['experts'] / boundaries:.2f} experts/boundary, "
            f"{s['copies'] / boundaries:.1f} copies/boundary, "
            f"{s['bytes'] / boundaries / 2**20:.2f} MiB/boundary, "
            f"{s['bytes'] / decode_steps / 2**20:.2f} MiB/decode-step, "
            f"active generation {self._active_generation}, "
            f"in-flight generation {self._inflight_generation}"
        )


def build_expert_puller(planner, mover) -> ExpertPuller:
    """Validate and wire the planner to the peer-copy path."""
    if planner is None or mover is None:
        raise ValueError("planner and mover are required")
    tables = mover.address_tables()
    puller = ExpertPuller(planner, tables, transport=mover.transport)

    logger.info(
        "IncrementalEPLB: side-stream pulls, up to %d destination slots/layer, "
        "%d regions/expert, %d physical slots/rank",
        puller.pull_width,
        tables.max_regions,
        tables.num_local,
    )
    return puller
