"""Opt-in, bounded transfer diagnostics; never wait for GPU timing events."""

import json
import logging
import os
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class RoundTrace:
    def __init__(self, step, event_factory):
        self.data = {"step": step, "moves": []}
        self.events = {}
        self.event_factory = event_factory

    def mark(self, name, stream):
        event = self.event_factory(enable_timing=True)
        event.record(stream)
        self.events[name] = event

    def timings(self):
        if not all(event.query() for event in self.events.values()):
            return None
        pairs = {
            "boundary_stream_ms": ("boundary_start", "boundary_end"),
            "plan_stream_ms": ("plan_start", "plan_end"),
            "target_replay_region_ms": ("target_start", "target_end"),
            "copy_stream_ms": ("copy_start", "copy_end"),
        }
        result = {
            name: self.events[start].elapsed_time(self.events[end])
            for name, (start, end) in pairs.items()
            if start in self.events and end in self.events
        }
        origin = self.events["boundary_start"]
        for name in ("plan_end", "target_end", "copy_end", "adoption_ready"):
            if name in self.events:
                result[name + "_from_boundary_ms"] = origin.elapsed_time(
                    self.events[name]
                )
        if "copy_end" in self.events:
            result["copy_done_minus_target_end_ms"] = (
                result["copy_end_from_boundary_ms"]
                - result["target_end_from_boundary_ms"]
            )
        return result


class TransferDebug:
    """One writer per EP rank; all per-round producers use nonblocking queues.

    Timing polling runs on the pull worker, after copy submission. Shutdown
    reports incomplete timings instead of synchronizing the device. Event
    recording and Python bookkeeping still perturb debug runs.
    """

    def __init__(
        self,
        directory,
        *,
        rank,
        ranks_per_node,
        event_factory,
        capacity=256,
    ):
        self.rank = rank
        self.ranks_per_node = ranks_per_node
        self.event_factory = event_factory
        self.capacity = capacity
        self.pending = deque()
        self.queue = queue.Queue(maxsize=capacity)
        self.closing = threading.Event()
        self.failed = threading.Event()
        self.dropped_records = 0
        self.incomplete_timings = 0
        self.closed = False
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (
            f"transfer-{socket.gethostname()}-ep{rank}-pid{os.getpid()}-"
            f"{time.time_ns()}.jsonl"
        )
        # Fail early on an unwritable path, before inference starts.
        output = self.path.open("x")
        self.thread = threading.Thread(
            target=self._write,
            args=(output,),
            name="incremental-eplb-debug",
            daemon=True,
        )
        self.emit(
            {
                "type": "header",
                "schema": 1,
                "rank": rank,
                "ranks_per_node": ranks_per_node,
                "apply_routing": True,
                "timing_scope": "target replay region, not full speculative decode step",
            }
        )
        self.thread.start()
        logger.info("IncrementalEPLB transfer diagnostics: %s", self.path)

    def _write(self, output):
        try:
            with output:
                while not self.closing.is_set() or not self.queue.empty():
                    try:
                        record = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")
                    output.flush()
                output.write(
                    json.dumps(
                        {
                            "type": "footer",
                            "rank": self.rank,
                            "dropped_records": self.dropped_records,
                            "incomplete_timings": self.incomplete_timings,
                        }
                    )
                    + "\n"
                )
        except Exception:
            self.failed.set()
            logger.exception("IncrementalEPLB debug writer failed: %s", self.path)

    def emit(self, record):
        if self.failed.is_set():
            self.dropped_records += 1
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped_records += 1

    def new_round(self, step):
        return RoundTrace(step, self.event_factory)

    def move(
        self, layer, source_rank, source_slot, destination_slot, nbytes, *, expert
    ):
        return {
            "layer": layer,
            "expert": expert,
            "src_rank": source_rank,
            "src_slot": source_slot,
            "dst_rank": self.rank,
            "dst_slot": destination_slot,
            "src_node": source_rank // self.ranks_per_node,
            "dst_node": self.rank // self.ranks_per_node,
            "bytes": nbytes,
        }

    def submitted(self, trace):
        self.emit({"type": "round", "rank": self.rank, **trace.data})
        self.poll()
        if len(self.pending) >= self.capacity:
            self._incomplete(self.pending.popleft(), "pending_limit")
        self.pending.append(trace)

    def _incomplete(self, trace, reason):
        self.incomplete_timings += 1
        self.emit(
            {
                "type": "timing_incomplete",
                "rank": self.rank,
                "step": trace.data["step"],
                "generation": trace.data["generation"],
                "reason": reason,
            }
        )

    def poll(self):
        # Do not reuse timing events, and never synchronize to read them.
        remaining = deque()
        while self.pending:
            trace = self.pending.popleft()
            timings = trace.timings()
            if timings is None:
                remaining.append(trace)
            else:
                self.emit(
                    {
                        "type": "timing",
                        "rank": self.rank,
                        "step": trace.data["step"],
                        "generation": trace.data["generation"],
                        **timings,
                    }
                )
        self.pending = remaining

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.poll()
        while self.pending:
            self._incomplete(self.pending.popleft(), "shutdown_not_ready")
        self.closing.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            logger.warning("IncrementalEPLB debug writer not drained: %s", self.path)
