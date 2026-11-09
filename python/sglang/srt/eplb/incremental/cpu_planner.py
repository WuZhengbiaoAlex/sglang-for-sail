# Copyright 2026 SGLang Team. Licensed under the Apache License, Version 2.0.
"""Background CPU planning from authoritative GPU placement snapshots."""

import logging
import threading
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(None)
def load_cpu_planner():
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_incremental_cpu_planner",
        sources=[str(Path(__file__).parent / "csrc" / "cpu_planner.cpp")],
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        with_cuda=False,
        verbose=False,
    )


class CpuPlanner:
    def __init__(self, *, threads):
        if type(threads) is not int or not 1 <= threads <= 1024:
            raise ValueError("CPU planner threads must be in [1, 1024]")
        self.library = load_cpu_planner()
        self.threads = threads
        self.shape = None
        logging.getLogger(__name__).info("CPU incremental planner: %d threads", threads)

    def plan(
        self,
        tokens,
        resident,
        owners,
        ranks,
        budget,
        moves,
        ranks_per_node,
        split_numerator,
        split_denominator,
        merge_numerator,
        merge_denominator,
        gain_divisor,
    ):
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("CPU planner must execute in the background worker")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CPU planner cannot execute inside CUDA graph capture")
        tensors = (tokens, resident, owners)
        if tokens.ndim != 2 or resident.ndim != 2 or owners.ndim != 3:
            raise ValueError("Invalid planner dimensions")
        layers, experts = tokens.shape
        physical = resident.shape[1]
        if not (
            layers > 0
            and 0 < experts <= 1024
            and 0 < ranks <= 64
            and 0 < physical <= 1152
            and physical % ranks == 0
            and physical // ranks <= 64
        ):
            raise ValueError("Unsupported planner geometry")
        if resident.shape != (layers, physical) or owners.shape != (
            layers,
            experts,
            ranks,
        ):
            raise ValueError("Inconsistent planner geometry")
        if not all(t.is_cuda and t.device == tokens.device for t in tensors):
            raise ValueError("Planner inputs must share one CUDA device")
        if (
            tokens.dtype != torch.int64
            or resident.dtype != torch.int32
            or owners.dtype != torch.int32
        ):
            raise ValueError("Expected int64 counts and int32 layouts")
        if not all(t.is_contiguous() for t in tensors):
            raise ValueError("Contiguous planner inputs required")
        shape = (layers, experts, physical, ranks, tokens.device)
        if self.shape is None:
            self.inputs = [
                torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors
            ]
            self.outputs = [
                torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                for t in (resident, owners, resident)
            ]
            self.stream = torch.cuda.current_stream()
            self.ready = torch.cuda.Event()
            self.shape = shape
        elif shape != self.shape:
            raise RuntimeError("CPU planner geometry changed")
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("CPU planner must use one ordered CUDA stream")
        # Snapshots belong to the runtime's generation, not a second CPU state.
        for dst, src in zip(self.inputs, tensors):
            dst.copy_(src, non_blocking=True)
        self.ready.record(torch.cuda.current_stream())
        # This also drains the preceding upload before its pinned buffers are reused.
        self.ready.synchronize()
        return self.library.plan_into(
            *self.inputs,
            *self.outputs,
            ranks,
            budget,
            moves,
            ranks_per_node,
            split_numerator,
            split_denominator,
            merge_numerator,
            merge_denominator,
            gain_divisor,
            self.threads,
        )

    def stage(self, placement, owners, pulls, experts, rank):
        """Prepare CPU copy records and enqueue the matching device layout."""
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("CPU staging must execute in the background worker")
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("CPU staging must use the planner stream")
        for dst, src in zip((placement, owners), self.outputs[:2]):
            if (
                dst.device != self.shape[-1]
                or dst.dtype != src.dtype
                or dst.shape != src.shape
                or not dst.is_contiguous()
            ):
                raise ValueError("Invalid device placement output")
        self.library.prepare_pulls(
            self.outputs[2],
            self.inputs[2],
            self.inputs[1],
            self.outputs[0],
            pulls,
            experts,
            self.shape[3],
            rank,
        )
        placement.copy_(self.outputs[0], non_blocking=True)
        owners.copy_(self.outputs[1], non_blocking=True)
