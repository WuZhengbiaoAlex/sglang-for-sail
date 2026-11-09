"""Opt-in decode diagnostics. Timed runs are not throughput baselines."""

import json
import logging
import socket
import time
from functools import wraps
from pathlib import Path

logger = logging.getLogger(__name__)


def decode_buffer_proxy(dispatcher):
    inners = getattr(dispatcher, "_inners", None)
    if inners is not None:
        if len(inners) != 1:
            raise ValueError("balance diagnostics does not support two-batch overlap")
        dispatcher = inners[0]
    return dispatcher._low_latency_dispatcher._incremental_eplb_buffer


def layer_record(layer, rank, local_slots, placement, dispatch, counts, timings):
    """Keep physical residency, execution ownership, and received work distinct."""
    base = rank * local_slots
    owned = set(dispatch)
    experts = [
        dict(
            slot=slot,
            expert=placement[base + slot],
            owns_routing=base + slot in owned,
            tokens=int(tokens),
            gemm_ms=None,
        )
        for slot, tokens in enumerate(counts)
    ]
    return dict(
        layer=layer,
        stored_slots=sum(e["expert"] >= 0 for e in experts),
        distinct_resident_experts=len(
            {e["expert"] for e in experts if e["expert"] >= 0}
        ),
        owned_experts=sum(e["owns_routing"] for e in experts),
        active_experts=sum(e["tokens"] > 0 for e in experts),
        total_tokens=sum(e["tokens"] for e in experts),
        experts=experts,
        **timings,
    )


class BalanceDebug:
    """Instrument only target decode graphs; no changes when not installed.

    External CUDA events remain graph nodes on replay. Each graph owns its own
    event set and token snapshot. Each replay synchronizes the main stream
    before reading them, so the next replay cannot overwrite the observation.
    This deliberate diagnostic synchronization invalidates end-to-end timing
    comparisons. GPU stage spans retain their actual stream scope.
    """

    def __init__(self, directory, planner, dispatch):
        import torch

        self.torch = torch
        self.planner, self.dispatch = planner, dispatch
        self.capture_key = None
        self.layer = None
        self.gemm_index = 0
        self.events, self.counts = {}, {}
        self.layers = []
        self.mega_moe = False
        self.batch_size = None
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / (
            f"balance-{socket.gethostname()}-ep{planner.ep_rank}-{time.time_ns()}.jsonl"
        )
        with self.path.open("x") as output:
            output.write(
                json.dumps(
                    dict(
                        type="header",
                        rank=planner.ep_rank,
                        schema=1,
                        timing_scope="GPU stream spans, instrumented decode; not baseline TPOT",
                        offset_scope="Per-device offsets from target replay start; not synchronized across devices",
                        per_expert_gemm_time="unavailable: experts share a grouped GEMM; not apportioned",
                        token_scope="received target-verification token assignments, not accepted output tokens",
                    )
                )
                + "\n"
            )
        logger.info("IncrementalEPLB balance diagnostics: %s", self.path)

    def measure(self, layer, name, function, *args, **kwargs):
        if self.capture_key is None:
            return function(*args, **kwargs)
        events = self.events.setdefault(self.capture_key, {})
        pair = events.get((layer, name))
        if pair is None:
            pair = tuple(
                self.torch.cuda.Event(enable_timing=True, external=True)
                for _ in range(2)
            )
            events[layer, name] = pair
        pair[0].record()
        result = function(*args, **kwargs)
        pair[1].record()
        return result

    def install(self, model):
        """Wrap this target model's expert cores and DeepEP proxy instances."""
        from sglang.srt.eplb.incremental.config import is_routed_expert_module
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.runtime_context import get_exec

        if get_exec().moe.moe_a2a_backend == "megamoe":
            self._install_mega(model)
            return

        for name in (
            "grouped_gemm_nt_f4f4bf16_masked",
            "grouped_gemm_nt_f8f8bf16_masked",
            "grouped_gemm_nt_bf16_masked",
        ):
            original = getattr(deep_gemm_wrapper, name, None)
            if original is not None:
                setattr(deep_gemm_wrapper, name, self._wrap_gemm(original))
        for module in model.modules():
            if not hasattr(module, "run_moe_core") or not is_routed_expert_module(
                module
            ):
                continue
            layer = module.layer_id
            proxy = decode_buffer_proxy(module.dispatcher)
            self.layers.append(layer)
            self._wrap_core(module, layer)
            self._wrap_dispatch(proxy, layer)
            self._wrap_combine(proxy, layer)
        if not self.layers or len(self.layers) != len(set(self.layers)):
            raise RuntimeError("balance diagnostics needs unique routed MoE layer IDs")

    def _install_mega(self, model):
        import deep_gemm
        from sglang.srt.eplb.incremental.mover import _iter_moe_modules

        self.mega_moe = True
        self.layers = [layer for layer, _ in _iter_moe_modules(model)]
        if not self.layers or len(self.layers) != len(set(self.layers)):
            raise RuntimeError("balance diagnostics needs unique routed MoE layer IDs")
        self._mega_count_layers = {
            self.planner.phys[layer].data_ptr(): layer for layer in self.layers
        }
        for name in ("fp4_fp4_mega_moe", "fp8_fp4_mega_moe"):
            original = getattr(deep_gemm, name, None)
            if original is not None:
                setattr(deep_gemm, name, self._wrap_mega(original))

    def _wrap_mega(self, original):
        @wraps(original)
        def kernel(*args, **kwargs):
            stats = kwargs.get("cumulative_local_expert_recv_stats")
            if self.capture_key is None or stats is None:
                return original(*args, **kwargs)
            layer = self._mega_count_layers.get(stats.data_ptr())
            if layer is None:
                raise RuntimeError("MegaMoE diagnostics received an unknown counter")
            before = stats.clone()
            result = self.measure(layer, "mega_moe_ms", original, *args, **kwargs)
            self.counts.setdefault(self.capture_key, {})[layer] = stats - before
            return result

        return kernel

    def _wrap_gemm(self, original):
        @wraps(original)
        def gemm(*args, **kwargs):
            if self.capture_key is None or self.layer is None:
                return original(*args, **kwargs)
            index = self.gemm_index
            self.gemm_index += 1
            if index > 1:
                raise RuntimeError(
                    "balance diagnostics expects two grouped GEMMs per layer"
                )
            return self.measure(
                self.layer, f"gemm{index}_ms", original, *args, **kwargs
            )

        return gemm

    def _wrap_core(self, module, layer):
        original = module.run_moe_core

        @wraps(original)
        def core(*args, **kwargs):
            old_layer, old_index = self.layer, self.gemm_index
            self.layer, self.gemm_index = layer, 0
            try:
                return self.measure(
                    layer, "expert_compute_ms", original, *args, **kwargs
                )
            finally:
                self.layer, self.gemm_index = old_layer, old_index

        module.run_moe_core = core

    def _wrap_dispatch(self, proxy, layer):
        original = proxy.low_latency_dispatch

        @wraps(original)
        def dispatch(*args, **kwargs):
            result = self.measure(layer, "dispatch_send_ms", original, *args, **kwargs)
            if self.capture_key is None:
                return result
            *outputs, hook = result
            if not kwargs.get("return_recv_hook") or hook is None:
                raise RuntimeError("balance diagnostics requires DeepEP receive hooks")

            @wraps(hook)
            def receive():
                result = self.measure(layer, "dispatch_receive_ms", hook)
                counts = outputs[1]
                snapshots = self.counts.setdefault(self.capture_key, {})
                if layer not in snapshots:
                    snapshots[layer] = self.torch.empty_like(counts)
                snapshots[layer].copy_(counts)
                return result

            return (*outputs, receive)

        proxy.low_latency_dispatch = dispatch

    def _wrap_combine(self, proxy, layer):
        def combine(*args, **kwargs):
            result = self.measure(
                layer,
                "combine_send_ms",
                proxy._buffer.low_latency_combine,
                *args,
                **kwargs,
            )
            if self.capture_key is None:
                return result
            output, event, hook = result
            if not kwargs.get("return_recv_hook") or hook is None:
                raise RuntimeError("balance diagnostics requires DeepEP receive hooks")

            @wraps(hook)
            def receive():
                return self.measure(layer, "combine_receive_ms", hook)

            return output, event, receive

        proxy.low_latency_combine = combine

    def install_runner(self, runner):
        if runner.enable_torch_compile:
            raise ValueError("balance diagnostics requires torch.compile disabled")
        original = runner.backend.capture_one

        @wraps(original)
        def capture(key, *args, **kwargs):
            old = self.capture_key
            self.capture_key = key
            try:
                return original(key, *args, **kwargs)
            finally:
                self.capture_key = old

        runner.backend.capture_one = capture

    def after_replay(self, key, puller):
        step = puller._nstep
        if key not in self.counts or set(self.counts[key]) != set(self.layers):
            raise RuntimeError("balance diagnostics did not capture all expert layers")
        # Polling alone cannot protect graph-owned events from the next replay.
        self.torch.cuda.current_stream().synchronize()
        placement = self.planner.resident_p2l.cpu().tolist()
        dispatch = self.dispatch.cpu().tolist()
        trace = getattr(puller, "_debug_round", None)
        origin = trace.events.get("target_start") if trace is not None else None
        rows = []
        for layer in sorted(self.layers):
            timings = {}
            for name in (
                "mega_moe_ms",
                "dispatch_send_ms",
                "dispatch_receive_ms",
                "expert_compute_ms",
                "gemm0_ms",
                "gemm1_ms",
                "combine_send_ms",
                "combine_receive_ms",
            ):
                pair = self.events[key].get((layer, name))
                if pair is None and (
                    self.mega_moe or name in ("mega_moe_ms", "gemm0_ms", "gemm1_ms")
                ):
                    timings[name] = None
                    continue
                if pair is None:
                    raise RuntimeError(f"missing balance timing: layer {layer} {name}")
                start, end = pair
                if not start.query() or not end.query():
                    raise RuntimeError(
                        f"incomplete balance timing for layer {layer}: {name}"
                    )
                timings[name] = start.elapsed_time(end)
                if origin is not None:
                    timings[name[:-3] + "_start_ms"] = origin.elapsed_time(start)
                    timings[name[:-3] + "_end_ms"] = origin.elapsed_time(end)
            timings["gemm_total_ms"] = (
                timings["gemm0_ms"] + timings["gemm1_ms"]
                if timings["gemm0_ms"] is not None and timings["gemm1_ms"] is not None
                else None
            )
            timings["dispatch_total_ms"] = (
                timings["dispatch_send_ms"] + timings["dispatch_receive_ms"]
                if not self.mega_moe
                else None
            )
            timings["combine_total_ms"] = (
                timings["combine_send_ms"] + timings["combine_receive_ms"]
                if not self.mega_moe
                else None
            )
            rows.append(
                layer_record(
                    layer,
                    self.planner.ep_rank,
                    self.planner.npe,
                    placement[layer],
                    dispatch[layer],
                    self.counts[key][layer].cpu().tolist(),
                    timings,
                )
            )
        record = dict(
            type="decode_balance",
            rank=self.planner.ep_rank,
            step=step,
            routing_generation=puller._active_generation,
            next_plan_generation=self.planner.published_count,
            batch_size=self.batch_size,
            graph_key=str(key),
            layers=rows,
        )
        with self.path.open("a") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
