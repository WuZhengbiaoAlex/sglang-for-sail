"""Host-side cost metadata associated with captured target-decode graphs."""

import logging
from contextlib import contextmanager

import torch


class GraphWorkloads:
    def __init__(self, layers, validator=None):
        self.layers = layers
        self.validator = validator
        self._profiles = {}
        self._recording = None

    @property
    def recording(self):
        return self._recording is not None

    @contextmanager
    def capture(self, graph_key):
        if self.recording:
            raise RuntimeError("Nested incremental workload capture")
        self._recording = {}
        try:
            yield
            values = [(1, 0, 1)] * self.layers
            for layer, profile in self._recording.items():
                values[layer] = profile
            if self.validator is not None:
                self.validator(graph_key, tuple(values))
            self._profiles[graph_key] = (
                torch.tensor(values, dtype=torch.int64, device="cpu")
                if any(value[1] for value in values)
                else None
            )
            if self._profiles[graph_key] is not None:
                logging.getLogger(__name__).info(
                    "Incremental EPLB kernel-cost profile: graph=%s, %d calibrated layers",
                    graph_key,
                    sum(value[1] > 0 for value in values),
                )
        finally:
            self._recording = None

    def record(self, layer, profile):
        if not self.recording:
            return
        if not 0 <= layer < self.layers:
            raise ValueError("Invalid incremental workload layer")
        if (
            len(profile) != 3
            or any(type(v) is not int for v in profile)
            or profile[0] <= 0
            or profile[1] < 0
            or profile[2] <= 0
        ):
            raise ValueError("Invalid kernel workload profile")
        previous = self._recording.get(layer)
        if previous is not None and previous != profile:
            raise RuntimeError("Kernel cost metadata changed within one graph capture")
        self._recording[layer] = profile

    def for_graph(self, graph_key):
        if graph_key not in self._profiles:
            raise RuntimeError("Missing workload metadata for captured decode graph")
        return self._profiles[graph_key]

    def for_batch(self, graph_key, global_num_tokens):
        # All planners must use the same objective. Mixed-size DP batches may
        # execute different graph shapes, so use token costs on every rank.
        if not global_num_tokens or min(global_num_tokens) <= 0:
            return None
        if len(set(global_num_tokens)) != 1:
            return None
        return self.for_graph(graph_key)


def validate_ep_profiles(graph_key, values):
    """One startup agreement check per graph; no decode-step collective."""
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().moe_ep_group.cpu_group
    local = (repr(graph_key), values)
    profiles = [None] * dist.get_world_size(group)
    dist.all_gather_object(profiles, local, group=group)
    if any(profile != local for profile in profiles):
        raise RuntimeError(
            "Incremental EPLB kernel cost profiles differ across EP ranks"
        )
