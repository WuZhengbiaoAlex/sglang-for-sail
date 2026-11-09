"""Target-decode routing and counters, isolated from primary-owner prefill dispatch.

Captured graphs retain separate routing pointers. Bindings require one Python
forward thread per worker; GPU streams may execute concurrently.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Optional

import torch
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.runtime_context import get_resources


@dataclass(frozen=True)
class IncrementalEPLBDecodeBindings:
    metadata: ExpertLocationMetadata
    counts: torch.Tensor
    balance_debug: object = None
    owners: Optional[torch.Tensor] = None
    source_rank: int = 0

    @classmethod
    def from_planner(cls, home_metadata, planner):
        live_map = planner._dispatch_out
        if live_map is None:
            raise RuntimeError("Attach the decode dispatch tensor before binding it")
        home_map = home_metadata.logical_to_rank_dispatch_physical_map
        if home_map is None:
            raise RuntimeError("Prefill expert routing metadata is missing")
        if live_map.data_ptr() == home_map.data_ptr():
            raise RuntimeError("Decode and prefill routing must not share storage")
        return cls(
            metadata=replace(
                home_metadata, logical_to_rank_dispatch_physical_map=live_map
            ),
            counts=planner.phys,
            owners=planner.owners,
            source_rank=planner.ep_rank,
        )

    @contextmanager
    def activate(self):
        """Bind decode launch arguments and restore the enclosing scope on exit."""
        global _CURRENT
        resources = get_resources()
        previous_metadata = resources.expert_location_metadata
        previous_bindings = _CURRENT
        resources.expert_location_metadata = self.metadata
        _CURRENT = self
        try:
            yield
        finally:
            _CURRENT = previous_bindings
            resources.expert_location_metadata = previous_metadata

    @contextmanager
    def capture(self):
        # Capture warmups must not contribute to measured decode counts.
        saved_counts = self.counts.clone()
        try:
            with self.activate():
                yield
        finally:
            self.counts.copy_(saved_counts)


# Plain Python state is traceable by torch.compile; ContextVar.get() is not.
_CURRENT: Optional[IncrementalEPLBDecodeBindings] = None


def dispatch_counts_buffer(layer_id: Optional[int]) -> Optional[torch.Tensor]:
    bindings = _CURRENT
    if bindings is None:
        return None
    if layer_id is None or not 0 <= layer_id < bindings.counts.shape[0]:
        raise RuntimeError(
            f"IncrementalEPLB received invalid expert layer id {layer_id}"
        )
    return bindings.counts[layer_id]


def route_decode_topk(topk_ids, layer_id, num_token_non_padded):
    """Return split-aware routes only in the target decode binding."""
    bindings = _CURRENT
    if bindings is None or bindings.owners is None:
        return None
    if not 0 <= layer_id < bindings.owners.shape[0]:
        raise RuntimeError(f"Invalid unified routing layer {layer_id}")
    if num_token_non_padded is None:
        raise RuntimeError("Unified decode routing requires the valid token count")
    from sglang.srt.eplb.incremental.planner import load_runtime_extension

    return load_runtime_extension().route(
        topk_ids,
        bindings.owners[layer_id],
        num_token_non_padded,
        bindings.source_rank,
    )
