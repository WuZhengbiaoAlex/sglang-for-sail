"""Static request partitions for the replicated PPU verification indexer.

The metadata belongs to one attention backend and one graph bucket. Its tensors
are refreshed in place before replay. Index-K projection and cache writes cover
all queries. Q and head weights may be projected only for this rank's requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndexerRequestShardMetadata:
    full_batch_size: int
    next_n: int
    tp_rank: int
    tp_size: int
    ctx_lens_2d: torch.Tensor
    schedule_metadata: torch.Tensor
    topk_v2_plan: torch.Tensor

    def __post_init__(self) -> None:
        if self.tp_size <= 0 or self.full_batch_size % self.tp_size:
            raise ValueError("Indexer request shards require equal request partitions")
        if not 0 <= self.tp_rank < self.tp_size or self.next_n <= 0:
            raise ValueError("Invalid indexer request shard rank or query width")
        if self.ctx_lens_2d.shape != (self.local_batch_size, self.next_n):
            raise ValueError("Indexer shard context lengths do not match its partition")

    @property
    def local_batch_size(self) -> int:
        return self.full_batch_size // self.tp_size

    @property
    def query_start(self) -> int:
        return self.tp_rank * self.local_batch_size * self.next_n

    @property
    def query_end(self) -> int:
        return self.query_start + self.local_batch_size * self.next_n


def map_logical_topk_to_physical(
    logical_indices: torch.Tensor,
    expanded_page_table: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    """Resolve gathered logical positions using this rank's compact page table.

    Both inputs have one row per query. Invalid positions are sanitized *before*
    gather, then restored to -1. This keeps TP ranks independent of one another's
    physical cache allocation. Allocations stay on the current stream and are
    capture-safe; no per-layer output buffer is shared with a subsequent layer.
    """
    if logical_indices.ndim != 2 or expanded_page_table.ndim != 2:
        raise ValueError("TopK positions and page tables must be two-dimensional")
    if logical_indices.shape[0] != expanded_page_table.shape[0]:
        raise ValueError("TopK positions and page tables must have equal query rows")
    if page_size <= 0 or expanded_page_table.shape[1] == 0:
        raise ValueError("A positive page size and nonempty page table are required")
    if logical_indices.dtype != torch.int32 or expanded_page_table.dtype != torch.int32:
        raise ValueError("TopK positions and compact page tables must use int32")
    if logical_indices.device != expanded_page_table.device:
        raise ValueError("TopK positions and page tables must be on the same device")

    if (
        logical_indices.is_cuda
        and expanded_page_table.shape[1] * page_size <= 2147483647
    ):
        from sglang.srt.utils import is_ppu

        if is_ppu():
            from sglang.srt.layers.attention.dsa.ppu_topk_map import (
                map_logical_topk_to_physical_triton,
            )

            return map_logical_topk_to_physical_triton(
                logical_indices, expanded_page_table, page_size
            )

    valid = (logical_indices >= 0) & (
        logical_indices < expanded_page_table.shape[1] * page_size
    )
    safe_indices = torch.where(valid, logical_indices, 0)
    page_indices = torch.div(safe_indices, page_size, rounding_mode="floor")
    pages = torch.gather(expanded_page_table, 1, page_indices.to(torch.int64))
    physical = pages * page_size + safe_indices.remainder(page_size)
    return torch.where(valid, physical, -1)
