from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


def vocab_parallel_top1(
    local_logits: torch.Tensor,
    *,
    vocab_start: int,
    valid_columns: int,
    group: GroupCoordinator,
) -> torch.Tensor:
    """Greedy token IDs for rank-ordered, contiguous original-vocabulary shards."""
    rows, width = local_logits.shape
    if not 0 <= valid_columns <= width:
        raise ValueError("valid_columns must fit within the local logits shard")
    if rows == 0:
        return local_logits.new_empty((0,), dtype=torch.long)

    if valid_columns:
        values, ids = local_logits[:, :valid_columns].max(dim=-1)
        ids.add_(vocab_start)
    else:
        # Padding-only shards follow all real IDs. An all-negative-infinity
        # row still chooses the first valid ID on the first nonempty shard.
        values = local_logits.new_full((rows,), float("-inf"))
        ids = local_logits.new_full((rows,), vocab_start, dtype=torch.long)

    if group.world_size == 1:
        return ids

    # Only two row vectors cross the TP group, regardless of vocabulary size.
    # Match full-vocab argmax ties: max picks the first local ID, then argmax
    # picks the first matching rank in the original global vocabulary order.
    shape = (group.world_size, rows)
    all_values = values.new_empty(shape)
    all_ids = ids.new_empty(shape)
    group.all_gather_into_tensor(all_values.view(-1), values)
    group.all_gather_into_tensor(all_ids.view(-1), ids)
    winner = all_values.argmax(dim=0, keepdim=True)
    return all_ids.gather(0, winner).squeeze(0)
