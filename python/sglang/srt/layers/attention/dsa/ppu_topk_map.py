"""One-launch logical TopK to rank-local KV slot mapping on PPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ppu_map_logical_topk_to_physical_kernel(
    logical,
    page_table,
    output,
    n_elements,
    topk: tl.constexpr,
    page_columns: tl.constexpr,
    logical_row_stride: tl.constexpr,
    logical_col_stride: tl.constexpr,
    table_row_stride: tl.constexpr,
    table_col_stride: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // topk
    col = offsets % topk
    index = tl.load(
        logical + row * logical_row_stride + col * logical_col_stride,
        mask=mask,
        other=-1,
    )
    valid = mask & (index >= 0) & (index < page_columns * PAGE_SIZE)
    safe_index = tl.where(valid, index, 0)
    page = tl.load(
        page_table
        + row * table_row_stride
        + (safe_index // PAGE_SIZE) * table_col_stride,
        mask=valid,
        other=0,
    )
    slot = page * PAGE_SIZE + safe_index % PAGE_SIZE
    tl.store(output + offsets, tl.where(valid, slot, -1), mask=mask)


def map_logical_topk_to_physical_triton(logical_indices, page_table, page_size):
    """Inputs are validated by map_logical_topk_to_physical before dispatch.

    Own each output allocation, including during graph capture. Strides are
    explicit so expanded query page tables and sliced inputs need no copies.
    """
    output = torch.empty(
        logical_indices.shape, dtype=torch.int32, device=logical_indices.device
    )
    count = logical_indices.numel()
    if count:
        _ppu_map_logical_topk_to_physical_kernel[(triton.cdiv(count, 256),)](
            logical_indices,
            page_table,
            output,
            count,
            logical_indices.shape[1],
            page_table.shape[1],
            *logical_indices.stride(),
            *page_table.stride(),
            PAGE_SIZE=page_size,
            BLOCK=256,
            num_warps=4,
        )
    return output
