"""BF16 FlashMLA for the causal, sliding-window DFlash MLA draft.

Keep Triton's established DFlash metadata and graph lifecycle. Only the draft
block attention is replaced: sparse indices enumerate the exact causal window,
not a learned DSA top-k. The target model and its KV cache are independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


def build_dflash_causal_window_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    out_cache_loc: torch.Tensor,
    *,
    block_size: int,
    window_left: int,
    num_query_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each rectangular draft query to its visible physical KV slots.

    ``kv_indices[kv_indptr[r]:kv_indptr[r+1]]`` contains only request r's
    committed prefix. Its tentative block is separately addressed by
    ``out_cache_loc[r * block_size:(r + 1) * block_size]``. For prefix length P
    and block query j, the inclusive visible interval is
    ``[max(0, P + j - window_left), P + j]``. DFlash's HF window 1024 has
    already been translated to SGLang's window_left=1023 by the model.

    No tensor values are read on the CPU, and all result shapes depend only on
    capture-static dimensions. Trailing query padding has -1 indices and zero
    length. The PPU BF16 sparse entry point requires top-k aligned to 128.
    """
    if block_size <= 0 or window_left < 0 or num_query_rows < 0:
        raise ValueError("Expected a positive block size and a finite causal window.")
    batch_size = kv_indptr.numel() - 1
    if batch_size < 0:
        raise ValueError("kv_indptr must contain at least its initial zero.")
    topk = ((window_left + 1 + 127) // 128) * 128
    device = kv_indptr.device
    if batch_size == 0 or num_query_rows == 0 or out_cache_loc.numel() == 0:
        return (
            torch.full((num_query_rows, 1, topk), -1, dtype=torch.int32, device=device),
            torch.zeros(num_query_rows, dtype=torch.int32, device=device),
        )

    query = torch.arange(num_query_rows, dtype=torch.int64, device=device)
    request = query.div(block_size, rounding_mode="floor")
    valid_row = (request < batch_size) & (query < out_cache_loc.numel())
    request = request.clamp(max=batch_size - 1)
    within_block = query.remainder(block_size)
    prefix_start = kv_indptr[request].to(torch.int64)
    prefix_len = (kv_indptr[request + 1] - kv_indptr[request]).to(torch.int64)
    query_position = prefix_len + within_block
    first_key = (query_position - window_left).clamp(min=0)
    count = (query_position + 1).clamp(max=window_left + 1)
    count = torch.where(valid_row, count, 0)
    offset = torch.arange(topk, dtype=torch.int64, device=device)
    key_position = first_key[:, None] + offset[None, :]
    in_prefix = key_position < prefix_len[:, None]

    # torch.where evaluates both operands. Clamp each gather independently so
    # short/empty prefixes, padding and the other branch never address OOB.
    if kv_indices.numel():
        prefix_index = (prefix_start[:, None] + key_position).clamp(
            min=0, max=kv_indices.numel() - 1
        )
        prefix_slot = kv_indices[prefix_index]
    else:
        prefix_slot = torch.zeros_like(key_position)
    block_index = (
        request[:, None] * block_size + key_position - prefix_len[:, None]
    ).clamp(min=0, max=out_cache_loc.numel() - 1)
    block_slot = out_cache_loc[block_index]
    slot = torch.where(in_prefix, prefix_slot, block_slot)
    slot = torch.where(offset[None, :] < count[:, None], slot, -1)
    return slot.to(torch.int32).unsqueeze(1), count.to(torch.int32)


class DFlashFlashMLABackend(TritonAttnBackend):
    """Use BF16 sparse FlashMLA for DFlash's fixed-width causal draft block."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(model_runner, skip_prefill, kv_indptr_buf)
        if not model_runner.is_draft_worker or not self.use_mla:
            raise ValueError("DFlashFlashMLABackend is only for an MLA draft runner.")
        if self.dcp_size != 1 or self.use_sliding_window_kv_pool:
            raise ValueError(
                "DFlash BF16 FlashMLA requires the unsharded draft MLA KV pool."
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks=None,
        score_mod=None,
        aux_tensors=None,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        # Selecting the "flashmla" backend makes DeepSeek's absorbed MLA core
        # pass NoPE and RoPE separately. Triton's path passes concatenated rows.
        if q_rope is not None:
            q = torch.cat(
                (
                    q.reshape(-1, layer.tp_q_head_num, layer.v_head_dim),
                    q_rope.reshape(
                        -1, layer.tp_q_head_num, layer.qk_head_dim - layer.v_head_dim
                    ),
                ),
                dim=-1,
            )
        if k_rope is not None:
            if k is None:
                raise ValueError("k_rope requires the latent NoPE key tensor.")
            k = torch.cat(
                (
                    k.reshape(-1, 1, layer.v_head_dim),
                    k_rope.reshape(-1, 1, layer.qk_head_dim - layer.v_head_dim),
                ),
                dim=-1,
            )

        if not (
            forward_batch.forward_mode.is_target_verify()
            and forward_batch.spec_algorithm.is_dflash()
        ):
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                sinks=sinks,
                score_mod=score_mod,
                aux_tensors=aux_tensors,
            )

        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if q.shape[0] == 0 or forward_batch.batch_size == 0:
            return q.new_zeros((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))

        if (
            layer.attn_type != AttentionType.DECODER
            or layer.is_cross_attention
            or layer.sliding_window_size is None
            or layer.sliding_window_size < 0
            or self.forward_metadata.custom_mask is not None
            or sinks is not None
            or score_mod is not None
            or aux_tensors is not None
            or layer.logit_cap != 0
            or (layer.xai_temperature_len is not None and layer.xai_temperature_len > 0)
        ):
            raise ValueError(
                "DFlash BF16 FlashMLA supports only plain causal sliding-window MLA."
            )
        if (
            q.dtype != torch.bfloat16
            or layer.qk_head_dim != 576
            or layer.v_head_dim != 512
            or layer.k_scale is not None
            or layer.v_scale is not None
        ):
            raise ValueError("DFlash BF16 FlashMLA requires unscaled BF16 MLA 576/512.")

        if (k is None) != (v is None):
            raise ValueError("Both k and v should be present, or both already cached.")
        if k is not None and save_kv_cache:
            # Match Triton's write route, including a translated physical loc
            # when the pool supplies it. Context-prefix writes remain worker-owned.
            loc_info = KVWriteLoc(
                forward_batch.out_cache_loc,
                self.forward_metadata.swa_out_cache_loc,
                full_loc=self.forward_metadata.out_cache_loc_full_physical,
            )
            self._set_kv_buffer(forward_batch, layer, loc_info, k, v)

        kv = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if kv.dtype != torch.bfloat16 or kv.ndim != 3 or kv.shape[1:] != (1, 576):
            raise ValueError(
                "DFlash BF16 FlashMLA requires a BF16 [slots, 1, 576] pool."
            )
        block_size = self._target_verify_num_tokens_per_req(forward_batch.spec_info)
        indices, topk_length = build_dflash_causal_window_indices(
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            forward_batch.out_cache_loc,
            block_size=block_size,
            window_left=layer.sliding_window_size,
            num_query_rows=q.shape[0],
        )

        # The existing PPU BF16 sparse kernel uses 64-head tiles. Dummy heads
        # have independent softmaxes and are discarded; 12 need not divide 64.
        heads = layer.tp_q_head_num
        padded_heads = ((heads + 63) // 64) * 64
        if padded_heads != heads:
            q_input = q.new_zeros((q.shape[0], padded_heads, q.shape[-1]))
            q_input[:, :heads].copy_(q)
        else:
            q_input = q.contiguous()

        from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
            flash_mla_sparse_fwd,
        )

        output, _, _ = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv,
            indices=indices,
            sm_scale=layer.scaling,
            d_v=512,
            topk_length=topk_length,
        )
        output = output[:, :heads]
        # Fully padded rows may have an undefined all-masked softmax. Select
        # zero explicitly (multiplying NaN by zero would not remove it).
        output = torch.where(topk_length[:, None, None] > 0, output, 0)
        output = output.reshape(q.shape[0], heads * layer.v_head_dim)
        attn_out = getattr(forward_batch, "_attn_output", None)
        if attn_out is not None:
            attn_out.copy_(output)
            return attn_out
        return output
