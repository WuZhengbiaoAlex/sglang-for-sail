from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_ppu

# FlashKDA chunk size. Sequences shorter than this fall back to Triton.
_FLASHKDA_CHUNK_SIZE = 64

# FlashKDA's max sequence length, Batches whose longest sequence exceeds this
# fall back to Triton for the whole batch.
_FLASHKDA_MAX_SEQ_LEN = 2048


def _load_flash_kda():
    """Import the optional ``flash_kda`` CUTLASS module."""
    try:
        import flash_kda
    except ImportError as e:
        raise ImportError(
            "The 'flashkda' KDA backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git"
        ) from e
    return flash_kda


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    beta_is_raw=False,
    return_intermediate_states=False,
    track_state=None,
    track_chunk_idx=None,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor
    (or (output, h) when return_intermediate_states is set).
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        output_intermediate_states=return_intermediate_states,
        track_state=track_state,
        track_chunk_idx=track_chunk_idx,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA fused KDA prefill and PPU dense-chain verify backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    Prefill is bf16, K == V == 128, HV == H (no GVA), and requires the safe
    (bounded) gate (``lower_bound`` set). Unsupported layouts and snapshot
    requirements fall back to Triton. PPU verify uses the wheel's strided ABI
    for dense speculative chains and falls back for tree/ragged layouts.
    """

    # Tracked batches always take the Triton fallback, which forwards the
    # fp32 snapshot arguments (see _triton_fallback).
    supports_track_state_snapshot: bool = True

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Use the PPU FlashKDA dense-chain verify ABI when rollback is safe."""
        flash_kda = _load_flash_kda()
        batch_size = cache_indices.numel()
        num_heads = q.shape[2]
        if (
            q.ndim == 4
            and a.ndim == 3
            and a.shape == (q.shape[0], q.shape[1], num_heads * 128)
        ):
            a = a.view(q.shape)
        native = (
            is_ppu()
            and hasattr(flash_kda, "verify_strided")
            and lower_bound is not None
            and intermediate_states_buffer is not None
            and retrieve_parent_token is None
            and not kwargs.get("cache_ring", False)
            and kwargs.get("native_correction") is None
            and 0 < batch_size * num_heads <= 768
            and 0 < cache_steps <= 8
            and q.shape == k.shape == v.shape == a.shape
            and q.ndim == 4
            and q.shape[0] == 1
            and q.shape[1] == batch_size * cache_steps
            and q.shape[3] == 128
            and b.numel() == batch_size * cache_steps * num_heads
            and q.dtype == k.dtype == v.dtype == a.dtype == torch.bfloat16
            and b.dtype == torch.bfloat16
            and ssm_states.dtype in (torch.bfloat16, torch.float32)
            and intermediate_states_buffer.dtype == ssm_states.dtype
        )
        if not native:
            from sglang.srt.layers.attention.linear.kernels.kda_triton import (
                TritonKDAKernel,
            )

            return TritonKDAKernel().target_verify(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token,
                lower_bound=lower_bound,
                **kwargs,
            )

        shape = (batch_size, cache_steps, num_heads, 128)
        if not getattr(self, "_native_verify_logged", False):
            import logging

            self._native_verify_logged = True
            logging.getLogger(__name__).info(
                "FlashKDA native verify active: H=%s T=%s state=%s",
                num_heads,
                cache_steps,
                ssm_states.dtype,
            )
        out = torch.empty(shape, dtype=torch.bfloat16, device=q.device)
        flash_kda.verify_strided(
            q.view(shape),
            k.view(shape),
            v.view(shape),
            a.view(shape),
            b.reshape(batch_size, cache_steps, num_heads).contiguous(),
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(num_heads, 128).float().contiguous(),
            128**-0.5,
            lower_bound,
            ssm_states,
            cache_indices.contiguous(),
            intermediate_states_buffer,
            intermediate_state_indices[:batch_size].contiguous(),
            out,
        )
        return out.flatten(0, 1).unsqueeze(0)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        flash_kda = _load_flash_kda()
        if not is_ppu() or not hasattr(flash_kda, "decode"):
            raise NotImplementedError("FlashKDA decode requires the PPU extension")
        if lower_bound is not None and not getattr(
            flash_kda, "supports_decode_safe_gate", False
        ):
            raise RuntimeError("FlashKDA decode wheel lacks safe-gate support")
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
            raise ValueError("FlashKDA decode requires [1, batch, heads, 128]")
        batch, heads = q.shape[1:3]
        if q.shape != k.shape or q.shape != v.shape or cache_indices.numel() != batch:
            raise ValueError("FlashKDA decode requires one Q/K/V token per state slot")
        packed = torch.cat(
            [x.reshape(batch, heads * 128) for x in (q, k, v)], dim=-1
        ).contiguous()
        out = q.new_empty((batch, 1, heads, 128))
        flash_kda.decode(
            packed,
            a.reshape(batch, heads * 128).contiguous(),
            b.reshape(batch, heads).contiguous(),
            A_log.reshape(-1).float().contiguous(),
            dt_bias.reshape(heads, 128).float().contiguous(),
            128**-0.5,
            ssm_states,
            cache_indices.contiguous(),
            out,
            lower_bound=lower_bound,
        )
        if not getattr(self, "_native_decode_logged", False):
            import logging

            self._native_decode_logged = True
            logging.getLogger(__name__).info(
                "FlashKDA native decode active: H=%s state=%s", heads, ssm_states.dtype
            )
        return out.transpose(0, 1)

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # The fused kernel cannot expose per-chunk states (h), which the mamba
        # radix extra_buffer track path may need; route only batches requiring
        # an intermediate snapshot through Triton. Aligned-only tracking can
        # copy the committed final state without an h tensor.
        track_ssm_h_src = kwargs.get("track_ssm_h_src")
        final_state_only_track = (
            return_intermediate_states
            and track_ssm_h_src is not None
            and track_ssm_h_src.numel() == 0
            and kwargs.get("track_state") is None
        )
        snapshot_fallback = return_intermediate_states and not final_state_only_track
        if (
            q.shape[1] > 1
            and snapshot_fallback
            and not getattr(self, "_snapshot_fallback_logged", False)
        ):
            import logging

            self._snapshot_fallback_logged = True
            logging.getLogger(__name__).info(
                "FlashKDA prefill snapshot fallback: tokens=%s h_rows=%s",
                q.shape[1],
                None if track_ssm_h_src is None else track_ssm_h_src.numel(),
            )
        unsupported_ppu_layout = is_ppu() and not self._ppu_native_layout_supported(
            q, k, v, g, beta, ssm_states, cache_indices, query_start_loc, A_log, dt_bias
        )
        if (
            snapshot_fallback
            or unsupported_ppu_layout
            or self._should_fall_back(
                lower_bound, is_spec_decode, query_start_loc, extend_seq_lens_cpu
            )
        ):
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                track_state=kwargs.get("track_state"),
                track_chunk_idx=kwargs.get("track_chunk_idx"),
            )

        # Preserve the (out) vs (out, h) contract: non-tracked batches consume
        # the output tensor directly, while aligned-only tracked batches unpack
        # a tuple and copy the final recurrent state instead of an h snapshot.
        out = self._flashkda_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
        )
        # The radix track needs no intermediate h when every tracked sequence
        # ends on a KDA chunk boundary; the caller copies the final state.
        return (out, None) if final_state_only_track else out

    @staticmethod
    def _ppu_native_layout_supported(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor],
        dt_bias: Optional[torch.Tensor],
    ) -> bool:
        """Keep non-K3 layouts on the existing Triton implementation."""
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
            return False
        heads, tokens = q.shape[2], q.shape[1]
        return (
            q.shape == k.shape == v.shape == g.shape
            and q.dtype == k.dtype == v.dtype == g.dtype == torch.bfloat16
            and beta.shape == (1, tokens, heads)
            and beta.dtype in (torch.bfloat16, torch.float32)
            and ssm_states.ndim == 4
            and ssm_states.shape[1:] == (heads, 128, 128)
            and ssm_states.dtype == torch.float32
            and cache_indices.numel() + 1 == query_start_loc.numel()
            and A_log is not None
            and A_log.numel() == heads
            and dt_bias is not None
            and dt_bias.numel() == heads * 128
        )

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
    ) -> bool:
        """Whether to use the Triton chunk_kda path instead of the fused kernel."""
        # Safe-gate only: the fused kernel does not support the unbounded gate
        # (-exp(A_log)*softplus); those models leave lower_bound unset.
        if lower_bound is None:
            return True
        # FlashKDA writes the committed recurrent state back in place, so it is
        # unsafe for speculative verify / draft-extend forwards (which must stay
        # rollback-able). Those reach this backend through forward_extend, so
        # gate them here rather than relying on the decode/target_verify stubs.
        if is_spec_decode:
            return True
        if is_ppu():
            return False
        # Short sequences (< chunk size) and long sequences (> the crossover
        # where Triton's chunked prefill wins) are faster on Triton. Read the
        # per-request lengths from the CPU-side extend_seq_lens to avoid a
        # GPU->CPU sync on every layer; derive from query_start_loc (one sync)
        # only if they are unavailable.
        if extend_seq_lens_cpu is not None:
            if torch.is_tensor(extend_seq_lens_cpu):
                lo = int(extend_seq_lens_cpu.min())
                hi = int(extend_seq_lens_cpu.max())
            else:
                lo = min(extend_seq_lens_cpu)
                hi = max(extend_seq_lens_cpu)
        else:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            lo_t, hi_t = torch.aminmax(seq_lens)
            lo, hi = int(lo_t), int(hi_t)
        return lo < _FLASHKDA_CHUNK_SIZE or hi > _FLASHKDA_MAX_SEQ_LEN

    def _flashkda_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
    ) -> torch.Tensor:
        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # FlashKDA applies sigmoid internally; invert only the already-activated
        # Kimi beta path.
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        beta = beta.to(torch.bfloat16).contiguous()

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float().contiguous()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float().contiguous()

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = ssm_states[cache_indices].contiguous()

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)

        flash_kda = _load_flash_kda()
        if q.shape[1] > 1 and not getattr(self, "_native_prefill_logged", False):
            import logging

            self._native_prefill_logged = True
            logging.getLogger(__name__).info(
                "FlashKDA native prefill active: H=%s tokens=%s",
                num_heads,
                q.shape[1],
            )
        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )

        ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        return out_buf
