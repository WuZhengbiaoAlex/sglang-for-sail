"""Opt-in PPU K3.1 VERIFY o_proj GEMM + attention-TP reduce-scatter.

The model owns the sharded-output contract. General RowParallelLinear calls
continue to return their usual shape, including draft, prefill and idle calls.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_sp_collective
from sglang.srt.runtime_context import (
    get_buffer,
    get_exec,
    get_flags,
    get_forward,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils import is_ppu

logger = logging.getLogger(__name__)


@dataclass
class _ProjectionCall:
    projection: "FusedOProjReduceScatter"
    full_rows: int
    shard_lo: int = -1


class _SharedWorkspace:
    def __init__(self, group, max_rows: int, width: int):
        from deep_gemm import (
            fp8_gemm_nt_reduce_scatter,
            get_workspace_for_gemm_reduce_scatter,
        )

        # Collective initialization is deliberately in model construction,
        # before any graph capture. Keep one allocation for all o_proj layers.
        self.workspace = get_workspace_for_gemm_reduce_scatter(
            group.device_group, max_rows, width
        )
        self.gemm = fp8_gemm_nt_reduce_scatter
        self.group = group
        self.max_rows = max_rows
        self.last_stream = None
        self.logged_shapes = set()

    def apply(self, x_q, x_scale, weight, weight_scale):
        stream = torch.cuda.current_stream()
        # The workspace's packet epochs are shared across layers. Preserve
        # their order even when a caller switches streams; graph capture
        # records the same dependency. No event is added on the usual stream.
        if self.last_stream is not None and self.last_stream != stream:
            stream.wait_stream(self.last_stream)
        out = torch.empty(
            (x_q.shape[0] // self.group.world_size, weight.shape[0]),
            dtype=torch.bfloat16,
            device=x_q.device,
        )
        self.gemm(
            (x_q, x_scale),
            (weight, weight_scale),
            out,
            workspace=self.workspace,
        )
        self.last_stream = stream
        shape = (x_q.shape[0], weight.shape[0], weight.shape[1])
        if shape not in self.logged_shapes:
            logger.info(
                "PPU K31 fused o_proj GEMM+RS runtime hit M=%d N=%d K=%d TP=%d "
                "attn_tp_rank=%d",
                *shape,
                self.group.world_size,
                self.group.rank_in_group,
            )
            self.logged_shapes.add(shape)
        return out


class FusedOProjReduceScatter:
    def __init__(self, resource: _SharedWorkspace):
        self.resource = resource

    @contextmanager
    def scope(self, forward_batch, full_rows: int):
        mode = forward_batch.forward_mode
        original_mode = getattr(forward_batch, "_original_forward_mode", None)
        call = None
        if (
            mode.is_target_verify()
            and not (original_mode is not None and original_mode.is_idle())
            and 0 < full_rows <= self.resource.max_rows
            and full_rows % self.resource.group.world_size == 0
            and not get_forward().sp_active
        ):
            call = _ProjectionCall(self, full_rows)
        with get_forward().scoped(k31_o_proj_rs_call=call):
            yield call

    def try_apply(self, layer, x_q, x_scale, weight, weight_scale):
        call = get_forward().k31_o_proj_rs_call
        if (
            call is None
            or call.projection is not self
            or call.shard_lo >= 0
            or layer.reduce_results
            or layer.use_decode_attn_tp
            or x_q.ndim != 2
            or x_q.shape[0] != call.full_rows
            or tuple(weight.shape) != (7168, 1536)
            or x_q.shape[1] != weight.shape[1]
            or x_q.dtype != torch.float8_e4m3fn
            or weight.dtype != torch.float8_e4m3fn
            or x_scale.dtype != torch.float32
            or weight_scale.dtype != torch.float32
            or tuple(x_scale.shape) != (call.full_rows, 1)
            or tuple(weight_scale.shape) != (weight.shape[0], 1)
            or not all(t.is_contiguous() for t in (x_q, x_scale, weight, weight_scale))
        ):
            return None
        out = self.resource.apply(x_q, x_scale, weight, weight_scale)
        call.shard_lo = self.resource.group.rank_in_group * out.shape[0]
        return out


def initialize_model(model) -> None:
    """Attach explicit sharded-output capabilities only to eligible main layers."""
    if not envs.SGLANG_PPU_K31_FUSE_O_PROJ_RS.get() or not is_ppu():
        return
    # The loader uses this same scope to stamp FusedMoE.is_draft. DFlash uses
    # its own model class, but other draft architectures may reuse K3's model.
    if get_flags().moe.in_speculative_scope:
        return
    if k3_sp_collective.enabled():
        raise RuntimeError(
            "PPU K31 fused o_proj GEMM+RS cannot use the K3 symmetric-output "
            "wrapper; disable SGLANG_K3_SP_COLLECTIVE"
        )
    config = model.config
    parallel = get_parallel()
    group = parallel.attn_tp_group
    if (
        getattr(config, "dsa_config", None) is None
        or config.attn_res_block_size is None
        or config.hidden_size != 7168
        or model.pp_group.world_size != 1
        or parallel.attn_cp_size != 1
        or parallel.attn_dcp_size != 1
        or group.world_size != 8
    ):
        logger.info(
            "PPU K31 fused o_proj GEMM+RS: unsupported model/topology; fallback"
        )
        return

    from sglang.srt.layers.quantization.ppu_dense_w8a8_fp8 import (
        PPUDenseW8A8FP8LinearMethod,
    )

    candidates = []
    for layer in model.layers:
        if not getattr(layer, "_sp_moe", False):
            continue
        proj = layer.self_attn.o_proj
        if (
            isinstance(proj.quant_method, PPUDenseW8A8FP8LinearMethod)
            and proj.bias is None
            and not proj.reduce_results
            and tuple(proj.weight.shape) == (7168, 1536)
        ):
            candidates.append((layer, proj))
    if not candidates:
        logger.info("PPU K31 fused o_proj GEMM+RS: no eligible MoE o_proj; fallback")
        return

    graph = get_exec().graph.cuda_graph_config
    graph_bs = (
        max([graph.decode.max_bs or 0, *(graph.decode.bs or [])])
        if graph is not None
        else 0
    )
    if graph_bs <= 0:
        graph_bs = max(
            1, (get_schedule().max_running_requests or 1) // parallel.attn_dp_size
        )
    max_rows = graph_bs * (get_spec().speculative_num_draft_tokens or 1)
    name = (
        f"k31_ppu_o_proj_rs:{group.device_group.group_name}:"
        f"{torch.cuda.current_device()}:{max_rows}:{config.hidden_size}"
    )
    resource = get_buffer(
        name, lambda: _SharedWorkspace(group, max_rows, config.hidden_size)
    )
    for layer, proj in candidates:
        handle = FusedOProjReduceScatter(resource)
        layer._ppu_o_proj_rs = handle
        proj._ppu_o_proj_rs = handle
    logger.info(
        "PPU K31 fused o_proj GEMM+RS initialized layers=%d max_rows=%d TP=%d "
        "attn_tp_rank=%d; graph warmup primes each captured VERIFY shape",
        len(candidates),
        max_rows,
        group.world_size,
        group.rank_in_group,
    )
