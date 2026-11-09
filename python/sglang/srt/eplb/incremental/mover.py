# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fixed-address expert arenas for peer or Mooncake transfers."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
from sglang.srt.eplb.incremental.config import is_routed_expert_module
from sglang.srt.eplb.incremental.copy_engine import (
    enable_peer_access,
    extension_error,
    is_available,
)
from sglang.srt.eplb.incremental.transport import PeerCopyTransport
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

logger = logging.getLogger(__name__)

_REGION_ALIGNMENT = 512


def _align_up(value: int) -> int:
    return (value + _REGION_ALIGNMENT - 1) // _REGION_ALIGNMENT * _REGION_ALIGNMENT


def _raw_bytes(t: torch.Tensor) -> torch.Tensor:
    """Return a dense storage view without changing element order."""
    flat = torch.as_strided(t, (t.numel(),), (1,), t.storage_offset())
    return flat.view(torch.uint8)


def _tiles_densely(sizes, strides) -> bool:
    """Return whether dimensions densely tile storage in any permutation."""
    pairs = sorted((int(st), int(sz)) for st, sz in zip(strides, sizes) if int(sz) > 1)
    expect = 1
    for st, sz in pairs:
        if st != expect:
            return False
        expect *= sz
    return True


# Supported per-expert weight tensors. Models may expose any subset.
_WEIGHT_ATTRS = (
    "w13_weight",
    "w13_weight_bias",
    "w13_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight",
    "w2_weight_bias",
    "w2_weight_scale",
    "w2_weight_scale_inv",
    "mega_w13_weight_scale_utccp",
    "mega_w2_weight_scale_utccp",
)

_SUPPORTED_QUANT_METHODS = {
    "UnquantizedFusedMoEMethod",
    "Fp8MoEMethod",
    "Mxfp4MoEMethod",
}


def _iter_moe_modules(model) -> List[tuple]:
    """(layer_id, experts_module) for every MoE layer, in layer order."""
    out = []
    for name, mod in model.named_modules():
        if not is_routed_expert_module(mod):
            continue
        looks_like_moe = hasattr(mod, "num_local_experts") and hasattr(
            mod, "quant_method"
        )
        if not isinstance(mod, FusedMoE):
            if looks_like_moe:
                raise ValueError(
                    "IncrementalEPLB does not support routed expert module "
                    f"{name} ({type(mod).__name__})"
                )
            continue
        if not hasattr(mod, "w13_weight"):
            raise ValueError(
                "IncrementalEPLB does not support the expert storage layout on "
                f"{name}: expected w13_weight/w2_weight tensors"
            )
        quant_method = type(getattr(mod, "quant_method", None)).__name__
        if quant_method not in _SUPPORTED_QUANT_METHODS:
            raise ValueError(
                f"IncrementalEPLB does not support {quant_method} on {name}; "
                "validated layouts are BF16, standard FP8, and MXFP4"
            )
        layer_id = getattr(mod, "layer_id", None)
        if layer_id is None:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_id = int(parts[i + 1])
                    break
        if layer_id is None:
            raise ValueError(
                f"IncrementalEPLB cannot determine the layer id for MoE module {name}"
            )
        out.append((int(layer_id), mod))
    out.sort(key=lambda x: x[0])
    return out


def validate_mega_moe_experts(model):
    """Require the MXFP4 MegaMoE adapter and its exact receive-count API."""
    import inspect

    import deep_gemm
    from sglang.srt.utils import is_ppu

    name = "fp4_fp4_mega_moe" if is_ppu() else "fp8_fp4_mega_moe"
    kernel = getattr(deep_gemm, name, None)
    if (
        kernel is None
        or "cumulative_local_expert_recv_stats"
        not in inspect.signature(kernel).parameters
    ):
        raise ValueError(
            f"Incremental EPLB requires DeepGEMM {name} with cumulative_local_expert_recv_stats"
        )
    for layer, module in _iter_moe_modules(model):
        if (
            type(module.quant_method).__name__ != "Mxfp4MoEMethod"
            or not getattr(module, "_mega_moe_weights_built", False)
            or not hasattr(module, "mega_l1_weights")
            or not hasattr(module, "mega_l2_weights")
        ):
            raise ValueError(
                f"Incremental EPLB with MegaMoE requires prepared MXFP4 expert weights on layer {layer}"
            )


def _untracked_expert_tensors(mod, num_local: int) -> List[str]:
    """Find post-load per-expert state that the mover would leave behind."""
    tensors = {}
    tensors.update(dict(mod.named_parameters(recurse=False)))
    tensors.update(dict(mod.named_buffers(recurse=False)))
    tensors.update(
        (name, value) for name, value in vars(mod).items() if torch.is_tensor(value)
    )
    return sorted(
        name
        for name, tensor in tensors.items()
        if name not in _WEIGHT_ATTRS
        and tensor.is_cuda
        and tensor.dim() > 0
        and int(tensor.shape[0]) == num_local
    )


def collect_layer_weights(
    model,
    num_layers: int,
    *,
    num_local_physical_experts: int,
) -> List[Optional[Dict[str, torch.Tensor]]]:
    """Collect per-expert tensors by model layer."""
    layer_weights: List[Optional[Dict[str, torch.Tensor]]] = [None] * num_layers
    seen_tensors: Dict[int, int] = {}
    moe_layers = set()
    for layer_id, mod in _iter_moe_modules(model):
        if not 0 <= layer_id < num_layers:
            raise ValueError(
                f"IncrementalEPLB MoE layer id {layer_id} is outside [0, {num_layers})"
            )
        moe_layers.add(layer_id)
        untracked = _untracked_expert_tensors(mod, num_local_physical_experts)
        if untracked:
            raise ValueError(
                "IncrementalEPLB does not support untracked per-expert state on "
                f"layer {layer_id}: {untracked}"
            )
        groups: Dict[str, torch.Tensor] = {}
        for attr in _WEIGHT_ATTRS:
            t = getattr(mod, attr, None)
            if t is None or not torch.is_tensor(t) or not t.is_cuda:
                continue
            key = id(t)
            owner = seen_tensors.get(key)
            if owner == layer_id:
                continue
            if owner is not None:
                raise ValueError(
                    f"IncrementalEPLB does not support one expert tensor shared by "
                    f"layers {owner} and {layer_id}"
                )
            seen_tensors[key] = layer_id
            groups[attr] = t
        if groups:
            existing = layer_weights[layer_id]
            if existing is None:
                layer_weights[layer_id] = groups
                continue
            overlap = existing.keys() & groups.keys()
            if overlap:
                raise ValueError(
                    f"layer {layer_id} exposes multiple tensors for {sorted(overlap)}"
                )
            existing.update(groups)
    missing = sorted(layer for layer in moe_layers if layer_weights[layer] is None)
    if missing:
        raise ValueError(
            f"IncrementalEPLB found no supported CUDA expert weights for layers {missing}"
        )
    return layer_weights


def collect_weight_aliases(model, layer_weights):
    """Match cached model-hook views to the owning tensors before rebinding."""
    declared = getattr(model, "routed_experts_weights_of_layer", None)

    def signature(tensor):
        return (
            tensor.data_ptr(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    owners = {
        layer: {signature(t): t for t in weights.values()}
        for layer, weights in enumerate(layer_weights)
        if weights
    }
    if declared is not None:
        expected = {
            layer: {signature(t) for t in tensors}
            for layer, tensors in declared.items()
        }
        if {layer: set(weights) for layer, weights in owners.items()} != expected:
            raise ValueError(
                "Incremental EPLB storage does not match the model's routed-expert weight contract"
            )
    aliases = {}

    def add_alias(layer, view):
        owner = owners.get(layer, {}).get(signature(view))
        if owner is None:
            raise ValueError(
                f"Incremental EPLB cannot rebind cached expert weights on layer {layer}"
            )
        if view is not owner:
            views = aliases.setdefault(id(owner), [])
            if not any(existing is view for existing in views):
                views.append(view)

    for layer, tensors in (declared or {}).items():
        for view in tensors:
            add_alias(layer, view)
    for layer, module in _iter_moe_modules(model):
        for attr in ("mega_l1_weights", "mega_l2_weights"):
            for view in getattr(module, attr, ()):
                add_alias(layer, view)
    return aliases


class _Region(msgspec.Struct, frozen=True):
    """One expert-major tensor's byte region inside a layer arena."""

    name: str
    offset: int
    bytes_per_expert: int
    total_bytes: int
    tensor: torch.Tensor


class AddressTables(msgspec.Struct, frozen=True):
    """Precomputed host addresses and offsets used by the per-step copy loop."""

    ep_size: int
    num_local: int
    num_homes: int
    local_base: Tuple[int, ...]
    peer_base: Tuple[Tuple[int, ...], ...]
    bytes_per_expert: Tuple[int, ...]
    slot_offset: Tuple[Tuple[Tuple[int, ...], ...], ...]
    region_bytes: Tuple[Tuple[int, ...], ...]
    moe_layers: Tuple[int, ...]
    max_regions: int


class ExpertWeightMover:
    """Re-home local expert tensors in symmetric, fixed-address arenas."""

    def __init__(
        self,
        layer_weights: Sequence[Optional[Dict[str, torch.Tensor]]],
        *,
        num_local_physical_experts: int,
        ep_group,
        ep_rank: int,
        ep_size: int,
        num_spare_slots: int,
        transport=None,
        weight_aliases=None,
    ):
        self.weight_aliases = weight_aliases or {}
        self.transport = PeerCopyTransport() if transport is None else transport
        self.layer_weights = list(layer_weights)
        self.num_local = num_local_physical_experts
        self.ep_group = ep_group
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.arena_bytes = 0
        if self.num_local <= 0 or self.ep_size <= 0:
            raise ValueError("num_local and ep_size must be positive")
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"EP rank {self.ep_rank} is outside EP size {self.ep_size}"
            )
        if self.ep_size > 1 and self.ep_group is None:
            raise ValueError("multi-rank expert pulls require an EP process group")

        # Initial homes and spare slots determine the allocated arena geometry.
        self.spares_total = int(num_spare_slots)
        if self.spares_total != 2 or self.num_local <= 2:
            raise ValueError(
                "Incremental EPLB requires exactly two initial spare slots per rank"
            )
        self.num_homes = self.num_local - self.spares_total

        self._regions: Dict[int, List[_Region]] = {}
        self._symm: Dict[int, torch.Tensor] = {}
        self._handles: Dict[int, object] = {}
        self._peer_base: Dict[Tuple[int, int], int] = {}
        self._tables: Optional[AddressTables] = None
        self._ready = False

    def _describe_layer(self, layer_id: int, groups):
        """Validate a layer and assign aligned byte regions in its arena."""
        offset = 0
        regions: List[_Region] = []
        layer_device = None
        for name, tensor in groups.items():
            if not tensor.is_cuda:
                raise ValueError(f"layer {layer_id} {name} must be on device")
            if layer_device is None:
                layer_device = tensor.device
            if tensor.device != layer_device:
                raise ValueError(
                    f"layer {layer_id} spans devices {layer_device} and {tensor.device}"
                )
            if tensor.dim() < 1 or tensor.shape[0] != self.num_local:
                raise ValueError(
                    f"layer {layer_id} {name}: shape {tuple(tensor.shape)} does "
                    "not have num_local_physical_experts "
                    f"{self.num_local} in its leading dimension"
                )
            if not tensor.is_contiguous():
                per_expert = tensor[0].numel() if tensor.dim() > 1 else 0
                densely_tiled = (
                    tensor.dim() > 1
                    and tensor.stride(0) == per_expert
                    and _tiles_densely(tensor.shape[1:], tensor.stride()[1:])
                )
                if not densely_tiled:
                    raise ValueError(
                        f"layer {layer_id} {name} shape {tuple(tensor.shape)} "
                        f"stride {tuple(tensor.stride())}: expert slices are not "
                        "contiguous and evenly strided"
                    )
            nbytes = tensor.numel() * tensor.element_size()
            tensor_slots = int(tensor.shape[0])
            if nbytes % tensor_slots:
                raise ValueError(
                    f"layer {layer_id} {name}: {nbytes} bytes does not divide "
                    f"evenly into {tensor_slots} expert slots"
                )
            offset = _align_up(offset)
            bytes_per_expert = nbytes // tensor_slots
            regions.append(
                _Region(
                    name=name,
                    offset=offset,
                    bytes_per_expert=bytes_per_expert,
                    total_bytes=nbytes,
                    tensor=tensor,
                )
            )
            offset += nbytes
        return regions, layer_device, offset

    def _allocate_arena(self, layer_id, size, device, symm_mem):
        if not self.transport.requires_peer_memory:
            self._handles[layer_id] = None
            return self.transport.allocate_arena(size, device)
        if self.ep_size == 1:
            self._handles[layer_id] = None
            return torch.empty(size, dtype=torch.uint8, device=device)
        buffer = symm_mem.empty(size, dtype=torch.uint8, device=device)
        self._handles[layer_id] = symm_mem.rendezvous(buffer, group=self.ep_group)
        return buffer

    def _rehome_regions(self, buffer, regions: List[_Region]) -> None:
        """Move raw storage into the arena without changing tensor layouts."""
        for region in regions:
            tensor = region.tensor
            nbytes = region.total_bytes
            # Some quantization scales are densely tiled but intentionally
            # transposed. A contiguous copy would silently change their layout.
            source = (
                tensor.view(torch.uint8)
                if tensor.is_contiguous()
                else _raw_bytes(tensor)
            )
            buffer[region.offset : region.offset + nbytes].copy_(source.reshape(-1))
            destination = buffer[region.offset : region.offset + nbytes].view(
                tensor.dtype
            )
            # Preserve the original strides expected by the fused GEMM.
            tensor.data = torch.as_strided(destination, tensor.shape, tensor.stride())
            # Cached .data views must release the old storage as well.
            for alias in self.weight_aliases.get(id(tensor), ()):
                alias.data = tensor.data

        self._initialize_spares(buffer, regions)

    def _initialize_spares(self, buffer, regions: List[_Region]) -> None:
        """Populate initial spare slots before placement can reuse home slots."""
        for slot in range(self.num_homes, self.num_local):
            home = (slot - self.num_homes) % self.num_homes
            for region in regions:
                size = region.bytes_per_expert
                src = region.offset + home * size
                dst = region.offset + slot * size
                buffer[dst : dst + size].copy_(buffer[src : src + size])

    def setup(self) -> None:
        if self._ready:
            return
        if self.transport.requires_peer_memory and not is_available():
            # Fail before rebinding any model storage.
            raise RuntimeError(
                "IncrementalEPLB requires the peer-copy CUDA Runtime binding: "
                f"{extension_error()}"
            )
        symm_mem = None
        if self.transport.requires_peer_memory:
            import torch.distributed._symmetric_memory as symm_mem

        layouts = {}
        for layer_id, groups in enumerate(self.layer_weights):
            if groups:
                regions, device, size = self._describe_layer(layer_id, groups)
                self._regions[layer_id] = regions
                layouts[layer_id] = (device, size)
        if not layouts:
            raise ValueError("no expert weight tensors given to the mover")
        if len({device for device, _ in layouts.values()}) != 1:
            raise ValueError("expert arenas must share one device per worker")
        # Check peer layouts before rendezvous or rebinding model weights.
        self._check_region_agreement()

        total = 0
        for layer_id, (device, size) in layouts.items():
            buffer = self._allocate_arena(layer_id, size, device, symm_mem)
            self._symm[layer_id] = buffer
            self._rehome_regions(buffer, self._regions[layer_id])
            total += size

        if not self.transport.requires_peer_memory:
            self.transport.prepare(self)
            n_enabled = 0
        else:
            self._prime_peer_bases()
            n_enabled = enable_peer_access()
        logger.info(
            "IncrementalEPLB: peer access newly enabled for %d device pair(s)",
            n_enabled,
        )
        if self.ep_group is not None:
            torch.distributed.barrier(self.ep_group)
        torch.cuda.synchronize()
        self._ready = True
        self.arena_bytes = total
        self._tables = self._build_address_tables()
        logger.info(
            "IncrementalEPLB mover ready: %d MoE layers, %.2f MiB/expert, %d slots/rank "
            "(%d initial homes + %d spares), arena %.2f GiB "
            "(%d per-layer symmetric buffers), %d regions/expert",
            len(self._regions),
            sum(r.bytes_per_expert for r in next(iter(self._regions.values()))) / 2**20,
            self.num_local,
            self.num_homes,
            self.spares_total,
            total / 2**30,
            len(self._symm),
            self._tables.max_regions,
        )

    def _peer(self, rank: int, layer_id: int) -> int:
        key = (rank, layer_id)
        base = self._peer_base.get(key)
        if base is None:
            local = self._symm[layer_id]
            if rank == self.ep_rank:
                base = local.data_ptr()
            else:
                handle = self._handles[layer_id]
                if handle is None:
                    raise RuntimeError("peer handle is missing")
                buf = handle.get_buffer(
                    rank, (local.numel(),), torch.uint8, storage_offset=0
                )
                base = buf.data_ptr()
            self._peer_base[key] = base
        return base

    def _prime_peer_bases(self) -> None:
        """Resolve every peer mapping during setup."""
        for layer_id in sorted(self._regions):
            for rank in range(self.ep_size):
                self._peer(rank, layer_id)

    def _check_region_agreement(self) -> None:
        """Verify that every rank uses the same arena layout."""
        if self.ep_group is None or self.ep_size <= 1:
            return
        sig = tuple(
            (
                layer_id,
                tuple(
                    (
                        r.name,
                        r.offset,
                        r.bytes_per_expert,
                        r.total_bytes,
                        self.num_local,
                    )
                    for r in self._regions[layer_id]
                ),
            )
            for layer_id in sorted(self._regions)
        )
        gathered: List[object] = [None] * self.ep_size
        torch.distributed.all_gather_object(gathered, sig, group=self.ep_group)
        for rank, other in enumerate(gathered):
            if other == sig:
                continue
            mine = dict(sig)
            theirs = dict(other or ())
            bad = sorted(
                set(mine) ^ set(theirs)
                | {k for k in set(mine) & set(theirs) if mine[k] != theirs[k]}
            )
            raise RuntimeError(
                f"IncrementalEPLB: rank {rank} laid out its expert arenas "
                f"differently from rank {self.ep_rank}; peer copies would move "
                f"the wrong bytes. First differing layers: {bad[:4]}"
            )

    def _build_address_tables(self) -> AddressTables:
        """Materialize immutable host address tables."""
        num_layers = len(self.layer_weights)
        moe_layers = sorted(self._regions)
        counts = [len(self._regions[i]) for i in moe_layers]
        max_regions = max(counts) if counts else 0

        local_base = [0] * num_layers
        peer_base: List[List[int]] = [[0] * self.ep_size for _ in range(num_layers)]
        per_expert = [0] * num_layers
        empty_slots = ((),) * self.num_local
        slot_offset: List[Tuple[Tuple[int, ...], ...]] = [
            empty_slots for _ in range(num_layers)
        ]
        region_bytes: List[Tuple[int, ...]] = [() for _ in range(num_layers)]

        for layer_id in moe_layers:
            regs = self._regions[layer_id]
            base = int(self._symm[layer_id].data_ptr())
            local_base[layer_id] = base
            row = [int(self._peer_base[(r, layer_id)]) for r in range(self.ep_size)]
            peer_base[layer_id] = row
            per_expert[layer_id] = sum(r.bytes_per_expert for r in regs)
            region_bytes[layer_id] = tuple(r.bytes_per_expert for r in regs)
            so = tuple(
                tuple(r.offset + slot * r.bytes_per_expert for r in regs)
                for slot in range(self.num_local)
            )
            slot_offset[layer_id] = so
        return AddressTables(
            ep_size=self.ep_size,
            num_local=self.num_local,
            num_homes=self.num_homes,
            local_base=tuple(local_base),
            peer_base=tuple(tuple(row) for row in peer_base),
            bytes_per_expert=tuple(per_expert),
            slot_offset=tuple(slot_offset),
            region_bytes=tuple(region_bytes),
            moe_layers=tuple(moe_layers),
            max_regions=max_regions,
        )

    def address_tables(self) -> AddressTables:
        """Return the immutable host address tables."""
        if not self._ready or self._tables is None:
            raise RuntimeError("call setup() first")
        return self._tables
