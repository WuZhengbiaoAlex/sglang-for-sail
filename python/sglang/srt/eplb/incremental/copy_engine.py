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
"""Host-issued peer copies for IncrementalEPLB expert-weight pulls."""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_EXT = None
_EXT_ERR: Optional[str] = None


def _load_extension():
    """Build the kernel-free CUDA Runtime binding once per process."""
    global _EXT, _EXT_ERR
    if _EXT is not None or _EXT_ERR is not None:
        return _EXT
    try:
        from torch.utils.cpp_extension import load

        src = os.path.join(os.path.dirname(__file__), "csrc", "copy_engine.cu")
        _EXT = load(
            name="sglang_incremental_eplb_copy_engine",
            sources=[src],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        logger.info("IncrementalEPLB copy-engine runtime built")
    except Exception as e:  # noqa: BLE001
        _EXT_ERR = str(e)
        logger.warning("IncrementalEPLB copy-engine runtime unavailable: %s", e)
    return _EXT


def is_available() -> bool:
    return _load_extension() is not None


def extension_error() -> Optional[str]:
    return _EXT_ERR


def enable_peer_access() -> int:
    """Enable direct peer access for every visible device pair."""
    ext = _load_extension()
    if ext is None:
        raise RuntimeError(
            f"IncrementalEPLB copy-engine runtime is unavailable: {extension_error()}"
        )
    enabled = int(ext.enable_peer_access())
    ext.prepare_copy_engine()
    return enabled


def issue_batch(
    src: Sequence[int],
    dst: Sequence[int],
    nbytes: Sequence[int],
    *,
    stream: Optional[torch.cuda.Stream],
) -> int:
    """Enqueue one peer-copy batch on the dedicated copy stream."""
    n = len(src)
    if len(dst) != n or len(nbytes) != n:
        raise ValueError(
            f"batch lengths differ: {n} src, {len(dst)} dst, {len(nbytes)} sizes"
        )
    if n == 0:
        return 0
    if stream is None:
        raise ValueError("IncrementalEPLB copies require a dedicated CUDA stream")

    ext = _load_extension()
    if ext is None:
        raise RuntimeError(
            f"IncrementalEPLB copy-engine runtime is unavailable: {extension_error()}"
        )
    ext.enqueue_memcpy_batch(src, dst, nbytes, int(stream.cuda_stream))
    return n
