"""Tokenizer kernel indirection.

Single choke point: `fused_bin(values, edges) -> int16 token ids`.

Mac / any CPU: uses np.searchsorted (vectorized, fast enough for dev).
H100 (Phase 3): dispatches to a persistent CUDA kernel that does a per-thread
                binary search in SMEM and is fused with the RDMA ring-buffer
                consumer. Drop-in replacement; same function signature.
"""
from __future__ import annotations

import numpy as np

from . import HAS_CUDA


def _fused_bin_cpu(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map float values to int16 bucket indices using interior edges only.

    edges[0] and edges[-1] are the ±inf sentinels; searchsorted works on
    the interior.
    """
    interior = edges[1:-1]
    idx = np.searchsorted(interior, values, side="right").astype(np.int16)
    return idx


def fused_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if HAS_CUDA:
        try:
            from odte.kernels import fused_bin as cuda_fused_bin  # type: ignore
            return cuda_fused_bin(values, edges)
        except Exception:
            pass  # fall back silently
    return _fused_bin_cpu(values, edges)
