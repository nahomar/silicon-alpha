"""Python wrapper for the fused binning kernel.

On H100 with the built C++ extension: dispatches to `_odte_fused_bin_cuda`
(one persistent block-per-feature, binary search on sorted edges in SMEM,
result int16 in HBM3 ring buffer).

On Mac/CPU: falls back to np.searchsorted — same signature, same result.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

HAS_CUDA_FUSED_BIN = False
_cuda_ext = None

try:  # optional C++ extension
    import torch
    from torch.utils import cpp_extension       # noqa: F401
    import odte_kernels_cu   # type: ignore     # built by odte/kernels/setup.py
    _cuda_ext = odte_kernels_cu
    HAS_CUDA_FUSED_BIN = True
except Exception:  # pragma: no cover (runs only on H100)
    _cuda_ext = None


def fused_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map float values to int16 bucket indices.

    edges[0] and edges[-1] are the ±inf sentinels; interior is searched.
    Accepts numpy or torch tensors; returns the same type on CUDA-backed
    tensors, else numpy int16.
    """
    if HAS_CUDA_FUSED_BIN and hasattr(values, "is_cuda") and values.is_cuda:
        return _cuda_ext.fused_bin(values, edges)
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(edges, dtype=np.float64)
    interior = e[1:-1]
    return np.searchsorted(interior, v, side="right").astype(np.int16)
