"""Budget-scale acceleration layer.

Numba-jitted hot paths (CPU/Mac) + optional Liger-Kernel integration (CUDA).
Gracefully degrades to numpy + eager torch when neither is installed.

Why this exists: skipping Phase-3 CUDA kernels costs ~$30-60k + 8-12 weeks.
Numba + Liger gets 80% of the speedup for free. Trade-off honestly noted
in the module docstrings.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    import numba  # type: ignore
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import liger_kernel  # type: ignore
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False

log.debug("accel flags: numba=%s liger=%s", HAS_NUMBA, HAS_LIGER)

from .numba_kernels import (
    fused_bin_numba, microprice_numba, ofi_numba, markout_numba,
    cum_signed_volume_numba,
)
from .liger_adapter import (
    patch_tradefm_with_liger, revert_tradefm_liger,
)
