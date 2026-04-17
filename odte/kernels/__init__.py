"""Hopper-native CUDA kernels + CPU stubs.

Three components, each with a compile-on-H100 `.cu` source, a pybind11/
torch-extension `.py` wrapper, and a CPU fallback so the same Python code
runs unchanged on Mac / non-CUDA hosts:

  fused_bin          quantile + log binning tokenizer
  persistent_decode  persistent inference kernel (weights resident in SMEM)
  rdma_ingest        GPUDirect RDMA receiver for OPRA/PITCH UDP feeds

On build-enabled hosts:
    cd odte/kernels && python setup.py build_ext --inplace
    # or: pip install -e .
On Mac/CPU: nothing to build — wrappers transparently fall back.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from .fused_bin import fused_bin, HAS_CUDA_FUSED_BIN
except ImportError as e:
    log.debug("fused_bin import: %s", e)
    fused_bin = None
    HAS_CUDA_FUSED_BIN = False

try:
    from .persistent_decode import (
        PersistentDecoder, load_weights_to_smem, HAS_CUDA_PERSISTENT_DECODE,
    )
except ImportError as e:
    log.debug("persistent_decode import: %s", e)
    PersistentDecoder = None
    load_weights_to_smem = None
    HAS_CUDA_PERSISTENT_DECODE = False

try:
    from .rdma_ingest import ConnectX7Receiver, HAS_CUDA_RDMA
except ImportError as e:
    log.debug("rdma_ingest import: %s", e)
    ConnectX7Receiver = None
    HAS_CUDA_RDMA = False
