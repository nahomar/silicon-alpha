"""0DTE alpha engine — Phase 0 local scaffolding.

Top-level package wiring HRT-style components:
  - HybridBinTokenizer (quantile for price, log-width for volume/time)
  - TradeFM decoder-only transformer (with MiniTradeFM for Mac dev)
  - DMLPricer (option price + Greeks with maturity gate)
  - WorldSim (digital-twin adversarial market sim)
  - DeterministicExecutor (constrained QP portfolio)
  - OPRAIngest (async pluggable feed)
  - synth_options (Heston + IV + Hawkes adversarial flow)

Feature flags detected at import. CUDA / FP8 / RDMA paths are gated so the
same code runs on Mac today and on an H100 box after Phase 3.
"""
from __future__ import annotations

import importlib
import logging
import os

log = logging.getLogger(__name__)

__version__ = "0.1.0"


def _is_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


# Hardware / library feature detection (evaluated once at import).
HAS_TORCH: bool = _is_importable("torch")
HAS_TE: bool = _is_importable("transformer_engine")
HAS_FLASH_ATTN: bool = _is_importable("flash_attn")
HAS_RDMA: bool = os.environ.get("ODTE_HAS_RDMA") == "1"

if HAS_TORCH:
    import torch  # noqa: F401
    HAS_CUDA: bool = torch.cuda.is_available()
    HAS_MPS: bool = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
else:
    HAS_CUDA = False
    HAS_MPS = False


def best_device() -> str:
    """Return the best device string for this host."""
    if HAS_CUDA:
        return "cuda"
    if HAS_MPS:
        return "mps"
    return "cpu"


log.debug(
    "odte flags — torch=%s cuda=%s mps=%s te=%s flash_attn=%s rdma=%s",
    HAS_TORCH, HAS_CUDA, HAS_MPS, HAS_TE, HAS_FLASH_ATTN, HAS_RDMA,
)
