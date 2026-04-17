"""Liger-Kernel drop-in replacement for TradeFM layers.

Liger (https://github.com/linkedin/Liger-Kernel) is an open-source Triton-
based kernel library. Drop-in replaces SwiGLU, RMSNorm, and the fused
linear + CE loss with ~20% throughput uplift and ~60% memory savings —
for free, no custom CUDA.

Requirements:
    pip install liger-kernel  # CUDA only (uses Triton)

Usage:
    from odte.accel import patch_tradefm_with_liger, revert_tradefm_liger
    patch_tradefm_with_liger(model)   # before training
    ...
    revert_tradefm_liger(model)       # if you need the reference layers back

On Mac / CPU: patch_tradefm_with_liger is a no-op (logs a warning).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    import liger_kernel  # type: ignore
    from liger_kernel.transformers import (  # type: ignore
        LigerRMSNorm, LigerSwiGLUMLP, LigerFusedLinearCrossEntropyLoss,
    )
    _HAS_LIGER = True
except Exception:
    _HAS_LIGER = False


def patch_tradefm_with_liger(model) -> bool:
    """In-place replace RMSNorm/SwiGLU blocks with Liger kernels.

    Returns True if Liger was applied. False means no-op (Mac / CPU / no
    liger_kernel installed). Records original modules on the model in
    `_liger_backup` so `revert_tradefm_liger` can restore.
    """
    if not _HAS_LIGER:
        log.warning("liger_kernel not importable; keeping reference kernels")
        return False
    import torch.nn as nn
    backup = {}
    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls == "RMSNorm":
            parent, attr = _resolve_parent(model, name)
            dim = module.weight.shape[0]
            liger = LigerRMSNorm(dim).to(module.weight.device)
            with backup_lock(backup, name, module):
                setattr(parent, attr, liger)
        elif cls == "SwiGLU":
            parent, attr = _resolve_parent(model, name)
            dim = module.w1.in_features
            hidden = module.w1.out_features
            liger = LigerSwiGLUMLP(hidden_size=dim, intermediate_size=hidden) \
                .to(next(module.parameters()).device)
            with backup_lock(backup, name, module):
                setattr(parent, attr, liger)
    model._liger_backup = backup
    log.info("Liger patch: replaced %d modules", len(backup))
    return True


def revert_tradefm_liger(model) -> None:
    backup = getattr(model, "_liger_backup", None)
    if not backup:
        return
    for name, original in backup.items():
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, original)
    del model._liger_backup
    log.info("Liger revert: restored %d modules", len(backup))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _resolve_parent(root, dotted: str):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


class backup_lock:
    """Trivial context manager to stash the original before we swap."""
    def __init__(self, d, key, module):
        self._d = d; self._key = key; self._module = module
    def __enter__(self):
        self._d[self._key] = self._module
        return self
    def __exit__(self, *_):
        return False
