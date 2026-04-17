"""PersistentInferenceStub — CPU/MPS fallback for the Hopper persistent kernel.

Same API as the CUDA version:
    stub = PersistentInferenceStub.from_checkpoint("ckpt.pt", device="mps")
    for tok in input_stream:
        logits = stub.step(tok_ids)           # (1, vocab)
        next_id = int(logits.argmax().item())

When Hopper kernels are available, the same code path automatically uses
the persistent_decode.cu kernel via odte.kernels.PersistentDecoder — this
stub delegates to it when HAS_CUDA_PERSISTENT_DECODE is True. Otherwise it
wraps a plain TradeFM forward pass.

Keeps a rolling context buffer so callers don't have to manage it.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from odte.kernels import HAS_CUDA_PERSISTENT_DECODE, PersistentDecoder

log = logging.getLogger(__name__)


@dataclass
class PersistentInferenceStub:
    ckpt_path: Path
    device: str = "cpu"
    ctx_len: int = 512
    _decoder: object = field(default=None, init=False)
    _ctx: deque = field(default_factory=lambda: deque(maxlen=512), init=False)
    _last_logits: object = field(default=None, init=False)

    def __post_init__(self):
        if PersistentDecoder is None:
            raise RuntimeError("odte.kernels.persistent_decode unavailable — "
                               "check torch install")
        self._decoder = PersistentDecoder(ckpt_path=Path(self.ckpt_path),
                                          device=self.device,
                                          ctx_len=self.ctx_len)
        self._ctx = deque(maxlen=self.ctx_len)
        log.info("PersistentInferenceStub ready  backend=%s  ctx=%d",
                 "cuda" if HAS_CUDA_PERSISTENT_DECODE else "cpu",
                 self.ctx_len)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device: str = "cpu",
                        ctx_len: int = 512) -> "PersistentInferenceStub":
        return cls(ckpt_path=Path(ckpt_path), device=device, ctx_len=ctx_len)

    def step(self, tokens: int | Sequence[int]):
        """Append tokens to the rolling context and return next-token logits."""
        if isinstance(tokens, int):
            self._ctx.append(int(tokens))
        else:
            for t in tokens:
                self._ctx.append(int(t))
        if len(self._ctx) < 2:       # need ≥ 2 tokens for causal LM
            return None
        import torch
        ctx = torch.tensor([list(self._ctx)], dtype=torch.long)
        logits = self._decoder.step(ctx)
        self._last_logits = logits
        return logits

    def reset(self) -> None:
        self._ctx.clear()
        self._last_logits = None

    def argmax(self) -> Optional[int]:
        if self._last_logits is None:
            return None
        return int(self._last_logits.argmax(dim=-1).item())
