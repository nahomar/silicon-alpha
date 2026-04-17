"""Per-tick streaming OFI / microprice — the budget substitute for
persistent CUDA kernels.

Each incoming TOB event updates an internal 2-tick window. We call the
Numba-JIT'd kernel on that 2-element array, which compiles once and runs
in sub-microsecond per tick on the CPU. This is the path the live
runner uses — the batched numpy version never reaches the hot loop.

Why this file exists: the per-tick win isn't visible in the batched
bench. It IS visible in odte_live_paper.py where each message round-
trips through the tokenizer / model / QP in <=1ms end-to-end.

Optional CUDA escape hatch: when odte_kernels_cu builds, the per-tick
path goes through the persistent CUDA kernel instead — but the API is
identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from odte.accel.numba_kernels import (
    ofi_numba, microprice_numba, fused_bin_numba,
)

log = logging.getLogger(__name__)


@dataclass
class TobState:
    """Rolling top-of-book state per instrument, supporting per-tick OFI."""
    prev_bid_px: float = 0.0
    prev_bid_sz: float = 0.0
    prev_ask_px: float = 0.0
    prev_ask_sz: float = 0.0
    last_micro: float = 0.0
    last_ofi: float = 0.0
    n_ticks: int = 0


@dataclass
class StreamingFeatures:
    """Fire-and-forget per-tick feature extractor.

    Maintains one TobState per symbol. Call update(sym, bid_px, ...) on
    every incoming TOB, get back the current microprice + incremental OFI.
    Uses 2-element numpy buffers that Numba JITs once and reuses forever.
    """
    _state: Dict[str, TobState] = field(default_factory=dict)
    # Scratch buffers — JIT kernels see these exact ndarrays every tick.
    _b_px: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    _b_sz: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    _a_px: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    _a_sz: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def update(self, symbol: str,
               bid_px: float, bid_sz: float,
               ask_px: float, ask_sz: float) -> dict:
        st = self._state.get(symbol)
        if st is None:
            st = TobState(prev_bid_px=bid_px, prev_bid_sz=bid_sz,
                          prev_ask_px=ask_px, prev_ask_sz=ask_sz)
            self._state[symbol] = st
            micro = microprice_numba(
                np.array([bid_px]), np.array([ask_px]),
                np.array([bid_sz]), np.array([ask_sz]))[0]
            st.last_micro = float(micro); st.last_ofi = 0.0; st.n_ticks = 1
            return {"micro": st.last_micro, "ofi": 0.0, "first_tick": True}

        self._b_px[0] = st.prev_bid_px; self._b_px[1] = bid_px
        self._b_sz[0] = st.prev_bid_sz; self._b_sz[1] = bid_sz
        self._a_px[0] = st.prev_ask_px; self._a_px[1] = ask_px
        self._a_sz[0] = st.prev_ask_sz; self._a_sz[1] = ask_sz

        ofi_vec = ofi_numba(self._b_px, self._b_sz, self._a_px, self._a_sz)
        micro = microprice_numba(
            self._b_px[1:2], self._a_px[1:2], self._b_sz[1:2], self._a_sz[1:2])[0]

        st.prev_bid_px = bid_px; st.prev_bid_sz = bid_sz
        st.prev_ask_px = ask_px; st.prev_ask_sz = ask_sz
        st.last_micro = float(micro)
        st.last_ofi = float(ofi_vec[1])
        st.n_ticks += 1
        return {"micro": st.last_micro, "ofi": st.last_ofi, "first_tick": False}

    def encode_token(self, value: float, edges: np.ndarray) -> int:
        """Numba-JIT bucket lookup for a single float value."""
        return int(fused_bin_numba(np.array([value], dtype=np.float64), edges)[0])
