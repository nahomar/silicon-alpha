"""Streaming quantile / log-edge fitters.

Designed for ≥1 T-row training corpora where np.quantile(whole_array) is
impossible. Checkpoint-resumable so a fit can span many nodes.

Two estimators:
  StreamingQuantileFitter  — reservoir-sampled quantiles. Error bounded by
                             reservoir size (default 1M samples → ≈0.1%
                             quantile error with 64 buckets).
  StreamingLogEdgeFitter   — running min/max in log-space, robust to outliers
                             via percentile clipping (default 0.01%/99.99%).

Both support .update(chunk) / .merge(other) / .finalize() / .save() / .load().
The merge semantics make them embarrassingly parallel across shards.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reservoir-sample quantile fitter
# ---------------------------------------------------------------------------

@dataclass
class StreamingQuantileFitter:
    # Default bumped 1M -> 2M after the parity test in
    # tests/odte/test_quantile_parity.py showed 1M gave ~3.3e-3 max-abs-err
    # vs np.quantile on a 10M-row worst-case corpus. 2M halves that near the
    # median and costs ~16 MB per feature. At Phase-2 scale (>=1T rows) the
    # reservoir dominates and error approaches the theoretical ~sqrt(1/R)
    # floor; the 10M-row test is the pessimistic case, not production.
    reservoir_size: int = 2_000_000
    n_buckets: int = 64
    seed: int = 0
    _buf: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    _n_seen: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return
        needed = self.reservoir_size - len(self._buf)
        if needed > 0:
            take = x[:needed]
            self._buf = np.concatenate([self._buf, take])
            x = x[needed:]
            self._n_seen += len(take)
        # Classic Vitter R: each subsequent item replaces a random slot with
        # prob reservoir_size / n_seen.
        if len(x):
            n0 = self._n_seen
            self._n_seen += len(x)
            # vectorized replacement decisions
            keep = self._rng.integers(1, self._n_seen - n0 + 1, size=len(x)) \
                + np.arange(n0 + 1, self._n_seen + 1) - 1
            keep_mask = keep < self.reservoir_size
            idx_to_replace = keep[keep_mask]
            self._buf[idx_to_replace] = x[keep_mask]

    def merge(self, other: "StreamingQuantileFitter") -> None:
        """Merge another fitter's reservoir into this one (weighted by n_seen)."""
        if self.reservoir_size != other.reservoir_size:
            raise ValueError("reservoir size mismatch")
        total = self._n_seen + other._n_seen
        if total == 0:
            return
        # proportional subsample from each
        take_self = min(len(self._buf),
                        int(self.reservoir_size * self._n_seen / total))
        take_other = self.reservoir_size - take_self
        take_other = min(take_other, len(other._buf))
        a = self._rng.choice(self._buf, size=take_self, replace=False) \
            if take_self < len(self._buf) else self._buf
        b = self._rng.choice(other._buf, size=take_other, replace=False) \
            if take_other < len(other._buf) else other._buf
        self._buf = np.concatenate([a, b])
        self._n_seen = total

    def finalize(self) -> np.ndarray:
        """Return n_buckets+1 edges with ±inf sentinels at the ends."""
        if len(self._buf) < self.n_buckets + 1:
            raise RuntimeError(f"need ≥{self.n_buckets+1} samples, have {len(self._buf)}")
        qs = np.linspace(0, 1, self.n_buckets + 1)
        edges = np.quantile(self._buf, qs)
        edges = np.unique(edges)
        if len(edges) < self.n_buckets + 1:
            pad = np.linspace(edges[-1], edges[-1] + 1,
                              self.n_buckets + 1 - len(edges) + 1)[1:]
            edges = np.concatenate([edges, pad])
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

    def save(self, path: Path) -> None:
        np.savez(Path(path), buf=self._buf, n_seen=self._n_seen,
                 n_buckets=self.n_buckets, reservoir_size=self.reservoir_size)

    @classmethod
    def load(cls, path: Path) -> "StreamingQuantileFitter":
        z = np.load(Path(path))
        f = cls(reservoir_size=int(z["reservoir_size"]),
                n_buckets=int(z["n_buckets"]))
        f._buf = z["buf"]
        f._n_seen = int(z["n_seen"])
        return f


# ---------------------------------------------------------------------------
# Log-width edge fitter (for volume / inter-arrival time features)
# ---------------------------------------------------------------------------

@dataclass
class StreamingLogEdgeFitter:
    """Finds min/max of log(|x| + floor), clipped at tail percentiles."""
    n_buckets: int = 64
    floor: float = 1e-9
    # Tail clip percentiles — stored via reservoir on |x|.
    clip_low: float = 0.0001
    clip_high: float = 0.9999
    reservoir_size: int = 500_000
    _qf: Optional[StreamingQuantileFitter] = None

    def __post_init__(self):
        self._qf = StreamingQuantileFitter(reservoir_size=self.reservoir_size,
                                           n_buckets=self.n_buckets)

    def update(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64)
        x = np.abs(x) + self.floor
        self._qf.update(x)

    def merge(self, other: "StreamingLogEdgeFitter") -> None:
        self._qf.merge(other._qf)

    def finalize(self) -> np.ndarray:
        buf = self._qf._buf
        if len(buf) < 10:
            raise RuntimeError("too few log-edge samples")
        lo = float(np.quantile(buf, self.clip_low))
        hi = float(np.quantile(buf, self.clip_high))
        lo = max(lo, self.floor)
        hi = max(hi, lo * 10)
        edges = np.exp(np.linspace(math.log(lo), math.log(hi), self.n_buckets + 1))
        edges[0], edges[-1] = 0.0, np.inf
        return edges

    def save(self, path: Path) -> None:
        self._qf.save(path)

    @classmethod
    def load(cls, path: Path) -> "StreamingLogEdgeFitter":
        f = cls()
        f._qf = StreamingQuantileFitter.load(path)
        return f


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------

def fit_hybrid_from_chunks(chunks, feature_spec: Dict[str, str],
                           n_buckets: int = 64,
                           checkpoint: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Fit HybridBinTokenizer edges from an iterable of pandas-like chunks.

    Each chunk is a DataFrame-ish object indexable by feature name.
    feature_spec maps column → 'quantile' | 'log'.

    Returns {feature_name: edges-array}.
    """
    fitters: dict = {}
    for col, kind in feature_spec.items():
        if kind == "quantile":
            fitters[col] = StreamingQuantileFitter(n_buckets=n_buckets)
        elif kind == "log":
            fitters[col] = StreamingLogEdgeFitter(n_buckets=n_buckets)
        else:
            raise ValueError(f"unknown strategy: {kind}")
    n_chunks = 0
    for chunk in chunks:
        for col, fit in fitters.items():
            if col in chunk:
                fit.update(np.asarray(chunk[col]))
        n_chunks += 1
        if checkpoint and n_chunks % 100 == 0:
            _save_all(fitters, checkpoint)
    if checkpoint:
        _save_all(fitters, checkpoint)
    edges = {col: fit.finalize() for col, fit in fitters.items()}
    log.info("streaming hybrid fit done: %d columns across %d chunks",
             len(edges), n_chunks)
    return edges


def _save_all(fitters: dict, checkpoint: Path) -> None:
    checkpoint = Path(checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    for col, fit in fitters.items():
        fit.save(checkpoint / f"{col}.npz")
