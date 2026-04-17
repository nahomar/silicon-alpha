"""HybridBinTokenizer — scale-invariant hybrid binning for 0DTE microstructure.

Two strategies routed per feature:

  QUANTILE (equal-frequency):
    Use for price-like features (mid, spread, log-return, micro-dev). Gives
    high resolution near the dense center of the distribution while
    gracefully handling fat tails.

  LOG-WIDTH (equal-width in log space):
    Use for multi-scale features (volume, inter-arrival time, trade size).
    Captures 6+ orders of magnitude without dedicating most bins to the
    bulk tiny-size bucket.

Reuses:
  - models.tokenizer.QuantileTokenizer (subclassed for the quantile path)
  - models.tokenizer.edges_to_json / json_to_edges for shared persistence
  - odte._kernel.fused_bin (CPU searchsorted now, CUDA later)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd

from models.tokenizer import QuantileTokenizer, edges_to_json, json_to_edges
from ._kernel import fused_bin

log = logging.getLogger(__name__)

Strategy = Literal["quantile", "log"]


@dataclass
class HybridBinTokenizer:
    """Per-feature binning.

    feature_spec: {feature_name: "quantile" | "log"}
    n_buckets:    number of buckets per feature (same for all — keeps the
                  transformer vocab flat; use a multi-vocab embedding if you
                  want feature-specific widths)
    log_floor:    additive floor for log features so log(0) never happens
    """

    n_buckets: int = 64
    feature_spec: Dict[str, Strategy] = field(default_factory=dict)
    edges: Dict[str, np.ndarray] = field(default_factory=dict)
    log_floor: float = 1e-9

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, feature_spec: Dict[str, Strategy] | None = None
            ) -> "HybridBinTokenizer":
        if feature_spec is not None:
            self.feature_spec = dict(feature_spec)
        if not self.feature_spec:
            raise ValueError("feature_spec is empty; pass {col: 'quantile'|'log'}")
        self.edges = {}
        for col, kind in self.feature_spec.items():
            if col not in df.columns:
                log.warning("skipping %s: not in DataFrame", col)
                continue
            x = df[col].dropna().values.astype(np.float64)
            if len(x) < max(2 * self.n_buckets, 50):
                log.warning("skipping %s: too few samples (%d)", col, len(x))
                continue
            if kind == "quantile":
                edges = self._fit_quantile(x)
            elif kind == "log":
                edges = self._fit_log(x)
            else:
                raise ValueError(f"unknown strategy for {col}: {kind!r}")
            self.edges[col] = edges
        return self

    def _fit_quantile(self, x: np.ndarray) -> np.ndarray:
        qs = np.linspace(0, 1, self.n_buckets + 1)
        edges = np.quantile(x, qs)
        # de-duplicate edges (happens when values repeat — e.g. many zeros)
        edges = np.unique(edges)
        if len(edges) < self.n_buckets + 1:
            # pad with linearly-interpolated extensions to preserve vocab size
            pad = np.linspace(edges[-1], edges[-1] + 1, self.n_buckets + 1 - len(edges) + 1)[1:]
            edges = np.concatenate([edges, pad])
        edges[0], edges[-1] = -np.inf, np.inf
        return edges

    def _fit_log(self, x: np.ndarray) -> np.ndarray:
        x = np.abs(x) + self.log_floor
        x = x[x > 0]
        if len(x) == 0:
            raise ValueError("all values <= 0 after floor; check log feature")
        lo, hi = float(np.log(np.min(x))), float(np.log(np.max(x)))
        if hi - lo < 1e-9:
            hi = lo + 1.0
        edges = np.exp(np.linspace(lo, hi, self.n_buckets + 1))
        edges[0], edges[-1] = 0.0, np.inf
        return edges

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col, kind in self.feature_spec.items():
            if col not in self.edges or col not in df.columns:
                continue
            vals = df[col].values.astype(np.float64)
            if kind == "log":
                vals = np.abs(vals) + self.log_floor
            out[col] = fused_bin(vals, self.edges[col])
        return pd.DataFrame(out, index=df.index).astype(np.int16)

    def tokenize_batch(self, df: pd.DataFrame,
                       feature_order: Iterable[str] | None = None) -> np.ndarray:
        """Return a 2-D int16 array of shape (T, F) — one col per feature."""
        codes = self.encode(df)
        order = list(feature_order) if feature_order else list(codes.columns)
        return codes[order].values.astype(np.int16)

    def decode(self, codes: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in codes.columns:
            e = self.edges.get(col)
            if e is None:
                continue
            mids = np.zeros(self.n_buckets)
            for i in range(self.n_buckets):
                lo, hi = e[i], e[i + 1]
                if np.isinf(lo):
                    mids[i] = e[i + 1] - (e[i + 2] - e[i + 1])
                elif np.isinf(hi):
                    mids[i] = e[i] + (e[i] - e[i - 1])
                else:
                    mids[i] = 0.5 * (lo + hi)
            idx = np.clip(codes[col].values.astype(int), 0, self.n_buckets - 1)
            out[col] = mids[idx]
        return pd.DataFrame(out, index=codes.index)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        payload = {
            "n_buckets": self.n_buckets,
            "feature_spec": self.feature_spec,
            "log_floor": self.log_floor,
            "edges": {k: v.tolist() for k, v in self.edges.items()},
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "HybridBinTokenizer":
        path = Path(path)
        p = json.loads(path.read_text())
        t = cls(n_buckets=p["n_buckets"], feature_spec=p["feature_spec"],
                log_floor=p.get("log_floor", 1e-9))
        t.edges = {k: np.array(v) for k, v in p["edges"].items()}
        return t

    # ------------------------------------------------------------------
    # interop with the legacy single-strategy tokenizer
    # ------------------------------------------------------------------
    @classmethod
    def from_quantile(cls, qt: QuantileTokenizer,
                      columns: Iterable[str] | None = None) -> "HybridBinTokenizer":
        """Adapt an existing QuantileTokenizer for reuse."""
        cols = list(columns) if columns else list(qt.edges.keys())
        t = cls(n_buckets=qt.n_buckets,
                feature_spec={c: "quantile" for c in cols})
        t.edges = {c: qt.edges[c] for c in cols if c in qt.edges}
        return t


def default_microstructure_spec() -> Dict[str, Strategy]:
    """Reasonable feature_spec for L2/trade streams (our smoke test)."""
    return {
        "ret": "quantile",
        "mid": "quantile",
        "micro_dev": "quantile",
        "spread": "log",
        "bid_sz": "log",
        "ask_sz": "log",
        "last_sz": "log",
        "inter_arrival_ms": "log",
    }
