"""Return tokenizer — discretize continuous returns into a finite codebook.

Uses quantile-based bucketing per ticker so each bucket has equal mass.
Stateful: learns the quantile edges on a training slice and applies them
at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import json
import numpy as np
import pandas as pd

from .config import Config


# ---------------------------------------------------------------------------
# Edge (de)serialization helpers — shared with odte.tokenizer.HybridBinTokenizer.
# ---------------------------------------------------------------------------

def edges_to_json(edges: Dict[str, np.ndarray]) -> str:
    """Serialize a {feature_name: float-edge-array} dict to JSON."""
    return json.dumps({k: v.tolist() for k, v in edges.items()})


def json_to_edges(payload: str) -> Dict[str, np.ndarray]:
    """Inverse of edges_to_json. Handles +/-inf sentinels in the array."""
    raw = json.loads(payload)
    out: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        out[k] = np.array(
            [float(x) if x not in ("Infinity", "-Infinity", None) else np.inf * (1 if x == "Infinity" else -1)
             for x in v]
        ) if isinstance(v, list) and any(isinstance(x, str) for x in v) else np.array(v)
    return out


@dataclass
class QuantileTokenizer:
    n_buckets: int = 64
    edges: Dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, returns: pd.DataFrame) -> "QuantileTokenizer":
        self.edges = {}
        qs = np.linspace(0, 1, self.n_buckets + 1)
        for col in returns.columns:
            x = returns[col].dropna().values
            if len(x) < 10:
                continue
            edges = np.quantile(x, qs)
            edges[0], edges[-1] = -np.inf, np.inf
            self.edges[col] = edges
        return self

    def encode(self, returns: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in returns.columns:
            if col not in self.edges:
                continue
            out[col] = np.clip(
                np.searchsorted(self.edges[col][1:-1], returns[col].values),
                0, self.n_buckets - 1,
            ).astype(np.int32)
        return pd.DataFrame(out, index=returns.index)

    def decode(self, codes: pd.DataFrame) -> pd.DataFrame:
        """Map token → midpoint of its bucket (expected return)."""
        out = {}
        for col in codes.columns:
            e = self.edges[col]
            mids = (e[1:] + e[:-1]) / 2
            # Handle inf endpoints by using finite interior means.
            mids[0] = e[1] - (e[2] - e[1])
            mids[-1] = e[-2] + (e[-2] - e[-3])
            out[col] = mids[codes[col].values]
        return pd.DataFrame(out, index=codes.index)

    def save(self, path: Path) -> None:
        payload = {k: v.tolist() for k, v in self.edges.items()}
        path.write_text(json.dumps({"n_buckets": self.n_buckets, "edges": payload}))

    @classmethod
    def load(cls, path: Path) -> "QuantileTokenizer":
        p = json.loads(path.read_text())
        t = cls(n_buckets=p["n_buckets"])
        t.edges = {k: np.array(v) for k, v in p["edges"].items()}
        return t


def build_tokenizer(returns: pd.DataFrame, cfg: Config | None = None) -> QuantileTokenizer:
    cfg = cfg or Config()
    return QuantileTokenizer(n_buckets=cfg.n_return_buckets).fit(returns)
