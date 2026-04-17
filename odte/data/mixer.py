"""Quality-weighted dataset mixer — the 90/10 rule.

Research consistently shows that 1B well-chosen tokens outperforms 10B
naively-sampled tokens. This module implements weighted sampling over
packed parquet shards based on per-row "quality" heuristics appropriate
for 0DTE microstructure:

  • expiry proximity — weight goes up as we approach 16:00 ET (gamma zone)
  • IV-rank regime     — high-IV-regime rows get heavier weight
  • trade activity     — rows bracketing real trade prints weighted higher
  • underlying-move magnitude — large-move rows (tails) get extra weight

Exposes a drop-in replacement for odte.train.pretrain_tradefm.ShardTokenDataset
that samples with these weights rather than uniformly.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

log = logging.getLogger(__name__)


@dataclass
class QualityWeights:
    """Named weights for each heuristic. Set any to 0 to disable."""
    expiry_proximity: float = 1.0
    iv_rank: float = 0.5
    trade_activity: float = 1.0
    move_magnitude: float = 0.75


def compute_row_weights(df: pd.DataFrame, w: QualityWeights) -> np.ndarray:
    """Given a packed shard DataFrame (with optional feature cols), return a
    per-row weight vector normalized to sum=1. Missing columns ignored."""
    n = len(df)
    if n == 0:
        return np.ones(0)
    weight = np.ones(n, dtype=np.float64)

    # Expiry proximity: bias toward last 90 minutes before close (0DTE zone).
    if w.expiry_proximity > 0 and "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], unit="ms", errors="coerce", utc=True)
        # seconds until 16:00 ET assuming same calendar day as ts
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        dist_to_close = (20 * 60) - minute_of_day  # 16:00 ET = 20:00 UTC
        dist_to_close = dist_to_close.clip(lower=0)
        prox = np.exp(-dist_to_close.values / 90.0)
        weight *= (1.0 + w.expiry_proximity * prox)

    # IV-rank: rows with iv well above shard median get extra weight.
    if w.iv_rank > 0 and "iv" in df.columns:
        iv = df["iv"].astype(float).fillna(df["iv"].median())
        z = (iv - iv.median()) / (iv.std() + 1e-9)
        weight *= (1.0 + w.iv_rank * np.clip(z.values, 0, 3))

    # Trade activity: presence of a real trade size bumps weight.
    if w.trade_activity > 0 and "last_sz" in df.columns:
        sz = df["last_sz"].astype(float).fillna(0.0).values
        weight *= (1.0 + w.trade_activity * np.log1p(sz) / np.log1p(sz.max() + 1))

    # Move magnitude: |Δmid| tails.
    if w.move_magnitude > 0 and "mid" in df.columns:
        mid = df["mid"].astype(float).ffill().fillna(0.0).values
        d = np.abs(np.diff(mid, prepend=mid[0]))
        if d.std() > 0:
            weight *= (1.0 + w.move_magnitude * np.clip(d / d.std(), 0, 3))

    weight /= weight.sum()
    return weight


class WeightedShardTokenDataset(IterableDataset):
    """Stream token contexts sampled with quality weights within each shard."""

    def __init__(self, shard_paths: Iterable[Path], ctx_len: int,
                 weights: Optional[QualityWeights] = None,
                 shuffle_buffer: int = 128, seed: int = 0):
        super().__init__()
        self.shards = sorted(Path(p) for p in shard_paths)
        self.ctx_len = ctx_len
        self.w = weights or QualityWeights()
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        shard_order = list(self.shards)
        rng.shuffle(shard_order)
        pack = np.empty(0, dtype=np.int32)
        for shard in shard_order:
            df = pd.read_parquet(shard)
            if df.empty:
                continue
            if "tokens" not in df.columns:
                continue
            w_vec = compute_row_weights(df, self.w)
            n_draws = min(len(df), int(len(df) * 1.0))  # whole shard, weighted order
            idx = rng.choice(len(df), size=n_draws, replace=False, p=w_vec)
            for i in idx:
                arr = np.asarray(df.iloc[int(i)]["tokens"], dtype=np.int32)
                pack = np.concatenate([pack, arr]) if len(pack) else arr
                while len(pack) >= self.ctx_len + 1:
                    yield torch.as_tensor(pack[: self.ctx_len + 1], dtype=torch.long)
                    pack = pack[self.ctx_len:]


def describe_weighting(shard_paths: Iterable[Path],
                       weights: Optional[QualityWeights] = None,
                       max_shards: int = 10) -> dict:
    """Diagnostic: what fraction of the weighted mass lands in the top-10% of rows?

    An informative target is ≥ 0.40 (i.e. top 10% gets 40%+ of weight).
    If it's ~0.10 we're effectively uniform and the mixer is no-op.
    """
    weights = weights or QualityWeights()
    shards = list(shard_paths)[:max_shards]
    concentrations = []
    for p in shards:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        w = compute_row_weights(df, weights)
        top = int(np.ceil(len(w) * 0.1))
        concentrations.append(float(np.sort(w)[-top:].sum()))
    return {
        "top10_mass_mean": float(np.mean(concentrations)) if concentrations else None,
        "top10_mass_min": float(np.min(concentrations)) if concentrations else None,
        "n_shards": len(concentrations),
    }
