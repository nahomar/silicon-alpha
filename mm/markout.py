"""Markout computation — the quant's truth serum.

For each fill, compute P&L at multiple forward horizons:
    markout(h) = side * (mid(t+h) - fill_px)

Interpretation:
  - Markout climbs monotonically → fills are benign, you're earning spread.
  - Markout dips at short h, recovers → queuing adverse selection.
  - Markout keeps falling → toxic flow; widen spreads and/or pull quotes.

Also produces the aggregated "edge curve": average markout as a function of
horizon, grouped by signal decile. That curve is the model's report card.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_markouts(fills: pd.DataFrame, tob: pd.DataFrame,
                     horizons: Iterable[int] = (1, 5, 25, 100, 500)) -> pd.DataFrame:
    """fills: columns [ts, side ('buy'/'sell'), fill_px]
       tob:   indexed on same ts resolution with a 'mid' column
    """
    horizons = list(horizons)
    mid = tob["mid"].reindex(fills["ts"]).reset_index(drop=True)
    out = fills.reset_index(drop=True).copy()
    out["mid_at_fill"] = mid.values
    # Build a lookup from position index in tob for each fill.
    tob_idx = tob.index
    pos = np.searchsorted(tob_idx, fills["ts"].values)
    for h in horizons:
        fwd_pos = np.clip(pos + h, 0, len(tob) - 1)
        fwd_mid = tob["mid"].values[fwd_pos]
        sign = np.where(out["side"].values == "buy", 1.0, -1.0)
        out[f"mkout_{h}"] = sign * (fwd_mid - out["fill_px"].values)
    return out


def edge_curve(markouts: pd.DataFrame, signal_col: str,
               horizons: Iterable[int] = (1, 5, 25, 100, 500), n_buckets: int = 5
               ) -> pd.DataFrame:
    """Mean markout by signal decile across horizons."""
    df = markouts.copy()
    df["bucket"] = pd.qcut(df[signal_col], n_buckets, duplicates="drop").astype(str)
    cols = [f"mkout_{h}" for h in horizons if f"mkout_{h}" in df.columns]
    return df.groupby("bucket")[cols].mean()
