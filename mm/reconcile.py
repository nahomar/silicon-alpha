"""Quote → fill reconciler.

Given a log of our emitted quotes (`quotes.parquet`) and the market's trade
prints (`trades.parquet`), determine for each quote whether it would have
been filled within a horizon Δt.

Simple-but-defensible rule:
  bid quote @ price P is filled if a SELL-aggressor trade prints at price ≤ P
  ask quote @ price P is filled if a BUY-aggressor  trade prints at price ≥ P
within `horizon_ms` of the quote timestamp.

Limitations:
  - Ignores queue position; assumes any cross of our price fills us.
    Overestimates fills at ToB. For a more honest simulator, track
    cumulative size at our price level from L2 and only fill once the
    market has consumed everything ahead of us.
  - Ignores partial fills and self-cancellation rules.

Outputs per-quote labels suitable for training the FillProbabilityModel.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_fill_labels(
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    horizon_ms: int = 2000,
    our_size: float = 1.0,
    queue_aware: bool = True,
) -> pd.DataFrame:
    """Return quotes with added columns:
       bid_fill, ask_fill (0/1),
       bid_fill_lat_ms, ask_fill_lat_ms (nan if not filled).

    Queue-aware rule (default): a bid at price P is filled only after
    cumulative sell-aggressor volume at price ≤ P within `horizon_ms`
    exceeds `bid_queue_ahead` (the resting size that was ahead of us when
    we posted). Partial fills are not modeled — a full `our_size` is
    required *after* the queue clears.
    """
    if quotes.empty or trades.empty:
        out = quotes.copy()
        out["bid_fill"] = 0
        out["ask_fill"] = 0
        return out

    q = quotes.sort_values("ts").reset_index(drop=True).copy()
    t = trades.sort_values("ts").reset_index(drop=True).copy()
    q_ts = q["ts"].values
    t_ts = t["ts"].values
    t_px = t["last_px"].values
    t_sz = t["last_sz"].values if "last_sz" in t.columns else np.ones(len(t))
    t_side = t["last_side"].values

    starts = np.searchsorted(t_ts, q_ts, side="left")
    ends = np.searchsorted(t_ts, q_ts + horizon_ms, side="right")

    n = len(q)
    bid_fill = np.zeros(n, dtype=np.int8)
    ask_fill = np.zeros(n, dtype=np.int8)
    bid_lat = np.full(n, np.nan)
    ask_lat = np.full(n, np.nan)

    q_bid = q["bid"].values
    q_ask = q["ask"].values
    bid_queue = q.get("bid_queue_ahead", pd.Series(np.zeros(n))).values
    ask_queue = q.get("ask_queue_ahead", pd.Series(np.zeros(n))).values

    for i in range(n):
        lo, hi = starts[i], ends[i]
        if hi <= lo:
            continue
        w_px = t_px[lo: hi]
        w_sz = t_sz[lo: hi]
        w_side = t_side[lo: hi]
        w_ts = t_ts[lo: hi]

        if queue_aware:
            # Bid side: sell-aggressors at price ≤ our bid consume queue
            bmask = (w_side == -1) & (w_px <= q_bid[i])
            if bmask.any():
                cum = np.cumsum(w_sz[bmask])
                need = bid_queue[i] + our_size
                hit = np.argmax(cum >= need) if (cum >= need).any() else -1
                if hit >= 0:
                    bid_fill[i] = 1
                    bid_lat[i] = w_ts[bmask][hit] - q_ts[i]
            amask = (w_side == +1) & (w_px >= q_ask[i])
            if amask.any():
                cum = np.cumsum(w_sz[amask])
                need = ask_queue[i] + our_size
                hit = np.argmax(cum >= need) if (cum >= need).any() else -1
                if hit >= 0:
                    ask_fill[i] = 1
                    ask_lat[i] = w_ts[amask][hit] - q_ts[i]
        else:
            bhits = (w_side == -1) & (w_px <= q_bid[i])
            if bhits.any():
                bid_fill[i] = 1
                bid_lat[i] = w_ts[np.argmax(bhits)] - q_ts[i]
            ahits = (w_side == +1) & (w_px >= q_ask[i])
            if ahits.any():
                ask_fill[i] = 1
                ask_lat[i] = w_ts[np.argmax(ahits)] - q_ts[i]

    q["bid_fill"] = bid_fill
    q["ask_fill"] = ask_fill
    q["bid_fill_lat_ms"] = bid_lat
    q["ask_fill_lat_ms"] = ask_lat
    return q


def to_training_frame(labeled_quotes: pd.DataFrame, tick_size: float = 0.01
                      ) -> pd.DataFrame:
    """Reshape labeled quotes into (features, label) pairs — one row per side."""
    rows = []
    for _, q in labeled_quotes.iterrows():
        micro = q.get("micro", (q["bid"] + q["ask"]) / 2)
        spread = q["ask"] - q["bid"]
        rel_spread = spread / max(micro, 1e-9)
        imb = q.get("imbalance", 0.0)
        vpin = q.get("vpin", 0.0)
        mu = q.get("mu", 0.0)
        # bid side
        rows.append({
            "side": "bid",
            "distance_ticks": (micro - q["bid"]) / tick_size,
            "imbalance": imb,
            "rel_spread": rel_spread,
            "micro_dev": q.get("micro", 0.0) - (q["bid"] + q["ask"]) / 2,
            "predicted_return": mu,
            "vpin": vpin,
            "queue_ahead": q.get("bid_queue_ahead", 0.0),
            "label": int(q["bid_fill"]),
        })
        # ask side
        rows.append({
            "side": "ask",
            "distance_ticks": (q["ask"] - micro) / tick_size,
            "imbalance": imb,
            "rel_spread": rel_spread,
            "micro_dev": q.get("micro", 0.0) - (q["bid"] + q["ask"]) / 2,
            "predicted_return": -mu,       # mirrored
            "vpin": vpin,
            "queue_ahead": q.get("ask_queue_ahead", 0.0),
            "label": int(q["ask_fill"]),
        })
    return pd.DataFrame(rows)
