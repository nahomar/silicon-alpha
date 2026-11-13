"""One live feature row, built by the backtest code from the quotes so far.

`chain` holds today's quotes in the historical schema (quote_datetime, root,
expiration, strike, option_type, bid, ask, bid_size, ask_size), recorded once
a minute just before each minute boundary. expiry_edge.minute_grid makes a
quote visible from the next whole minute, exactly as in the backtests.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def day_context(day, events: pd.DataFrame | None = None, wide: pd.DataFrame | None = None) -> dict:
    """Everything known before the open: context features and today's events."""
    from odte.data import calendar as cal
    from odte.data import context as ctx
    from odte.eval.expiry_edge import load_early_closes
    day = pd.Timestamp(day).normalize()
    events = cal.load() if events is None else events
    wide = ctx.load() if wide is None else wide
    timed = events[(events["session"] == day) & events["session_time"].notna()]
    return {"day": day, "events": events, "ctx": ctx.features_for(wide, day),
            "event_times": sorted(set(timed["session_time"])),
            "close_hhmm": load_early_closes().get(day)}


def feature_row(chain: pd.DataFrame, hhmm: str, dctx: dict,
                prev_settle: float = float("nan"), prev_ratio: float = float("nan")) -> dict | None:
    """The decision row at hhmm (features, strikes, prices), or None if unquoted."""
    from odte.data import calendar as cal
    from odte.eval.expiry_edge import implied_forward, minute_grid, session_rows
    day = dctx["day"]
    t = day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
    seen = chain[pd.to_datetime(chain["quote_datetime"]) <= t]     # nothing after t
    if seen.empty:
        return None
    grid = minute_grid(seen)
    F = implied_forward(grid)
    marks = [day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:])) for h in dctx["event_times"]]
    # The close must be given: session_rows otherwise infers an early close
    # from the last quote's time, which on a partial day is always "now".
    rows = session_rows(grid, F, float("nan"), prev_settle, prev_ratio, times=(hhmm,),
                        event_marks=[m for m in marks if m <= t],
                        close_hhmm=dctx["close_hhmm"] or "16:00")
    if not rows:
        return None
    r = rows[0]
    r.update(cal.features_for(dctx["events"], day, hhmm))
    r.update(dctx["ctx"])
    ann = (r["implied_move"] / 0.8) / math.sqrt(max(r["minutes_to_close"], 1.0) / (390 * 252)) * 100.0
    v = r.get("vix1d_prev")
    r["iv_vs_vix1d"] = ann / v if v and np.isfinite(v) and v != 0 else np.nan
    return r
