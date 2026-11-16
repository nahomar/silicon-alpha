"""Safety guards for live trading (paper orders only; the shadow book keeps its research rules).

1. Surprise-move (shock) guard: a 5-minute SPX move at or above the 99.9th percentile
   of 5-minute moves over the previous 20 sessions (the research rule, expiry_edge
   SHOCK_*), searched 09:35-14:55, at most 2 a day, 30 min apart. No new orders for
   60 minutes after one, the same as after a scheduled release.
2. Big-day guard: SPX 4% or more from yesterday's close (a level 1 circuit breaker is
   7%): no new orders for the rest of the day.
3. Wide-spread guard: the SPX fly's quoted gap above 5% of mid (about 3x today's
   typical 1.69%) means a stressed or illiquid moment: skip the order.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SHOCK_Q, SHOCK_LOOKBACK, MAX_SHOCKS, SHOCK_GAP_MIN = 0.999, 20, 2, 30
FALLBACK_THR, THR_FLOOR, THR_CAP = 0.0025, 0.0015, 0.01     # 0.25% (pre-exam Aug 2026 p99.9 was 0.20%)
BIG_DAY_MOVE = 0.04
WIDE_GAP = 0.05


def shock_threshold(day) -> tuple:
    """(threshold, source) from Yahoo 5-minute S&P bars over the previous 20 sessions."""
    try:
        import yfinance as yf
        day = pd.Timestamp(day).normalize()
        h = yf.Ticker("^GSPC").history(period="60d", interval="5m", auto_adjust=False)
        if h.empty:
            raise RuntimeError("no 5-minute bars")
        idx = pd.DatetimeIndex(h.index)
        idx = idx.tz_convert("America/New_York").tz_localize(None) if idx.tz is not None else idx
        c = pd.Series(h["Close"].to_numpy(), index=idx)
        c = c[c.index.normalize() < day]
        sessions = sorted(set(c.index.normalize()))[-SHOCK_LOOKBACK:]
        if len(sessions) < 10:
            raise RuntimeError(f"only {len(sessions)} sessions")
        moves = []
        for s in sessions:
            x = c[c.index.normalize() == s]
            moves.append(np.abs(np.diff(np.log(x.to_numpy()))))      # within the session only
        thr = float(np.quantile(np.concatenate(moves), SHOCK_Q))
        return float(min(max(thr, THR_FLOOR), THR_CAP)), f"yahoo 5m, {len(sessions)} sessions"
    except Exception as e:
        return FALLBACK_THR, f"fallback ({type(e).__name__})"


class ShockWatch:
    """Feed one price per minute; returns the minute of a new shock, else None."""

    def __init__(self, thr: float, day):
        self.thr = thr
        day = pd.Timestamp(day).normalize()
        self.lo, self.hi = day + pd.Timedelta(hours=9, minutes=35), day + pd.Timedelta(hours=14, minutes=55)
        self.prices: dict = {}
        self.marks: list = []

    def update(self, t, price: float):
        m = pd.Timestamp(t).floor("min")
        if not (price and math.isfinite(price) and price > 0):
            return None
        self.prices[m] = price
        prev = self.prices.get(m - pd.Timedelta(minutes=5))
        if prev is None or not (self.lo <= m <= self.hi) or len(self.marks) >= MAX_SHOCKS:
            return None
        if self.marks and (m - self.marks[-1]) < pd.Timedelta(minutes=SHOCK_GAP_MIN):
            return None
        if abs(math.log(price / prev)) >= self.thr:
            self.marks.append(m)
            return m
        return None


def parity_price(snap: pd.DataFrame) -> float:
    """Index level from one option snapshot: K + C - P at the strike where C and P are closest."""
    try:
        q = snap.dropna(subset=["bid", "ask"])
        q = q[(q["bid"] > 0) & (q["ask"] >= q["bid"])]
        mid = ((q["bid"] + q["ask"]) / 2).groupby([q["option_type"], q["strike"]]).last()
        c, p = mid.xs("C"), mid.xs("P")
        both = c.index.intersection(p.index)
        if not len(both):
            return float("nan")
        k = min(both, key=lambda k: abs(c[k] - p[k]))
        return float(k + c[k] - p[k])
    except Exception:
        return float("nan")


def big_day(price: float, prev_close: float) -> bool:
    return bool(math.isfinite(price) and math.isfinite(prev_close) and prev_close > 0
                and abs(price / prev_close - 1.0) >= BIG_DAY_MOVE)


def too_wide(row: dict) -> bool:
    cm, ct = row.get("fly_credit_mid"), row.get("fly_credit")
    try:
        return bool(cm > 0 and (cm - ct) / cm > WIDE_GAP)
    except TypeError:
        return False
