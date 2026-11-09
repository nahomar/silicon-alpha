"""Strategies the bot shadow-trades every day on live quotes.

Each decision time, every strategy looks at the same decision row and model
prediction and says SELL the fly (+1), BUY the fly (-1) or nothing (0).
Shadow trades place no orders. After the close each is priced on the
settlement at three entry prices (mid, half the gap, full touch), sized to
the same worst case, so they compare like for like. Paper trading produces
new unseen days, the honest test for all of them at once.

Record on unseen SPX data 2017H1-2021H1 (today's cost), for reference:
  sell_only       #1 sell-only            -$5.3k / 637 days (always-sell: -$36.9k)
  full_model      #1 (sells and buys)     -$43.0k / 713 days (always-sell: -$50.1k)
  always_sell     no model, -postevt      lost; the reference every model must beat
  event_seller    CPI/NFP/FOMC days only, no model; sealed NDX/RUT 2025-26 PASS
                  (event days +$581/day over other days at mid, CI [53, 1088])
"""
from __future__ import annotations

import math

MARGIN = 0.15


EVENT_FLAGS = ("ev_cpi", "ev_nfp", "ev_fomc", "ev_fomc_sep")


# `sig` comes from odte.live.model.signal: +1 overpriced, -1 big move expected, 0 neither.
def sell_only(sig, after_event, row=None):
    if sig == 1 and not after_event:
        return 1, "overpriced"
    return 0, "after event" if sig == 1 else "not overpriced enough"


def full_model(sig, after_event, row=None):
    if sig == -1:
        return -1, "big move expected"           # buys are allowed after events
    return sell_only(sig, after_event)


def always_sell(sig, after_event, row=None):
    return (0, "after event") if after_event else (1, "always")


def event_seller(sig, after_event, row=None):
    """Sell on scheduled CPI / jobs / Fed days, outside the hour after a
    release, unless yesterday's VIX/VIX3M > 1 (stress). No model."""
    row = row or {}
    if not any(float(row.get(k, 0) or 0) > 0 for k in EVENT_FLAGS):
        return 0, "not an event day"
    if after_event:
        return 0, "within 60 min after the release"
    v = row.get("vix_over_vix3m")
    if v is not None and math.isfinite(v) and v > 1.0:
        return 0, "stress: VIX above VIX3M"
    return 1, "event day"


STRATEGIES = {"sell_only": sell_only, "full_model": full_model, "always_sell": always_sell,
              "event_seller": event_seller}
ORDER_STRATEGIES = ("event_seller", "sell_only")   # paper orders when either says sell


def shadow_pnl(side: int, row: dict, settle: float, budget: float = 300.0, fee: float = 1.20) -> dict:
    """$ P&L of one shadow trade at mid / half-gap / touch, each sized to `budget`
    worst case. Prices in signal-symbol points; settle = that index's close."""
    K, W, cm = row["K"], row["W"], row["fly_credit_mid"]
    if side == 0 or not (math.isfinite(W) and cm > 0):
        return {}
    intrinsic = min(abs(settle - K), W)
    fees4 = 4 * fee
    out = {}
    if side == 1:
        touch = row["fly_credit"]
        for name, c in (("mid", cm), ("half", cm - 0.5 * (cm - touch)), ("touch", touch)):
            risk = (W - c) * 100 + fees4
            out[name] = ((c - intrinsic) * 100 - fees4) / risk * budget if risk > 0 else float("nan")
    else:
        touch = row["fly_debit"]
        for name, d in (("mid", cm), ("half", cm + 0.5 * (touch - cm)), ("touch", touch)):
            risk = d * 100 + fees4
            out[name] = ((intrinsic - d) * 100 - fees4) / risk * budget if risk > 0 else float("nan")
    return out
