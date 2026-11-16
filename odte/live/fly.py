"""Iron fly orders: which contracts, how many, and the limit-price ladder.

The fly (strike K, wing width W) comes from the decision row, so it is the
fly the backtests priced. We SELL it for a credit:
  short K call + short K put, long K+W call + long K-W put.
For XSP (1/10 SPX) strikes and prices are divided by 10.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

FEE_PER_LEG = 1.20          # $ per contract per leg, as in the backtests
MULT = 100.0


@dataclass
class FlyOrder:
    symbol: str
    K: float
    W: float
    qty: int
    mid: float              # mid credit (points of the traded symbol)
    touch: float            # credit if crossing every leg now
    ladder: list            # limit credits to try, best first
    worst_case: float       # $ max loss of the whole order at the lowest ladder price


def scale(symbol: str) -> float:
    return 0.1 if symbol == "XSP" else 1.0


def _round_tick(x: float, tick: float, down: bool = True) -> float:
    n = x / tick
    return round((math.floor(n + 1e-9) if down else math.ceil(n - 1e-9)) * tick, 2)


def ladder(mid: float, touch: float, steps: int, max_concession: float, tick: float) -> list:
    """Limit credits from mid down toward the touch, never past
    mid - max_concession x (mid - touch). Rounded to the tick, deduplicated."""
    floor_ = mid - max_concession * max(mid - touch, 0.0)
    out = []
    for i in range(steps + 1):
        p = mid - (mid - floor_) * i / max(steps, 1)
        p = round(round(p / tick) * tick, 2) if i == 0 else _round_tick(p, tick, down=True)
        p = max(p, _round_tick(floor_, tick, down=False))
        if not out or p < out[-1]:
            out.append(p)
    return out


def worst_case_per_fly(W: float, credit: float) -> float:
    return (W - credit) * MULT + 4 * FEE_PER_LEG


def fly_from_quotes(q, F: float, wing_mult: float = 1.5) -> dict | None:
    """The backtests' iron fly built from one symbol's own quotes.

    Body at the strike nearest the forward with both a call and a put quoted;
    wings at the smallest symmetric distance >= wing_mult x straddle mid.
    q: rows with strike, option_type, bid, ask (latest per contract)."""
    import numpy as np
    if q is None or len(q) == 0 or not (F > 0):
        return None
    q = q[(q["bid"] > 0) & (q["ask"] >= q["bid"])]
    c = q[q["option_type"] == "C"].set_index("strike")
    p = q[q["option_type"] == "P"].set_index("strike")
    both = c.index.intersection(p.index)
    if not len(both):
        return None
    K = float(both[np.argmin(np.abs(both.to_numpy(float) - F))])
    mid = lambda df, k: (df.at[k, "bid"] + df.at[k, "ask"]) / 2
    straddle = mid(c, K) + mid(p, K)
    dists = sorted(d for d in {round(k - K, 4) for k in c.index} & {round(K - k, 4) for k in p.index}
                   if d >= wing_mult * straddle)
    if not dists or straddle <= 0:
        return None
    W = dists[0]
    return {"K": K, "W": W, "straddle_mid": straddle,
            "fly_credit_mid": straddle - (mid(c, K + W) + mid(p, K - W)),
            "fly_credit": c.at[K, "bid"] + p.at[K, "bid"] - c.at[K + W, "ask"] - p.at[K - W, "ask"],
            "fly_debit": c.at[K, "ask"] + p.at[K, "ask"] - c.at[K + W, "bid"] - p.at[K - W, "bid"]}


def build(row: dict, symbol: str, max_loss: float, steps: int, max_concession: float,
          tick: float, native: bool = False) -> FlyOrder | None:
    """Order for selling the fly in `row`, or None. native=True: row is already
    priced in the traded symbol (its own quotes); otherwise SPX row x scale."""
    s = 1.0 if native else scale(symbol)
    K, W = row["K"] * s, row["W"] * s
    mid, touch = row["fly_credit_mid"] * s, row["fly_credit"] * s
    if not (math.isfinite(W) and mid > 0 and touch > 0):
        return None
    lad = ladder(mid, touch, steps, max_concession, tick)
    per = worst_case_per_fly(W, lad[-1])
    qty = int(max_loss // per) if per > 0 else 0
    if qty < 1:
        return None
    return FlyOrder(symbol, K, W, qty, mid, touch, lad, qty * per)


def settle_pnl(order_credit: float, qty: int, K: float, W: float, settle: float) -> float:
    """$ P&L at expiry of a short fly sold for order_credit (traded-symbol points)."""
    intrinsic = min(abs(settle - K), W)
    return (order_credit - intrinsic) * MULT * qty - 4 * FEE_PER_LEG * qty


def vertical_from_quotes(q, K: float, W: float) -> dict | None:
    """Put credit spread (sell P K, buy P K-W) and call credit spread (sell C K,
    buy C K+W) from one symbol's latest quotes: mid and touch credits."""
    if q is None or len(q) == 0:
        return None
    last = q.sort_values("quote_datetime").groupby(["option_type", "strike"]).tail(1) \
        if "quote_datetime" in q else q
    b = last.set_index(["option_type", "strike"])
    try:
        pk, pw = b.loc[("P", K)], b.loc[("P", K - W)]
        ck, cw = b.loc[("C", K)], b.loc[("C", K + W)]
    except KeyError:
        return None
    mid = lambda r: (r["bid"] + r["ask"]) / 2
    out = {"K": K, "W": W,
           "put_mid": mid(pk) - mid(pw), "put_touch": pk["bid"] - pw["ask"],
           "call_mid": mid(ck) - mid(cw), "call_touch": ck["bid"] - cw["ask"]}
    return out if out["put_mid"] > 0 and out["call_mid"] > 0 else None
