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


def build(row: dict, symbol: str, max_loss: float, steps: int, max_concession: float,
          tick: float) -> FlyOrder | None:
    """Order for selling the fly in `row` (from features.feature_row), or None."""
    s = scale(symbol)
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
