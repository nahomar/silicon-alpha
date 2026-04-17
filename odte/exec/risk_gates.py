"""Pre-trade risk gates for 0DTE execution.

Upgrades per the Phase-5 spec:

  1. Time-decaying gamma cap — tightens linearly from 14:30 ET to 15:55 ET
     so we're forced to de-risk before the 4:00 PM gamma explosion.
  2. pin_score (0-100) — OI-weighted proximity to high-OI "magnet" strikes
     with a time-to-expiry amplifier. Score > 70 recommends auto closeout.
  3. close_for_pennies helper — produces market-order specs to flatten
     positions whose quote has collapsed toward zero.
  4. Broker-table linear margin approximation (via k-coefficients) is
     already enforced upstream in QPExecutor; we double-check here.

Every gate is a hard veto — failure blocks the candidate order.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

from .qp import InstrumentGreeks

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gamma_cap_scale(minute_of_day: int,
                     tighten_start: int = 300,
                     tighten_end: int = 385,
                     floor: float = 0.10) -> float:
    """Multiplier applied to the base gamma_dollar_cap.

    1.0 until 14:30 ET, then linearly decays to `floor` (default 0.10) by
    15:55 ET. At the close the cap is 10× tighter than the opening bell.
    """
    if minute_of_day < tighten_start:
        return 1.0
    if minute_of_day >= tighten_end:
        return floor
    span = tighten_end - tighten_start
    frac = (minute_of_day - tighten_start) / span
    return 1.0 - (1.0 - floor) * frac


def pin_score(spot: float,
              strikes: np.ndarray,
              open_interest: np.ndarray,
              minutes_to_expiry: float,
              max_window_min: float = 120.0) -> float:
    """0-100 pin-risk score.

    Combines (a) OI-weighted exponential proximity of spot to strikes and
    (b) time-to-expiry amplification within the last 2 hours before expiry.
    Score ≥ 70 means settlement risk is material; the caller should flatten
    the offending legs.
    """
    if minutes_to_expiry > max_window_min or len(strikes) == 0:
        return 0.0
    strikes = np.asarray(strikes, dtype=np.float64)
    oi = np.clip(np.asarray(open_interest, dtype=np.float64), 0, None)
    if oi.sum() <= 0:
        return 0.0
    scale = max(spot * 1e-3, 1e-6)
    proximity = np.exp(-np.abs(strikes - spot) / scale)
    oi_weight = oi / oi.sum()
    base = float((proximity * oi_weight).sum()) * 100.0
    time_amp = max(0.0, (max_window_min - minutes_to_expiry) / max_window_min)
    return float(min(100.0, base * (1.0 + time_amp)))


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    ok: bool
    failed: List[str]
    details: Dict[str, float] = field(default_factory=dict)
    # When the gate vetoes for pin-score, it also reports which symbols
    # the live runner should flatten via close_for_pennies_orders().
    recommend_close_for_pennies: List[str] = field(default_factory=list)


@dataclass
class RiskGates:
    # Static caps
    gross_cap: float = 5e6
    delta_dollar_cap: float = 1e6
    vega_cap: float = 5_000.0
    gamma_dollar_cap: float = 5e7
    per_symbol_cap: float = 100.0
    pin_dist_cap: float = 0.001
    notional_velocity_cap: float = 2e6
    orders_per_sec_cap: int = 20
    equity_buffer_pct: float = 0.95

    # End-of-day gamma tightening
    gamma_tighten_start_min: int = 300
    gamma_tighten_end_min: int = 385
    gamma_floor_scale: float = 0.10

    # Pin risk
    pin_score_warn: float = 50.0
    pin_score_block: float = 70.0
    pin_window_min: float = 120.0

    _order_stamps: deque = field(default_factory=lambda: deque(maxlen=2000), init=False)
    _recent_fills: deque = field(default_factory=lambda: deque(maxlen=2000), init=False)

    # -------------------------------------------------------------------
    def check(self,
              w: np.ndarray,
              g: InstrumentGreeks,
              required_margin: float,
              equity: float,
              strikes: np.ndarray,
              spot: float,
              minute_of_day: int = 0,
              minutes_to_expiry: float = 9999.0,
              open_interest: Optional[np.ndarray] = None,
              symbols: Optional[List[str]] = None) -> GateResult:
        failed: List[str] = []
        det: Dict[str, float] = {}
        aw = np.abs(w)

        gross = float((aw * g.spot * g.multiplier).sum())
        dd = float((aw * np.abs(g.delta) * g.spot * g.multiplier).sum())
        vg = float((aw * np.abs(g.vega) * g.multiplier).sum())
        gd = float((aw * np.abs(g.gamma) * g.spot * g.spot * g.multiplier).sum())
        det.update(gross=gross, delta_dollars=dd, vega=vg, gamma_dollars=gd,
                   required_margin=required_margin, equity=equity)

        if gross > self.gross_cap: failed.append("gross_cap")
        if dd > self.delta_dollar_cap: failed.append("delta_dollar_cap")
        if vg > self.vega_cap: failed.append("vega_cap")

        # Time-decaying gamma cap
        g_scale = gamma_cap_scale(minute_of_day,
                                   self.gamma_tighten_start_min,
                                   self.gamma_tighten_end_min,
                                   self.gamma_floor_scale)
        dyn_gamma_cap = self.gamma_dollar_cap * g_scale
        det["gamma_cap_scale"] = g_scale
        det["gamma_cap_dynamic"] = dyn_gamma_cap
        if gd > dyn_gamma_cap: failed.append("gamma_dollar_cap_eod")

        # Per-symbol cap
        if aw.max(initial=0.0) > self.per_symbol_cap:
            failed.append("per_symbol_cap")

        # Margin buffer
        if required_margin > equity * self.equity_buffer_pct:
            failed.append("margin_buffer")
        det["equity_used_pct"] = required_margin / max(equity, 1e-9)

        # Pin risk: distance + OI-weighted pin score
        active = aw > 1e-6
        close_recs: List[str] = []
        if active.any() and len(strikes) == len(w):
            dists = np.abs(strikes[active] - spot) / max(spot, 1e-9)
            det["min_pin_dist"] = float(dists.min())
            if dists.min() < self.pin_dist_cap:
                failed.append("pin_dist")
            if open_interest is not None and len(open_interest) == len(w):
                score = pin_score(spot, np.asarray(strikes)[active],
                                   np.asarray(open_interest)[active],
                                   minutes_to_expiry, self.pin_window_min)
                det["pin_score"] = score
                if score > self.pin_score_block:
                    failed.append("pin_score_block")
                    if symbols is not None:
                        close_recs = [symbols[i] for i, x in enumerate(active) if x]

        # Velocity + rate limit
        now = time.time()
        while self._recent_fills and now - self._recent_fills[0][0] > 60.0:
            self._recent_fills.popleft()
        last_min_notional = sum(n for _, n in self._recent_fills) + gross
        det["notional_velocity_1m"] = float(last_min_notional)
        if last_min_notional > self.notional_velocity_cap:
            failed.append("notional_velocity")

        while self._order_stamps and now - self._order_stamps[0] > 1.0:
            self._order_stamps.popleft()
        det["orders_in_last_1s"] = float(len(self._order_stamps))
        if len(self._order_stamps) >= self.orders_per_sec_cap:
            failed.append("order_rate_limit")

        result = GateResult(ok=not failed, failed=failed, details=det,
                            recommend_close_for_pennies=close_recs)
        if result.ok:
            self._order_stamps.append(now)
            self._recent_fills.append((now, gross))
        return result


# ---------------------------------------------------------------------------
# Auto-flatten recommendation helper
# ---------------------------------------------------------------------------

def close_for_pennies_orders(symbols: Iterable[str],
                              current_positions: Dict[str, float],
                              bids: Dict[str, float],
                              asks: Dict[str, float],
                              penny_threshold: float = 0.05) -> List[dict]:
    """Produce market-order specs to flatten positions when the quote has
    collapsed toward zero.

    Rules:
      - long qty + spread ≤ `penny_threshold` → sell at bid
      - short qty + ask ≤ `penny_threshold` → buy back at ask

    Returned dicts plug directly into PaperBroker.submit_market(...).
    """
    orders: List[dict] = []
    for sym in symbols:
        qty = current_positions.get(sym, 0.0)
        if abs(qty) < 1e-9:
            continue
        b = bids.get(sym, 0.0); a = asks.get(sym, 0.0)
        if b <= 0 or a <= 0:
            continue
        spread = a - b
        if qty > 0 and spread <= penny_threshold:
            orders.append({"symbol": sym, "side": "sell", "qty": abs(qty),
                            "reason": "pin_flatten_long"})
        elif qty < 0 and a <= penny_threshold:
            orders.append({"symbol": sym, "side": "buy", "qty": abs(qty),
                            "reason": "pin_flatten_short_penny"})
    return orders
