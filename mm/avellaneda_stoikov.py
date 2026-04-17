"""Avellaneda-Stoikov (2008) optimal market-making quotes, extended.

Reservation price: r(q, t) = S - q * γ * σ² * (T - t)
Half-spread:       δ*  = (γ σ²(T-t))/2 + (1/γ) ln(1 + γ/κ)
Optimal bid:       bid = r(q,t) - δ*
Optimal ask:       ask = r(q,t) + δ*

Extensions vs vanilla AS:
  - signal overlay: shift r by our predicted drift μ·Δt
  - toxicity: widen δ when VPIN / Kyle λ are high
  - inventory hard cap: snap quotes that would push inventory past limit
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ASParams:
    gamma: float = 0.1        # risk aversion
    sigma: float = 0.01       # short-horizon volatility (in price units)
    kappa: float = 1.5        # market depth / fill intensity parameter
    horizon: float = 1.0      # normalized time-to-EOD
    inv_limit: int = 100
    tick_size: float = 0.01


def as_quotes(
    s: float,
    q: int,
    t: float,
    params: ASParams,
    predicted_drift: float = 0.0,
    toxicity: float = 0.0,
) -> Tuple[float, float, dict]:
    """Return (bid, ask, diagnostics).

    s : current microprice (fair value)
    q : current inventory (positive = long)
    t : current time (0..params.horizon)
    predicted_drift : μ over the quote horizon (in price units, not log)
    toxicity : normalized in [0, 1] — widens spread linearly by up to 2x
    """
    tau = max(params.horizon - t, 1e-9)
    # Reservation price shifted by our short-horizon forecast
    r = s + predicted_drift - q * params.gamma * (params.sigma ** 2) * tau
    base_half = 0.5 * params.gamma * (params.sigma ** 2) * tau
    info_half = (1 / params.gamma) * np.log(1 + params.gamma / params.kappa)
    half_spread = base_half + info_half
    # toxicity widening: at max toxicity (1.0), double the spread
    half_spread *= (1.0 + toxicity)

    bid = r - half_spread
    ask = r + half_spread

    # Snap to tick grid
    bid = np.floor(bid / params.tick_size) * params.tick_size
    ask = np.ceil(ask / params.tick_size) * params.tick_size

    # Hard inventory cap: pull the side that would make it worse
    if q >= params.inv_limit:
        bid = -np.inf   # don't buy any more
    if q <= -params.inv_limit:
        ask = np.inf    # don't sell any more

    info = dict(reservation=r, half_spread=half_spread, tau=tau,
                inv=q, toxicity=toxicity, drift=predicted_drift)
    return float(bid), float(ask), info
