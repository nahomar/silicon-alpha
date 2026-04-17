"""Online calibration of Avellaneda-Stoikov parameters from real TOB data.

sigma
-----
In A-S, σ is the instantaneous volatility of the fair-value process in
PRICE UNITS per time step. We estimate it from realized log-return vol of
the microprice over the warmup window and map it to price units via the
current price level.

kappa
-----
Fill intensity decays exponentially with distance from the touch:
    λ(δ) = A · exp(-κ · δ)
A naive bootstrap estimator: assume at δ = 1 tick, P(fill within Δt) = p1,
and at δ = 2 ticks, P(fill) = p2. Then κ = -ln(p2 / p1) / tick_size.
When fill data is thin, fall back to a sensible default for crypto majors.

tick_size
---------
Inferred from the minimum non-zero spread observed in the warmup.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .avellaneda_stoikov import ASParams

log = logging.getLogger(__name__)


def infer_tick_size(tob: pd.DataFrame) -> float:
    spreads = (tob["ask_px"] - tob["bid_px"]).values
    positive = spreads[spreads > 0]
    if len(positive) == 0:
        return 0.01
    # The tick size is at most the minimum positive spread.
    # Use a rounded estimate to avoid floating-point garbage.
    m = float(np.min(positive))
    # Round to a power of 10 below m.
    if m <= 0:
        return 0.01
    exp = int(np.floor(np.log10(m)))
    return float(10 ** exp)


def realized_sigma_price_units(tob: pd.DataFrame, horizon_steps: int = 1) -> float:
    """σ in price units per step. σ_price ≈ std(Δ log mid) * mid.

    horizon_steps lets you scale to a multi-step horizon via √t.
    """
    mid = (tob["bid_px"] + tob["ask_px"]) / 2.0
    log_ret = np.log(mid).diff().dropna()
    sigma_log = float(log_ret.std())
    price = float(mid.iloc[-1])
    return sigma_log * price * np.sqrt(horizon_steps)


def bootstrap_kappa(
    tob: pd.DataFrame,
    trades: pd.DataFrame,
    window_ms: int = 2000,
    default: float = 500.0,
) -> float:
    """Estimate κ from empirical decay of fill-like events vs distance.

    Approximation (no real fill data yet):
      rate(δ) = # of aggressor trades whose print price is ≥ best_ask - δ
               (for asks) / total time, within the warmup.
    Regress log(rate) on δ → slope = -κ.
    """
    if trades.empty or tob.empty:
        return default
    tick = infer_tick_size(tob)
    mid_series = (tob["bid_px"] + tob["ask_px"]) / 2.0
    ask_series = tob["ask_px"]
    # Align each trade to the nearest TOB snapshot before it.
    tob_ts = tob["ts"].values
    t_ts = trades["ts"].values
    idx = np.searchsorted(tob_ts, t_ts, side="right") - 1
    idx = np.clip(idx, 0, len(tob) - 1)
    mid_at = mid_series.values[idx]
    ask_at = ask_series.values[idx]
    # Distance from ask of buy-aggressor trades, in ticks.
    buy_mask = trades["last_side"].values == +1
    if buy_mask.sum() < 10:
        return default
    delta_ticks = np.clip(
        np.round((trades["last_px"].values[buy_mask] - ask_at[buy_mask]) / tick), 0, 20
    )
    if len(np.unique(delta_ticks)) < 2:
        return default
    counts = pd.Series(delta_ticks).value_counts().sort_index()
    # Only regress if we have at least 2 distance bins with ≥3 events
    counts = counts[counts >= 3]
    if len(counts) < 2:
        return default
    x = counts.index.values * tick          # distance in price units
    y = np.log(counts.values.astype(float))
    slope, _ = np.polyfit(x, y, 1)
    kappa = -float(slope)
    # Sanity: κ must be positive and in a sane range.
    if not np.isfinite(kappa) or kappa <= 0 or kappa > 1e6:
        return default
    return kappa


def calibrate_as_params(
    tob: pd.DataFrame,
    trades: pd.DataFrame,
    gamma: float = 0.02,
    horizon: float = 1.0,
    inv_limit: int = 5,
    sigma_scale: float = 1.0,
) -> ASParams:
    """Produce A-S params tuned to the current regime."""
    tick = infer_tick_size(tob)
    sigma = realized_sigma_price_units(tob) * sigma_scale
    kappa = bootstrap_kappa(tob, trades)
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = tick * 0.5
    log.info("calibrated  tick=%g  sigma=%g  kappa=%g", tick, sigma, kappa)
    return ASParams(gamma=gamma, sigma=sigma, kappa=kappa,
                    horizon=horizon, inv_limit=inv_limit, tick_size=tick)
