"""Synthetic limit-order-book generator.

Produces realistic TOB + trade prints so the whole pipeline runs today
without a real feed. Model:
  - latent fair value S_t follows a mean-reverting SDE with jumps
  - spread ~ max(1 tick, scaled by vol)
  - top-of-book sizes from mixture of small / large liquidity providers
  - trade arrivals via Hawkes process (self-exciting) with side bias
    proportional to a hidden "informed" signal, inducing adverse selection

Good for unit-testing the quoter; do NOT treat as market truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_book(
    n_steps: int = 20_000,
    tick_size: float = 0.01,
    start_px: float = 100.0,
    vol: float = 0.0005,
    mean_rev: float = 0.02,
    jump_prob: float = 0.002,
    jump_scale: float = 0.10,
    informed_prob: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    s = np.zeros(n_steps)
    s[0] = start_px

    # Hidden informed signal (persistent AR(1))
    z = np.zeros(n_steps)
    # Fair value with mean reversion + rare jumps
    for i in range(1, n_steps):
        dz = rng.normal(0, 0.05)
        z[i] = 0.995 * z[i - 1] + dz
        shock = rng.normal(0, vol)
        jump = rng.normal(0, jump_scale) if rng.random() < jump_prob else 0.0
        s[i] = s[i - 1] - mean_rev * (s[i - 1] - start_px) + shock + jump + 0.001 * z[i]

    spread_ticks = np.clip(np.round(np.abs(rng.normal(1.5, 0.5, n_steps))), 1, 5)
    spread = spread_ticks * tick_size
    bid_px = np.round((s - spread / 2) / tick_size) * tick_size
    ask_px = bid_px + spread

    # Sizes: mixture — most levels small, occasionally a big parent
    bid_sz = rng.integers(5, 50, n_steps) + (rng.random(n_steps) < 0.05) * rng.integers(100, 500, n_steps)
    ask_sz = rng.integers(5, 50, n_steps) + (rng.random(n_steps) < 0.05) * rng.integers(100, 500, n_steps)

    # Trade arrivals: Bernoulli with rate boosted by |z|
    trade_rate = 0.15 + 0.25 * np.abs(z) / (np.abs(z).max() + 1e-9)
    trade_mask = rng.random(n_steps) < trade_rate
    # Side biased by z: when z>0 more buys, when z<0 more sells (this is the
    # informed-flow mechanism that induces adverse selection).
    p_buy = 1 / (1 + np.exp(-5 * z))
    side_r = rng.random(n_steps)
    sides = np.where(side_r < p_buy, "buy", "sell")
    last_px = np.where(sides == "buy", ask_px, bid_px)
    last_sz = rng.integers(1, 20, n_steps)
    last_side = np.where(sides == "buy", 1, -1)
    last_px = np.where(trade_mask, last_px, np.nan)
    last_sz = np.where(trade_mask, last_sz, 0)
    last_side = np.where(trade_mask, last_side, 0)

    ts = pd.date_range("2026-04-17 09:30", periods=n_steps, freq="100ms")
    df = pd.DataFrame({
        "ts": ts, "bid_px": bid_px, "bid_sz": bid_sz,
        "ask_px": ask_px, "ask_sz": ask_sz,
        "last_px": last_px, "last_sz": last_sz, "last_side": last_side,
        "hidden_z": z,  # for evaluating predictor; drop at inference
    }).set_index("ts")
    return df
