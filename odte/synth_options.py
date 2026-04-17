"""Synthetic 0DTE SPX tape generator.

Produces a full digital-twin of a 0DTE trading day:
  - Underlying index path  ... Heston SV
  - IV surface snapshots   ... stochastic, skewed, mean-reverting
  - Order / trade tape     ... Hawkes process with informed & uninformed flow

Output: parquet files under reports/odte_synth/ that match the column schema
the rest of the stack expects (same fields as feeds/coinbase_feed.py plus
option-chain specifics).

This is explicitly a toy model. Use real CBOE DataShop / OPRA data before
anything resembling live trading.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "odte_synth"


# ---------------------------------------------------------------------------
# Heston underlying (Euler discretization, full-truncation on variance)
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    mu: float = 0.0              # drift (risk-neutral set to r-d; leave 0 for 0DTE)
    kappa: float = 4.0           # mean-reversion speed of variance
    theta: float = 0.04          # long-run variance (20% vol squared)
    xi: float = 0.6              # vol-of-vol
    rho: float = -0.7            # price-vol correlation (leverage effect)
    v0: float = 0.04             # initial variance


class Heston:
    def __init__(self, params: HestonParams | None = None):
        self.p = params or HestonParams()

    def simulate_paths(self, S0: float, T: float, steps: int, n_paths: int = 1,
                       seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (S, v) arrays of shape (n_paths, steps+1).

        T is in YEARS (so a 6.5-hour trading day is T = 6.5/24/252 ≈ 1.07e-3).
        """
        rng = np.random.default_rng(seed)
        p = self.p
        dt = T / steps
        sqrt_dt = math.sqrt(dt)
        S = np.zeros((n_paths, steps + 1))
        v = np.zeros((n_paths, steps + 1))
        S[:, 0] = S0
        v[:, 0] = p.v0
        for t in range(steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            # correlated Brownian increment for the variance process
            w_v = p.rho * z1 + math.sqrt(1 - p.rho ** 2) * z2
            v_next = v[:, t] + p.kappa * (p.theta - np.maximum(v[:, t], 0)) * dt \
                + p.xi * np.sqrt(np.maximum(v[:, t], 0)) * sqrt_dt * w_v
            v_next = np.maximum(v_next, 0.0)
            S[:, t + 1] = S[:, t] * np.exp((p.mu - 0.5 * v[:, t]) * dt
                                           + np.sqrt(v[:, t]) * sqrt_dt * z1)
            v[:, t + 1] = v_next
        return S, v


# ---------------------------------------------------------------------------
# Stochastic IV surface (SABR-like smile whose level tracks Heston variance)
# ---------------------------------------------------------------------------

@dataclass
class IVSurfaceParams:
    base_skew: float = -0.15
    smile_curvature: float = 0.7
    iv_noise: float = 0.03


class StochasticIVSurface:
    def __init__(self, params: IVSurfaceParams | None = None):
        self.p = params or IVSurfaceParams()

    def sample_surface(self, S: float, variance: float,
                       strikes: np.ndarray, tau: float,
                       rng: np.random.Generator) -> np.ndarray:
        """Return IVs for `strikes` at spot S, instantaneous variance v,
        time-to-expiry tau (years).
        """
        base_iv = math.sqrt(max(variance, 1e-6))
        log_moneyness = np.log(strikes / S)
        skew = self.p.base_skew * log_moneyness
        smile = self.p.smile_curvature * log_moneyness ** 2
        noise = rng.normal(0.0, self.p.iv_noise, size=strikes.shape)
        iv = base_iv + skew + smile + noise
        # Time-to-expiry dampens smile for near-zero tau
        iv = base_iv + (iv - base_iv) * math.sqrt(max(tau, 1e-6) / (1.0 / 365))
        return np.clip(iv, 0.02, 3.0)


# ---------------------------------------------------------------------------
# Adversarial flow — Hawkes process with informed / uninformed agents
# ---------------------------------------------------------------------------

@dataclass
class HawkesFlowParams:
    baseline: float = 0.4
    self_excite: float = 0.2
    decay: float = 2.0
    informed_fraction: float = 0.05
    informed_edge_bps: float = 2.0   # advance-knowledge of next Δprice


class AdversarialFlowHawkes:
    """Simulate informed & uninformed order arrivals.

    Uninformed flow: pure Hawkes on both sides.
    Informed flow:   Hawkes whose side is correlated with the next Δprice
                     with an edge of `informed_edge_bps` basis points.
    Together they induce the toxicity the market maker must defend against.
    """

    def __init__(self, params: HawkesFlowParams | None = None):
        self.p = params or HawkesFlowParams()

    def simulate_orders(self, path: np.ndarray, dt_seconds: float,
                        rng: np.random.Generator) -> pd.DataFrame:
        n = len(path) - 1
        future_dp = np.diff(path)                               # next-step Δprice
        uninformed_intensity = np.full(n, self.p.baseline)
        informed_bias = np.tanh(future_dp / (path[:-1] * self.p.informed_edge_bps / 1e4))
        rows: List[dict] = []
        # decayed self-excitation state
        lam = self.p.baseline
        for t in range(n):
            # Inter-arrival: Poisson with time-varying λ
            lam = self.p.baseline + (lam - self.p.baseline) * math.exp(-self.p.decay * dt_seconds)
            k = rng.poisson(lam * dt_seconds)
            for _ in range(k):
                is_informed = rng.random() < self.p.informed_fraction
                if is_informed:
                    p_buy = 0.5 + 0.5 * float(np.clip(informed_bias[t], -0.95, 0.95))
                else:
                    p_buy = 0.5 + 0.02 * rng.standard_normal()
                side = +1 if rng.random() < p_buy else -1
                size = float(np.exp(rng.normal(math.log(0.1), 1.2)))  # heavy tail
                rows.append({
                    "ts_sec": t * dt_seconds + rng.uniform(0, dt_seconds),
                    "last_px": float(path[t]),
                    "last_sz": size,
                    "last_side": int(side),
                    "informed": bool(is_informed),
                })
                lam += self.p.self_excite
        return pd.DataFrame(rows).sort_values("ts_sec").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level: generate a full 0DTE trading session
# ---------------------------------------------------------------------------

@dataclass
class SessionSpec:
    S0: float = 5500.0
    n_steps: int = 6 * 60 * 10       # ~6 hours × 10 ticks/minute = 3600 ticks
    dt_seconds: float = 6.0          # 6 second step
    strike_grid_pct: Tuple[float, float, int] = (0.98, 1.02, 21)
    seed: int = 7


def generate_session(spec: SessionSpec | None = None, write: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (underlying_df, trade_df, chain_df)."""
    spec = spec or SessionSpec()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)

    heston = Heston()
    iv_gen = StochasticIVSurface()
    flow = AdversarialFlowHawkes()

    T_years = (spec.n_steps * spec.dt_seconds) / (365 * 24 * 3600)
    S, v = heston.simulate_paths(spec.S0, T_years, spec.n_steps, n_paths=1, seed=spec.seed)
    S = S[0]; v = v[0]

    # Build a strike grid anchored at S0
    lo, hi, n_strikes = spec.strike_grid_pct
    strikes = np.linspace(spec.S0 * lo, spec.S0 * hi, n_strikes)

    # Sample IV surface every ~1 minute
    snap_every = max(1, 60 // int(spec.dt_seconds))
    chain_rows: List[dict] = []
    for t in range(0, spec.n_steps, snap_every):
        iv = iv_gen.sample_surface(S[t], v[t], strikes, tau=max((spec.n_steps - t) * spec.dt_seconds
                                                                / (365 * 24 * 3600), 1e-6),
                                   rng=rng)
        for k_idx, K in enumerate(strikes):
            for cp in ("C", "P"):
                chain_rows.append({
                    "ts_sec": t * spec.dt_seconds,
                    "underlying": "SPX",
                    "strike": float(K), "cp": cp,
                    "tau_years": max((spec.n_steps - t) * spec.dt_seconds
                                     / (365 * 24 * 3600), 1e-9),
                    "iv": float(iv[k_idx]),
                    "S": float(S[t]),
                })
    chain = pd.DataFrame(chain_rows)

    trades = flow.simulate_orders(S, spec.dt_seconds, rng)
    under = pd.DataFrame({"ts_sec": np.arange(spec.n_steps + 1) * spec.dt_seconds,
                          "S": S, "v_inst": v})

    if write:
        ts = time.strftime("%Y%m%dT%H%M%S")
        under.to_parquet(OUT_DIR / f"underlying_{ts}.parquet", index=False)
        chain.to_parquet(OUT_DIR / f"chain_{ts}.parquet", index=False)
        trades.to_parquet(OUT_DIR / f"trades_{ts}.parquet", index=False)
        log.info("synth session → %s (steps=%d trades=%d chain=%d)",
                 OUT_DIR, spec.n_steps, len(trades), len(chain))

    return under, trades, chain
