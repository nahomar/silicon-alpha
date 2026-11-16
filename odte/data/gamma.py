"""Dealer-gamma features from pre-open open interest (data/oi, fetch_oi.py).

Dealer hedging is the most 0DTE-specific force there is. When dealers are
net long gamma they sell rallies and buy dips, which damps moves and pins
the index near large strikes. When they are net short gamma they chase the
move and amplify it.

Approximations, the standard ones for public-data gamma estimates:
  * Positioning: dealers long calls and short puts (customers sell calls,
    buy puts). Real positioning is unknown; this is the usual proxy.
  * Volatility: one implied vol for every strike and expiry, backed out of
    the session's own ATM straddle at the decision minute.
  * Open interest: as of the previous close (published pre-open), so known
    before the session.

Features (all known at decision time t):
  gex_net        net dealer gamma, $bn per 1% move, next ~5 expiries
  gex_0dte       the same, today's expiry only
  gex_flip_dist  (zero-gamma level - F) / (F x implied move); NaN if no flip
                 within +/-5%
  maxoi_dist     (largest-0DTE-OI strike - F) / (F x implied move): the magnet
  oi_pcr_0dte    log(0DTE put OI / call OI) within +/-2% of F
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OI_DIR = ROOT / "data" / "oi"
GEX_FEATURES = ["gex_net", "gex_0dte", "gex_flip_dist", "maxoi_dist", "oi_pcr_0dte"]
_MIN_PER_YEAR = 390.0 * 252.0


def _load(day) -> pd.DataFrame | None:
    p = OI_DIR / f"{pd.Timestamp(day):%Y-%m-%d}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    return d[d["oi"] > 0] if len(d) else None


def _gamma(S, K, sigma, tau):
    """Black-Scholes gamma, vectorised; tau in years."""
    st = sigma * np.sqrt(np.maximum(tau, 1e-8))
    d1 = (np.log(S / K) + 0.5 * st ** 2) / st
    return np.exp(-0.5 * d1 ** 2) / (math.sqrt(2 * math.pi) * S * st)


def features_for(oi: pd.DataFrame | None, day, F: float, implied_move: float,
                 minutes_to_close: float) -> dict:
    out = {k: np.nan for k in GEX_FEATURES}
    if oi is None or not (F > 0) or not (implied_move > 0) or minutes_to_close <= 0:
        return out
    day = pd.Timestamp(day).normalize()
    tau0 = minutes_to_close / _MIN_PER_YEAR
    sigma = implied_move / (0.8 * math.sqrt(tau0))           # straddle ~ 0.8 sigma sqrt(tau) F
    bdays = np.busday_count(day.date(), oi["expiration"].dt.date.to_numpy().astype("datetime64[D]"))
    tau = (minutes_to_close + 390.0 * np.maximum(bdays, 0)) / _MIN_PER_YEAR
    K = oi["strike"].to_numpy(float)
    sign = np.where(oi["option_type"].to_numpy() == "C", 1.0, -1.0)   # dealers long C, short P
    w = oi["oi"].to_numpy(float) * 100.0
    is0 = (oi["expiration"].dt.normalize() == day).to_numpy()

    def net(S, mask=None):
        g = _gamma(S, K, sigma, tau) * sign * w * S * S * 0.01
        return float(g[mask].sum() if mask is not None else g.sum())

    out["gex_net"] = net(F) / 1e9
    out["gex_0dte"] = net(F, is0) / 1e9 if is0.any() else np.nan
    grid = F * np.linspace(0.95, 1.05, 41)
    vals = np.array([net(S) for S in grid])
    flips = np.flatnonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))
    if len(flips):
        i = flips[np.argmin(np.abs(grid[flips] - F))]
        x0, x1, y0, y1 = grid[i], grid[i + 1], vals[i], vals[i + 1]
        zero = x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x0
        out["gex_flip_dist"] = (zero - F) / (F * implied_move)
    if is0.any():
        z = oi[is0]
        by_k = z.groupby("strike")["oi"].sum()
        out["maxoi_dist"] = (float(by_k.idxmax()) - F) / (F * implied_move)
        near = z[(z["strike"] - F).abs() <= 0.02 * F]
        c, p = near.loc[near["option_type"] == "C", "oi"].sum(), near.loc[near["option_type"] == "P", "oi"].sum()
        out["oi_pcr_0dte"] = float(np.log1p(p) - np.log1p(c))   # log ratio: stays bounded
    return out


def gamma_features(ds: pd.DataFrame) -> pd.DataFrame:
    cache, rows = {}, []
    for d, F, im, mtc in zip(ds["day"], ds["F"], ds["implied_move"], ds["minutes_to_close"]):
        if d not in cache:
            cache[d] = _load(d)
        rows.append(features_for(cache[d], d, F, im, mtc))
    return pd.DataFrame(rows, index=ds.index)[GEX_FEATURES]
