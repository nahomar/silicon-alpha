"""Adverse-selection / toxicity metrics.

- VPIN (Easley, López de Prado, O'Hara, 2012): probability that the flow in
  a volume bucket is informed. Uses bulk volume classification.
- Kyle's lambda: price impact per unit signed volume.
- Realized variance and variance ratio for regime detection.

These surface as inputs to the quoter: widen spreads when toxicity is high.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


def bulk_volume_classify(trades: pd.DataFrame, price_col: str = "last_px",
                          size_col: str = "last_sz",
                          sigma_window: int = 50) -> pd.DataFrame:
    """Assign buy/sell probabilities to trades using the BVC method.

    Pr(buy) = Φ(Δp / (σ·√Δt)); we approximate Δt as constant.
    """
    dp = trades[price_col].diff().fillna(0)
    sigma = dp.rolling(sigma_window).std().replace(0, np.nan).bfill().fillna(dp.std() or 1e-9)
    z = dp / sigma
    p_buy = pd.Series(norm.cdf(z), index=trades.index)
    buy_v = trades[size_col] * p_buy
    sell_v = trades[size_col] * (1 - p_buy)
    return pd.DataFrame({"buy_v": buy_v, "sell_v": sell_v, "p_buy": p_buy})


def vpin(trades: pd.DataFrame, bucket_size: Optional[float] = None,
         window: int = 50) -> pd.Series:
    """Volume-Synchronized Probability of Informed Trading.

    1. Classify volume with BVC.
    2. Group by equal-volume buckets.
    3. VPIN = moving avg of |buy - sell| / (buy + sell) across last `window`
       buckets.
    """
    bvc = bulk_volume_classify(trades)
    v = trades["last_sz"].cumsum()
    if bucket_size is None:
        bucket_size = float(v.iloc[-1] / max(len(trades) / 20, 1))
    bvc["bucket"] = (v // bucket_size).astype(int)
    agg = bvc.groupby("bucket")[["buy_v", "sell_v"]].sum()
    imb = (agg["buy_v"] - agg["sell_v"]).abs() / (agg["buy_v"] + agg["sell_v"]).replace(0, np.nan)
    vp = imb.rolling(window, min_periods=1).mean()
    # Map bucket-level VPIN back to trade timestamps
    out = pd.Series(np.nan, index=trades.index)
    mapping = dict(zip(agg.index, vp.values))
    out.loc[:] = bvc["bucket"].map(mapping).values
    return out.ffill().fillna(0.0).rename("vpin")


def kyle_lambda(trades: pd.DataFrame, window: int = 200) -> pd.Series:
    """Rolling Kyle lambda = slope of |Δprice| on signed volume.

    High lambda ⇒ flow is moving the price ⇒ informed / toxic.
    """
    dp = trades["last_px"].diff()
    side = np.sign(trades.get("last_side", pd.Series(np.sign(dp), index=trades.index)))
    signed_v = trades["last_sz"] * side.replace(0, np.nan).ffill().fillna(0)
    num = (dp.abs() * signed_v.abs()).rolling(window, min_periods=10).sum()
    den = (signed_v ** 2).rolling(window, min_periods=10).sum().replace(0, np.nan)
    return (num / den).rename("kyle_lambda").fillna(method="ffill").fillna(0.0)


def realized_variance(mid: pd.Series, window: int = 100) -> pd.Series:
    r = np.log(mid).diff()
    return (r ** 2).rolling(window, min_periods=5).sum().rename("rv")


def variance_ratio(mid: pd.Series, q: int = 4, window: int = 200) -> pd.Series:
    """Lo-MacKinlay variance ratio: VR(q) = Var(r_q)/(q·Var(r_1)).
    VR > 1 ⇒ trend; VR < 1 ⇒ mean reversion.
    """
    r1 = np.log(mid).diff()
    rq = np.log(mid).diff(q)
    v1 = r1.rolling(window, min_periods=10).var()
    vq = rq.rolling(window, min_periods=10).var()
    return (vq / (q * v1)).rename("vr")
