"""Correlation + pattern-finding on price panels.

Functions:
  - pairwise_corr: symmetric Pearson matrix of daily returns.
  - rolling_corr: rolling-window correlation of two series.
  - lead_lag: max-correlation lag (cross-correlation up to N days).
  - cluster_by_corr: KMeans on returns to group tickers by co-movement.
  - anomaly_zscores: z-score of today's return vs trailing window.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def pairwise_corr(ret: pd.DataFrame) -> pd.DataFrame:
    return ret.corr(method="pearson")


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 21) -> pd.Series:
    return a.rolling(window).corr(b)


def lead_lag(a: pd.Series, b: pd.Series, max_lag: int = 5) -> Tuple[int, float]:
    """Return (best_lag, corr_at_best_lag). Positive lag = a leads b."""
    a = a.dropna()
    b = b.dropna()
    best_lag, best_c = 0, 0.0
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            x = a.iloc[: len(a) - k]
            y = b.iloc[k:]
        else:
            x = a.iloc[-k:]
            y = b.iloc[: len(b) + k]
        n = min(len(x), len(y))
        if n < 10:
            continue
        c = np.corrcoef(x.values[:n], y.values[:n])[0, 1]
        if np.isnan(c):
            continue
        if abs(c) > abs(best_c):
            best_c, best_lag = float(c), k
    return best_lag, best_c


def cluster_by_corr(ret: pd.DataFrame, k: int = 4) -> Dict[str, int]:
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise RuntimeError("scikit-learn required: pip install scikit-learn") from e
    r = ret.dropna(axis=1, how="any")
    # drop zero-variance columns so StandardScaler doesn't divide by zero
    variances = r.var()
    r = r.loc[:, variances > 1e-12]
    if r.shape[1] < k:
        return {t: 0 for t in r.columns}
    X = StandardScaler().fit_transform(r.T.values)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    return dict(zip(r.columns, map(int, labels)))


def anomaly_zscores(ret: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    mu = ret.rolling(window).mean()
    sd = ret.rolling(window).std()
    z = (ret - mu) / sd
    return z


def top_anomalies(ret: pd.DataFrame, window: int = 63, n: int = 15) -> pd.DataFrame:
    z = anomaly_zscores(ret, window)
    last = z.iloc[-1].dropna().abs().sort_values(ascending=False)
    return pd.DataFrame({
        "ticker": last.index[:n],
        "z_score": last.values[:n],
        "return": [ret.iloc[-1][t] for t in last.index[:n]],
    })


def correlated_pairs(ret: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    c = pairwise_corr(ret)
    out = []
    cols = c.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = c.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                out.append((a, b, float(v)))
    out.sort(key=lambda x: -abs(x[2]))
    return out
