"""Turn a per-ticker-day signal into a dollar-neutral book, net of costs.

Accuracy is not alpha. A model can be right 53% of the time and still lose
money, because being right on small moves and wrong on large ones nets to a
loss, and because rebalancing a cross-sectional book every day costs real
money. This module exists to make that gap measurable rather than assumed.

Construction:
  - each day, rank the cross-section by signal, go long the top and short the
    bottom, dollar-neutral, gross exposure 1 (0.5 long, 0.5 short);
  - hold one day, close at the next close, rebalance;
  - charge `cost_bps` on every unit of turnover, where turnover is the
    absolute weight change name by name. A position fully exited and replaced
    costs twice: once out, once in.

Dollar-neutrality is not decoration. A long-only version of this signal would
mostly measure whether the market rose during the test window, which says
nothing about the signal. Neutralizing removes that and leaves the part of the
return attributable to picking names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import day_block_bootstrap


def cross_sectional_weights(signal: np.ndarray, scheme: str = "rank",
                            top_frac: float = 0.2) -> np.ndarray:
    """Weights for one day's cross-section, summing to 0 with gross 1.

    `rank`  demeaned rank, scaled to gross 1 -- uses the whole cross-section.
    `topk`  equal weight on the extreme `top_frac` at each end, nothing in the
            middle -- concentrated, higher turnover, closer to how a discrete
            signal is traded in practice.
    """
    n = len(signal)
    if n < 2:
        return np.zeros(n)

    if scheme == "topk":
        k = max(1, int(round(top_frac * n)))
        if 2 * k > n:
            k = n // 2
        if k == 0:
            return np.zeros(n)
        order = np.argsort(signal, kind="mergesort")
        w = np.zeros(n)
        w[order[-k:]] = 0.5 / k
        w[order[:k]] = -0.5 / k
        return w

    ranks = pd.Series(signal).rank(method="average").to_numpy()
    centered = ranks - ranks.mean()
    gross = np.abs(centered).sum()
    return centered / gross if gross > 0 else np.zeros(n)


def run_backtest(df: pd.DataFrame, signal_col: str = "signal",
                 ret_col: str = "fwd_ret", scheme: str = "rank",
                 top_frac: float = 0.2, cost_bps: float = 0.0,
                 min_names: int = 10) -> dict:
    """Daily long/short backtest. `df` needs date, ticker, signal, fwd_ret.

    Days with fewer than `min_names` tradeable names are skipped -- a
    "cross-sectional" book over three stocks is a coin flip, and including
    those days flatters the variance estimate.
    """
    need = {"date", "ticker", signal_col, ret_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"backtest input missing columns: {sorted(missing)}")

    data = df.loc[:, ["date", "ticker", signal_col, ret_col]].dropna()
    data = data.sort_values(["date", "ticker"])

    prev_w: dict[str, float] = {}
    rows = []
    n_skipped = 0

    for date, g in data.groupby("date", sort=True):
        if len(g) < min_names:
            n_skipped += 1
            continue
        w = cross_sectional_weights(g[signal_col].to_numpy(), scheme=scheme,
                                    top_frac=top_frac)
        tickers = g["ticker"].to_numpy()
        rets = g[ret_col].to_numpy()

        gross_ret = float(np.dot(w, rets))
        cur = dict(zip(tickers, w))
        names = set(cur) | set(prev_w)
        turnover = float(sum(abs(cur.get(t, 0.0) - prev_w.get(t, 0.0))
                             for t in names))
        cost = turnover * cost_bps / 1e4
        rows.append({"date": date, "gross_ret": gross_ret,
                     "turnover": turnover, "cost": cost,
                     "net_ret": gross_ret - cost, "n_names": int(len(g))})
        prev_w = cur

    if not rows:
        return {"n_days": 0, "note": "no tradeable days"}

    daily = pd.DataFrame(rows)
    net = daily["net_ret"].to_numpy()
    gross = daily["gross_ret"].to_numpy()

    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())

    boot = day_block_bootstrap(net)
    return {
        "n_days": int(len(daily)),
        "n_days_skipped": int(n_skipped),
        "mean_daily_gross": float(gross.mean()),
        "mean_daily_net": float(net.mean()),
        "ann_return_net": float(net.mean() * 252),
        "ann_vol": float(net.std(ddof=1) * np.sqrt(252)),
        "sharpe_gross": _sharpe(gross),
        "sharpe_net": _sharpe(net),
        "sharpe_net_ci": [boot["ci_low"], boot["ci_high"]],
        "p_sharpe_le_0": boot["p_sharpe_le_0"],
        "hit_rate_days": float((net > 0).mean()),
        "avg_turnover": float(daily["turnover"].mean()),
        "total_cost_drag_ann": float(daily["cost"].mean() * 252),
        "max_drawdown": max_dd,
        "cost_bps": float(cost_bps),
        "scheme": scheme,
        "_daily": daily,
    }


def _sharpe(r: np.ndarray, periods_per_year: int = 252) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def breakeven_cost_bps(df: pd.DataFrame, signal_col: str = "signal",
                       scheme: str = "rank", hi: float = 200.0) -> float:
    """Cost level at which the strategy's net Sharpe hits zero.

    More informative than a pass/fail at one assumed cost: it says how much
    execution quality the signal can tolerate before it stops being a
    strategy. Large-cap US equities round-trip somewhere near 5-20bp all-in,
    so a breakeven below that range means the edge is not tradeable.
    """
    base = run_backtest(df, signal_col=signal_col, scheme=scheme, cost_bps=0.0)
    if base.get("n_days", 0) == 0 or base["mean_daily_gross"] <= 0:
        return 0.0
    daily = base["_daily"]
    turnover = daily["turnover"].mean()
    if turnover <= 0:
        return float(hi)
    # Net mean return is linear in cost, so breakeven is closed-form.
    return float(min(hi, base["mean_daily_gross"] / turnover * 1e4))
