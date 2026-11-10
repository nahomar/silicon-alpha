"""Trend following, the fixed rule pre-registered in docs/trend_following.md.

  python3 -m trend.backtest fetch     # download free daily prices (yfinance), check them
  python3 -m trend.backtest run       # the one pre-registered run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "trend"
OUT = ROOT / "reports" / "trend"
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM",          # stocks
            "SHY", "IEF", "TLT", "TIP", "LQD",          # bonds
            "GLD", "SLV", "USO", "DBA", "DBB",          # commodities
            "UUP", "FXE", "FXY", "FXA",                 # currencies
            "VNQ"]                                       # real estate
RF = "^IRX"
START, END = "2003-01-01", "2026-10-30"
LAST_MONTH = "2026-10"
PRIMARY_FROM = "2013-01"
LOOKBACKS = (21, 63, 252)
VOL_COM, VOL_MIN = 60, 60
TARGET_VOL = 0.40
COST_BP = 10.0
MIN_ASSETS = 10
SEED, N_BOOT, BLOCK = 0, 10_000, 6


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def fetch() -> pd.DataFrame:
    import yfinance as yf
    DATA.mkdir(parents=True, exist_ok=True)
    raw = yf.download(UNIVERSE + [RF], start=START, end=pd.Timestamp(END) + pd.Timedelta(days=1),
                      auto_adjust=True, progress=False, threads=True)
    px = raw["Close"].copy()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[px.index <= pd.Timestamp(END)]
    px.to_parquet(DATA / "prices.parquet")
    return px


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA / "prices.parquet")


def check(px: pd.DataFrame) -> pd.DataFrame:
    """First date per ticker and every daily move above 25% (bad-data spot check)."""
    r = px[UNIVERSE].pct_change(fill_method=None)
    big = r.stack()
    big = big[big.abs() > 0.25]
    print("first dates:", {t: f"{px[t].first_valid_index():%Y-%m-%d}" for t in UNIVERSE})
    print(f"cash rate (^IRX) days: {px[RF].notna().sum()}, last {px[RF].dropna().iloc[-1]:.2f}%")
    print(f"daily moves > 25%: {len(big)}")
    if len(big):
        print(big.to_string())
    return big.to_frame("ret")


# ---------------------------------------------------------------------------
# rule
# ---------------------------------------------------------------------------

def returns(px: pd.DataFrame) -> tuple:
    """Daily total returns, cash rate (yesterday's T-bill yield) and excess returns."""
    px = px[px[UNIVERSE].notna().any(axis=1)]                # stock-market days only
    r = px[UNIVERSE].pct_change(fill_method=None)
    rf = (px[RF].ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    ex = r.sub(rf, axis=0)
    return r, rf, ex


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).max().to_numpy())


def signals(ex: pd.DataFrame, at: pd.DatetimeIndex) -> pd.DataFrame:
    """Mean of sign(1m, 3m, 12m excess return) at the given closes; NaN without 252 days."""
    cum = np.log1p(ex).cumsum()
    n_obs = ex.notna().cumsum()
    sig = []
    for k in LOOKBACKS:
        sig.append(np.sign(cum - cum.shift(k)))
    s = sum(sig) / len(sig)
    s = s.where(n_obs >= max(LOOKBACKS))
    return s.reindex(at)


def ewma_vol(ex: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(ex.pow(2).ewm(com=VOL_COM, min_periods=VOL_MIN).mean() * 252.0)


def target_weights(s: pd.DataFrame, vol: pd.DataFrame, version: str) -> pd.DataFrame:
    """Weights decided at each month-end close. A: long/short vol-targeted; B: long/flat 1/N."""
    ok = s.notna() & vol.notna() & (vol > 0)
    n = ok.sum(axis=1).replace(0, np.nan)
    if version == "A":
        w = s * (TARGET_VOL / vol)
    elif version == "B":
        w = s.clip(lower=0.0)
    else:
        raise ValueError(version)
    return w.where(ok, 0.0).div(n, axis=0).fillna(0.0)


def daily_weights(w_me: pd.DataFrame, index: pd.DatetimeIndex) -> tuple:
    """Weights decided at close t trade at close t+1, so they earn returns from day t+2.
    Returns (daily weights, the trade day for each decision)."""
    pos = index.get_indexer(w_me.index)
    W = pd.DataFrame(np.nan, index=index, columns=w_me.columns)
    trade_days = {}
    for d, p in zip(w_me.index, pos):
        if p < 0 or p + 2 >= len(index):
            continue
        W.iloc[p + 2] = w_me.loc[d].to_numpy()
        trade_days[index[p + 1]] = d
    return W.ffill().fillna(0.0), trade_days


def run(r: pd.DataFrame, rf: pd.Series, ex: pd.DataFrame, w_me: pd.DataFrame, version: str,
        cost_bp: float = COST_BP) -> pd.DataFrame:
    """Daily strategy returns: 'excess' and 'total', after costs, plus turnover."""
    W, trade_days = daily_weights(w_me, r.index)
    r0, ex0 = r.fillna(0.0), ex.fillna(0.0)
    if version == "A":                                       # futures-like: P&L on excess returns
        excess = (W * ex0).sum(axis=1)
        total = excess + rf
    else:                                                    # cash account: rest in T-bills
        total = (W * r0).sum(axis=1) + (1.0 - W.sum(axis=1)) * rf
        excess = total - rf
    turnover = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    cost = turnover * cost_bp / 1e4
    return pd.DataFrame({"excess": excess - cost, "total": total - cost, "turnover": turnover,
                         "gross_excess": excess})


def benchmarks(r: pd.DataFrame, rf: pd.Series, eligible_me: pd.DataFrame) -> pd.DataFrame:
    """SPY, equal-weight of eligible ETFs, 60/40 SPY/IEF (monthly rebalanced, no costs)."""
    ew_me = eligible_me.astype(float).div(eligible_me.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    W, _ = daily_weights(ew_me, r.index)
    r0 = r.fillna(0.0)
    out = pd.DataFrame(index=r.index)
    out["SPY"] = r0["SPY"]
    out["equal_weight"] = (W * r0).sum(axis=1) + (1.0 - W.sum(axis=1)) * rf
    out["60_40"] = 0.6 * r0["SPY"] + 0.4 * r0["IEF"]
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def monthly(x: pd.Series) -> pd.Series:
    return (1.0 + x).groupby(x.index.to_period("M")).prod() - 1.0


def sharpe(m: np.ndarray) -> float:
    m = np.asarray(m, float)
    sd = m.std(ddof=1)
    return float(m.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan")


def block_bootstrap_sharpe(m: pd.Series, block: int = BLOCK, n: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(m, float)
    T = len(x)
    rng = np.random.default_rng(seed)
    k = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    bs = x[idx]
    s = bs.mean(axis=1) / bs.std(axis=1, ddof=1) * np.sqrt(12)
    return {"sharpe": sharpe(x), "ci": [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))],
            "months": T}


def iid_mean_ci(x: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "months": int(len(x))}


def max_drawdown(total_m: pd.Series) -> float:
    w = (1.0 + total_m).cumprod()
    return float((w / w.cummax() - 1.0).min())


def describe(total_m: pd.Series, excess_m: pd.Series) -> dict:
    yrs = len(total_m) / 12.0
    return {"cagr": float((1.0 + total_m).prod() ** (1.0 / yrs) - 1.0),
            "vol": float(excess_m.std(ddof=1) * np.sqrt(12)),
            "sharpe": sharpe(excess_m), "max_drawdown": max_drawdown(total_m),
            "worst_month": float(total_m.min())}


# ---------------------------------------------------------------------------
# the one run
# ---------------------------------------------------------------------------

def main_run() -> dict:
    px = load()
    r, rf, ex = returns(px)
    me = month_ends(r.index)
    me = me[me <= pd.Timestamp(END)]
    s = signals(ex, me)
    vol = ewma_vol(ex).reindex(me)
    eligible = s.notna() & vol.notna() & (vol > 0)
    start_me = eligible.sum(axis=1)
    first = start_me[start_me >= MIN_ASSETS].index[0]
    first_month = (first + pd.offsets.MonthBegin(1)).to_period("M")      # first month fully held
    months = pd.period_range(first_month, LAST_MONTH, freq="M")
    prim = months[months >= pd.Period(PRIMARY_FROM, "M")]
    res = {"first_month": str(first_month), "last_month": LAST_MONTH, "assets_by_year": {}}
    for y in range(first.year, 2027, 4):
        d = start_me[start_me.index.year == y]
        if len(d):
            res["assets_by_year"][str(y)] = int(d.iloc[-1])
    rf_m = monthly(rf)
    strat = {}
    for v in ("A", "B"):
        w = target_weights(s, vol, v)
        d = run(r, rf, ex, w, v)
        strat[v] = {"excess": monthly(d["excess"]), "total": monthly(d["total"]),
                    "gross_excess": monthly(d["gross_excess"]),
                    "turnover_per_year": float(d["turnover"].groupby(d.index.year).sum().loc[
                        first.year + 1:2025].mean()),
                    "orders_per_rebalance": float((w.diff().abs() > 1e-9).sum(axis=1).loc[first:].mean()),
                    "gross_leverage": float(w.abs().sum(axis=1).loc[first:].mean())}
    bm = benchmarks(r, rf, eligible)
    bench = {k: {"total": monthly(bm[k]), "excess": monthly(bm[k]) - rf_m} for k in bm.columns}

    def block(name, x, period):
        t, e = x["total"].reindex(period), x["excess"].reindex(period)
        out = describe(t, e)
        out["sharpe_ci"] = block_bootstrap_sharpe(e)["ci"]
        return out

    for label, period in (("full", months), ("post_2013", prim)):
        res[label] = {}
        for v in ("A", "B"):
            res[label][f"trend_{v}"] = block(v, strat[v], period)
        for k in bench:
            res[label][k] = block(k, bench[k], period)
        res[label]["trend_A"]["gross_sharpe"] = sharpe(strat["A"]["gross_excess"].reindex(period))
    for v in ("A", "B"):
        res[f"trend_{v}_costs"] = {k: strat[v][k] for k in
                                   ("turnover_per_year", "orders_per_rebalance", "gross_leverage")}

    # G1: A's Sharpe after publication
    g1 = block_bootstrap_sharpe(strat["A"]["excess"].reindex(prim))
    res["G1"] = {**g1, "passes": bool(g1["ci"][0] > 0)}
    # G2: A in the worst 10% of SPY months (full period)
    spy_ex = bench["SPY"]["excess"].reindex(months)
    worst = spy_ex[spy_ex <= spy_ex.quantile(0.10)].index
    g2 = iid_mean_ci(strat["A"]["excess"].reindex(worst).to_numpy())
    res["G2"] = {**g2, "spy_mean_in_those_months": float(spy_ex.reindex(worst).mean()),
                 "passes": bool(g2["mean"] > 0 and g2["ci"][0] > 0)}
    res["verdict"] = ("second engine: shadow slot" if res["G1"]["passes"] and res["G2"]["passes"]
                      else "insurance only" if res["G2"]["passes"] else "closed")
    # yearly table and correlation with SPY
    yr = {k: (1 + x["total"].reindex(months)).groupby(months.year).prod() - 1
          for k, x in {"trend_A": strat["A"], "trend_B": strat["B"], **bench}.items()}
    res["by_year"] = pd.DataFrame(yr).round(4).to_dict(orient="index")
    res["corr_A_with_SPY_monthly"] = float(np.corrcoef(strat["A"]["excess"].reindex(months),
                                                       spy_ex)[0, 1])
    # fixed-fee drag for the small-account version (about $1 per ETF order)
    opy = res["trend_B_costs"]["orders_per_rebalance"] * 12
    res["B_fee_drag_per_year"] = {f"${a:,}": round(opy * 1.0 / a, 4) for a in (1_000, 5_000, 25_000, 100_000)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trend.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        check(fetch())
    elif cmd == "check":
        check(load())
    elif cmd == "run":
        res = main_run()
        print(json.dumps({k: v for k, v in res.items() if k != "by_year"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
