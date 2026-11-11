"""H1 "Lazy Prices" test (docs/filings_plan.md). Pre-registered; run once.

  python3 -m filings.lazy prices     # DBN -> compact daily closes parquet (needs the live venv's databento)
  python3 -m filings.lazy run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "filings" / "tenk_features.parquet"
DBN = ROOT / "data" / "prices" / "xnas_ohlcv1d.dbn.zst"
SYMB = ROOT / "data" / "prices" / "xnas_symbology.parquet"
DAILY = ROOT / "data" / "prices" / "xnas_daily.parquet"
OUT = ROOT / "reports" / "filings"
FIRST, LAST = pd.Timestamp("2019-08-31"), pd.Timestamp("2026-07-31")   # formation month-ends
SPLIT = pd.Timestamp("2023-01-01")
SPLIT_RATIOS = (2, 3, 4, 5, 10, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10)
MIN_PRICE, MIN_DV = 5.0, 1e6
COST = 0.0020


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_daily() -> pd.DataFrame:
    import databento as db
    df = db.DBNStore.from_file(DBN).to_df(price_type="float", pretty_ts=True, map_symbols=False)
    df = df.reset_index()[["ts_event", "instrument_id", "close", "volume"]]
    sym = pd.read_parquet(SYMB)[["instrument_id", "symbol"]].drop_duplicates("instrument_id")
    df = df.merge(sym, on="instrument_id", how="inner")
    df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
    out = df[["date", "symbol", "close", "volume"]].sort_values(["symbol", "date"])
    out.to_parquet(DAILY, index=False)
    return out


def split_adjusted_returns(px: pd.Series) -> pd.Series:
    """Daily close-to-close returns; split-like jumps (within 2% of a split ratio) become 0."""
    r = px / px.shift(1)
    split = pd.Series(False, index=px.index)
    for k in SPLIT_RATIOS:
        split |= (r / (1 / k) - 1).abs() < 0.02          # price falls 1/k on a k-for-1 split
    return (r - 1).where(~split, 0.0)


def similarities(f: pd.DataFrame) -> pd.DataFrame:
    """Each 10-K vs the same company's previous 10-K filed 9-15 months earlier: Jaccard via MinHash."""
    f = f[f["minhash"].notna()].sort_values(["cik", "filed"])
    rows = []
    for cik, g in f.groupby("cik"):
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                gap = (r["filed"] - prev["filed"]).days
                if 270 <= gap <= 460:
                    a = np.frombuffer(r["minhash"], np.uint32)
                    b = np.frombuffer(prev["minhash"], np.uint32)
                    rows.append({"cik": cik, "filed": r["filed"], "ticker": r.get("ticker"),
                                 "sim": float(np.mean(a == b))})
            prev = r
    return pd.DataFrame(rows)


def monthly_panel(daily: pd.DataFrame, month_ends: pd.DatetimeIndex) -> tuple:
    """(next-month return, close at t, 21-day median dollar volume at t) per (t, symbol).
    A stock that stops trading earns its last price (cash) for the rest of the month."""
    rets, closes, dvs = {}, {}, {}
    for sym, g in daily.groupby("symbol"):
        g = g.set_index("date").sort_index()
        r = split_adjusted_returns(g["close"])
        wealth = (1 + r.fillna(0)).cumprod()
        dv = (g["close"] * g["volume"]).rolling(21, min_periods=15).median()
        w_at = wealth.reindex(month_ends, method="ffill")
        last_day = g.index.max()
        for i, t in enumerate(month_ends[:-1]):
            if t < g.index.min() or t > last_day:
                continue
            t1 = month_ends[i + 1]
            w0 = w_at.iloc[i]
            w1 = wealth[wealth.index <= t1].iloc[-1]          # last price if delisted mid-month
            rets[(t, sym)] = w1 / w0 - 1
            closes[(t, sym)] = g["close"][g.index <= t].iloc[-1]
            dvs[(t, sym)] = dv[dv.index <= t].iloc[-1] if (dv.index <= t).any() else np.nan
    idx = pd.MultiIndex.from_tuples(list(rets), names=["t", "symbol"])
    return pd.DataFrame({"ret": list(rets.values()), "close": [closes[k] for k in rets],
                         "dv": [dvs[k] for k in rets]}, index=idx)


# ---------------------------------------------------------------------------
# portfolios and stats
# ---------------------------------------------------------------------------

def newey_west_t(x: np.ndarray, lags: int = 3) -> float:
    x = np.asarray(x, float)
    n, mu = len(x), x.mean()
    e = x - mu
    s = e @ e / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return float(mu / np.sqrt(s / n)) if s > 0 else float("nan")


def portfolios(sims: pd.DataFrame, panel: pd.DataFrame, month_ends) -> pd.DataFrame:
    rows, prev_w = [], {"L": {}, "S": {}, "U": {}}
    for t in month_ends[:-1]:
        if t < FIRST or t > LAST:
            continue
        fresh = sims[(sims["filed"] < t) & (sims["filed"] >= t - pd.DateOffset(months=3))]
        fresh = fresh[fresh["ticker"].notna()].sort_values("filed").drop_duplicates("ticker", keep="last")
        if t not in panel.index.get_level_values(0):
            continue
        p = panel.loc[t]
        u = fresh.merge(p, left_on="ticker", right_index=True, how="inner")
        u = u[(u["close"] >= MIN_PRICE) & (u["dv"] >= MIN_DV) & u["ret"].notna()]
        if len(u) < 50:
            continue
        u["q"] = pd.qcut(u["sim"].rank(method="first"), 5, labels=False) + 1
        legs = {"L": u[u["q"] == 5], "S": u[u["q"] == 1], "U": u}
        rec = {"t": t, "n": len(u)}
        for k, g in legs.items():
            w = {s: 1 / len(g) for s in g["ticker"]}
            turn = sum(abs(w.get(s, 0) - prev_w[k].get(s, 0)) for s in set(w) | set(prev_w[k]))
            rec[f"r_{k}"] = float(g["ret"].mean())
            rec[f"cost_{k}"] = COST * turn
            prev_w[k] = w
        for q in range(1, 6):
            rec[f"q{q}"] = float(u.loc[u["q"] == q, "ret"].mean())
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("t")
    out["ls"] = (out["r_L"] - out["cost_L"]) - (out["r_S"] + out["cost_S"])
    out["long_only"] = (out["r_L"] - out["cost_L"]) - out["r_U"]
    return out


def run() -> dict:
    f = pd.read_parquet(FEAT)
    sims = similarities(f)
    daily = pd.read_parquet(DAILY)
    me = pd.DatetimeIndex(sorted(pd.date_range("2019-07-31", "2026-08-31", freq="ME")))
    panel = monthly_panel(daily[daily["symbol"].isin(set(sims["ticker"].dropna()))], me)
    port = portfolios(sims, panel, me)
    ls = port["ls"].to_numpy()
    h1, h2 = port[port.index < SPLIT]["ls"], port[port.index >= SPLIT]["ls"]
    res = {"months": int(len(port)), "avg_universe": float(port["n"].mean()),
           "ls_mean_monthly": float(ls.mean()), "ls_nw_t": newey_west_t(ls), "ls_ann": float(ls.mean() * 12),
           "ls_first_half_mean": float(h1.mean()), "ls_second_half_mean": float(h2.mean()),
           "long_only_mean_monthly": float(port["long_only"].mean()),
           "long_only_nw_t": newey_west_t(port["long_only"].to_numpy()),
           "quintile_means": {f"q{q}": float(port[f"q{q}"].mean()) for q in range(1, 6)},
           "avg_cost_ls_monthly": float((port["cost_L"] + port["cost_S"]).mean()),
           "similarities": int(len(sims)), "with_ticker": int(sims["ticker"].notna().sum())}
    res["passes"] = bool(res["ls_mean_monthly"] > 0 and res["ls_nw_t"] >= 2.0
                         and res["ls_first_half_mean"] > 0 and res["ls_second_half_mean"] > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    port.to_csv(OUT / "lazy_prices_monthly.csv")
    (OUT / "lazy_prices.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "prices":
        d = build_daily()
        print(f"{len(d):,} rows, {d['symbol'].nunique():,} symbols, {d['date'].min().date()} .. {d['date'].max().date()}")
    else:
        print(json.dumps(run(), indent=2, default=float))
