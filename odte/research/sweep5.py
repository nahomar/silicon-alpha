"""Sweep 5 (docs/sweep_plan.md, "Sweep 5"): Q short-sale volume cross-sections (FINRA), K crypto cross-sections
(Coinbase).

    python3 -m odte.research.sweep5 Q      # -> reports/sweep5/Q.json
    python3 -m odte.research.sweep5 K      # -> reports/sweep5/K.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.sweep3 import p_two, tstat

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "sweep5"
STABLE = {"USDT", "USDC", "DAI", "PAX", "USDP", "PYUSD", "GUSD", "BUSD", "TUSD", "UST", "EURC", "EUROC", "WBTC",
          "CBETH", "LSETH", "MSOL", "WAMPL", "EUR", "GBP", "USD"}


def xs_dates(sig: pd.DataFrame, per_ret: pd.DataFrame, k_of, cost: float, high_long: bool, min_n: int = 6) -> pd.Series:
    """Rebalance at each sig row date t_k; hold to the next row date t_{k+1}. per_ret.loc[t_{k+1}] = each asset's
    return from the close of t_k to the close of t_{k+1}. Net long-short return per period (turnover with drift)."""
    idx = list(sig.index)
    out, prev_w = {}, pd.Series(dtype=float)
    for a, b in zip(idx, idx[1:]):
        s = sig.loc[a].dropna()
        r = per_ret.loc[b]
        s = s[s.index.isin(r.dropna().index)]
        if len(s) < min_n:
            prev_w = pd.Series(dtype=float)
            continue
        k = k_of(len(s))
        o = s.sort_values(ascending=not high_long)
        w = pd.Series(0.0, index=s.index)
        w[o.index[:k]], w[o.index[-k:]] = 1.0 / k, -1.0 / k
        u = w.index.union(prev_w.index)
        turn = float((w.reindex(u, fill_value=0) - prev_w.reindex(u, fill_value=0)).abs().sum())
        out[b] = float((w * r[w.index]).sum()) - turn * cost
        dw = w * (1 + r[w.index])
        prev_w = dw / max(dw.abs().sum() / 2, 1e-12)
    return pd.Series(out, dtype=float)


def two_sided(v_fwd: pd.Series, v_rev: pd.Series) -> dict:
    tf, tr = tstat(v_fwd), tstat(v_rev)
    d, v = ("as written", v_fwd) if (tf if np.isfinite(tf) else -9) >= (tr if np.isfinite(tr) else -9) else ("reversed", v_rev)
    v = v.dropna()
    t = tstat(v)
    mid = v.index[0] + (v.index[-1] - v.index[0]) / 2
    a, b = v[v.index < mid], v[v.index >= mid]
    both = bool(a.mean() > 0 and b.mean() > 0)
    return {"direction": d, "n": int(len(v)), "start": str(v.index[0].date()), "end": str(v.index[-1].date()),
            "mean": float(v.mean()), "t": t, "p": p_two(t), "half1_t": tstat(a), "half2_t": tstat(b),
            "verdict": "PASS" if t >= 2.0 and both else ("LEAD" if p_two(t) < 0.05 and both else "FAIL")}


def score_ls(sig, per_ret, k_of, cost, high_long, per_year) -> dict:
    fwd = xs_dates(sig, per_ret, k_of, cost, high_long)
    rev = xs_dates(sig, per_ret, k_of, cost, not high_long)
    e = two_sided(fwd, rev)
    e["pct_per_year"] = e["mean"] * per_year * 100
    return e


# ----------------------------------------------------------------------------------------------- Q short volume
def part_q() -> int:
    from odte.research.sweep3 import stock_panel
    OUT.mkdir(parents=True, exist_ok=True)
    ret, uni = stock_panel()
    dates = ret.index
    sh = pd.read_parquet(ROOT / "data" / "short" / "finra_daily.parquet")
    sh = sh[sh["total"] > 0]
    sh["sr"] = sh["short"] / sh["total"]
    SR = sh.pivot_table(index="date", columns="symbol", values="sr", aggfunc="last")
    SR.index = pd.DatetimeIndex(SR.index)
    SR = SR.reindex(index=dates, columns=ret.columns)
    first = SR.dropna(how="all").index[0]
    cum = (1 + ret.fillna(0)).cumprod()
    ue = sorted(uni)

    def member_mask(ts: list) -> pd.DataFrame:
        m = pd.DataFrame(False, index=ts, columns=ret.columns)
        for t in ts:
            k = [u for u in ue if u <= t]
            if k:
                m.loc[t, list(uni[k[-1]] & set(ret.columns))] = True
        return m

    sr5 = SR.rolling(5, min_periods=3).mean()
    base = SR.shift(5).rolling(60, min_periods=40).mean()
    sr20 = SR.rolling(20, min_periods=15).mean()
    wk = pd.Series(dates, index=dates)
    fri = list(wk.groupby(dates.to_period("W-FRI")).max())
    fri = [d for d in fri if d >= first + pd.Timedelta(days=100)]
    mon = list(wk.groupby(dates.to_period("M")).max())
    mon = [d for d in mon if d >= first + pd.Timedelta(days=40)]
    res = {"first_short_day": str(first.date())}
    dec = lambda n: max(n // 10, 1)                              # noqa: E731
    for name, sig_all, ts, per_year in (("Q1 5-day short ratio (weekly)", sr5, fri, 52),
                                        ("Q2 abnormal short ratio (weekly)", sr5 - base, fri, 52),
                                        ("Q3 20-day short ratio (monthly)", sr20, mon, 12)):
        mm = member_mask(ts)
        sig = sig_all.loc[ts].where(mm)
        c = cum.loc[ts]
        per = c / c.shift(1) - 1
        res[name] = score_ls(sig, per, dec, 0.001, False, per_year)    # as written: long the LOWEST short ratio
        e = res[name]
        print(f"{name:36} {e['start']}..{e['end']} n {e['n']:4d} {e['pct_per_year']:6.2f}%/yr t {e['t']:5.2f} "
              f"({e['direction']}) halves {e['half1_t']:5.2f}/{e['half2_t']:5.2f} {e['verdict']}")
    (OUT / "Q.json").write_text(json.dumps(res, indent=2, default=float))
    return 0


# ----------------------------------------------------------------------------------------------- K crypto
def part_k() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(ROOT / "data" / "xasset" / "coinbase_daily.parquet")
    d = d[~d["base"].isin(STABLE)]
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    C = d.pivot_table(index="date", columns="product", values="close", aggfunc="last").sort_index()
    V = d.pivot_table(index="date", columns="product", values="volume", aggfunc="last").reindex(C.index)
    C = C[(C.index >= pd.Timestamp("2019-01-01")) & (C.index <= pd.Timestamp("2026-11-06"))]
    V = V.reindex(C.index)
    full = pd.date_range(C.index[0], C.index[-1], freq="D")
    C, V = C.reindex(full), V.reindex(full)
    r = C.pct_change(fill_method=None)
    r = r.where(r.abs() < 5)                                     # a >500% day is a data error
    dv = (C * V).rolling(30, min_periods=20).mean()
    age = C.notna().cumsum()
    sundays = [t for t in full if t.dayofweek == 6 and t >= full[0] + pd.Timedelta(days=60)]
    elig = {}
    for t in sundays:
        a = dv.loc[t].where(age.loc[t] >= 60).dropna()
        elig[t] = list(a.sort_values(ascending=False).index[:20])
    mask = pd.DataFrame(False, index=sundays, columns=C.columns)
    for t, cs in elig.items():
        mask.loc[t, cs] = True
    cum = (1 + r.fillna(0)).cumprod()
    per = cum.loc[sundays] / cum.loc[sundays].shift(1) - 1
    quarter = lambda n: max(n // 4, 1)                           # noqa: E731
    res = {"products": int(C.shape[1]), "start": str(full[0].date()), "end": str(full[-1].date())}
    sig1 = (C.loc[sundays] / C.shift(7).loc[sundays] - 1).where(mask)
    sig2 = (C.loc[sundays] / C.shift(28).loc[sundays] - 1).where(mask)
    vol = r.rolling(30, min_periods=20).std().loc[sundays].where(mask)
    for name, sig, hl in (("K1 1-week momentum", sig1, True), ("K2 4-week momentum", sig2, True),
                          ("K3 low volatility", vol, False)):
        res[name] = score_ls(sig, per, quarter, 0.0025, hl, 52)
        res[name]["at_60bps"] = score_ls(sig, per, quarter, 0.006, hl, 52)["t"]
    # K4: BTC up on day d -> long the other coins (this week's universe) on day d+1, else short
    btc = r["BTC-USD"]
    week_of = pd.Series(pd.NaT, index=full)
    for t in sundays:
        week_of[t + pd.Timedelta(days=1):t + pd.Timedelta(days=7)] = t
    basket = pd.Series(np.nan, index=full)
    for day in full[1:]:
        w = week_of.get(day)
        if pd.isna(w):
            continue
        cs = [c for c in elig[w] if c != "BTC-USD"]
        x = r.loc[day, cs].dropna()
        if len(x) >= 5:
            basket[day] = x.mean()
    pos = np.sign(btc).shift(1).reindex(full)
    g = pos * basket
    cost4 = pos.diff().abs() * 0.0025
    v4f, v4r = (g - cost4).dropna(), (-g - cost4).dropna()
    res["K4 Bitcoin lead"] = two_sided(v4f, v4r)
    res["K4 Bitcoin lead"]["pct_per_year"] = res["K4 Bitcoin lead"]["mean"] * 365 * 100
    # K5: time-series momentum, weekly
    sig5 = np.sign(C.loc[sundays] / C.shift(20).loc[sundays] - 1).where(mask)
    out5f, out5r, prev = {}, {}, pd.Series(dtype=float)
    for a, b in zip(sundays, sundays[1:]):
        s = sig5.loc[a].dropna()
        rr = per.loc[b, s.index].dropna()
        s = s[rr.index]
        if len(s) < 6:
            prev = pd.Series(dtype=float)
            continue
        w = s / len(s)
        u = w.index.union(prev.index)
        turn = float((w.reindex(u, fill_value=0) - prev.reindex(u, fill_value=0)).abs().sum())
        gr = float((w * rr).sum())
        out5f[b], out5r[b] = gr - turn * 0.0025, -gr - turn * 0.0025
        prev = w
    res["K5 time-series momentum (weekly)"] = two_sided(pd.Series(out5f), pd.Series(out5r))
    res["K5 time-series momentum (weekly)"]["pct_per_year"] = res["K5 time-series momentum (weekly)"]["mean"] * 52 * 100
    for k, e in res.items():
        if isinstance(e, dict):
            print(f"{k:34} {e['start']}..{e['end']} n {e['n']:4d} {e['pct_per_year']:8.2f}%/yr t {e['t']:5.2f} "
                  f"({e['direction']}) halves {e['half1_t']:5.2f}/{e['half2_t']:5.2f} {e['verdict']}"
                  + (f"  t at 60bps {e['at_60bps']:.2f}" if "at_60bps" in e else ""))
    (OUT / "K.json").write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"Q": part_q, "K": part_k}.get(arg, lambda: print(__doc__) or 1)())
