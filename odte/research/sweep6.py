"""Sweep 6 (docs/sweep_plan.md, "Sweep 6"): T Treasury auction cycle, I S&P 500 changes, J January effect.

    python3 -m odte.research.sweep6        # -> reports/sweep6/sweep6.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.sweep3 import p_two, tstat

ROOT = Path(__file__).resolve().parents[2]
X = ROOT / "data" / "xasset"
OUT = ROOT / "reports" / "sweep6"


def judge(v: pd.Series, two_sided: bool) -> dict:
    v = v.dropna().sort_index()
    d = "as written"
    if two_sided and tstat(v) < 0:
        v, d = -v, "reversed"
    return _halves(v, d)


def judge_costed(gross: pd.Series, cost: float, two_sided: bool) -> dict:
    """Costs always subtract, whichever direction is scored."""
    g = gross.dropna().sort_index()
    f, r = g - cost, -g - cost
    if two_sided and tstat(r) > tstat(f):
        return _halves(r, "reversed")
    return _halves(f, "as written")


def _halves(v: pd.Series, d: str) -> dict:
    t = tstat(v)
    mid = v.index[0] + (v.index[-1] - v.index[0]) / 2
    a, b = v[v.index < mid], v[v.index >= mid]
    both = bool(a.mean() > 0 and b.mean() > 0)
    return {"direction": d, "n": int(len(v)), "start": str(v.index[0].date()), "end": str(v.index[-1].date()),
            "mean_pct": float(v.mean() * 100), "t": t, "p": p_two(t), "half1_t": tstat(a), "half2_t": tstat(b),
            "verdict": "PASS" if t >= 2.0 and both else ("LEAD" if p_two(t) < 0.05 and both else "FAIL")}


def part_t() -> dict:
    a = pd.read_parquet(X / "treasury_auctions.parquet")
    a["bid_to_cover_ratio"] = pd.to_numeric(a["bid_to_cover_ratio"], errors="coerce")
    a = a.sort_values("auction_date")
    px = pd.read_parquet(X / "treasury_etf.parquet")
    px.index = pd.DatetimeIndex(px.index)
    px = px[px.index <= pd.Timestamp("2026-10-31")]
    dates = px.index
    out, cost = {}, 0.0004                                       # 2 bps per side
    btc_rows = []
    for term, etf, k1, k2 in (("10-Year", "IEF", "T1", "T2"), ("30-Year", "TLT", "T3", "T4")):
        z = a[(a["original_security_term"] == term) & (a["auction_date"] >= pd.Timestamp("2003-01-01"))].copy()
        z["prev6"] = z["bid_to_cover_ratio"].shift(1).rolling(6, min_periods=6).mean()
        p = px[etf]
        post, pre = {}, {}
        for r in z.itertuples():
            i = int(np.searchsorted(dates.values, np.datetime64(r.auction_date)))
            if i >= len(dates) or dates[i] != r.auction_date or i - 5 < 0 or i + 3 >= len(dates):
                continue
            post[r.auction_date] = p.iloc[i + 3] / p.iloc[i] - 1
            pre[r.auction_date] = p.iloc[i - 1] / p.iloc[i - 5] - 1
            if np.isfinite(r.prev6) and np.isfinite(r.bid_to_cover_ratio):
                sgn = 1.0 if r.bid_to_cover_ratio >= r.prev6 else -1.0
                btc_rows.append((r.auction_date, sgn * (p.iloc[i + 3] / p.iloc[i] - 1)))
        out[f"{k1} after the {term} auction, long {etf} 3 days"] = judge(pd.Series(post) - cost, False)
        out[f"{k2} before the {term} auction, short {etf} days -5..-1"] = judge(-pd.Series(pre) - cost, False)
    g = pd.Series([v for _, v in btc_rows], index=pd.DatetimeIndex([d for d, _ in btc_rows]))
    g = g.groupby(level=0).sum()                                 # same-day 10y/30y never happens; safety only
    out["T5 weak/strong auction (bid-to-cover vs prior 6), 3 days"] = judge_costed(g, cost, True)
    return out


def part_i() -> dict:
    from odte.research.sweep3 import stock_panel
    ret, uni = stock_panel(1000, 0.5)
    dates = ret.index
    ue = sorted(uni)
    mkt = pd.Series(np.nan, index=dates)
    for a, b in zip(ue, ue[1:] + [dates[-1] + pd.Timedelta(days=1)]):
        idx = dates[(dates > a) & (dates <= b)]
        cols = [c for c in uni[a] if c in ret.columns]
        mkt[idx] = ret.loc[idx, cols].mean(axis=1)
    ch = pd.read_parquet(X / "sp500_changes.parquet")
    ch["eff"] = pd.to_datetime(ch["Effective Date_Effective Date"], errors="coerce")
    ch = ch[(ch["eff"] >= pd.Timestamp("2018-06-01")) & (ch["eff"] <= pd.Timestamp("2026-08-31"))]
    out, counts = {}, {}
    for k, colname in (("I1 added to the S&P 500", "Added_Ticker"), ("I2 deleted from the S&P 500", "Removed_Ticker")):
        vals = {}
        for r in ch.dropna(subset=[colname]).itertuples():
            sym = str(getattr(r, colname)).strip().replace("-", ".")
            if sym not in ret.columns:
                continue
            i0 = int(np.searchsorted(dates.values, np.datetime64(r.eff)))
            if i0 + 20 >= len(dates):
                continue
            rr = ret[sym].iloc[i0 + 1:i0 + 21]
            if np.isnan(ret[sym].iloc[i0]) and np.isnan(ret[sym].iloc[i0 - 1:i0 + 1]).all() or rr.notna().sum() < 15:
                continue
            ab = float(np.prod(1 + rr.fillna(0)) - np.prod(1 + mkt.iloc[i0 + 1:i0 + 21].fillna(0)))
            d0 = dates[i0]
            while d0 in vals:                                    # two changes effective the same day
                d0 = d0 + pd.Timedelta(microseconds=1)
            vals[d0] = ab
        counts[k] = len(vals)
        out[f"{k}, 20 days after, vs universe"] = judge_costed(pd.Series(vals), 0.002, True)
    out["_events"] = counts
    return out


def part_j() -> dict:
    raw = pd.read_parquet(ROOT / "data" / "indices" / "yahoo_daily.parquet")
    c = {t: g.set_index("date")["close"].sort_index() for t, g in raw.groupby("ticker") if t in ("^RUT", "^GSPC")}
    px = pd.DataFrame(c).dropna()
    px.index = pd.DatetimeIndex(px.index)
    dates = px.index
    j1, j2 = {}, {}
    for y in range(1988, 2026):
        a = dates[dates <= pd.Timestamp(f"{y}-12-15")]
        b = dates[dates <= pd.Timestamp(f"{y + 1}-01-31")]
        if len(a) == 0 or len(b) == 0 or b[-1] < pd.Timestamp(f"{y + 1}-01-25"):
            continue
        r = px.loc[b[-1]] / px.loc[a[-1]] - 1
        j1[b[-1]] = r["^RUT"] - r["^GSPC"]
        dec = dates[(dates.year == y) & (dates.month == 12)]
        jan = dates[(dates.year == y + 1) & (dates.month == 1)]
        if len(dec) >= 6 and len(jan) >= 5:
            s, e = dec[-6], jan[4]                               # close before the last 5 Dec days -> 5th Jan close
            r2 = px.loc[e] / px.loc[s] - 1
            j2[e] = r2["^RUT"] - r2["^GSPC"]
    cost = 0.0008                                                # 2 bps per side per leg
    return {"J1 small minus large, Dec 15 -> Jan 31": judge(pd.Series(j1) - cost, False),
            "J2 small minus large, last 5 Dec + first 5 Jan days": judge(pd.Series(j2) - cost, False)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {**part_t(), **part_j(), **part_i()}
    sc = {k: v for k, v in res.items() if isinstance(v, dict) and "p" in v}
    ks = sorted(sc, key=lambda k: sc[k]["p"])
    prev = 1.0
    for i, k in reversed(list(enumerate(ks, 1))):
        prev = min(prev, sc[k]["p"] * len(ks) / i)
        sc[k]["q"] = prev
    (OUT / "sweep6.json").write_text(json.dumps(res, indent=2, default=float))
    for k, e in sc.items():
        print(f"{k:58} n {e['n']:4d} {e['mean_pct']:7.3f}%/trade t {e['t']:5.2f} q {e['q']:.2f} ({e['direction']}) "
              f"halves {e['half1_t']:5.2f}/{e['half2_t']:5.2f} {e['verdict']}")
    print("S&P change events used:", res.get("_events"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
