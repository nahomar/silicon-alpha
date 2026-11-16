"""Sweep 2 (docs/sweep_plan.md, "Sweep 2"): the close-only family-B price ideas on data sweep 1 never saw:
the S&P 1990-2015 (H1), other US indices (H2), 12 markets abroad (H3), and an option version on the S&P (H4).

    python3 -m odte.research.sweep2 fetch     # Yahoo daily index prices -> data/indices/yahoo_daily.parquet
    python3 -m odte.research.sweep2           # -> reports/sweep2/sweep2.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "indices" / "yahoo_daily.parquet"
OUT = ROOT / "reports" / "sweep2"
US_OTHER = ["^NDX", "^RUT", "^DJI", "^MID"]
ABROAD = ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^N225", "^HSI", "^AXJO", "^GSPTSE", "^KS11", "^BSESN", "^BVSP", "^TWII"]
EXTRA = ["^SSMI", "^IBEX", "FTSEMIB.MI", "^AEX", "^OMXSPI", "^STI", "000001.SS", "^JKSE", "^KLSE", "^NZ50"]
REGIONS = {"Europe": ["^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^SSMI", "^IBEX", "FTSEMIB.MI", "^AEX", "^OMXSPI"],
           "Asia": ["^N225", "^HSI", "^KS11", "^BSESN", "^TWII", "^STI", "000001.SS", "^JKSE", "^KLSE"]}
TICKERS = ["^GSPC"] + US_OTHER + ABROAD + EXTRA
IDEAS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B16"]
DESC = {"B1": "200-day trend, long or flat", "B2": "200-day trend, long or short", "B3": "10-month rule",
        "B4": "12-month momentum", "B5": "1-day reversal", "B6": "5-day reversal", "B7": "three down closes -> long",
        "B8": "internal bar strength", "B9": "RSI(2) < 10 -> long", "B10": "252-day high -> long",
        "B16": "big-move reversal"}
COST = 1e-4                                   # per unit of position change
SWEEP_END = pd.Timestamp("2026-11-06")
H1 = (pd.Timestamp("1990-01-01"), pd.Timestamp("2015-12-31"))
H1_OLD = (pd.Timestamp("1962-01-01"), pd.Timestamp("1989-12-31"))
H2_START, H3_START = pd.Timestamp("1990-01-01"), pd.Timestamp("2000-01-01")


def fetch(only=None) -> int:
    import yfinance as yf
    DATA.parent.mkdir(parents=True, exist_ok=True)
    out = []
    if only and DATA.exists():
        old = pd.read_parquet(DATA)
        out.append(old[~old["ticker"].isin(only)])
    for t in (only or TICKERS):
        h = yf.Ticker(t).history(period="max", auto_adjust=False)
        if h.empty:
            print(t, "EMPTY")
            continue
        h.index = pd.DatetimeIndex(h.index.date)
        h = h[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        h["ticker"] = t
        out.append(h.rename_axis("date").reset_index())
        print(t, h.index.min().date(), h.index.max().date(), len(h))
    pd.concat(out).to_parquet(DATA)
    return 0


def clean(h: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    h = h.dropna(subset=["close"]).sort_index()
    h = h[h["close"] > 0]
    h = h[h["close"] != h["close"].shift(1)]                     # holiday placeholders
    r = h["close"].pct_change()
    bad = h.index[r.abs() > 0.25]
    return h.drop(bad), [d.date().isoformat() for d in bad]


def positions(h: pd.DataFrame) -> pd.DataFrame:
    """pos[d] = position held over day d (prior close -> close d), decided from closes up to d-1. Same rules as
    sweep1.fam_B, on this market's own trading days."""
    x = h["close"]
    r = x.pct_change()
    ma = x.rolling(200).mean()
    above = x > ma
    p = {"B1": above.astype(float).shift(1), "B2": (above.astype(float) * 2 - 1).where(ma.notna()).shift(1)}
    me = x.groupby([x.index.year, x.index.month]).last()
    me.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in me.index])
    faber = (me > me.rolling(10).mean()).astype(float).where(me.rolling(10).mean().notna())
    tsm = np.sign(me.pct_change(12))
    per = x.index.to_period("M") - 1
    p["B3"] = pd.Series(faber.reindex(per).to_numpy(), index=x.index)
    p["B4"] = pd.Series(tsm.reindex(per).to_numpy(), index=x.index)
    p["B5"] = (-np.sign(r)).shift(1)
    p["B6"] = (-np.sign(x.pct_change(5))).shift(1)
    p["B7"] = ((r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0)).astype(float).shift(1)
    rng = h["high"] - h["low"]
    ibs = ((x - h["low"]) / rng).where((rng > 0) & (h["high"] >= x) & (h["low"] <= x))
    p["B8"] = pd.Series(np.where(ibs < 0.2, 1.0, np.where(ibs > 0.8, -1.0, 0.0)), index=x.index).shift(1)
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    p["B9"] = (rsi < 10).astype(float).shift(1)
    p["B10"] = (x >= x.rolling(252).max()).astype(float).shift(1)
    big = r.abs() > 2 * r.rolling(20).std().shift(1)
    p["B16"] = (-np.sign(r)).where(big, 0.0).shift(1)
    return pd.DataFrame(p).fillna(0.0)


def tstat(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * math.sqrt(len(v)))


def welch(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float((a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))


def nets(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    pos = positions(h)
    r = h["close"].pct_change()
    net = pos.mul(r, axis=0) - pos.diff().abs().fillna(pos.abs()) * COST
    return net, pos, r


def score(net: pd.Series, pos: pd.Series, r: pd.Series) -> dict:
    held = pos != 0
    n = len(net)
    return {"days": int(n), "held_days": int(held.sum()), "net_bps_per_day": float(net.mean() * 1e4),
            "net_pct_per_year": float(net.mean() * 252 * 100), "t": tstat(net),
            "timing_t": welch((pos * r)[held].dropna(), r[~held].dropna())}


def window(df, a, b):
    return df[(df.index >= a) & (df.index <= b)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(DATA)
    mk, dropped, span = {}, {}, {}
    for t, g in raw.groupby("ticker"):
        h, bad = clean(g.set_index("date")[["open", "high", "low", "close"]])
        mk[t] = h
        dropped[t] = bad
        span[t] = [h.index.min().date().isoformat(), h.index.max().date().isoformat(), int(len(h))]
    res = {"data": {"span": span, "dropped_over_25pct": {k: v for k, v in dropped.items() if v}}, "ideas": {}}
    N = {t: nets(h) for t, h in mk.items()}

    pooled_p = {}
    for k in IDEAS:
        net, pos, r = N["^GSPC"]
        e = {"desc": DESC[k],
             "H1_1990_2015": score(*(window(z[k] if isinstance(z, pd.DataFrame) else z, *H1) for z in (net, pos, r))),
             "H1_1962_1989_report": score(*(window(z[k] if isinstance(z, pd.DataFrame) else z, *H1_OLD) for z in (net, pos, r)))}
        e["H2_report"] = {}
        for t in US_OTHER:
            if t in N:
                net, pos, r = N[t]
                e["H2_report"][t] = score(*(window(z[k] if isinstance(z, pd.DataFrame) else z, H2_START, SWEEP_END)
                                            for z in (net, pos, r)))
        per, cols = {}, {}
        for t in ABROAD:
            if t not in N:
                continue
            net, pos, r = N[t]
            per[t] = score(*(window(z[k] if isinstance(z, pd.DataFrame) else z, H3_START, SWEEP_END) for z in (net, pos, r)))
            cols[t] = window(net[k], H3_START, SWEEP_END)
        pool = pd.DataFrame(cols).mean(axis=1, skipna=True)
        e["H3"] = {"markets": len(per), "markets_positive": int(sum(v["net_bps_per_day"] > 0 for v in per.values())),
                   "pooled_days": int(len(pool)), "pooled_net_bps_per_day": float(pool.mean() * 1e4),
                   "pooled_net_pct_per_year": float(pool.mean() * 252 * 100), "pooled_t": tstat(pool),
                   "per_market": {t: {"t": round(v["t"], 2), "net_pct_per_year": round(v["net_pct_per_year"], 2),
                                      "held_days": v["held_days"]} for t, v in per.items()}}
        e["H3b_report"] = {}
        for t in EXTRA:
            if t in N:
                net, pos, r = N[t]
                v = score(*(window(z[k] if isinstance(z, pd.DataFrame) else z, H3_START, SWEEP_END) for z in (net, pos, r)))
                e["H3b_report"][t] = {"t": round(v["t"], 2), "net_pct_per_year": round(v["net_pct_per_year"], 2),
                                      "held_days": v["held_days"], "days": v["days"]}
                cols[t] = window(net[k], H3_START, SWEEP_END)
        e["regions_report"] = {}
        for reg, ts in REGIONS.items():
            ts = [t for t in ts if t in cols]
            pr = pd.DataFrame({t: cols[t] for t in ts}).mean(axis=1, skipna=True)
            pos_n = sum(window(N[t][0][k], H3_START, SWEEP_END).mean() > 0 for t in ts)
            e["regions_report"][reg] = {"markets": len(ts), "markets_positive": int(pos_n), "pooled_t": tstat(pr),
                                        "pooled_net_pct_per_year": float(pr.mean() * 252 * 100),
                                        "holds_up": bool(tstat(pr) >= 2.0 and pos_n > len(ts) / 2)}
        pooled_p[k] = 2 * (1 - 0.5 * (1 + math.erf(abs(e["H3"]["pooled_t"]) / math.sqrt(2))))
        h1 = e["H1_1990_2015"]["t"] >= 2.0
        h3 = e["H3"]["pooled_t"] >= 2.0 and e["H3"]["markets_positive"] >= 7
        e["verdict"] = "CONFIRMED" if h1 and h3 else ("PARTLY" if h1 or h3 else "NOT CONFIRMED")
        e["H1_pass"], e["H3_pass"] = bool(h1), bool(h3)
        res["ideas"][k] = e
    # BH q across the 11 pooled H3 tests (report only)
    ks = sorted(pooled_p, key=pooled_p.get)
    m, q, prev = len(ks), {}, 1.0
    for i, k in reversed(list(enumerate(ks, 1))):
        prev = min(prev, pooled_p[k] * m / i)
        q[k] = prev
    for k in IDEAS:
        res["ideas"][k]["H3"]["bh_q_report"] = q[k]
    res["H4_report"] = h4()
    (OUT / "sweep2.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"{'idea':5} {'H1 t':>6} {'old t':>6} {'H3 t':>6} {'+mkts':>5} {'H3 %/yr':>7}  verdict")
    for k, e in res["ideas"].items():
        print(f"{k:5} {e['H1_1990_2015']['t']:6.2f} {e['H1_1962_1989_report']['t']:6.2f} {e['H3']['pooled_t']:6.2f} "
              f"{e['H3']['markets_positive']:5d} {e['H3']['pooled_net_pct_per_year']:7.2f}  {e['verdict']}  ({e['desc']})")
    print(json.dumps({"data": res["data"], "H4_report": res["H4_report"]}, indent=1, default=float))
    return 0


def h4() -> dict:
    """Option version on the S&P 2016-01-04..2026-11-06: signal days from SPX closes (as the sweep), 0DTE credit
    spreads sold at 10:00, return per $1k worst case."""
    from odte.research.forward_leads import closes, signals
    lib = pd.read_parquet(ROOT / "data" / "research" / "library_SPX.parquet",
                          columns=["day", "hhmm", "y_put_spread_touch", "y_call_spread_touch"])
    lib["day"] = pd.to_datetime(lib["day"])
    z = lib[(lib["hhmm"] == "10:00") & (lib["day"] >= pd.Timestamp("2016-01-04")) & (lib["day"] <= SWEEP_END)]
    z = z.set_index("day").sort_index()
    z = z[~z.index.duplicated()]
    x, _, _ = closes()
    pos = signals(x).shift(1).reindex(z.index).fillna(0.0)
    out = {}
    for k, side, col in (("B7", 1, "y_put_spread_touch"), ("B9", 1, "y_put_spread_touch"),
                         ("B16", 1, "y_put_spread_touch"), ("B16", -1, "y_call_spread_touch")):
        on = pos[k] == side
        y, other = z.loc[on, col].dropna(), z.loc[~on, col].dropna()
        t = tstat(y)
        out[f"{k}{'-long' if side == 1 else '-short'} sell {col[2:-6].replace('_', ' ')}"] = {
            "trades": int(len(y)), "mean_per_1k": float(y.mean()) if len(y) else None, "t_vs_zero": t,
            "p_one_sided": 1 - 0.5 * (1 + math.erf(t / math.sqrt(2))) if np.isfinite(t) else None,
            "other_days_mean_per_1k": float(other.mean()), "welch_t_vs_other_days": welch(y, other)}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(fetch(a[1:] or None) if a[:1] == ["fetch"] else main())
