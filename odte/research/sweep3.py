"""Sweep 3 (docs/sweep_plan.md, "Sweep 3"): P partner B3, X country/sector/currency cross-sections,
E earnings-announcement ideas.

    python3 -m odte.research.sweep3 P          # -> reports/sweep3/P.json
    python3 -m odte.research.sweep3 fetchX     # Yahoo ETFs + FRED currencies -> data/xasset/
    python3 -m odte.research.sweep3 X          # -> reports/sweep3/X.json
    python3 -m odte.research.sweep3 E          # -> reports/sweep3/E.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "sweep3"
XDIR = ROOT / "data" / "xasset"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
SEED, MES = 20261115, 5.0
SCALE_END = pd.Timestamp("2021-12-31")


def tstat(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or v.std(ddof=1) == 0:
        return float("nan")
    return float(v.mean() / v.std(ddof=1) * math.sqrt(len(v)))


def p_two(t: float) -> float:
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))) if np.isfinite(t) else 1.0


def verdict(v: pd.Series) -> dict:
    """PASS: t >= 2 and both halves positive; LEAD: p < 0.05 and both halves positive."""
    v = v.dropna()
    mid = v.index[0] + (v.index[-1] - v.index[0]) / 2
    a, b = v[v.index < mid], v[v.index >= mid]
    t = tstat(v)
    both = bool(a.mean() > 0 and b.mean() > 0)
    return {"n": int(len(v)), "start": v.index[0].date().isoformat(), "end": v.index[-1].date().isoformat(),
            "mean": float(v.mean()), "t": t, "p": p_two(t), "half1_mean": float(a.mean()), "half2_mean": float(b.mean()),
            "half1_t": tstat(a), "half2_t": tstat(b),
            "verdict": "PASS" if t >= 2.0 and both else ("LEAD" if p_two(t) < 0.05 and t > 0 and both else "FAIL")}


# ----------------------------------------------------------------------------------------------- P partner
def part_p() -> int:
    from odte.research.partner_b import marks
    from odte.research.partner_t import book
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    from odte.research.sweep1 import SP, fam_B
    OUT.mkdir(parents=True, exist_ok=True)
    sp = SP()
    ideas = {i["id"]: i for i in fam_B(sp) if i["id"] in ("B3", "B7")}
    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    early = b0.index <= SCALE_END
    m = marks(sp.dt)
    ser = {"B": book(b0.index, m["day"], m["net"] * MES)}
    for k in ("B3", "B7"):
        net = ideas[k]["g"] - ideas[k]["c"]
        ser[k] = book(b0.index, pd.Series(net.index, index=net.index), net)
    w = {k: float(b0[early].std(ddof=1) / v[early].std(ddof=1)) for k, v in ser.items()}

    def p_loss(x):
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        return float((yr.sum(axis=1) < 0).mean())

    def mdd(x):
        e = x.cumsum()
        return float((e - e.cummax()).min())

    combos = {"B0": b0, "B0+B": b0 + w["B"] * ser["B"], "B0+B3": b0 + w["B3"] * ser["B3"],
              "B0+B+B7": b0 + w["B"] * ser["B"] + w["B7"] * ser["B7"], "B0+B3+B7": b0 + w["B3"] * ser["B3"] + w["B7"] * ser["B7"]}
    res = {"weights_mes_per_b0_unit": w, "corr_B3_B0": float(np.corrcoef(b0, ser["B3"])[0, 1]),
           "combos": {k: {"sharpe": sharpe(v), "p_losing_year": p_loss(v), "max_dd_per_b0_unit": mdd(v),
                          "mean_per_day": float(v.mean())} for k, v in combos.items()},
           "by_year": {int(y): {k: round(float(v[v.index.year == y].sum()), 1) for k, v in combos.items()}
                       for y in sorted(set(b0.index.year))}}
    c = res["combos"]
    res["P1_replace_B_with_B3"] = bool(c["B0+B3"]["sharpe"] > c["B0+B"]["sharpe"]
                                        and c["B0+B3"]["p_losing_year"] < c["B0+B"]["p_losing_year"])
    (OUT / "P.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------------------- X other markets
COUNTRY = ["EWJ", "EWG", "EWU", "EWQ", "EWH", "EWY", "EWT", "EWA", "EWC", "EWZ", "EWS", "EWM", "EWI", "EWP", "EWL",
           "EWN", "EWD", "EWW", "EWK", "EWO", "FXI", "EZA", "TUR", "INDA", "SPY"]
SECTOR = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
FX = {"AUD": ("DEXUSAL", False, "IR3TIB01AUM156N"), "CAD": ("DEXCAUS", True, "IR3TIB01CAM156N"),
      "JPY": ("DEXJPUS", True, "IR3TIB01JPM156N"), "GBP": ("DEXUSUK", False, "IR3TIB01GBM156N"),
      "CHF": ("DEXSZUS", True, "IR3TIB01CHM156N"), "EUR": ("DEXUSEU", False, "IR3TIB01EZM156N"),
      "NZD": ("DEXUSNZ", False, "IR3TIB01NZM156N"), "NOK": ("DEXNOUS", True, "IR3TIB01NOM156N"),
      "SEK": ("DEXSDUS", True, "IR3TIB01SEM156N")}          # (spot series, quoted as foreign per USD?, 3m rate)
US_RATE = "IR3TIB01USM156N"


def fred(series: str) -> pd.Series:
    import io
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = urllib.request.urlopen(url, timeout=60).read()      # FRED stalls on some custom User-Agents
    d = pd.read_csv(io.BytesIO(raw))
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"])
    return pd.to_numeric(d.set_index("date")["v"], errors="coerce").dropna()


def fetch_x() -> int:
    import yfinance as yf
    XDIR.mkdir(parents=True, exist_ok=True)
    px = {}
    for t in ([] if (XDIR / "etf_close.parquet").exists() else COUNTRY + SECTOR):
        h = yf.Ticker(t).history(period="max", auto_adjust=True)
        if h.empty:
            print(t, "EMPTY")
            continue
        h.index = pd.DatetimeIndex(h.index.date)
        px[t] = h["Close"]
        print(t, h.index.min().date(), len(h))
    if px:
        pd.DataFrame(px).to_parquet(XDIR / "etf_close.parquet")
    fx = {}
    for c, (s, inv, r) in FX.items():
        for name in (s, r):
            try:
                fx[name] = fred(name)
                print(name, fx[name].index.min().date(), fx[name].index.max().date(), len(fx[name]))
            except Exception as e:                               # noqa: BLE001
                print(name, "FAILED", type(e).__name__)
    fx[US_RATE] = fred(US_RATE)
    print(US_RATE, fx[US_RATE].index.min().date(), fx[US_RATE].index.max().date())
    pd.concat(fx, axis=1).to_parquet(XDIR / "fred_fx.parquet")
    return 0


X_END = pd.Period("2026-10", freq="M")


def xs_book(signal: pd.DataFrame, ret: pd.DataFrame, k_of, cost: float, high_long: bool) -> pd.Series:
    """Cross-sectional long-short: at each month-end m-1, rank eligible assets on signal[m-1]; hold over month m.
    k_of(n) = names per side. high_long: the highest signal goes long. Net monthly return."""
    out, prev_w = {}, pd.Series(dtype=float)
    for m in ret.index[1:]:
        s = signal.loc[m - 1].dropna()
        r = ret.loc[m]
        s = s[s.index.isin(r.dropna().index)]
        if len(s) < 6:
            prev_w = pd.Series(dtype=float)
            continue
        k = k_of(len(s))
        o = s.sort_values(ascending=not high_long)
        longs, shorts = o.index[:k], o.index[-k:]
        w = pd.Series(0.0, index=s.index)
        w[longs], w[shorts] = 1.0 / k, -1.0 / k
        allk = w.index.union(prev_w.index)
        turn = float((w.reindex(allk, fill_value=0) - prev_w.reindex(allk, fill_value=0)).abs().sum())
        out[m] = float((w * r[w.index]).sum()) - turn * cost
        prev_w = w * (1 + r[w.index])                            # drift to month end
        prev_w = prev_w / max(prev_w.abs().sum() / 2, 1e-12)      # back to unit gross per side
    return pd.Series(out, dtype=float)


def part_x() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(XDIR / "etf_close.parquet")
    px.index = pd.DatetimeIndex(px.index)
    me = px.groupby(px.index.to_period("M")).last()
    me = me[me.index <= X_END]
    ret = me.pct_change(fill_method=None)
    hist = me.notna().cumsum() >= 13                             # 13 month-ends of history
    mom = (me.shift(1) / me.shift(12) - 1).where(hist)
    rev = ret.where(hist)
    third = lambda n: n // 3                                    # noqa: E731
    three = lambda n: 3                                         # noqa: E731
    C, S = [c for c in COUNTRY if c in me], [c for c in SECTOR if c in me]
    ser = {"X1 country 12-1 momentum": xs_book(mom[C], ret[C], third, 0.001, True),
           "X2 country 1-month reversal": xs_book(rev[C], ret[C], third, 0.001, False),
           "X3 sector 12-1 momentum": xs_book(mom[S], ret[S], three, 0.001, True),
           "X4 sector 1-month reversal": xs_book(rev[S], ret[S], three, 0.001, False)}

    f = pd.read_parquet(XDIR / "fred_fx.parquet")
    f.index = pd.DatetimeIndex(f.index)
    spot, rate = {}, {}
    for c, (s, inv, r) in FX.items():
        d = f[s].dropna()
        d = (1.0 / d) if inv else d                             # USD per unit of foreign currency
        spot[c] = d.groupby(d.index.to_period("M")).last()
        rr = f[r].dropna()
        rate[c] = rr.groupby(rr.index.to_period("M")).last()
    us = f[US_RATE].dropna()
    us = us.groupby(us.index.to_period("M")).last()
    months = pd.period_range("1999-01", X_END, freq="M")
    S_ = pd.DataFrame(spot).reindex(months)
    R_ = pd.DataFrame(rate).reindex(months).ffill(limit=12)
    U_ = us.reindex(months).ffill(limit=12)
    R_ = R_.where(pd.DataFrame(rate).reindex(months).notna().cumsum() > 0)   # no back-fill before a series starts
    diff = R_.sub(U_, axis=0)                                    # % a year, month m
    xr = S_.pct_change(fill_method=None) + diff.shift(1) / 1200.0  # excess return over month m, accrual at m-1 rates
    xr = xr.where(diff.shift(1).notna())
    cum = (1 + xr.fillna(0)).cumprod().where(xr.notna().cumsum() > 0)
    fxmom = (cum.shift(1) / cum.shift(12) - 1).where(xr.notna().rolling(12).sum() >= 12)
    val = (S_ / S_.shift(60) - 1).where(diff.notna())
    ser["X5 currency carry"] = xs_book(diff, xr, three, 0.0005, True)
    ser["X6 currency momentum"] = xs_book(fxmom, xr, three, 0.0005, True)
    ser["X7 currency value (5-year reversal)"] = xs_book(val, xr, three, 0.0005, False)

    res = {}
    for k, v in ser.items():
        v.index = v.index.to_timestamp(how="end").normalize()
        e = verdict(v)
        e["pct_per_year"] = e["mean"] * 12 * 100
        e["sharpe"] = float(v.mean() / v.std(ddof=1) * math.sqrt(12))
        eq = (1 + v).cumprod()
        e["max_dd_pct"] = float((eq / eq.cummax() - 1).min() * 100)
        res[k] = e
        print(f"{k:38} {e['start']}..{e['end']} n {e['n']:3d}  {e['pct_per_year']:6.2f}%/yr  t {e['t']:5.2f}  "
              f"halves t {e['half1_t']:5.2f} / {e['half2_t']:5.2f}  Sharpe {e['sharpe']:.2f}  {e['verdict']}")
    (OUT / "X.json").write_text(json.dumps(res, indent=2, default=float))
    return 0


# ----------------------------------------------------------------------------------------------- E earnings
def stock_panel():
    from odte.research.sweep1 import SPLITS
    x = pd.read_parquet(ROOT / "data" / "prices" / "xnas_daily.parquet")
    x = x[(x["close"] > 0)].drop_duplicates(["date", "symbol"])
    C = x.pivot(index="date", columns="symbol", values="close").sort_index().astype("float32")
    V = x.pivot(index="date", columns="symbol", values="volume").reindex(C.index).astype("float32")
    del x
    C.index = pd.DatetimeIndex(C.index)
    V.index = C.index
    from odte.research.partner_t import nyse_days
    keep = C.index.isin(nyse_days("2018-01-01", "2026-12-31"))   # the file has ~150 sparse weekend/holiday dates
    C, V = C[keep], V[keep]
    ratio = C / C.shift(1)
    ret = ratio - 1
    split = pd.DataFrame(False, index=C.index, columns=C.columns)
    big = ret.abs() >= 0.3
    for s in SPLITS:
        split |= big & ((ratio / s - 1).abs() < 0.03)
    ret = ret.mask(split).astype("float32")
    adv20 = (C * V).rolling(20, min_periods=15).mean()
    dates = C.index
    last = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max()
    uni = {}
    for d in last:
        a = adv20.loc[d].where(C.loc[d] >= 5).dropna()
        uni[d] = set(a.sort_values(ascending=False).index[:1000])
    return ret, uni


def events(ret: pd.DataFrame, uni: dict) -> pd.DataFrame:
    e = pd.read_parquet(ROOT / "data" / "filings" / "earnings_8k.parquet")
    e["sym"] = e["symbols"].str.split(",").str[0]
    acc = pd.to_datetime(e["acceptanceDateTime"], utc=True).dt.tz_convert("America/New_York")
    dates = ret.index
    # day 0: first trading day whose 16:00 close is after acceptance
    cut = acc.dt.tz_localize(None)
    after = (cut.dt.hour >= 16)
    base = cut.dt.normalize() + pd.to_timedelta(after.astype(int), unit="D")
    i0 = np.searchsorted(dates.values, base.values.astype("datetime64[ns]"))
    e["i0"] = i0
    e = e[(e["i0"] >= 1) & (e["i0"] + 62 < len(dates))].copy()
    e["day0"] = dates[e["i0"].to_numpy()]
    e = e.sort_values(["cik", "day0"])
    gap = e.groupby("cik")["day0"].diff().dt.days
    e = e[gap.isna() | (gap > 30)]                               # first release of a cluster only
    ue = sorted(uni)
    def member(row):
        k = np.searchsorted(np.array(ue, dtype="datetime64[ns]"), np.datetime64(row.day0), side="left") - 1
        return k >= 0 and row.sym in uni[ue[k]]
    e = e[[member(r) for r in e.itertuples()]]
    return e.reset_index(drop=True)


def part_e() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ret, uni = stock_panel()
    dates = ret.index
    ue = sorted(uni)
    # equal-weight universe return each day (membership from the latest month-end before the day)
    mkt = pd.Series(np.nan, index=dates)
    for a, b in zip(ue, ue[1:] + [dates[-1] + pd.Timedelta(days=1)]):
        idx = dates[(dates > a) & (dates <= b)]
        cols = [c for c in uni[a] if c in ret.columns]
        mkt[idx] = ret.loc[idx, cols].mean(axis=1)
    ev = events(ret, uni)
    col = {s: j for j, s in enumerate(ret.columns)}
    R = ret.to_numpy(float)
    M = mkt.to_numpy(float)
    ev = ev[ev["sym"].isin(col)].copy()
    j = ev["sym"].map(col).to_numpy()
    i0 = ev["i0"].to_numpy()
    r0, r1, m0, m1 = R[i0, j], R[i0 + 1, j], M[i0], M[i0 + 1]
    ev["ar"] = (1 + r0) * (1 + r1) - (1 + m0) * (1 + m1)
    ev = ev[np.isfinite(ev["ar"])].reset_index(drop=True)
    ev["day1"] = dates[ev["i0"].to_numpy() + 1]
    # decile thresholds from the prior 91 calendar days of announcements
    d1 = ev["day1"].to_numpy()
    ar = ev["ar"].to_numpy()
    order = np.argsort(d1, kind="stable")
    side = np.zeros(len(ev))
    for k in order:
        lo = np.searchsorted(d1[order], d1[k] - np.timedelta64(91, "D"), side="left")
        hi = np.searchsorted(d1[order], d1[k], side="left")          # strictly earlier day-1 dates
        past = ar[order[lo:hi]]
        if len(past) >= 200:
            q10, q90 = np.quantile(past, [0.1, 0.9])
            side[k] = 1.0 if ar[k] >= q90 else (-1.0 if ar[k] <= q10 else 0.0)
    ev["side"] = side

    def calendar_book(entries, hold, cost_rt):
        """Long-only vs the universe: daily mean over active entries of (r - universe r), each paying cost_rt
        per round trip spread over its holding days. Days with a position only."""
        n = len(dates)
        tot, cnt = np.zeros(n), np.zeros(n)
        for s0, jj, _ in entries:
            s1 = min(s0 + hold, n)
            rr = R[s0:s1, jj] - M[s0:s1]
            tot[s0:s1] += np.where(np.isfinite(rr), rr, 0.0) - cost_rt / hold
            cnt[s0:s1] += 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return pd.Series(np.where(cnt > 0, tot / cnt, np.nan), index=dates).dropna()

    big = ev[ev["side"] != 0]
    i1 = big["i0"].to_numpy() + 1
    jj = big["sym"].map(col).to_numpy()
    sg = big["side"].astype(int).to_numpy()
    # E1: long top decile, short bottom decile, days 2..61. For a short, the leg's return is -r and pays costs
    longs = [(a + 1, b, 1) for a, b, s in zip(i1, jj, sg) if s == 1]
    shorts = [(a + 1, b, -1) for a, b, s in zip(i1, jj, sg) if s == -1]
    e1 = e1_book(R, dates, longs, shorts, hold=60, cost_rt=0.002)
    # E3: long bottom decile, days 2..6, minus universe
    e3 = calendar_book([(a + 1, b, 1) for a, b, s in zip(i1, jj, sg) if s == -1], 5, 0.002)
    # E2: each release forecasts the next one 364 days later; long days exp-5 .. exp+1 minus universe
    ent2 = []
    for r in ev.itertuples():
        exp = r.day0 + pd.Timedelta(days=364)
        ie = int(np.searchsorted(dates.values, np.datetime64(exp), side="left"))
        if ie - 5 >= 1 and ie + 2 <= len(dates):
            ent2.append((ie - 5, col[r.sym], 1))
    e2 = calendar_book(ent2, 7, 0.002)
    res = {"events": int(len(ev)), "companies": int(ev["cik"].nunique()),
           "first_day0": ev["day0"].min().date().isoformat(), "last_day0": ev["day0"].max().date().isoformat(),
           "e1_longs": len(longs), "e1_shorts": len(shorts), "e2_trades": len(ent2)}
    for k, v in {"E1 announcement-return drift (long-short, days 2-61)": e1,
                 "E2 announcement premium (days exp-5..exp+1, vs universe)": e2,
                 "E3 big losers bounce (days 2-6, vs universe)": e3}.items():
        x = verdict(v)
        x["bps_per_day"] = x["mean"] * 1e4
        x["pct_per_year_if_always_on"] = x["mean"] * 252 * 100
        res[k] = x
        print(f"{k:58} n {x['n']:4d} {x['bps_per_day']:6.2f} bps/day  t {x['t']:5.2f}  halves t "
              f"{x['half1_t']:5.2f} / {x['half2_t']:5.2f}  {x['verdict']}")
    (OUT / "E.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))
    return 0


def e1_book(R, dates, longs, shorts, hold, cost_rt):
    """Long-short calendar-time book: daily mean of active longs minus mean of active shorts; each leg pays
    cost_rt per round trip spread over its holding days. Days with both sides active only."""
    n = len(dates)
    ls, lc, ss, sc = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for book_s, book_c, entries in ((ls, lc, longs), (ss, sc, shorts)):
        for s0, jj, _ in entries:
            s1 = min(s0 + hold, n)
            rr = R[s0:s1, jj]
            book_s[s0:s1] += np.where(np.isfinite(rr), rr, 0.0)
            book_c[s0:s1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ls / lc - ss / sc - 2 * cost_rt / hold
    return pd.Series(np.where((lc > 0) & (sc > 0), v, np.nan), index=dates).dropna()


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"P": part_p, "fetchX": fetch_x, "X": part_x, "E": part_e}.get(arg, lambda: print(__doc__) or 1)())
