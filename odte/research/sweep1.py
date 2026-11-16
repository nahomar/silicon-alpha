"""Sweep 1 (docs/sweep_plan.md): ~115 fixed-rule ideas, one scoring rule, a false-discovery correction.

    python3 -m odte.research.sweep1            # -> reports/sweep1/sweep1.csv, sweep1.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research.gamma_momentum import es_bars
from odte.research.partner_o import legs
from odte.research.partner_t import CLOSURES, NYSE, book, day_table, nyse_days

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "sweep1"
FIRST, END = pd.Timestamp("2016-01-04"), pd.Timestamp("2026-11-06")
ERA = pd.Timestamp("2022-01-01")
MES, COST, HALF = 5.0, 0.75, 0.375
Q_FDR, MIN_TRADES, SEED = 0.10, 20, 20261115
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
ET = "America/New_York"


# ----------------------------------------------------------------------------------------------- S&P data
class SP:
    def __init__(self):
        self.bars = es_bars()
        b = self.bars
        self.O, self.C, self.I = (b[c].unstack("hhmm") for c in ("open", "close", "instrument_id"))
        self.cal = nyse_days("2012-01-01", "2026-12-31")
        self.days = self.cal[(self.cal >= FIRST) & (self.cal <= END)]
        self.dt = day_table(b)
        spx = pd.read_csv(ROOT / "data" / "odte_0dte" / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
        ctx = pd.read_parquet(ROOT / "data" / "context" / "daily.parquet")
        ctx.index = pd.DatetimeIndex(pd.to_datetime(ctx.index).date)
        self.ctx = ctx[~ctx.index.duplicated()].reindex(self.cal).ffill(limit=3)
        self.spxc = self.ctx["spx"]                                       # index closes for signals (2012+)
        spx = spx.reindex(self.cal).fillna(self.spxc)
        es = self.dt.reindex(self.cal)
        ok = es["close"].notna() & es["close"].shift(1).notna() & (es["inst"] == es["inst"].shift(1))
        self.mark = pd.Series(np.where(ok, es["close"] - es["close"].shift(1), spx - spx.shift(1)), index=self.cal)
        lg = legs(b)
        self.legs = lg.set_index("day")                                   # night (ends on day), dayleg

    def nxt(self, dates) -> pd.DatetimeIndex:
        """First NYSE day on or after each date."""
        i = np.searchsorted(self.cal.values, pd.DatetimeIndex(dates).values)
        i = np.minimum(i, len(self.cal) - 1)
        return self.cal[i]

    def intraday(self, d_in: str, d_out: str) -> pd.Series:
        """Points from the open of bar d_in to the close of bar d_out, same day, same contract."""
        g = self.C[d_out] - self.O[d_in]
        return g.where(self.I[d_out] == self.I[d_in])

    def bar_close(self, h: str) -> pd.Series:
        return self.C[h]


def idea(fam, iid, desc, g, c, n1, n2, index, split=ERA, one_sided=False, unit="$ per MES"):
    return {"family": fam, "id": iid, "desc": desc, "g": g.reindex(index).fillna(0.0), "c": c.reindex(index).fillna(0.0),
            "n1": int(n1), "n2": int(n2), "split": split, "one_sided": one_sided, "unit": unit}


def sp_trades(sp: SP, fam, iid, desc, gross_pts: pd.Series):
    """Trades: gross points per trade indexed by date; booked on the first NYSE day on or after it."""
    s = gross_pts.dropna()
    s = s[(s.index >= FIRST) & (s.index <= END)]
    bd = sp.nxt(s.index)
    g = pd.Series(s.to_numpy() * MES, index=bd).groupby(level=0).sum()
    c = pd.Series(COST * MES, index=bd).groupby(level=0).sum()
    return idea(fam, iid, desc, g, c, (s.index < ERA).sum(), (s.index >= ERA).sum(), sp.days)


def sp_pos(sp: SP, fam, iid, desc, pos: pd.Series, index=None, split=ERA):
    """pos[d] = position held over day d (prior close -> close d), decided from data up to d-1."""
    index = sp.days if index is None else index
    p = pos.reindex(sp.cal).fillna(0.0)
    g = p * sp.mark * MES
    c = (p - p.shift(1).fillna(0.0)).abs() * HALF * MES
    held = p.reindex(index)
    return idea(fam, iid, desc, g, c, ((held != 0) & (held.index < split)).sum(), ((held != 0) & (held.index >= split)).sum(),
                index, split)


def on_days(sp: SP, dates) -> pd.Series:
    d = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(dates)))).intersection(sp.cal)
    return pd.Series(1.0, index=d)


# ----------------------------------------------------------------------------------------------- A calendar
def fam_A(sp: SP) -> list:
    out = []
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    sess = sp.intraday("09:30", "15:59")
    sess = sess[sess.index.isin(sp.cal)]
    for k in range(5):
        out.append(sp_trades(sp, "A", f"A{k + 1}", f"session on {wd[k]}", sess[sess.index.dayofweek == k]))
    n = sp.legs["night"].dropna()
    n = n[n.index.isin(sp.cal)]
    for k in range(5):
        out.append(sp_trades(sp, "A", f"A{k + 6}", f"night ending {wd[k]}", n[n.index.dayofweek == k]))
    hol = sorted(set(NYSE().holidays("2015-01-01", "2026-12-31")) | CLOSURES)
    cal = sp.cal
    pre = [cal[np.searchsorted(cal.values, np.datetime64(h)) - 1] for h in hol]
    post = [cal[min(np.searchsorted(cal.values, np.datetime64(h)), len(cal) - 1)] for h in hol]
    out.append(sp_pos(sp, "A", "A11", "pre-holiday day", on_days(sp, pre)))
    out.append(sp_pos(sp, "A", "A12", "post-holiday day", on_days(sp, post)))
    s = pd.Series(cal, index=cal)
    santa = []
    for y in range(2015, 2026):
        dec = s[(s.index.year == y) & (s.index.month == 12)]
        jan = s[(s.index.year == y + 1) & (s.index.month == 1)]
        santa += list(dec.index[-5:]) + list(jan.index[:2])
    out.append(sp_pos(sp, "A", "A13", "Santa Claus rally", on_days(sp, santa)))
    out.append(sp_pos(sp, "A", "A14", "Halloween (Nov-Apr long)", on_days(sp, cal[cal.month.isin([11, 12, 1, 2, 3, 4])])))
    opex_wk, after_wk = [], []
    for m in pd.date_range("2015-12-01", "2026-12-01", freq="MS"):
        fr = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="W-FRI")[2]
        opex_wk += list(pd.date_range(fr - pd.Timedelta(days=4), fr))
        after_wk += list(pd.date_range(fr + pd.Timedelta(days=3), fr + pd.Timedelta(days=7)))
    out.append(sp_pos(sp, "A", "A15", "monthly opex week", on_days(sp, opex_wk)))
    out.append(sp_pos(sp, "A", "A16", "week after opex", on_days(sp, after_wk)))
    for iid, months, desc in (("A17", range(1, 13), "month-end rebalancing (against MTD)"),
                              ("A18", (3, 6, 9, 12), "quarter-end rebalancing (against QTD)")):
        pos = pd.Series(0.0, index=cal)
        grp = s.groupby([cal.year, cal.month])
        keys = list(grp.groups.keys())
        for j, key in enumerate(keys):
            if key[1] not in months or j < 3:
                continue
            md = grp.get_group(key).index
            if len(md) < 5:
                continue
            back = 1 if iid == "A17" else 3
            base_day = grp.get_group(keys[j - back]).index[-1]
            sig = sp.spxc.get(md[-4]) / sp.spxc.get(base_day) - 1.0
            if np.isfinite(sig) and sig != 0:
                pos.loc[md[-3:]] = -np.sign(sig)
        out.append(sp_pos(sp, "A", iid, desc, pos))
    ev = pd.read_parquet(ROOT / "data" / "calendar" / "events.parquet")
    ev["date"] = pd.to_datetime(ev["date"])
    fomc = pd.DatetimeIndex(sorted(set(ev.loc[ev["event"].isin(["fomc", "fomc_sep"]), "date"]))).intersection(cal)
    c1359 = sp.bar_close("13:59")
    pre_g = {}
    for f in fomc:
        i = cal.get_loc(f)
        p = cal[i - 1]
        try:
            if sp.I.at[p, "13:59"] == sp.I.at[f, "13:59"]:
                pre_g[f] = c1359[f] - c1359[p]
        except KeyError:
            pass
    out.append(sp_trades(sp, "A", "A19", "pre-FOMC drift (13:59 day before -> 13:59)", pd.Series(pre_g, dtype=float)))
    react = np.sign(sp.C["14:29"] - sp.C["13:59"]) * sp.intraday("14:30", "15:59")
    react = react.where(sp.I["14:29"] == sp.I["13:59"])
    out.append(sp_trades(sp, "A", "A20", "FOMC reaction (follow 14:00-14:30)", react[react.index.isin(fomc)]))
    cyc = []
    for f in fomc:
        i = cal.get_loc(f)
        for a, b in ((-1, 3), (9, 13), (19, 23), (29, 33)):
            cyc += list(cal[max(i + a, 0): i + b + 1])
    out.append(sp_pos(sp, "A", "A21", "FOMC cycle even weeks", on_days(sp, cyc)))
    evd = lambda e: pd.DatetimeIndex(sorted(set(ev.loc[ev["event"] == e, "date"])))
    out.append(sp_pos(sp, "A", "A22", "CPI day", on_days(sp, evd("cpi"))))
    out.append(sp_pos(sp, "A", "A23", "jobs-report day", on_days(sp, evd("nfp"))))
    out.append(sp_pos(sp, "A", "A24", "day after FOMC", on_days(sp, [cal[cal.get_loc(f) + 1] for f in fomc])))
    out.append(sp_pos(sp, "A", "A25", "triple-witching day", on_days(sp, evd("quad_witching"))))
    mc = evd("megacap_earnings")
    out.append(sp_pos(sp, "A", "A26", "day after megacap earnings",
                      on_days(sp, cal[np.minimum(np.searchsorted(cal.values, mc.values, side="right"), len(cal) - 1)])))
    return out


# ----------------------------------------------------------------------------------------------- B price patterns
def fam_B(sp: SP) -> list:
    out, x = [], sp.spxc
    r = x.pct_change()
    above = (x > x.rolling(200).mean())
    out.append(sp_pos(sp, "B", "B1", "200-day trend, long or flat", above.astype(float).shift(1)))
    out.append(sp_pos(sp, "B", "B2", "200-day trend, long or short", (above.astype(float) * 2 - 1).where(x.rolling(200).mean().notna()).shift(1)))
    me = x.groupby([x.index.year, x.index.month]).last()
    me.index = pd.PeriodIndex([pd.Period(year=y, month=m, freq="M") for y, m in me.index])
    faber = (me > me.rolling(10).mean()).astype(float).where(me.rolling(10).mean().notna())
    tsm = np.sign(me.pct_change(12))
    per = x.index.to_period("M") - 1
    out.append(sp_pos(sp, "B", "B3", "10-month rule", pd.Series(faber.reindex(per).to_numpy(), index=x.index)))
    out.append(sp_pos(sp, "B", "B4", "12-month time-series momentum", pd.Series(tsm.reindex(per).to_numpy(), index=x.index)))
    out.append(sp_pos(sp, "B", "B5", "1-day reversal", (-np.sign(r)).shift(1)))
    out.append(sp_pos(sp, "B", "B6", "5-day reversal", (-np.sign(x.pct_change(5))).shift(1)))
    out.append(sp_pos(sp, "B", "B7", "three down closes -> long", ((r < 0) & (r.shift(1) < 0) & (r.shift(2) < 0)).astype(float).shift(1)))
    hi = np.fmax(sp.O.max(axis=1), sp.C.max(axis=1))
    lo = np.fmin(sp.O.min(axis=1), sp.C.min(axis=1))
    cl = sp.dt["close"].reindex(hi.index)
    ibs = ((cl - lo) / (hi - lo)).reindex(sp.cal)
    ibs_pos = pd.Series(np.where(ibs < 0.2, 1.0, np.where(ibs > 0.8, -1.0, 0.0)), index=sp.cal)
    out.append(sp_pos(sp, "B", "B8", "internal bar strength", ibs_pos.shift(1)))
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn)
    out.append(sp_pos(sp, "B", "B9", "RSI(2) < 10 -> long", (rsi < 10).astype(float).shift(1)))
    out.append(sp_pos(sp, "B", "B10", "252-day high -> long", (x >= x.rolling(252).max()).astype(float).shift(1)))
    lg = sp.legs
    out.append(sp_trades(sp, "B", "B11", "gap fade", (-np.sign(lg["night"]) * lg["dayleg"]).where(lg["night"] != 0)))
    out.append(sp_trades(sp, "B", "B12", "opening drive (first 30 min)",
                         np.sign(sp.C["09:59"] - sp.O["09:30"]).where(sp.I["09:59"] == sp.I["09:30"]) * sp.intraday("10:00", "15:59")))
    out.append(sp_trades(sp, "B", "B13", "morning direction held afternoon",
                         np.sign(sp.C["11:59"] - sp.O["09:30"]).where(sp.I["11:59"] == sp.I["09:30"]) * sp.intraday("12:00", "15:59")))
    cols = list(sp.C.columns)
    i0, i1, ie = cols.index("09:30"), cols.index("10:00"), cols.index("15:59")
    Cv, Ov, Iv = sp.C.to_numpy(float), sp.O.to_numpy(float), sp.I.to_numpy(float)
    rh = np.fmax(np.nanmax(Cv[:, i0:i1], axis=1), np.nanmax(Ov[:, i0:i1], axis=1))
    rl = np.fmin(np.nanmin(Cv[:, i0:i1], axis=1), np.nanmin(Ov[:, i0:i1], axis=1))
    after = Cv[:, i1:ie]
    brk_up, brk_dn = after > rh[:, None], after < rl[:, None]
    anyb = (brk_up | brk_dn)
    first = np.where(anyb.any(axis=1), anyb.argmax(axis=1), -1)
    orb = np.full(len(Cv), np.nan)
    for k in np.flatnonzero(first >= 0):
        j = i1 + first[k]
        sgn = 1.0 if brk_up[k, first[k]] else -1.0
        if Iv[k, j] == Iv[k, ie] and np.isfinite(Cv[k, ie]):
            orb[k] = sgn * (Cv[k, ie] - Cv[k, j])
    out.append(sp_trades(sp, "B", "B14", "30-min opening-range breakout", pd.Series(orb, index=sp.C.index)))
    nn = lg["night"].shift(-1)
    out.append(sp_trades(sp, "B", "B15", "day direction held overnight",
                         pd.Series((np.sign(lg["dayleg"]) * nn).to_numpy(), index=lg.index).shift(1)))
    sd = r.rolling(20).std().shift(1)
    big = (r.abs() > 2 * sd)
    out.append(sp_pos(sp, "B", "B16", "big-move reversal", (-np.sign(r)).where(big, 0.0).shift(1)))
    return out


# ----------------------------------------------------------------------------------------------- C other markets
def btc_minutes() -> pd.Series:
    b = pd.read_parquet(ROOT / "data" / "alt" / "btc_1m.parquet", columns=["ts", "close"]).sort_values("ts")
    return pd.Series(b["close"].to_numpy(float), index=b["ts"].to_numpy(np.int64))


def btc_at(btc: pd.Series, when: pd.DatetimeIndex, stale_s: int = 1800) -> np.ndarray:
    """BTC price at ET timestamps (last minute at or before), NaN if older than stale_s."""
    t = when.tz_localize(ET, nonexistent="shift_forward", ambiguous="NaT").tz_convert("UTC").asi8 // 10**9
    i = np.searchsorted(btc.index.values, t, side="right") - 1
    ok = (i >= 0)
    out = np.full(len(t), np.nan)
    ii = np.where(ok, i, 0)
    good = ok & ((t - btc.index.values[ii]) <= stale_s)
    out[good] = btc.values[ii[good]]
    return out


def fam_C(sp: SP, btc: pd.Series) -> list:
    out, c = [], sp.ctx
    L1 = lambda s: s.shift(1)
    L2 = lambda s: s.shift(2)
    out.append(sp_pos(sp, "C", "C1", "VIX below VIX3M -> long", L1((c["vix"] < c["vix3m"]).astype(float))))
    out.append(sp_pos(sp, "C", "C2", "VIX9D below VIX -> long", L1((c["vix9d"] < c["vix"]).astype(float))))
    out.append(sp_pos(sp, "C", "C3", "VIX spike -> long", L1((c["vix"] > 1.2 * c["vix"].rolling(10).mean()).astype(float))))
    for iid, col in (("C4", "vix"), ("C5", "vvix"), ("C6", "skew")):
        out.append(sp_pos(sp, "C", iid, f"{col.upper()} above 1-yr median -> long",
                          L1((c[col] > c[col].rolling(252).median()).astype(float))))
    rv = sp.spxc.pct_change().rolling(21).std() * np.sqrt(252) * 100
    vrp = c["vix"] - rv
    out.append(sp_pos(sp, "C", "C7", "variance premium above median -> long", L1((vrp > vrp.rolling(252).median()).astype(float))))
    out.append(sp_pos(sp, "C", "C8", "credit (HYG 5d) leads", L1(np.sign(c["hyg"].diff(5)))))
    out.append(sp_pos(sp, "C", "C9", "dollar 5d, opposite", L2(-np.sign(c["dxy"].diff(5)))))
    out.append(sp_pos(sp, "C", "C10", "10y yield 5d, opposite", L2(-np.sign(c["tnx"].diff(5)))))
    out.append(sp_pos(sp, "C", "C11", "oil 5d", L2(np.sign(c["crude"].diff(5)))))
    out.append(sp_pos(sp, "C", "C12", "gold 5d, opposite", L2(-np.sign(c["gold"].diff(5)))))
    out.append(sp_pos(sp, "C", "C13", "Bitcoin 1d", L2(np.sign(c["btc"].diff(1)))))
    # C14: Bitcoin 16:00 (prev NYSE day) -> 09:30 ET sign, held over the S&P session
    lg = sp.legs
    days = lg.index[lg.index.isin(sp.cal)]
    prev = sp.cal[np.searchsorted(sp.cal.values, days.values) - 1]
    p0 = btc_at(btc, prev + pd.Timedelta(hours=16))
    p1 = btc_at(btc, days + pd.Timedelta(hours=9, minutes=30))
    sig = pd.Series(np.sign(p1 - p0), index=days)
    s = (sig * lg["dayleg"].reindex(days)).dropna()
    s = s[s.index >= pd.Timestamp("2022-04-21")]
    bidx = sp.days[sp.days >= pd.Timestamp("2022-04-21")]
    bsplit = pd.Timestamp("2024-07-01")
    g = s * MES
    c14 = pd.Series(COST * MES, index=s.index)
    out.append(idea("C", "C14", "Bitcoin overnight -> S&P session", g, c14, (s.index < bsplit).sum(), (s.index >= bsplit).sum(),
                    bidx, bsplit))
    v1 = c["vix1d"]
    vidx = sp.days[sp.days >= pd.Timestamp("2023-04-25")]
    out.append(sp_pos(sp, "C", "C15", "VIX1D below VIX -> long", L1((v1 < c["vix"]).astype(float).where(v1.notna())),
                      index=vidx, split=pd.Timestamp("2025-01-01")))
    return out


# ----------------------------------------------------------------------------------------------- D dealer positioning
def fam_D(sp: SP) -> list:
    lib = pd.read_parquet(ROOT / "data" / "research" / "library_SPX.parquet",
                          columns=["day", "hhmm", "gex_net", "oi_pcr_0dte", "skew", "maxoi_dist", "gex_flip_dist"])
    lib["day"] = pd.to_datetime(lib["day"])
    at = lambda h, col: lib[lib["hhmm"] == h].set_index("day")[col].sort_index()
    out = []
    g1530 = at("15:30", "gex_net")
    nd = sp.cal[np.minimum(np.searchsorted(sp.cal.values, g1530.index.values, side="right"), len(sp.cal) - 1)]
    out.append(sp_pos(sp, "D", "D1", "sign of gex 15:30 -> next day", pd.Series(np.sign(g1530.to_numpy()), index=nd)))
    s10 = sp.intraday("10:00", "15:59")
    out.append(sp_trades(sp, "D", "D2", "sign of gex 10:00 -> rest of day", np.sign(at("10:00", "gex_net")) * s10.reindex(at("10:00", "gex_net").index)))
    for iid, col, desc in (("D3", "oi_pcr_0dte", "put/call OI above median -> long"), ("D4", "skew", "put skew above median -> long")):
        v = at("10:00", col)
        med = v.rolling(252, min_periods=120).median().shift(1)
        flag = (v > med)
        out.append(sp_trades(sp, "D", iid, desc, s10.reindex(v.index).where(flag & med.notna())))
    m14 = at("14:00", "maxoi_dist")
    out.append(sp_trades(sp, "D", "D5", "toward max-OI strike 14:00", np.sign(m14) * sp.intraday("14:00", "15:59").reindex(m14.index)))
    fl = at("10:00", "gex_flip_dist")
    out.append(sp_trades(sp, "D", "D6", "above zero-gamma -> long", -np.sign(fl) * s10.reindex(fl.index)))
    return out


# ----------------------------------------------------------------------------------------------- E event-day options
EVENTS = ["cpi", "nfp", "ppi", "jolts", "claims", "ism_mfg", "ism_svc", "fomc", "fomc_minutes", "auction_long", "opex",
          "quad_witching", "vix_expiry", "month_end", "quarter_end", "pre_holiday", "post_holiday", "megacap_earnings"]


def fam_E() -> list:
    cols = ["day", "hhmm", "y_fly_touch", "y_long_straddle_touch", "megacap_earnings", "ev_fomc_sep"] + [f"ev_{e}" for e in EVENTS if e != "megacap_earnings"]
    lib = pd.read_parquet(ROOT / "data" / "research" / "library_SPX.parquet", columns=cols)
    lib["day"] = pd.to_datetime(lib["day"])
    lib = lib[(lib["day"] >= FIRST) & (lib["day"] <= END)]
    days = pd.DatetimeIndex(sorted(lib["day"].unique()))
    out = []
    for e in EVENTS:
        h = "14:30" if e in ("fomc", "fomc_minutes") else "10:00"
        flag = "megacap_earnings" if e == "megacap_earnings" else f"ev_{e}"
        on = lib[flag].fillna(0) > 0
        if e == "fomc":
            on = on | (lib["ev_fomc_sep"].fillna(0) > 0)
        z = lib[(lib["hhmm"] == h) & on].set_index("day")
        for side, col in (("sell", "y_fly_touch"), ("buy", "y_long_straddle_touch")):
            y = z[col].dropna()
            out.append(idea("E", f"E-{e}-{side}", f"{side} {'fly' if side == 'sell' else 'straddle'} at {h} on {e} days", y,
                            pd.Series(0.0, index=y.index), (y.index < ERA).sum(), (y.index >= ERA).sum(), days,
                            one_sided=True, unit="$ per $1k risk"))
    return out


# ----------------------------------------------------------------------------------------------- F US stocks
SPLITS = [1 / 2, 1 / 3, 2 / 3, 1 / 4, 1 / 5, 1 / 10, 3 / 2, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]


def fam_F(nyse_only: bool = False) -> list:
    """nyse_only=True drops the file's ~150 sparse non-NYSE dates (post-hoc fix; the scored sweep used False)."""
    x = pd.read_parquet(ROOT / "data" / "prices" / "xnas_daily.parquet")
    x = x[(x["close"] > 0)].drop_duplicates(["date", "symbol"])
    C = x.pivot(index="date", columns="symbol", values="close").sort_index().astype("float32")
    V = x.pivot(index="date", columns="symbol", values="volume").reindex(C.index).astype("float32")
    del x
    if nyse_only:
        keep = pd.DatetimeIndex(C.index).isin(nyse_days("2018-01-01", "2026-12-31"))
        C, V = C[keep], V[keep]
    ratio = C / C.shift(1)
    ret = ratio - 1
    split = pd.DataFrame(False, index=C.index, columns=C.columns)
    big = ret.abs() >= 0.3
    for s in SPLITS:
        split |= big & ((ratio / s - 1).abs() < 0.03)
    ret = ret.mask(split)
    P = (1 + ret.fillna(0)).cumprod().where(C.notna())
    DV = C * V
    adv20 = DV.rolling(20, min_periods=15).mean()
    dates = C.index
    month_last = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max().to_numpy()
    reb_m = [dates.get_loc(d) for d in month_last][:-1]
    reb_w = list(range(260, len(dates) - 1, 5))
    MR = P.groupby([dates.year, dates.month]).last().pct_change()
    MR.index = pd.MultiIndex.from_tuples(MR.index)

    def sig(name: str, i: int) -> pd.Series:
        if name == "F1":
            return P.iloc[i - 21] / P.iloc[i - 252] - 1 if i >= 252 else None
        if name == "F2":
            return P.iloc[i] / P.iloc[i - 21] - 1
        if name == "F3":
            return P.iloc[i] / P.iloc[i - 5] - 1
        if name == "F4":
            w = ret.iloc[i - 62:i + 1]
            return w.std().where(w.count() >= 50)
        if name == "F5":
            w = ret.iloc[i - 20:i + 1]
            return w.max().where(w.count() >= 15)
        if name == "F6":
            return V.iloc[i - 4:i + 1].mean() / V.iloc[i - 54:i - 4].mean()
        if name == "F7":
            return (P.iloc[i] / P.iloc[i - 251:i + 1].max()) if i >= 251 else None
        if name == "F8":
            return adv20.iloc[i]
        if name == "F9":
            nxt = dates[i] + pd.offsets.MonthBegin(1)
            past = MR[(MR.index.get_level_values(1) == nxt.month) & (MR.index.get_level_values(0) < nxt.year)]
            return past.mean().where(past.count() >= 2)
        raise KeyError(name)

    spec = {"F1": ("momentum 12-1", reb_m, +1), "F2": ("1-month reversal", reb_m, -1), "F3": ("1-week reversal", reb_w, -1),
            "F4": ("low volatility", reb_m, -1), "F5": ("MAX (lottery) effect", reb_m, -1), "F6": ("volume shock", reb_w, +1),
            "F7": ("52-week high", reb_m, +1), "F8": ("low liquidity in top 1000", reb_m, -1),
            "F9": ("same-month seasonality", reb_m, +1)}
    out = []
    idx = dates[(dates >= pd.Timestamp("2018-05-01"))]
    for iid, (desc, reb, direction) in spec.items():
        g = pd.Series(0.0, index=dates)
        cst = pd.Series(0.0, index=dates)
        held = pd.Series(0.0, index=dates)
        wl_old, ws_old = pd.Series(dtype=float), pd.Series(dtype=float)
        reb = [i for i in reb if i >= 70]
        for a, b in zip(reb, reb[1:] + [len(dates) - 1]):
            s = sig(iid, a)
            if s is None:
                continue
            univ = (C.iloc[a] >= 5) & (adv20.iloc[a].rank(ascending=False) <= 1000)
            s = s[univ & s.notna() & np.isfinite(s)]
            if len(s) < 100:
                continue
            k = len(s) // 10
            hi, lo = s.nlargest(k).index, s.nsmallest(k).index
            lg_, sh_ = (hi, lo) if direction > 0 else (lo, hi)
            rows = ret.iloc[a + 1:b + 1]
            g.iloc[a + 1:b + 1] += 10000 * (rows[lg_].mean(axis=1).fillna(0) - rows[sh_].mean(axis=1).fillna(0)).to_numpy()
            wl, ws = pd.Series(1 / k, index=lg_), pd.Series(1 / k, index=sh_)
            turn = wl.sub(wl_old, fill_value=0).abs().sum() + ws.sub(ws_old, fill_value=0).abs().sum()
            cst.iloc[a + 1] += 0.001 * 10000 * turn
            held.iloc[a + 1:b + 1] = 1
            wl_old, ws_old = wl, ws
        h = held.reindex(idx)
        out.append(idea("F", iid, desc, g, cst, ((h > 0) & (h.index < ERA)).sum(), ((h > 0) & (h.index >= ERA)).sum(),
                        idx, unit="$ per $10k a side"))
    return out


# ----------------------------------------------------------------------------------------------- G alt data, Bitcoin
def fam_G(sp: SP, btc: pd.Series) -> list:
    out = []
    files = sorted(glob.glob(str(ROOT / "data" / "africa" / "gdelt" / "events_daily" / "*.parquet")))
    cnt = {}
    for f in files:
        z = pd.read_parquet(f, columns=["EventRootCode"])
        cnt[pd.Timestamp(Path(f).stem)] = int(z["EventRootCode"].astype(str).isin(["14", "18", "19", "20"]).sum())
    cnt = pd.Series(cnt).sort_index()
    spike_days = cnt.index[(cnt > 2 * cnt.rolling(30).mean().shift(1)).to_numpy()]
    rets = []
    for f in sorted(glob.glob(str(ROOT / "data" / "market" / "history" / "*.parquet"))):
        h = pd.read_parquet(f, columns=["date", "adj_close"])
        h["date"] = pd.to_datetime(h["date"].astype(str).str[:10])
        s = h.drop_duplicates("date").set_index("date")["adj_close"].sort_index()
        rets.append(s.pct_change().rename(Path(f).stem))
    R = pd.concat(rets, axis=1)
    basket = R.mean(axis=1).where(R.count(axis=1) >= 10).dropna()
    gold = sp.ctx["gold"].pct_change()
    for iid, ser, cost_rt, sign, desc in (("G1", basket, 0.002, -1.0, "Africa unrest spike -> short miners 5d"),
                                          ("G2", gold.dropna(), 0.0005, 1.0, "Africa unrest spike -> long gold 5d")):
        ser = ser[(ser.index >= FIRST) & (ser.index <= END)]
        tdays = ser.index
        pos = pd.Series(0.0, index=tdays)
        for d in spike_days:
            i = np.searchsorted(tdays.values, np.datetime64(d), side="right")
            pos.iloc[i + 1:i + 6] = sign
        g = pos * ser * 10000
        c = (pos - pos.shift(1).fillna(0)).abs() * cost_rt / 2 * 10000
        out.append(idea("G", iid, desc, g, c, ((pos != 0) & (tdays < ERA)).sum(), ((pos != 0) & (tdays >= ERA)).sum(),
                        tdays, unit="$ per $10k"))
    # G3 Reddit crowding
    rd = pd.read_parquet(ROOT / "data" / "social" / "agg" / "reddit_hourly" / "wallstreetbets_posts.parquet")
    t = pd.to_datetime(rd["hour_utc"])
    t = (t.dt.tz_localize("UTC") if t.dt.tz is None else t).dt.tz_convert(ET)
    day = (t + pd.Timedelta(hours=8)).dt.tz_localize(None).dt.normalize()          # 16:00 ET d-1 .. 16:00 ET d -> d
    dc = rd.groupby(day.to_numpy())["count"].sum()
    dc = dc[dc.index.isin(sp.cal)]
    spike = dc > 1.5 * dc.rolling(20).mean().shift(1)
    sd = dc.index[spike.to_numpy()]
    nd = sp.cal[np.minimum(np.searchsorted(sp.cal.values, sd.values, side="right"), len(sp.cal) - 1)]
    ridx = sp.days[sp.days >= dc.index[20]] if len(dc) > 20 else sp.days[-1:]
    out.append(sp_pos(sp, "G", "G3", "WSB post spike -> short S&P next day", pd.Series(-1.0, index=nd), index=ridx,
                      split=pd.Timestamp("2026-04-01")))
    # Bitcoin as its own market
    bstart, bend, bsplit = pd.Timestamp("2022-04-21"), END, pd.Timestamp("2024-07-01")
    cidx = pd.date_range(bstart, bend, freq="D")

    def btc_trades(iid, desc, t0: pd.DatetimeIndex, t1: pd.DatetimeIndex, book_on: pd.DatetimeIndex, sgn=None):
        p0, p1 = btc_at(btc, t0), btc_at(btc, t1)
        r = (p1 / p0 - 1) * (1.0 if sgn is None else sgn)
        s = pd.Series(r * 10000, index=book_on).dropna()
        s = s[(s.index >= bstart) & (s.index <= bend)]
        s = s.groupby(level=0).sum()
        return idea("G", iid, desc, s, pd.Series(10.0, index=s.index), (s.index < bsplit).sum(), (s.index >= bsplit).sum(),
                    cidx, bsplit, unit="$ per $10k")

    fri = cidx[cidx.dayofweek == 4]
    out.append(btc_trades("G4", "Bitcoin weekend (Fri 16:00 -> Mon 09:30)", fri + pd.Timedelta(hours=16),
                          fri + pd.Timedelta(days=3, hours=9, minutes=30), fri + pd.Timedelta(days=3)))
    wk = cidx[cidx.dayofweek < 5]
    out.append(btc_trades("G5", "Bitcoin during US hours", wk + pd.Timedelta(hours=9, minutes=30), wk + pd.Timedelta(hours=16), wk))
    mt = cidx[cidx.dayofweek < 4]
    out.append(btc_trades("G6", "Bitcoin weekday nights", mt + pd.Timedelta(hours=16), mt + pd.Timedelta(days=1, hours=9, minutes=30),
                          mt + pd.Timedelta(days=1)))
    allc = pd.date_range(bstart - pd.Timedelta(days=260), bend, freq="D")
    cl = pd.Series(btc_at(btc, allc + pd.Timedelta(hours=16)), index=allc)
    r1 = cl.pct_change()
    for iid, pos, desc in (("G7", (cl > cl.rolling(200).mean()).astype(float).where(cl.rolling(200).mean().notna()).shift(1),
                            "Bitcoin 200-day trend"),
                           ("G8", (-np.sign(r1)).shift(1), "Bitcoin 1-day reversal")):
        pos = pos.fillna(0)
        g = pos * r1.fillna(0) * 10000
        c = (pos - pos.shift(1).fillna(0)).abs() * 5.0
        held = pos.reindex(cidx)
        out.append(idea("G", iid, desc, g, c, ((held != 0) & (cidx < bsplit)).sum(), ((held != 0) & (cidx >= bsplit)).sum(),
                        cidx, bsplit, unit="$ per $10k"))
    lg = sp.legs
    d9 = lg.index[lg.index.isin(sp.cal) & (lg.index >= bstart)]
    nx = sp.cal[np.minimum(np.searchsorted(sp.cal.values, d9.values, side="right"), len(sp.cal) - 1)]
    sgn = np.sign(lg["dayleg"].reindex(d9).to_numpy())
    out.append(btc_trades("G9", "Bitcoin follows the S&P session overnight", d9 + pd.Timedelta(hours=16),
                          nx + pd.Timedelta(hours=9, minutes=30), nx, sgn))
    return out


# ----------------------------------------------------------------------------------------------- scoring
def tstat(v: np.ndarray) -> float:
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(len(v))) if sd > 0 and len(v) > 2 else 0.0


def p_upper(t: float) -> float:
    return 0.5 * math.erfc(t / math.sqrt(2))


def score(it: dict) -> dict:
    g, c, split = it["g"], it["c"], it["split"]
    fwd, rev = g - c, -g - c
    tf, tr = tstat(fwd.to_numpy()), tstat(rev.to_numpy())
    if it["one_sided"]:
        direction, net, t = "as written", fwd, tf
        p = p_upper(tf)
    else:
        direction, net, t = ("as written", fwd, tf) if tf >= tr else ("reversed", rev, tr)
        p = min(1.0, 2 * p_upper(t))
    e1, e2 = net[net.index < split], net[net.index >= split]
    years = max((net.index[-1] - net.index[0]).days / 365.25, 1e-9)
    ntr = it["n1"] + it["n2"]
    row = {"id": it["id"], "family": it["family"], "desc": it["desc"], "unit": it["unit"], "direction": direction,
           "trades_era1": it["n1"], "trades_era2": it["n2"], "net_per_year": float(net.sum() / years),
           "net_per_trade": float(net.sum() / ntr) if ntr else np.nan, "gross_t_as_written": tstat(g.to_numpy()),
           "t": t, "p": p, "era1_mean": float(e1.mean()) if len(e1) else np.nan,
           "era2_mean": float(e2.mean()) if len(e2) else np.nan, "split": str(split.date()),
           "scored": bool(it["n1"] >= MIN_TRADES and it["n2"] >= MIN_TRADES)}
    row["both_eras"] = bool(row["era1_mean"] > 0 and row["era2_mean"] > 0)
    return row


def bh(p: np.ndarray) -> np.ndarray:
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    run = 1.0
    for rank in range(m, 0, -1):
        i = o[rank - 1]
        run = min(run, p[i] * m / rank)
        q[i] = run
    return q


def partner_check(items: dict, ids: list, sp: SP) -> dict:
    from odte.research.partner_b import marks as b_marks
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    early = b0.index <= pd.Timestamp("2021-12-31")
    m = b_marks(sp.dt)
    b = book(b0.index, m["day"], m["net"] * MES)
    base = b0 + float(b0[early].std(ddof=1) / b[early].std(ddof=1)) * b

    def pl(x):
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        return float((yr.sum(axis=1) < 0).mean())

    res = {"B0+B": {"sharpe": sharpe(base), "p_losing_year": pl(base)}}
    for iid, direction in ids:
        it = items[iid]
        net = (it["g"] - it["c"]) if direction == "as written" else (-it["g"] - it["c"])
        x = book(b0.index, pd.Series(net.index, index=net.index), net)
        live = x != 0
        win = early & live
        if win.sum() < 60:
            first = np.flatnonzero(live.to_numpy())
            win = pd.Series(False, index=b0.index)
            if len(first):
                win.iloc[first[0]:first[0] + 252] = True
        sx = x[win].std(ddof=1)
        if not np.isfinite(sx) or sx == 0:
            res[iid] = {"note": "no overlap with B0 days"}
            continue
        comb = base + float(b0[win].std(ddof=1) / sx) * x
        res[iid] = {"corr_with_B0": float(np.corrcoef(b0, x)[0, 1]), "sharpe": sharpe(comb), "p_losing_year": pl(comb)}
        res[iid]["helps"] = bool(res[iid]["sharpe"] > res["B0+B"]["sharpe"] and res[iid]["p_losing_year"] < res["B0+B"]["p_losing_year"])
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sp = SP()
    btc = btc_minutes()
    items = []
    for name, fn in (("A", lambda: fam_A(sp)), ("B", lambda: fam_B(sp)), ("C", lambda: fam_C(sp, btc)), ("D", lambda: fam_D(sp)),
                     ("E", fam_E), ("F", fam_F), ("G", lambda: fam_G(sp, btc))):
        got = fn()
        print(f"family {name}: {len(got)} ideas", flush=True)
        items += got
    rows = [score(it) for it in items]
    tab = pd.DataFrame(rows)
    sc = tab["scored"]
    tab["q"] = np.nan
    tab.loc[sc, "q"] = bh(tab.loc[sc, "p"].to_numpy())
    tab["verdict"] = np.where(~sc, "NOT SCORED",
                              np.where((tab["q"] <= Q_FDR) & tab["both_eras"], "PASS",
                                       np.where((tab["p"] < 0.05) & tab["both_eras"], "LEAD", "FAIL")))
    tab.to_csv(OUT / "sweep1.csv", index=False)
    by_id = {it["id"]: it for it in items}
    chk = [(r["id"], r["direction"]) for _, r in tab.iterrows() if r["verdict"] in ("PASS", "LEAD")]
    pc = partner_check(by_id, chk, sp) if chk else {}
    res = {"ideas": int(len(tab)), "scored": int(sc.sum()), "verdicts": tab["verdict"].value_counts().to_dict(),
           "partner_check": pc, "table": tab.to_dict("records")}
    (OUT / "sweep1.json").write_text(json.dumps(res, indent=1, default=float))
    pd.set_option("display.width", 250)
    show = tab[["id", "desc", "direction", "trades_era1", "trades_era2", "net_per_year", "t", "p", "q", "era1_mean",
                "era2_mean", "verdict"]]
    print(show.sort_values("p").to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(json.dumps({"verdicts": res["verdicts"], "partner_check": pc}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
