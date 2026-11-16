"""Partner T (docs/partner_plan.md): the turn of the month. Long 1 MES from the close of day -2 (the day before
the month's last trading day) to the close of day +3 (the third trading day of the next month). Does it make
money on its own (T1), and does it make B0 steadier (T2)?

    python3 -m odte.research.partner_t      # -> reports/partners/t.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday,
                                    sunday_to_monday)

from odte.research.gamma_momentum import es_bars
from odte.research.partner_o import legs

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "partners"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
COST, MES, SEED = 0.75, 5.0, 20261115
FIRST, END = pd.Timestamp("2016-01-01"), pd.Timestamp("2026-11-06")
ERA = pd.Timestamp("2022-01-01")
SCALE_END = pd.Timestamp("2021-12-31")
CLOSURES = {pd.Timestamp("2018-12-05"), pd.Timestamp("2025-01-09")}


class NYSE(AbstractHolidayCalendar):
    rules = [Holiday("NewYear", month=1, day=1, observance=sunday_to_monday), USMartinLutherKingJr, USPresidentsDay,
             GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("July4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


def nyse_days(start="2015-11-01", end="2026-12-31") -> pd.DatetimeIndex:
    hol = set(NYSE().holidays(start, end)) | CLOSURES
    d = pd.bdate_range(start, end)
    return d[~d.isin(hol)]


def day_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Per ES session that reaches 13:10 (full or half day): last regular-hours close, its contract, and the 09:30
    open and its contract."""
    r = bars.reset_index()
    last = r.groupby("date").tail(1).set_index("date")
    last = last[last["hhmm"] >= "13:10"]
    o = r[r["hhmm"] == "09:30"].set_index("date")
    return pd.DataFrame({"close": last["close"], "inst": last["instrument_id"],
                         "open": o["open"].reindex(last.index), "open_inst": o["instrument_id"].reindex(last.index)})


def windows(cal: pd.DatetimeIndex, dt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per window (gross points) and one row per held day (close-to-close mark, night and day parts)."""
    s = pd.Series(cal, index=cal)
    pos = {d: i for i, d in enumerate(cal)}
    lasts = [pd.Timestamp(d) for d in s.groupby([cal.year, cal.month]).max()]
    rows, marks = [], []
    for L in lasts:
        i = pos[L]
        if i < 1 or i + 3 >= len(cal):
            continue
        entry, held = cal[i - 1], [cal[i], cal[i + 1], cal[i + 2], cal[i + 3]]
        ex = held[-1]
        if not (FIRST <= ex <= END):
            continue
        r = {"entry": entry, "exit": ex, "gross": np.nan, "skip": ""}
        if entry not in dt.index or any(d not in dt.index for d in held):
            r["skip"] = "missing close"
        elif len({dt.at[d, "inst"] for d in [entry] + held}) > 1:
            r["skip"] = "contract roll"
        else:
            r["gross"] = dt.at[ex, "close"] - dt.at[entry, "close"]
            prev = entry
            for k, d in zip((-1, 1, 2, 3), held):
                m = {"day": d, "k": k, "mark": dt.at[d, "close"] - dt.at[prev, "close"], "exit": d == ex,
                     "night": np.nan, "session": np.nan}
                if dt.at[d, "open_inst"] == dt.at[d, "inst"] and np.isfinite(dt.at[d, "open"]):
                    m["night"] = dt.at[d, "open"] - dt.at[prev, "close"]
                    m["session"] = dt.at[d, "close"] - dt.at[d, "open"]
                marks.append(m)
                prev = d
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(marks)


def st(v, cost=COST, per_year=12) -> dict:
    """per_year: 12 for windows, about 48 for held days (4 a month)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)] - cost
    if len(v) < 3:
        return {"n": int(len(v))}
    eq = np.cumsum(v * MES)
    return {"n": int(len(v)), "mean_pts": float(v.mean()), "t": float(v.mean() / v.std(ddof=1) * np.sqrt(len(v))),
            "usd_per_mes_per_year": float(v.mean() * MES * per_year),
            "hit": float((v > 0).mean()), "worst_pts": float(v.min()),
            "max_dd_usd_per_mes": float((eq - np.maximum.accumulate(eq)).min())}


def book(b0_index: pd.DatetimeIndex, days: pd.Series, pnl: pd.Series) -> pd.Series:
    """Book each P&L on the first B0 day on or after its date; drop dates outside B0's span."""
    out = pd.Series(0.0, index=b0_index)
    keep = (days > b0_index[0] - pd.Timedelta(days=1)) & (days <= b0_index[-1])
    p = np.searchsorted(b0_index.values, days[keep].values)
    np.add.at(out.values, p, pnl[keep].to_numpy())
    return out


def main() -> int:
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    OUT.mkdir(parents=True, exist_ok=True)
    bars = es_bars()
    cal = nyse_days()
    dt = day_table(bars)
    w, mk = windows(cal, dt)
    ok = w.dropna(subset=["gross"])
    era = {"2016-2021": st(ok.loc[ok["exit"] < ERA, "gross"]), "2022-2026": st(ok.loc[ok["exit"] >= ERA, "gross"])}
    res = {"windows": int(len(w)), "skipped": w.loc[w["skip"] != "", "skip"].value_counts().to_dict(),
           "T1": {"all": st(ok["gross"]), **era}}
    res["T1"]["pass"] = bool(era["2016-2021"]["mean_pts"] > 0 and era["2022-2026"]["mean_pts"] > 0
                             and res["T1"]["all"]["t"] >= 2.0)

    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    mk["net"] = (mk["mark"] - np.where(mk["exit"], COST, 0.0)) * MES
    t = book(b0.index, mk["day"], mk["net"])
    early = b0.index <= SCALE_END
    wt = float(b0[early].std(ddof=1) / t[early].std(ddof=1))
    comb = b0 + wt * t

    def p_loss(x: pd.Series) -> float:
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        return float((yr.sum(axis=1) < 0).mean())

    pl = {"B0": p_loss(b0), "B0+T": p_loss(comb)}
    res["T2"] = {"held_days_booked": int(((mk["day"] >= b0.index[0]) & (mk["day"] <= b0.index[-1])).sum()),
                 "corr": float(np.corrcoef(b0, t)[0, 1]), "weight_mes_per_b0_unit": wt,
                 "sharpe": {"B0": sharpe(b0), "T": sharpe(t), "B0+T": sharpe(comb)},
                 "mean_per_day": {"B0": float(b0.mean()), "T_scaled": float((wt * t).mean()), "B0+T": float(comb.mean())},
                 "p_losing_year": pl,
                 "by_year": {int(y): {"B0": float(b0[b0.index.year == y].mean()),
                                      "T_scaled": float((wt * t)[t.index.year == y].mean()),
                                      "B0+T": float(comb[comb.index.year == y].mean())} for y in sorted(set(b0.index.year))}}
    r2 = res["T2"]
    r2["pass"] = bool(r2["corr"] <= 0.10 and r2["sharpe"]["B0+T"] > r2["sharpe"]["B0"] and pl["B0+T"] < pl["B0"])
    e = res["T1"]
    lead = e["2016-2021"]["mean_pts"] > 0 and e["2022-2026"]["mean_pts"] > 0 and e["all"]["t"] >= 1.0 and r2["pass"]
    res["verdict"] = "PASS" if e["pass"] and r2["pass"] else ("LEAD" if lead else "FAIL")

    # ---- reported only ----
    # calendar or just long? daily close-to-close % returns, window days vs all other trading days
    prev = dt.shift(1)
    same = dt["inst"] == prev["inst"]
    gap_ok = pd.Series(dt.index, index=dt.index).diff().dt.days <= 5
    ret = (dt["close"] / prev["close"] - 1.0)[same & gap_ok]
    ret = ret[(ret.index >= FIRST) & (ret.index <= END)]
    in_w = ret.index.isin(set(mk["day"]))
    a, b = ret[in_w].to_numpy(), ret[~in_w].to_numpy()
    welch = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    rep = {"window_vs_other_days_pct": {"window_mean": float(a.mean() * 100), "other_mean": float(b.mean() * 100),
                                         "n_window": int(len(a)), "n_other": int(len(b)), "welch_t": float(welch),
                                         "ratio": float(a.mean() / b.mean()) if b.mean() != 0 else None},
           "by_day_k_gross_pts": {str(k): st(z["mark"], 0.0) for k, z in mk.groupby("k")},  # $ figure: 12 such days a year
           "nights_gross_pts": st(mk["night"], 0.0, 48), "sessions_gross_pts": st(mk["session"], 0.0, 48),
           "costs": {str(c): st(ok["gross"], c) for c in (0.0, COST, 1.5)},
           "by_year": {int(y): st(z["gross"]) for y, z in ok.groupby(ok["exit"].dt.year)}}
    # O's daily series (as booked in partner_o) for overlap and the three-way combination
    x = legs(bars).dropna(subset=["night"])
    o = book(b0.index, x["day"], (x["night"] - COST) * MES)
    wo = float(b0[early].std(ddof=1) / o[early].std(ddof=1))
    comb3 = b0 + wo * o + wt * t
    rep["corr_T_O"] = float(np.corrcoef(t, o)[0, 1])
    rep["B0+O+T"] = {"sharpe": {"B0": sharpe(b0), "B0+O": sharpe(b0 + wo * o), "B0+O+T": sharpe(comb3)},
                     "p_losing_year": {"B0": pl["B0"], "B0+O": p_loss(b0 + wo * o), "B0+O+T": p_loss(comb3)}}
    res["report_only"] = rep
    (OUT / "t.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
