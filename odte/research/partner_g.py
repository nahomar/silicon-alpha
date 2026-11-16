"""Partner G (docs/partner_plan.md): sell the standard fly on same-day GLD and TLT options at 10:00, 12:00 and
14:00, touch costs, legs in the money at 15:59 closed then (these options settle into shares). Does it make
money on its own (G1), and does it make B0 steadier (G2)?

    python3 -m odte.research.partner_g      # -> reports/partners/g.json, g_trades.parquet
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "partners"
ETF = ROOT / "data" / "odte_etf"
B0 = ROOT / "reports" / "stress" / "stress_spx_daily.parquet"
FLY = {"structure": "iron_fly", "wing": 1.5}
TIMES = ("10:00", "12:00", "14:00")
PRIMARY = ("GLD", "TLT")
OTHERS = ("USO", "EEM", "XLF")
START, END = pd.Timestamp("2024-11-11"), pd.Timestamp("2026-11-06")
HALF2 = pd.Timestamp("2025-11-01")
SEED = 20261115


def settles(sym: str) -> pd.Series:
    return pd.read_csv(ETF / sym / "settlements.csv", parse_dates=["date"]).set_index("date")["close"]


def score_day(args) -> list[dict]:
    """Every decision of one (symbol, day): P&L per $1k risk at touch and half, ITM legs closed at 15:59; and the
    held-to-close upper bound. Also the day's 10:00 straddle and open-to-close move (for the rich-only filter)."""
    sym, day = args
    from odte.research import zoo
    from odte.research.struct4 import price
    st = settles(sym)
    f = ETF / sym / f"{day:%Y-%m-%d}.parquet"
    if not f.exists() or day not in st.index:
        return []
    ch = pd.read_parquet(f)
    ch["quote_datetime"] = pd.to_datetime(ch["quote_datetime"])
    ch = ch[(ch["bid"] > 0) & (ch["ask"] >= ch["bid"])].sort_values("quote_datetime")
    S_c = float(st[day])
    try:
        zd = zoo.Day(day, ch, S_c, np.nan, {})
    except Exception:
        return []
    t_x = zd.at("15:59")
    if t_x not in zd.minutes:
        return []
    S_x = zd.forward(t_x)
    m, h = zd.mid.loc[t_x], zd.half.loc[t_x]
    t10 = zd.at("10:00")
    st10 = zd.straddle(t10, zd.forward(t10))[1] if t10 in zd.minutes else np.nan
    open_ = zd.forward(zd.at("09:31"))
    base = {"sym": sym, "day": day, "straddle_10": st10, "abs_move_oc": abs(S_c - open_) if np.isfinite(open_) else np.nan}
    out = []
    for hh in TIMES:
        r = dict(base, hhmm=hh)
        for share, tag in ((0.5, "half"), (1.0, "touch")):
            p = price(FLY, zd, hh, 1.0, share)
            if p is None:
                r[f"itm_{tag}"] = r[f"hold_{tag}"] = np.nan
                continue
            cash, n_close, rest = p["cash_in"], 0, 0.0
            for right, k, q in p["legs"]:
                itm = (S_x > k) if right == "C" else (S_x < k)
                if itm:                                        # close at 15:59: mid -/+ half-spread * share
                    try:
                        mid, hs = float(m[(right, k)]), float(h[(right, k)])
                    except KeyError:
                        mid, hs = np.nan, np.nan
                    if not np.isfinite(mid):
                        mid, hs = max((S_x - k) if right == "C" else (k - S_x), 0.0), 0.0
                    cash += q * mid - abs(q) * hs * share
                    n_close += abs(q)
                else:                                          # expires; intrinsic at the close if it crossed late
                    rest += q * (max(S_c - k, 0.0) if right == "C" else max(k - S_c, 0.0))
            pnl = (cash + rest) * 100 - p["fees"] - n_close * zoo.FEE
            r[f"itm_{tag}"] = pnl / p["risk"] * 1000
            r[f"hold_{tag}"] = ((p["cash_in"] + zoo.payoff(p["legs"], S_c)) * 100 - p["fees"]) / p["risk"] * 1000
            r["risk_per_contract"] = p["risk"]
        out.append(r)
    return out


def days_for(sym: str) -> list:
    return sorted(d for d in (pd.Timestamp(p.stem) for p in (ETF / sym).glob("20??-??-??.parquet")) if START <= d <= END)


def tstat(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else float("nan")


def daily(t: pd.DataFrame, col: str, syms=PRIMARY) -> pd.Series:
    z = t[t["sym"].isin(syms)]
    return z.groupby("day")[col].sum(min_count=1).dropna()


def halves(s: pd.Series) -> dict:
    a, b = s[s.index < HALF2], s[s.index >= HALF2]
    return {"first": {"n": int(len(a)), "mean": float(a.mean()), "t": tstat(a)},
            "second": {"n": int(len(b)), "mean": float(b.mean()), "t": tstat(b)},
            "all": {"n": int(len(s)), "mean": float(s.mean()), "t": tstat(s)}}


def main() -> int:
    from odte.research.stress_spx import sharpe, stationary_bootstrap
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, d) for s in PRIMARY + OTHERS for d in days_for(s)]
    with ProcessPoolExecutor(10) as ex:
        t = pd.DataFrame([r for part in ex.map(score_day, jobs, chunksize=4) for r in part])
    t.to_parquet(OUT / "g_trades.parquet", index=False)

    g = daily(t, "itm_touch")
    res = {"jobs": len(jobs), "decisions": int(t["itm_touch"].notna().sum()),
           "decisions_by_sym": t.groupby("sym")["itm_touch"].count().to_dict(),
           "G1": halves(g)}
    res["G1"]["pass"] = bool(res["G1"]["first"]["mean"] > 0 and res["G1"]["second"]["mean"] > 0
                             and res["G1"]["all"]["t"] >= 2.0)

    b0 = pd.read_parquet(B0)["pnl_per_1k"]
    b0.index = pd.DatetimeIndex(b0.index)
    b0 = b0[(b0.index >= START) & (b0.index <= END)]
    gd = pd.Series(0.0, index=b0.index)
    common = gd.index.intersection(g.index)
    gd.loc[common] = g.reindex(common)
    first = b0.index < HALF2
    w = float(b0[first].std(ddof=1) / gd[first].std(ddof=1))
    comb = b0 + w * gd
    p_loss = {}
    for name, x in (("B0", b0), ("B0+G", comb)):
        yr = stationary_bootstrap(x.to_numpy(), 10000, 252, 20.0, np.random.default_rng(SEED))
        p_loss[name] = float((yr.sum(axis=1) < 0).mean())
    both = (b0 != 0) & (gd != 0)
    res["G2"] = {"days": int(len(b0)), "g_days": int((gd != 0).sum()), "g_days_not_in_b0": int(len(g.index.difference(b0.index))),
                 "corr": float(np.corrcoef(b0, gd)[0, 1]),
                 "corr_on_overlap": float(np.corrcoef(b0[both], gd[both])[0, 1]) if both.sum() > 2 else None,
                 "weight": w, "sharpe": {"B0": sharpe(b0), "G": sharpe(gd), "B0+G": sharpe(comb)},
                 "mean_per_day": {"B0": float(b0.mean()), "G_scaled": float((w * gd).mean()), "B0+G": float(comb.mean())},
                 "p_losing_year": p_loss,
                 "halves": {"B0": halves(b0), "B0+G": halves(comb)}}
    r2 = res["G2"]
    r2["pass"] = bool(r2["corr"] <= 0.10 and r2["sharpe"]["B0+G"] > r2["sharpe"]["B0"] and p_loss["B0+G"] < p_loss["B0"])
    g1 = res["G1"]
    lead = g1["first"]["mean"] > 0 and g1["second"]["mean"] > 0 and g1["all"]["t"] >= 1.0 and r2["pass"]
    res["verdict"] = "PASS" if g1["pass"] and r2["pass"] else ("LEAD" if lead else "FAIL")

    # reported only
    rep = {"by_sym_touch": {s: halves(daily(t, "itm_touch", (s,))) for s in PRIMARY + OTHERS},
           "by_time_touch": {hh: halves(daily(t[t["hhmm"] == hh], "itm_touch")) for hh in TIMES},
           "half_gap": halves(daily(t, "itm_half")),
           "hold_to_close_touch": halves(daily(t, "hold_touch")),
           "per_decision_touch": {s: float(z["itm_touch"].mean()) for s, z in t.groupby("sym")},
           "per_decision_half": {s: float(z["itm_half"].mean()) for s, z in t.groupby("sym")},
           "median_risk_per_contract": {s: float(z["risk_per_contract"].median()) for s, z in t.groupby("sym")}}
    # rich-only: today's 10:00 straddle > mean |open-to-close| of the previous 20 sessions of that symbol
    rich = []
    for s, z in t.groupby("sym"):
        dd = z.drop_duplicates("day").set_index("day").sort_index()
        prev = dd["abs_move_oc"].shift(1).rolling(20, min_periods=20).mean()
        ok = (dd["straddle_10"] > prev)
        rich.append(z[z["day"].map(ok).fillna(False).astype(bool)])
    rr = pd.concat(rich)
    rep["rich_only_touch"] = halves(daily(rr, "itm_touch"))
    rep["rich_share_of_days"] = float(rr["day"].nunique() / t["day"].nunique())
    res["report_only"] = rep
    (OUT / "g.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
