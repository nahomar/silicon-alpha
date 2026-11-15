"""Stress test of the SPX fly strategy (docs/stress_plan.md): walk-forward by year, worse costs and
fills, Monte Carlo drawdowns, parameter sensitivity. Development data only.

    python3 -m odte.research.stress_spx        # -> reports/stress/stress_spx.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import exam, meta

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "stress"
EXAM_FROM, EXAM_TO = pd.Timestamp("2026-09-08"), pd.Timestamp("2026-11-06")
ALPHAS = (100, 300, 1000, 3000, 10000)
THRESHOLDS = (-20, -10, 0, 10, 20, 40)
PER_DAY = (1, 2, 3, 99)
WINGS = {"1.0": "fly_narrow", "1.5": "fly", "2.0": "fly_wide"}
CENTRE = {"alpha": 1000, "thr": 0, "per_day": 2, "wing": "1.5"}


def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def max_dd(x) -> float:
    x = np.asarray(x, float)
    return float(_dd_rows(x[None, :])[0]) if len(x) else 0.0


def _dd_rows(P: np.ndarray) -> np.ndarray:
    """Max drawdown of each row of a (paths x days) P&L matrix (peak includes the start at 0)."""
    eq = np.cumsum(P, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((len(P), 1)), eq], axis=1), axis=1)[:, 1:]
    return (eq - peak).min(axis=1)


def predictions(d: pd.DataFrame, alpha: float) -> dict:
    feats = exam.feature_list(d, exam.GROUPS)
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Y = {k: d[f"y_{k}_half"].to_numpy(float) for k in exam.MENU}
    return meta.walk_forward(d, X, Y, f"ridge_a{alpha:g}")


def trades(d: pd.DataFrame, preds: dict, thr: float, per_day: int) -> pd.DataFrame:
    """SPX rows where the best predicted shape is the fly and its prediction > thr; first `per_day`."""
    P = np.column_stack([preds[k] for k in exam.MENU])
    has = np.isfinite(P).any(axis=1)
    Pf = np.where(np.isfinite(P), P, -np.inf)
    best = np.array(exam.MENU)[Pf.argmax(axis=1)]
    take = has & (Pf.max(axis=1) > thr) & (best == "fly") & ~d["post"].to_numpy() & (d["index"] == "SPX").to_numpy()
    t = d.loc[take, ["day", "hhmm"]].copy()
    t["row"] = np.flatnonzero(take)
    t = t.sort_values(["day", "hhmm"])
    return t[t.groupby("day").cumcount() < per_day]


def ret(d: pd.DataFrame, labels: pd.DataFrame, rows: np.ndarray, wing: str = "1.5", share: float = 1.0) -> np.ndarray:
    k = WINGS[wing]
    src = d if k == "fly" else labels
    h = src[f"y_{k}_half"].to_numpy(float)[rows]
    t = src[f"y_{k}_touch"].to_numpy(float)[rows]
    r = h + (share - 0.5) * 2.0 * (t - h)                 # linear in the cost share (0.5 half, 1.0 touch)
    return np.where(np.isfinite(r), r, 0.0)


def daily(days: np.ndarray, t: pd.DataFrame, r: np.ndarray) -> pd.Series:
    return pd.Series(r, index=t["day"].to_numpy()).groupby(level=0).sum().reindex(days, fill_value=0.0)


def summary(x: pd.Series) -> dict:
    return {"days": int(len(x)), "trade_days": int((x != 0).sum()), "mean_day": float(x.mean()),
            "per_year": float(x.mean() * 252), "sharpe": sharpe(x), "max_dd": max_dd(x.to_numpy()),
            "worst_day": float(x.min()), "win_days": float((x[x != 0] > 0).mean()) if (x != 0).any() else float("nan")}


def stationary_bootstrap(x: np.ndarray, n_paths: int, length: int, mean_block: float, rng) -> np.ndarray:
    n = len(x)
    idx = np.empty((n_paths, length), int)
    idx[:, 0] = rng.integers(0, n, n_paths)
    jump = rng.random((n_paths, length)) < 1.0 / mean_block
    fresh = rng.integers(0, n, (n_paths, length))
    for j in range(1, length):
        idx[:, j] = np.where(jump[:, j], fresh[:, j], (idx[:, j - 1] + 1) % n)
    return x[idx]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = meta.load().reset_index(drop=True)
    labels = d[["day", "index", "hhmm"]].merge(
        pd.read_parquet(ROOT / "data" / "research" / "menu3_labels.parquet"), on=["day", "index", "hhmm"], how="left")
    assert len(labels) == len(d)
    rng = np.random.default_rng(20261114)
    preds = {a: predictions(d, a) for a in ALPHAS}
    first = min(d.loc[np.isfinite(preds[1000]["fly"]), "day"])
    days = pd.DatetimeIndex(sorted(d.loc[(d["index"] == "SPX") & (d["day"] >= first), "day"].unique()))

    def strat(alpha=1000, thr=0, per_day=2, wing="1.5", share=1.0):
        t = trades(d, preds[alpha], thr, per_day)
        return t, daily(days, t, ret(d, labels, t["row"].to_numpy(), wing, share))

    res = {"first_predicted_day": str(first.date()), "days": int(len(days))}
    t0, x0 = strat()
    res["centre"] = summary(x0)
    res["centre"]["trades"] = int(len(t0))

    # T1 walk-forward by year
    yrs = x0.groupby(x0.index.year)
    by_year = {int(y): summary(v) for y, v in yrs}
    # full calendar years = every year after the first predicted one (before 2022 SPX had 0DTE
    # expiries only on Mon/Wed/Fri, so those years have ~160 sessions)
    full = {y: v for y, v in by_year.items() if y > first.year}
    s_a = sharpe(x0[(days >= pd.Timestamp("2018-01-01")) & (days < pd.Timestamp("2022-01-01"))])
    s_b = sharpe(x0[days >= pd.Timestamp("2022-01-01")])
    ex = x0[(days >= EXAM_FROM) & (days <= EXAM_TO)]
    pos_share = float(np.mean([v["mean_day"] > 0 for v in full.values()]))
    res["T1"] = {"by_year": by_year, "positive_full_years": f"{sum(v['mean_day'] > 0 for v in full.values())}/{len(full)}",
                 "sharpe_2018_21": s_a, "sharpe_2022_26": s_b, "exam_window": summary(ex),
                 "pass": bool(pos_share >= 0.7 and s_a > 0.5 and s_b > 0.5)}

    # T2 worse costs and fills
    rows0 = t0["row"].to_numpy()
    r_touch = ret(d, labels, rows0, "1.5", 1.0)
    miss = []
    win = np.flatnonzero(r_touch > 0)
    for _ in range(1000):
        r = r_touch.copy()
        r[rng.choice(win, int(round(0.2 * len(win))), replace=False)] = 0.0
        miss.append(daily(days, t0, r).mean())
    t2 = {"half": summary(strat(share=0.5)[1]), "touch": res["centre"],
          "touch_x1.5": summary(strat(share=1.5)[1]), "touch_x2": summary(strat(share=2.0)[1]),
          "miss_20pct_winners_mean_day_median": float(np.median(miss)),
          "miss_20pct_winners_mean_day_p5": float(np.percentile(miss, 5))}
    t2["pass"] = bool(t2["touch_x1.5"]["mean_day"] > 0 and t2["miss_20pct_winners_mean_day_median"] > 0)
    res["T2"] = t2

    # T3 Monte Carlo
    x = x0.to_numpy()
    sh = np.array([rng.permutation(x) for _ in range(10000)])
    dd_sh = _dd_rows(sh)
    yr = stationary_bootstrap(x, 10000, 252, 20.0, rng)
    pnl_yr, dd_yr = yr.sum(axis=1), _dd_rows(yr)
    res["T3"] = {"actual_max_dd": max_dd(x),
                 "shuffled_max_dd": {"median": float(np.median(dd_sh)), "p95": float(np.percentile(dd_sh, 5)),
                                     "p99": float(np.percentile(dd_sh, 1))},
                 "one_year": {"p_loss": float((pnl_yr < 0).mean()), "pnl_p5": float(np.percentile(pnl_yr, 5)),
                              "pnl_median": float(np.median(pnl_yr)), "max_dd_p95": float(np.percentile(dd_yr, 5)),
                              "max_dd_p99": float(np.percentile(dd_yr, 1))}}
    res["T3"]["pass"] = bool(res["T3"]["one_year"]["p_loss"] <= 0.20)

    # T4 parameter sensitivity
    def cell(**kw):
        s = summary(strat(**kw)[1])
        return {"mean_day": s["mean_day"], "sharpe": s["sharpe"], "trades": int(len(strat(**kw)[0]))}
    grid = {f"a{a}_t{t}": cell(alpha=a, thr=t) for a in ALPHAS for t in THRESHOLDS}
    one = {"per_day": {str(n): cell(per_day=n) for n in PER_DAY}, "wing": {w: cell(wing=w) for w in WINGS}}
    c = grid["a1000_t0"]
    nbrs = {"alpha 300": grid["a300_t0"], "alpha 3000": grid["a3000_t0"], "thr -10": grid["a1000_t-10"],
            "thr +10": grid["a1000_t10"], "1 per day": one["per_day"]["1"], "3 per day": one["per_day"]["3"],
            "wing 1.0": one["wing"]["1.0"], "wing 2.0": one["wing"]["2.0"]}
    ok = {k: bool(v["mean_day"] > 0 and v["sharpe"] >= 0.5 * c["sharpe"]) for k, v in nbrs.items()}
    res["T4"] = {"alpha_x_threshold": grid, "one_at_a_time": one, "neighbours_ok": ok, "pass": bool(all(ok.values()))}

    res["overall"] = {k: res[k]["pass"] for k in ("T1", "T2", "T3", "T4")}
    (OUT / "stress_spx.json").write_text(json.dumps(res, indent=2, default=float))
    x0.rename("pnl_per_1k").to_frame().to_parquet(OUT / "stress_spx_daily.parquet")

    c0 = res["centre"]
    print(f"centre: {c0['trades']} trades, {c0['trade_days']}/{c0['days']} days, ${c0['mean_day']:+.1f}/day per $1k/trade "
          f"(${c0['per_year']:+,.0f}/yr), Sharpe {c0['sharpe']:.2f}, max DD ${c0['max_dd']:+,.0f}, from {first.date()}")
    print("T1", res["T1"]["positive_full_years"], "years positive; Sharpe 2018-21", round(s_a, 2), "2022-26", round(s_b, 2),
          "| exam window", round(res["T1"]["exam_window"]["mean_day"], 1), "/day ->", "PASS" if res["T1"]["pass"] else "FAIL")
    for y, v in by_year.items():
        print(f"   {y}: ${v['mean_day']:+6.1f}/day  Sharpe {v['sharpe']:+.2f}  DD ${v['max_dd']:+,.0f}  ({v['days']} days)")
    print("T2 half", round(t2["half"]["mean_day"], 1), "touch", round(c0["mean_day"], 1), "x1.5", round(t2["touch_x1.5"]["mean_day"], 1),
          "x2", round(t2["touch_x2"]["mean_day"], 1), "| miss 20% winners median", round(t2["miss_20pct_winners_mean_day_median"], 1),
          "->", "PASS" if t2["pass"] else "FAIL")
    t3 = res["T3"]
    print("T3 actual DD", round(t3["actual_max_dd"]), "| shuffled DD median/p95/p99", {k: round(v) for k, v in t3["shuffled_max_dd"].items()},
          "| 1-yr:", {k: round(v, 3) if k == "p_loss" else round(v) for k, v in t3["one_year"].items()}, "->", "PASS" if t3["pass"] else "FAIL")
    print("T4 neighbours:", {k: (round(nbrs[k]["mean_day"], 1), round(nbrs[k]["sharpe"], 2)) for k in nbrs}, "centre Sharpe", round(c["sharpe"], 2),
          "->", "PASS" if res["T4"]["pass"] else "FAIL")
    print("   alpha x threshold Sharpe:")
    for a in ALPHAS:
        print("   ", a, " ".join(f"{grid[f'a{a}_t{t}']['sharpe']:+.2f}" for t in THRESHOLDS))
    print("   per day", {k: round(v["sharpe"], 2) for k, v in one["per_day"].items()}, "wing", {k: round(v["sharpe"], 2) for k, v in one["wing"].items()})
    print("OVERALL", res["overall"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
