"""Pair trades across 0DTE indices: sell the most overpriced, buy the least.

Each index's ridge model (expiry_edge, walk-forward) predicts realized /
implied for the rest of the day. At each fixed decision time, among the
indices quoted (SPX, NDX, RUT):
  short leg  the iron fly of the index with the LOWEST predicted ratio
  long leg   the iron fly of the index with the HIGHEST predicted ratio
  only if    predicted spread (high - low) >= MARGIN
Sizing: equal premium on both legs (n_i x mid fly price_i equal), scaled so the
pair's worst case (short leg max loss + long leg premium) is $1,000. On a
crash day both legs move the same way, so the long leg pays for most of the
short leg's loss.

Execution, both reported:
  touch    sell at the bid / buy at the ask (expiry_edge pnl_*_fly)
  passive  rest at mid, filled only if the market trades through it, else
           cross 5 min later (expiry_edge pnl_*_fly_midx)

Benchmarks (same rows, same sizing):
  random   random short / long index (mean over 20 seeds)
  carry    no model: short the index whose last 20 sessions' realized /
           implied was lowest, long the highest (known after each close)
  each index's plain always-short fly, $1k max loss (for reference)

Usage:
  python3 -m odte.eval.pairs --tags pair-dev-spx pair-dev-ndx pair-dev-rut
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEC = ROOT / "reports" / "expiry_edge"
MODEL = "ml_ridge+cal+ctx"
MARGIN = 0.15
BUDGET = 1000.0
COLS = ["fly_credit_mid", "ratio", "pnl_short_fly", "pnl_long_fly", "risk_short_fly", "risk_long_fly",
        "pnl_short_fly_midx", "pnl_long_fly_midx", "risk_short_fly_midx", "risk_long_fly_midx",
        "pnl_short_fly_t5", "pnl_long_fly_t5", "risk_short_fly_t5", "risk_long_fly_t5",
        "fill_mid_short", "fill_mid_long"]


def load(tags) -> dict:
    """{index: frame indexed by (day, hhmm)} with fixed6, tested rows only."""
    out = {}
    for tag in tags:
        d = pd.read_parquet(DEC / f"decisions-{tag}.parquet")
        d = d[d["on6"] & d["tested6"]].copy()
        d["pred"] = d[f"pred6__{MODEL}"]
        d["day"] = pd.to_datetime(d["day"])
        # Passive execution, as documented: mid only where the market traded
        # through it (fill_mid_*), otherwise cross 5 minutes later (t5).
        for side in ("short", "long"):
            f = d[f"fill_mid_{side}"].fillna(False).astype(bool)
            for c in ("pnl", "risk"):
                d[f"{c}_{side}_fly_pas"] = np.where(f, d[f"{c}_{side}_fly_midx"], d[f"{c}_{side}_fly_t5"])
        cols = ["pred", *COLS] + [f"{c}_{s}_fly_pas" for c in ("pnl", "risk") for s in ("short", "long")]
        out[tag.split("-")[-1]] = d.set_index(["day", "hhmm"])[cols]
    return out


def carry_scores(frames: dict) -> dict:
    """Each index's mean realized/implied over its previous 20 sessions."""
    out = {}
    for k, f in frames.items():
        daily = f["ratio"].groupby(level="day").mean()
        out[k] = daily.rolling(20, min_periods=10).mean().shift(1)
    return out


def pair_pnl(frames: dict, choose, exe: str = "touch") -> pd.Series:
    """Daily pair P&L. choose(row_scores) -> (short_index, long_index) or None."""
    suf = "" if exe == "touch" else "_pas"
    keys = sorted(frames)
    idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
    pnl = {}
    for key in idx:
        avail = {k: frames[k].loc[key] for k in keys if key in frames[k].index}
        avail = {k: r for k, r in avail.items()
                 if np.isfinite(r[f"pnl_short_fly{suf}"]) and np.isfinite(r[f"pnl_long_fly{suf}"])
                 and r["fly_credit_mid"] > 0}
        if len(avail) < 2:
            continue
        pick = choose(key, avail)
        if pick is None:
            continue
        s, l = avail[pick[0]], avail[pick[1]]
        per_k = (s[f"risk_short_fly{suf}"] / s["fly_credit_mid"]
                 + l[f"risk_long_fly{suf}"] / l["fly_credit_mid"])
        K = BUDGET / per_k                                   # equal premium per leg
        v = K / s["fly_credit_mid"] * s[f"pnl_short_fly{suf}"] + K / l["fly_credit_mid"] * l[f"pnl_long_fly{suf}"]
        pnl[key] = v
    s = pd.Series(pnl, dtype=float)
    if s.empty:
        return s
    return s.groupby(level=0).sum()


def model_choice(margin=MARGIN):
    def f(key, avail):
        p = {k: r["pred"] for k, r in avail.items() if np.isfinite(r["pred"])}
        if len(p) < 2:
            return None
        lo, hi = min(p, key=p.get), max(p, key=p.get)
        return (lo, hi) if p[hi] - p[lo] >= margin else None
    return f


def carry_choice(scores):
    def f(key, avail):
        c = {k: scores[k].get(key[0], np.nan) for k in avail}
        c = {k: v for k, v in c.items() if np.isfinite(v)}
        if len(c) < 2:
            return None
        return min(c, key=c.get), max(c, key=c.get)
    return f


def random_choice(seed):
    rng = np.random.default_rng(seed)

    def f(key, avail):
        k = list(avail)
        a, b = rng.choice(len(k), 2, replace=False)
        return k[a], k[b]
    return f


def summary(daily: pd.Series) -> dict:
    from odte.eval.panel_stats import day_block_bootstrap
    if daily.empty:
        return {"days": 0}
    b = day_block_bootstrap(daily.to_numpy())
    cum = daily.sort_index().cumsum()
    return {"days": int(len(daily)), "total": float(daily.sum()), "mean_day": float(daily.mean()),
            "sharpe": b["sharpe"], "sharpe_ci": [b["ci_low"], b["ci_high"]],
            "max_drawdown": float((cum - cum.cummax()).min()), "worst_day": float(daily.min())}


def run(tags, margin=MARGIN) -> dict:
    from odte.eval.expiry_edge import paired_diff
    fr = load(tags)
    res = {"indices": {k: int(len(f)) for k, f in fr.items()}}
    for exe in ("touch", "passive"):
        m = pair_pnl(fr, model_choice(margin), exe)
        c = pair_pnl(fr, carry_choice(carry_scores(fr)), exe)
        rnd = pd.concat([pair_pnl(fr, random_choice(s), exe) for s in range(20)], axis=1).mean(axis=1)
        r = {"model": summary(m), "carry": summary(c), "random": summary(rnd),
             "model_vs_random": paired_diff(m, rnd.reindex(m.index).fillna(0)),
             "model_vs_carry": paired_diff(m, c.reindex(m.index).fillna(0))}
        suf = "" if exe == "touch" else "_pas"
        for k, f in fr.items():
            one = (f[f"pnl_short_fly{suf}"] / f[f"risk_short_fly{suf}"] * BUDGET).groupby(level="day").sum()
            r[f"always_short_{k}"] = summary(one)
        res[exe] = r
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = run(a.tags, a.margin)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print("rows per index:", res["indices"])
    for exe in ("touch", "passive"):
        print(f"\n== {exe} execution ($1k worst case per pair / per trade)")
        for k, v in res[exe].items():
            if "days" in v and v["days"]:
                print(f"  {k:20s} {v['days']:4d} days  total ${v['total']:>9,.0f}  Sharpe {v['sharpe']:+.2f} "
                      f"[{v['sharpe_ci'][0]:+.2f}, {v['sharpe_ci'][1]:+.2f}]  max DD ${v['max_drawdown']:,.0f}  "
                      f"worst ${v['worst_day']:,.0f}")
            elif "mean" in v:
                print(f"  {k:20s} {v['mean']:+.1f}/day  CI [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
