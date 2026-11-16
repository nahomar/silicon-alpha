"""T1: does the leader time long straddles better than chance? (pre-registered in docs/master_plan.md)

Stratified AUC: positives and negatives are compared only within the same (index, decision time),
so time-of-day or index base rates cannot create apparent skill. 90% CI from a day-block bootstrap.

  python3 -m odte.research.skill
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from odte.research import meta

KEYS = ["day", "index", "hhmm"]
N_BOOT = 500


def strat_auc(score: np.ndarray, event: np.ndarray, strata: np.ndarray) -> float:
    u_sum, pairs = 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        y, x = event[m], score[m]
        npos = int(y.sum())
        nneg = len(y) - npos
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(x)
        u_sum += r[y].sum() - npos * (npos + 1) / 2
        pairs += npos * nneg
    return u_sum / pairs if pairs else np.nan


def pick_lift(pick: np.ndarray, event: np.ndarray, strata: np.ndarray) -> float:
    if pick.sum() == 0:
        return np.nan
    base = pd.Series(event.astype(float)).groupby(strata).transform("mean").to_numpy()
    return float(event[pick].mean() / base[pick].mean())


def boot(fn, days: np.ndarray, rng) -> list:
    """90% CI of fn(rows) resampling whole days."""
    by_day = pd.Series(np.arange(len(days))).groupby(days).apply(np.array).to_list()
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(by_day), len(by_day))
        vals.append(fn(np.concatenate([by_day[i] for i in pick])))
    return [float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))]


def run() -> dict:
    d = meta.load()
    p = d[KEYS].merge(pd.read_parquet(meta.OUT / "preds_leader.parquet"), on=KEYS, how="left", validate="one_to_one")
    preds = {k: p[k].to_numpy(float) for k in meta.TRADES}
    dec = meta.decide(d, preds, "half")
    y = d["y_long_straddle_half"].to_numpy(float)
    score = preds["long_straddle"]
    bench = -pd.to_numeric(d["iv_over_rv"], errors="coerce").to_numpy(float)
    ok = ~d["post"].to_numpy() & np.isfinite(score) & np.isfinite(y)
    strata = (d["index"] + " " + d["hhmm"]).to_numpy()
    pick = (dec["take"] & (dec["trade"] == "long_straddle")).to_numpy()
    rng = np.random.default_rng(0)
    out = {}
    for era, em in (("2022_26", d["day"] >= meta.EVAL_FROM), ("2018_21", d["day"] < meta.EVAL_FROM)):
        m = ok & em.to_numpy()
        days = d["day"].to_numpy()[m]
        S, B, Y, ST, PK = score[m], bench[m], y[m], strata[m], pick[m]
        Bf = np.where(np.isfinite(B), B, np.nanmedian(B))
        res = {"rows": int(m.sum()), "days": int(len(np.unique(days))), "straddle_picks": int(PK.sum())}
        for ev_name, ev in (("E1_profitable", Y > 0), ("E2_doubles", Y > 1000)):
            r = {"base_rate": float(ev.mean()),
                 "auc_model": strat_auc(S, ev, ST),
                 "auc_model_ci90": boot(lambda i: strat_auc(S[i], ev[i], ST[i]), days, rng),
                 "auc_benchmark": strat_auc(Bf, ev, ST),
                 "auc_benchmark_ci90": boot(lambda i: strat_auc(Bf[i], ev[i], ST[i]), days, rng),
                 "pick_lift": pick_lift(PK, ev, ST),
                 "pick_lift_ci90": boot(lambda i: pick_lift(PK[i], ev[i], ST[i]), days, rng)}
            res[ev_name] = r
        out[era] = res
    e1 = [out[e]["E1_profitable"]["auc_model_ci90"][0] for e in out]
    out["verdict"] = "PASS" if all(lo > 0.5 for lo in e1) else "FAIL"
    (meta.OUT / "skill_t1.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    r = run()
    for era in ("2022_26", "2018_21"):
        x = r[era]
        print(f"\n{era}: {x['rows']:,} decisions, {x['days']:,} days, {x['straddle_picks']:,} straddle picks")
        for ev in ("E1_profitable", "E2_doubles"):
            e = x[ev]
            print(f"  {ev:14s} base {e['base_rate']:.1%} | model AUC {e['auc_model']:.3f} {np.round(e['auc_model_ci90'], 3)}"
                  f" | benchmark AUC {e['auc_benchmark']:.3f} {np.round(e['auc_benchmark_ci90'], 3)}"
                  f" | picks lift {e['pick_lift']:.2f}x {np.round(e['pick_lift_ci90'], 2)}")
    print("\nT1 verdict:", r["verdict"])
