"""Edge #11: take profit at 50% of the credit (pre-registered 2026-11-10, docs/expiry_edge_plan.md).

  python3 -m odte.research.early_exit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
CHAINS = ROOT / "data" / "odte_0dte"
TODAY_COST, FEES4, BUDGET = 0.0169, 4 * 1.20, 1000.0
TP, SL = 0.5, 2.0
N_BOOT, SEED = 10_000, 0


def first_hit(values: np.ndarray, level: float, above: bool) -> int:
    """Index of the first value at/below (or at/above) level, -1 if none. NaNs never hit."""
    v = np.asarray(values, float)
    hit = (v >= level) if above else (v <= level)
    hit &= np.isfinite(v)
    return int(np.argmax(hit)) if hit.any() else -1


def fly_paths(grid: dict, t: pd.Timestamp, t_last: pd.Timestamp, K: float, W: float):
    """Buy-back value of the short fly (mid, touch) for minutes strictly after t up to t_last."""
    from odte.eval.expiry_edge import _mids
    legs = [("C", K, 1), ("P", K, 1), ("C", K + W, -1), ("P", K - W, -1)]
    cols = [(r, k) for r, k, _ in legs]
    mids, bid, ask = _mids(grid), grid["bid"], grid["ask"]
    if not all(c in mids.columns for c in cols):
        return None
    win = mids.index[(mids.index > t) & (mids.index <= t_last)]
    m = mids.loc[win, cols].to_numpy(float)
    b, a = bid.loc[win, cols].to_numpy(float), ask.loc[win, cols].to_numpy(float)
    sign = np.array([s for _, _, s in legs], float)
    value_mid = (m * sign).sum(axis=1)
    value_touch = np.where(sign > 0, a, -b).sum(axis=1)       # buy the body at the ask, sell wings at the bid
    ok = np.isfinite(m).all(axis=1) & np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    value_mid[~ok] = np.nan                                    # a minute counts only with all 8 quotes
    return win, value_mid, value_touch


def exit_pnl(credit_mid, credit_touch, W, intrinsic, path, rule: str):
    """(pnl $, risk $, exited) for one decision under rule 'hold' | 'E1' | 'E2' | 'E12'."""
    c = credit_mid * (1 - 0.5 * TODAY_COST)
    risk = (W - c) * 100 + FEES4
    hold = (c - intrinsic) * 100 - FEES4
    if rule == "hold" or path is None:
        return hold, risk, False
    _, vm, vt = path
    hits = []
    if rule in ("E1", "E12"):
        hits.append(first_hit(vm, TP * credit_mid, above=False))
    if rule in ("E2", "E12"):
        hits.append(first_hit(vm, SL * credit_mid, above=True))
    hits = [h for h in hits if h >= 0]
    if not hits:
        return hold, risk, False
    i = min(hits)
    q_gap = credit_mid - credit_touch
    k = (TODAY_COST * credit_mid / q_gap) if q_gap > 0 else 1.0
    cost = vm[i] + 0.5 * max(vt[i] - vm[i], 0.0) * k
    return (c - cost) * 100 - 2 * FEES4, risk, True


def mean_ci(x, n=N_BOOT, seed=SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "days": int(len(x)), "total": float(x.sum())}


def main() -> int:
    from odte.eval.expiry_edge import minute_grid
    d = fx.load()
    d = d[~(d["min_since_event"] <= 60).fillna(False)]
    d = d[(d["fly_credit_mid"] > 0) & np.isfinite(d["W"])]
    rules = ("hold", "E1", "E2", "E12")
    rec = []
    for day, g in d.groupby("day"):
        p = CHAINS / f"{day:%Y-%m-%d}.parquet"
        grid = minute_grid(pd.read_parquet(p)) if p.exists() else None
        for _, r in g.iterrows():
            t = day + pd.Timedelta(hours=int(r["hhmm"][:2]), minutes=int(r["hhmm"][3:]))
            t_last = (t + pd.Timedelta(minutes=float(r["minutes_to_close"])) - pd.Timedelta(minutes=1)).floor("min")
            path = fly_paths(grid, t, t_last, float(r["K"]), float(r["W"])) if grid is not None else None
            cm = float(r["fly_credit_mid"])
            intrinsic = cm - float(r["pnl_short_fly_mid"]) / 100
            row = {"day": day}
            for rule in rules:
                pnl, risk, ex = exit_pnl(cm, float(r["fly_credit"]), float(r["W"]), intrinsic, path, rule)
                row[rule] = pnl / risk * BUDGET if risk > 0 else np.nan
                row[f"x_{rule}"] = ex
            rec.append(row)
    x = pd.DataFrame(rec)
    daily = x.groupby("day")[list(rules)].sum()
    era1 = daily.index < fx.ERA_SPLIT
    res = {"decisions": int(len(x)), "days": int(len(daily)),
           "exit_rate": {k: float(x[f"x_{k}"].mean()) for k in rules if k != "hold"}}
    for k in rules:
        res[k] = {**mean_ci(daily[k]), "2016-21": float(daily.loc[era1, k].sum()),
                  "2022-26": float(daily.loc[~era1, k].sum())}
    for k in ("E1", "E2", "E12"):
        res[f"{k}_minus_hold"] = mean_ci((daily[k] - daily["hold"]).to_numpy())
    e = res["E1"]
    res["passes"] = bool(e["ci"][0] > 0 and e["2016-21"] > 0 and e["2022-26"] > 0)
    (ROOT / "reports" / "factory" / "early_exit.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
