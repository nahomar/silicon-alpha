"""Edge #10: sell SPX only when the model ranks SPX the most overpriced of SPX/NDX/RUT.

Pre-registered 2026-11-10 (docs/expiry_edge_plan.md). Seen pair-trade development rows.

  python3 -m odte.research.xindex
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEC = ROOT / "reports" / "expiry_edge"
PRED = "pred6__ml_ridge+cal+ctx"
TODAY_COST, FEES4, BUDGET = 0.0169, 4 * 1.20, 1000.0
SPLIT = pd.Timestamp("2023-09-01")
OWN_MARGIN = 0.95
N_BOOT, SEED = 10_000, 0


def load(tag: str) -> pd.DataFrame:
    d = pd.read_parquet(DEC / f"decisions-pair-dev-{tag}.parquet")
    d = d[d["on6"] & d["tested6"]].copy()
    d["day"] = pd.to_datetime(d["day"])
    return d.set_index(["day", "hhmm"])


def spx_ror(d: pd.DataFrame, share: float) -> pd.Series:
    """Return on risk of selling the SPX fly, paying `share` of today's gap."""
    cm, W = d["fly_credit_mid"].astype(float), d["W"].astype(float)
    c = cm * (1 - share * TODAY_COST)
    intrinsic = cm - d["pnl_short_fly_mid"].astype(float) / 100
    risk = (W - c) * 100 + FEES4
    return ((c - intrinsic) * 100 - FEES4) / risk.where(risk > 0)


def rule_r1(p_spx: pd.Series, p_ndx: pd.Series, p_rut: pd.Series) -> pd.Series:
    return (p_spx < p_ndx) & (p_spx < p_rut)


def mean_ci(x, n=N_BOOT, seed=SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "days": int(len(x)), "total": float(x.sum())}


def main() -> int:
    s, n, r = load("spx"), load("ndx"), load("rut")
    idx = s.index.intersection(n.index).intersection(r.index)
    s = s.loc[idx]
    ok = s[PRED].notna() & n.loc[idx, PRED].notna() & r.loc[idx, PRED].notna()
    ok &= ~(s["min_since_event"] <= 60).fillna(False)
    s = s[ok]
    p_s, p_n, p_r = s[PRED], n.loc[s.index, PRED], r.loc[s.index, PRED]
    rules = {"R1_spx_richest": rule_r1(p_s, p_n, p_r), "always_sell": pd.Series(True, index=s.index),
             "own_model": p_s < OWN_MARGIN}
    days = s.index.get_level_values("day")
    res = {"rows": int(len(s)), "days": int(days.nunique()),
           "r1_share_of_rows": float(rules["R1_spx_richest"].mean())}
    for px, share in (("half", 0.5), ("touch", 1.0)):
        ror = spx_ror(s, share).fillna(0.0)
        daily = {k: (ror * v.astype(float) * BUDGET).groupby(days).sum() for k, v in rules.items()}
        out = {}
        for k, x in daily.items():
            out[k] = {**mean_ci(x), "first_half": float(x[x.index < SPLIT].sum()),
                      "second_half": float(x[x.index >= SPLIT].sum())}
        out["R1_minus_always"] = mean_ci((daily["R1_spx_richest"] - daily["always_sell"]).to_numpy())
        res[px] = out
    h = res["half"]
    res["passes"] = bool(h["R1_spx_richest"]["ci"][0] > 0 and h["R1_minus_always"]["ci"][0] > 0
                         and h["R1_spx_richest"]["first_half"] > 0 and h["R1_spx_richest"]["second_half"] > 0)
    p = ROOT / "reports" / "factory" / "xindex.json"
    p.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
