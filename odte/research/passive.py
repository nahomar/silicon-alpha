"""Edge #14: passive-only seller (pre-registered 2026-11-10, docs/expiry_edge_plan.md).

  python3 -m odte.research.passive
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 1000.0
PRED = "pred6__ml_ridge+cal+ctx"
N_BOOT, SEED = 10_000, 0


def passive_ror(d: pd.DataFrame) -> pd.Series:
    """Return on risk if the resting mid order filled (strict rule), else 0 (no trade)."""
    r = d["pnl_short_fly_midx"] / d["risk_short_fly_midx"].where(d["risk_short_fly_midx"] > 0)
    return r.where(d["fill_mid_short"].fillna(False).astype(bool), 0.0).fillna(0.0)


def mid_ror(d: pd.DataFrame) -> pd.Series:
    return (d["pnl_short_fly_midx"] / d["risk_short_fly_midx"].where(d["risk_short_fly_midx"] > 0)).fillna(0.0)


def mean_ci(x, n=N_BOOT, seed=SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "days": int(len(x)), "total": float(x.sum())}


def main() -> int:
    d = fx.load()
    d = d[~(d["min_since_event"] <= 60).fillna(False)]
    d = d[np.isfinite(d["pnl_short_fly_midx"]) & (d["risk_short_fly_midx"] > 0)].copy()
    days = d["day"]
    era1 = days < fx.ERA_SPLIT
    filled = d["fill_mid_short"].fillna(False).astype(bool)
    ev = (d[["ev_cpi", "ev_nfp", "ev_fomc", "ev_fomc_sep"]].fillna(0).sum(axis=1) > 0)
    model = (d[PRED] < 0.85) if PRED in d else pd.Series(False, index=d.index)
    strategies = {
        "passive": passive_ror(d),
        "always_half_gap": d["T_fly"].fillna(0.0),
        "passive_model": passive_ror(d).where(model, 0.0),
        "passive_events": passive_ror(d).where(ev, 0.0),
    }
    res = {"decisions": int(len(d)), "fill_rate": float(filled.mean()),
           "fill_rate_2016_21": float(filled[era1].mean()), "fill_rate_2022_26": float(filled[~era1].mean()),
           "adverse_selection_mid_ror": {"filled_rows": float(mid_ror(d)[filled].mean()),
                                         "all_rows": float(mid_ror(d).mean()),
                                         "unfilled_rows": float(mid_ror(d)[~filled].mean())}}
    for k, r in strategies.items():
        x = (r * BUDGET).groupby(days).sum()
        res[k] = {**mean_ci(x), "2016-21": float(x[x.index < fx.ERA_SPLIT].sum()),
                  "2022-26": float(x[x.index >= fx.ERA_SPLIT].sum())}
    p = res["passive"]
    res["passes"] = bool(p["ci"][0] > 0 and p["2016-21"] > 0 and p["2022-26"] > 0)
    (ROOT / "reports" / "factory" / "passive.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
