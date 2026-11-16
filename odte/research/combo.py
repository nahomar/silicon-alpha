"""Edge #16: one walk-forward model on every options signal (pre-registered 2026-11-11).

  python3 -m odte.research.combo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
EVAL_FROM, BUDGET = pd.Timestamp("2022-01-01"), 1000.0
MIN_DAYS, REFIT = 500, 63
N_BOOT, SEED = 10_000, 0


def walk_forward(d: pd.DataFrame, X: pd.DataFrame, y: pd.Series, model: str) -> np.ndarray:
    """Expanding-window predictions: each block of REFIT days is predicted by a model fitted on
    all earlier days only."""
    from odte.eval.expiry_edge import _make_model
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    out = np.full(len(d), np.nan)
    Xa, ya = X.to_numpy(float), y.to_numpy(float)
    for k in range(MIN_DAYS, len(days), REFIT):
        tr = (dd < days[k]) & np.isfinite(ya)
        te = (dd >= days[k]) & (dd < days[min(k + REFIT, len(days) - 1)] if k + REFIT < len(days) else dd >= days[k])
        m = _make_model(model).fit(Xa[tr], ya[tr])
        out[te] = m.predict(Xa[te])
    return out


def mean_ci(x, n=N_BOOT, seed=SEED) -> dict:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "days": int(len(x)), "total": float(x.sum())}


def main() -> int:
    from odte.live.model import feature_list
    from odte.research import term_structure as ts
    d = fx.load()
    d = d[~(d["min_since_event"] <= 60).fillna(False)].reset_index(drop=True)
    S = fx.signals(d)
    T = ts.build(d)[ts.SIGNALS]
    X = pd.concat([d[[c for c in feature_list() if c in d.columns]], S, T.reindex(d.index)], axis=1)
    X = X.loc[:, ~X.columns.duplicated()].apply(pd.to_numeric, errors="coerce")
    y = d["T_fly"]
    # touch version of the same trade (full gap) for the report
    cm = d["fly_credit_mid"].astype(float)
    intrinsic = cm - d["pnl_short_fly_mid"].astype(float) / 100
    c_touch = cm * (1 - 0.0169)
    risk_t = (d["W"] - c_touch) * 100 + 4.8
    touch = (((c_touch - intrinsic) * 100 - 4.8) / risk_t.where(risk_t > 0)).fillna(0.0)
    res = {"rows": int(len(d)), "features": int(X.shape[1])}
    days = d["day"]
    always = (y.fillna(0.0) * BUDGET).groupby(days).sum()
    for name in ("ml_lgbm", "ml_ridge"):
        pred = walk_forward(d, X, y, name)
        sell = np.isfinite(pred) & (pred > 0)
        tested = np.isfinite(pred)
        daily = (np.where(sell, y.fillna(0.0), 0.0) * BUDGET)
        daily = pd.Series(daily).groupby(days).sum()
        tdays = pd.Series(tested).groupby(days).any()
        daily, alw = daily[tdays], always[tdays]
        dtouch = pd.Series(np.where(sell, touch, 0.0) * BUDGET).groupby(days).sum()[tdays]
        post, pre = daily.index >= EVAL_FROM, daily.index < EVAL_FROM
        res[name] = {"share_sold": float(sell[tested].mean()),
                     "eval_2022_26": mean_ci(daily[post].to_numpy()),
                     "always_2022_26": mean_ci(alw[post].to_numpy()),
                     "minus_always_2022_26": mean_ci((daily[post] - alw[post]).to_numpy()),
                     "total_2016_21_walkforward": float(daily[pre].sum()),
                     "touch_total_2022_26": float(dtouch[post].sum()),
                     "oos_ic": fx.ic_test(pd.Series(pred[tested]), y[tested].reset_index(drop=True),
                                          days[tested].reset_index(drop=True))["ic"]}
    g = res["ml_lgbm"]
    res["passes"] = bool(g["eval_2022_26"]["ci"][0] > 0 and g["minus_always_2022_26"]["ci"][0] > 0
                         and g["total_2016_21_walkforward"] > 0)
    (ROOT / "reports" / "factory" / "combo.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
