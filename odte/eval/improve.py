"""Model improvement round: three candidates against the bot's current model.

All on seen SPX data, walk-forward by day, scored at today's cost entering
at half the bid-ask gap (pre-registered in docs/expiry_edge_plan.md, 2026-11-09).

  C0  current bot: ridge on realized/implied (55 inputs), sell if pred < 0.85
  C1  profit target: ridge predicts the half-gap return on risk; sell if > 0
  C2  danger detector: logistic P(half-gap return on risk <= -0.5);
      sell unless P > 2 x training base rate
  C3  small model: C0's rule on 22 hand-picked inputs
  always_sell (reference)
Every strategy keeps the no-short-within-60-min-after-an-event rule.

  python3 -m odte.eval.improve
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / "reports" / "expiry_edge" / "decisions-lockbox-odte_0dte-2021H1.parquet",
         ROOT / "reports" / "expiry_edge" / "decisions-dev-gex-es.parquet"]
SEALED = [("2021-07-01", "2022-05-13"), ("2026-09-08", "2100-01-01")]
TODAY_COST, FEES4, BUDGET = 0.0169, 4 * 1.20, 1000.0
MIN_TRAIN, REFIT = 250, 21
SMALL = ["ev_fomc", "ev_cpi", "ev_nfp", "events_ahead", "minutes_to_next_event", "vix_prev",
         "vix1d_over_vix", "vix_chg", "spx_ret_prev", "spx_rv5", "iv_vs_vix1d", "megacap_earnings"]


def load(files=FILES, allow_sealed=False) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d["on6"]].copy()
    d["day"] = pd.to_datetime(d["day"])
    for lo, hi in SEALED:
        if d["day"].between(lo, hi).any() and not allow_sealed:
            raise SystemExit("REFUSED: sealed days present")
    d = d.drop_duplicates(["day", "hhmm"]).sort_values(["day", "hhmm"]).reset_index(drop=True)
    cm, W = d["fly_credit_mid"].to_numpy(float), d["W"].to_numpy(float)
    intrinsic = cm - d["pnl_short_fly_mid"].to_numpy(float) / 100
    for name, gap in (("half", 0.5 * TODAY_COST), ("touch", TODAY_COST)):
        c = cm * (1 - gap)
        pnl = (c - intrinsic) * 100 - FEES4
        risk = (W - c) * 100 + FEES4
        d[f"ror_{name}"] = np.where(risk > 0, pnl / risk, np.nan)
    d["post"] = (d["min_since_event"] <= 60).fillna(False).to_numpy()
    return d[np.isfinite(d["ror_half"])].reset_index(drop=True)


def walk_forward(d, cols, y, kind) -> np.ndarray:
    from odte.eval.expiry_edge import _make_model
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    X = d[cols].to_numpy(float)
    out = np.full(len(d), np.nan)
    for k in range(MIN_TRAIN, len(days), REFIT):
        tr = (dd < days[k]) & np.isfinite(y)
        te = dd >= days[k]
        if k + REFIT < len(days):
            te &= dd < days[k + REFIT]
        if kind == "ridge":
            m = _make_model("ml_ridge").fit(X[tr], y[tr])
            out[te] = m.predict(X[te])
        else:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=1000)).fit(X[tr], y[tr].astype(int))
            p = m.predict_proba(X[te])[:, 1]
            out[te] = p / max(float(y[tr].mean()), 1e-6)           # as a multiple of the base rate
    return out


def daily(d, sell, col) -> pd.Series:
    pnl = np.where(sell, d[col].to_numpy() * BUDGET, 0.0)
    tested = d["tested"].to_numpy()
    return pd.Series(pnl[tested]).groupby(d["day"].to_numpy()[tested]).sum()


def stats(x: pd.Series) -> dict:
    from odte.eval.panel_stats import day_block_bootstrap
    b = day_block_bootstrap(x.to_numpy())
    cum = x.cumsum()
    return {"total": float(x.sum()), "sharpe": b["sharpe"], "sharpe_ci": [b["ci_low"], b["ci_high"]],
            "max_dd": float((cum - cum.cummax()).min())}


def main() -> int:
    from odte.eval.expiry_edge import FEATURES, paired_diff
    from odte.live.model import feature_list
    d = load()
    full = feature_list()
    small = list(FEATURES) + SMALL
    days = np.array(sorted(d["day"].unique()))
    d["tested"] = d["day"] >= days[MIN_TRAIN]
    print(f"{d['day'].nunique()} days, {len(d):,} rows; tested from {days[MIN_TRAIN]:%Y-%m-%d}", flush=True)
    ok = ~d["post"].to_numpy()
    ratio = d["ratio"].clip(upper=4.0).to_numpy()
    p0 = walk_forward(d, full, ratio, "ridge")
    p1 = walk_forward(d, full, d["ror_half"].to_numpy(), "ridge")
    p2 = walk_forward(d, full, (d["ror_half"] <= -0.5).astype(float).to_numpy(), "logit")
    p3 = walk_forward(d, small, ratio, "ridge")
    sells = {"always_sell": ok.copy(),
             "C0_current": ok & (p0 < 0.85),
             "C1_profit": ok & (p1 > 0),
             "C2_danger": ok & np.isfinite(p2) & (p2 <= 2.0),
             "C3_small": ok & (p3 < 0.85)}
    res = {}
    for cost in ("half", "touch"):
        ser = {k: daily(d, v, f"ror_{cost}") for k, v in sells.items()}
        r = {}
        for k, x in ser.items():
            r[k] = stats(x)
            r[k]["trades"] = int((sells[k] & d["tested"].to_numpy()).sum())
            r[k]["vs_C0"] = paired_diff(x, ser["C0_current"])
            r[k]["vs_always"] = paired_diff(x, ser["always_sell"])
            for era, (lo, hi) in {"2017-2021": ("2016-01-01", "2021-12-31"),
                                  "2022-2026": ("2022-01-01", "2026-12-31")}.items():
                xs = x[(x.index >= pd.Timestamp(lo)) & (x.index <= pd.Timestamp(hi))]
                r[k][era] = float(xs.sum())
        res[cost] = r
    cands = [k for k in ("C1_profit", "C2_danger", "C3_small")
             if res["half"][k]["vs_C0"]["mean"] > 0 and res["half"][k]["vs_always"]["mean"] > 0]
    res["winner"] = max(cands, key=lambda k: res["half"][k]["sharpe"]) if cands else "C0_current"
    (ROOT / "reports" / "improve_dev.json").write_text(json.dumps(res, indent=2, default=float))
    for cost in ("half", "touch"):
        print(f"\n== {cost} of today's gap ($1k worst case per trade)")
        for k, v in res[cost].items():
            print(f"  {k:12s} {v['trades']:5d} trades  total ${v['total']:>9,.0f}  Sharpe {v['sharpe']:+.2f} "
                  f"[{v['sharpe_ci'][0]:+.2f}, {v['sharpe_ci'][1]:+.2f}]  maxDD ${v['max_dd']:,.0f}  "
                  f"vs C0 {v['vs_C0']['mean']:+.1f}/day  vs always {v['vs_always']['mean']:+.1f}/day  "
                  f"| 2017-21 ${v['2017-2021']:,.0f}  2022-26 ${v['2022-2026']:,.0f}")
    print(f"\nWINNER (pre-registered rule): {res['winner']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
