"""Direction stage A: does order flow or headline content make 30-60 min direction callable?

Same rows, targets and gate as odte/eval/direction.py, on the development
year only (SPX sessions 2025-09-02 .. 2026-09-04), with two new input groups:

  ES order flow (odte/data/es_flow.py; a minute ending at ts+60 is known then):
    of_imb5/15/30/60   (aggressive buy - sell) / total, last 5/15/30/60 min
    of_big15/60        the same for trades >= 20 lots
    of_imb_night       previous 16:00 -> 09:30
    of_vol15_rel       last-15-min volume / median of the same slot on the
                       previous 20 sessions
  Headlines (odte/data/news_titles.py): NEWS_FEATURES

Walk-forward inside the year: first 100 days training only, refit every 21
days. Models: logistic and LightGBM, with and without the new inputs.
Gate (pre-registered 2026-11-08): the step 1 gate applied to the "+new"
models only.

Usage:
  python3 -m odte.eval.direction_stage_a reports/expiry_edge/decisions-dev-gex-es.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.eval import direction as dr

ROOT = Path(__file__).resolve().parents[2]
YEAR = (pd.Timestamp("2025-09-02"), pd.Timestamp("2026-09-04"))
FLOW_FEATURES = ["of_imb5", "of_imb15", "of_imb30", "of_imb60", "of_big15", "of_big60",
                 "of_imb_night", "of_vol15_rel"]


def _utc(et) -> np.ndarray:
    return (pd.DatetimeIndex(et).tz_localize("America/New_York").tz_convert("UTC")
            .asi8 // 10**9)


def flow_features(rows: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    ts = flow["ts"].to_numpy(np.int64)
    c = {k: np.r_[0.0, np.cumsum(flow[k].to_numpy(float))] for k in ("buy", "sell", "big_buy", "big_sell")}

    def win(a_utc, b_utc, kind=""):
        i = np.searchsorted(ts, a_utc, side="left")
        j = np.searchsorted(ts, b_utc - 60, side="right")     # minute must have ended by b
        B, S = c[f"{kind}buy"][j] - c[f"{kind}buy"][i], c[f"{kind}sell"][j] - c[f"{kind}sell"][i]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(B + S > 0, (B - S) / (B + S), np.nan), B + S
    t = _utc(rows["t"])
    out = {}
    for w in (5, 15, 30, 60):
        out[f"of_imb{w}"], vol = win(t - 60 * w, t)
        if w == 15:
            vol15 = vol
    out["of_big15"], _ = win(t - 900, t, "big_")
    out["of_big60"], _ = win(t - 3600, t, "big_")
    day = pd.to_datetime(rows["day"]).dt.normalize()
    prev = day - pd.to_timedelta(np.where(day.dt.dayofweek == 0, 3, 1), unit="D")
    out["of_imb_night"], _ = win(_utc(prev + pd.Timedelta(hours=16)),
                                 _utc(day + pd.Timedelta(hours=9, minutes=30)))
    f = pd.DataFrame(out, index=rows.index)
    v = pd.DataFrame({"day": day, "slot": rows["t"].dt.strftime("%H:%M"), "v": vol15}, index=rows.index)
    v = v.sort_values("day")
    norm = v.groupby("slot")["v"].transform(lambda s: s.rolling(20, min_periods=5).median().shift(1))
    f["of_vol15_rel"] = (v["v"] / norm).reindex(rows.index)
    return f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions")
    ap.add_argument("--out", default=str(ROOT / "reports" / "direction_stage_a.json"))
    a = ap.parse_args()
    from odte.data.es_flow import load as load_flow
    from odte.data.news_titles import NEWS_FEATURES, features_for
    d = dr.add_targets(dr.load([a.decisions]))
    d = d[d["day"].between(*YEAR)].reset_index(drop=True)
    d = pd.concat([d, flow_features(d, load_flow()), features_for(d)], axis=1)
    base = dr.feature_cols(d)
    new = FLOW_FEATURES + NEWS_FEATURES
    cov = d[new].notna().mean().round(3).to_dict()
    print(f"{d['day'].nunique()} days, {len(d):,} rows; {len(base)} base + {len(new)} new inputs; "
          f"coverage {cov}", flush=True)
    dr.MIN_TRAIN_DAYS, dr.REFIT_DAYS = 100, 21
    preds = {}
    for h in dr.HORIZONS:
        for kind in ("logit", "lgbm"):
            preds[(kind, h)] = dr.walk_forward(d, base, h, kind)
            preds[(f"{kind}+new", h)] = dr.walk_forward(d, base + new, h, kind)
    res = dr.evaluate(d, preds)
    res["verdict"] = "PASS" if any(res[f"H{h}"][k]["passes"] for h in dr.HORIZONS
                                   for k in ("logit+new", "lgbm+new")) else "FAIL"
    res["coverage"] = cov
    Path(a.out).write_text(json.dumps(res, indent=2))
    for h in dr.HORIZONS:
        r = res[f"H{h}"]
        print(f"\nH={h} min: {r['n_rows']:,} rows / {r['n_days']} test days; up {r['up_share']:.1%}; "
              f"break-even ~{r['median_hurdle_bp']:.1f} bp")
        for k, v in r["rules"].items():
            print(f"   {k:10s} accuracy {v['accuracy']:.2%}  capture {v['capture_bp']:+.2f} bp")
        for k in ("logit+new", "lgbm+new"):
            v = r[k]
            print(f"   {k}: vs best rule accuracy {v['accuracy_minus_best_rule']:+.2%} "
                  f"[{v['ci'][0]:+.2%}, {v['ci'][1]:+.2%}], capture {v['capture_minus_best_rule_bp']:+.2f} bp "
                  f"[{v['capture_ci'][0]:+.2f}, {v['capture_ci'][1]:+.2f}]; top 20%: "
                  f"{v['top20pct_accuracy']:.1%}, {v['top20pct_capture_bp']:+.2f} bp vs break-even "
                  f"{v['top20pct_hurdle_bp']:.1f} bp -> {'PASS' if v['passes'] else 'fail'}")
    print(f"\nVERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
