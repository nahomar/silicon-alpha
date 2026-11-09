"""Step 1 of the option-buying idea: can we call the S&P's direction 30-60 minutes ahead?

Buying a call or put and selling it later the same day only works if the
direction of the next 30-60 minutes is predictable, by enough to pay the
bid-ask (twice) and the hour's time decay. This checks the first part on the
underlying alone; step 2 (real option prices) runs only if this passes.

Data: development days only (already seen):
  * SPX 2016-01 .. 2021-06 (lockbox blocks already opened)
  * SPX 2022-05-16 .. 2026-09-04 (the dev period)
Sealed and NOT used: SPX 2021-07 .. 2022-05-13 (the last two lockbox blocks,
kept for this strategy's real test), the exam (2026-09-08 on), NDX and RUT.

Rows: every 15 minutes, 10:00 .. 15:00. Price = the forward implied by
put-call parity in the 0DTE chain at that minute (moves with SPX).
Target: sign of log(F[t+H] / F[t]), H = 30 and 60 min (the 4 PM settlement when
t+H is the close).
Features, all known at t: today's option and price state (move so far, last
15/30/60-min moves, realized and implied vol, skew, gap), calendar, market
backdrop (VIX family, rates, dollar, oil, gold, credit, BTC), ES overnight,
dealer gamma, time since events, and news/social counts (GDELT, Truth Social,
Kalshi; 2022 on only).
Models: logistic regression and LightGBM, walk-forward by day (train on all
earlier days, refit every 21 days, first 250 days training only).

Pre-registered gate (written before the run):
  PASS if, for H = 30 or 60, logistic or LightGBM beats the best simple rule
  (always-up, momentum, reversal) on BOTH accuracy and average captured move
  (bp, prediction sign x realized move), each with a 99% day-resampled CI
  above 0. 99% rather than 95% because 4 model x horizon pairs are tried.
  The break-even needed by an option buyer at today's costs is reported
  alongside. If no pair passes: stop; buying same-day options on these
  inputs has no basis. Plan B (user): a sequence transformer on minute paths.

Usage:
  python3 -m odte.eval.direction \\
      reports/expiry_edge/decisions-lockbox-odte_0dte-2021H1.parquet \\
      reports/expiry_edge/decisions-dev-gex-es.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEALED = [(pd.Timestamp("2021-07-01"), pd.Timestamp("2022-05-13")),
          (pd.Timestamp("2026-09-08"), pd.Timestamp("2100-01-01"))]
HORIZONS = (30, 60)
MIN_TRAIN_DAYS, REFIT_DAYS = 250, 21
TODAY_SPREAD = 0.02          # ATM straddle bid-ask / mid today (2025-26: 1.4-1.9%)
N_BOOT, CI = 2000, 0.99


def load(paths) -> pd.DataFrame:
    parts = []
    for p in paths:
        d = pd.read_parquet(p)
        parts.append(d[d["trigger"] == "grid"].drop(columns=[c for c in d if c.startswith("act6__")
                                                              or c == "tested6"]))
    d = pd.concat(parts, ignore_index=True)
    d["day"] = pd.to_datetime(d["day"])
    for lo, hi in SEALED:
        bad = d["day"].between(lo, hi)
        if bad.any():
            raise SystemExit(f"REFUSED: sealed days present ({d.loc[bad, 'day'].min():%Y-%m-%d} ..)")
    return d.drop_duplicates(["day", "t"]).sort_values(["day", "t"]).reset_index(drop=True)


def add_targets(d: pd.DataFrame) -> pd.DataFrame:
    """Lagged moves (features) and forward moves (targets), within each day."""
    d = d.copy()
    lf = np.log(d["F"].astype(float))
    key = d.set_index(["day", "t"]).index
    pos = pd.Series(np.arange(len(d)), index=key)
    for m in (15, 30, 60):
        j = pos.reindex(list(zip(d["day"], d["t"] - pd.Timedelta(minutes=m)))).to_numpy()
        ok = ~np.isnan(j)
        v = np.full(len(d), np.nan)
        v[ok] = lf.to_numpy()[ok] - lf.to_numpy()[j[ok].astype(int)]
        d[f"ret{m}"] = v * 1e4
    settle = np.log(d["settle"].astype(float)).to_numpy()
    for h in HORIZONS:
        j = pos.reindex(list(zip(d["day"], d["t"] + pd.Timedelta(minutes=h)))).to_numpy()
        fwd = np.full(len(d), np.nan)
        ok = ~np.isnan(j)
        fwd[ok] = lf.to_numpy()[j[ok].astype(int)]
        at_close = np.isclose(d["minutes_to_close"].to_numpy(float), h)
        fwd[at_close] = settle[at_close]
        d[f"fwd{h}"] = (fwd - lf.to_numpy()) * 1e4
        # break-even move for an ATM call/put buyer held h minutes at today's
        # cost: round-trip spread plus the hour's decay, divided by delta 0.5
        mtc = d["minutes_to_close"].to_numpy(float)
        decay = d["straddle_mid"] * (1 - np.sqrt(np.clip(1 - h / mtc, 0, 1)))
        d[f"hurdle{h}"] = (TODAY_SPREAD * d["straddle_mid"] + decay) / d["F"] * 1e4
    return d


def feature_cols(d: pd.DataFrame) -> list:
    from odte.data.calendar import CAL_FEATURES
    from odte.data.context import CTX_FEATURES
    from odte.data.futures import ES_FEATURES
    from odte.data.gamma import GEX_FEATURES
    from odte.eval.expiry_edge import FEATURES
    news = ["kx_move_ratio", "kx_down_prob", "fed_entropy_next", "fed_entropy_chg5", "tr_n_night",
            "tr_n_2h", "tr_mkt_today", "tr_fear_today", "gd_market_1h", "gd_tariff_1h", "gd_fed_1h",
            "gd_geo_1h", "gd_all_night", "btc_ret_night", "btc_rv_night", "btc_ret_1h", "btc_rv_1h"]
    cols = [*FEATURES, "ret15", "ret30", "ret60", "min_since_event", "move_since_event",
            *CAL_FEATURES, *CTX_FEATURES, "iv_vs_vix1d", *ES_FEATURES, *GEX_FEATURES, *news]
    return [c for c in dict.fromkeys(cols) if c in d]


def _model(kind):
    if kind == "logit":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.03,
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1)


def walk_forward(d: pd.DataFrame, cols, h: int, kind: str) -> np.ndarray:
    """P(up over the next h minutes), predicted only from earlier days."""
    days = np.array(sorted(d["day"].unique()))
    y = d[f"fwd{h}"].to_numpy()
    p = np.full(len(d), np.nan)
    X = d[cols].to_numpy(float)
    dd = d["day"].to_numpy()
    for k in range(MIN_TRAIN_DAYS, len(days), REFIT_DAYS):
        tr = (dd < days[k]) & np.isfinite(y) & (y != 0)
        te = dd >= days[k]
        if k + REFIT_DAYS < len(days):
            te &= dd < days[k + REFIT_DAYS]
        m = _model(kind).fit(X[tr], (y[tr] > 0).astype(int))
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def _boot(day, a, b, seed=0):
    """Mean of (a - b) per row and its CI, resampling whole days."""
    df = pd.DataFrame({"day": day, "x": a - b}).groupby("day")["x"].agg(["sum", "count"])
    s, n = df["sum"].to_numpy(), df["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(df), size=(N_BOOT, len(df)))
    m = s[idx].sum(1) / n[idx].sum(1)
    q = (1 - CI) / 2
    return float(s.sum() / n.sum()), [float(np.quantile(m, q)), float(np.quantile(m, 1 - q))]


def evaluate(d: pd.DataFrame, preds: dict) -> dict:
    out = {}
    for h in HORIZONS:
        y = d[f"fwd{h}"].to_numpy()
        lag = d["ret30" if h == 30 else "ret60"].to_numpy()
        valid = np.isfinite(y) & (y != 0) & np.isfinite(preds[("logit", h)]) & np.isfinite(lag)
        r = {"n_rows": int(valid.sum()), "n_days": int(d.loc[valid, "day"].nunique()),
             "up_share": float((y[valid] > 0).mean()),
             "mean_abs_move_bp": float(np.abs(y[valid]).mean()),
             "median_hurdle_bp": float(np.nanmedian(d[f"hurdle{h}"].to_numpy()[valid]))}
        sign = {"always_up": np.ones(valid.sum()), "momentum": np.sign(lag[valid]),
                "reversal": -np.sign(lag[valid])}
        for kind in ("logit", "lgbm"):
            sign[kind] = np.where(preds[(kind, h)][valid] > 0.5, 1.0, -1.0)
        yv, day = y[valid], d["day"].to_numpy()[valid]
        acc = {k: (s == np.sign(yv)).astype(float) for k, s in sign.items()}
        cap = {k: s * yv for k, s in sign.items()}
        r["rules"] = {k: {"accuracy": float(acc[k].mean()), "capture_bp": float(cap[k].mean())}
                      for k in sign}
        best_acc = max(("always_up", "momentum", "reversal"), key=lambda k: acc[k].mean())
        best_cap = max(("always_up", "momentum", "reversal"), key=lambda k: cap[k].mean())
        r["best_rule_accuracy"], r["best_rule_capture"] = best_acc, best_cap
        for kind in ("logit", "lgbm"):
            da, ca = _boot(day, acc[kind], acc[best_acc])
            dc, cc = _boot(day, cap[kind], cap[best_cap])
            conf = np.abs(preds[(kind, h)][valid] - 0.5)
            top = conf >= np.quantile(conf, 0.8)
            r[kind] = {"accuracy_minus_best_rule": da, "ci": ca,
                       "capture_minus_best_rule_bp": dc, "capture_ci": cc,
                       "top20pct_accuracy": float(acc[kind][top].mean()),
                       "top20pct_capture_bp": float(cap[kind][top].mean()),
                       "top20pct_hurdle_bp": float(np.nanmedian(d[f"hurdle{h}"].to_numpy()[valid][top])),
                       "passes": bool(ca[0] > 0 and cc[0] > 0)}
        out[f"H{h}"] = r
    out["verdict"] = "PASS" if any(out[f"H{h}"][k]["passes"] for h in HORIZONS
                                   for k in ("logit", "lgbm")) else "FAIL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "reports" / "direction_step1.json"))
    a = ap.parse_args()
    d = add_targets(load(a.decisions))
    cols = feature_cols(d)
    print(f"{d['day'].nunique()} days, {len(d):,} rows, {len(cols)} features", flush=True)
    preds = {}
    for h in HORIZONS:
        for kind in ("logit", "lgbm"):
            preds[(kind, h)] = walk_forward(d, cols, h, kind)
            print(f"  fitted {kind} H={h}", flush=True)
    res = evaluate(d, preds)
    res["features"] = cols
    Path(a.out).write_text(json.dumps(res, indent=2))
    for h in HORIZONS:
        r = res[f"H{h}"]
        print(f"\nH={h} min: {r['n_rows']:,} rows / {r['n_days']} days; up {r['up_share']:.1%}; "
              f"avg |move| {r['mean_abs_move_bp']:.1f} bp; option break-even ~{r['median_hurdle_bp']:.1f} bp")
        for k, v in r["rules"].items():
            print(f"   {k:10s} accuracy {v['accuracy']:.2%}  capture {v['capture_bp']:+.2f} bp")
        for kind in ("logit", "lgbm"):
            v = r[kind]
            print(f"   {kind}: vs best rule accuracy {v['accuracy_minus_best_rule']:+.2%} "
                  f"[{v['ci'][0]:+.2%}, {v['ci'][1]:+.2%}], capture {v['capture_minus_best_rule_bp']:+.2f} bp "
                  f"[{v['capture_ci'][0]:+.2f}, {v['capture_ci'][1]:+.2f}]; most confident 20%: accuracy "
                  f"{v['top20pct_accuracy']:.1%}, capture {v['top20pct_capture_bp']:+.2f} bp vs break-even "
                  f"{v['top20pct_hurdle_bp']:.1f} bp -> {'PASS' if v['passes'] else 'fail'}")
    print(f"\nVERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
