"""Phase 5: the one-time exam of the frozen model (docs/master_plan.md).

Everything that decides the result is fixed here, in the freeze commit:

  model    ridge, alpha 1000 (median impute -> standardise -> ridge), one model per trade,
           trained on ALL development rows (day < 2026-09-08), never on post-event rows
  inputs   groups calendar, gex, term, flow, identity (exact column list taken from the
           development matrix; every column must exist in the exam rows)
  menu     iron fly, iron condor, put credit spread, call credit spread, or no trade
           (long straddles excluded per T1)
  rule     take the trade with the highest predicted return if it is > 0; never within
           60 minutes after a scheduled release
  cost     half of today's gap (primary); the touch is reported
  sample   SPX, NDX, RUT, every 30 minutes 10:00-15:30 ET, sessions 2026-09-08 .. 2026-11-06

Outcome (half-gap, pooled daily P&L = sum over indexes and decisions, $1k worst case each):
  STRONG PASS  mean > 0, 90% day-bootstrap CI lower bound > 0, and mean > always-sell's
  WEAK PASS    mean > 0 and mean > always-sell's, CI includes 0
  FAIL         otherwise
"always-sell" = an iron fly at every allowed decision; it is also the best single always-trade
strategy in development (2022-26), so it serves as "the best single strategy" too.
Importance: each input group is removed in turn, the model refit on development data, and the
change in exam mean reported.

  python3 -m odte.research.exam dryrun      # same pipeline on the last 44 development days
  EXAM_UNLOCK=<freeze commit> python3 -m odte.research.exam run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import exam_guard as eg
from odte.research import library, meta

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "exam"
GROUPS = ["calendar", "gex", "term", "flow", "identity"]
ABLATE = ["calendar", "gex", "term", "flow"]
MENU = ["fly", "condor", "put_spread", "call_spread"]
MODEL = "ridge_a1000"
N_BOOT = 10000
GEX_COLS = ("gex_0dte", "gex_flip_dist", "gex_net", "maxoi_dist", "oi_pcr_0dte")


def sessions_between(a, b) -> list:
    s = pd.DatetimeIndex(pd.read_parquet(ROOT / "data" / "calendar" / "sessions.parquet")["session"])
    return [d for d in s if pd.Timestamp(a) <= d <= pd.Timestamp(b)]


def build_rows(days: list) -> pd.DataFrame:
    """Decision rows for `days`, through the same code as the development matrix (library rows,
    SPX term structure, identity, post flag). One warm-up session before the window carries
    prev_ratio exactly as in the development build; it is dropped afterwards."""
    from odte.research import term_structure as ts
    days = sorted(pd.to_datetime(days))
    parts = []
    for sym in meta.INDEXES:
        R = float(pd.read_parquet(meta.LIB / f"library_{sym}.parquet", columns=["R_today"])["R_today"].iloc[0])
        have = sorted(pd.Timestamp(p.stem) for p in library.DIRS[sym].glob("20??-??-??.parquet"))
        warm = [x for x in have if x < days[0]][-1:]
        d = library.build_index(sym, limit_days=warm + days, allow_exam=True, R_today=R)
        d = d[d["day"].isin(days)]
        for c in ("es_on_range", "es_on_rv", "es_on_ret", "es_on_vol", "es_last15") + GEX_COLS:
            if c not in d.columns:
                d[c] = np.nan
        d = d.sort_values(["day", "hhmm"]).reset_index(drop=True)
        if sym == "SPX":
            d = d.join(ts.build(d)[ts.SIGNALS], how="left")
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    for s in meta.INDEXES:
        d[f"is_{s}"] = (d["index"] == s).astype(float)
    d["tod"] = d["hhmm"].str[:2].astype(int) * 60 + d["hhmm"].str[3:].astype(int)
    d["post"] = (d["min_since_event"] <= 60).fillna(False)
    return d


def feature_list(dev: pd.DataFrame, groups: list) -> list:
    g = meta.feature_groups(dev.columns)
    return [c for k in groups for c in g.get(k, [])]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list) -> dict:
    missing = [c for c in feats if c not in test.columns]
    if missing:
        raise RuntimeError(f"exam rows lack inputs: {missing}")
    Xtr = train[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Xte = test[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    preds = {"long_straddle": np.full(len(test), np.nan)}
    post = train["post"].to_numpy()
    for k in MENU:
        y = train[f"y_{k}_half"].to_numpy(float)
        ok = ~post & np.isfinite(y)
        preds[k] = meta.make(MODEL).fit(Xtr[ok], y[ok]).predict(Xte)
    return preds


def boot_ci(x: np.ndarray, rng) -> list:
    m = rng.integers(0, len(x), (N_BOOT, len(x)))
    v = x[m].mean(axis=1)
    return [float(np.percentile(v, 5)), float(np.percentile(v, 95))]


def evaluate(test: pd.DataFrame, preds: dict) -> dict:
    rng = np.random.default_rng(20260908)
    days = sorted(test["day"].unique())
    res = {}
    for tag in ("half", "touch"):
        dec = meta.decide(test, preds, tag)
        daily = dec.groupby("day")["ret"].sum().reindex(days, fill_value=0.0)
        always = test.assign(r=np.where(test["post"], 0.0, test[f"y_fly_{tag}"].fillna(0.0))).groupby("day")["r"].sum().reindex(days, fill_value=0.0)
        diff = (daily - always).to_numpy()
        r = {"days": len(days), "mean_day": float(daily.mean()), "ci90": boot_ci(daily.to_numpy(), rng),
             "total": float(daily.sum()), "win_days": float((daily > 0).mean()),
             "always_sell_mean_day": float(always.mean()), "vs_always_mean": float(diff.mean()),
             "vs_always_ci90": boot_ci(diff, rng),
             "by_index": {}, "take_rate": float(dec["take"].mean()),
             "trade_mix": dec.loc[dec["take"], "trade"].value_counts(normalize=True).round(3).to_dict(),
             "per_trade_mean": float(dec.loc[dec["take"], "ret"].mean()) if dec["take"].any() else float("nan"),
             "per_trade_win": float((dec.loc[dec["take"], "ret"] > 0).mean()) if dec["take"].any() else float("nan")}
        for s, g in dec.groupby("index"):
            dd = g.groupby("day")["ret"].sum()
            r["by_index"][s] = {"mean_day": float(dd.mean()), "total": float(dd.sum()), "days": int(len(dd))}
        res[tag] = r
    h = res["half"]
    if h["mean_day"] > 0 and h["ci90"][0] > 0 and h["mean_day"] > h["always_sell_mean_day"]:
        res["outcome"] = "STRONG PASS"
    elif h["mean_day"] > 0 and h["mean_day"] > h["always_sell_mean_day"]:
        res["outcome"] = "WEAK PASS"
    else:
        res["outcome"] = "FAIL"
    return res


def run(train: pd.DataFrame, test: pd.DataFrame, tag: str) -> dict:
    feats = feature_list(train, GROUPS)
    out = {"spec": {"model": MODEL, "groups": GROUPS, "menu": MENU, "features": feats,
                    "train_days": [str(train["day"].min().date()), str(train["day"].max().date())],
                    "test_days": [str(test["day"].min().date()), str(test["day"].max().date())],
                    "test_rows": int(len(test))}}
    out["result"] = evaluate(test, fit_predict(train, test, feats))
    base = out["result"]["half"]["mean_day"]
    out["importance"] = {}
    for g in ABLATE:
        r = evaluate(test, fit_predict(train, test, feature_list(train, [x for x in GROUPS if x != g])))
        out["importance"][g] = {"mean_day_without": r["half"]["mean_day"], "contribution": base - r["half"]["mean_day"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def show(out: dict) -> None:
    r = out["result"]
    print(f"test {out['spec']['test_days'][0]} .. {out['spec']['test_days'][1]}: {r['half']['days']} days, {out['spec']['test_rows']} rows")
    for tag in ("half", "touch"):
        x = r[tag]
        print(f"  {tag:5s}: ${x['mean_day']:+,.0f}/day [{x['ci90'][0]:+,.0f}, {x['ci90'][1]:+,.0f}] total ${x['total']:+,.0f}"
              f" | always-sell ${x['always_sell_mean_day']:+,.0f}/day | model - always ${x['vs_always_mean']:+,.0f} "
              f"[{x['vs_always_ci90'][0]:+,.0f}, {x['vs_always_ci90'][1]:+,.0f}] | win days {x['win_days']:.0%}")
    h = r["half"]
    print("  by index:", {k: round(v["mean_day"]) for k, v in h["by_index"].items()},
          "| take", f"{h['take_rate']:.0%}", "| per trade", f"${h['per_trade_mean']:+.0f}", f"win {h['per_trade_win']:.0%}", "| mix", h["trade_mix"])
    print("  importance (exam $/day contributed):", {k: round(v["contribution"]) for k, v in out["importance"].items()})
    print("  OUTCOME:", r["outcome"])


def main() -> int:
    if sys.argv[1:2] == ["dryrun"]:
        dev = meta.load()
        test_days = sorted(dev["day"].unique())[-44:]
        built = build_rows(test_days)
        # the exam builder must reproduce the development matrix on these days
        key = ["day", "index", "hhmm"]
        cols = feature_list(dev, GROUPS) + [f"y_{k}_{t}" for k in MENU + ["long_straddle"] for t in ("half", "touch")] + ["post"]
        m = dev[dev["day"].isin(test_days)][key + cols].merge(built[key + cols], on=key, suffixes=("_dev", "_new"), how="outer", indicator=True)
        print("rows: dev", int((m["_merge"] != "right_only").sum()), "| rebuilt", int((m["_merge"] != "left_only").sum()), "| unmatched", int((m["_merge"] != "both").sum()))
        worst = {}
        for c in cols:
            a = pd.to_numeric(m[f"{c}_dev"], errors="coerce").to_numpy(float)
            b = pd.to_numeric(m[f"{c}_new"], errors="coerce").to_numpy(float)
            nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
            diff = np.nanmax(np.abs(a - b)) if np.isfinite(a - b).any() else 0.0
            if nan_mismatch or diff > 1e-6:
                worst[c] = {"max_abs_diff": float(diff), "nan_mismatch": nan_mismatch}
        print("columns that differ:", worst if worst else "none")
        train = dev[dev["day"] < test_days[0]]
        show(run(train, built, "dryrun"))
        return 0
    if sys.argv[1:2] == ["run"]:
        if not eg.unlocked():
            raise eg.ExamLocked("set EXAM_UNLOCK to the freeze commit recorded in docs/master_plan.md")
        if (OUT / "exam.json").exists():
            raise SystemExit("The exam has already been run (reports/exam/exam.json exists). It runs once.")
        dev = meta.load()
        test = build_rows(sessions_between(eg.EXAM_FROM, eg.EXAM_TO))
        out = run(dev, test, "exam")
        out["freeze_commit"] = eg.freeze_commit()
        (OUT / "exam.json").write_text(json.dumps(out, indent=2, default=float))
        show(out)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
