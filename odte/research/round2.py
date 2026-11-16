"""Round 2 (docs/round2_plan.md): challengers vs the locked model B0, walk-forward on development
data through 2026-11-06. Each candidate changes one thing vs B0.

  python3 -m odte.research.round2            # B0, C1..C7 -> reports/round2/
  python3 -m odte.research.round2 c8 C2,C4   # the pre-registered combination of the winners
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

from odte.research import meta

OUT = meta.ROOT / "reports" / "round2"
MENU = ("fly", "condor", "put_spread", "call_spread")
B0_GROUPS = ["calendar", "gex", "term", "flow", "identity"]
LIVE_GROUPS = {"session", "calendar", "context", "gex", "term", "flow", "strategy", "identity"}
WIN = (pd.Timestamp("2023-07-03"), pd.Timestamp("2026-11-06"))
HALVES = ((pd.Timestamp("2023-07-03"), pd.Timestamp("2024-12-31")), (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-11-06")))
ERA = pd.Timestamp("2022-05-16")
LATE = ("15:00", "15:30")


def features(d, groups):
    g = meta.feature_groups(d.columns)
    feats = [c for k in groups for c in g.get(k, [])]
    return d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def ridge(alpha):
    return meta.make(f"ridge_a{alpha:g}")


def walk(d, X, labels="half", alpha=1000.0, train_from=None, half_life_years=None, kind="ridge"):
    """meta.walk_forward's schedule (first fit after MIN_DAYS sessions, refit every REFIT),
    with the round-2 options. Returns per-trade predictions and, per row, the training window's
    median positive best prediction (for C4 sizing)."""
    days = np.array(sorted(d["day"].unique()))
    dd = d["day"].to_numpy()
    ok_rows = ~d["post"].to_numpy()
    if train_from is not None:
        ok_rows &= dd >= np.datetime64(train_from)
    Y = {k: d[f"y_{k}_{labels}"].to_numpy(float) for k in MENU}
    Yhalf = {k: d[f"y_{k}_half"].to_numpy(float) for k in MENU}
    P = {k: np.full(len(d), np.nan) for k in MENU}
    scale = np.full(len(d), np.nan)
    for k0 in range(meta.MIN_DAYS, len(days), meta.REFIT):
        tr_days = dd < days[k0]
        te = (dd >= days[k0]) & ((dd < days[k0 + meta.REFIT]) if k0 + meta.REFIT < len(days) else True)
        fitted = {}
        for k in MENU:
            tr = tr_days & ok_rows & np.isfinite(Y[k])
            if tr.sum() < 1000:
                continue
            w = None
            if half_life_years:
                age = (np.datetime64(pd.Timestamp(days[k0]), "ns") - dd[tr]) / np.timedelta64(1, "D") / 365.25
                w = 0.5 ** (age / half_life_years)
            if kind == "ridge":
                m = ridge(alpha)
                m.fit(X[tr], Y[k][tr], **({"ridge__sample_weight": w} if w is not None else {}))
                P[k][te] = m.predict(X[te])
                fitted[k] = (m, tr)
            else:                                                     # C6: logistic expected value
                from sklearn.impute import SimpleImputer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                y = Yhalf[k][tr]
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
                m.fit(X[tr], (y > 0).astype(int))
                win, loss = y[y > 0].mean(), -y[y <= 0].mean()
                p = m.predict_proba(X[te])[:, 1]
                P[k][te] = p * win - (1 - p) * loss
        if fitted and kind == "ridge":                                # C4: typical positive edge in training
            trm = tr_days & ok_rows
            best = np.max(np.column_stack([fitted[k][0].predict(X[trm]) if k in fitted else np.full(trm.sum(), -np.inf)
                                           for k in MENU]), axis=1)
            pos = best[best > 0]
            scale[te] = np.median(pos) if len(pos) else np.nan
    return P, scale


def decide(d, P, tag="half", size=None, drop_late=False):
    preds = dict(P, long_straddle=np.full(len(d), np.nan))
    dec = meta.decide(d, preds, tag)
    if drop_late:
        late = d["hhmm"].isin(LATE).to_numpy()
        dec.loc[late, "ret"] = 0.0
        dec.loc[late, "take"] = False
    if size is not None:
        dec["ret"] = dec["ret"].to_numpy() * np.where(dec["take"].to_numpy(), size, 0.0)
    dec["hhmm"] = d["hhmm"].to_numpy()
    return dec


def sharpe(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 20 and x.std(ddof=1) > 0 else np.nan


def capped(x):
    return sharpe(x.clip(upper=x.quantile(0.99)))


def metrics(d, P, size=None, drop_late=False) -> dict:
    out = {}
    all_days = pd.DatetimeIndex(sorted(d["day"].unique()))
    win_days = all_days[(all_days >= WIN[0]) & (all_days <= WIN[1])]
    for tag in ("half", "touch"):
        dec = decide(d, P, tag, size, drop_late)
        daily = dec[dec["tested"]].groupby("day")["ret"].sum().reindex(win_days, fill_value=0.0)
        out[tag] = daily
        if tag == "half":
            spx = dec[dec["tested"] & (dec["index"] == "SPX")].groupby("day")["ret"].sum().reindex(win_days, fill_value=0.0)
            t = dec[dec["take"] & dec["day"].between(*WIN)]
            out["spx"] = spx
            out["mix"] = t["trade"].value_counts(normalize=True).round(3).to_dict()
            out["take"] = float(dec.loc[dec["tested"] & dec["day"].between(*WIN), "take"].mean())
    h, tch = out["half"], out["touch"]
    top = h.sort_values().tail(10).index
    res = {"primary_capped_sharpe": capped(h), "sharpe": sharpe(h), "mean_day": float(h.mean()),
           "touch_mean_day": float(tch.mean()), "touch_sharpe": sharpe(tch),
           "no_top10_sharpe": sharpe(h.drop(top)), "spx_sharpe": sharpe(out["spx"]), "spx_mean_day": float(out["spx"].mean()),
           "half1_capped": capped(h[(h.index >= HALVES[0][0]) & (h.index <= HALVES[0][1])]),
           "half2_capped": capped(h[(h.index >= HALVES[1][0]) & (h.index <= HALVES[1][1])]),
           "exam_window_mean": float(h[h.index >= pd.Timestamp("2026-09-08")].mean()),
           "take_rate": out["take"], "mix": out["mix"]}
    return res, h


def c3_configs(n=10) -> list:
    t = pd.read_csv(meta.OUT / "t2_table.csv")
    t = t[t["eligible"]].copy()
    t["gs"] = t["groups"].fillna("").apply(lambda g: [x for x in str(g).split(",") if x])
    t = t[t["gs"].apply(lambda gs: set(gs) <= LIVE_GROUPS)]
    t = t.sort_values("sell_cap22", ascending=False).head(n)
    return [(r["gs"] + ["identity"], float(r["alpha"])) for _, r in t.iterrows()]


def candidates(d) -> dict:
    X0 = features(d, B0_GROUPS)
    out = {}
    P, sc = walk(d, X0)
    out["B0"] = (P, None, False)
    out["C1_era_only"] = (walk(d, X0, train_from=ERA)[0], None, False)
    out["C2_recency_1y"] = (walk(d, X0, half_life_years=1.0)[0], None, False)
    ens = [walk(d, features(d, gs), alpha=a)[0] for gs, a in c3_configs()]
    out["C3_plateau_ensemble"] = ({k: np.nanmean([e[k] for e in ens], axis=0) for k in MENU}, None, False)
    best = np.nanmax(np.column_stack([P[k] for k in MENU]), axis=1)
    out["C4_confidence_size"] = (P, np.clip(best / sc, 0, 2), False)
    out["C5_cost_aware_touch"] = (walk(d, X0, labels="touch")[0], None, False)
    out["C6_meta_label"] = (walk(d, X0, kind="logistic")[0], None, False)
    out["C7_no_late"] = (P, None, True)
    return out


def deflated(h: pd.Series, trial_sharpes: np.ndarray) -> float:
    sr = h.mean() / h.std(ddof=1)
    g3, g4 = stats.skew(h), stats.kurtosis(h, fisher=False)
    n = len(trial_sharpes)
    v = np.nanvar(trial_sharpes / np.sqrt(252))
    em = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / n) + em * stats.norm.ppf(1 - 1 / (n * np.e)))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(len(h) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def show(name, r):
    print(f"{name:22s} capped {r['primary_capped_sharpe']:+.2f} | Sh {r['sharpe']:+.2f} ${r['mean_day']:+5.0f}/d | touch ${r['touch_mean_day']:+5.0f}"
          f" | no-top10 {r['no_top10_sharpe']:+.2f} | SPX {r['spx_sharpe']:+.2f} | halves {r['half1_capped']:+.2f} / {r['half2_capped']:+.2f}"
          f" | Sep-Nov ${r['exam_window_mean']:+5.0f}/d | take {r['take_rate']:.0%}", flush=True)


def combine(d, parts: list) -> int:
    """C8 (pre-registered): every candidate that beat B0 on the primary metric, combined."""
    X0 = features(d, B0_GROUPS)
    kw = {"labels": "touch" if "C5" in parts else "half"}
    if "C1" in parts:
        kw["train_from"] = ERA
    if "C2" in parts:
        kw["half_life_years"] = 1.0
    P, sc = walk(d, X0, **kw)
    best = np.nanmax(np.column_stack([P[k] for k in MENU]), axis=1)
    size = np.clip(best / sc, 0, 2) if "C4" in parts else None
    r, h = metrics(d, P, size, "C7" in parts)
    res = json.loads((OUT / "candidates.json").read_text())
    b0 = res["B0"]
    r["beats_b0_primary"] = r["primary_capped_sharpe"] - b0["primary_capped_sharpe"]
    r["beats_b0_both_halves"] = bool(r["half1_capped"] > b0["half1_capped"] and r["half2_capped"] > b0["half2_capped"])
    daily = pd.read_parquet(OUT / "daily.parquet")
    daily["C8"] = h
    trials = np.array([sharpe(daily[c]) for c in daily.columns])
    r["deflated_p"] = deflated(h, trials)
    r["parts"] = parts
    res["C8_" + "+".join(parts)] = r
    show("C8 " + "+".join(parts), r)
    print(f"   vs B0 {r['beats_b0_primary']:+.2f}, both halves: {r['beats_b0_both_halves']}, deflated P(real) {r['deflated_p']:.0%} (N={len(trials)})")
    (OUT / "candidates.json").write_text(json.dumps(res, indent=2, default=float))
    daily.to_parquet(OUT / "daily.parquet")
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = meta.load()
    if sys.argv[1:2] == ["c8"]:
        return combine(d, sys.argv[2].split(","))
    res, series = {}, {}
    for name, (P, size, late) in candidates(d).items():
        r, h = metrics(d, P, size, late)
        res[name], series[name] = r, h
        show(name, r)
    b0 = res["B0"]
    trials = np.array([r["sharpe"] for r in res.values()])
    for name, r in res.items():
        r["beats_b0_primary"] = r["primary_capped_sharpe"] - b0["primary_capped_sharpe"]
        r["beats_b0_both_halves"] = bool(r["half1_capped"] > b0["half1_capped"] and r["half2_capped"] > b0["half2_capped"])
        r["deflated_p"] = deflated(series[name], trials)
    print("\nvs B0 (primary +0.10 needed, and both halves):")
    for name, r in res.items():
        if name != "B0":
            print(f"   {name:22s} {r['beats_b0_primary']:+.2f}  both halves: {r['beats_b0_both_halves']}  deflated P(real) {r['deflated_p']:.0%}")
    (OUT / "candidates.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(series).to_parquet(OUT / "daily.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
