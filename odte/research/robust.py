"""Robustness report for one meta-model configuration (development data only).

  python3 -m odte.research.robust "session,calendar,context,flow,strategy,factory" 1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx
from odte.research import meta

ROOT = Path(__file__).resolve().parents[2]


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def run(groups: list, alpha: float, tag: str = "leader") -> dict:
    d = meta.load()
    g = meta.feature_groups(d.columns)
    feats = [c for k in groups + ["identity"] for c in g.get(k, [])]
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    Y = {k: d[f"y_{k}_half"].to_numpy(float) for k in meta.TRADES}
    preds = meta.walk_forward(d, X, Y, f"ridge_a{alpha:g}")
    pd.DataFrame(preds).assign(day=d["day"], index=d["index"], hhmm=d["hhmm"]).to_parquet(
        meta.OUT / f"preds_{tag}.parquet", index=False)
    half, touch = meta.decide(d, preds, "half"), meta.decide(d, preds, "touch")
    half["hhmm"] = d["hhmm"].to_numpy()
    t = half[half["tested"]]
    t_touch = touch[touch["tested"]]
    res = {"groups": groups, "alpha": alpha, "features": len(feats)}
    daily = t.groupby("day")["ret"].sum()
    always = d.assign(r=np.where(d["post"], 0.0, d["y_fly_half"].fillna(0.0))).groupby("day")["r"].sum().reindex(daily.index)
    res["by_year"] = {int(y): {"model": round(float(v.sum()), 0), "always_fly": round(float(always[v.index].sum()), 0),
                               "sharpe": round(sharpe(v), 2)} for y, v in daily.groupby(daily.index.year)}
    post = t[t["day"] >= meta.EVAL_FROM]
    res["by_index_2022_26"] = {}
    for s, g_ in post.groupby("index"):
        dd = g_.groupby("day")["ret"].sum()
        res["by_index_2022_26"][s] = {"mean_index_day": round(float(dd.mean()), 1), "sharpe": round(sharpe(dd), 2),
                                      "days": int(len(dd))}
    res["by_time_2022_26"] = {h: round(float(g_["ret"].mean()), 1) for h, g_ in post.groupby("hhmm")}
    res["trade_mix"] = {"2022_26": post.loc[post["take"], "trade"].value_counts(normalize=True).round(3).to_dict(),
                        "2016_21": t[(t["day"] < meta.EVAL_FROM) & t["take"]]["trade"].value_counts(normalize=True).round(3).to_dict()}
    res["by_trade_2022_26"] = {k: round(float(g_["ret"].mean()), 1) for k, g_ in post[post["take"]].groupby("trade")}
    td = t_touch.groupby("day")["ret"].sum()
    res["touch_2022_26"] = {"mean_day": round(float(td[td.index >= meta.EVAL_FROM].mean()), 1),
                            "sharpe": round(sharpe(td[td.index >= meta.EVAL_FROM]), 2)}
    b = fx.summarize(daily[daily.index >= meta.EVAL_FROM])
    res["eval_2022_26"] = {"mean_day": round(float(daily[daily.index >= meta.EVAL_FROM].mean()), 1),
                           "sharpe": b.get("sharpe"), "sharpe_ci": b.get("sharpe_ci")}
    # stricter entry: only take trades whose predicted return clears a margin
    P = np.column_stack([preds[k] for k in meta.TRADES])
    best = np.nanmax(np.where(np.isfinite(P), P, -np.inf), axis=1)
    res["margin_sensitivity_2022_26"] = {}
    for m in (0.0, 0.02, 0.05, 0.10):
        take = np.isfinite(best) & (best > m) & ~d["post"].to_numpy()
        r = np.where(take, half["ret"].to_numpy(), 0.0)
        s = pd.Series(r).groupby(d["day"].to_numpy()).sum()
        s = s[s.index >= meta.EVAL_FROM]
        res["margin_sensitivity_2022_26"][str(m)] = {"take_rate": round(float(take[d["day"] >= meta.EVAL_FROM].mean()), 3),
                                                     "mean_day": round(float(s.mean()), 1), "sharpe": round(sharpe(s), 2)}
    (meta.OUT / f"robust_{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1].split(","), float(sys.argv[2])), indent=2, default=float))
