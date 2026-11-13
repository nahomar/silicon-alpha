"""T2: tail-robust re-ranking of the 2,048-point grid (rules in docs/master_plan.md, fixed before results).

  python3 -m odte.research.t2
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from scipy import stats

from odte.research import meta

DIR = meta.OUT / "search_daily"


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 20 and x.std(ddof=1) > 0 else np.nan


def capped_sharpe(x: pd.Series) -> float:
    return sharpe(x.clip(upper=x.quantile(0.99)))


def no_top10(x: pd.Series) -> float:
    return sharpe(x.drop(x.sort_values().tail(10).index))


def always_fly() -> pd.Series:
    d = meta.load()
    return d.assign(r=np.where(d["post"], 0.0, d["y_fly_half"].fillna(0.0))).groupby("day")["r"].sum()


def deflated(x: pd.Series, sr_trials_annual: np.ndarray, n: int) -> float:
    """Bailey & Lopez de Prado deflated Sharpe probability, daily units."""
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = stats.skew(x), stats.kurtosis(x, fisher=False)
    v = np.nanvar(sr_trials_annual / np.sqrt(252))
    em = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / n) + em * stats.norm.ppf(1 - 1 / (n * np.e)))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def table() -> pd.DataFrame:
    summ = pd.read_csv(meta.OUT / "search_summary.csv").set_index("task")
    af = always_fly()
    rows = []
    for f in sorted(glob.glob(str(DIR / "s*.parquet"))):
        task = int(f.split("/s")[-1][:5])
        s = pd.read_parquet(f).set_index("day")
        r = {"task": task, "alpha": summ.loc[task, "alpha"], "groups": summ.loc[task, "groups"]}
        for prefix in ("", "sell_"):
            x = s[prefix + "half"]
            a, b = x[x.index >= meta.EVAL_FROM], x[x.index < meta.EVAL_FROM]
            r[prefix + "sh22"], r[prefix + "sh18"] = sharpe(a), sharpe(b)
            r[prefix + "cap22"], r[prefix + "cap18"] = capped_sharpe(a), capped_sharpe(b)
            r[prefix + "no10_22"] = no_top10(a)
            r[prefix + "mean22"], r[prefix + "mean18"] = a.mean(), b.mean()
            r[prefix + "excess18"] = float((b - af.reindex(b.index).fillna(0.0)).mean())
            t = s[prefix + "touch"]
            r[prefix + "touch22"] = t[t.index >= meta.EVAL_FROM].mean()
        rows.append(r)
    x = pd.DataFrame(rows)
    x["robust5"] = x[["cap22", "cap18"]].min(axis=1)               # pre-registered T2, 5-trade menu
    x["eligible"] = x["sell_excess18"] > 0                          # amended rule's guard
    return x


if __name__ == "__main__":
    x = table()
    print(f"{len(x)} configurations")
    cols5 = ["robust5", "cap22", "cap18", "no10_22", "mean22", "touch22", "alpha", "groups"]
    print("\n== T2 as pre-registered (5-trade menu), rank by worse-era capped Sharpe")
    print(x.sort_values("robust5", ascending=False).head(8)[cols5].round(2).to_string(index=False))
    e = x[x["eligible"]].sort_values("sell_cap22", ascending=False)
    print(f"\n== Amended rule (sell-only): {len(e)} of {len(x)} pass the 2018-21 guard; rank by 2022-26 capped Sharpe")
    cols = ["sell_cap22", "sell_sh22", "sell_no10_22", "sell_mean22", "sell_touch22", "sell_sh18", "sell_excess18", "alpha", "groups"]
    print(e.head(12)[cols].round(2).to_string(index=False))
    if len(e):
        best = e.iloc[0]
        s = pd.read_parquet(DIR / f"s{int(best['task']):05d}.parquet").set_index("day")["sell_half"]
        a = s[s.index >= meta.EVAL_FROM]
        trials = x["sell_sh22"].to_numpy()
        print(f"\nleader task {int(best['task'])}: deflated Sharpe probability real (2022-26) "
              + ", ".join(f"N={n}: {deflated(a, trials, n):.0%}" for n in (10, 50, 200, 2048)))
        # plateau: same alpha, groups differing by exactly one
        g0 = set(str(best["groups"]).split(","))
        nb = x[(x["alpha"] == best["alpha"]) & x["groups"].apply(lambda g: len(set(str(g).split(",")) ^ g0) == 1)]
        print(f"plateau: {len(nb)} one-group neighbours, sell_cap22 median {nb['sell_cap22'].median():.2f}"
              f" (min {nb['sell_cap22'].min():.2f}, max {nb['sell_cap22'].max():.2f}); leader {best['sell_cap22']:.2f}")
    print(f"\nsell-only grid overall: median 2022-26 capped Sharpe {x['sell_cap22'].median():.2f}; "
          f"share > 0: {(x['sell_cap22'] > 0).mean():.0%}; share beating always-fly in 2018-21: {x['eligible'].mean():.0%}")
    x.to_csv(meta.OUT / "t2_table.csv", index=False)
    (meta.OUT / "t2_top.json").write_text(json.dumps(e.head(20).to_dict(orient="records"), indent=2, default=float))
