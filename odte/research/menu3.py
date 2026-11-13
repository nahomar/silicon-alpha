"""Round 3 labels: 10 more defined-risk shapes for every development decision row
(docs/round2_plan.md, "Round 3"). Same pricing and cost scaling as odte/research/library.py; the
four existing shapes are recomputed in the same pass as an exactness check.

  python3 -m odte.research.menu3          # -> data/research/menu3_labels.parquet
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "research" / "menu3_labels.parquet"
NEW = {
    "fly_narrow": {"structure": "iron_fly", "wing": 1.0},
    "fly_wide": {"structure": "iron_fly", "wing": 2.0},
    "condor_tight": {"structure": "iron_condor", "short_k": 0.6, "wing": 0.5},
    "condor_wide": {"structure": "iron_condor", "short_k": 1.4, "wing": 0.5},
    "put_spread_near": {"structure": "put_credit_spread", "short_k": 0.5, "wing": 0.5},
    "put_spread_far": {"structure": "put_credit_spread", "short_k": 1.2, "wing": 0.5},
    "call_spread_near": {"structure": "call_credit_spread", "short_k": 0.5, "wing": 0.5},
    "call_spread_far": {"structure": "call_credit_spread", "short_k": 1.2, "wing": 0.5},
    "pin_put_fly": {"structure": "long_put_butterfly", "center_k": -0.5, "wing": 0.5},
    "pin_call_fly": {"structure": "long_call_butterfly", "center_k": 0.5, "wing": 0.5},
}
CHECK = ("fly", "condor", "put_spread", "call_spread")


def _one(job):
    sym, day, times, R = job
    from odte.research import library, zoo
    folder = library.DIRS[sym]
    st = pd.read_csv(folder / "settlements.csv", parse_dates=["date"]).set_index("date")["spx_close"]
    prev = st[st.index < day]
    zd = zoo.Day(day, pd.read_parquet(folder / f"{day:%Y-%m-%d}.parquet"), float(st[day]),
                 float(prev.iloc[-1]) if len(prev) else np.nan, {})
    r10 = zd.straddle_rel_spread(day + pd.Timedelta(hours=10))
    k_day = float(np.clip(R / r10, 0.1, 2.0)) if np.isfinite(r10) and r10 > 0 else 1.0
    rules = {**NEW, **{k: library.TRADES[k] for k in CHECK}}
    rows = []
    for hhmm in times:
        r = {"day": day, "index": sym, "hhmm": hhmm}
        for name, rule in rules.items():
            for share, tag in ((0.5, "half"), (1.0, "touch")):
                tr = zoo.trade(rule, zd, hhmm, k_day, share)
                r[f"y_{name}_{tag}" if name in NEW else f"chk_{name}_{tag}"] = tr["ret"] if tr else np.nan
        rows.append(r)
    return rows


def build(workers: int = 10) -> pd.DataFrame:
    from odte.research import exam_guard as eg
    from odte.research import meta
    m = pd.read_parquet(meta.MATRIX, columns=["day", "index", "hhmm"] + [f"y_{k}_{t}" for k in CHECK for t in ("half", "touch")])
    eg.check(m["day"])
    R = {s: float(pd.read_parquet(meta.LIB / f"library_{s}.parquet", columns=["R_today"])["R_today"].iloc[0]) for s in meta.INDEXES}
    jobs = [(s, d, sorted(g["hhmm"]), R[s]) for (s, d), g in m.groupby(["index", "day"])]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for i, part in enumerate(ex.map(_one, jobs, chunksize=8)):
            rows += part
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)} index-days", flush=True)
    out = pd.DataFrame(rows)
    chk = m.merge(out, on=["day", "index", "hhmm"], how="left", validate="one_to_one")
    worst = 0.0
    for k in CHECK:
        for t in ("half", "touch"):
            a, b = chk[f"y_{k}_{t}"].to_numpy(float), chk[f"chk_{k}_{t}"].to_numpy(float)
            if (np.isnan(a) != np.isnan(b)).any():
                raise SystemExit(f"exactness check failed: {k}_{t} missing pattern differs")
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
    if worst > 1e-6:
        raise SystemExit(f"exactness check failed: max difference {worst}")
    out = out.drop(columns=[c for c in out.columns if c.startswith("chk_")])
    out.to_parquet(OUT, index=False)
    print(f"{len(out):,} rows, {len(NEW)} new shapes -> {OUT}; exactness check passed (max diff {worst:.1e})")
    return out


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
