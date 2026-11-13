"""Combine one strategy's daily P&L across underlyings (SPX, NDX, RUT).

Each study run (expiry_edge --write --tag NAME) stores daily P&L for its key
strategies. This sums them by date, at equal risk per trade per underlying,
and scores the portfolio against always-short on the same underlyings. More
independent bets per day is the point: the edge per trade is small and noisy,
so breadth is what can make it provable and scalable.

Usage:
  python3 -m odte.eval.combine --tags spx16 ndx rut --strategy "ml_ridge+cal+ctx -postevt"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .expiry_edge import OUT, paired_diff
from .panel_stats import day_block_bootstrap


def load_daily(tag: str, strategy: str, mode: str = "fixed6") -> pd.Series:
    rep = json.loads((OUT / f"study-{tag}.json").read_text())
    d = rep["modes"][mode]["daily"].get(strategy)
    if d is None:
        raise KeyError(f"{tag}: no daily series for {strategy!r} "
                       f"(have {list(rep['modes'][mode]['daily'])})")
    return pd.Series(d, dtype=float).rename(tag).set_axis(pd.to_datetime(list(d)))


def combine(tags, strategy, ref="always_short_fly", mode="fixed6") -> dict:
    s = pd.concat([load_daily(t, strategy, mode) for t in tags], axis=1).fillna(0.0)
    r = pd.concat([load_daily(t, ref, mode) for t in tags], axis=1).fillna(0.0)
    port, base = s.sum(axis=1), r.sum(axis=1).reindex(s.index).fillna(0.0)
    boot = day_block_bootstrap(port.to_numpy())
    cum = port.cumsum()
    return {"tags": list(tags), "strategy": strategy, "days": int(len(port)),
            "total": float(port.sum()), "per_day": float(port.mean()),
            "sharpe": boot["sharpe"], "sharpe_ci": [boot["ci_low"], boot["ci_high"]],
            "max_drawdown": float((cum - cum.cummax()).min()),
            "vs_ref": paired_diff(port, base),
            "ref_total": float(base.sum()),
            "corr_between": s.corr().round(2).to_dict(),
            "per_underlying_total": {t: float(s[t].sum()) for t in s}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--mode", default="fixed6")
    a = ap.parse_args()
    out = combine(a.tags, a.strategy, mode=a.mode)
    v = out["vs_ref"]
    print(f"{out['strategy']} on {', '.join(out['tags'])}: {out['days']} days, total ${out['total']:,.0f} "
          f"(${out['per_day']:.0f}/day), Sharpe {out['sharpe']:+.2f} "
          f"[{out['sharpe_ci'][0]:+.2f}, {out['sharpe_ci'][1]:+.2f}], max DD ${out['max_drawdown']:,.0f}")
    print(f"  vs always-short on the same underlyings (total ${out['ref_total']:,.0f}): "
          f"{v['mean']:+.1f}/day CI [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")
    print(f"  per underlying: {out['per_underlying_total']}")
    print(f"  daily P&L correlation: {out['corr_between']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
