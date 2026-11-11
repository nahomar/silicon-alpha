"""Volatility premium through SVXY with a VIX term-structure filter (docs/vol_etf_plan.md).

  python3 -m trend.volatility fetch
  python3 -m trend.volatility run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trend import backtest as tb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "trend" / "voletf.parquet"
OUT = ROOT / "reports" / "trend"
START, END = "2011-10-01", "2026-10-30"
POST_FROM, LAST = "2018-03", "2026-10"
COST_BP = 5.0


def fetch() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(["SVXY", "^VIX", "^VIX3M", "SPY", "^IRX"], start=START,
                      end=pd.Timestamp(END) + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    px = raw["Close"].copy()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px[px.index <= pd.Timestamp(END)]
    DATA.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(DATA)
    return px


def strategy(px: pd.DataFrame, thr: float, cost_bp: float = COST_BP) -> pd.DataFrame:
    """Daily total/excess returns. Signal at close d, trade at close d+1 -> earns from d+2."""
    px = px[px["SVXY"].notna()]
    r = px["SVXY"].pct_change(fill_method=None).fillna(0.0)
    rf = (px["^IRX"].ffill() / 100.0 / 252.0).shift(1).fillna(0.0)
    sig = (px["^VIX"] / px["^VIX3M"] < thr).astype(float)
    pos = sig.shift(2).fillna(0.0)
    cost = pos.diff().abs().fillna(pos.abs()) * cost_bp / 1e4
    total = pos * r + (1 - pos) * rf - cost
    return pd.DataFrame({"total": total, "excess": total - rf, "pos": pos, "rf": rf})


def run() -> dict:
    px = pd.read_parquet(DATA)
    big = px["SVXY"].pct_change(fill_method=None)
    res = {"svxy_daily_moves_over_25pct": {str(k.date()): round(float(v), 4) for k, v in big[big.abs() > 0.25].items()}}
    out = {}
    for name, thr in (("V1_vix_lt_vix3m", 1.0), ("V2_ratio_lt_0.90", 0.90)):
        out[name] = strategy(px, thr)
    rf = out["V1_vix_lt_vix3m"]["rf"]
    for b in ("SVXY", "SPY"):
        tot = px[b].pct_change(fill_method=None).reindex(rf.index).fillna(0.0)
        out[f"{b}_buy_hold"] = pd.DataFrame({"total": tot, "excess": tot - rf})
    first = out["V1_vix_lt_vix3m"].index[2].to_period("M") + 1
    periods = {"full": pd.period_range(first, LAST, freq="M"), "post_2018_03": pd.period_range(POST_FROM, LAST, freq="M")}
    for label, months in periods.items():
        res[label] = {}
        for k, d in out.items():
            t, e = tb.monthly(d["total"]).reindex(months), tb.monthly(d["excess"]).reindex(months)
            row = tb.describe(t, e)
            row["sharpe_ci"] = tb.block_bootstrap_sharpe(e)["ci"]
            dd = d["total"][(d.index >= months[0].start_time) & (d.index <= months[-1].end_time)]
            row["worst_day"] = float(dd.min())
            row["excess_total"] = float((1 + e).prod() - 1)
            if "pos" in d:
                row["time_in_market"] = float(d["pos"][dd.index].mean())
            res[label][k] = row
    episodes = {"2018-02 Volmageddon": ("2018-02-01", "2018-02-28"), "2020-03 Covid": ("2020-02-20", "2020-03-31"),
                "2024-08 yen carry": ("2024-07-25", "2024-08-15")}
    res["episodes"] = {ep: {k: float((1 + d["total"][a:b]).prod() - 1) for k, d in out.items()}
                       for ep, (a, b) in episodes.items()}
    v1 = res["post_2018_03"]["V1_vix_lt_vix3m"]
    res["passes"] = bool(v1["sharpe_ci"][0] > 0 and res["full"]["V1_vix_lt_vix3m"]["excess_total"] > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "volatility.json").write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        f = fetch()
        print(f"{len(f)} days {f.index.min().date()} .. {f.index.max().date()}; first SVXY "
              f"{f['SVXY'].first_valid_index().date()}, first VIX3M {f['^VIX3M'].first_valid_index().date()}")
    else:
        print(json.dumps(run(), indent=2, default=str))
