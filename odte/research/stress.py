"""Edge #13: new stress gauges as 0DTE signals (pre-registered 2026-11-10, docs/expiry_edge_plan.md).

  python3 -m odte.research.stress fetch
  python3 -m odte.research.stress run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from odte.research import factory as fx

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "research" / "stress.parquet"
TICKERS = ["^VIX", "^VVIX", "^SKEW", "^MOVE", "^OVX", "^GVZ", "^VXN"]
SIGNALS = ["vvix", "vvix_over_vix", "skew", "skew_z20", "move", "move_chg5",
           "ovx_over_vix", "gvz_over_vix", "vxn_over_vix"]
T_MIN = 3.0


def fetch() -> pd.DataFrame:
    import yfinance as yf
    out = {}
    for t in TICKERS:
        h = yf.Ticker(t).history(start="2015-01-01", end="2026-09-05", auto_adjust=False)
        s = h["Close"]
        s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        out[t.strip("^").lower()] = s
    d = pd.DataFrame(out).sort_index()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(DATA)
    return d


def daily_signals(c: pd.DataFrame) -> pd.DataFrame:
    """Signals from each close; shifted later so day d only sees closes before d."""
    s = pd.DataFrame(index=c.index)
    s["vvix"] = c["vvix"]
    s["vvix_over_vix"] = c["vvix"] / c["vix"]
    s["skew"] = c["skew"]
    m, sd = c["skew"].rolling(20, min_periods=15).mean(), c["skew"].rolling(20, min_periods=15).std()
    s["skew_z20"] = (c["skew"] - m) / sd
    s["move"] = c["move"]
    s["move_chg5"] = np.log(c["move"] / c["move"].shift(5))
    s["ovx_over_vix"] = c["ovx"] / c["vix"]
    s["gvz_over_vix"] = c["gvz"] / c["vix"]
    s["vxn_over_vix"] = c["vxn"] / c["vix"]
    return s


def as_of_previous_close(sig: pd.DataFrame, days: pd.Series) -> pd.DataFrame:
    """For each decision day, the latest signal row strictly before that day."""
    sig = sig.sort_index()
    pos = sig.index.searchsorted(pd.DatetimeIndex(days), side="left") - 1
    vals = np.where(pos[:, None] >= 0, sig.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.DataFrame(vals, index=days.index, columns=sig.columns)


def run() -> dict:
    d = fx.load()
    sig = as_of_previous_close(daily_signals(pd.read_parquet(DATA)), d["day"])
    d = pd.concat([d, sig], axis=1)
    era1 = d["day"] < fx.ERA_SPLIT
    rows = []
    for t in ("T_fly", "T_dir", "T_size", "T_tail"):
        for s in SIGNALS:
            r = fx.ic_test(d[s], d[t], d["day"])
            r1 = fx.ic_test(d[s][era1], d[t][era1], d["day"][era1])
            r2 = fx.ic_test(d[s][~era1], d[t][~era1], d["day"][~era1])
            rows.append({"target": t, "signal": s, **r, "ic_2016_21": r1["ic"], "ic_2022_26": r2["ic"]})
    tab = pd.DataFrame(rows)
    tab["fdr_pass"] = fx.bh(tab["p"].to_numpy(), fx.FDR_Q)
    tab["survives"] = (tab["fdr_pass"] & (tab["t"].abs() >= T_MIN)
                       & (np.sign(tab["ic_2016_21"]) == np.sign(tab["ic_2022_26"]))
                       & (np.sign(tab["ic"]) == np.sign(tab["ic_2016_21"])))
    out = ROOT / "reports" / "factory"
    tab.to_csv(out / "stress_tests.csv", index=False)
    print(tab.sort_values("p")[["target", "signal", "ic", "t", "ic_2016_21", "ic_2022_26", "fdr_pass",
                                "survives"]].round(4).to_string(index=False))
    res = {"survivors": tab[tab["survives"]][["target", "signal", "ic", "t"]].to_dict("records")}
    (out / "stress.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nSURVIVORS (pre-registered gate): {len(res['survivors'])}")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "fetch":
        f = fetch()
        print(f"{len(f)} days, {f.index.min().date()} .. {f.index.max().date()}")
    else:
        run()
