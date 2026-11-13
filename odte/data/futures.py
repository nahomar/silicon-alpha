"""S&P 500 futures (ES) overnight features, from front-month 1-minute bars.

ES trades nearly around the clock, so it records the overnight session that
the cash index and its options never see. The options data already gives
the *net* overnight move (the opening gap). ES adds the *path*: how violent
the night was, its range, its volume. Overnight turbulence is a known
predictor of the next day's realized volatility.

Data: Databento GLBX.MDP3 ohlcv-1m, continuous front month (ES.c.0), saved to
data/futures/es_1m.parquet. A bar stamped m covers [m, m+1 min) and is used
only once m + 1 min <= the cut-off. Returns are taken within one contract
(`instrument_id`), so quarterly rolls do not appear as jumps.

Features for session d (window: previous session 16:00 ET -> d 09:30 ET):
  es_on_ret    overnight log return, bp
  es_on_rv     overnight realized vol, sqrt(sum r^2) in bp
  es_on_range  (high - low) / prior close, bp
  es_on_vol    overnight volume / median of the previous 20 nights
  es_last15    09:15 -> 09:30 log return, bp

Usage:
  python3 -m odte.data.futures fetch --start 2015-12-01 --end 2026-11-07 --max-usd 16
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "futures"
TZ = "America/New_York"
ES_FEATURES = ["es_on_ret", "es_on_rv", "es_on_range", "es_on_vol", "es_last15"]


def fetch(start, end, max_usd: float) -> pd.DataFrame:
    import databento as db
    c = db.Historical(os.environ["DATABENTO_API_KEY"])
    kw = dict(dataset="GLBX.MDP3", schema="ohlcv-1m", symbols=["ES.c.0"], stype_in="continuous")
    cost = float(c.metadata.get_cost(start=start, end=end, **kw))
    print(f"ES.c.0 ohlcv-1m {start}..{end}: ${cost:.2f} (cap ${max_usd:.2f})", flush=True)
    if cost > max_usd:
        raise SystemExit("ABORT: estimate exceeds the cap; nothing was downloaded.")
    parts = []
    for y0 in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        lo = max(pd.Timestamp(start), pd.Timestamp(f"{y0}-01-01"))
        hi = min(pd.Timestamp(end), pd.Timestamp(f"{y0 + 1}-01-01"))
        if lo >= hi:
            continue
        df = c.timeseries.get_range(start=lo.isoformat(), end=hi.isoformat(), **kw).to_df()
        df = df.reset_index()[["ts_event", "instrument_id", "open", "high", "low", "close", "volume"]]
        parts.append(df)
        print(f"  {y0}: {len(df):,} bars", flush=True)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["ts_event"]).sort_values("ts_event")
    from odte.timeutil import epoch_s
    out["ts"] = epoch_s(pd.to_datetime(out["ts_event"], utc=True))
    out = out.drop(columns="ts_event")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "es_1m.parquet", index=False)
    print(f"ES: {len(out):,} bars -> {OUT/'es_1m.parquet'}")
    return out


CROSS = {"ZN": "10-yr Treasury", "ZT": "2-yr Treasury", "NQ": "Nasdaq-100", "RTY": "Russell 2000",
         "CL": "crude oil", "6E": "euro/dollar", "GC": "gold"}


def fetch_cross(start, end, max_usd: float, roots=tuple(CROSS)) -> None:
    """Other markets' 1-minute bars (most-traded contract, `ROOT.v.0`), saved
    one file per market as data/futures/<root>_1m.parquet (ts, instrument_id,
    high, low, close, volume). The whole cost is checked against the cap first."""
    import databento as db
    c = db.Historical(os.environ["DATABENTO_API_KEY"])
    kw = dict(dataset="GLBX.MDP3", schema="ohlcv-1m", stype_in="continuous")
    costs = {r: float(c.metadata.get_cost(symbols=[f"{r}.v.0"], start=start, end=end, **kw)) for r in roots}
    for r, x in costs.items():
        print(f"  {r:4s} {CROSS[r]:15s} ${x:.2f}", flush=True)
    print(f"total ${sum(costs.values()):.2f} (cap ${max_usd:.2f})", flush=True)
    if sum(costs.values()) > max_usd:
        raise SystemExit("ABORT: estimate exceeds the cap; nothing was downloaded.")
    OUT.mkdir(parents=True, exist_ok=True)
    for r in roots:
        p = OUT / f"{r}_1m.parquet"
        if p.exists():
            continue
        parts = []
        for y0 in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            lo = max(pd.Timestamp(start), pd.Timestamp(f"{y0}-01-01"))
            hi = min(pd.Timestamp(end), pd.Timestamp(f"{y0 + 1}-01-01"))
            if lo < hi:
                df = c.timeseries.get_range(symbols=[f"{r}.v.0"], start=lo.isoformat(),
                                            end=hi.isoformat(), **kw).to_df().reset_index()
                parts.append(df[["ts_event", "instrument_id", "high", "low", "close", "volume"]])
        out = pd.concat(parts, ignore_index=True).drop_duplicates("ts_event").sort_values("ts_event")
        from odte.timeutil import epoch_s
        out["ts"] = epoch_s(pd.to_datetime(out["ts_event"], utc=True))
        out.drop(columns="ts_event").to_parquet(p.with_suffix(".tmp"), index=False)
        p.with_suffix(".tmp").replace(p)
        print(f"  {r}: {len(out):,} bars -> {p}", flush=True)


def _et(day, hhmm) -> int:
    return int(pd.Timestamp(f"{pd.Timestamp(day):%Y-%m-%d} {hhmm}").tz_localize(TZ)
               .tz_convert("UTC").timestamp())


def es_features(days: list) -> pd.DataFrame:
    """One row per session (index = session date)."""
    p = OUT / "es_1m.parquet"
    days = sorted(pd.Timestamp(d).normalize() for d in days)
    if not p.exists():
        return pd.DataFrame(index=days, columns=ES_FEATURES, dtype=float)
    es = pd.read_parquet(p).sort_values("ts")
    t_end = es["ts"].to_numpy() + 60                       # a bar is known at its end
    inst = es["instrument_id"].to_numpy()
    lc = np.log(es["close"].to_numpy(float))
    r = np.diff(lc, prepend=lc[0])
    r[1:][inst[1:] != inst[:-1]] = 0.0                     # no return across a roll
    r[0] = 0.0
    hi, lo, vol = es["high"].to_numpy(float), es["low"].to_numpy(float), es["volume"].to_numpy(float)
    rows, night_vols = [], []
    for i, d in enumerate(days):
        f = {k: np.nan for k in ES_FEATURES}
        if i == 0:
            rows.append(f)
            continue
        a, b = _et(days[i - 1], "16:00"), _et(d, "09:30")
        j0, j1 = np.searchsorted(t_end, [a, b], side="right")
        if j1 - j0 > 60:
            seg_r = r[j0 + 1:j1]
            f["es_on_ret"] = float(seg_r.sum() * 1e4)
            f["es_on_rv"] = float(np.sqrt((seg_r ** 2).sum()) * 1e4)
            f["es_on_range"] = float((hi[j0:j1].max() - lo[j0:j1].min()) / np.exp(lc[j0]) * 1e4)
            v = float(vol[j0:j1].sum())
            f["es_on_vol"] = v / np.median(night_vols[-20:]) if len(night_vols) >= 5 else np.nan
            night_vols.append(v)
            k0 = np.searchsorted(t_end, _et(d, "09:15"), side="right")
            f["es_last15"] = float(r[k0 + 1:j1].sum() * 1e4) if j1 > k0 else np.nan
        rows.append(f)
    return pd.DataFrame(rows, index=days)[ES_FEATURES]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "fetch-cross"):
        f = sp.add_parser(name)
        f.add_argument("--start", default="2015-12-01")
        f.add_argument("--end", default="2026-11-07")
        f.add_argument("--max-usd", type=float, required=True)
    a = ap.parse_args()
    (fetch if a.cmd == "fetch" else fetch_cross)(a.start, a.end, a.max_usd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
