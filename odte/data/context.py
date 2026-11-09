"""Daily cross-asset context for SPX 0DTE sessions, known before the open.

These are the inputs a vol desk glances at before the bell. The most
directly relevant is VIX1D: Cboe's 1-day implied vol of SPX, which at
yesterday's close is the market's own forecast of today's move, the same
quantity this study's straddle measures intraday.

  vol surface   VIX1D, VIX9D, VIX, VIX3M (term structure), VVIX (vol of
                vol), SKEW (tail pricing)
  cross-asset   10y yield (^TNX), dollar index, crude, gold, high-yield
                credit (HYG), Bitcoin
  SPX itself    previous return, 5- and 20-day realized vol

Leakage rule: session d sees only bars dated strictly before d. That holds
for every series here, including the 24/7 ones. Yahoo's BTC-USD bar for
d-1 closes at 00:00 UTC on d, which is 19:00-20:00 ET on d-1, before d's
open.

Source: Yahoo via yfinance (free). Cached at data/context/daily.parquet.

Usage:
  python3 -m odte.data.context fetch --start 2025-09-01 --end 2026-11-30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "context"
TICKERS = {
    "vix1d": "^VIX1D", "vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M",
    "vvix": "^VVIX", "skew": "^SKEW", "tnx": "^TNX", "dxy": "DX-Y.NYB",
    "crude": "CL=F", "gold": "GC=F", "hyg": "HYG", "btc": "BTC-USD",
    "spx": "^GSPC",
}
CTX_FEATURES = [
    "vix1d_prev", "vix9d_prev", "vix_prev", "vvix_prev", "skew_prev",
    "vix1d_over_vix", "vix9d_over_vix", "vix_over_vix3m", "vix_chg",
    "tnx_chg", "dxy_ret", "crude_ret", "gold_ret", "hyg_ret", "btc_ret",
    "spx_ret_prev", "spx_rv5", "spx_rv20",
]


def fetch(start, end, save: bool = True) -> pd.DataFrame:
    import yfinance as yf
    cols = {}
    for name, tk in TICKERS.items():
        h = yf.Ticker(tk).history(start=start, end=end, auto_adjust=False)
        if h.empty:
            print(f"  {tk}: no data")
            continue
        s = h["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        cols[name] = s[~s.index.duplicated(keep="last")]
    wide = pd.DataFrame(cols).sort_index()
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        wide.to_parquet(OUT / "daily.parquet")
    return wide


def refresh(lookback_days: int = 30) -> pd.DataFrame:
    """Update the last `lookback_days` of closes without dropping older history
    (fetch() alone would overwrite the file with only its own range)."""
    today = pd.Timestamp.today().normalize()
    new = fetch(today - pd.Timedelta(days=lookback_days), today + pd.Timedelta(days=1), save=False)
    p = OUT / "daily.parquet"
    old = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    wide = new.combine_first(old).sort_index() if len(old) else new
    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(p)
    return wide


def load() -> pd.DataFrame:
    p = OUT / "daily.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run: python3 -m odte.data.context fetch")
    return pd.read_parquet(p)


def _last_two(s: pd.Series, day: pd.Timestamp):
    s = s[s.index < day].dropna()
    if len(s) < 2:
        return (float(s.iloc[-1]) if len(s) else np.nan), np.nan
    return float(s.iloc[-1]), float(s.iloc[-2])


def features_for(wide: pd.DataFrame, day: pd.Timestamp) -> dict:
    """Context features for session `day`, from bars dated strictly before it."""
    day = pd.Timestamp(day).normalize()
    g = {}
    for c in wide.columns:
        g[c] = _last_two(wide[c], day)
    out = {f"{k}_prev": g[k][0] if k in g else np.nan
           for k in ("vix1d", "vix9d", "vix", "vvix", "skew")}
    vix = g.get("vix", (np.nan, np.nan))[0]
    out["vix1d_over_vix"] = g.get("vix1d", (np.nan,))[0] / vix if vix else np.nan
    out["vix9d_over_vix"] = g.get("vix9d", (np.nan,))[0] / vix if vix else np.nan
    v3 = g.get("vix3m", (np.nan,))[0]
    out["vix_over_vix3m"] = vix / v3 if v3 else np.nan

    def lr(k):
        a, b = g.get(k, (np.nan, np.nan))
        return float(np.log(a / b)) if a and b and a > 0 and b > 0 else np.nan
    out["vix_chg"] = lr("vix")
    a, b = g.get("tnx", (np.nan, np.nan))
    # ^TNX has been quoted both as yield x10 (41.5) and in percent (4.15);
    # detect which, and express the change in basis points either way.
    scale = 10.0 if np.isfinite(a) and a > 20 else 100.0
    out["tnx_chg"] = (a - b) * scale if np.isfinite(a) and np.isfinite(b) else np.nan
    for k in ("dxy", "crude", "gold", "hyg", "btc"):
        out[f"{k}_ret"] = lr(k)
    spx = wide["spx"][wide.index < day].dropna() if "spx" in wide else pd.Series(dtype=float)
    r = np.diff(np.log(spx.to_numpy())) if len(spx) > 1 else np.array([])
    out["spx_ret_prev"] = float(r[-1]) if len(r) else np.nan
    out["spx_rv5"] = float(r[-5:].std(ddof=1) * np.sqrt(252)) if len(r) >= 5 else np.nan
    out["spx_rv20"] = float(r[-20:].std(ddof=1) * np.sqrt(252)) if len(r) >= 20 else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fetch")
    f.add_argument("--start", default="2025-09-01")
    f.add_argument("--end", default="2026-11-30")
    a = ap.parse_args()
    wide = fetch(a.start, a.end)
    print(f"{wide.shape[0]} days x {wide.shape[1]} series -> {OUT/'daily.parquet'}")
    print(wide.notna().sum().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
