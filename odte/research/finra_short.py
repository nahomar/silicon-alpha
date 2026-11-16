"""FINRA Reg SHO daily short-sale volume (consolidated CNMS files), for sweep 5 family Q.

    python3 -m odte.research.finra_short fetch     # -> data/short/finra_daily.parquet (date, symbol, short, total)
"""
from __future__ import annotations

import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from odte.research.partner_t import nyse_days

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "short"
URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
HDR = {"User-Agent": "python-research-client/1.0"}          # the CDN refuses Python's default client name


def fetch() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts, missing = [], []
    days = nyse_days("2018-06-01", "2026-08-31")
    for i, d in enumerate(days):
        cache = OUT / "raw" / f"{d:%Y%m%d}.parquet"
        cache.parent.mkdir(exist_ok=True)
        if cache.exists():
            parts.append(pd.read_parquet(cache))
            continue
        try:
            raw = urllib.request.urlopen(urllib.request.Request(URL.format(d=f"{d:%Y%m%d}"), headers=HDR), timeout=30).read()
        except urllib.error.HTTPError:
            missing.append(d.date().isoformat())
            continue
        except Exception:                                        # noqa: BLE001
            time.sleep(5)
            missing.append(d.date().isoformat())
            continue
        x = pd.read_csv(io.BytesIO(raw), sep="|", dtype={"Symbol": str})
        x = x[pd.to_numeric(x["Date"], errors="coerce").notna()]   # drop the trailer line
        x = pd.DataFrame({"date": d, "symbol": x["Symbol"].astype(str),
                          "short": pd.to_numeric(x["ShortVolume"], errors="coerce"),
                          "total": pd.to_numeric(x["TotalVolume"], errors="coerce")})
        x.to_parquet(cache)
        parts.append(x)
        time.sleep(0.25)
        if i % 200 == 0:
            print(f"{i}/{len(days)} {d.date()}", flush=True)
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(OUT / "finra_daily.parquet")
    print(f"done: {len(res)} rows, {res['date'].nunique()} days; missing {len(missing)}: "
          f"{missing[:3]} .. {missing[-3:]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(fetch() if sys.argv[1:] == ["fetch"] else 0)
