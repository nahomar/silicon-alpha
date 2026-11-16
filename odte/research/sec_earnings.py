"""Earnings-release dates from SEC EDGAR (docs/sweep_plan.md, sweep 3 family E): 8-K filings with Item 2.02
and their acceptance times, for the most traded US stocks in data/prices/xnas_daily.parquet.

    python3 -m odte.research.sec_earnings fetch     # -> data/filings/earnings_8k.parquet  (~10 min, polite rate)

The SEC asks automated clients to identify themselves; the User-Agent is read from
~/.config/silicon-alpha/sec_user_agent and never printed or stored in the repo.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "filings" / "earnings_8k.parquet"
RAW = ROOT / "data" / "filings" / "submissions"
UA = (Path.home() / ".config" / "silicon-alpha" / "sec_user_agent").read_text().strip()
N_SYMBOLS, SINCE, GAP = 2500, "2017-06-01", 0.13          # <= ~8 requests a second


def get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"failed: {url}")


def symbols() -> list[str]:
    x = pd.read_parquet(ROOT / "data" / "prices" / "xnas_daily.parquet", columns=["symbol", "close", "volume"])
    dv = (x["close"] * x["volume"]).groupby(x["symbol"]).mean()
    return list(dv.sort_values(ascending=False).index[:N_SYMBOLS])


def rows(block: dict) -> pd.DataFrame:
    keep = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return pd.DataFrame({k: block.get(k, []) for k in keep})


def fetch() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    tick = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    cik_of = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in tick.values()}
    syms = symbols()
    want = {}
    for s in syms:
        c = cik_of.get(s.upper()) or cik_of.get(s.upper().replace(".", "-"))
        if c:
            want.setdefault(c, []).append(s)
    print(f"{len(syms)} symbols, {len(want)} matched to a CIK", flush=True)
    out = []
    for i, (cik, ss) in enumerate(want.items()):
        f = RAW / f"CIK{cik:010d}.json"
        try:
            if not f.exists():
                f.write_bytes(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
                time.sleep(GAP)
            j = json.loads(f.read_bytes())
            parts = [rows(j["filings"]["recent"])]
            for extra in j["filings"].get("files", []):
                if extra.get("filingTo", "9999") < SINCE:
                    continue
                g = RAW / extra["name"]
                if not g.exists():
                    g.write_bytes(get(f"https://data.sec.gov/submissions/{extra['name']}"))
                    time.sleep(GAP)
                parts.append(rows(json.loads(g.read_bytes())))
            d = pd.concat(parts)
            d = d[(d["form"] == "8-K") & d["items"].fillna("").str.contains("2.02", regex=False)
                  & (d["filingDate"] >= SINCE)]
            d["cik"], d["symbols"] = cik, ",".join(ss)
            out.append(d)
        except Exception as e:                                   # noqa: BLE001
            print(f"CIK {cik} {ss}: {type(e).__name__}", flush=True)
        if i % 250 == 0:
            print(f"{i}/{len(want)}", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT)
    print(f"done: {len(res)} Item 2.02 8-Ks from {res['cik'].nunique()} companies", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(fetch() if sys.argv[1:] == ["fetch"] else 0)
