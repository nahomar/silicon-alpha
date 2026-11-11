"""Polite EDGAR client (SEC fair-access policy: declared User-Agent with contact, <= 10 requests/s).

The User-Agent line comes from ~/.config/silicon-alpha/sec_user_agent (owner-only file).
If the SEC answers 403 the client stops: it never retries around a block.
"""
from __future__ import annotations

import gzip
import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "filings"
UA_FILE = Path.home() / ".config" / "silicon-alpha" / "sec_user_agent"
INTERVAL = 0.2                      # 5 requests/s, half the SEC limit
FORMS = ("10-K", "10-Q", "8-K", "10-K/A", "10-Q/A")


class Blocked(RuntimeError):
    pass


class Edgar:
    def __init__(self, interval: float = INTERVAL):
        ua = UA_FILE.read_text().strip() if UA_FILE.exists() else ""
        if "@" not in ua:
            raise RuntimeError(f"missing contact User-Agent in {UA_FILE}")
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.interval, self.last, self.calls = interval, 0.0, 0

    def get(self, url: str, binary: bool = False, retries: int = 6):
        for attempt in range(retries):
            wait = self.interval - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.calls += 1
            try:
                r = self.s.get(url, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r.content if binary else r.text
            if r.status_code == 403:
                raise Blocked(f"403 from SEC for {url}: stopping, not retrying around a block")
            if r.status_code == 404:
                return None
            time.sleep(2 ** attempt * (5 if r.status_code == 429 else 1))
        raise RuntimeError(f"failed after {retries} tries: {url}")


def parse_master(text: str) -> pd.DataFrame:
    """EDGAR master.idx: CIK|Company Name|Form Type|Date Filed|Filename after a dashed header."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("-----")) + 1
    rows = [l.split("|") for l in lines[start:] if l.count("|") == 4]
    d = pd.DataFrame(rows, columns=["cik", "company", "form", "filed", "path"])
    d["cik"] = d["cik"].astype(int)
    d["filed"] = pd.to_datetime(d["filed"])
    d["accession"] = d["path"].str.extract(r"(\d{10}-\d{2}-\d{6})")[0]
    return d


def fetch_index(years, forms=FORMS) -> pd.DataFrame:
    """Quarterly master indexes, kept only for `forms`, saved compactly to data/filings/index.parquet."""
    ed = Edgar()
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "index.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done = set(zip(have["filed"].dt.year, have["filed"].dt.quarter)) if len(have) else set()
    parts = [have] if len(have) else []
    for y in years:
        for q in (1, 2, 3, 4):
            if (y, q) in done or pd.Timestamp(year=y, month=3 * q - 2, day=1) > pd.Timestamp.now():
                continue
            raw = ed.get(f"https://www.sec.gov/Archives/edgar/full-index/{y}/QTR{q}/master.gz", binary=True)
            if raw is None:
                continue
            d = parse_master(gzip.GzipFile(fileobj=io.BytesIO(raw)).read().decode("latin-1"))
            parts.append(d[d["form"].isin(forms)])
            print(f"{y}Q{q}: {int(d['form'].isin(forms).sum()):,} filings kept ({ed.calls} calls)", flush=True)
            pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["index"]:
        y0, y1 = int(sys.argv[2]), int(sys.argv[3])
        d = fetch_index(range(y0, y1 + 1))
        print(d.groupby([d["filed"].dt.year, "form"]).size().unstack(fill_value=0).to_string())
