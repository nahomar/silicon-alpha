"""Crawl 10-K primary documents and keep only text fingerprints (resumable).

  python3 -m filings.crawl tenk 2017-01-01
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from filings import text as tx
from filings.edgar import DATA, Edgar

OUT = DATA / "tenk_features.parquet"


def primary_docs(ed: Edgar, cik: int, wanted: set) -> dict:
    """accession -> primary document name, from the submissions API (recent + older pages if needed)."""
    js = ed.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if not js:
        return {}
    import json
    d = json.loads(js)
    out = {}

    def take(block):
        for acc, doc in zip(block.get("accessionNumber", []), block.get("primaryDocument", [])):
            if acc in wanted:
                out[acc] = doc
    take(d.get("filings", {}).get("recent", {}))
    for f in d.get("filings", {}).get("files", []):
        if wanted - set(out):
            more = ed.get(f"https://data.sec.gov/submissions/{f['name']}")
            if more:
                take(json.loads(more))
    return out


def crawl_tenk(since: str) -> None:
    idx = pd.read_parquet(DATA / "index.parquet")
    tenk = idx[(idx["form"] == "10-K") & (idx["filed"] >= pd.Timestamp(since))]
    have = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["accession"])
    done = set(have["accession"])
    todo = tenk[~tenk["accession"].isin(done)]
    print(f"10-Ks since {since}: {len(tenk):,}; done {len(done):,}; to do {len(todo):,}", flush=True)
    ed = Edgar()
    rows, n = [], 0
    for cik, g in todo.groupby("cik"):
        docs = primary_docs(ed, int(cik), set(g["accession"]))
        for _, r in g.iterrows():
            doc = docs.get(r["accession"])
            rec = {"cik": int(cik), "accession": r["accession"], "filed": r["filed"], "company": r["company"]}
            if doc:
                html = ed.get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                              f"{r['accession'].replace('-', '')}/{doc}")
                if html:
                    f = tx.features(html)
                    rec.update({k: f[k] for k in ("n_words", "n_unique", "ticker", "exchange")})
                    rec["minhash"] = f["minhash"].tobytes()
            rows.append(rec)
            n += 1
        if len(rows) >= 100:
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            have.to_parquet(OUT, index=False)
            rows = []
            print(f"{len(have):,} done ({ed.calls:,} calls)", flush=True)
    if rows:
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(OUT, index=False)
    print("crawl done", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["tenk"]:
        crawl_tenk(sys.argv[2] if len(sys.argv) > 2 else "2017-01-01")
