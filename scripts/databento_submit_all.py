"""Submit the Phase-2 Databento batch jobs in parallel.

Fires 5 OPRA cmbp-1 batch jobs (SPX.OPT + SPXW.OPT parent symbology)
for the chosen regime-stratified corpus. No data is downloaded locally
— all downloading happens later on Modal.

Idempotent: rerunning this after jobs are already submitted just re-reads
the existing job IDs from Databento. No double-charging.

Prints a job-id manifest to stdout AND writes it to
`scripts/databento_jobs.json` so the Modal packer can consume it.

Usage:
    cd /Users/nahom/silicon-alpha && set -a && . ./.env && set +a && \
    python3 scripts/databento_submit_all.py
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import databento as db

# ---------------------------------------------------------------------------
# CORPUS DEFINITION — synced with scripts/databento_cost_estimate.py
# ---------------------------------------------------------------------------

DATASET = "OPRA.PILLAR"
SCHEMA = "cmbp-1"
STYPE_IN = "parent"
SYMBOLS = ["SPX.OPT", "SPXW.OPT"]

DAYS = [
    ("2026-04-16", "held-out eval (day before OPEX)",      "eval"),
    ("2026-04-17", "April OPEX Friday",                    "train"),
    ("2026-04-21", "Tuesday (heavy 166 GB)",               "train"),
    ("2026-04-22", "Wednesday",                            "train"),
    ("2026-04-23", "Thursday (latest available)",          "train"),
]

MANIFEST_PATH = Path(__file__).parent / "databento_jobs.json"


def _find_existing(client, start: str, end: str) -> Optional[dict]:
    """Return the most recent matching job, if any, across all states."""
    # Search common states where our job could live.
    for st in ("done", "processing", "queued", "received"):
        try:
            jobs = client.batch.list_jobs(states=[st])
        except Exception:
            continue
        for j in jobs:
            if (j.get("dataset") == DATASET
                    and j.get("schema") == SCHEMA
                    and j.get("stype_in") == STYPE_IN
                    and sorted(j.get("symbols") or []) == sorted(SYMBOLS)
                    and (j.get("start") or "").startswith(start)
                    and (j.get("end") or "").startswith(end)):
                return j
    return None


def main() -> None:
    if not os.environ.get("DATABENTO_API_KEY"):
        print("ERROR: DATABENTO_API_KEY not set; source .env first.",
              file=sys.stderr)
        sys.exit(1)

    client = db.Historical()
    manifest: list[dict] = []

    print(f"Submitting {len(DAYS)} batch jobs "
          f"({DATASET} {SCHEMA} parent {SYMBOLS})...")
    print()

    for day, note, split in DAYS:
        d0 = _dt.date.fromisoformat(day)
        d1 = d0 + _dt.timedelta(days=1)
        start = f"{d0.isoformat()}T00:00:00"
        end = f"{d1.isoformat()}T00:00:00"

        print(f"[{day}] ({split:<5}) {note}")
        # Idempotent check: if a matching job already exists we reuse it.
        existing = _find_existing(client, start, end)
        if existing:
            print(f"    reusing existing job {existing.get('id')} "
                  f"state={existing.get('state')}")
            manifest.append({
                "day": day, "note": note, "split": split,
                "job_id": existing["id"],
                "state": existing.get("state"),
                "cost_usd": existing.get("cost_usd"),
                "billed_size": existing.get("billed_size"),
                "reused": True,
            })
            continue

        try:
            job = client.batch.submit_job(
                dataset=DATASET, schema=SCHEMA,
                start=start, end=end,
                symbols=SYMBOLS, stype_in=STYPE_IN,
                encoding="dbn", compression="zstd",
                split_duration="day",
                delivery="download",
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            manifest.append({
                "day": day, "note": note, "split": split,
                "job_id": None, "state": "submit_failed",
                "error": str(e),
            })
            continue

        print(f"    submitted job {job.get('id')} "
              f"cost=${float(job.get('cost_usd') or 0):.2f} "
              f"size={float(job.get('billed_size') or 0) / 1e9:.1f} GB")
        manifest.append({
            "day": day, "note": note, "split": split,
            "job_id": job["id"],
            "state": job.get("state"),
            "cost_usd": job.get("cost_usd"),
            "billed_size": job.get("billed_size"),
            "reused": False,
        })
        # Be polite — don't hammer the submit endpoint.
        time.sleep(0.5)

    print()
    print("=== MANIFEST ===")
    total_cost = 0.0
    for row in manifest:
        jid = row["job_id"] or "FAILED"
        c = row.get("cost_usd") or 0.0
        total_cost += float(c)
        print(f"  {row['day']}  {row['split']:<5}  ${float(c):>6.2f}  "
              f"{jid}  ({row['note']})")
    print()
    print(f"Total (Databento-reported): ${total_cost:.2f}")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Manifest written to {MANIFEST_PATH}")
    print()
    print("Next steps:")
    print("  1. Wait 20-60 min for Databento to process the batches.")
    print("  2. Check status: python3 scripts/databento_status.py")
    print("  3. Once all 'done', run the Modal packer:")
    print("     python3 -m modal run infra/modal/phase2_smoke.py::dryrun_pack_all")


if __name__ == "__main__":
    main()
