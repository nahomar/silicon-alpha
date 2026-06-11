"""Poll Databento for the status of the Phase-2 corpus batch jobs.

Reads `scripts/databento_jobs.json` (written by databento_submit_all.py)
and prints the current state of each job. Run periodically until all
are `done`.

Usage:
    cd /Users/nahom/silicon-alpha && set -a && . ./.env && set +a && \
    python3 scripts/databento_status.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import databento as db

MANIFEST_PATH = Path(__file__).parent / "databento_jobs.json"


def main() -> None:
    if not os.environ.get("DATABENTO_API_KEY"):
        print("ERROR: DATABENTO_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    if not MANIFEST_PATH.exists():
        print(f"ERROR: no manifest at {MANIFEST_PATH}; "
              f"run databento_submit_all.py first.", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text())
    target_ids = {row["job_id"] for row in manifest if row.get("job_id")}

    client = db.Historical()
    # Fetch every job we might care about across all queryable states.
    jobs_by_id: dict[str, dict] = {}
    for st in ("done", "processing", "queued", "received", "expired"):
        try:
            for j in client.batch.list_jobs(states=[st]):
                if j.get("id") in target_ids:
                    jobs_by_id[j["id"]] = j
        except Exception:
            continue

    print(f"Job status across {len(manifest)} days:")
    print()
    all_done = True
    for row in manifest:
        jid = row.get("job_id")
        if not jid:
            print(f"  {row['day']}  (NO JOB — {row.get('error', 'unknown')})")
            all_done = False
            continue
        j = jobs_by_id.get(jid)
        state = (j.get("state") if j else "unknown") or "unknown"
        progress = (j.get("progress") if j else None) or 0
        cost = float((j.get("cost_usd") if j else None) or 0)
        size_gb = float((j.get("billed_size") if j else None) or 0) / 1e9
        mark = "✓" if state == "done" else "…" if state in ("queued", "processing", "received") else "✗"
        if state != "done":
            all_done = False
        print(f"  {mark}  {row['day']}  {row['split']:<5}  {state:<12}  "
              f"progress={int(progress):>3}%  ${cost:>6.2f}  {size_gb:>6.2f} GB  "
              f"{jid}")

    print()
    if all_done:
        print("ALL DONE. Fire the Modal packer:")
        print("  python3 -m modal run infra/modal/phase2_smoke.py::dryrun_pack_all")
    else:
        print("Not yet — re-run this script in a few minutes.")


if __name__ == "__main__":
    main()
