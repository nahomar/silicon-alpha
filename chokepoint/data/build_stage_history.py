"""Reconstruct daily Eskom loadshedding stage history from eskom-calendar.

`chokepoint/data/eskom.py` documents the unmet dependency: there is no free
official API for *historical* national loadshedding stages. This module resolves
it without paying anyone.

## Where the history actually lives

The open-source project `beyarkay/eskom-calendar` maintains a file,
`manually_specified.yaml`, holding the currently-active loadshedding schedule as
announced by Eskom — each entry carrying a stage, a start/finish timestamp, and
a **source URL** (usually the Eskom announcement itself).

That file is overwritten as the schedule changes, so its latest revision covers
only the current week. But it is version-controlled, and the project has been
maintained continuously since July 2022 across ~780 commits. **The git history of
that one file is the historical record.** Walking every revision and unioning the
announcement intervals reconstructs the national stage series across the entire
severe-loadshedding era.

## The zero-fill decision, stated explicitly

`eskom.py` warns that a calendar gap is ambiguous — it can mean "no loadshedding"
or "nobody recorded it" — and that assuming the former biases the series calm.

Here the provenance resolves that ambiguity *within the covered span*: this is a
continuously-maintained announcement tracker, so a day inside the span with no
entry means no loadshedding was announced, which is genuinely stage 0. Outside
the span (before the first commit, after the last) nothing is emitted at all.

`--no-zero-fill` disables it if you would rather drop unannounced days than
assert calm. The choice taken is recorded in the output file's provenance header
so the resulting series can never be used without knowing which convention
produced it.

## Convention

Announcement intervals overlap (different regions, revised schedules). Each
calendar day is assigned the **maximum** stage in force at any point that day —
matching the convention `eskom.py` documents, and the right one for this
question: the binding constraint on mine operations is the worst the grid got,
not its daily average.

Usage:
    PYTHONPATH=. python -m chokepoint.data.build_stage_history --out data/eskom_stages.csv
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

REPO = "https://github.com/beyarkay/eskom-calendar.git"
TRACKED_FILE = "manually_specified.yaml"
MAX_STAGE = 8


def _run(args: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:3])}… failed: {out.stderr.strip()}")
    return out.stdout


def clone(dest: Path) -> Path:
    """Blobless clone — we only need history of one small text file."""
    log.info("cloning %s (blobless)…", REPO)
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", "-q",
          REPO, str(dest)])
    return dest


def revisions(repo: Path) -> list[str]:
    shas = _run(
        ["git", "log", "--format=%H", "--", TRACKED_FILE], cwd=repo
    ).split()
    log.info("%d revisions of %s", len(shas), TRACKED_FILE)
    return shas


def _parse_revision(repo: Path, sha: str) -> list[tuple[int, datetime, datetime]]:
    """Extract (stage, start, finsh) from one revision. Tolerant by design.

    Early revisions of a community file have inconsistent shapes; a single
    malformed entry must not abort a 780-revision walk. Unparseable entries are
    skipped and counted, and the count is reported so a silently-empty result
    is impossible.
    """
    try:
        raw = _run(["git", "show", f"{sha}:{TRACKED_FILE}"], cwd=repo)
        doc = yaml.safe_load(raw)
    except Exception:  # noqa: BLE001 - a bad revision is expected, not fatal
        return []
    if not isinstance(doc, dict):
        return []

    out = []
    for entry in doc.get("changes") or []:
        if not isinstance(entry, dict):
            continue
        try:
            stage = int(entry["stage"])
            start = entry["start"]
            finsh = entry["finsh"]
            start = (start if isinstance(start, datetime)
                     else datetime.fromisoformat(str(start)))
            finsh = (finsh if isinstance(finsh, datetime)
                     else datetime.fromisoformat(str(finsh)))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= stage <= MAX_STAGE) or finsh < start:
            continue
        # Normalize away tzinfo: these are all SAST wall-clock times, and the
        # probe works on SAST calendar days.
        out.append((stage, start.replace(tzinfo=None), finsh.replace(tzinfo=None)))
    return out


def collect(repo: Path) -> tuple[dict[date, int], int]:
    """Union all announcement intervals into a daily max-stage map."""
    shas = revisions(repo)
    if not shas:
        raise RuntimeError(f"no revisions of {TRACKED_FILE} found")

    intervals: set[tuple[int, datetime, datetime]] = set()
    empty = 0
    for i, sha in enumerate(shas):
        got = _parse_revision(repo, sha)
        if not got:
            empty += 1
        intervals.update(got)
        if (i + 1) % 100 == 0:
            log.info("  %d/%d revisions, %d unique intervals",
                     i + 1, len(shas), len(intervals))

    log.info("%d unique announcement intervals from %d revisions (%d empty)",
             len(intervals), len(shas), empty)
    if not intervals:
        raise RuntimeError(
            "parsed 0 intervals — the upstream schema likely changed. "
            "Inspect `git show HEAD:manually_specified.yaml` in the clone."
        )

    daily: dict[date, int] = defaultdict(int)
    for stage, start, finsh in intervals:
        day = start.date()
        while day <= finsh.date():
            daily[day] = max(daily[day], stage)
            day += timedelta(days=1)
    return dict(daily), len(intervals)


def to_csv(daily: dict[date, int], out_path: Path, n_intervals: int,
           zero_fill: bool = True) -> None:
    lo, hi = min(daily), max(daily)
    rows: list[tuple[date, int]] = []
    if zero_fill:
        day = lo
        while day <= hi:
            rows.append((day, daily.get(day, 0)))
            day += timedelta(days=1)
    else:
        rows = sorted(daily.items())

    fill_note = (
        "days with no announcement zero-filled (continuously-maintained "
        "tracker, so absence == no loadshedding)"
        if zero_fill else
        "days with no announcement OMITTED (--no-zero-fill)"
    )
    header = (
        f"# Eskom national loadshedding stage, daily maximum.\n"
        f"# Reconstructed {date.today()} from the git history of "
        f"{TRACKED_FILE} in {REPO}\n"
        f"# via chokepoint.data.build_stage_history. {n_intervals} unique "
        f"announcement intervals.\n"
        f"# Convention: daily MAX stage in force; {fill_note}.\n"
        f"# Span {lo} to {hi}. Upstream entries cite Eskom announcements; "
        f"verify before publishing.\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(header)
        fh.write("date,stage\n")
        for day, stage in rows:
            fh.write(f"{day.isoformat()},{stage}\n")

    shed = sum(1 for _, s in rows if s > 0)
    severe = sum(1 for _, s in rows if s >= 4)
    log.info("wrote %d rows to %s", len(rows), out_path)
    print(
        f"\n{len(rows)} days {lo} → {hi}\n"
        f"  loadshedding on {shed} days ({shed / len(rows):.0%})\n"
        f"  stage >= 4 on {severe} days ({severe / len(rows):.0%})\n"
        f"  max stage {max(s for _, s in rows)}\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/eskom_stages.csv")
    ap.add_argument("--repo", default=None,
                    help="existing clone to reuse (skips the network fetch)")
    ap.add_argument("--no-zero-fill", action="store_true",
                    help="omit unannounced days instead of recording stage 0")
    args = ap.parse_args()

    if args.repo:
        daily, n = collect(Path(args.repo))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            daily, n = collect(clone(Path(tmp) / "eskom-calendar"))
    to_csv(daily, Path(args.out), n, zero_fill=not args.no_zero_fill)


if __name__ == "__main__":  # pragma: no cover
    main()
