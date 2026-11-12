"""Exam lock: no research code may read dates >= EXAM_FROM unless the final model is frozen.

Unlock only by setting EXAM_UNLOCK to the freeze commit written in docs/master_plan.md.

Once the one-time exam has been run (docs/exam_result.json exists) its window is development
data. Days after EXAM_TO belong to round 2's forward test (docs/round2_plan.md) and stay locked
unless ROUND2_UNLOCK equals the challenger freeze commit written in that plan.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

EXAM_FROM = pd.Timestamp("2026-09-08")
EXAM_TO = pd.Timestamp("2026-11-06")
PLAN = Path(__file__).resolve().parents[2] / "docs" / "master_plan.md"
RESULT = Path(__file__).resolve().parents[2] / "docs" / "exam_result.json"
ROUND2 = Path(__file__).resolve().parents[2] / "docs" / "round2_plan.md"


class ExamLocked(RuntimeError):
    pass


def freeze_commit() -> str | None:
    m = re.search(r"Freeze commit:\s*([0-9a-f]{7,40})", PLAN.read_text()) if PLAN.exists() else None
    return m.group(1) if m else None


def unlocked() -> bool:
    fc = freeze_commit()
    return bool(fc) and os.environ.get("EXAM_UNLOCK", "") == fc


def spent() -> bool:
    """The one-time exam has been run: its window is now development data."""
    return RESULT.exists()


def round2_unlocked() -> bool:
    m = re.search(r"Challenger freeze commit:\s*([0-9a-f]{7,40})", ROUND2.read_text()) if ROUND2.exists() else None
    return bool(m) and os.environ.get("ROUND2_UNLOCK", "") == m.group(1)


def dev_end() -> pd.Timestamp:
    """Last development day."""
    return EXAM_TO if spent() else EXAM_FROM - pd.Timedelta(days=1)


def check(days) -> None:
    """Raise if any of `days` is locked: the exam window before the exam is run, and any day
    after it (round 2's forward test) unless round 2 is unlocked."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(days)).dropna().unique())
    if not len(d):
        return
    if (d > EXAM_TO).any() and not round2_unlocked():
        raise ExamLocked(f"{int((d > EXAM_TO).sum())} day(s) after {EXAM_TO:%Y-%m-%d} requested: round 2's forward "
                         "test is locked until the challenger freeze is recorded in docs/round2_plan.md")
    if (d >= EXAM_FROM).any() and not (spent() or unlocked()):
        raise ExamLocked(f"{int((d >= EXAM_FROM).sum())} exam day(s) requested; the exam is locked until "
                         "the freeze commit is recorded in docs/master_plan.md and EXAM_UNLOCK matches it")


def dev_days(days) -> list:
    """Filter to development days (always allowed)."""
    return [x for x in days if pd.Timestamp(x) <= dev_end()]
