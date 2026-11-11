"""Exam lock: no research code may read dates >= EXAM_FROM unless the final model is frozen.

Unlock only by setting EXAM_UNLOCK to the freeze commit written in docs/master_plan.md.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

EXAM_FROM = pd.Timestamp("2026-09-08")
EXAM_TO = pd.Timestamp("2026-11-06")
PLAN = Path(__file__).resolve().parents[2] / "docs" / "master_plan.md"


class ExamLocked(RuntimeError):
    pass


def freeze_commit() -> str | None:
    m = re.search(r"Freeze commit:\s*([0-9a-f]{7,40})", PLAN.read_text()) if PLAN.exists() else None
    return m.group(1) if m else None


def unlocked() -> bool:
    fc = freeze_commit()
    return bool(fc) and os.environ.get("EXAM_UNLOCK", "") == fc


def check(days) -> None:
    """Raise if any of `days` is in the exam window and the exam is not unlocked."""
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(days)).dropna().unique())
    if len(d) and (d >= EXAM_FROM).any() and not unlocked():
        raise ExamLocked(f"{int((d >= EXAM_FROM).sum())} exam day(s) requested; the exam is locked until "
                         "the freeze commit is recorded in docs/master_plan.md and EXAM_UNLOCK matches it")


def dev_days(days) -> list:
    """Filter to development days (always allowed)."""
    return [x for x in days if pd.Timestamp(x) < EXAM_FROM]
