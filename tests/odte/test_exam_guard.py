import pandas as pd
import pytest

from odte.research import exam_guard as eg


def test_exam_days_are_refused_without_unlock(monkeypatch):
    monkeypatch.delenv("EXAM_UNLOCK", raising=False)
    eg.check(["2026-09-04", "2025-01-02"])                       # development: fine
    with pytest.raises(eg.ExamLocked):
        eg.check(["2026-09-04", "2026-09-08"])
    assert eg.dev_days(["2026-09-04", "2026-09-08"]) == ["2026-09-04"]


def test_unlock_needs_the_recorded_freeze_commit(monkeypatch):
    monkeypatch.setattr(eg, "freeze_commit", lambda: "abc1234")
    monkeypatch.setenv("EXAM_UNLOCK", "wrong")
    with pytest.raises(eg.ExamLocked):
        eg.check(["2026-10-01"])
    monkeypatch.setenv("EXAM_UNLOCK", "abc1234")
    eg.check(["2026-10-01"])
