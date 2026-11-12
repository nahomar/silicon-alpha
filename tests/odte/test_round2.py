import numpy as np
import pandas as pd

from odte.research import round2_forward as rf


def test_choose_and_late_rule():
    assert rf._choose({"fly": 5.0, "condor": 9.0, "put_spread": -1.0, "call_spread": 0.0}, False) == "condor"
    assert rf._choose({"fly": 5.0}, True) == "none"
    assert rf._choose({"fly": -5.0}, False) == "none"


def test_head_to_head_outcome_needs_44_sessions():
    days = pd.bdate_range("2026-11-12", periods=44)
    s = pd.DataFrame({"day": days, "index": "SPX", "hhmm": "10:00",
                      "B0_half": 0.0, "C7_half": 10.0, "B0_touch": 0.0, "C7_touch": 10.0})
    assert rf.head_to_head(s.head(10))["outcome"].startswith("in progress")
    assert rf.head_to_head(s)["outcome"] == "WIN"
