"""The multi-underlying combiner sums aligned daily P&L and fills gaps with 0."""
import json

import pandas as pd

from odte.eval import combine as cb


def _write(tmp, tag, strat, ref):
    rep = {"modes": {"fixed6": {"daily": {"S": strat, "always_short_fly": ref}}}}
    (tmp / f"study-{tag}.json").write_text(json.dumps(rep))


def test_combine_sums_by_date(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "OUT", tmp_path)
    _write(tmp_path, "a", {"2024-01-02": 100, "2024-01-03": -50, "2024-01-04": 20},
           {"2024-01-02": 10, "2024-01-03": 10, "2024-01-04": 10})
    _write(tmp_path, "b", {"2024-01-02": 30, "2024-01-04": 40},
           {"2024-01-02": 0, "2024-01-04": 0})
    out = cb.combine(["a", "b"], "S")
    assert out["days"] == 3 and out["total"] == 140
    assert out["per_underlying_total"] == {"a": 70.0, "b": 70.0}
    assert out["ref_total"] == 30
