"""The lockbox splits half-years and refuses to reopen a block."""
import json

import pytest

from odte.eval import lockbox as lb


def test_half_year_blocks_clip_to_range():
    b = lb.blocks("2016-01-04", "2017-03-31")
    assert [x["block"] for x in b] == ["2016H1", "2016H2", "2017H1"]
    assert b[0]["start"] == "2016-01-04" and b[-1]["end"] == "2017-03-31"


def test_reopening_a_block_is_refused(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps([{"data": "odte_0dte", "block": "2016H2",
                                   "opened_at": "2026-11-08 20:00", "commit": "abc1234"}]))
    monkeypatch.setattr(lb, "LEDGER", ledger)
    with pytest.raises(SystemExit, match="REFUSED"):
        lb.open_block("data/odte_0dte", "2016H2", "x", "2016-01-01", "2022-05-13")
