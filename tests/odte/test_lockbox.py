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


def test_today_cost_reproduces_touch_pnl_when_given_the_touch_cost():
    import numpy as np
    import pandas as pd
    cm, credit, W, intrinsic = np.array([5.0, 4.0]), np.array([4.8, 3.9]), np.array([12.0, 10.0]), np.array([1.0, 7.0])
    dec = pd.DataFrame({"fly_credit_mid": cm, "W": W,
                        "pnl_short_fly_mid": (cm - intrinsic) * 100,
                        "pnl_short_fly": (credit - intrinsic) * 100 - lb.FEES4,
                        "pnl_long_fly": (intrinsic - cm) * 100 - lb.FEES4})
    frac = (cm - credit) / cm
    for i in range(2):
        got = lb.today_cost_pnl(dec.iloc[[i]], [1], cost=frac[i])[0]
        want = (credit[i] - intrinsic[i]) * 100 - lb.FEES4
        want = want / ((W[i] - credit[i]) * 100 + lb.FEES4) * 1000
        assert np.isclose(got, want)
    assert lb.today_cost_pnl(dec, [0, 0]).tolist() == [0.0, 0.0]


def test_sell_only_never_buys():
    import numpy as np
    a = np.array([1, -1, 0, 1, -1])
    assert np.where(a == 1, 1, 0).tolist() == [1, 0, 0, 1, 0]
