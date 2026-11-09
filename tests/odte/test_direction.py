"""Direction step 1: targets look forward only, lags look back only, sealed days refused."""
import numpy as np
import pandas as pd
import pytest

from odte.eval import direction as dr


def _day(day, prices, settle):
    t = pd.date_range(f"{day} 10:00", periods=len(prices), freq="15min")
    return pd.DataFrame({"day": pd.Timestamp(day), "t": t, "trigger": "grid", "F": prices,
                         "settle": settle, "straddle_mid": 20.0,
                         "minutes_to_close": [(pd.Timestamp(f"{day} 16:00") - x).seconds / 60 for x in t]})


def test_forward_and_lagged_moves():
    prices = np.linspace(5000, 5021, 22)[:21]                # 10:00 .. 15:00
    d = dr.add_targets(_day("2023-03-15", prices, 5050.0))
    i = d.index[d["t"] == pd.Timestamp("2023-03-15 11:00")][0]
    assert np.isclose(d.loc[i, "fwd30"], np.log(prices[6] / prices[4]) * 1e4)
    assert np.isclose(d.loc[i, "ret60"], np.log(prices[4] / prices[0]) * 1e4)
    last = d.index[d["t"] == pd.Timestamp("2023-03-15 15:00")][0]
    assert np.isclose(d.loc[last, "fwd60"], np.log(5050.0 / prices[20]) * 1e4)   # settlement
    assert np.isnan(d.loc[0, "ret15"])                                        # nothing before 10:00


def test_sealed_days_are_refused(tmp_path):
    p = tmp_path / "x.parquet"
    _day("2021-09-15", np.full(21, 4400.0), 4400.0).to_parquet(p)
    with pytest.raises(SystemExit, match="REFUSED"):
        dr.load([p])


def test_sequence_tokens_never_see_after_t():
    from odte.eval import direction_seq as ds
    idx = pd.date_range("2023-03-15 09:30", "2023-03-15 16:00", freq="min")
    path = pd.DataFrame({"F": np.linspace(4000, 4010, len(idx)), "straddle": 20.0, "spread": 0.4}, index=idx)
    t = pd.Timestamp("2023-03-15 11:00")
    a, ma = ds.tokens(path, t)
    later = path.copy()
    later.loc[later.index > t, "F"] = 9999.0
    b, mb = ds.tokens(later, t)
    assert np.array_equal(a, b) and np.array_equal(ma, mb) and ma.sum() == ds.L
