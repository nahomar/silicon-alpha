"""ES overnight features use only bars that finished before 09:30."""
import numpy as np
import pandas as pd

from odte.data import futures as fu


def _bars(day0, day1, jump_after_open=0.0):
    t = pd.date_range(f"{day0} 15:00", f"{day1} 12:00", freq="min", tz="America/New_York")
    px = 6000 + np.cumsum(np.full(len(t), 0.25))
    open_ = pd.Timestamp(f"{day1} 09:30", tz="America/New_York")
    px = np.where(t >= open_, px + jump_after_open, px)
    return pd.DataFrame({"ts": (t.tz_convert("UTC").astype("int64") // 10**9).to_numpy(),
                         "instrument_id": 1, "open": px, "high": px + 0.5, "low": px - 0.5,
                         "close": px, "volume": 10.0})


def test_overnight_features_ignore_bars_after_the_open(tmp_path, monkeypatch):
    monkeypatch.setattr(fu, "OUT", tmp_path)
    days = ["2026-03-17", "2026-03-18"]
    _bars(*days).to_parquet(tmp_path / "es_1m.parquet")
    a = fu.es_features(days)
    _bars(*days, jump_after_open=500.0).to_parquet(tmp_path / "es_1m.parquet")
    b = fu.es_features(days)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[pd.Timestamp("2026-03-18"), "es_on_ret"] > 0
