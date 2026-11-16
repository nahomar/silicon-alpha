"""Epoch conversion is right whatever unit pandas stores times in."""
import pandas as pd

from odte.timeutil import epoch_s


def test_epoch_seconds_ignore_the_storage_unit():
    t = pd.Timestamp("2026-03-17 15:00", tz="UTC")
    for unit in ("s", "ms", "us", "ns"):
        idx = pd.DatetimeIndex([t]).as_unit(unit)
        assert epoch_s(idx)[0] == int(t.timestamp())
        assert epoch_s(pd.Series(idx))[0] == int(t.timestamp())
