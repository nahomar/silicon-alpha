"""Datetime -> integer epoch conversions that do not depend on pandas' time unit.

pandas 3 stores datetimes parsed from text in microseconds, not nanoseconds,
so `x.astype("int64") // 10**9` silently returns the wrong number of seconds.
Convert through an explicit nanosecond unit instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def epoch_ns(x) -> np.ndarray:
    """Nanoseconds since 1970-01-01 UTC (naive values are taken as UTC)."""
    if isinstance(x, pd.Series):
        return pd.to_datetime(x).dt.as_unit("ns").astype("int64").to_numpy()
    return pd.DatetimeIndex(pd.to_datetime(x)).as_unit("ns").asi8


def epoch_s(x) -> np.ndarray:
    return epoch_ns(x) // 10**9
