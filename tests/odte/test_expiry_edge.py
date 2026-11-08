"""Tests for the hold-to-expiry 0DTE study and the Databento 0DTE reducer."""
import numpy as np
import pandas as pd
import pytest

from odte.data.fetch_0dte import reduce_frame, session_window
from odte.eval import expiry_edge as ee


def test_selftest_passes():
    assert ee.selftest(verbose=False)


def test_quote_visible_only_from_next_whole_minute():
    day = pd.Timestamp("2026-11-06")
    chain = pd.DataFrame({
        "quote_datetime": [day + pd.Timedelta(hours=10, seconds=30)],
        "strike": [6000.0], "option_type": ["C"], "bid": [10.0], "ask": [10.5],
    })
    g = ee.minute_grid(chain)
    assert np.isnan(g["bid"].loc[day + pd.Timedelta(hours=10)].iloc[0])
    assert g["bid"].loc[day + pd.Timedelta(hours=10, minutes=1)].iloc[0] == 10.0
    # carried at most 5 minutes
    assert np.isnan(g["bid"].loc[day + pd.Timedelta(hours=10, minutes=7)].iloc[0])


def test_reduce_frame_parses_osi_and_keeps_only_same_day_expiry():
    ts = pd.Timestamp("2026-11-06 15:00:00", tz="UTC")   # 10:00 New York
    df = pd.DataFrame({
        "ts_recv": [ts, ts, ts],
        "symbol": ["SPXW  261106C05800000", "SPXW  261106P05795000",
                   "SPXW  261109C05800000"],
        "bid_px_00": [12.3, 9.8, 30.0], "ask_px_00": [12.5, 10.0, 30.5],
        "bid_sz_00": [10, 20, 5], "ask_sz_00": [11, 21, 6],
    })
    out = reduce_frame(df, pd.Timestamp("2026-11-06"))
    assert len(out) == 2
    assert out["strike"].tolist() == [5800.0, 5795.0]
    assert out["option_type"].tolist() == ["C", "P"]
    assert (out["quote_datetime"] == pd.Timestamp("2026-11-06 10:00")).all()
    assert (out["root"] == "SPXW").all()


def test_session_window_handles_dst():
    # 2026-10-30 is EDT (UTC-4); 2026-11-02 is EST (UTC-5).
    assert session_window(pd.Timestamp("2026-10-30"))[0].startswith("2026-10-30T13:25")
    assert session_window(pd.Timestamp("2026-11-02"))[0].startswith("2026-11-02T14:25")
