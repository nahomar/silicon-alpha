import numpy as np
import pandas as pd

from odte.research import stress as st


def test_signals_use_only_earlier_closes():
    idx = pd.bdate_range("2020-01-01", periods=30)
    sig = pd.DataFrame({"x": np.arange(30, dtype=float)}, index=idx)
    days = pd.Series([idx[0], idx[10], idx[10] + pd.Timedelta(hours=0)], index=[5, 6, 7])
    got = st.as_of_previous_close(sig, days)
    assert np.isnan(got.loc[5, "x"])                    # nothing before the first day
    assert got.loc[6, "x"] == 9.0                        # day 10 sees day 9's close, never its own


def test_daily_signal_math():
    idx = pd.bdate_range("2020-01-01", periods=25)
    c = pd.DataFrame({"vix": 20.0, "vvix": 100.0, "skew": np.linspace(120, 144, 25), "move": 100.0,
                      "ovx": 40.0, "gvz": 15.0, "vxn": 25.0}, index=idx)
    s = st.daily_signals(c)
    assert s["g_vvix_over_vix"].iloc[-1] == 5.0 and s["g_vxn_over_vix"].iloc[-1] == 1.25
    assert s["g_move_chg5"].iloc[-1] == 0.0 and s["g_skew_z20"].iloc[-1] > 1
