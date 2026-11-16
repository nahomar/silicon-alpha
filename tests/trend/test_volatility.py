import numpy as np
import pandas as pd

from trend import volatility as tv


def test_signal_is_lagged_two_closes_and_costs_charged():
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.DataFrame({"SVXY": [100, 110, 121, 133.1, 146.41, 161.051], "^VIX": [15, 30, 15, 15, 15, 15],
                       "^VIX3M": [20, 20, 20, 20, 20, 20], "^IRX": 0.0, "SPY": 1.0}, index=idx)
    d = tv.strategy(px, 1.0, cost_bp=0.0)
    # contango on day 0 -> position from day 2; backwardation on day 1 -> flat on day 3
    assert d["pos"].tolist() == [0, 0, 1, 0, 1, 1]
    assert np.isclose(d["total"].iloc[2], 0.10) and np.isclose(d["total"].iloc[3], 0.0)
    c = tv.strategy(px, 1.0, cost_bp=10.0)
    assert np.isclose((d["total"] - c["total"]).sum(), 3 * 10 / 1e4)     # three switches
