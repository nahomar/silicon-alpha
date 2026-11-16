"""Dealer-gamma sign convention and the magnet strike."""
import numpy as np
import pandas as pd

from odte.data.gamma import features_for


def _oi(rows):
    return pd.DataFrame(rows, columns=["expiration", "strike", "option_type", "oi"]).assign(
        expiration=lambda d: pd.to_datetime(d["expiration"]))


def test_calls_give_positive_puts_negative_dealer_gamma():
    day = "2026-03-18"
    calls = _oi([(day, k, "C", 1000) for k in range(5950, 6051, 5)])
    puts = _oi([(day, k, "P", 1000) for k in range(5950, 6051, 5)])
    assert features_for(calls, day, 6000.0, 0.008, 240)["gex_net"] > 0
    assert features_for(puts, day, 6000.0, 0.008, 240)["gex_net"] < 0


def test_magnet_points_at_the_biggest_0dte_strike():
    day = "2026-03-18"
    oi = _oi([(day, 6000, "C", 10), (day, 6050, "C", 9999), (day, 5950, "P", 10)])
    f = features_for(oi, day, 6000.0, 0.008, 240)
    assert np.isclose(f["maxoi_dist"], 50 / (6000 * 0.008))
