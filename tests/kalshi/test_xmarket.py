import numpy as np
import pandas as pd

from kalshi import xmarket as xm


def test_threshold_parsing():
    assert xm.thresholds("KXINX-26SEP10H1600-B7337", "between", "7,325 to 7,349.9999") == ("between", 7325.0, 7349.9999)
    k, a, _ = xm.thresholds("KXINXU-26SEP10H1600-T7504.9999", "greater_or_equal", "7,505 or above")
    assert k == "above" and a == 7504.9999
    k, a, _ = xm.thresholds("INX-22APR28-T4150", "less", "4149.99 or lower")
    assert k == "below" and a == 4150.0
    assert xm.thresholds("INXW-22APR29-T4,150", "less", "x") is None


def _grid(t, F, sigma=40.0):
    """Normal-ish smile-free chain around F for the digital test."""
    from math import erf, exp, pi, sqrt
    ks = np.arange(F - 200, F + 205, 5.0)
    def call(k):
        z = (F - k) / sigma
        n = exp(-z * z / 2) / sqrt(2 * pi)
        N = 0.5 * (1 + erf(z / sqrt(2)))
        return (F - k) * N + sigma * n
    q = {}
    for k in ks:
        c = call(k)
        q[("C", float(k))] = (c - 0.05, c + 0.05)
        q[("P", float(k))] = (c - F + k - 0.05, c - F + k + 0.05)
    cols = pd.MultiIndex.from_tuples(list(q), names=["right", "strike"])
    bid = pd.DataFrame([[max(v[0], 0.0) for v in q.values()]], index=[t], columns=cols)
    ask = pd.DataFrame([[v[1] for v in q.values()]], index=[t], columns=cols)
    return {"bid": bid, "ask": ask}


def test_digital_curve_recovers_probabilities():
    t = pd.Timestamp("2025-03-03 11:00")
    curve = xm.digital_curve(_grid(t, 5000.0), t, 5000.0)
    assert abs(xm.fair(curve, "above", 5000.0, np.nan) - 0.5) < 0.03
    assert abs(xm.fair(curve, "above", 5040.0, np.nan) - 0.159) < 0.03       # +1 sigma
    assert abs(xm.fair(curve, "between", 4960.0, 5040.0) - 0.683) < 0.04
    assert np.isnan(xm.fair(curve, "above", 6000.0, np.nan))                  # outside the chain


def test_trade_rule_needs_margin_after_fees():
    assert xm.trade(0.60, 0.50, 0.52) == ("yes", 0.52, 0.0175)               # 0.52 + 0.0175 <= 0.57
    assert xm.trade(0.56, 0.50, 0.52) is None                                # not cheap enough
    side, px, fee = xm.trade(0.30, 0.40, 0.42)                               # NO at 0.60 vs fair 0.70
    assert side == "no" and px == 0.6 and fee > 0
    assert xm.trade(0.6, 0.52, 0.50) is None                                 # crossed quote
