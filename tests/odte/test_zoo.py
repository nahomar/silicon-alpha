import numpy as np
import pandas as pd

from odte.research import zoo


def _chain(day="2024-03-05", F=5000.0, vol_pts=40.0, drift=0.0):
    """Synthetic 0DTE chain: Bachelier-style prices around a forward, 5-point strikes, every minute."""
    from math import erf, exp, pi, sqrt
    day = pd.Timestamp(day)
    rows = []
    ts = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(hours=16), freq="min")
    for i, t in enumerate(ts):
        left = max((len(ts) - 1 - i) / (len(ts) - 1), 1e-3)
        s = vol_pts * sqrt(left)
        f = F + drift * i / len(ts)
        for k in np.arange(F - 150, F + 155, 5.0):
            z = (f - k) / s
            N, n = 0.5 * (1 + erf(z / sqrt(2))), exp(-z * z / 2) / sqrt(2 * pi)
            c = (f - k) * N + s * n
            p = c - (f - k)
            for r, v in (("C", c), ("P", p)):
                v = max(v, 0.05)
                rows.append({"quote_datetime": t, "option_type": r, "strike": float(k), "bid": v - 0.05, "ask": v + 0.05,
                             "root": "SPXW", "expiration": day, "bid_size": 10, "ask_size": 10})
    return pd.DataFrame(rows)


def test_iron_condor_held_to_expiry_keeps_credit_when_pinned():
    d = zoo.Day(pd.Timestamp("2024-03-05"), _chain(), settle=5000.0, prev_settle=5000.0, info={"vix_prev": 15})
    rule = {"structure": "iron_condor", "short_k": 1.0, "wing": 1.0}
    tr = zoo.trade(rule, d, "10:00", k_day=1.0, share=0.5)
    assert tr is not None and tr["pnl"] > 0 and tr["risk"] > 0       # expires between the shorts
    big = zoo.Day(pd.Timestamp("2024-03-05"), _chain(), settle=5150.0, prev_settle=5000.0, info={})
    loss = zoo.trade(rule, big, "10:00", k_day=1.0, share=0.5)
    assert loss["pnl"] < 0 and abs(loss["pnl"] + loss["risk"]) < 1e-6  # max loss = the risk used for sizing


def test_take_profit_exits_early_and_pays_exit_costs():
    d = zoo.Day(pd.Timestamp("2024-03-05"), _chain(), settle=5000.0, prev_settle=5000.0, info={})
    hold = zoo.trade({"structure": "iron_fly", "wing": 1.5}, d, "10:00", 1.0, 0.5)
    tp = zoo.trade({"structure": "iron_fly", "wing": 1.5, "exit": {"take_profit": 0.25}}, d, "10:00", 1.0, 0.5)
    assert tp["exited"] == "tp/sl" and hold["exited"] is None
    assert tp["pnl"] < hold["pnl"]                                    # pinned at K: holding keeps more


def test_direction_and_filters():
    d = zoo.Day(pd.Timestamp("2024-03-05"), _chain(drift=30.0), settle=5030.0, prev_settle=5000.0,
                info={"vix_prev": 30, "event_day": True})
    up_put = zoo.trade({"structure": "put_credit_spread", "short_k": 1.0, "wing": 0.5,
                        "direction": "with_morning_move"}, d, "12:00", 1.0, 0.5)
    assert up_put is not None and up_put["pnl"] > 0                   # market rose: sold puts, they expire
    assert not zoo.passes_filters({"filters": {"vix_max": 20}}, d, "12:00")
    assert not zoo.passes_filters({"filters": {"skip_event_days": True}}, d, "12:00")
    assert zoo.passes_filters({"filters": {"days_of_week": [1]}}, d, "12:00")   # 2024-03-05 is a Tuesday


def test_unbounded_structures_are_rejected():
    legs = [("C", 100.0, -1)]
    assert zoo.worst_payoff(legs, 100.0)[1] is False
