import numpy as np
import pandas as pd

from odte.research import calendar as cal

F0 = {"s0_sell": 0.0, "s1_buy": 0.0, "sx_sell": 0.0}


def test_calendar_profits_when_nothing_moves_and_loss_is_capped_by_the_debit():
    # sell 0DTE straddle 10, buy next 14; nothing moves: 0DTE worthless, next worth 13 at the close
    pnl, risk = cal.calendar_pnl(10.0, 14.0, 13.0, 5000.0, 5000.0, F0)
    assert np.isclose(pnl, (10 - 0 + 13 - 14) * 100 - cal.FEES6)
    assert np.isclose(risk, (14 - 10) * 100 + cal.FEES6)
    # huge move: next straddle ~ intrinsic (300 + tiny), 0DTE pays 300 -> lose about the debit
    pnl, risk = cal.calendar_pnl(10.0, 14.0, 300.2, 5300.0, 5000.0, F0)
    assert pnl < 0 and -pnl <= risk


def test_costs_lower_the_pnl():
    f = {"s0_sell": 0.02, "s1_buy": 0.02, "sx_sell": 0.02}
    a, _ = cal.calendar_pnl(10.0, 14.0, 13.0, 5000.0, 5000.0, F0)
    b, _ = cal.calendar_pnl(10.0, 14.0, 13.0, 5000.0, 5000.0, f, share=0.5)
    c, _ = cal.calendar_pnl(10.0, 14.0, 13.0, 5000.0, 5000.0, f, share=1.0)
    assert a > b > c


def test_signal_threshold_never_uses_the_same_day():
    days = pd.Series(pd.bdate_range("2020-01-01", periods=300))
    s = pd.Series(np.random.default_rng(0).normal(size=300))
    hh = pd.Series(["10:00"] * 300)
    t1 = cal.signal_threshold(s, days, hh)
    s2 = s.copy()
    s2.iloc[200] = 99.0                      # change today's value only
    t2 = cal.signal_threshold(s2, days, hh)
    assert t1.iloc[200] == t2.iloc[200]      # today's threshold unchanged
    assert t1.iloc[:120].isna().all() and np.isfinite(t1.iloc[121])
