import numpy as np
import pandas as pd

from trend import backtest as tb


def _frame(n=600, seed=0):
    idx = pd.bdate_range("2010-01-01", periods=n)
    rng = np.random.default_rng(seed)
    r = pd.DataFrame(rng.normal(0, 0.01, (n, 2)), index=idx, columns=["X", "Y"])
    rf = pd.Series(0.0001, index=idx)
    return r, rf, r.sub(rf, axis=0)


def test_signal_is_plus_one_in_a_steady_rise_and_minus_one_in_a_fall():
    idx = pd.bdate_range("2010-01-01", periods=400)
    ex = pd.DataFrame({"UP": 0.001, "DOWN": -0.001}, index=idx)
    s = tb.signals(ex, idx[[100, 399]])
    assert s.iloc[0].isna().all()                       # not enough history yet
    assert s.iloc[1]["UP"] == 1.0 and s.iloc[1]["DOWN"] == -1.0


def test_new_weights_earn_only_from_two_days_after_the_signal():
    r, rf, ex = _frame()
    me = tb.month_ends(r.index)
    w = pd.DataFrame(0.0, index=me, columns=r.columns)
    d = me[10]
    w.loc[d:, "X"] = 1.0
    W, trades = tb.daily_weights(w, r.index)
    p = r.index.get_loc(d)
    assert W.iloc[p + 1]["X"] == 0.0                   # the trade happens at this close
    assert W.iloc[p + 2]["X"] == 1.0                   # first return earned
    assert trades[r.index[p + 1]] == d


def test_a_jump_right_after_month_end_is_not_captured():
    r, rf, ex = _frame()
    me = tb.month_ends(r.index)
    d = me[10]
    p = r.index.get_loc(d)
    r.iloc[p + 1, 0] = 0.50                            # big move before we can trade
    ex = r.sub(rf, axis=0)
    w = pd.DataFrame(0.0, index=me, columns=r.columns)
    w.loc[d:, "X"] = 1.0
    out = tb.run(r, rf, ex, w, "A", cost_bp=0.0)
    assert abs(out["excess"].iloc[p + 1]) < 1e-12


def test_small_account_version_never_shorts_or_levers():
    idx = pd.bdate_range("2010-01-01", periods=5)
    s = pd.DataFrame({"X": [1.0, -1.0, 1 / 3, np.nan, -1 / 3], "Y": [1.0, 1.0, -1.0, 1.0, 1 / 3]}, index=idx)
    vol = pd.DataFrame(0.2, index=idx, columns=["X", "Y"])
    w = tb.target_weights(s, vol, "B")
    assert (w >= 0).all().all() and (w.sum(axis=1) <= 1 + 1e-12).all()
    wa = tb.target_weights(s, vol, "A")
    assert wa.iloc[0]["X"] == 1.0 * 0.40 / 0.2 / 2


def test_costs_are_charged_on_turnover():
    r, rf, ex = _frame()
    me = tb.month_ends(r.index)
    w = pd.DataFrame(0.0, index=me, columns=r.columns)
    w.loc[me[5]:, "X"] = 1.0
    a = tb.run(r, rf, ex, w, "A", cost_bp=0.0)
    b = tb.run(r, rf, ex, w, "A", cost_bp=10.0)
    assert np.isclose((a["excess"] - b["excess"]).sum(), 1.0 * 10 / 1e4)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    m = pd.Series(np.random.default_rng(1).normal(0.01, 0.03, 160))
    a, b = tb.block_bootstrap_sharpe(m), tb.block_bootstrap_sharpe(m)
    assert a == b and a["ci"][0] < a["sharpe"] < a["ci"][1]
