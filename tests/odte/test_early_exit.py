import numpy as np
import pandas as pd

from odte.research import early_exit as ee


def _grid(times, quotes):
    cols = pd.MultiIndex.from_tuples(list(quotes), names=["right", "strike"])
    bid = pd.DataFrame({c: [v[i][0] for i in range(len(times))] for c, v in quotes.items()}, index=times)[cols]
    ask = pd.DataFrame({c: [v[i][1] for i in range(len(times))] for c, v in quotes.items()}, index=times)[cols]
    return {"bid": bid, "ask": ask}


def test_first_hit():
    assert ee.first_hit([5, 4, 2, 1], 2.5, above=False) == 2
    assert ee.first_hit([5, np.nan, 9], 8, above=True) == 2
    assert ee.first_hit([5, 6], 1, above=False) == -1


def test_path_starts_after_the_decision_minute():
    times = pd.date_range("2024-01-02 10:00", periods=3, freq="min")
    q = {("C", 100.0): [(1.0, 1.2)] * 3, ("P", 100.0): [(1.0, 1.2)] * 3,
         ("C", 110.0): [(0.1, 0.2)] * 3, ("P", 90.0): [(0.1, 0.2)] * 3}
    win, vm, vt = ee.fly_paths(_grid(times, q), times[0], times[-1], 100.0, 10.0)
    assert win[0] == times[1] and len(win) == 2            # never the decision minute itself
    assert np.allclose(vm, 2.2 - 0.3) and np.allclose(vt, 2.4 - 0.2)


def test_take_profit_pays_exit_costs_and_holds_otherwise():
    path = (None, np.array([3.0, 1.9, 1.0]), np.array([3.2, 2.1, 1.2]))
    pnl_hold, risk, ex = ee.exit_pnl(4.0, 3.9, 10.0, 0.0, path, "hold")
    pnl_e1, _, ex1 = ee.exit_pnl(4.0, 3.9, 10.0, 0.0, path, "E1")
    assert not ex and ex1
    assert pnl_e1 < pnl_hold                               # pinned at K: holding keeps the full credit
    never = (None, np.array([3.0, 3.5]), np.array([3.2, 3.7]))
    assert ee.exit_pnl(4.0, 3.9, 10.0, 0.0, never, "E1")[0] == pnl_hold
