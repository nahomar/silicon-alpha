"""Pair sizing: equal premium, worst case = budget; crash moves offset."""
import numpy as np
import pandas as pd

from odte.eval import pairs as pr


def _row(pred, cm, pnl_s, pnl_l, rs, rl):
    return pd.Series({"pred": pred, "fly_credit_mid": cm, "ratio": 1.0, "pnl_short_fly": pnl_s,
                      "pnl_long_fly": pnl_l, "risk_short_fly": rs, "risk_long_fly": rl,
                      "pnl_short_fly_midx": pnl_s, "pnl_long_fly_midx": pnl_l, "pnl_short_fly_t5": pnl_s, "pnl_long_fly_t5": pnl_l,
                      "risk_short_fly_t5": rs, "risk_long_fly_t5": rl, "fill_mid_short": False, "fill_mid_long": False,
                      "risk_short_fly_midx": rs, "risk_long_fly_midx": rl})


def _frames(a, b):
    key = pd.MultiIndex.from_tuples([(pd.Timestamp("2023-03-15"), "10:00")], names=["day", "hhmm"])
    return {"spx": pd.DataFrame([a], index=key), "ndx": pd.DataFrame([b], index=key)}


def test_worst_case_equals_budget_and_crash_partly_offsets():
    # SPX predicted cheap -> long; NDX predicted rich -> short. Both lose / gain their max.
    fr = _frames(_row(1.3, 10.0, -500.0, 1500.0, 500.0, 1000.0),
                 _row(0.7, 40.0, -2000.0, 6000.0, 2000.0, 4000.0))
    pick = pr.model_choice(0.15)(None, {k: f.iloc[0] for k, f in fr.items()})
    assert pick == ("ndx", "spx")
    K = pr.BUDGET / (2000.0 / 40.0 + 1000.0 / 10.0)
    worst = K / 40.0 * 2000.0 + K / 10.0 * 1000.0
    assert np.isclose(worst, pr.BUDGET)
    v = pr.pair_pnl(fr, pr.model_choice(0.15)).iloc[0]
    assert np.isclose(v, K / 40.0 * -2000.0 + K / 10.0 * 1500.0) and v > -pr.BUDGET


def test_no_trade_below_margin():
    fr = _frames(_row(1.0, 10.0, 1, 1, 5, 5), _row(0.95, 40.0, 1, 1, 5, 5))
    assert pr.pair_pnl(fr, pr.model_choice(0.15)).empty
