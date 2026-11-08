"""Tests for sizing, execution and risk limits in the money view."""
import numpy as np
import pandas as pd

from odte.eval import portfolio as pf


def _trades(n_days=60, per_day=3, ror=0.05, unit_risk=1500.0, cap=np.nan, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i, d in enumerate(pd.bdate_range("2026-01-05", periods=n_days)):
        for k in range(per_day):
            rows.append({"row": len(rows), "day": d, "t": d + pd.Timedelta(hours=10 + k),
                         "side": "short", "ror": ror + rng.normal(0, 0.3),
                         "unit_risk": unit_risk, "cap": cap, "strength": rng.random()})
    return pd.DataFrame(rows)


def test_daily_risk_cap_is_never_exceeded():
    tr = _trades(per_day=10)
    fr = np.full(len(tr), pf.MAX_RISK_PER_TRADE)          # every trade wants the max
    capital = 1_000_000
    res = pf.simulate(tr, fr, capital, sorted(tr["day"].unique()))
    # 10 trades x 2% would be 20% a day; opened risk must stay within 6%.
    assert res["max_risk_opened_in_a_day"] <= pf.MAX_RISK_PER_DAY + 1e-12
    assert res["max_risk_opened_in_a_day"] > 0.05


def test_whole_lots_and_liquidity_cap():
    tr = _trades(cap=2.0)
    fr = np.full(len(tr), pf.MAX_RISK_PER_TRADE)
    res = pf.simulate(tr, fr, 10_000_000, sorted(tr["day"].unique()))
    assert res["pct_trades_capped_by_liquidity"] == 100          # wanted ~133 lots, 2 shown
    small = pf.simulate(tr, np.full(len(tr), 0.001), 100_000, sorted(tr["day"].unique()))
    assert small["n_trades"] == 0                                # $100 risk < 1 lot of $1,500


def test_drawdown_stop_pauses_trading():
    tr = _trades(ror=-1.0)                                       # every trade loses its max
    tr["ror"] = -1.0
    fr = np.full(len(tr), pf.MAX_RISK_PER_TRADE)
    res = pf.simulate(tr, fr, 1_000_000, sorted(tr["day"].unique()))
    assert res["drawdown_pauses"] >= 1
    assert res["n_trades"] < len(tr)


def test_kelly_uses_only_earlier_days():
    tr = _trades(n_days=40, per_day=3)
    f1 = pf.kelly_fractions(tr)
    tr2 = tr.copy()
    last = tr2["day"] == tr2["day"].max()
    tr2.loc[last, "ror"] = 50.0                                  # rewrite the last day's outcomes
    f2 = pf.kelly_fractions(tr2)
    assert np.allclose(f1[~last.to_numpy()], f2[~last.to_numpy()])
    assert (f1 <= pf.MAX_RISK_PER_TRADE + 1e-12).all() and (f1 >= 0).all()


def test_mid_or_skip_drops_unfilled_trades():
    ds = pd.DataFrame({
        "day": pd.to_datetime(["2026-01-05"] * 2), "t": pd.to_datetime(["2026-01-05 10:00", "2026-01-05 11:00"]),
        "pnl_short_fly": [100.0, 100.0], "risk_short_fly": [1000.0, 1000.0],
        "pnl_short_fly_midx": [150.0, 150.0], "risk_short_fly_midx": [950.0, 950.0],
        "pnl_short_fly_t5": [90.0, 90.0], "risk_short_fly_t5": [1010.0, 1010.0],
        "fill_mid_short": [True, False], "cap_short": [5.0, 5.0]})
    act = np.array([1, 1])
    assert len(pf.trade_table(ds, act, None, "fly", "mid_or_skip")) == 1
    t = pf.trade_table(ds, act, None, "fly", "mid_or_cross")
    assert np.allclose(t["ror"], [150 / 950, 90 / 1010])        # mid if filled, else t+5 touch
