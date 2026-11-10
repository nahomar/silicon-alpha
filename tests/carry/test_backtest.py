import numpy as np
import pandas as pd

from carry import backtest as cb


def _funding(start="2021-01-01", n_days=40, rate=0.0001, hours=8):
    t = pd.date_range(start, periods=n_days * 24 // hours, freq=f"{hours}h")
    return pd.Series(rate, index=t)


# --- no look-ahead -------------------------------------------------------------------------------

def test_switch_decision_uses_only_funding_settled_before_the_day_starts():
    f = _funding()
    idx = cb.days("2021-01-10", "2021-02-05")
    base = cb.signal(f, idx)
    d = pd.Timestamp("2021-01-20")
    g = f.copy()
    g[g.index >= d] = 0.05                       # huge funding from d 00:00 onwards, incl. the 00:00 print
    s = cb.signal(g, idx)
    assert np.allclose(s[s.index <= d], base[base.index <= d])          # nothing at or after d leaks in
    assert s[d + pd.Timedelta(days=1)] > base[d + pd.Timedelta(days=1)]  # it shows up the next day
    st = cb.c2_state(s)
    assert st[d] == cb.c2_state(base)[d]


def test_signal_annualizes_a_constant_rate_and_needs_a_full_week():
    f = _funding(rate=0.0001)                    # 0.01% per 8h
    idx = cb.days("2021-01-01", "2021-01-20")
    s = cb.signal(f, idx)
    assert s[:"2021-01-07"].isna().all()         # window not yet covered
    assert np.isclose(s["2021-01-08"], 0.0001 * 3 * 365)
    assert np.isclose(s["2021-01-15"], 0.0001 * 3 * 365)


def test_hold_only_when_signal_is_strictly_above_ten_percent():
    sig = pd.Series([0.05, 0.10, 0.1000001, np.nan], index=cb.days("2021-01-01", "2021-01-04"))
    assert cb.c2_state(sig).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_a_day_earns_the_next_midnight_print_but_not_its_own():
    t = pd.to_datetime(["2021-01-02 00:00", "2021-01-02 08:00", "2021-01-02 16:00", "2021-01-03 00:00"])
    f = pd.Series([1.0, 2.0, 4.0, 8.0], index=t)
    e = cb.earned_by_day(f, cb.days("2021-01-01", "2021-01-02"))
    assert e["2021-01-01"] == 1.0                # the 01-02 00:00 print belongs to day 01-01
    assert e["2021-01-02"] == 2.0 + 4.0 + 8.0


# --- funding sign --------------------------------------------------------------------------------

def test_positive_funding_pays_the_short_and_negative_funding_costs_it():
    idx = cb.days("2021-01-01", "2021-01-10")
    zero = pd.Series(0.0, index=idx)
    on = pd.Series(1.0, index=idx)
    pos = cb.simulate(pd.Series(0.0003, index=idx), zero, on, zero)
    neg = cb.simulate(pd.Series(-0.0003, index=idx), zero, on, zero)
    assert (pos["total"] > 0).all() and (neg["total"] < 0).all()
    assert np.isclose(pos["total"].iloc[0], 0.0003 / cb.CAP)       # on capital = 1.333 x notional


def test_flat_days_earn_t_bills_and_zero_excess():
    idx = cb.days("2021-01-01", "2021-01-05")
    rf = pd.Series(0.0001, index=idx)
    off = pd.Series(0.0, index=idx)
    s = cb.simulate(pd.Series(0.01, index=idx), rf, off, pd.Series(0.002, index=idx))
    assert np.allclose(s["total"], 0.0001) and np.allclose(s["excess"], 0.0) and s["cost"].sum() == 0


# --- costs ---------------------------------------------------------------------------------------

def test_always_on_pays_four_trades_once():
    idx = cb.days("2021-01-01", "2021-03-31")
    zero = pd.Series(0.0, index=idx)
    cost = cb.switch_cost("binance", "BTC", pd.Series(30000.0, index=idx))
    s = cb.simulate(zero, zero, pd.Series(1.0, index=idx), cost)
    four_trades = 2 * (cb.BINANCE_SPOT_TAKER + cb.SLIP) + 2 * (cb.BINANCE_PERP_TAKER + cb.SLIP)
    assert np.isclose(s["cost"].sum(), four_trades / cb.CAP)
    assert s["cost"].iloc[0] > 0 and s["cost"].iloc[-1] > 0 and s["cost"].iloc[1:-1].sum() == 0
    assert s["changes"].sum() == 2


def test_each_switch_costs_two_legs_including_the_final_exit():
    idx = cb.days("2021-01-01", "2021-01-08")
    zero = pd.Series(0.0, index=idx)
    st = pd.Series([0, 1, 1, 0, 0, 1, 0, 1], index=idx, dtype=float)    # 5 changes + exit at the end
    c = 0.0019
    s = cb.simulate(zero, zero, st, pd.Series(c, index=idx))
    assert s["changes"].sum() == 6
    assert np.isclose(s["cost"].sum(), 6 * c / cb.CAP)
    assert np.isclose(cb.switch_cost("binance", "ETH", pd.Series([1.0])).iloc[0], 0.0019)


def test_coinbase_per_contract_fee_scales_with_price():
    p = pd.Series([100_000.0, 50_000.0])
    c = cb.switch_cost("coinbase", "BTC", p)
    base = cb.CB_SPOT_TAKER + cb.SLIP + cb.CDE_PERP_PCT + cb.SLIP
    assert np.isclose(c.iloc[0], base + 0.15 / 1000.0)               # 0.01 BTC x $100k = $1,000
    assert np.isclose(c.iloc[1], base + 0.15 / 500.0)


# --- cash rate, statistics, margin risk --------------------------------------------------------------

def test_t_bill_uses_the_previous_trading_day():
    irx = pd.Series([5.0, 4.0], index=pd.to_datetime(["2021-01-04", "2021-01-08"]))   # Mon, Fri
    rf = cb.tbill_daily(irx, cb.days("2021-01-04", "2021-01-10"))
    assert np.isnan(rf["2021-01-04"])
    assert np.isclose(rf["2021-01-05"], 0.05 / 365) and np.isclose(rf["2021-01-08"], 0.05 / 365)
    assert np.isclose(rf["2021-01-09"], 0.04 / 365) and np.isclose(rf["2021-01-10"], 0.04 / 365)


def test_bootstrap_is_reproducible_and_uses_the_bonferroni_level():
    x = np.random.default_rng(1).normal(0.0002, 0.001, 900)
    a, b = cb.block_bootstrap_mean(x), cb.block_bootstrap_mean(x)
    assert a == b and a["ci"][0] < a["mean"] < a["ci"][1]
    wide = cb.block_bootstrap_mean(x, level=0.975)
    narrow = cb.block_bootstrap_mean(x, level=0.95)
    assert wide["ci"][0] < narrow["ci"][0] and a["level"] == 0.975


def test_margin_risk_counts_one_episode_for_a_single_spike():
    idx = cb.days("2021-01-01", "2021-03-01")
    close = pd.Series(100.0, index=idx)
    close["2021-02-01":] = 130.0
    spot = pd.DataFrame({"high": close, "close": close})
    m = cb.margin_risk(spot, cb.days("2021-01-01", "2021-02-20"), pd.Series(1.0, index=idx))
    assert m["episodes"] == 1 and m["flagged_days"] == 7               # 01-25 .. 01-31 see the jump
    assert np.isclose(m["largest_7d_rise"], 0.30)
    assert m["episodes_started_while_c2_invested"] == 1


def test_describe_on_a_known_path():
    idx = cb.days("2021-01-01", "2021-12-31")
    sim = pd.DataFrame({"total": 0.0001, "rf": 0.0001, "excess": 0.0, "gross": 0.0001,
                        "gross_excess": 0.0, "state": 0.0, "changes": 0.0, "cost": 0.0}, index=idx)
    d = cb.describe(sim)
    assert np.isclose(d["ann_return"], 1.0001 ** 365 - 1) and d["max_drawdown"] == 0.0
    assert d["pct_invested"] == 0.0 and np.isclose(d["ann_return_minus_tbill"], 0.0)
