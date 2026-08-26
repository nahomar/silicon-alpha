"""Tests for the 0DTE option edge-existence harness.

The harness exists to give honest "no" answers, so these tests are mostly
about the ways it could give a dishonest "yes":

  leakage        using the exit bar's information to make the entry decision
  mid pricing    reporting a return nobody could have realized
  circularity    ranking on the spread, then scoring returns net of spread
  dependence     treating every contract in a snapshot as an independent bet

Each has a test below that fails if the guard is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from odte.eval.option_edge import (backtest_signal, baseline_signals,
                                   breakeven_spread_capture, evaluate_signal,
                                   inject_oracle_signal, selftest,
                                   synthetic_chain)
from odte.eval.option_panel import (PanelFilters, build_option_panel,
                                    spread_cost_summary)
from odte.eval.panel_stats import (group_indices, permutation_ic, spearman_ic)


@pytest.fixture(scope="module")
def chain():
    return synthetic_chain(n_bars=24, n_strikes=11, seed=1)


@pytest.fixture(scope="module")
def panel(chain):
    return build_option_panel(chain, interval="5min")


def _hand_chain(rows):
    """Minimal chain frame from (bar_index, strike, bid, ask) tuples."""
    start = pd.Timestamp("2024-06-03 09:30")
    recs = []
    for i, k, bid, ask in rows:
        recs.append({
            "quote_datetime": start + pd.Timedelta(minutes=5 * i),
            "root": "SPXW", "expiration": pd.Timestamp("2024-06-03 16:00"),
            "strike": float(k), "option_type": "C",
            "bid": bid, "ask": ask, "bid_size": 10.0, "ask_size": 10.0,
            "trade_volume": 1.0, "underlying_price": 5500.0, "delta": 0.5,
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Executable pricing
# ---------------------------------------------------------------------------

def test_long_return_buys_at_ask_and_sells_at_bid():
    ch = _hand_chain([(0, 5500, 1.00, 1.20), (1, 5500, 2.00, 2.20)]
                     + [(i, 5500 + 10 * j, 1.0, 1.2)
                        for i in (0, 1) for j in (1, 2, 3)])
    p = build_option_panel(ch, interval="5min")
    row = p[(p["strike"] == 5500) & (p["bar"] == p["bar"].min())].iloc[0]
    # entry ask 1.20, exit bid 2.00
    assert row["ret_long_net"] == pytest.approx((2.00 - 1.20) / 1.20)


def test_short_return_sells_at_bid_and_buys_at_ask():
    ch = _hand_chain([(0, 5500, 1.00, 1.20), (1, 5500, 2.00, 2.20)]
                     + [(i, 5500 + 10 * j, 1.0, 1.2)
                        for i in (0, 1) for j in (1, 2, 3)])
    p = build_option_panel(ch, interval="5min")
    row = p[(p["strike"] == 5500) & (p["bar"] == p["bar"].min())].iloc[0]
    # entry bid 1.00, exit ask 2.20
    assert row["ret_short_net"] == pytest.approx((1.00 - 2.20) / 1.00)


def test_net_return_is_always_worse_than_mid(panel):
    """Crossing the spread both ways can never beat the mid-to-mid number.

    If this ever fails, the harness is quoting a return nobody could realize.
    """
    assert (panel["ret_long_net"] < panel["ret_mid"] + 1e-12).all()


def test_spread_summary_reports_the_hurdle(panel):
    s = spread_cost_summary(panel)
    assert s["median_spread_pct"] > 0
    assert s["ratio_spread_to_move"] > 0
    assert s["n"] == len(panel)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_exit_uses_the_immediately_next_bar_only():
    ch = _hand_chain([(0, 5500, 1.00, 1.20), (1, 5500, 2.00, 2.20),
                      (2, 5500, 9.00, 9.20)]
                     + [(i, 5500 + 10 * j, 1.0, 1.2)
                        for i in (0, 1, 2) for j in (1, 2, 3)])
    p = build_option_panel(ch, interval="5min")
    first = p[(p["strike"] == 5500)].sort_values("bar").iloc[0]
    # Must use bar 1 (bid 2.00), never bar 2's 9.00.
    assert first["ret_long_net"] == pytest.approx((2.00 - 1.20) / 1.20)


def test_non_contiguous_bars_are_not_used_as_exits():
    """A contract that stops quoting must not 'exit' at a much later bar."""
    ch = _hand_chain([(0, 5500, 1.00, 1.20), (5, 5500, 9.00, 9.20)]
                     + [(i, 5500 + 10 * j, 1.0, 1.2)
                        for i in (0, 5) for j in (1, 2, 3)])
    p = build_option_panel(ch, interval="5min")
    assert p[p["strike"] == 5500].empty, "a 25-minute gap was treated as one bar"
    assert p.attrs["drop_reasons"].get("no_contiguous_next_bar", 0) > 0


def test_features_do_not_change_when_the_future_changes(chain):
    """Perturbing only bar t+1 must leave every feature at bar t untouched.

    This is the direct test for look-ahead: if any feature reads forward, the
    two frames will differ.
    """
    base = build_option_panel(chain, interval="5min")
    bars = np.sort(chain["quote_datetime"].unique())
    last_two = set(bars[-2:])

    tampered = chain.copy()
    hit = tampered["quote_datetime"].isin(last_two)
    tampered.loc[hit, "bid"] *= 3.0
    tampered.loc[hit, "ask"] *= 3.0
    tamp = build_option_panel(tampered, interval="5min")

    keep_bars = sorted(set(base["bar"]) & set(tamp["bar"]))[:-2]
    cols = ["moneyness", "spread_pct", "size_imbalance", "opt_ret_1",
            "opt_ret_3", "und_ret_1", "log_mid"]
    a = base[base["bar"].isin(keep_bars)].sort_values(["bar", "contract_id"])
    b = tamp[tamp["bar"].isin(keep_bars)].sort_values(["bar", "contract_id"])
    assert len(a) == len(b) and len(a) > 0
    for c in cols:
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(),
                                   rtol=1e-9, atol=1e-9,
                                   err_msg=f"feature {c} reads the future")


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_crossed_and_cheap_quotes_are_dropped_with_reasons():
    ch = _hand_chain([(i, 5500 + 10 * j, 1.0, 1.2)
                      for i in (0, 1) for j in range(6)])
    ch.loc[(ch["strike"] == 5500), "bid"] = 0.01          # below min_bid
    ch.loc[(ch["strike"] == 5510), ["bid", "ask"]] = [2.0, 1.0]   # crossed
    p = build_option_panel(ch, interval="5min")
    assert 5500.0 not in set(p["strike"])
    assert 5510.0 not in set(p["strike"])
    reasons = p.attrs["drop_reasons"]
    assert reasons.get("bid_below_min", 0) > 0
    assert p.attrs["n_dropped"] > 0


def test_thin_bars_are_dropped():
    ch = _hand_chain([(i, 5500 + 10 * j, 1.0, 1.2)
                      for i in (0, 1) for j in range(2)])
    p = build_option_panel(ch, interval="5min",
                           filters=PanelFilters(min_contracts_per_bar=4))
    assert p.empty


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_spearman_ic_known_cases():
    assert spearman_ic([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman_ic([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Monotone but non-linear: rank correlation still 1, Pearson would not be.
    assert spearman_ic([1, 2, 3, 4], [1, 10, 1000, 1e6]) == pytest.approx(1.0)


def test_group_indices_partitions_every_row():
    keys = np.array([3, 1, 3, 2, 1])
    groups = group_indices(keys)
    assert sorted(np.concatenate(groups).tolist()) == [0, 1, 2, 3, 4]
    assert sorted(len(g) for g in groups) == [1, 2, 2]


def test_permutation_does_not_flag_noise():
    rng = np.random.default_rng(0)
    n = 600
    ts = np.repeat(np.arange(30), 20)
    res = permutation_ic(rng.normal(size=n), rng.normal(size=n), ts,
                         n_perm=200, seed=1)
    assert res["p_value"] > 0.05
    assert abs(res["null_mean"]) < 0.05


def test_permutation_preserves_timestamp_level_effects():
    """A signal that only knows the timestamp must NOT look significant.

    Every contract in a bar shares the underlying's move. A "signal" that is
    constant within each bar carries no ability to choose between contracts,
    and within-bar shuffling leaves it unchanged -- so the test must return a
    non-significant p-value even though the raw IC can be large.
    """
    rng = np.random.default_rng(5)
    ts = np.repeat(np.arange(40), 15)
    bar_effect = rng.normal(size=40)
    signal = bar_effect[ts]                       # constant within a bar
    ret = bar_effect[ts] + rng.normal(0, 0.1, size=len(ts))
    res = permutation_ic(signal, ret, ts, n_perm=200, seed=2)
    assert res["ic"] > 0.5, "setup failed: raw IC should look impressive"
    assert res["p_value"] > 0.05, "timestamp-level effect was called skill"


# ---------------------------------------------------------------------------
# Harness behaviour
# ---------------------------------------------------------------------------

def test_random_signal_reports_null(panel):
    rng = np.random.default_rng(0)
    r = evaluate_signal(panel, rng.normal(size=len(panel)), "random",
                        n_perm=200)
    assert abs(r["ic_vs_net"]) < 0.08
    assert r["permutation_net"]["p_value"] > 0.05


def test_injected_edge_is_detected(panel):
    oracle = inject_oracle_signal(panel, strength=0.6, seed=2)
    r = evaluate_signal(panel, oracle, "oracle", n_perm=200)
    assert r["ic_vs_mid"] > 0.2
    assert r["permutation_net"]["p_value"] < 0.05


def test_spread_erodes_a_perfect_mid_forecast(panel):
    perfect = inject_oracle_signal(panel, strength=1.0, seed=3)
    be = breakeven_spread_capture(panel, perfect)
    assert be["mean_ret_mid"] > be["mean_ret_net"]
    assert be["spread_drag_per_trade"] > 0
    assert be["n_trades"] > 0


def test_spread_derived_signal_is_flagged_mechanical(panel):
    """Ranking on the spread and scoring net of spread is circular.

    `tight_spread` earns net-IC purely because the tight contracts are
    charged less, not because it forecasts anything. The guard must catch it.
    """
    sig = -panel["spread_pct"].to_numpy()
    r = evaluate_signal(panel, sig, "tight_spread", n_perm=100)
    assert abs(r["corr_with_spread"]) > 0.9
    assert r["mechanical_spread_effect"] is True
    assert abs(r["ic_vs_net"]) > abs(r["ic_vs_mid"])


def test_genuine_signal_is_not_flagged_mechanical(panel):
    oracle = inject_oracle_signal(panel, strength=0.8, seed=7)
    r = evaluate_signal(panel, oracle, "oracle", n_perm=100)
    assert r["mechanical_spread_effect"] is False


def test_backtest_makes_money_on_a_net_oracle(panel):
    """A signal equal to the realized NET return must profit."""
    sig = np.nan_to_num(panel["ret_long_net"].to_numpy())
    res = backtest_signal(panel, sig)
    assert res["mean_bar_net"] > 0


def test_backtest_loses_on_an_inverted_signal(panel):
    sig = -np.nan_to_num(panel["ret_long_net"].to_numpy())
    res = backtest_signal(panel, sig)
    assert res["mean_bar_net"] < 0


def test_signal_length_mismatch_is_an_error(panel):
    with pytest.raises(ValueError, match="signal length"):
        backtest_signal(panel, np.zeros(len(panel) + 1))


def test_baseline_ladder_covers_the_delta_confound(panel):
    sigs = baseline_signals(panel)
    for expected in ("random", "always_long", "reversal", "delta_proxy"):
        assert expected in sigs
        assert len(sigs[expected]) == len(panel)


def test_selftest_passes():
    assert selftest(verbose=False) is True


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
