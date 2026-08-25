"""Tests for the 0DTE pre-trade risk gates.

This module is what stops the book from carrying an unbounded gamma position
into the close, and it had no test coverage at all. These tests pin the three
behaviours that actually protect capital:

  1. the gamma cap really does tighten through the afternoon, so a position
     that is acceptable at the open is rejected near the bell;
  2. the gate fails *closed* when its clock inputs go missing, rather than
     silently restoring the loosest cap;
  3. pin risk escalates as expiry approaches and as open interest piles up
     next to spot.

Gamma exposure throughout is *gamma dollars*, sum(|w| * |gamma| * S^2 * mult),
not raw gamma. The units matter: `odte.executor.LegacyRiskGates` caps the raw
quantity, and for SPX near 5500 the two differ by about ten orders of
magnitude.
"""
from __future__ import annotations

import numpy as np
import pytest

from odte.exec.qp import InstrumentGreeks
from odte.exec.risk_gates import (
    GateResult, RiskGates, close_for_pennies_orders, gamma_cap_scale,
    pin_score,
)

SPOT = 5500.0
OPEN_BELL = 0          # 09:30 ET
TIGHTEN_START = 300    # 14:30 ET
TIGHTEN_END = 385      # 15:55 ET


def _greeks(n: int = 1, gamma: float = 1e-4, delta: float = 0.5,
            vega: float = 0.1, spot: float = SPOT) -> InstrumentGreeks:
    return InstrumentGreeks(
        spot=np.full(n, spot),
        delta=np.full(n, delta),
        gamma=np.full(n, gamma),
        vega=np.full(n, vega),
        multiplier=np.full(n, 100.0),
    )


def _gamma_dollars(w, g) -> float:
    return float((np.abs(w) * np.abs(g.gamma) * g.spot * g.spot
                  * g.multiplier).sum())


def _gate(**kw) -> RiskGates:
    """A gate with the unrelated caps opened up, isolating what's under test."""
    base = dict(gross_cap=1e12, delta_dollar_cap=1e12, vega_cap=1e12,
                per_symbol_cap=1e12, notional_velocity_cap=1e12,
                orders_per_sec_cap=10**6)
    base.update(kw)
    return RiskGates(**base)


def _check(gate, w, g, **kw) -> GateResult:
    params = dict(required_margin=0.0, equity=1e9,
                  strikes=np.full(len(w), SPOT * 2.0), spot=SPOT)
    params.update(kw)
    return gate.check(w, g, **params)


# ---------------------------------------------------------------------------
# The time-decay schedule
# ---------------------------------------------------------------------------

def test_gamma_scale_is_full_before_the_tightening_window():
    for minute in (0, 100, TIGHTEN_START - 1):
        assert gamma_cap_scale(minute) == 1.0


def test_gamma_scale_reaches_the_floor_at_and_after_the_window():
    for minute in (TIGHTEN_END, TIGHTEN_END + 1, 500):
        assert gamma_cap_scale(minute) == pytest.approx(0.10)


def test_gamma_scale_is_linear_across_the_window():
    mid = (TIGHTEN_START + TIGHTEN_END) // 2
    assert gamma_cap_scale(mid) == pytest.approx(0.55, abs=0.01)
    quarter = TIGHTEN_START + (TIGHTEN_END - TIGHTEN_START) // 4
    assert gamma_cap_scale(quarter) == pytest.approx(0.775, abs=0.01)


def test_gamma_scale_never_increases_through_the_day():
    scales = [gamma_cap_scale(m) for m in range(0, 400)]
    assert all(a >= b for a, b in zip(scales, scales[1:])), "cap loosened"
    assert scales[0] / scales[-1] == pytest.approx(10.0, rel=1e-6)


def test_gamma_scale_honours_custom_window():
    assert gamma_cap_scale(100, tighten_start=100, tighten_end=200,
                           floor=0.5) == pytest.approx(1.0)
    assert gamma_cap_scale(150, tighten_start=100, tighten_end=200,
                           floor=0.5) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# The gamma cap in the gate
# ---------------------------------------------------------------------------

def test_gamma_dollars_use_spot_squared_and_multiplier():
    g = _greeks(n=1, gamma=1e-4)
    w = np.array([2.0])
    res = _check(_gate(), w, g, minute_of_day=OPEN_BELL)
    expected = 2.0 * 1e-4 * SPOT * SPOT * 100.0
    assert res.details["gamma_dollars"] == pytest.approx(expected)


def test_position_allowed_at_the_open_is_vetoed_near_the_close():
    """The behaviour the whole schedule exists for."""
    g = _greeks(n=1, gamma=1e-4)
    w = np.array([1.0])
    exposure = _gamma_dollars(w, g)
    # Cap sits just above the position at full scale, so only the EOD
    # tightening can push the position over the line.
    gate = _gate(gamma_dollar_cap=exposure * 1.05)

    opening = _check(gate, w, g, minute_of_day=OPEN_BELL)
    closing = _check(gate, w, g, minute_of_day=TIGHTEN_END)

    assert opening.ok, opening.failed
    assert not closing.ok
    assert "gamma_dollar_cap_eod" in closing.failed
    assert closing.details["gamma_cap_dynamic"] < opening.details["gamma_cap_dynamic"]


def test_gamma_veto_names_the_eod_reason():
    g = _greeks(n=1, gamma=1e-3)
    w = np.array([10.0])
    res = _check(_gate(gamma_dollar_cap=1.0), w, g, minute_of_day=OPEN_BELL)
    assert not res.ok
    assert res.failed == ["gamma_dollar_cap_eod"]


def test_flat_book_passes_every_gate():
    g = _greeks(n=3)
    res = _check(_gate(), np.zeros(3), g, minute_of_day=OPEN_BELL,
                 strikes=np.full(3, SPOT * 2))
    assert res.ok, res.failed
    assert res.details["gamma_dollars"] == 0.0


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------

def test_missing_clock_applies_the_tightest_gamma_cap():
    """A dropped field must not restore the opening-bell cap.

    This is the regression that motivated the change: the old defaults
    (minute 0, 9999 minutes to expiry) were the loosest possible values, so a
    mis-keyed field disabled the gate while the logs still showed it passing.
    """
    g = _greeks(n=1, gamma=1e-4)
    w = np.array([1.0])
    gate = _gate(gamma_dollar_cap=_gamma_dollars(w, g) * 1.05)

    missing = _check(gate, w, g)                       # no minute_of_day
    at_close = _check(gate, w, g, minute_of_day=TIGHTEN_END)

    assert not missing.ok
    assert "gamma_dollar_cap_eod" in missing.failed
    assert missing.details["clock_missing"] == 1.0
    assert missing.details["gamma_cap_dynamic"] == pytest.approx(
        at_close.details["gamma_cap_dynamic"])


def test_supplied_clock_is_not_flagged_as_missing():
    g = _greeks(n=1)
    res = _check(_gate(), np.array([1.0]), g, minute_of_day=OPEN_BELL)
    assert res.details["clock_missing"] == 0.0


def test_missing_time_to_expiry_scores_pin_as_imminent():
    """Absent expiry must maximize the pin amplifier, not zero the score."""
    g = _greeks(n=1)
    w = np.array([1.0])
    strikes = np.array([SPOT])
    oi = np.array([10_000.0])

    missing = _check(_gate(), w, g, strikes=strikes, open_interest=oi,
                     minute_of_day=OPEN_BELL)
    far_away = _check(_gate(), w, g, strikes=strikes, open_interest=oi,
                      minute_of_day=OPEN_BELL, minutes_to_expiry=9999.0)

    assert missing.details["pin_score"] > 0.0
    assert far_away.details["pin_score"] == 0.0
    assert missing.details["pin_score"] > far_away.details["pin_score"]


# ---------------------------------------------------------------------------
# Pin risk
# ---------------------------------------------------------------------------

def test_pin_score_is_zero_outside_the_window():
    assert pin_score(SPOT, np.array([SPOT]), np.array([1e4]),
                     minutes_to_expiry=121.0) == 0.0


def test_pin_score_is_zero_without_open_interest():
    assert pin_score(SPOT, np.array([SPOT]), np.array([0.0]),
                     minutes_to_expiry=5.0) == 0.0
    assert pin_score(SPOT, np.array([]), np.array([]),
                     minutes_to_expiry=5.0) == 0.0


def test_pin_score_rises_as_expiry_approaches():
    # Spot is held one proximity-scale (spot * 1e-3) away from the strike so
    # the base score is ~37 rather than 100. Sitting exactly on the strike
    # saturates the score and hides the time amplifier entirely -- see
    # test_pin_score_saturates_on_an_atm_magnet_strike.
    args = (SPOT, np.array([SPOT + SPOT * 1e-3]), np.array([1e4]))
    far = pin_score(*args, minutes_to_expiry=110.0)
    near = pin_score(*args, minutes_to_expiry=5.0)
    assert near > far > 0.0


def test_pin_score_saturates_on_an_atm_magnet_strike():
    """A single strike exactly at spot pins the score at 100 regardless of time.

    Proximity is 1.0 and the OI weight is 1.0, so the base is already 100 and
    the amplifier is clipped away. Worth knowing: inside the two-hour window
    an exactly-ATM magnet blocks on pin score at any time-to-expiry, so the
    score cannot be used to rank urgency in that configuration.
    """
    args = (SPOT, np.array([SPOT]), np.array([1e4]))
    assert pin_score(*args, minutes_to_expiry=119.0) == pytest.approx(100.0)
    assert pin_score(*args, minutes_to_expiry=1.0) == pytest.approx(100.0)


def test_pin_score_falls_as_spot_moves_away_from_the_magnet():
    on_strike = pin_score(SPOT, np.array([SPOT]), np.array([1e4]), 10.0)
    off_strike = pin_score(SPOT, np.array([SPOT + 100.0]), np.array([1e4]), 10.0)
    assert on_strike > off_strike


def test_pin_score_is_bounded_to_0_100():
    score = pin_score(SPOT, np.array([SPOT]), np.array([1e9]),
                      minutes_to_expiry=0.0)
    assert 0.0 <= score <= 100.0


def test_high_pin_score_blocks_and_recommends_flattening():
    g = _greeks(n=1)
    res = _check(_gate(), np.array([1.0]), g,
                 strikes=np.array([SPOT]), open_interest=np.array([1e5]),
                 minute_of_day=OPEN_BELL, minutes_to_expiry=1.0,
                 symbols=["SPXW_C5500"])
    assert "pin_score_block" in res.failed
    assert res.recommend_close_for_pennies == ["SPXW_C5500"]


def test_pin_distance_gate_fires_when_spot_sits_on_the_strike():
    g = _greeks(n=1)
    res = _check(_gate(), np.array([1.0]), g, strikes=np.array([SPOT]),
                 minute_of_day=OPEN_BELL)
    assert "pin_dist" in res.failed


# ---------------------------------------------------------------------------
# Margin and rate limits
# ---------------------------------------------------------------------------

def test_margin_buffer_blocks_above_the_equity_fraction():
    g = _greeks(n=1)
    gate = _gate(equity_buffer_pct=0.95)
    ok = _check(gate, np.array([1.0]), g, required_margin=94.0, equity=100.0,
                minute_of_day=OPEN_BELL)
    bad = _check(gate, np.array([1.0]), g, required_margin=96.0, equity=100.0,
                 minute_of_day=OPEN_BELL)
    assert "margin_buffer" not in ok.failed
    assert "margin_buffer" in bad.failed
    assert bad.details["equity_used_pct"] == pytest.approx(0.96)


def test_order_rate_limit_trips_after_the_cap():
    g = _greeks(n=1)
    gate = _gate(orders_per_sec_cap=3)
    w = np.array([0.001])
    passes = sum(_check(gate, w, g, minute_of_day=OPEN_BELL).ok
                 for _ in range(10))
    assert passes == 3, f"expected exactly 3 accepted orders, got {passes}"


def test_rejected_orders_do_not_consume_rate_budget():
    """Only accepted orders should count toward the per-second limit."""
    g = _greeks(n=1, gamma=1e-3)
    gate = _gate(orders_per_sec_cap=2, gamma_dollar_cap=1.0)
    for _ in range(5):
        assert not _check(gate, np.array([10.0]), g,
                          minute_of_day=OPEN_BELL).ok
    # Budget untouched, so a compliant order still gets through.
    gate.gamma_dollar_cap = 1e12
    assert _check(gate, np.array([0.001]), g, minute_of_day=OPEN_BELL).ok


# ---------------------------------------------------------------------------
# close_for_pennies
# ---------------------------------------------------------------------------

def test_close_for_pennies_sells_a_long_with_a_tight_spread():
    orders = close_for_pennies_orders(
        ["A"], {"A": 5.0}, {"A": 0.03}, {"A": 0.05})
    assert orders == [{"symbol": "A", "side": "sell", "qty": 5.0,
                       "reason": "pin_flatten_long"}]


def test_close_for_pennies_buys_back_a_short_at_a_penny_ask():
    orders = close_for_pennies_orders(
        ["A"], {"A": -4.0}, {"A": 0.01}, {"A": 0.04})
    assert orders[0]["side"] == "buy"
    assert orders[0]["qty"] == 4.0


def test_close_for_pennies_skips_flat_and_unquoted_positions():
    assert close_for_pennies_orders(["A"], {"A": 0.0}, {"A": 0.01},
                                    {"A": 0.02}) == []
    assert close_for_pennies_orders(["A"], {"A": 5.0}, {"A": 0.0},
                                    {"A": 0.0}) == []


def test_close_for_pennies_leaves_a_wide_long_alone():
    """A long with a wide spread is not a penny position; don't dump it."""
    assert close_for_pennies_orders(["A"], {"A": 5.0}, {"A": 1.00},
                                    {"A": 3.00}) == []


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
