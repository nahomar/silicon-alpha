"""Tests for the model-free static-arbitrage scan (odte.eval.parity_scan).

The scan's job is to decide whether an option chain contradicts its own
arithmetic. Two failure modes matter and both are tested here:

  false positives -- crying arbitrage on a sound chain. Guarded by scanning a
      synthetic chain that is arbitrage-free *by construction*: bid/ask are
      rounded outward from exact Black-76 values, so the executable band is a
      strict superset of the true price and any flag is definitionally wrong.
  false negatives -- missing a real dislocation. Guarded by injecting known
      violations (cheap box, rich box, one corrupted strike) and requiring the
      scan to name them.

Also pinned here is the identifiability result the module is built around: at
0DTE the discount factor is below the resolution of the quote grid, so the
implied rate is not recoverable and the fit must say so rather than report
noise. The same code recovers a 5% rate cleanly on a one-year chain, which is
what makes the 0DTE refusal a property of the data rather than of the fit.

Run:
    PYTHONPATH=. pytest tests/odte/test_parity_scan.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from odte.eval.parity_scan import (
    DEFAULT_T,
    fit_parity_line,
    perturb_quote,
    pivot_chain,
    run_scan,
    scan_boxes,
    scan_parity,
    selftest,
    synthetic_chain,
)


@pytest.fixture
def chain():
    """Arbitrage-free 0DTE chain: F = 5000, 41 strikes, 25-point spacing."""
    return synthetic_chain()


@pytest.fixture
def strikes(chain):
    return np.sort(chain["strike"].unique())


def _deep_itm_call_strike(strikes):
    """A strike whose call is worth hundreds of points.

    Injections need headroom: shifting an ATM 0DTE call (worth ~12 points) by
    40 would drive its bid negative, and the scan would drop the strike as
    unquotable rather than flag it. Deep in the money there is room to move.
    """
    return float(strikes[2])


# ---------------------------------------------------------------------------
# No false positives on a sound chain
# ---------------------------------------------------------------------------

def test_arbitrage_free_chain_has_no_box_violations(chain):
    boxes = scan_boxes(chain)
    assert len(boxes) > 0, "no box pairs were priced at all"
    flagged = boxes[boxes["arb_cheap"] | boxes["arb_rich"]]
    assert len(flagged) == 0, (
        f"{len(flagged)} phantom box arb(s) on an arbitrage-free chain, worst "
        f"edge {flagged['edge'].max():.4f} at "
        f"K={flagged['k1'].iloc[0]:.0f}/{flagged['k2'].iloc[0]:.0f}"
    )


def test_arbitrage_free_chain_has_no_parity_violations(chain):
    par = scan_parity(chain)
    flagged = par[par["violation"]]
    assert len(flagged) == 0, (
        f"{len(flagged)} phantom parity violation(s); worst is "
        f"{flagged['edge'].max():.4f} pts at K={flagged['strike'].iloc[0]:.0f}"
    )


def test_every_box_cost_sits_inside_the_no_arbitrage_band(chain):
    """Cost to buy must exceed 0 and the sell credit must stay under width."""
    boxes = scan_boxes(chain)
    assert (boxes["cost_buy"] > 0).all()
    assert (boxes["proceeds_sell"] < boxes["width"]).all()
    # The bid/ask spread must make buying strictly costlier than selling.
    assert (boxes["cost_buy"] > boxes["proceeds_sell"]).all()


# ---------------------------------------------------------------------------
# The box identity the whole scan rests on
# ---------------------------------------------------------------------------

def test_box_payoff_is_width_invariant():
    """+C(K1) -C(K2) +P(K2) -P(K1) pays K2-K1 wherever the underlying lands."""
    k1, k2 = 4900.0, 5000.0
    underlying = np.linspace(0.0, 10_000.0, 2001)
    payoff = (np.maximum(underlying - k1, 0.0) - np.maximum(underlying - k2, 0.0)
              + np.maximum(k2 - underlying, 0.0) - np.maximum(k1 - underlying, 0.0))
    assert np.allclose(payoff, k2 - k1), (
        f"box payoff is not constant: min={payoff.min()} max={payoff.max()}"
    )


# ---------------------------------------------------------------------------
# Real violations must be caught
# ---------------------------------------------------------------------------

def test_injected_cheap_box_is_flagged(chain, strikes):
    """A call marked 40 points too low makes its box buyable for a credit."""
    k1 = _deep_itm_call_strike(strikes)
    rigged = perturb_quote(chain, k1, "C", bid_delta=-40.0, ask_delta=-40.0)
    boxes = scan_boxes(rigged)
    hit = boxes[np.isclose(boxes["k1"], k1) & boxes["arb_cheap"]]
    assert len(hit) > 0, "underpriced box went unflagged"
    assert (hit["cost_buy"] <= 0).all()
    assert hit["edge"].max() > 30.0


def test_injected_rich_box_is_flagged(chain, strikes):
    """A call marked 40 points too high makes its box sellable above width."""
    k1 = _deep_itm_call_strike(strikes)
    rigged = perturb_quote(chain, k1, "C", bid_delta=40.0, ask_delta=40.0)
    boxes = scan_boxes(rigged)
    hit = boxes[np.isclose(boxes["k1"], k1) & boxes["arb_rich"]]
    assert len(hit) > 0, "overpriced box went unflagged"
    assert (hit["proceeds_sell"] >= hit["width"]).all()


@pytest.mark.parametrize("idx_frac, delta, side", [
    (1 / 3, 30.0, "cheap"),     # OTM put marked up -> combo buyable too cheap
    (2 / 3, -2.0, "rich"),      # ITM put marked down -> combo sellable too rich
])
def test_parity_violation_is_isolated_to_one_strike(chain, strikes,
                                                    idx_frac, delta, side):
    """One bad strike is flagged, the 40 sound ones are not, with the right side.

    Shifting the *put* moves the C - P combo the opposite way: an overpriced
    put means the combo can be bought below the fitted line (`cheap`), an
    underpriced one makes it `rich`.

    The two cases must use different strikes. Marking a put *down* only works
    where the put has value to give up -- above the forward, where it is in the
    money. Below the forward a 0DTE put is quoted at the exchange floor, and
    subtracting from $0.05 produces a negative bid, so the scan would discard
    the strike as unquotable instead of flagging it.
    """
    k_bad = float(strikes[int(len(strikes) * idx_frac)])
    skewed = perturb_quote(chain, k_bad, "P", bid_delta=delta, ask_delta=delta)
    par = scan_parity(skewed)
    flagged = par[par["violation"]]
    assert len(flagged) == 1, (
        f"expected exactly 1 violation, got {len(flagged)}: "
        f"{flagged['strike'].tolist()[:8]}"
    )
    assert float(flagged["strike"].iloc[0]) == pytest.approx(k_bad)
    assert flagged["side"].iloc[0] == side
    assert flagged["edge"].iloc[0] > 0.0


# ---------------------------------------------------------------------------
# Executable prices, not mids
# ---------------------------------------------------------------------------

def test_wide_spreads_suppress_a_marginal_flag(strikes):
    """The same mispricing is arb at a tight touch and not at a wide one.

    This is the point of pricing on touch prices: 27 points of edge is free
    money against a $0.25 half-spread and is swallowed whole by a $20 one.
    """
    k1 = _deep_itm_call_strike(strikes)
    tight = perturb_quote(synthetic_chain(half_spread=0.25), k1, "C",
                          bid_delta=-27.0, ask_delta=-27.0)
    wide = perturb_quote(synthetic_chain(half_spread=20.0), k1, "C",
                         bid_delta=-27.0, ask_delta=-27.0)
    tight_hits = scan_boxes(tight)
    wide_hits = scan_boxes(wide)
    assert tight_hits["arb_cheap"].any(), "tight-market arb not detected"
    assert not wide_hits["arb_cheap"].any(), (
        "flagged an arb that the quoted spread makes untradeable"
    )


def test_fees_reduce_reported_edge(chain, strikes):
    """Edge is quoted net of fees, so a fee must shrink it by 4 legs' worth."""
    k1 = _deep_itm_call_strike(strikes)
    rigged = perturb_quote(chain, k1, "C", bid_delta=-40.0, ask_delta=-40.0)
    gross = scan_boxes(rigged, fee_per_leg=0.0)["edge"].max()
    net = scan_boxes(rigged, fee_per_leg=1.0)["edge"].max()
    assert net == pytest.approx(gross - 4.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Feed hygiene
# ---------------------------------------------------------------------------

def test_crossed_and_nonpositive_quotes_are_dropped(chain, strikes):
    """A crossed or zero-bid strike is discarded, never silently repaired."""
    n_before = len(pivot_chain(chain))
    k_crossed = float(strikes[10])
    k_zero = float(strikes[20])

    dirty = perturb_quote(chain, k_crossed, "C", ask_delta=-1e6)   # ask << bid
    dirty = perturb_quote(dirty, k_zero, "P", bid_delta=-1e6)      # bid <= 0
    after = pivot_chain(dirty)

    assert len(after) == n_before - 2
    assert k_crossed not in set(after["strike"])
    assert k_zero not in set(after["strike"])


def test_missing_required_column_is_an_error(chain):
    with pytest.raises(ValueError, match="missing required column"):
        pivot_chain(chain.drop(columns=["bid"]))


# ---------------------------------------------------------------------------
# What the parity fit can and cannot recover
# ---------------------------------------------------------------------------

def test_forward_is_recovered_at_zero_dte(chain):
    fit = fit_parity_line(chain)
    assert fit.ok
    assert fit.forward == pytest.approx(5000.0, abs=0.05)


def test_rate_is_not_identifiable_at_zero_dte(chain):
    """The headline result: over one session, D is below quote resolution.

    A 5% rate displaces the parity line by ~0.015 index points across the full
    strike span -- under a third of one $0.05 tick. The fit must decline to
    report a rate rather than annualize that noise by dividing by T ~ 1e-3.
    """
    fit = fit_parity_line(chain, T=DEFAULT_T)
    assert fit.ok
    assert not fit.rate_identifiable
    assert fit.discount == 1.0, "D should be pinned, not estimated, at 0DTE"

    span = 500.0                      # furthest strike from the forward
    true_effect = abs(1.0 - math.exp(-0.05 * DEFAULT_T)) * span
    assert true_effect < fit.quote_resolution, (
        f"a 5% rate moves the line {true_effect:.4f} pts, which is not below "
        f"the {fit.quote_resolution:.4f} pts of quote resolution -- the "
        f"premise of pinning D no longer holds"
    )


def test_rate_is_identifiable_on_a_one_year_chain():
    """Same estimator, longer horizon: the rate comes back cleanly.

    This is the control for the test above. If the 0DTE refusal were a defect
    in the fit rather than a fact about the data, this would fail too.
    """
    T, r = 1.0, 0.05
    dated = synthetic_chain(discount=math.exp(-r * T), T=T)
    fit = fit_parity_line(dated, T=T)
    assert fit.ok and fit.rate_identifiable
    assert fit.discount == pytest.approx(math.exp(-r * T), abs=1e-4)
    assert fit.rate == pytest.approx(r, abs=5e-4)
    assert fit.forward == pytest.approx(5000.0, abs=0.05)


def test_forward_is_robust_to_one_corrupted_strike(chain, strikes):
    """A 30-point error on one strike must not move the forward."""
    k_bad = float(strikes[len(strikes) // 3])
    skewed = perturb_quote(chain, k_bad, "P", bid_delta=30.0, ask_delta=30.0)
    fit = fit_parity_line(skewed)
    assert fit.ok
    assert fit.forward == pytest.approx(5000.0, abs=0.05), (
        f"one bad strike dragged the forward to {fit.forward:.3f}"
    )


# ---------------------------------------------------------------------------
# Summary contract and self-test
# ---------------------------------------------------------------------------

def test_run_scan_summary_is_clean_and_json_shaped(chain):
    s = run_scan(chain)
    assert s["verdict"].startswith("CLEAN")
    assert s["n_strikes_usable"] == 41
    assert s["n_strikes_dropped"] == 0
    assert s["boxes"]["n_arb"] == 0
    assert s["parity"]["n_violations"] == 0
    assert s["parity"]["ok"] is True
    assert s["parity"]["rate_identifiable"] is False
    # Every leaf must survive a JSON round-trip (no numpy scalars, no NaN).
    import json
    assert json.loads(json.dumps(s))["verdict"] == s["verdict"]


def test_run_scan_reports_injected_arbitrage(chain, strikes):
    k1 = _deep_itm_call_strike(strikes)
    rigged = perturb_quote(chain, k1, "C", bid_delta=-40.0, ask_delta=-40.0)
    s = run_scan(rigged)
    assert s["verdict"].startswith("INCONSISTENT")
    assert s["boxes"]["n_arb"] > 0
    assert s["boxes"]["max_edge"] > 30.0


def test_selftest_passes():
    assert selftest(verbose=False) is True


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
