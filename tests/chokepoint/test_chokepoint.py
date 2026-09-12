"""Offline tests for the CHOKEPOINT track.

Deliberately network-free so they can live in CI, unlike the probe itself
(`odte/eval/signal_probe.py` set the precedent that network-dependent probes
stay out of CI).

These test the two things that would silently produce a wrong answer: the HAC
estimator behind every t-statistic, and the guards that are supposed to stop
fabricated or look-ahead data from reaching a result.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from chokepoint import countries
from chokepoint.data import eskom, sources
from chokepoint.probe.eskom_pgm import ols
from chokepoint.research import concentration


# ---------------------------------------------------------------------------
# Country dependency map
# ---------------------------------------------------------------------------

def _shares(rows):
    return pd.DataFrame(rows, columns=["commodity", "country", "share_pct"])


def test_chokehold_is_the_largest_single_commodity_share():
    df = _shares([
        ("cobalt", "DR Congo", 70), ("cobalt", "Australia", 3),
        ("tantalum", "DR Congo", 40),
    ])
    drc = next(p for p in countries.build_from(df) if p.country == "DR Congo")
    assert drc.chokehold == pytest.approx(0.70)
    assert drc.chokehold_commodity == "cobalt"
    assert drc.is_chokepoint


def test_breadth_counts_only_material_shares():
    """A 5% share is not leverage — other producers absorb that."""
    df = _shares([
        ("a", "X", 60), ("b", "X", 25), ("c", "X", 5), ("d", "X", 1),
    ])
    x = countries.build_from(df)[0]
    assert x.breadth == 2, "only shares >= 20% should count"
    assert x.supply_at_risk == pytest.approx(0.85)


def test_supply_at_risk_can_exceed_one():
    """It sums across DIFFERENT commodities, so >1.0 is correct, not a bug.

    Guards the formatting fix: this was rendered as '458%' and read as a
    broken calculation.
    """
    df = _shares([("a", "X", 80), ("b", "X", 70), ("c", "X", 60)])
    assert countries.build_from(df)[0].supply_at_risk > 1.0


def test_actionable_requires_both_leverage_and_visibility():
    """Uses a country with no registered source, deliberately.

    An earlier version asserted DR Congo was unwatchable. That was true until
    CME cobalt futures were registered, and then this test failed — correctly,
    because it had pinned a fact about the world rather than the rule. Which
    countries are visible changes every time a source is added; that 'leverage
    without visibility is not actionable' does not.
    """
    df = _shares([("unobtainium", "Ruritania", 70)])
    country = countries.build_from(df)[0]
    assert country.is_chokepoint
    # No registered source -> falls back to USGS annual -> invisible.
    assert not country.is_watchable
    assert not country.actionable, "leverage without visibility is not actionable"


def test_registered_fast_source_makes_a_country_watchable():
    """The other half of the rule, pinned against the registry's own contents."""
    df = _shares([("platinum", "South Africa", 70)])
    sa = countries.build_from(df)[0]
    assert sa.is_watchable and sa.actionable


# ---------------------------------------------------------------------------
# ComexStat (offline parts only — the network path is not in CI)
# ---------------------------------------------------------------------------

def test_unknown_commodity_raises_before_any_network_call():
    from chokepoint.data import comexstat

    with pytest.raises(KeyError, match="unknown commodity"):
        comexstat.fetch("unobtainium", "2024-01", "2024-06")


def test_monthly_source_refuses_a_daily_signal():
    """The horizon guard has to hold through the wrapper, not just the registry."""
    from chokepoint.data import comexstat

    with pytest.raises(sources.SourceUnusableError, match="predicting the past"):
        comexstat.require_horizon(1)
    comexstat.require_horizon(90)  # within reach — must not raise


def test_ncm_codes_are_unique_ints():
    """A duplicated code would silently make two commodities the same series."""
    from chokepoint.data import comexstat

    codes = list(comexstat.NCM.values())
    assert all(isinstance(c, int) for c in codes)
    assert len(set(codes)) == len(codes)


def test_low_share_country_is_not_a_chokepoint():
    df = _shares([("gold", "Mali", 2), ("gold", "Ghana", 4)])
    assert not any(p.is_chokepoint for p in countries.build_from(df))


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(size=n)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=n)
    fit = ols(y, np.column_stack([np.ones(n), x]), ["const", "x"], hac=False)
    assert fit.beta_of("const") == pytest.approx(2.0, abs=0.05)
    assert fit.beta_of("x") == pytest.approx(3.0, abs=0.05)


def test_newey_west_widens_errors_under_autocorrelation():
    """The entire reason the probe uses HAC.

    Loadshedding stage is highly persistent. With autocorrelated residuals,
    classical OLS errors are understated and t-statistics inflated — which is
    how a persistent regressor manufactures 'significance' from noise.
    """
    rng = np.random.default_rng(1)
    n = 1200
    # AR(1) regressor and AR(1) errors, independent of each other.
    x = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.95 * x[t - 1] + rng.normal()
        e[t] = 0.95 * e[t - 1] + rng.normal()
    X = np.column_stack([np.ones(n), x])

    classical = ols(e, X, ["const", "x"], hac=False)
    hac = ols(e, X, ["const", "x"], hac=True)

    i = hac.names.index("x")
    assert hac.se[i] > classical.se[i], (
        "HAC standard error should exceed classical under positive "
        "autocorrelation; if not, the Bartlett kernel is misapplied"
    )


def test_no_relationship_is_not_significant():
    """A null must read as a null. Guards against a sign/scale error that
    would make every run look like a discovery."""
    rng = np.random.default_rng(2)
    n = 800
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # independent of x by construction
    fit = ols(y, np.column_stack([np.ones(n), x]), ["const", "x"])
    assert abs(fit.t_of("x")) < 2.0


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

def test_long_tail_commodity_is_not_flagged_concentrated():
    """Regression test for the residual-bucket bug.

    A commodity whose listed shares sum well below 100% (many small producers)
    must NOT score as concentrated. Lumping the tail into one synthetic
    producer squared it as a single firm and flagged unconcentrated gold at
    HHI 3056.
    """
    df = pd.DataFrame({
        "commodity": ["gold"] * 5,
        "country": ["China", "Russia", "Australia", "Canada", "Ghana"],
        "share_pct": [10, 9, 9, 6, 4],  # sums to 38 — long fragmented tail
    })
    prof = concentration.profile(df)[0]
    assert prof.hhi < 1000, f"long-tail commodity scored HHI {prof.hhi:.0f}"
    assert not prof.is_concentrated
    assert not prof.is_african_chokepoint


def test_single_dominant_african_producer_is_a_chokepoint():
    df = pd.DataFrame({
        "commodity": ["platinum"] * 3,
        "country": ["South Africa", "Russia", "Zimbabwe"],
        "share_pct": [70, 10, 8],
    })
    prof = concentration.profile(df)[0]
    assert prof.is_concentrated
    assert prof.is_african_chokepoint
    assert prof.top_producer == "South Africa"
    assert prof.africa_share == pytest.approx(0.78)


def test_hhi_is_a_lower_bound():
    """Adding a previously-omitted tail producer can only raise HHI."""
    base = pd.DataFrame({
        "commodity": ["x"] * 2, "country": ["A", "B"], "share_pct": [50, 30],
    })
    extended = pd.concat([base, pd.DataFrame({
        "commodity": ["x"], "country": ["C"], "share_pct": [10],
    })])
    assert concentration.profile(extended)[0].hhi >= concentration.profile(base)[0].hhi


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

def test_missing_stage_file_raises_rather_than_fabricating():
    """The most important test here.

    Silently plausible fake data is what `docs/data_integrity_finding.md`
    already cost this repo once. Absent data must be loud.
    """
    with pytest.raises(eskom.MissingStageData) as exc:
        eskom.load_stages("/nonexistent/eskom_stages.csv")
    assert "date,stage" in str(exc.value), "error must explain the schema"


def test_zero_lag_is_rejected_as_lookahead(tmp_path):
    p = tmp_path / "stages.csv"
    p.write_text("date,stage\n2023-01-02,4\n2023-01-03,6\n2023-01-04,2\n")
    hist = eskom.load_stages(p)
    sessions = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
    with pytest.raises(ValueError, match="look-ahead"):
        eskom.align_to_sessions(hist, sessions, lag_days=0)


def test_stage_alignment_uses_only_prior_information(tmp_path):
    p = tmp_path / "stages.csv"
    p.write_text("date,stage\n2023-01-02,1\n2023-01-03,5\n2023-01-04,8\n")
    hist = eskom.load_stages(p)
    sessions = pd.DatetimeIndex(["2023-01-03", "2023-01-04"])
    aligned = eskom.align_to_sessions(hist, sessions, lag_days=1)
    # The session on the 4th may only know the stage as of the 3rd.
    assert aligned.loc["2023-01-04"] == 5.0
    assert aligned.loc["2023-01-04"] != 8.0


def test_sessions_beyond_stage_record_are_not_forward_filled(tmp_path):
    """Regression test: reindex(method='ffill') extrapolated forever.

    The first live run carried the final stage across ~350 sessions past the
    end of the record, fabricating a zero-variance tail that the probe then
    read as evidence.
    """
    p = tmp_path / "stages.csv"
    p.write_text("date,stage\n2023-01-02,4\n2023-01-03,6\n")
    hist = eskom.load_stages(p)
    sessions = pd.DatetimeIndex(["2023-01-03", "2023-01-04", "2023-06-01"])
    aligned = eskom.align_to_sessions(hist, sessions, lag_days=1)
    assert np.isnan(aligned.loc["2023-06-01"]), (
        "sessions after the stage record must be NaN, not forward-filled"
    )


def test_degenerate_eval_split_refuses_to_render_a_verdict():
    """A null is only evidence if the test could have detected an effect."""
    from chokepoint.probe.eskom_pgm import InsufficientPower, _require_power

    x = np.concatenate([np.random.default_rng(0).normal(size=200),
                        np.zeros(100)])  # eval split is all zeros
    with pytest.raises(InsufficientPower, match="absence of an experiment"):
        _require_power(x, split=200)


def test_power_check_passes_when_both_splits_have_variance():
    from chokepoint.probe.eskom_pgm import _require_power

    x = np.random.default_rng(0).normal(size=300)
    _require_power(x, split=200)  # must not raise


def test_out_of_range_stage_rejected(tmp_path):
    p = tmp_path / "stages.csv"
    p.write_text("date,stage\n2023-01-02,4\n2023-01-03,99\n")
    with pytest.raises(ValueError, match="outside"):
        eskom.load_stages(p)


def test_annual_source_cannot_serve_a_daily_signal():
    usgs = sources.get("usgs_mcs")
    assert not usgs.is_signal_capable
    with pytest.raises(sources.SourceUnusableError, match="predicting the past"):
        usgs.require_horizon(timedelta(days=1))


def test_daily_source_can_serve_a_daily_signal():
    sources.get("eskom_stages").require_horizon(timedelta(days=1))


def test_registry_invariant_holds():
    sources._check_registry_invariant()
