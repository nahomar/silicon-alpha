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


def test_firing_placebo_invalidates_the_test_even_with_no_real_hits(capsys):
    """Regression: the placebo check was gated behind real hits.

    The live run produced FCX/SCCO/TECK all insignificant and the GDX placebo
    at t=+2.96. Because no real target fired, the verdict skipped the placebo
    branch entirely and printed "NO PREDICTIVE CONTENT" — blind in exactly the
    case the placebo exists to catch. A null from an unsound test is not
    evidence of absence.
    """
    from chokepoint.probe.inventory_miners import Result, report

    results = [
        Result("FCX", t_oos=-0.87, beta_oos=-0.048, t_is=-0.23,
               n_train=114, n_eval=49),
        Result("GDX", t_oos=2.96, beta_oos=0.119, t_is=-0.64,
               n_train=114, n_eval=49, is_placebo=True),
    ]
    report(results, n=163)
    out = capsys.readouterr().out
    assert "UNSOUND" in out
    assert "NO PREDICTIVE CONTENT" not in out


def test_clean_placebo_allows_a_null_verdict(capsys):
    from chokepoint.probe.inventory_miners import Result, report

    results = [
        Result("FCX", t_oos=-0.87, beta_oos=-0.048, t_is=-0.23,
               n_train=114, n_eval=49),
        Result("GDX", t_oos=0.40, beta_oos=0.01, t_is=-0.10,
               n_train=114, n_eval=49, is_placebo=True),
    ]
    report(results, n=163)
    out = capsys.readouterr().out
    assert "NO PREDICTIVE CONTENT" in out
    assert "UNSOUND" not in out


def test_european_numbers_parse_correctly():
    """'88.950' is eighty-eight thousand, not 88.95."""
    from chokepoint.data.cochilco import _euro_number

    assert _euro_number("88.950") == 88950.0
    assert _euro_number("1.414,4") == pytest.approx(1414.4)
    assert _euro_number("-19.000") == -19000.0
    assert _euro_number("-25") == -25.0


def test_missing_values_are_none_not_zero():
    """Zero inventory is a market event; missing data is not.

    Reading a blank as 0.0 would print an empty warehouse into the series.
    """
    from chokepoint.data.cochilco import _euro_number

    for blank in ("", "   ", "-", "n/d", "s/i"):
        assert _euro_number(blank) is None


def test_period_parsing_handles_years_and_spanish_months():
    from chokepoint.data.cochilco import _parse_period

    assert _parse_period("2021") == pd.Timestamp("2021-12-31")
    assert _parse_period("ENE/JAN 2024") == pd.Timestamp("2024-01-31")
    assert _parse_period("DIC/DEC 2023") == pd.Timestamp("2023-12-31")
    assert _parse_period("not a period") is None


def test_br_split_cells_align_positionally():
    """A <td> is a column fragment, not a value — the core parsing assumption."""
    from chokepoint.data.cochilco import _rows_from_html

    html = "<tr><td>2021<br>2022</td><td>88.950<br>88.925</td></tr>"
    rows = _rows_from_html(html)
    assert rows == [[["2021", "2022"], ["88.950", "88.925"]]]


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


# ---------------------------------------------------------------------------
# Universe audit + recorder
# ---------------------------------------------------------------------------

def test_subunit_currencies_are_divided_by_one_hundred():
    """Regression: London quotes pence, Johannesburg quotes cents.

    The first audit ranked AAL.L at "8,319,447,954" against SBSW at
    "17,953,282" and called the first more liquid. Those were pence and
    dollars. Without the subunit divisor the entire cross-venue liquidity
    ranking is meaningless, and two genuinely illiquid names (PDL.L, GEMD.L)
    passed the screen.
    """
    from chokepoint.universe import audit

    assert audit.SUBUNIT["GBp"] == ("GBP", 100.0)
    assert audit.SUBUNIT["ZAc"] == ("ZAR", 100.0)


def test_usd_needs_no_fx_lookup():
    from chokepoint.universe.audit import fx_rate

    assert fx_rate("USD") == 1.0


def test_blank_currency_has_no_rate():
    """No currency means no USD volume, so the ticker cannot pass the screen."""
    import numpy as np
    from chokepoint.universe.audit import fx_rate

    assert np.isnan(fx_rate(""))


def test_tradeable_flag_needs_data_liquidity_and_freshness():
    from chokepoint.universe.audit import TickerAudit

    base = dict(ticker="X", name="X", listing="foreign", country="Ghana",
                commodity="gold", africa_share=1.0, ok=True, rows=1000,
                stale_days=1, median_usd_vol=1e6)
    assert TickerAudit(**base).tradeable

    assert not TickerAudit(**{**base, "median_usd_vol": 1_000.0}).tradeable
    assert not TickerAudit(**{**base, "rows": 10}).tradeable
    assert not TickerAudit(**{**base, "stale_days": 400}).tradeable
    assert not TickerAudit(**{**base, "ok": False}).tradeable


def test_recorder_refuses_to_run_without_a_verified_universe(tmp_path):
    """It must not silently fall back to the candidate list.

    An unverified ticker that returns nothing is indistinguishable, once it is
    in the store, from a real outage in a ticker that normally works.
    """
    from chokepoint.record import recorder

    with pytest.raises(recorder.NoUniverse, match="universe.audit"):
        recorder.tradeable_universe(tmp_path / "missing.csv")


def test_recorder_rejects_an_audit_with_no_survivors(tmp_path):
    from chokepoint.record import recorder

    p = tmp_path / "audit.csv"
    pd.DataFrame({"ticker": ["A"], "tradeable": [False]}).to_csv(p, index=False)
    with pytest.raises(recorder.NoUniverse, match="no tradeable"):
        recorder.tradeable_universe(p)


# ---------------------------------------------------------------------------
# Supply vintage store
# ---------------------------------------------------------------------------

def _obs(rows):
    return pd.DataFrame(rows, columns=["series", "period", "value"])


def test_first_print_is_recorded(tmp_path, monkeypatch):
    from chokepoint.record import supply

    monkeypatch.setattr(supply, "STORE", tmp_path)
    rep = supply.append_observations("t", _obs([("x", "2024-01-31", 100.0)]))
    assert rep.first_prints == 1 and rep.revisions == 0
    assert len(supply.load("t")) == 1


def test_unchanged_value_writes_nothing(tmp_path, monkeypatch):
    """Polling a monthly series every day must not grow the store."""
    from chokepoint.record import supply

    monkeypatch.setattr(supply, "STORE", tmp_path)
    o = _obs([("x", "2024-01-31", 100.0)])
    supply.append_observations("t", o)
    rep = supply.append_observations("t", o)

    assert rep.unchanged == 1 and rep.first_prints == 0 and rep.revisions == 0
    assert len(supply.load("t")) == 1, "re-poll duplicated a row"


def test_revision_appends_without_destroying_the_first_print(tmp_path, monkeypatch):
    """The entire reason this store exists.

    Official statistics get restated. If a revision overwrote the original we
    could never reconstruct what was knowable at the time, and every backtest
    would quietly use numbers nobody had.
    """
    from chokepoint.record import supply

    monkeypatch.setattr(supply, "STORE", tmp_path)
    supply.append_observations("t", _obs([("x", "2024-01-31", 100.0)]))
    rep = supply.append_observations("t", _obs([("x", "2024-01-31", 118.0)]))

    assert rep.revisions == 1
    df = supply.load("t")
    assert len(df) == 2, "revision should append, not overwrite"
    assert sorted(df.value) == [100.0, 118.0], "first print was lost"


def test_vintage_lets_you_reconstruct_what_was_known(tmp_path, monkeypatch):
    """Point-in-time reconstruction: the first vintage is the first print."""
    from chokepoint.record import supply

    monkeypatch.setattr(supply, "STORE", tmp_path)
    supply.append_observations("t", _obs([("x", "2024-01-31", 100.0)]))
    supply.append_observations("t", _obs([("x", "2024-01-31", 118.0)]))

    df = supply.load("t").sort_values("captured_at")
    as_first_known = df.iloc[0].value
    as_known_now = df.iloc[-1].value
    assert as_first_known == 100.0
    assert as_known_now == 118.0


def test_float_noise_is_not_treated_as_a_revision(tmp_path, monkeypatch):
    """Otherwise a parquet round-trip appends a row on every single poll."""
    from chokepoint.record import supply

    monkeypatch.setattr(supply, "STORE", tmp_path)
    supply.append_observations("t", _obs([("x", "2024-01-31", 100.0)]))
    rep = supply.append_observations("t", _obs([("x", "2024-01-31", 100.0 + 1e-13)]))
    assert rep.unchanged == 1 and rep.revisions == 0


def test_empty_poll_is_an_error_not_a_silent_success(tmp_path, monkeypatch):
    """A source returning nothing must not read as 'nothing happened'."""
    from chokepoint.record import supply

    monkeypatch.setattr(supply, "STORE", tmp_path)
    rep = supply.append_observations("t", _obs([]))
    assert rep.error and "no observations" in rep.error


def test_quarantined_ncm_codes_are_not_polled():
    """Guessed codes that returned nothing must stay out of the live map.

    A code that silently returns an empty series is worse than an absent one:
    the gap later reads as 'Brazil exported no bauxite' rather than 'we asked
    the wrong question'.
    """
    from chokepoint.data import comexstat

    assert set(comexstat.NCM) & set(comexstat.UNVERIFIED_NCM) == set()
    assert "bauxite" in comexstat.UNVERIFIED_NCM
    assert "ferroniobium" in comexstat.NCM
