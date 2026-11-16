"""Tests for the event calendar, daily context, and social activity features.

All offline: sessions are synthetic business days with known holidays, so
the rules can be checked against dates that are known by hand.
"""
import numpy as np
import pandas as pd

from odte.data import calendar as cal
from odte.data import context as ctx
from social import features as sf

HOLIDAYS = pd.to_datetime(["2025-11-27", "2025-12-25", "2026-01-01", "2026-04-03",
                           "2026-06-19", "2026-07-03"])
SESSIONS = pd.bdate_range("2025-11-03", "2026-07-31").difference(HOLIDAYS)


def test_events_map_to_the_session_they_affect():
    # Jobs report on Good Friday (market closed) -> next session, pre-open.
    assert cal.map_to_session(SESSIONS, "2026-04-03", "08:30") == (pd.Timestamp("2026-04-06"), None)
    # After-close earnings -> next session.
    assert cal.map_to_session(SESSIONS, "2026-01-29", "16:00") == (pd.Timestamp("2026-01-30"), None)
    # 08:30 on a session day is pre-open news for that day.
    assert cal.map_to_session(SESSIONS, "2026-01-13", "08:30") == (pd.Timestamp("2026-01-13"), None)
    # 14:00 FOMC keeps its intraday time.
    assert cal.map_to_session(SESSIONS, "2026-03-18", "14:00") == (pd.Timestamp("2026-03-18"), "14:00")


def test_rules_respect_holidays():
    r = cal.rule_events(SESSIONS)
    on = lambda ev: set(r.loc[r["event"] == ev, "date"])
    # June 2026 opex: 3rd Friday is Juneteenth -> Thursday the 18th, quad witching.
    assert pd.Timestamp("2026-06-18") in on("opex") and pd.Timestamp("2026-06-18") in on("quad_witching")
    # Thanksgiving week: claims move to Wednesday.
    assert pd.Timestamp("2025-11-26") in on("claims")
    # VIX expiry for March 2026: 30 days before Fri 17-Apr-2026.
    assert pd.Timestamp("2026-03-18") in on("vix_expiry")
    assert pd.Timestamp("2026-04-02") in on("pre_holiday")
    assert pd.Timestamp("2026-04-06") in on("post_holiday")


def test_fomc_is_ahead_only_before_the_statement():
    ev = pd.DataFrame({"session": [pd.Timestamp("2026-03-18")] * 2,
                       "session_time": ["14:00", None],
                       "event": ["fomc_sep", "ppi"]})
    f13 = cal.features_for(ev, "2026-03-18", "13:00")
    f15 = cal.features_for(ev, "2026-03-18", "15:00")
    assert f13["fomc_ahead"] == 1 and f13["events_ahead"] == 1 and f13["minutes_to_next_event"] == 60
    assert f15["fomc_ahead"] == 0 and f15["events_ahead"] == 0
    assert f13["ev_fomc"] == 1 and f13["ev_fomc_sep"] == 1 and f13["ev_ppi"] == 1
    assert set(cal.CAL_FEATURES) == set(f13)


def test_context_uses_only_prior_days():
    idx = pd.bdate_range("2026-01-01", periods=40)
    rng = np.random.default_rng(0)
    wide = pd.DataFrame({k: 10 + rng.random(40) for k in ctx.TICKERS}, index=idx)
    day = idx[30]
    a = ctx.features_for(wide, day)
    w2 = wide.copy()
    w2.loc[w2.index >= day] *= 3.0                 # rewrite today and the future
    b = ctx.features_for(w2, day)
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a)
    assert set(ctx.CTX_FEATURES) <= set(a)


def test_hourly_rate_ignores_the_incomplete_hour():
    hours = pd.date_range("2026-10-01 13:00", periods=4, freq="h")      # UTC
    h = sf._Hourly(pd.Series([10, 20, 30, 1000], index=hours))
    # Window [14:00, 16:00): buckets 14 and 15 only; 16:00 bucket is not complete.
    assert h.rate(pd.Timestamp("2026-10-01 14:00"), pd.Timestamp("2026-10-01 16:00")) == 25
    # Outside coverage -> unknown, not zero.
    assert np.isnan(h.rate(pd.Timestamp("2026-09-30 10:00"), pd.Timestamp("2026-09-30 12:00")))


def test_slot_zscore_uses_prior_sessions_only():
    days = pd.bdate_range("2026-01-05", periods=30)
    keys = pd.DataFrame({"day": days, "hhmm": "10:00"})
    raw = pd.DataFrame({"rd_wsb_c_1h": np.arange(30, dtype=float) + 100})
    z1 = sf.zscore_by_slot(raw, keys)
    raw2 = raw.copy()
    raw2.iloc[25:] = 1e6                          # change the future only
    z2 = sf.zscore_by_slot(raw2, keys)
    pd.testing.assert_series_equal(z1["rd_wsb_c_1h"].iloc[:25], z2["rd_wsb_c_1h"].iloc[:25])
    assert z1["rd_wsb_c_1h"].iloc[:5].isna().all()      # not enough history yet
