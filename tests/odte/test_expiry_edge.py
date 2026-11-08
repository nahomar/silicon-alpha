"""Tests for the hold-to-expiry 0DTE study and the Databento 0DTE reducer."""
import numpy as np
import pandas as pd
import pytest

from odte.data.fetch_0dte import reduce_frame, session_window
from odte.eval import expiry_edge as ee


def test_selftest_passes():
    assert ee.selftest(verbose=False)


def test_quote_visible_only_from_next_whole_minute():
    day = pd.Timestamp("2026-11-06")
    chain = pd.DataFrame({
        "quote_datetime": [day + pd.Timedelta(hours=10, seconds=30)],
        "strike": [6000.0], "option_type": ["C"], "bid": [10.0], "ask": [10.5],
    })
    g = ee.minute_grid(chain)
    assert np.isnan(g["bid"].loc[day + pd.Timedelta(hours=10)].iloc[0])
    assert g["bid"].loc[day + pd.Timedelta(hours=10, minutes=1)].iloc[0] == 10.0
    # carried at most 5 minutes
    assert np.isnan(g["bid"].loc[day + pd.Timedelta(hours=10, minutes=7)].iloc[0])


def test_reduce_frame_parses_osi_and_keeps_only_same_day_expiry():
    ts = pd.Timestamp("2026-11-06 15:00:00", tz="UTC")   # 10:00 New York
    df = pd.DataFrame({
        "ts_recv": [ts, ts, ts],
        "symbol": ["SPXW  261106C05800000", "SPXW  261106P05795000",
                   "SPXW  261109C05800000"],
        "bid_px_00": [12.3, 9.8, 30.0], "ask_px_00": [12.5, 10.0, 30.5],
        "bid_sz_00": [10, 20, 5], "ask_sz_00": [11, 21, 6],
    })
    out = reduce_frame(df, pd.Timestamp("2026-11-06"))
    assert len(out) == 2
    assert out["strike"].tolist() == [5800.0, 5795.0]
    assert out["option_type"].tolist() == ["C", "P"]
    assert (out["quote_datetime"] == pd.Timestamp("2026-11-06 10:00")).all()
    assert (out["root"] == "SPXW").all()


def test_session_window_handles_dst():
    # 2026-10-30 is EDT (UTC-4); 2026-11-02 is EST (UTC-5).
    assert session_window(pd.Timestamp("2026-10-30"))[0].startswith("2026-10-30T13:25")
    assert session_window(pd.Timestamp("2026-11-02"))[0].startswith("2026-11-02T14:25")


def test_sequence_tokens_ignore_minutes_after_t():
    from odte.eval.seq_model import _tokens, minute_paths
    rng = np.random.default_rng(5)
    day = pd.Timestamp("2025-03-03")
    chain, close, S = ee.synthetic_session(day, 6000.0, 0.2, 0.2, rng)
    S2 = S.copy()
    S2[200:] *= 1.02                                    # change the future (after 12:50)
    chain2, _, _ = ee.synthetic_session(day, 6000.0, 0.2, 0.2,
                                        np.random.default_rng(5), path_override=S2)
    t = day + pd.Timedelta(hours=12)
    a, b = _tokens(minute_paths(chain), t), _tokens(minute_paths(chain2), t)
    assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0])
    later = day + pd.Timedelta(hours=14)
    assert not np.allclose(_tokens(minute_paths(chain), later)[0],
                           _tokens(minute_paths(chain2), later)[0])


def _days_with_jump(jump_day_idx=6, jump_min=150, size=0.006, n=8):
    rng = np.random.default_rng(9)
    chains, settles = {}, {}
    for i, d in enumerate(pd.bdate_range("2025-03-03", periods=n)):
        z = rng.normal(0, 0.15 * np.sqrt(1 / (252 * 390)), 390)
        S = 6000 * np.exp(np.concatenate([[0], np.cumsum(z)]))
        if i == jump_day_idx:
            S[jump_min:] *= 1 + size                      # a sudden 0.6% jump at 12:00
        c, close, _ = ee.synthetic_session(d, 6000, 0.15, 0.15, rng, path_override=S)
        chains[d], settles[d] = c, close
    return chains, settles


def test_event_triggers_and_post_event_features():
    chains, settles = _days_with_jump(n=2, jump_day_idx=-1)
    d = sorted(chains)[1]
    ds = ee.build_dataset(chains, settles, times=ee.GRID15,
                          event_times={d: ["14:00"]}, shocks=False)
    day = ds[ds["day"] == d].set_index("hhmm")
    assert day.loc["14:05", "trigger"] == "event" and day.loc["14:30", "trigger"] == "event"
    assert day.loc["14:30", "min_since_event"] == 30
    assert np.isnan(day.loc["13:00", "min_since_event"])     # nothing before the event
    assert day.loc["14:05", "is_event_row"] == 1 and day.loc["13:00", "is_event_row"] == 0


def test_shock_trigger_uses_prior_days_and_fires_after_the_jump():
    chains, settles = _days_with_jump()
    ds = ee.build_dataset(chains, settles, times=ee.GRID15, shocks=True)
    jump_day = sorted(chains)[6]
    sh = ds[(ds["trigger"] == "shock")]
    assert (sh["day"] == jump_day).any()
    times = set(sh[sh["day"] == jump_day]["t"])
    assert jump_day + pd.Timedelta(hours=12, minutes=5) in times  # jump at 12:00 -> decide 12:05
    # Rare by construction: shocks on well under half of the eligible days.
    eligible = sorted(chains)[5:]
    assert sh["day"].nunique() <= len(eligible) // 2 + 1
    first_days = sorted(chains)[:5]                            # no history yet -> no shocks
    assert not sh["day"].isin(first_days).any()


def test_every15_trades_at_most_once_a_day():
    ds = pd.DataFrame({"day": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-03"] * 2)})
    act = np.array([0, -1, 1, 1, 1])
    assert ee._first_per_day(ds, act).tolist() == [0, -1, 0, 1, 0]


def test_early_close_drops_decisions_after_the_bell():
    rng = np.random.default_rng(3)
    day = pd.Timestamp("2026-07-02")
    chain, close, _ = ee.synthetic_session(day, 6000.0, 0.15, 0.15, rng)   # quotes run to 16:00
    ds = ee.build_dataset({day: chain}, {day: close}, times=ee.GRID15,
                          early_closes={day: "13:00"})
    assert ds["hhmm"].max() < "13:00"
    r = ds[ds["hhmm"] == "12:00"].iloc[0]
    assert r["minutes_to_close"] == 60
