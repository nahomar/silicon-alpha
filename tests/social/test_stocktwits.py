"""Tests for the Stocktwits recorder's backward paging and resume rules.

The fake stream mimics the real one: 30 newest messages before `max`
(inclusive), newest first.
"""
import pandas as pd
import pytest

from social import stocktwits as stw

BASE = 1790800000   # 2026-09-30 ~20:26Z


def _msg(i):
    return {"id": i, "created_at": pd.Timestamp(BASE + 600 * i, unit="s", tz="UTC").isoformat(),
            "body": f"m{i}", "user": {"username": f"u{i % 5}"},
            "entities": {"sentiment": {"basic": "Bullish"} if i % 3 == 0 else None},
            "symbols": [{"symbol": "SPY"}], "likes": {"total": 99}}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    state = {"top": 200, "calls": 0, "fail_after": None}

    def _get(symbol, max_id=None, attempts=6):
        state["calls"] += 1
        if state["fail_after"] is not None and state["calls"] > state["fail_after"]:
            raise RuntimeError("simulated crash")
        hi = state["top"] if max_id is None else min(max_id, state["top"])
        return [_msg(i) for i in range(hi, max(hi - 30, 0), -1)]

    monkeypatch.setattr(stw, "_get", _get)
    monkeypatch.setattr(stw, "OUT", tmp_path)
    monkeypatch.setattr(stw, "PAUSE_S", 0)
    monkeypatch.setattr(stw, "WRITE_EVERY", 2)
    return state


def _saved_ids(tmp_path, sym="SPY"):
    files = sorted((tmp_path / sym).glob("20??-??-??.parquet"))
    return sorted(pd.concat(pd.read_parquet(f) for f in files)["id"]) if files else []


def _ts(i):
    return pd.Timestamp(BASE + 600 * i, unit="s", tz="UTC")


def test_backfill_reaches_start_and_stops(fake, tmp_path):
    start = _ts(50).strftime("%Y-%m-%dT%H:%M:%S")
    stw.sync_symbol("SPY", start)
    assert _saved_ids(tmp_path) == list(range(50, 201))


def test_catch_up_fills_the_gap_exactly(fake, tmp_path):
    stw.sync_symbol("SPY", _ts(150).strftime("%Y-%m-%dT%H:%M:%S"))
    fake["top"] = 330                       # 130 new messages arrive
    stw.sync_symbol("SPY")
    assert _saved_ids(tmp_path) == list(range(150, 331))


def test_interrupted_catch_up_does_not_skip_the_gap(fake, tmp_path):
    stw.sync_symbol("SPY", _ts(190).strftime("%Y-%m-%dT%H:%M:%S"))
    fake["top"] = 400
    fake["fail_after"] = fake["calls"] + 3  # crash partway down the gap
    with pytest.raises(RuntimeError):
        stw.sync_symbol("SPY")
    assert stw._load_state("SPY")["newest_id"] == 200   # not advanced
    fake["fail_after"] = None
    stw.sync_symbol("SPY")
    ids = _saved_ids(tmp_path)
    # The first run keeps its whole top page (ids 171-200), so the floor sits
    # below --start; what matters is no hole anywhere up to the new top.
    assert ids == list(range(ids[0], 401)) and ids[0] <= 190


def test_rows_drop_future_counts_and_hash_users():
    r = stw._row(_msg(3), "SPY", 0)
    assert set(r) == set(stw.COLUMNS) and "likes" not in r
    assert r["sentiment"] == "Bullish" and r["user_hash"] != "u3"


def test_rate_limit_stops_the_run_and_resumes_without_a_gap(fake, tmp_path, monkeypatch):
    stw.sync_symbol("SPY", start="2026-09-30")
    fake["top"] = 400                                   # 200 new messages arrive
    calls = {"n": 0}
    real = stw._get

    def limited(symbol, max_id=None, attempts=6):
        calls["n"] += 1
        if calls["n"] == 4:
            raise stw.RateLimited(symbol)
        return real(symbol, max_id)
    monkeypatch.setattr(stw, "_get", limited)
    out = stw.sync(symbols=("SPY", "VIX"))
    assert out == {} and calls["n"] == 4                # stopped at once, VIX not tried
    monkeypatch.setattr(stw, "_get", real)
    stw.sync_symbol("SPY", start="2026-09-30")
    assert set(range(1, 401)) <= set(_saved_ids(tmp_path))


def test_get_raises_on_429_without_sleeping(monkeypatch):
    import urllib.error
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 429, "Too Many", {}, None)
    monkeypatch.setattr(stw.urllib.request, "urlopen", boom)
    monkeypatch.setattr(stw.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    with pytest.raises(stw.RateLimited):
        stw._get("SPY")
