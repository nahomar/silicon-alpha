import numpy as np
import pandas as pd
import pytest

from kalshi import backtest as kb


def _candle(end, bid, ask, vol=10.0, oi=100.0, live=False):
    if live:
        return {"end_period_ts": end, "yes_bid": {"close_dollars": str(bid)},
                "yes_ask": {"close_dollars": str(ask)}, "volume_fp": str(vol), "open_interest_fp": str(oi)}
    return {"end_period_ts": end, "yes_bid": {"close": str(bid)}, "yes_ask": {"close": str(ask)},
            "volume": str(vol), "open_interest": str(oi)}


# --- no look-ahead ----------------------------------------------------------

def test_decision_quote_never_uses_a_candle_after_close_minus_24h():
    close = 1_750_000_000 - (1_750_000_000 % 3600)          # on the hour
    d = close - 24 * 3600
    cs = [_candle(d - 7200, 0.10, 0.12), _candle(d - 3600, 0.11, 0.13), _candle(d, 0.12, 0.14),
          _candle(d + 3600, 0.90, 0.95), _candle(close, 0.99, 0.995)]   # later candles must be ignored
    q = kb.decision_quote(cs, d)
    assert q["end"] == d and q["end"] <= close - 24 * 3600
    assert (q["bid"], q["ask"]) == (0.12, 0.14)
    assert q["stale_s"] == 0


def test_close_off_the_hour_uses_the_hour_ending_before_d():
    close = 1_750_000_000 - (1_750_000_000 % 3600) + 1799   # 29:59 past the hour
    d = close - 24 * 3600
    hour_end_before = d - 1799
    cs = [_candle(hour_end_before, 0.20, 0.22), _candle(hour_end_before + 3600, 0.50, 0.52)]
    q = kb.decision_quote(cs, d)
    assert q["end"] == hour_end_before and q["end"] <= d
    assert q["bid"] == 0.20


def test_sparse_candles_carry_the_last_quote_but_not_beyond_24h():
    d = 1_750_000_000
    assert kb.decision_quote([_candle(d - 5 * 3600, 0.3, 0.4)], d)["stale_s"] == 5 * 3600
    assert kb.decision_quote([_candle(d - 24 * 3600 - 1, 0.3, 0.4)], d) is None
    assert kb.decision_quote([_candle(d + 1, 0.3, 0.4)], d) is None


def test_fetch_window_ends_at_d():
    close = 1_750_000_000
    start, end = kb.candle_window(close)
    assert end == close - 24 * 3600 and start == end - 24 * 3600


def test_volume_before_d_sums_only_the_24_hours_up_to_d():
    d = 1_750_000_000
    cs = [_candle(d - 24 * 3600, 0.1, 0.2, vol=1000), _candle(d - 3600, 0.1, 0.2, vol=5),
          _candle(d, 0.1, 0.2, vol=7), _candle(d + 3600, 0.1, 0.2, vol=99)]
    assert kb.decision_quote(cs, d)["vol24"] == 12


def test_live_and_historical_candle_formats_parse_the_same():
    a = kb.parse_candle(_candle(1, 0.05, 0.07))
    b = kb.parse_candle(_candle(1, 0.05, 0.07, live=True))
    assert a == b


# --- quotes and signals -------------------------------------------------------

def test_quote_validity_and_bands():
    assert kb.valid_quote(0.04, 0.06)
    assert not kb.valid_quote(None, 0.06)
    assert not kb.valid_quote(0.0, 0.06)
    assert not kb.valid_quote(0.06, 0.06)
    assert not kb.valid_quote(0.95, 1.0)
    assert kb.signal(kb.mid(0.02, 0.04)) == "K1"           # 0.03 is inside (float-safe)
    assert kb.signal(kb.mid(0.14, 0.16)) == "K1"           # 0.15 is inside
    assert kb.signal(0.1501) is None
    assert kb.signal(kb.mid(0.84, 0.86)) == "K2"
    assert kb.signal(kb.mid(0.96, 0.98)) == "K2"
    assert kb.signal(0.5) is None and kb.signal(0.02) is None and kb.signal(0.98) is None


def test_entry_prices_taker_and_maker():
    assert kb.entry("K1", 0.08, 0.10) == ("no", 0.92)                 # no_ask = 1 - yes_bid
    assert kb.entry("K1", 0.08, 0.10, maker=True) == ("no", 0.90)     # no_bid = 1 - yes_ask
    assert kb.entry("K2", 0.90, 0.92) == ("yes", 0.92)
    assert kb.entry("K2", 0.90, 0.92, maker=True) == ("yes", 0.90)


# --- fees ---------------------------------------------------------------------

def test_taker_fee_matches_the_published_formula():
    # 0.07 x 0.50 x 0.50 = 0.0175 (CFTC filing 2026-10-14)
    assert kb.taker_fee(0.50) == pytest.approx(0.0175)
    # 0.07 x 0.92 x 0.08 = 0.005152 -> rounded up to $0.0001
    assert kb.taker_fee(0.92) == pytest.approx(0.0052)
    # exact grid value is not bumped up by float noise: 0.07 x 0.9 x 0.1 = 0.0063
    assert kb.taker_fee(0.90) == pytest.approx(0.0063)
    # symmetric in P and 1 - P
    assert kb.taker_fee(0.08) == kb.taker_fee(0.92)
    # series multiplier
    assert kb.taker_fee(0.50, multiplier=0.5) == pytest.approx(0.0088)   # 0.00875 -> 0.0088
    assert kb.taker_fee(0.50, multiplier=0.0) == 0.0


def test_conservative_cent_rounding_on_one_contract():
    assert kb.taker_fee(0.92, step=kb.FEE_STEP_CONSERVATIVE) == pytest.approx(0.01)
    assert kb.taker_fee(0.50, step=kb.FEE_STEP_CONSERVATIVE) == pytest.approx(0.02)


def test_maker_fee_by_fee_type():
    assert kb.maker_fee(0.50, "quadratic") == 0.0
    assert kb.maker_fee(0.50, "quadratic_with_maker_fees") == pytest.approx(0.0044)   # 0.004375
    assert kb.maker_fee(0.50, "quadratic_with_combo_maker_fees") == pytest.approx(0.0088)


# --- payoff and return --------------------------------------------------------

def test_payoff_and_return_on_capital():
    assert kb.payoff("no", "no") == 1.0 and kb.payoff("no", "yes") == 0.0
    assert kb.payoff("yes", "yes") == 1.0 and kb.payoff("yes", "no") == 0.0
    with pytest.raises(ValueError):
        kb.payoff("no", "void")
    # K1: buy NO at 0.92 with fee 0.0052; NO wins -> (1 - 0.9252) / 0.9252
    assert kb.roc(1.0, 0.92, 0.0052) == pytest.approx(0.0748 / 0.9252)
    # loses everything when YES happens
    assert kb.roc(0.0, 0.92, 0.0052) == pytest.approx(-1.0)


# --- universe and sampling -----------------------------------------------------

def _markets():
    base = dict(market_type="binary", status="finalized", result="no", volume_fp="5000.00",
                open_time="2024-03-01T00:00:00Z", close_time="2024-03-10T00:00:00Z",
                event_ticker="E1", ticker="A")
    rows = [base,
            dict(base, ticker="B", result="scalar"),
            dict(base, ticker="C", result=""),
            dict(base, ticker="D", volume_fp="999.99"),
            dict(base, ticker="E", open_time="2024-03-08T00:00:01Z"),            # < 48 h open
            dict(base, ticker="F", close_time="2023-12-31T23:59:59Z", open_time="2023-12-01T00:00:00Z"),
            dict(base, ticker="G", close_time="2026-11-01T00:00:00Z", open_time="2026-10-01T00:00:00Z"),
            dict(base, ticker="H", market_type="scalar"),
            dict(base, ticker="I", status="determined"),
            dict(base, ticker="J", open_time="2024-03-08T00:00:00Z", result="yes")]  # exactly 48 h
    return pd.DataFrame(rows)


def test_universe_filters():
    u, counts = kb.universe(_markets())
    assert sorted(u["ticker"]) == ["A", "J"]
    assert counts["qualifying"] == 2 and counts["listed"] == 10


def test_stratified_sample_is_reproducible_and_proportional():
    rng = np.random.default_rng(1)
    n = 5000
    close = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    df = pd.DataFrame({"ticker": [f"T{i:05d}" for i in range(n)], "close": close})
    a = kb.stratified_sample(df, n_max=1000, seed=0)
    b = kb.stratified_sample(df.sample(frac=1.0, random_state=3), n_max=1000, seed=0)
    assert len(a) == 1000 and a["ticker"].is_unique
    assert list(a["ticker"]) == list(b["ticker"])                       # input order does not matter
    share_pop = df["close"].dt.month.value_counts(normalize=True).sort_index()
    share_smp = a["close"].dt.month.value_counts(normalize=True).sort_index()
    assert (share_pop - share_smp).abs().max() < 0.002
    small = kb.stratified_sample(df.head(10), n_max=1000)
    assert len(small) == 10


# --- bootstrap ------------------------------------------------------------------

def test_cluster_bootstrap_resamples_whole_events():
    # two events, all-or-nothing outcomes inside each: the CI must be wide
    r = np.array([1.0] * 50 + [-1.0] * 50)
    cl = np.array(["A"] * 50 + ["B"] * 50)
    b = kb.cluster_bootstrap(r, cl, n=2000)
    assert b["clusters"] == 2 and b["mean"] == 0.0
    assert b["ci"][0] <= -0.99 and b["ci"][1] >= 0.99
    # independent markets: CI close to the iid one
    rng = np.random.default_rng(0)
    r2 = rng.normal(0.05, 0.2, 4000)
    b2 = kb.cluster_bootstrap(r2, np.arange(4000), n=2000)
    assert b2["ci"][0] > 0.03 and b2["ci"][1] < 0.07
    # gate CI is wider than the 95% one
    assert b2["ci"][0] <= b2["ci95"][0] and b2["ci"][1] >= b2["ci95"][1]


def test_bootstrap_is_reproducible():
    r = np.random.default_rng(5).normal(size=300)
    cl = np.repeat(np.arange(100), 3)
    assert kb.cluster_bootstrap(r, cl, n=500) == kb.cluster_bootstrap(r, cl, n=500)


# --- end to end on a tiny synthetic set -------------------------------------------

def test_build_trades_end_to_end():
    close = pd.Timestamp("2025-08-01T00:00:00Z")
    d = int(close.timestamp()) - 24 * 3600
    sample = pd.DataFrame({"ticker": ["M1", "M2", "M3"], "event_ticker": ["S-1", "S-1", "S-2"],
                           "close": [close] * 3, "result": ["no", "yes", "yes"],
                           "volume": [2000.0] * 3, "open_interest_fp": ["10.00"] * 3,
                           "can_close_early": [False, False, True], "source": ["hist"] * 3})
    events = pd.DataFrame({"event_ticker": ["S-1", "S-2"], "series_ticker": ["S", "S"],
                           "category": ["Weather", "Weather"], "title": ["", ""]})
    series = pd.DataFrame({"ticker": ["S"], "fee_type": ["quadratic"], "fee_multiplier": [1.0],
                           "category": ["Climate and Weather"]})
    candles = {"M1": {"route": "hist", "candlesticks": [_candle(d, 0.08, 0.10), _candle(d + 3600, 0.5, 0.6)]},
               "M2": {"route": "hist", "candlesticks": [_candle(d, 0.90, 0.92)]},
               "M3": {"route": "hist", "candlesticks": [_candle(d, 0.40, 0.45)]}}
    q, gaps = kb.build_trades(sample, events, series, candles)
    assert sum(gaps.values()) == 0
    m1 = q.set_index("ticker").loc["M1"]
    assert m1["strategy"] == "K1" and m1["price"] == 0.92 and m1["payoff"] == 1.0
    assert m1["fee"] == pytest.approx(0.0052) and m1["half"] == "H2"
    m2 = q.set_index("ticker").loc["M2"]
    assert m2["strategy"] == "K2" and m2["price"] == 0.92 and m2["payoff"] == 1.0
    assert pd.isna(q.set_index("ticker").loc["M3"]["strategy"])


def test_main_run_on_synthetic_files(tmp_path, monkeypatch):
    import json
    rng = np.random.default_rng(0)
    data, out = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    monkeypatch.setattr(kb, "DATA", data)
    monkeypatch.setattr(kb, "OUT", out)
    n = 400
    close = pd.Timestamp("2024-02-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
    close = close.floor("h")
    p = rng.choice([0.05, 0.10, 0.5, 0.9, 0.95], n)
    res = np.where(rng.random(n) < p, "yes", "no")
    sample = pd.DataFrame({"ticker": [f"S{i % 7}-{i}-M" for i in range(n)],
                           "event_ticker": [f"S{i % 7}-{i // 3}" for i in range(n)],
                           "close": close, "result": res, "volume": 5000.0, "open_interest_fp": "50.00",
                           "can_close_early": rng.random(n) < 0.2, "source": "hist"})
    sample.to_parquet(data / "sample.parquet", index=False)
    pd.DataFrame({"event_ticker": sample["event_ticker"].unique(),
                  "series_ticker": [e.split("-")[0] for e in sample["event_ticker"].unique()],
                  "category": "X", "title": ""}).to_parquet(data / "events.parquet", index=False)
    (data / "series.json").write_text(json.dumps({"series": [
        {"ticker": f"S{k}", "fee_type": "quadratic_with_maker_fees" if k == 0 else "quadratic",
         "fee_multiplier": 1, "category": "Cat%d" % (k % 3)} for k in range(6)]}))   # S6 missing on purpose
    (data / "universe_counts.json").write_text(json.dumps({"qualifying": n}))
    (data / "cutoff.json").write_text(json.dumps({"market_settled_ts": "2026-09-11T00:00:00Z"}))
    with open(data / "candles.jsonl", "w") as fh:
        for m, pi in zip(sample.itertuples(), p):
            d = int(m.close.timestamp()) - 24 * 3600
            cs = [] if m.Index % 50 == 0 else [_candle(d - 3600, round(pi - 0.01, 2), round(pi + 0.01, 2))]
            fh.write(json.dumps({"ticker": m.ticker, "route": "hist", "candlesticks": cs}) + "\n")
    with open(data / "candles_ext.jsonl", "w") as fh:
        for m, pi in zip(sample.itertuples(), p):
            if m.Index % 50 == 0:
                d = int(m.close.timestamp()) - 24 * 3600
                fh.write(json.dumps({"ticker": m.ticker, "route": "hist", "candlesticks": [
                    _candle(d - 5 * 86400, round(pi - 0.01, 2), round(pi + 0.01, 2))]}) + "\n")
    r = kb.main_run()
    assert r["data_gaps"]["no candle in [D-24h, D]"] == 8
    assert r["extended_lookback"]["added_vs_gate_set"] == 8
    assert r["extended_lookback"]["K1"]["full"]["n"] >= r["K1"]["gate"]["full"]["n"]
    assert r["K1"]["gate"]["H1"]["n"] > 0 and r["K2"]["gate"]["H2"]["n"] > 0
    assert r["series_not_in_fee_snapshot"] > 0
    assert set(r["verdict"].values()) <= {"pass", "fail"}
    assert (out / "kalshi.json").exists()


def test_load_universe_merges_tiers_and_drops_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "DATA", tmp_path)
    m = _markets()
    m["mve_collection_ticker"] = None
    combo = m.iloc[[0]].assign(ticker="K", mve_collection_ticker="KXMVECOMBO")
    for folder, rows in (("markets_hist", pd.concat([m.iloc[:6], combo])), ("markets_live", m.iloc[[0, 6, 7, 8, 9]])):
        (tmp_path / folder).mkdir()
        rows.to_parquet(tmp_path / folder / "part-00000.parquet", index=False)
    u, counts = kb.load_universe()
    assert sorted(u["ticker"]) == ["A", "J"]
    assert u.set_index("ticker").loc["A", "source"] == "hist"
    assert counts["qualifying_duplicates_removed"] == 1 and counts["listed"] == 12


def test_limiter_spaces_requests_across_threads():
    import threading, time
    lim = kb.Limiter(0.05)
    stamps = []
    def go():
        for _ in range(5):
            lim.wait()
            stamps.append(time.monotonic())
    th = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in th]; [t.join() for t in th]
    stamps.sort()
    assert min(b - a for a, b in zip(stamps, stamps[1:])) > 0.04


def test_extended_lookback_uses_the_standing_quote_but_never_after_d():
    d = 1_750_000_000
    cs = [_candle(d - 10 * 86400, 0.05, 0.07, vol=3), _candle(d + 60, 0.5, 0.6)]
    assert kb.decision_quote(cs, d) is None                                  # gate rule: 24 h cap
    q = kb.decision_quote(cs, d, lookback=kb.EXT_LOOKBACK_SECONDS)
    assert q["end"] == d - 10 * 86400 and q["bid"] == 0.05 and q["vol24"] == 0
    assert kb.decision_quote(cs, d, lookback=5 * 86400) is None
    assert kb.ext_window(close_ts=d + 86400, open_ts=d - 3 * 86400) == (d - 3 * 86400, d)
    assert kb.ext_window(close_ts=d + 86400, open_ts=0) == (d - 30 * 86400, d)
