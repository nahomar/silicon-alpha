"""Tests for the NLP alpha pipeline, with leakage first.

Every test here runs on synthetic data and needs no corpus, so CI can enforce
the guarantees without a 653MB checkout.

The leakage tests are the important ones. A backtest cannot detect its own
look-ahead: a model fed tomorrow's information reports a beautiful Sharpe and
no error. The only defense is to assert the causal properties directly, which
is what the first section does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlpalpha.backtest import cross_sectional_weights, run_backtest
from nlpalpha.data import (DEADBAND_DOWN, DEADBAND_UP, assign_decision_date,
                           build_panel, make_split)
from nlpalpha.evaluate import matthews_corrcoef, permutation_test_auc
from nlpalpha.features import standardize
from nlpalpha.text_model import SPECIALS, Vocab, mask_tokens, pad_batch
from nlpalpha.train import build_day_tensors, roc_auc

TRADING_DAYS = pd.DatetimeIndex(pd.bdate_range("2014-01-02", periods=40))


def _tweets(times, ticker="AAA"):
    return pd.DataFrame({
        "ticker": ticker,
        "ts_utc": pd.to_datetime(times, utc=True),
        "user_id": [str(i) for i in range(len(times))],
        "tokens": [["good", "news"] for _ in times],
    })


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_tweet_before_cutoff_is_usable_same_day():
    tw = _tweets(["2014-01-02 13:00:00", "2014-01-02 19:59:00"])
    got = assign_decision_date(tw["ts_utc"], TRADING_DAYS, cutoff_hour_utc=20)
    assert (got == pd.Timestamp("2014-01-02")).all()


def test_tweet_after_cutoff_rolls_to_next_trading_day():
    """The single most important assertion in the package.

    A tweet at 21:00 UTC is published after the 20:00 cutoff, hence after the
    close we trade. Bucketing it into day t -- what naive calendar-day
    grouping does -- would let the model read the market's reaction before
    predicting it.
    """
    tw = _tweets(["2014-01-02 20:00:01", "2014-01-02 23:30:00"])
    got = assign_decision_date(tw["ts_utc"], TRADING_DAYS, cutoff_hour_utc=20)
    assert (got == pd.Timestamp("2014-01-03")).all()


def test_weekend_tweets_roll_to_monday():
    # 2014-01-03 is a Friday; 01-06 is the following Monday.
    tw = _tweets(["2014-01-04 12:00:00", "2014-01-05 12:00:00"])
    got = assign_decision_date(tw["ts_utc"], TRADING_DAYS, cutoff_hour_utc=20)
    assert (got == pd.Timestamp("2014-01-06")).all()


def test_decision_date_is_never_before_the_tweet():
    """Exhaustive causality check over a dense grid of timestamps."""
    stamps = pd.date_range("2014-01-02", "2014-02-20", freq="7h", tz="UTC")
    tw = _tweets(stamps)
    got = assign_decision_date(tw["ts_utc"], TRADING_DAYS, cutoff_hour_utc=20)
    ok = got.notna()
    cutoffs = got[ok].dt.tz_localize("UTC") + pd.Timedelta(hours=20)
    assert (tw["ts_utc"][ok] <= cutoffs).all(), "a tweet was used before it existed"


def test_looser_cutoff_admits_strictly_more_information():
    """cutoff=24 is the leaky calendar-day variant; it must differ."""
    tw = _tweets(["2014-01-02 22:00:00"])
    strict = assign_decision_date(tw["ts_utc"], TRADING_DAYS, cutoff_hour_utc=20)
    loose = assign_decision_date(tw["ts_utc"], TRADING_DAYS, cutoff_hour_utc=24)
    assert strict.iloc[0] == pd.Timestamp("2014-01-03")
    assert loose.iloc[0] == pd.Timestamp("2014-01-02")


# ---------------------------------------------------------------------------
# Target and splits
# ---------------------------------------------------------------------------

def _panel_fixture():
    n = 30
    px = pd.DataFrame({
        "ticker": "AAA",
        "date": TRADING_DAYS[:n],
        "open": np.linspace(100, 130, n),
        "high": np.linspace(101, 131, n),
        "low": np.linspace(99, 129, n),
        "close": np.linspace(100, 130, n),
        "adj_close": np.linspace(100, 130, n),
        "volume": np.full(n, 1e6),
    })
    tw = _tweets([d + pd.Timedelta(hours=12) for d in TRADING_DAYS[:n]])
    return build_panel(px, tw)


def test_forward_return_is_next_day_and_never_same_day():
    panel = _panel_fixture()
    adj = dict(zip(panel["date"], panel["adj_close"]))
    dates = sorted(adj)
    for d, nxt in zip(dates, dates[1:]):
        row = panel.loc[panel["date"] == d, "fwd_ret"]
        if len(row):
            assert row.iloc[0] == pytest.approx(adj[nxt] / adj[d] - 1, rel=1e-9)


def test_deadband_labels_match_thresholds():
    panel = _panel_fixture()
    inside = panel["in_deadband"]
    assert (panel.loc[inside, "fwd_ret"] > DEADBAND_DOWN).all()
    assert (panel.loc[inside, "fwd_ret"] < DEADBAND_UP).all()
    assert panel.loc[inside, "label_deadband"].isna().all()
    assert panel.loc[~inside, "label_deadband"].notna().all()


def test_splits_are_disjoint_and_chronological():
    panel = _panel_fixture()
    sp = make_split(panel)
    assert not (sp.train & sp.val).any()
    assert not (sp.val & sp.test).any()
    assert not (sp.train & sp.test).any()
    d = panel["date"].to_numpy()
    if sp.train.any() and sp.val.any():
        assert d[sp.train].max() < d[sp.val].min()
    if sp.val.any() and sp.test.any():
        assert d[sp.val].max() < d[sp.test].min()


# ---------------------------------------------------------------------------
# Fitting hygiene
# ---------------------------------------------------------------------------

def test_standardize_uses_only_training_moments():
    """Test moments must not influence the scaling of anything."""
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, size=(200, 3))
    test = rng.normal(50, 10, size=(80, 3))     # wildly different distribution
    tr_s, te_s, mu, sd = standardize(train, test)
    assert np.allclose(tr_s.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(mu, train.mean(axis=0))
    # If test moments had leaked in, the test block would be centered too.
    assert np.abs(te_s.mean()) > 1.0


def test_vocab_excludes_words_only_seen_after_training():
    train_tokens = [["earnings", "beat"] for _ in range(10)]
    vocab = Vocab.build(train_tokens, min_freq=2)
    assert "earnings" in vocab.stoi
    assert "bankruptcy" not in vocab.stoi
    ids = vocab.encode(["bankruptcy"], max_len=8)
    assert ids[1] == vocab.stoi["<unk>"]


def test_mlm_never_masks_special_tokens():
    vocab = Vocab.build([["alpha", "beta"] * 5 for _ in range(20)], min_freq=1)
    ids, _ = pad_batch([vocab.encode(["alpha", "beta"], 8)] * 32,
                       vocab.pad_id, 8)
    gen = __import__("torch").Generator().manual_seed(0)
    corrupted, labels = mask_tokens(ids, vocab, prob=0.9, generator=gen)
    for sid in (vocab.pad_id, vocab.cls_id):
        assert (labels[ids.eq(sid)] == -100).all()
    assert (corrupted[ids.eq(vocab.pad_id)] == vocab.pad_id).all()


def test_day_tensor_mask_matches_tweet_counts():
    panel = pd.DataFrame({"tweet_idx": [[0], [1, 2, 3], []]})
    emb = np.arange(12, dtype=np.float32).reshape(4, 3)
    x, mask = build_day_tensors(panel, emb, max_tweets=5)
    assert mask.sum(axis=1).tolist() == [1, 3, 0]
    assert np.allclose(x[1, :3], emb[[1, 2, 3]])
    assert np.allclose(x[0, 1:], 0.0)


def test_day_tensor_truncation_keeps_most_recent():
    panel = pd.DataFrame({"tweet_idx": [[0, 1, 2, 3]]})
    emb = np.arange(4, dtype=np.float32).reshape(4, 1)
    x, mask = build_day_tensors(panel, emb, max_tweets=2)
    assert mask[0].tolist() == [True, True]
    assert x[0, :, 0].tolist() == [2.0, 3.0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_roc_auc_matches_known_cases():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(0.0)
    assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5)


def test_mcc_is_zero_for_constant_prediction():
    """The reason MCC is reported: always-up scores >50% accuracy but 0 MCC."""
    y = [1] * 51 + [0] * 49
    assert matthews_corrcoef(y, [1] * 100) == pytest.approx(0.0)


def test_permutation_test_does_not_flag_noise():
    rng = np.random.default_rng(3)
    n = 600
    dates = np.repeat(pd.date_range("2015-10-01", periods=30), 20)
    y = rng.integers(0, 2, n)
    res = permutation_test_auc(y, rng.normal(size=n), dates, n_perm=200, seed=1)
    assert res["p_value"] > 0.05, "pure noise was called significant"
    assert res["null_mean"] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def test_weights_are_dollar_neutral_with_unit_gross():
    for scheme in ("rank", "topk"):
        w = cross_sectional_weights(np.array([5.0, 1.0, 3.0, 2.0, 4.0]),
                                    scheme=scheme)
        assert w.sum() == pytest.approx(0.0, abs=1e-12)
        assert np.abs(w).sum() == pytest.approx(1.0, abs=1e-9)


def test_weights_rank_the_cross_section_correctly():
    w = cross_sectional_weights(np.array([1.0, 5.0, 3.0]), scheme="rank")
    assert w[1] > w[2] > w[0]


def test_backtest_profits_from_a_perfect_signal():
    """A signal equal to the realized return must make money gross."""
    dates = pd.date_range("2015-10-01", periods=25, freq="B")
    rng = np.random.default_rng(0)
    rows = []
    for d in dates:
        for t in [f"T{i}" for i in range(12)]:
            r = float(rng.normal(0, 0.02))
            rows.append({"date": d, "ticker": t, "fwd_ret": r, "signal": r})
    df = pd.DataFrame(rows)
    res = run_backtest(df, cost_bps=0.0)
    assert res["mean_daily_gross"] > 0
    assert res["sharpe_gross"] > 3


def test_backtest_loses_on_an_inverted_signal():
    dates = pd.date_range("2015-10-01", periods=25, freq="B")
    rng = np.random.default_rng(1)
    rows = []
    for d in dates:
        for t in [f"T{i}" for i in range(12)]:
            r = float(rng.normal(0, 0.02))
            rows.append({"date": d, "ticker": t, "fwd_ret": r, "signal": -r})
    res = run_backtest(pd.DataFrame(rows), cost_bps=0.0)
    assert res["mean_daily_gross"] < 0


def test_costs_monotonically_reduce_net_return():
    dates = pd.date_range("2015-10-01", periods=20, freq="B")
    rng = np.random.default_rng(2)
    rows = []
    for d in dates:
        for t in [f"T{i}" for i in range(10)]:
            rows.append({"date": d, "ticker": t,
                         "fwd_ret": float(rng.normal(0, 0.02)),
                         "signal": float(rng.normal())})
    df = pd.DataFrame(rows)
    nets = [run_backtest(df, cost_bps=c)["mean_daily_net"]
            for c in (0, 5, 10, 20)]
    assert all(a > b for a, b in zip(nets, nets[1:])), nets


def test_backtest_requires_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(pd.DataFrame({"date": [], "ticker": []}))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
