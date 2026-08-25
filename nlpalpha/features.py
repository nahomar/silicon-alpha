"""Causal price, volume and attention features.

Every column here is computed from information available at or before the
decision point -- the close of day t, which is when the trade is placed. The
target (`fwd_ret`, spanning t to t+1) is never an input.

Two families, deliberately kept apart so the study can attribute any edge:

  price/volume  the standard technical block -- trailing returns, realized
                volatility, volume surprise, trend, plus VIX as a market
                regime control. This is the *baseline to beat*: if text adds
                nothing over these, H1 fails.
  attention     counts derived from the tweet stream without reading a single
                word -- how many tweets, how many distinct users, how unusual
                today's volume of chatter is. These separate "people are
                talking about this stock" from "people are saying good things
                about this stock", which is the H2 distinction.

One honest caveat on timing. Features derived from day t's close (returns,
realized vol, volume) are known only *at* the close we trade. Real execution
means submitting a market-on-close order before the auction, so a strategy
using close(t) to trade at close(t) is marginally optimistic. `feature_lag=1`
shifts the whole price block back a day to measure how much that assumption is
worth; the study reports both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "rvol_5", "rvol_20", "range_pct", "vol_z", "mom_20", "gap",
    "vix", "vix_chg",
]
ATTENTION_FEATURES = [
    "log_n_tweets", "tweet_z", "log_n_users", "users_per_tweet", "rt_share",
]
ALL_FEATURES = PRICE_FEATURES + ATTENTION_FEATURES

_EPS = 1e-9


def build_price_features(prices: pd.DataFrame, vix: pd.DataFrame | None = None,
                         feature_lag: int = 0) -> pd.DataFrame:
    """Per-ticker technical features on the full history.

    Computed on the *whole* price frame -- including the pre-2014 warm-up the
    dataset ships -- so that a 20-day window at the first trading day of the
    study is a real 20-day window rather than a partially-filled one. Merging
    onto the study panel afterwards keeps the warm-up out of the sample while
    still letting it inform the features.
    """
    px = prices.sort_values(["ticker", "date"]).copy()
    g = px.groupby("ticker", sort=False)

    logp = np.log(px["adj_close"].clip(lower=_EPS))
    px["_logp"] = logp
    r1 = g["_logp"].diff()
    px["ret_1"] = r1
    px["ret_2"] = g["_logp"].diff(2)
    px["ret_3"] = g["_logp"].diff(3)
    px["ret_5"] = g["_logp"].diff(5)
    px["ret_10"] = g["_logp"].diff(10)

    px["_r1"] = r1
    px["rvol_5"] = g["_r1"].transform(lambda s: s.rolling(5).std())
    px["rvol_20"] = g["_r1"].transform(lambda s: s.rolling(20).std())

    px["range_pct"] = (px["high"] - px["low"]) / px["close"].clip(lower=_EPS)

    vmean = g["volume"].transform(lambda s: s.rolling(20).mean())
    vstd = g["volume"].transform(lambda s: s.rolling(20).std())
    px["vol_z"] = (px["volume"] - vmean) / (vstd + _EPS)

    ma20 = g["adj_close"].transform(lambda s: s.rolling(20).mean())
    px["mom_20"] = px["adj_close"] / (ma20 + _EPS) - 1.0

    prev_close = g["close"].shift(1)
    px["gap"] = px["open"] / (prev_close + _EPS) - 1.0

    if feature_lag > 0:
        cols = [c for c in PRICE_FEATURES if c not in ("vix", "vix_chg")]
        px[cols] = px.groupby("ticker", sort=False)[cols].shift(feature_lag)

    if vix is not None:
        v = vix.sort_values("date").copy()
        v["vix_chg"] = v["vix"].pct_change()
        px = px.merge(v, on="date", how="left")
        # VIX is a market close too, so lag it exactly like the price block.
        if feature_lag > 0:
            px[["vix", "vix_chg"]] = px[["vix", "vix_chg"]].shift(feature_lag)
    else:
        px["vix"] = np.nan
        px["vix_chg"] = np.nan

    drop = [c for c in ("_logp", "_r1") if c in px.columns]
    return px.drop(columns=drop)


def build_attention_features(panel: pd.DataFrame, tweets: pd.DataFrame,
                             window: int = 20) -> pd.DataFrame:
    """Counting features over the tweet stream -- no text is read.

    `tweet_z` is the one that matters: raw tweet counts are dominated by how
    famous the company is (AAPL always outdraws AEP), which is a constant, not
    a signal. Standardizing against each ticker's own trailing mean turns that
    into *abnormal* attention, which is the quantity the literature actually
    finds predictive of volatility.

    The trailing window uses strictly prior rows, so today's own count never
    enters its own baseline.
    """
    out = panel.sort_values(["ticker", "date"]).copy()

    rt_flags = tweets["tokens"].map(
        lambda toks: bool(toks) and str(toks[0]).lower() == "rt")
    rt_arr = rt_flags.to_numpy()
    out["rt_share"] = [
        float(rt_arr[idx].mean()) if len(idx) else 0.0 for idx in out["tweet_idx"]
    ]

    out["log_n_tweets"] = np.log1p(out["n_tweets"])
    out["log_n_users"] = np.log1p(out["n_users"])
    out["users_per_tweet"] = out["n_users"] / out["n_tweets"].clip(lower=1)

    g = out.groupby("ticker", sort=False)["log_n_tweets"]
    base = g.transform(lambda s: s.shift(1).rolling(window, min_periods=5).mean())
    scale = g.transform(lambda s: s.shift(1).rolling(window, min_periods=5).std())
    out["tweet_z"] = (out["log_n_tweets"] - base) / (scale + _EPS)

    return out


def assemble(panel: pd.DataFrame, prices: pd.DataFrame, tweets: pd.DataFrame,
             vix: pd.DataFrame | None = None,
             feature_lag: int = 0) -> pd.DataFrame:
    """Panel joined to both feature families, rows with any NaN dropped."""
    pf = build_price_features(prices, vix=vix, feature_lag=feature_lag)
    cols = ["ticker", "date"] + PRICE_FEATURES
    merged = panel.merge(pf.loc[:, cols], on=["ticker", "date"], how="left")
    merged = build_attention_features(merged, tweets)
    before = len(merged)
    merged = merged.dropna(subset=ALL_FEATURES + ["fwd_ret"]).reset_index(drop=True)
    merged.attrs["n_dropped_incomplete"] = int(before - len(merged))
    return merged


def standardize(train_x: np.ndarray, *others: np.ndarray):
    """Z-score using train-set moments only.

    Fitting the scaler on the full panel is a subtle and very common leak: the
    test set's mean and variance are future information. The moments come from
    train and are applied unchanged everywhere else.
    """
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd < _EPS, 1.0, sd)
    scaled = [(train_x - mu) / sd]
    scaled.extend((o - mu) / sd for o in others)
    return (*scaled, mu, sd)
