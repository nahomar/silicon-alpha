"""Load the StockNet tweet/price corpus into a leakage-safe daily panel.

Source: Xu & Cohen (ACL 2018), "Stock Movement Prediction from Tweets and
Historical Prices" -- 88 large-cap US tickers across 9 sectors, tweets and
Yahoo OHLCV covering 2014-01-01 to 2016-01-01. Cloned from
https://github.com/yumoxu/stocknet-dataset; point `STOCKNET_ROOT` at the
checkout.

The one thing this module exists to get right is *when* each piece of text
became usable. Everything else here is bookkeeping.

The alignment rule
------------------
A trade is executed at the close of trading day t, and the position is held to
the close of t+1. The information set for that trade is every tweet stamped
after the previous trading day's cutoff and at or before day t's cutoff, where
the cutoff is 20:00 UTC.

20:00 UTC is the conservative choice: the US cash close is 21:00 UTC under EST
and 20:00 UTC under EDT, so a 20:00 cutoff is at-or-before the close all year
and never admits a tweet published after the bar we trade on. Tweets landing
after the cutoff (evenings, weekends, holidays) roll forward into the next
trading day's set, which is exactly right -- you could not have traded on them
any earlier.

Getting this wrong is the classic way these studies manufacture alpha: bucket
tweets by calendar day, and every tweet from the evening of day t -- published
*after* the close you claim to trade -- silently enters the feature set. That
variant is available as `cutoff_hour_utc=24` so the damage can be measured
rather than assumed; see `nlpalpha/run_study.py`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Official StockNet temporal splits (Xu & Cohen 2018, section 4.1). Reused
# verbatim so results sit next to published numbers on the same boundaries.
TRAIN_START = pd.Timestamp("2014-01-01")
VAL_START = pd.Timestamp("2015-08-01")
TEST_START = pd.Timestamp("2015-10-01")
TEST_END = pd.Timestamp("2016-01-01")

# Movement labels in the paper use a deadband: moves inside it are discarded as
# noise. Kept for comparability, and measured against full-sample in H4.
DEADBAND_DOWN = -0.005
DEADBAND_UP = 0.0055

CUTOFF_HOUR_UTC = 20


def stocknet_root(root: str | Path | None = None) -> Path:
    """Resolve the dataset checkout from an argument or $STOCKNET_ROOT."""
    if root is not None:
        return Path(root)
    env = os.environ.get("STOCKNET_ROOT")
    if not env:
        raise RuntimeError(
            "StockNet data location unknown. Set STOCKNET_ROOT to a checkout of "
            "https://github.com/yumoxu/stocknet-dataset, or pass root=."
        )
    return Path(env)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def load_prices(root: str | Path | None = None) -> pd.DataFrame:
    """Long-form daily OHLCV for all tickers: one row per (ticker, date).

    Returns are always computed from Adj Close, which carries dividends and
    splits; the raw Close in this feed does not. Using raw Close would print a
    fake -3% on every ex-dividend date and a fake -50% on every split, and a
    directional model would happily learn the dividend calendar.
    """
    base = stocknet_root(root) / "price" / "raw"
    files = sorted(base.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no price CSVs under {base}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = path.stem
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.rename(columns={"adj_close": "adj_close"})
    keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    out = out.loc[:, keep].dropna(subset=["adj_close"])
    out = out[out["adj_close"] > 0]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def cross_check_prices(prices: pd.DataFrame, alt_csv: str | Path,
                       tol: float = 0.01) -> dict:
    """Compare StockNet closes against an independent feed.

    `alt_csv` is the Kaggle S&P-500 daily file (columns date/close/Name), a
    separate collection from a separate vendor. Two feeds agreeing on a price
    is weak evidence; two feeds *disagreeing* is strong evidence something is
    wrong, and that is what this is for. Both closes are unadjusted, so they
    should agree closely wherever they overlap.
    """
    alt = pd.read_csv(alt_csv)
    alt["date"] = pd.to_datetime(alt["date"])
    alt = alt.rename(columns={"Name": "ticker", "close": "alt_close"})
    merged = prices.merge(alt[["ticker", "date", "alt_close"]],
                          on=["ticker", "date"], how="inner")
    if merged.empty:
        return {"n_overlap": 0, "note": "no overlapping (ticker, date) pairs"}

    rel = (merged["close"] - merged["alt_close"]).abs() / merged["alt_close"]
    bad = merged.loc[rel > tol]
    return {
        "n_overlap": int(len(merged)),
        "n_tickers": int(merged["ticker"].nunique()),
        "median_rel_err": float(rel.median()),
        "p99_rel_err": float(rel.quantile(0.99)),
        "n_beyond_tol": int(len(bad)),
        "frac_beyond_tol": float(len(bad) / len(merged)),
        "worst_tickers": bad["ticker"].value_counts().head(5).to_dict(),
    }


# ---------------------------------------------------------------------------
# Tweets
# ---------------------------------------------------------------------------

def load_tweets(root: str | Path | None = None,
                tickers: list[str] | None = None) -> pd.DataFrame:
    """Every preprocessed tweet as (ticker, ts_utc, tokens, user_id).

    The corpus ships pre-tokenized and anonymized -- user handles collapse to
    AT_USER and links to URL -- which removes the two fields most likely to
    leak identity, and means the vocabulary is fixed by the release rather
    than by choices made here.
    """
    base = stocknet_root(root) / "tweet" / "preprocessed"
    if not base.is_dir():
        raise FileNotFoundError(f"no tweet directory under {base}")

    wanted = set(tickers) if tickers else None
    records = []
    for tdir in sorted(base.iterdir()):
        if not tdir.is_dir() or (wanted and tdir.name not in wanted):
            continue
        for day_file in sorted(tdir.iterdir()):
            if not day_file.is_file():
                continue
            with open(day_file, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tokens = obj.get("text")
                    created = obj.get("created_at")
                    if not tokens or not created:
                        continue
                    records.append((tdir.name, created,
                                    obj.get("user_id_str", ""), tokens))

    if not records:
        raise RuntimeError(f"no tweets parsed under {base}")
    df = pd.DataFrame(records, columns=["ticker", "created_at", "user_id", "tokens"])
    # Twitter's format, e.g. "Wed Jan 01 03:59:03 +0000 2014".
    df["ts_utc"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y",
                                  utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).drop(columns=["created_at"])
    return df.sort_values(["ticker", "ts_utc"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def assign_decision_date(ts_utc: pd.Series, trading_days: pd.DatetimeIndex,
                         cutoff_hour_utc: int = CUTOFF_HOUR_UTC) -> pd.Series:
    """Map each tweet to the first trading day whose cutoff it precedes.

    This is the whole leakage guarantee in one function. A tweet is usable for
    the trade at the close of day d only if it was published at or before d's
    cutoff; otherwise it rolls to the next trading day.
    """
    if len(trading_days) == 0:
        return pd.Series(pd.NaT, index=ts_utc.index, dtype="datetime64[ns]")

    cutoffs = (trading_days.tz_localize("UTC")
               + pd.Timedelta(hours=cutoff_hour_utc))
    # side="left": a tweet exactly at the cutoff is still usable that day.
    idx = np.searchsorted(cutoffs.values, ts_utc.values, side="left")
    out = np.full(len(ts_utc), np.datetime64("NaT"), dtype="datetime64[ns]")
    inside = idx < len(trading_days)
    out[inside] = trading_days.values[idx[inside]]
    return pd.Series(out, index=ts_utc.index)


def build_panel(prices: pd.DataFrame, tweets: pd.DataFrame,
                cutoff_hour_utc: int = CUTOFF_HOUR_UTC,
                min_tweets: int = 1) -> pd.DataFrame:
    """Join text to prices on the decision date and attach the forward target.

    One row per (ticker, decision_date) that has both a next-day price and at
    least `min_tweets` tweets in its information set. Columns:

        ticker, date          the decision date; trade at this day's close
        adj_close             close we trade at
        fwd_ret               adj_close(t+1) / adj_close(t) - 1, the target
        up                    1 if fwd_ret > 0 else 0 (full-sample label)
        in_deadband           True if |fwd_ret| falls inside the paper's band
        label_deadband        1/0 outside the band, NaN inside
        tweet_idx             row indices into `tweets` for this cell
        n_tweets, n_users     activity counts

    `fwd_ret` is strictly forward-looking and must never be fed to a model as
    an input; it is the thing being predicted.
    """
    panels = []
    tweets = tweets.reset_index(drop=True)
    by_ticker = {t: g for t, g in tweets.groupby("ticker", sort=False)}

    for ticker, px in prices.groupby("ticker", sort=False):
        px = px.sort_values("date").reset_index(drop=True)
        if len(px) < 3:
            continue
        trading_days = pd.DatetimeIndex(px["date"])

        grp = by_ticker.get(ticker)
        if grp is None or grp.empty:
            continue
        decision = assign_decision_date(grp["ts_utc"], trading_days,
                                        cutoff_hour_utc)
        valid = decision.notna()
        if not valid.any():
            continue

        agg = pd.DataFrame({
            "date": decision[valid].values,
            "row": grp.index[valid].values,
            "user_id": grp["user_id"].values[valid.values],
        })
        grouped = agg.groupby("date")
        cell = pd.DataFrame({
            "date": list(grouped.groups.keys()),
            "tweet_idx": [g["row"].tolist() for _, g in grouped],
            "n_tweets": grouped.size().values,
            "n_users": grouped["user_id"].nunique().values,
        })

        px["fwd_ret"] = px["adj_close"].shift(-1) / px["adj_close"] - 1.0
        merged = px.merge(cell, on="date", how="inner").dropna(subset=["fwd_ret"])
        panels.append(merged)

    if not panels:
        raise RuntimeError("no (ticker, date) cells survived alignment")

    out = pd.concat(panels, ignore_index=True)
    out = out[out["n_tweets"] >= min_tweets]
    out["up"] = (out["fwd_ret"] > 0).astype(int)
    out["in_deadband"] = (out["fwd_ret"] > DEADBAND_DOWN) & (out["fwd_ret"] < DEADBAND_UP)
    out["label_deadband"] = np.where(out["in_deadband"], np.nan,
                                     (out["fwd_ret"] >= DEADBAND_UP).astype(float))
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Split:
    """Row masks for one temporal split. Never shuffled across boundaries."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict:
        return {"train": int(self.train.sum()), "val": int(self.val.sum()),
                "test": int(self.test.sum())}


def make_split(panel: pd.DataFrame) -> Split:
    """Official StockNet date boundaries, applied to the decision date."""
    d = panel["date"]
    return Split(
        train=((d >= TRAIN_START) & (d < VAL_START)).to_numpy(),
        val=((d >= VAL_START) & (d < TEST_START)).to_numpy(),
        test=((d >= TEST_START) & (d < TEST_END)).to_numpy(),
    )


def load_vix(path: str | Path) -> pd.DataFrame:
    """Daily VIX close, used as a market-regime control."""
    vix = pd.read_csv(path)
    vix.columns = [c.strip().lower() for c in vix.columns]
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.loc[:, ["date", "close"]].rename(columns={"close": "vix"})
    return vix.dropna().sort_values("date").reset_index(drop=True)
