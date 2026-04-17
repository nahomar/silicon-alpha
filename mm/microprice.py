"""Microprice and book-imbalance features.

- microprice = (ask_px * bid_sz + bid_px * ask_sz) / (bid_sz + ask_sz)
  Intuition: the side with more size is less likely to move away, so the
  fair price is tilted toward the smaller side. Stoikov (2018) shows the
  microprice is a better short-horizon fair-value estimator than mid.
- book_imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz), in [-1, 1].
- ofi (order flow imbalance) = Cont/Kukanov/Stoikov's signed change in
  top-of-book size; strong short-term predictor of the next price move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def microprice(bid_px, ask_px, bid_sz, ask_sz):
    total = np.asarray(bid_sz) + np.asarray(ask_sz)
    total = np.where(total == 0, 1.0, total)
    return (np.asarray(ask_px) * np.asarray(bid_sz) + np.asarray(bid_px) * np.asarray(ask_sz)) / total


def book_imbalance(bid_sz, ask_sz):
    bid_sz = np.asarray(bid_sz, dtype=float)
    ask_sz = np.asarray(ask_sz, dtype=float)
    total = bid_sz + ask_sz
    total = np.where(total == 0, 1.0, total)
    return (bid_sz - ask_sz) / total


def ofi(df: pd.DataFrame) -> pd.Series:
    """Order-flow imbalance at the top of book.

    Cont, Kukanov, Stoikov (2014): OFI_n = eN - eS where
      eN (bid side event): Δbid_sz if bid_px unchanged, bid_sz if bid_px↑, -bid_sz_prev if bid_px↓
      eS (ask side event): symmetric with sign flip.
    """
    bp, bs = df["bid_px"].values, df["bid_sz"].values
    ap, as_ = df["ask_px"].values, df["ask_sz"].values
    n = len(df)
    e_bid = np.zeros(n); e_ask = np.zeros(n)
    for i in range(1, n):
        if bp[i] > bp[i - 1]:
            e_bid[i] = bs[i]
        elif bp[i] == bp[i - 1]:
            e_bid[i] = bs[i] - bs[i - 1]
        else:
            e_bid[i] = -bs[i - 1]
        if ap[i] < ap[i - 1]:
            e_ask[i] = as_[i]
        elif ap[i] == ap[i - 1]:
            e_ask[i] = as_[i] - as_[i - 1]
        else:
            e_ask[i] = -as_[i - 1]
    return pd.Series(e_bid - e_ask, index=df.index, name="ofi")


def microprice_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the standard feature bundle from a TOB DataFrame."""
    mid = (df["bid_px"] + df["ask_px"]) / 2.0
    micro = microprice(df["bid_px"], df["ask_px"], df["bid_sz"], df["ask_sz"])
    imb = book_imbalance(df["bid_sz"], df["ask_sz"])
    spread = df["ask_px"] - df["bid_px"]
    rel_spread = spread / mid
    out = pd.DataFrame({
        "mid": mid, "micro": micro, "imbalance": imb,
        "spread": spread, "rel_spread": rel_spread,
        "micro_dev": micro - mid,            # microprice deviation from mid
        "ofi": ofi(df),
    }, index=df.index)
    # Forward microprice return over 1 step — label for ST predictor.
    out["r_1step"] = np.log(out["micro"].shift(-1) / out["micro"])
    return out
