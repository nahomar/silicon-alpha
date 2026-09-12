"""Leakage-safe 0DTE option panel priced on executable touch prices.

Consumes the canonical chain schema the repo's feeds already emit
(`odte/data/datashop_pack.py`, `odte/data/databento_pack.py`):

    quote_datetime, root, expiration, strike, option_type,
    bid, ask, bid_size, ask_size, trade_volume  (+ underlying price)

and produces one row per (contract, bar) carrying the forward return you would
*actually realize*, not the one a mid-to-mid backtest reports.

Why that distinction is the whole point
---------------------------------------
On equities you apply an assumed cost in basis points. On 0DTE options you do
not have to assume anything -- the cost is quoted, sitting in the data, and it
is enormous. A $1.20/$1.35 market is a 12% round trip. A signal that predicts
a 3% move in an option's mid is not a 3% edge; it is a large loss, because
buying at 1.35 and selling at 1.20 loses 11% before the signal is consulted.

So this module computes three returns per bar and the study reports all three:

    ret_mid        mid to mid           -- the flattering number, for contrast
    ret_long_net   buy at ask, sell at bid   -- what a long actually earns
    ret_short_net  sell at bid, buy at ask   -- what a short actually earns

`ret_mid` is deliberately kept, not discarded: the gap between it and the net
returns is the single most useful diagnostic this harness produces, because it
tells you how large a mid-edge has to be before it is worth anything.

The delta trap
--------------
An option's return is mostly delta times the underlying's move. "Predicting
option returns" therefore mostly means predicting SPX, and a model can score
well on options while carrying no options-specific information at all. Two
defenses:

  - `delta_hedged_pnl` removes the first-order directional component, leaving
    the gamma/theta/vega P&L that is actually specific to the contract;
  - `delta_proxy` is exposed as an explicit baseline in `option_edge`, so any
    claimed edge has to beat "just bet on the underlying".

Leakage rule
------------
The decision is made at the close of bar t using only data stamped at or
before t. Entry executes on bar t's touch, exit on bar t+1's touch. Bars are
never forward-filled across the t/t+1 boundary: a stale quote carried into the
exit bar would be a look-ahead, because in reality you would have executed
against whatever was quoted, not against the last price you saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ("quote_datetime", "strike", "option_type", "bid", "ask")
CONTRACT_KEYS = ("root", "expiration", "strike", "option_type")

FEATURES = [
    "moneyness", "minutes_to_expiry", "spread_pct", "size_imbalance",
    "opt_ret_1", "opt_ret_3", "und_ret_1", "und_ret_5", "und_rvol_10",
    "log_volume", "log_mid",
]

_EPS = 1e-12


@dataclass(frozen=True)
class PanelFilters:
    """Every row this drops is reported, never silently discarded."""
    min_bid: float = 0.05          # cannot sell what has no bid
    min_mid: float = 0.10          # penny options are percentage noise
    max_spread_pct: float = 1.00   # wider than 100% of mid is not a market
    min_contracts_per_bar: int = 4


def _normalize(chain: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in chain.columns]
    if missing:
        raise ValueError(
            f"chain missing required column(s) {missing}; expected at least "
            f"{list(REQUIRED)}, got {sorted(chain.columns)}")
    df = chain.copy()
    df["quote_datetime"] = pd.to_datetime(df["quote_datetime"])
    for col in ("bid", "ask", "strike", "bid_size", "ask_size",
                "trade_volume", "underlying_price", "delta"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("root", "expiration"):
        if col not in df.columns:
            df[col] = ""
    df["option_type"] = (df["option_type"].astype(str).str.strip()
                         .str.upper().str[0])
    return df


def _bar_aggregate(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Last quote in each bar per contract -- the touch you would trade on."""
    df = df.sort_values("quote_datetime")
    df["bar"] = df["quote_datetime"].dt.floor(interval)

    agg = {"bid": "last", "ask": "last"}
    for col, how in (("bid_size", "last"), ("ask_size", "last"),
                     ("underlying_price", "last"), ("delta", "last"),
                     ("trade_volume", "sum")):
        if col in df.columns:
            agg[col] = how

    keys = ["bar", *CONTRACT_KEYS]
    out = df.groupby(keys, sort=False).agg(agg).reset_index()
    return out.sort_values(["bar", *CONTRACT_KEYS]).reset_index(drop=True)


def build_option_panel(chain: pd.DataFrame, interval: str = "5min",
                       filters: PanelFilters | None = None) -> pd.DataFrame:
    """Bar panel with executable forward returns and causal features."""
    filters = filters or PanelFilters()
    df = _normalize(chain)
    bars = _bar_aggregate(df, interval)

    if "underlying_price" not in bars.columns:
        raise ValueError(
            "chain has no `underlying_price`; it is required for moneyness, "
            "delta-hedging and the delta_proxy baseline")

    bars["mid"] = (bars["bid"] + bars["ask"]) / 2.0
    bars["spread"] = bars["ask"] - bars["bid"]
    bars["spread_pct"] = bars["spread"] / bars["mid"].clip(lower=_EPS)

    cid = (bars["root"].astype(str) + "|" + bars["expiration"].astype(str)
           + "|" + bars["strike"].astype(str) + "|" + bars["option_type"])
    bars["contract_id"] = cid

    bars = bars.sort_values(["contract_id", "bar"]).reset_index(drop=True)
    g = bars.groupby("contract_id", sort=False)

    # --- exit leg: strictly the NEXT bar, never a forward fill -------------
    nxt_bar = g["bar"].shift(-1)
    nxt_bid = g["bid"].shift(-1)
    nxt_ask = g["ask"].shift(-1)
    nxt_mid = g["mid"].shift(-1)
    nxt_und = g["underlying_price"].shift(-1)

    # A gap larger than one interval means the contract stopped quoting; the
    # "next" row is not the next bar and must not be used as an exit.
    step = pd.Timedelta(interval)
    contiguous = (nxt_bar - bars["bar"]) <= step

    bars["ret_mid"] = nxt_mid / bars["mid"].clip(lower=_EPS) - 1.0
    bars["ret_long_net"] = (nxt_bid - bars["ask"]) / bars["ask"].clip(lower=_EPS)
    bars["ret_short_net"] = (bars["bid"] - nxt_ask) / bars["bid"].clip(lower=_EPS)

    # Dollar P&L per contract, and the delta-hedged version that strips out
    # the first-order underlying move.
    bars["pnl_mid"] = nxt_mid - bars["mid"]
    d_und = nxt_und - bars["underlying_price"]
    bars["und_move"] = d_und
    if "delta" in bars.columns:
        bars["delta_hedged_pnl"] = bars["pnl_mid"] - bars["delta"] * d_und
        bars["delta_proxy"] = bars["delta"] * d_und
    else:
        bars["delta_hedged_pnl"] = np.nan
        bars["delta_proxy"] = np.nan

    # --- causal features: everything below uses bar t or earlier ----------
    bars["log_mid"] = np.log(bars["mid"].clip(lower=_EPS))
    # Regrouped: `g` above predates log_mid, so it cannot serve these.
    gm = bars.groupby("contract_id", sort=False)["log_mid"]
    bars["opt_ret_1"] = gm.diff(1)
    bars["opt_ret_3"] = gm.diff(3)

    if "bid_size" in bars.columns and "ask_size" in bars.columns:
        tot = (bars["bid_size"] + bars["ask_size"]).clip(lower=_EPS)
        bars["size_imbalance"] = (bars["bid_size"] - bars["ask_size"]) / tot
    else:
        bars["size_imbalance"] = 0.0

    bars["log_volume"] = np.log1p(bars.get("trade_volume", pd.Series(0.0,
                                                                    index=bars.index)))
    bars["moneyness"] = np.log(bars["strike"].clip(lower=_EPS)
                               / bars["underlying_price"].clip(lower=_EPS))

    # Underlying series is common to all contracts; build it once per bar.
    und = (bars.groupby("bar", sort=True)["underlying_price"].last()
           .rename("S").to_frame())
    und["logS"] = np.log(und["S"].clip(lower=_EPS))
    und["und_ret_1"] = und["logS"].diff(1)
    und["und_ret_5"] = und["logS"].diff(5)
    und["und_rvol_10"] = und["und_ret_1"].rolling(10).std()
    bars = bars.merge(und[["und_ret_1", "und_ret_5", "und_rvol_10"]],
                      left_on="bar", right_index=True, how="left")

    if "minutes_to_expiry" not in bars.columns:
        exp = pd.to_datetime(bars["expiration"], errors="coerce")
        bars["minutes_to_expiry"] = np.where(
            exp.notna(), (exp - bars["bar"]).dt.total_seconds() / 60.0, np.nan)

    # --- filters, each counted -------------------------------------------
    n0 = len(bars)
    reasons: dict[str, int] = {}

    def _drop(mask, name):
        nonlocal bars
        bad = int((~mask).sum())
        if bad:
            reasons[name] = bad
        bars = bars.loc[mask]

    _drop(contiguous.reindex(bars.index).fillna(False), "no_contiguous_next_bar")
    _drop(bars["bid"] >= filters.min_bid, "bid_below_min")
    _drop(bars["ask"] >= bars["bid"], "crossed_quote")
    _drop(bars["mid"] >= filters.min_mid, "mid_below_min")
    _drop(bars["spread_pct"] <= filters.max_spread_pct, "spread_too_wide")
    _drop(bars["ret_long_net"].notna() & bars["ret_short_net"].notna(),
          "missing_exit_price")

    counts = bars.groupby("bar")["contract_id"].transform("size")
    _drop(counts >= filters.min_contracts_per_bar, "thin_bar")

    bars = bars.reset_index(drop=True)
    bars.attrs["n_rows_in"] = int(n0)
    bars.attrs["n_dropped"] = int(n0 - len(bars))
    bars.attrs["drop_reasons"] = reasons
    bars.attrs["interval"] = interval
    return bars


def spread_cost_summary(panel: pd.DataFrame) -> dict:
    """How much a round trip costs, in the units a signal has to overcome.

    `mean_round_trip_pct` is the headline: it is the return a perfectly
    accurate mid-forecast must exceed before a long position makes a cent.
    """
    if panel.empty:
        return {"n": 0}
    rt = panel["spread"] / panel["mid"].clip(lower=_EPS)
    return {
        "n": int(len(panel)),
        "median_spread_pct": float(panel["spread_pct"].median()),
        "mean_round_trip_pct": float(rt.mean()),
        "p90_spread_pct": float(panel["spread_pct"].quantile(0.90)),
        "median_mid": float(panel["mid"].median()),
        "median_abs_ret_mid": float(panel["ret_mid"].abs().median()),
        "ratio_spread_to_move": float(
            panel["spread_pct"].median()
            / max(panel["ret_mid"].abs().median(), _EPS)),
    }
