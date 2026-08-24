"""Model-free static-arbitrage scan on a 0DTE option chain.

Before any forecasting work is worth doing, the chain has to be arithmetically
consistent with itself. This module asks that question and nothing else: given
a snapshot of quotes, does the chain violate relations that hold for *any*
arbitrage-free market, with no model, no vol surface, and no view on the
underlying?

Two complementary tests, both evaluated on **executable touch prices** (you buy
at the ask and sell at the bid — never at the mid):

  1. Box spreads (pairwise, exact).
     For strikes K1 < K2, the box  +C(K1) -C(K2) +P(K2) -P(K1)  pays exactly
     W = K2 - K1 at expiry regardless of where the underlying lands. So with
     non-negative rates its fair cost lies in (0, W]. Two rate-agnostic
     violations follow:
        cost to BUY   <= 0   -> paid to receive a guaranteed W later
        proceeds SELL >= W   -> received more today than you owe at expiry
     Neither requires knowing the discount factor, which is what makes this
     the strict test. Everything flagged here is a genuine static arbitrage
     up to fees, borrow, and the assumption that the quotes are hittable.

  2. Put-call parity line (chain-wide, localizing).
     Parity says  C(K) - P(K) = D*(F - K)  -- a straight line in K with slope
     -D and intercept D*F. Fitting that line across all strikes recovers the
     chain's own implied discount factor and forward without a model. A strike
     whose executable combo band [c_bid - p_ask, c_ask - p_bid] excludes the
     fitted line disagrees with the rest of the chain by more than its own
     spread.

The two are the same relation at different scales: differencing parity at two
strikes gives (C-P)(K1) - (C-P)(K2) = D*W, which *is* the box. The box scan is
therefore the exact pairwise statement, and the line fit is the many-strike
version that says *which* strike is the odd one out. Read them together: a box
violation names a tradeable pair, a parity residual names a suspect strike.

What the parity fit can and cannot tell you at 0DTE. The forward comes out
sharp -- on the synthetic chain it is recovered to well under a tick. The
implied *rate* does not, and the reason is structural rather than statistical:
discounting displaces the parity line by |1 - D| * |F - K|, which for a 5%
rate over a six-and-a-half-hour session is about 0.015 index points across a
1000-point strike span. That is under a third of one $0.05 tick, so D is
indistinguishable from 1 in any real quote grid, and dividing that
indistinguishable-from-zero error by T ~ 1e-3 to annualize it produces a rate
that is pure amplified noise. The scan therefore reports `rate_identifiable`
alongside `implied_rate`, and on 0DTE input it will normally be False. Read
the forward; treat D as 1.0.

Honest limits -- what a flag here does and does not mean:
  - A box violation is a real static arbitrage *in the quoted snapshot*. It is
    not a filled trade. Stale quotes, size-one touches, locked/crossed books
    around the open, and fees routinely explain what looks like free money.
    Pass `fee_per_leg` to price that in rather than admiring gross edge.
  - A parity-line residual is a *relative-value* finding, not an arbitrage.
    The fitted line is not a traded instrument; trading against it means
    trading that strike's combo versus the strikes that set the line.
  - Absence of flags is the expected, healthy outcome on a liquid chain. The
    scan earns its keep as a data-quality gate: if SPX 0DTE appears to
    misprice its own arithmetic, the feed is far more likely to be wrong than
    the market, and every downstream model inherits that error.

Dependencies: numpy and pandas only. No market data required -- the self-test
builds an arbitrage-free synthetic chain via Black-76 and injects known
violations into it.

Usage:
    PYTHONPATH=. python -m odte.eval.parity_scan --selftest
    PYTHONPATH=. python -m odte.eval.parity_scan --chain path/to/chain.parquet
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "parity_scan"

# A 0DTE session is a fraction of a trading year; 6.5h ~= 1.07e-3 years.
DEFAULT_T = 6.5 / 24.0 / 252.0

# Default pairing budget for the box scan. All-pairs is O(n^2) and dominated by
# far-apart strikes whose quotes are wide and stale; nearby pairs are where a
# real dislocation shows up. Skipped pairs are reported, never silently capped.
DEFAULT_MAX_PAIRS_AHEAD = 8
DEFAULT_MAX_WIDTH = 100.0

REQUIRED_COLUMNS = ("strike", "right", "bid", "ask")


# ---------------------------------------------------------------------------
# Chain normalization
# ---------------------------------------------------------------------------

def _normalize_right(s: pd.Series) -> pd.Series:
    """Map assorted call/put spellings onto 'C' / 'P'."""
    up = s.astype(str).str.strip().str.upper().str[0]
    return up.where(up.isin(["C", "P"]))


def pivot_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Collapse a long-form quote table to one row per strike.

    Input columns: strike, right ('C'/'P'/'call'/'put'), bid, ask.
    Output columns: strike, c_bid, c_ask, p_bid, p_ask -- sorted by strike,
    restricted to strikes carrying a usable two-sided quote on *both* legs.

    Rows are dropped (never repaired) when a quote is non-finite, non-positive
    on the bid, or crossed (ask < bid). A crossed book is a feed artifact; a
    scan that "fixed" it would manufacture the very edge it is looking for.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(
            f"chain is missing required column(s) {missing}; "
            f"expected {list(REQUIRED_COLUMNS)}, got {list(chain.columns)}"
        )

    df = chain.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["right"] = _normalize_right(df["right"])
    for col in ("strike", "bid", "ask"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ok = (
        df["right"].notna()
        & np.isfinite(df["strike"])
        & np.isfinite(df["bid"])
        & np.isfinite(df["ask"])
        & (df["bid"] > 0.0)
        & (df["ask"] >= df["bid"])
    )
    df = df.loc[ok]
    if df.empty:
        return pd.DataFrame(columns=["strike", "c_bid", "c_ask", "p_bid", "p_ask"])

    # Keep the tightest quote if a strike/right appears more than once.
    df = df.assign(_spread=df["ask"] - df["bid"])
    df = (df.sort_values("_spread")
            .drop_duplicates(subset=["strike", "right"], keep="first"))

    wide = df.pivot(index="strike", columns="right", values=["bid", "ask"])
    have_both = {("bid", "C"), ("ask", "C"), ("bid", "P"), ("ask", "P")}
    if not have_both.issubset(set(wide.columns)):
        return pd.DataFrame(columns=["strike", "c_bid", "c_ask", "p_bid", "p_ask"])

    out = pd.DataFrame({
        "c_bid": wide[("bid", "C")],
        "c_ask": wide[("ask", "C")],
        "p_bid": wide[("bid", "P")],
        "p_ask": wide[("ask", "P")],
    }).dropna()
    return out.reset_index().sort_values("strike").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Box spreads -- the strict, rate-agnostic test
# ---------------------------------------------------------------------------

def scan_boxes(chain: pd.DataFrame,
               fee_per_leg: float = 0.0,
               max_pairs_ahead: int = DEFAULT_MAX_PAIRS_AHEAD,
               max_width: float = DEFAULT_MAX_WIDTH,
               T: float = DEFAULT_T) -> pd.DataFrame:
    """Price every candidate box on executable touch prices.

    Returns one row per (K1, K2) pair with:
        width           K2 - K1, the guaranteed expiry payoff
        cost_buy        net debit to establish the long box (4 legs, fees in)
        proceeds_sell   net credit for the short box (4 legs, fees out)
        implied_rate    continuous rate implied by cost_buy over T, or NaN
        arb_cheap       cost_buy <= 0   -> paid to receive `width` later
        arb_rich        proceeds_sell >= width -> credit exceeds what you owe
        edge            dollars of guaranteed profit per box, 0.0 if no arb

    `arb_cheap` / `arb_rich` are the only rate-free claims here. `implied_rate`
    is a diagnostic: a box priced at a 40% financing rate is more likely a
    stale quote than a lending opportunity, but that judgement needs a view on
    plausible rates, so it is reported and never used to flag.
    """
    p = pivot_chain(chain)
    n = len(p)
    if n < 2:
        return _empty_box_frame()

    K = p["strike"].to_numpy(float)
    c_bid = p["c_bid"].to_numpy(float)
    c_ask = p["c_ask"].to_numpy(float)
    p_bid = p["p_bid"].to_numpy(float)
    p_ask = p["p_ask"].to_numpy(float)

    i_idx, j_idx = [], []
    skipped_width = 0
    for i in range(n - 1):
        for j in range(i + 1, min(i + 1 + max_pairs_ahead, n)):
            if K[j] - K[i] > max_width:
                skipped_width += 1
                continue
            i_idx.append(i)
            j_idx.append(j)
    if not i_idx:
        return _empty_box_frame()

    i_arr = np.asarray(i_idx)
    j_arr = np.asarray(j_idx)
    width = K[j_arr] - K[i_arr]

    # Long box: buy C(K1), sell C(K2), buy P(K2), sell P(K1). Four legs of fee
    # are paid on the way in and again on the way out.
    fees = 4.0 * float(fee_per_leg)
    cost_buy = (c_ask[i_arr] - c_bid[j_arr]
                + p_ask[j_arr] - p_bid[i_arr]) + fees
    proceeds_sell = (c_bid[i_arr] - c_ask[j_arr]
                     + p_bid[j_arr] - p_ask[i_arr]) - fees

    arb_cheap = cost_buy <= 0.0
    arb_rich = proceeds_sell >= width

    # Guaranteed profit per box, in dollars of index points.
    edge = np.zeros_like(width)
    edge[arb_cheap] = (width - cost_buy)[arb_cheap]
    edge[arb_rich] = (proceeds_sell - width)[arb_rich]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cost_buy > 0.0, width / cost_buy, np.nan)
        implied_rate = np.where(np.isfinite(ratio) & (ratio > 0.0),
                                np.log(ratio) / max(T, 1e-12), np.nan)

    out = pd.DataFrame({
        "k1": K[i_arr],
        "k2": K[j_arr],
        "width": width,
        "cost_buy": cost_buy,
        "proceeds_sell": proceeds_sell,
        "implied_rate": implied_rate,
        "arb_cheap": arb_cheap,
        "arb_rich": arb_rich,
        "edge": edge,
    })
    out.attrs["skipped_pairs_over_max_width"] = int(skipped_width)
    return out.sort_values("edge", ascending=False).reset_index(drop=True)


def _empty_box_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "k1": pd.Series(dtype=float), "k2": pd.Series(dtype=float),
        "width": pd.Series(dtype=float), "cost_buy": pd.Series(dtype=float),
        "proceeds_sell": pd.Series(dtype=float),
        "implied_rate": pd.Series(dtype=float),
        "arb_cheap": pd.Series(dtype=bool), "arb_rich": pd.Series(dtype=bool),
        "edge": pd.Series(dtype=float),
    })
    frame.attrs["skipped_pairs_over_max_width"] = 0
    return frame


# ---------------------------------------------------------------------------
# 2. Put-call parity line -- the chain-wide test
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParityFit:
    """Discount factor and forward implied by the chain's own parity line."""
    intercept: float          # D * F
    slope: float              # -D
    discount: float           # D
    forward: float            # F
    rate: float               # continuous r = -ln(D) / T
    n_used: int               # strikes surviving the trim
    n_trimmed: int            # strikes down-weighted as outliers
    rmse: float               # residual RMSE on the kept strikes, index points
    ok: bool                  # False when the fit is degenerate
    # Identifiability of the *rate*, which at 0DTE is a separate question from
    # the quality of the fit. See `rate_identifiable` below.
    discount_effect: float = float("nan")   # |1-D| * max|F-K|, index points
    quote_resolution: float = float("nan")  # median half-width of combo bands
    rate_identifiable: bool = False

    def value_at(self, strike):
        """Fitted fair value of the C - P combo at `strike`."""
        return self.intercept + self.slope * np.asarray(strike, dtype=float)


def _ols_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares intercept, slope for y = a + b*x. Raises if degenerate."""
    mx = float(x.mean())
    my = float(y.mean())
    sxx = float(((x - mx) ** 2).sum())
    if sxx <= 0.0:
        raise ValueError("cannot fit a line through a single distinct strike")
    sxy = float(((x - mx) * (y - my)).sum())
    slope = sxy / sxx
    return my - slope * mx, slope


def fit_parity_line(chain: pd.DataFrame,
                    T: float = DEFAULT_T,
                    trim: float = 0.10,
                    n_iter: int = 3,
                    min_leg_price: float = 0.10,
                    max_abs_rate: float = 0.25) -> ParityFit:
    """Recover (D, F) from C - P = D*(F - K) by trimmed least squares.

    Two robustness measures, both of which matter on a real chain:

      - Legs quoted at the exchange floor are excluded. A strike whose call or
        put sits at the minimum tick carries no information about the forward:
        its true value is far below one tick, so the quoted mid is a rounding
        artifact biased *away* from zero. That bias has opposite sign at the
        two ends of the chain, so it tilts the slope directly. On the synthetic
        chain, dropping floored legs moves the recovered discount factor from
        4.8e-4 too low to 3e-5 of truth and pins the forward exactly. The
        filter is abandoned if it would leave fewer than four strikes.
      - A single badly-quoted strike would drag a plain OLS line toward itself
        and then mask its own residual, so the fit is iterated: fit, measure
        absolute residuals, drop the worst `trim` fraction, refit.

    The fit deliberately runs on *mids* -- it is estimating the chain's central
    tendency, not claiming a trade. Executable prices enter in `scan_parity`,
    where the fitted line is compared against each strike's touch band.

    Note which output you may trust. The forward is well identified; the rate
    generally is not (see `rate_identifiable`).
    """
    p = pivot_chain(chain)
    if len(p) < 3:
        return ParityFit(np.nan, np.nan, np.nan, np.nan, np.nan,
                         len(p), 0, np.nan, False)

    valuable = (p["c_bid"] >= min_leg_price) & (p["p_bid"] >= min_leg_price)
    if int(valuable.sum()) >= 4:
        p = p.loc[valuable].reset_index(drop=True)

    K = p["strike"].to_numpy(float)
    y = ((p["c_bid"] + p["c_ask"]) / 2.0
         - (p["p_bid"] + p["p_ask"]) / 2.0).to_numpy(float)

    # Quote resolution: how far the line must move before the chain could see
    # it. Used below to decide whether the slope is worth estimating at all.
    half_band = ((p["c_ask"] - p["c_bid"]) + (p["p_ask"] - p["p_bid"])) / 2.0
    quote_resolution = float(np.median(half_band.to_numpy(float)))

    # Slope-free forward. Parity with D = 1 says y + K = F at every strike, and
    # D is within 1e-4 of 1 for any plausible short-dated rate. The median
    # makes this immune to a handful of corrupted strikes.
    forward_robust = float(np.median(y + K))
    span = float(np.abs(forward_robust - K).max())

    # Is the slope identifiable *in principle*? Bound D by a maximum plausible
    # rate; if even the most extreme admissible D displaces the line less than
    # the chain can resolve, then no estimator can recover it and fitting the
    # slope only imports noise. Note this asks what the data could resolve --
    # not how large the fitted slope happens to be. A wildly wrong slope also
    # produces a large displacement, so testing the estimate would let exactly
    # the broken fits through that the check exists to catch.
    d_lo = math.exp(-abs(max_abs_rate) * max(T, 0.0))
    d_hi = math.exp(abs(max_abs_rate) * max(T, 0.0))
    max_plausible_effect = max(abs(1.0 - d_lo), abs(1.0 - d_hi)) * span
    identifiable = bool(max_plausible_effect > quote_resolution)

    # --- Pass 1: free two-parameter fit, trimmed. ---------------------------
    keep = np.ones(len(K), dtype=bool)
    intercept = slope = np.nan
    for _ in range(max(1, n_iter)):
        try:
            intercept, slope = _ols_line(K[keep], y[keep])
        except ValueError:
            return ParityFit(np.nan, np.nan, np.nan, np.nan, np.nan,
                             int(keep.sum()), 0, np.nan, False)
        resid = np.abs(y - (intercept + slope * K))
        if trim <= 0.0:
            break
        # Cutoff is taken over *all* strikes each pass, not over the survivors
        # of the previous pass -- otherwise the kept set shrinks geometrically
        # and three iterations can strip a six-strike chain down to three.
        cutoff = float(np.quantile(resid, 1.0 - trim))
        nxt = resid <= max(cutoff, 1e-12)
        # Never trim below three points -- two would fit any line exactly.
        if nxt.sum() < 3:
            break
        keep = nxt

    discount = -slope
    credible = bool(np.isfinite(discount) and d_lo <= discount <= d_hi)

    # --- Pass 2: constrain the slope when it cannot be trusted. -------------
    # Either the chain cannot resolve the slope, or the fitted one implies a
    # rate outside anything a real market would price. In both cases stop
    # estimating it: pin D = 1 and read off the robust forward alone. At 0DTE
    # this is the normal path, and it is the honest one -- the alternative is
    # a rate that is quote noise amplified by 1/T.
    if identifiable and credible:
        forward = intercept / discount
        rate = -math.log(discount) / max(T, 1e-12)
        rate_identifiable = True
        discount_effect = abs(1.0 - discount) * span
    else:
        forward = forward_robust
        discount, slope, intercept = 1.0, -1.0, forward
        rate, rate_identifiable = 0.0, False
        keep = np.ones(len(K), dtype=bool)
        discount_effect = 0.0

    resid_keep = y[keep] - (intercept + slope * K[keep])
    rmse = float(np.sqrt(np.mean(resid_keep ** 2)))
    return ParityFit(float(intercept), float(slope), float(discount),
                     float(forward), float(rate), int(keep.sum()),
                     int((~keep).sum()), rmse, True,
                     float(discount_effect), quote_resolution,
                     bool(rate_identifiable))


def scan_parity(chain: pd.DataFrame,
                fit: ParityFit | None = None,
                fee_per_leg: float = 0.0,
                T: float = DEFAULT_T) -> pd.DataFrame:
    """Flag strikes whose executable combo band excludes the fitted line.

    For each strike the C - P combo can be bought for at most
    `c_ask - p_bid` and sold for at least `c_bid - p_ask`. If the fitted fair
    value sits above the buy price the combo is cheap versus the rest of the
    chain; below the sell price, rich. Anything inside the band is consistent
    with the chain to within its own spread and is not reported as a
    violation.
    """
    p = pivot_chain(chain)
    if p.empty:
        return _empty_parity_frame()
    if fit is None:
        fit = fit_parity_line(chain, T=T)
    if not fit.ok:
        return _empty_parity_frame()

    K = p["strike"].to_numpy(float)
    fees = 2.0 * float(fee_per_leg)
    band_hi = (p["c_ask"] - p["p_bid"]).to_numpy(float) + fees   # cost to buy
    band_lo = (p["c_bid"] - p["p_ask"]).to_numpy(float) - fees   # sell credit
    line = fit.value_at(K)

    cheap = line > band_hi          # buy the combo below its fitted value
    rich = line < band_lo           # sell the combo above its fitted value
    edge = np.where(cheap, line - band_hi,
                    np.where(rich, band_lo - line, 0.0))

    return pd.DataFrame({
        "strike": K,
        "combo_bid": band_lo,
        "combo_ask": band_hi,
        "fitted": line,
        "mid_residual": ((p["c_bid"] + p["c_ask"]) / 2.0
                         - (p["p_bid"] + p["p_ask"]) / 2.0).to_numpy(float) - line,
        "violation": cheap | rich,
        "side": np.where(cheap, "cheap", np.where(rich, "rich", "")),
        "edge": edge,
    }).sort_values("edge", ascending=False).reset_index(drop=True)


def _empty_parity_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "strike": pd.Series(dtype=float), "combo_bid": pd.Series(dtype=float),
        "combo_ask": pd.Series(dtype=float), "fitted": pd.Series(dtype=float),
        "mid_residual": pd.Series(dtype=float),
        "violation": pd.Series(dtype=bool), "side": pd.Series(dtype=object),
        "edge": pd.Series(dtype=float),
    })


# ---------------------------------------------------------------------------
# Top-level scan
# ---------------------------------------------------------------------------

def run_scan(chain: pd.DataFrame,
             T: float = DEFAULT_T,
             fee_per_leg: float = 0.0,
             max_pairs_ahead: int = DEFAULT_MAX_PAIRS_AHEAD,
             max_width: float = DEFAULT_MAX_WIDTH,
             write: bool = False) -> dict:
    """Run both tests and return a JSON-serializable summary."""
    p = pivot_chain(chain)
    boxes = scan_boxes(chain, fee_per_leg=fee_per_leg,
                       max_pairs_ahead=max_pairs_ahead,
                       max_width=max_width, T=T)
    fit = fit_parity_line(chain, T=T)
    parity = scan_parity(chain, fit=fit, fee_per_leg=fee_per_leg, T=T)

    box_arb = boxes[boxes["arb_cheap"] | boxes["arb_rich"]] if len(boxes) else boxes
    par_arb = parity[parity["violation"]] if len(parity) else parity

    n_box_arb = int(len(box_arb))
    n_par_arb = int(len(par_arb))
    clean = (n_box_arb == 0 and n_par_arb == 0)

    summary = {
        "n_quotes_in": int(len(chain)),
        "n_strikes_in": int(chain["strike"].nunique()) if len(chain) else 0,
        "n_strikes_usable": int(len(p)),
        "n_strikes_dropped": (int(chain["strike"].nunique()) - int(len(p))
                              if len(chain) else 0),
        "T_years": float(T),
        "fee_per_leg": float(fee_per_leg),
        "boxes": {
            "n_pairs": int(len(boxes)),
            "n_skipped_over_max_width":
                int(boxes.attrs.get("skipped_pairs_over_max_width", 0)),
            "n_arb": n_box_arb,
            "n_cheap": int(boxes["arb_cheap"].sum()) if len(boxes) else 0,
            "n_rich": int(boxes["arb_rich"].sum()) if len(boxes) else 0,
            "max_edge": float(box_arb["edge"].max()) if n_box_arb else 0.0,
            "median_implied_rate":
                (float(np.nanmedian(boxes["implied_rate"]))
                 if len(boxes) and np.isfinite(boxes["implied_rate"]).any()
                 else None),
        },
        "parity": {
            "ok": bool(fit.ok),
            "discount": _jsonable(fit.discount),
            "forward": _jsonable(fit.forward),
            "implied_rate": _jsonable(fit.rate),
            "rate_identifiable": bool(fit.rate_identifiable),
            "discount_effect_pts": _jsonable(fit.discount_effect),
            "quote_resolution_pts": _jsonable(fit.quote_resolution),
            "rmse": _jsonable(fit.rmse),
            "n_used": int(fit.n_used),
            "n_trimmed": int(fit.n_trimmed),
            "n_violations": n_par_arb,
            "max_edge": float(par_arb["edge"].max()) if n_par_arb else 0.0,
            "violating_strikes": [float(s) for s in par_arb["strike"].tolist()[:20]],
        },
        "verdict": ("CLEAN -- chain is internally consistent" if clean else
                    f"INCONSISTENT -- {n_box_arb} box arb(s), "
                    f"{n_par_arb} parity violation(s)"),
        "note": ("Box flags are static arbitrage in the quoted snapshot, not "
                 "filled trades; parity flags are relative-value, not "
                 "arbitrage. On a liquid chain CLEAN is the expected result "
                 "and a flag usually indicts the feed, not the market."),
    }
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        boxes.to_csv(OUT / "boxes.csv", index=False)
        parity.to_csv(OUT / "parity.csv", index=False)
    return summary


def _jsonable(x):
    """None for non-finite floats so the summary round-trips through JSON."""
    xf = float(x)
    return xf if np.isfinite(xf) else None


# ---------------------------------------------------------------------------
# Synthetic arbitrage-free chain (for the self-test and the unit tests)
# ---------------------------------------------------------------------------

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erf -- avoids a scipy dependency."""
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def black76(F: float, K: np.ndarray, T: float, sigma: np.ndarray,
            D: float) -> tuple[np.ndarray, np.ndarray]:
    """Black-76 call/put on a forward. Satisfies C - P = D*(F - K) exactly."""
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrt_t = math.sqrt(max(T, 1e-12))
    vol = np.maximum(sigma, 1e-8) * sqrt_t
    d1 = (np.log(F / K) + 0.5 * vol ** 2) / vol
    d2 = d1 - vol
    call = D * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    put = D * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))
    return call, put


def synthetic_chain(forward: float = 5000.0,
                    discount: float = 0.99997,
                    T: float = DEFAULT_T,
                    n_strikes: int = 41,
                    strike_step: float = 25.0,
                    base_vol: float = 0.18,
                    skew: float = 0.35,
                    half_spread: float = 0.25,
                    tick: float = 0.05) -> pd.DataFrame:
    """Build an arbitrage-free long-form chain with a realistic smile.

    Bid/ask are rounded *outward* from the true Black-76 value: the bid floors
    and the ask ceilings. That keeps the executable band a strict superset of
    the true price, so the chain is arbitrage-free by construction and any flag
    the scan raises on it is a false positive by definition -- which is exactly
    what the no-false-positive tests need.
    """
    lo = forward - strike_step * (n_strikes // 2)
    K = lo + strike_step * np.arange(n_strikes, dtype=float)
    moneyness = np.log(K / forward)
    sigma = base_vol - skew * moneyness          # downside skew, upside smirk
    sigma = np.clip(sigma, 0.05, 1.50)

    call, put = black76(forward, K, T, sigma, discount)
    c_bid = np.floor((call - half_spread) / tick) * tick
    c_ask = np.ceil((call + half_spread) / tick) * tick
    p_bid = np.floor((put - half_spread) / tick) * tick
    p_ask = np.ceil((put + half_spread) / tick) * tick

    floor = tick
    rows = pd.DataFrame({
        "strike": np.concatenate([K, K]),
        "right": ["C"] * len(K) + ["P"] * len(K),
        "bid": np.maximum(np.concatenate([c_bid, p_bid]), floor),
        "ask": np.maximum(np.concatenate([c_ask, p_ask]), floor * 2),
    })
    return rows.reset_index(drop=True)


def perturb_quote(chain: pd.DataFrame, strike: float, right: str,
                  bid_delta: float = 0.0, ask_delta: float = 0.0) -> pd.DataFrame:
    """Return a copy of `chain` with one leg's touch shifted. Test helper."""
    out = chain.copy()
    mask = (np.isclose(out["strike"], strike)
            & (_normalize_right(out["right"]) == right.upper()[0]))
    if not mask.any():
        raise KeyError(f"no {right} quote at strike {strike}")
    out.loc[mask, "bid"] = out.loc[mask, "bid"] + bid_delta
    out.loc[mask, "ask"] = out.loc[mask, "ask"] + ask_delta
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> bool:
    """Three checks: no false positives, box arb caught, parity arb caught."""
    results: list[tuple[str, bool, str]] = []

    # 1. A chain that is arbitrage-free by construction must come back CLEAN.
    clean_chain = synthetic_chain()
    s = run_scan(clean_chain)
    ok1 = (s["boxes"]["n_arb"] == 0 and s["parity"]["n_violations"] == 0
           and s["parity"]["ok"])
    results.append((
        "arbitrage-free chain scans CLEAN",
        ok1,
        f"box_arb={s['boxes']['n_arb']} parity_viol={s['parity']['n_violations']} "
        f"D={s['parity']['discount']:.6f} F={s['parity']['forward']:.2f}",
    ))

    # 2. Mark a deep-ITM call 40 points below where the rest of the chain says
    #    it belongs, so the K/K+25 box can be bought for a credit -- a textbook
    #    cheap box the scan must name. The strike is chosen deep in the money
    #    precisely so the shifted quote stays positive: an ATM call is only
    #    worth ~12 points, so the same shift would push its bid negative and
    #    `pivot_chain` would discard the strike, hiding the injected arb.
    strikes = np.sort(clean_chain["strike"].unique())
    k1 = float(strikes[2])
    rigged = perturb_quote(clean_chain, k1, "C", bid_delta=-40.0, ask_delta=-40.0)
    boxes = scan_boxes(rigged)
    hit = boxes[(np.isclose(boxes["k1"], k1)) & boxes["arb_cheap"]]
    ok2 = len(hit) > 0
    results.append((
        "injected cheap box is flagged",
        ok2,
        f"flagged={int(boxes['arb_cheap'].sum())} at k1={k1:.0f} "
        f"edge={float(hit['edge'].max()) if ok2 else 0.0:.2f}",
    ))

    # 3. Lift one strike's put quotes so its combo band no longer spans the
    #    line the other 40 strikes agree on.
    k_bad = float(strikes[len(strikes) // 3])
    skewed = perturb_quote(clean_chain, k_bad, "P", bid_delta=30.0, ask_delta=30.0)
    fit = fit_parity_line(skewed)
    par = scan_parity(skewed, fit=fit)
    flagged = par[par["violation"]]
    ok3 = (len(flagged) == 1
           and bool(np.isclose(float(flagged["strike"].iloc[0]), k_bad)))
    got = float(flagged["strike"].iloc[0]) if len(flagged) else float("nan")
    results.append((
        "injected parity violation is isolated to one strike",
        ok3,
        f"violations={len(flagged)} strike={got:.0f} (expected {k_bad:.0f})",
    ))

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    if verbose:
        print("\n=== parity_scan self-test ===")
        for name, ok, detail in results:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")
            print(f"         {detail}")
        print(f"\n{'PASS' if n_pass == n_total else 'FAIL'} ({n_pass}/{n_total})")
    return n_pass == n_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_chain(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="run the built-in checks against a synthetic chain")
    ap.add_argument("--chain", default=None,
                    help="path to a long-form chain (.csv or .parquet)")
    ap.add_argument("--T", type=float, default=DEFAULT_T,
                    help="time to expiry in years (default: one 0DTE session)")
    ap.add_argument("--fee-per-leg", type=float, default=0.0,
                    help="per-contract fee charged into executable prices")
    ap.add_argument("--write", action="store_true",
                    help=f"write summary/CSVs under {OUT}")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest() else 1

    chain = _load_chain(a.chain) if a.chain else synthetic_chain(T=a.T)
    if a.chain is None:
        print("(no --chain given; scanning a synthetic arbitrage-free chain)")

    s = run_scan(chain, T=a.T, fee_per_leg=a.fee_per_leg, write=a.write)
    print("\n=== 0DTE model-free static-arbitrage scan ===")
    print(f"strikes usable          : {s['n_strikes_usable']}")
    print(f"box pairs priced        : {s['boxes']['n_pairs']} "
          f"(skipped {s['boxes']['n_skipped_over_max_width']} over max width)")
    print(f"box arbitrages          : {s['boxes']['n_arb']} "
          f"({s['boxes']['n_cheap']} cheap / {s['boxes']['n_rich']} rich), "
          f"max edge {s['boxes']['max_edge']:.2f}")
    if s["parity"]["ok"]:
        print(f"implied discount factor : {s['parity']['discount']:.6f}")
        print(f"implied forward         : {s['parity']['forward']:.2f}")
        if s["parity"]["rate_identifiable"]:
            print(f"implied rate            : "
                  f"{s['parity']['implied_rate']*100:.2f}%")
        else:
            print(f"implied rate            : "
                  f"{s['parity']['implied_rate']*100:.2f}%  "
                  f"NOT IDENTIFIABLE -- discounting moves the line "
                  f"{s['parity']['discount_effect_pts']:.4f} pts vs "
                  f"{s['parity']['quote_resolution_pts']:.4f} pts of quote "
                  f"resolution; treat D as 1.0")
        print(f"parity fit RMSE         : {s['parity']['rmse']:.4f} "
              f"({s['parity']['n_used']} strikes, "
              f"{s['parity']['n_trimmed']} trimmed)")
    else:
        print("parity fit              : DEGENERATE (too few usable strikes)")
    print(f"parity violations       : {s['parity']['n_violations']}, "
          f"max edge {s['parity']['max_edge']:.2f}")
    print(f"VERDICT                 : {s['verdict']}")
    print(f"\n{s['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
