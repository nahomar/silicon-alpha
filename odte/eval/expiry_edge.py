"""Hold-to-expiry edge test for SPXW 0DTE.

One question, asked so it can come back "no":

    At a decision time t, can a model tell whether the move left in the day
    will be larger or smaller than the options are pricing -- well enough to
    make money after the spread and fees?

Why hold to expiry
------------------
`option_edge.py` trades bar to bar and pays the spread on entry and exit; on
its synthetic session a perfect next-bar forecast lost 82% of its edge to the
spread. SPXW 0DTE options cash-settle at the SPX close, so a position held to
the bell crosses the spread once and pays nothing to exit. That changes the
economics enough to be worth testing on its own.

Trade unit: an iron fly at the at-the-money strike K, wings at K +/- W
---------------------------------------------------------------------
  short fly  sell the K call and put at the bid, buy the wings at the ask.
             Wins when the close lands near K. Max loss W - credit.
  long fly   the mirror image, paid at the touch the other way.
Both are defined-risk. The bare straddle is kept only as a diagnostic.

Model
-----
Target: realized / implied  r = |S_T - K| / straddle_mid(t).  r < 1 means
the options overpriced the rest of the day. The model forecasts r from
features known at t. The rule sells the fly below 1 - margin, buys it above
1 + margin, and stays flat otherwise. Walk-forward by day: the model used on
day d was fit only on days before d.

Leakage rule
------------
A quote stamped at time s is first visible at ceil(s) to the minute, so a
10:00 decision sees nothing stamped after 10:00. Features use the implied
forward path up to and including t plus the previous session's settlement.
The day's own settlement appears only in targets and P&L.

POWER NOTE (measured on the synthetic world, 2026-11-07)
--------------------------------------------------------
With a planted edge (market prices each day halfway between its true vol and
the long-run level, plus a 10% premium), the model's own Sharpe is clearly
positive within ~60-120 sessions. Proving it *beats always-short-fly* is far
harder: at 250 sessions only 1 of 3 seeds cleared 95%. Fly payoffs are lumpy
and selling premium is already a strong baseline. Read a 5-week run as a
smoke test of the pipeline, not as evidence either way.

Usage:
    PYTHONPATH=. python3 -m odte.eval.expiry_edge --selftest
    PYTHONPATH=. python3 -m odte.eval.expiry_edge --data-dir data/odte_0dte --write
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .panel_stats import day_block_bootstrap, spearman_ic

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "odte_0dte"
OUT = ROOT / "reports" / "expiry_edge"

MULT = 100.0                 # SPX option multiplier, $ per index point
FEE_PER_CONTRACT = 1.20      # exchange + regulatory + commission, $ per leg
DECISION_TIMES = ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00")
FEATURES = [
    "implied_move", "minutes_to_close", "iv_per_rootmin", "rv_so_far",
    "rv_last30", "iv_over_rv", "move_so_far", "range_so_far", "gap",
    "skew", "prev_ratio", "dow",
]
# Every 15 minutes, 10:00 .. 15:00 (21 checks). Mode "every15" trades at
# most once a day, on the first strong signal (STRONG_MARGIN).
GRID15 = tuple(f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 15, 30, 45)
               if h < 15 or m == 0)
STRONG_MARGIN = 0.30
# Mode "events": decide EVENT_OFFSETS_MIN after a scheduled intraday event
# (FOMC 14:00, minutes 14:00, long auctions 13:00, 10:00 releases), and 5 min
# after a market shock: a 5-minute move above the SHOCK_QUANTILE of all
# 5-minute moves in the previous SHOCK_LOOKBACK sessions (prior days only).
# A multiple of the median was tried first and rejected in testing before
# any real result: at 4x it fires ~twice a day on pure noise.
EVENT_OFFSETS_MIN = (5, 30)
SHOCK_QUANTILE, SHOCK_LOOKBACK, MAX_SHOCKS, SHOCK_GAP_MIN = 0.999, 20, 2, 30
EVENT_FEATURES = ["min_since_event", "move_since_event", "ivchg_since_event", "is_event_row"]
# Objective B ("danger detector"), pre-registered 2026-11-08 before any result:
# a day-move counts as danger when realized/implied >= DANGER_RATIO. Default
# is to sell the fly; skip when P(danger) > SKIP_X x the training base rate,
# flip to buying when > FLIP_X x (flip variant only).
DANGER_RATIO, SKIP_X, FLIP_X = 1.25, 2.0, 3.0
# Items 1-3, pre-registered 2026-11-08 before any result:
#   exits: close when profit >= EXIT_TP x max profit or loss >= EXIT_SL x max
#          loss, checked every minute, crossing the spread (4 more fees);
#          otherwise hold to the close.
#   post-event: no short within POST_EVENT_MIN minutes after an event/shock.
EXIT_TP, EXIT_SL, POST_EVENT_MIN = 0.5, 0.5, 60
PRIMARY_MODEL = "ml_ridge"   # pre-registered: tiny samples favour a linear model
RISK_BUDGET = 1000.0         # $ max loss per trade; each trade is sized to it
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Session -> minute grid -> implied forward
# ---------------------------------------------------------------------------

def minute_grid(day_chain: pd.DataFrame, max_stale_min: int = 5) -> dict:
    """As-of bid/ask for every (minute, contract) in one session.

    Returns {"bid": df, "ask": df}: index is each minute 09:30..16:00, columns
    are a MultiIndex (option_type, strike). A quote becomes visible at the
    first whole minute at or after its stamp and is carried forward at most
    `max_stale_min` minutes, after which the contract counts as unquoted.
    """
    c = day_chain.sort_values("quote_datetime")
    t = pd.to_datetime(c["quote_datetime"]).dt.ceil("min")
    day = t.iloc[0].normalize()
    idx = pd.date_range(day + pd.Timedelta(hours=9, minutes=30),
                        day + pd.Timedelta(hours=16), freq="min")
    keys = [t.to_numpy(), c["option_type"].to_numpy(),
            c["strike"].to_numpy(dtype=float)]
    wide = {}
    sides = ["bid", "ask"] + [f for f in ("bid_size", "ask_size") if f in day_chain.columns]
    for side in sides:
        s = pd.Series(c[side].to_numpy(dtype=float),
                      index=pd.MultiIndex.from_arrays(keys))
        s = s[~s.index.duplicated(keep="last")]
        w = s.unstack([1, 2])
        w = (w.reindex(w.index.union(idx)).sort_index()
             .ffill(limit=max_stale_min).reindex(idx))
        wide[side] = w
    cols = wide["bid"].columns.union(wide["ask"].columns)
    return {k: v.reindex(columns=cols).sort_index(axis=1)
            for k, v in wide.items()}


def _leg(grid: dict, side: str, t: pd.Timestamp, right: str, k: float) -> float:
    """One quote field for one contract at minute t (NaN if absent)."""
    try:
        v = grid[side].at[t, (right, k)]
    except KeyError:
        return float("nan")
    return float(v) if v is not None else float("nan")


def _mids(grid: dict) -> pd.DataFrame:
    b, a = grid["bid"], grid["ask"]
    ok = (b > 0) & (a >= b)
    return ((b + a) / 2.0).where(ok)


def implied_forward(grid: dict, n_strikes: int = 3) -> pd.Series:
    """Put-call parity forward per minute: F = K + C_mid - P_mid.

    Uses the `n_strikes` strikes where call and put are closest in price
    (the ones nearest the money, where parity is least sensitive to quote
    noise) and takes the median. Interest over a few hours is ignored; at 4%
    a year it moves F by under 0.01 point.
    """
    mid = _mids(grid)
    types = set(mid.columns.get_level_values(0))
    if not {"C", "P"} <= types:
        return pd.Series(np.nan, index=mid.index)
    C, P = mid["C"], mid["P"]
    ks = C.columns.intersection(P.columns)
    d = C[ks].to_numpy() - P[ks].to_numpy()
    fk = ks.to_numpy(dtype=float)[None, :] + d
    absd = np.where(np.isfinite(d), np.abs(d), np.inf)
    order = np.argsort(absd, axis=1)[:, :n_strikes]
    picked = np.take_along_axis(fk, order, axis=1)
    finite = np.isfinite(np.take_along_axis(absd, order, axis=1))
    picked = np.where(finite, picked, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        F = np.nanmedian(picked, axis=1)
    return pd.Series(F, index=mid.index)


# ---------------------------------------------------------------------------
# Decision rows: features at t, payoff at the close
# ---------------------------------------------------------------------------

def _two_sided(grid: dict, t: pd.Timestamp):
    b, a = grid["bid"].loc[t], grid["ask"].loc[t]
    ok = b.notna() & a.notna() & (b > 0) & (a >= b)
    b, a = b[ok], a[ok]
    out = {}
    for right in ("C", "P"):
        if right in b.index.get_level_values(0):
            out[right] = (b.xs(right), a.xs(right))
        else:
            out[right] = (pd.Series(dtype=float), pd.Series(dtype=float))
    return out


def _nearest(series_index: pd.Index, target: float):
    if len(series_index) == 0:
        return None
    arr = series_index.to_numpy(dtype=float)
    return float(arr[np.argmin(np.abs(arr - target))])


def _atm_iv(grid: dict, F: pd.Series, t: pd.Timestamp, close_t: pd.Timestamp) -> float:
    """ATM straddle / F / sqrt(minutes left) at minute t (NaN if unquoted)."""
    if t not in F.index or not np.isfinite(F.at[t]):
        return float("nan")
    q = _two_sided(grid, t)
    (cb, ca), (pb, pa) = q["C"], q["P"]
    K = _nearest(cb.index.intersection(pb.index), float(F.at[t]))
    if K is None:
        return float("nan")
    mid = (cb[K] + ca[K] + pb[K] + pa[K]) / 2.0
    mtc = max((close_t - t).total_seconds() / 60.0, 1.0)
    return mid / float(F.at[t]) / math.sqrt(mtc)


def _exit_pnl(grid: dict, t, close_t, K, W, credit, debit, intrinsic, fee):
    """Fly P&L (points, before entry fees) with take-profit / stop-loss exits.

    Walks the minutes after entry; at each one the close-out price is read
    from the touch (short fly: buy the body at the ask, sell the wings at the
    bid). Returns (short_pnl, long_pnl, short_exited, long_exited), with the
    exit-leg fees already subtracted in points.
    """
    legs = [("C", K), ("P", K), ("C", K + W), ("P", K - W)]
    sl = slice(t + pd.Timedelta(minutes=1), close_t - pd.Timedelta(minutes=1))
    try:
        b = grid["bid"].loc[sl, legs].to_numpy(dtype=float)
        a = grid["ask"].loc[sl, legs].to_numpy(dtype=float)
    except KeyError:
        return credit - intrinsic, intrinsic - debit, False, False
    ok = np.isfinite(b).all(axis=1) & np.isfinite(a).all(axis=1)
    fee_pts = 4 * fee / MULT
    s_cost = a[:, 0] + a[:, 1] - b[:, 2] - b[:, 3]           # buy back a short fly
    l_val = b[:, 0] + b[:, 1] - a[:, 2] - a[:, 3]            # sell out a long fly
    s_un, l_un = credit - s_cost, l_val - debit
    s_hit = ok & ((s_un >= EXIT_TP * credit) | (s_un <= -EXIT_SL * (W - credit)))
    l_hit = ok & ((l_un >= EXIT_TP * (W - debit)) | (l_un <= -EXIT_SL * debit))
    sp = float(s_un[np.argmax(s_hit)]) - fee_pts if s_hit.any() else credit - intrinsic
    lp = float(l_un[np.argmax(l_hit)]) - fee_pts if l_hit.any() else intrinsic - debit
    return sp, lp, bool(s_hit.any()), bool(l_hit.any())


def _exit_cols(grid, t, close_t, K, W, credit, debit, intrinsic, fee, fees4, fly_ok) -> dict:
    if not fly_ok:
        return {"pnl_short_fly_x": np.nan, "pnl_long_fly_x": np.nan,
                "exit_short": False, "exit_long": False}
    sp, lp, se, le = _exit_pnl(grid, t, close_t, K, W, credit, debit, intrinsic, fee)
    return {"pnl_short_fly_x": sp * MULT - fees4, "pnl_long_fly_x": lp * MULT - fees4,
            "exit_short": se, "exit_long": le}


def session_rows(grid: dict, F: pd.Series, settle: float,
                 prev_settle: float = float("nan"),
                 prev_ratio: float = float("nan"),
                 times=DECISION_TIMES, wing_mult: float = 1.5,
                 fee: float = FEE_PER_CONTRACT, triggers=None,
                 event_marks=(), close_hhmm=None) -> list:
    """One row per decision time: causal features, then settlement P&L.

    `triggers` maps hhmm -> "grid" | "event" | "shock" (default: `times`, all
    "grid"). `event_marks` are today's event times (scheduled or shock); each
    row gets features relative to the last one at or before its own time.
    """
    triggers = triggers or {h: "grid" for h in times}
    marks = sorted(pd.Timestamp(m) for m in event_marks)
    day = grid["bid"].index[0].normalize()
    valid_F = F.dropna()
    if valid_F.empty:
        return []
    # Early-close sessions (day after Thanksgiving, Christmas Eve) stop
    # quoting at 13:00; minutes-to-close must count to the real bell.
    close_t = day + pd.Timedelta(hours=13 if valid_F.index[-1] < day + pd.Timedelta(hours=14) else 16)
    if close_hhmm:
        # Verified early close. The options feed can keep showing quotes after
        # the 13:00 bell (2026-07-02: until 16:00), so quotes alone cannot
        # identify these days, and a decision after the bell would already
        # know the settlement.
        close_t = day + pd.Timedelta(hours=int(close_hhmm[:2]), minutes=int(close_hhmm[3:]))
        triggers = {h: v for h, v in triggers.items() if h < close_hhmm}
    F_open = float(valid_F.iloc[0])
    rows = []
    for hhmm in sorted(triggers):
        t = day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))
        if t not in F.index or not np.isfinite(F.at[t]):
            continue
        Ft = float(F.at[t])
        path = F.loc[:t].dropna().to_numpy()
        r1 = np.diff(np.log(path)) if len(path) > 2 else np.array([])
        rv = float(r1.std(ddof=1)) if len(r1) > 5 else float("nan")
        rv30 = float(r1[-30:].std(ddof=1)) if len(r1) > 10 else float("nan")
        mtc = (close_t - t).total_seconds() / 60.0

        q = _two_sided(grid, t)
        (cb, ca), (pb, pa) = q["C"], q["P"]
        both = cb.index.intersection(pb.index)
        K = _nearest(both, Ft)
        if K is None:
            continue
        straddle_bid = cb[K] + pb[K]
        straddle_ask = ca[K] + pa[K]
        straddle_mid = (straddle_bid + straddle_ask) / 2.0
        if straddle_mid <= 0:
            continue
        implied_move = straddle_mid / Ft
        iv_rm = implied_move / math.sqrt(max(mtc, 1.0))

        # Wings: smallest symmetric distance >= wing_mult * straddle.
        c_d = set(np.round(cb.index.to_numpy(dtype=float) - K, 4))
        p_d = set(np.round(K - pb.index.to_numpy(dtype=float), 4))
        dists = sorted(d for d in (c_d & p_d) if d >= wing_mult * straddle_mid)
        W = dists[0] if dists else float("nan")
        if np.isfinite(W):
            cw_b, cw_a = cb[K + W], ca[K + W]
            pw_b, pw_a = pb[K - W], pa[K - W]
            credit = straddle_bid - (cw_a + pw_a)
            credit_mid = straddle_mid - ((cw_b + cw_a) + (pw_b + pw_a)) / 2.0
            debit = straddle_ask - (cw_b + pw_b)
        else:
            credit = credit_mid = debit = float("nan")

        # Skew: OTM put vs OTM call one implied move away.
        kp = _nearest(pb.index, K - straddle_mid)
        kc = _nearest(cb.index, K + straddle_mid)
        skew = (((pb[kp] + pa[kp]) - (cb[kc] + ca[kc])) / 2.0 / straddle_mid
                if kp is not None and kc is not None else float("nan"))

        # Post-event features: relative to the last event at or before t.
        past = [m for m in marks if m <= t]
        if past:
            e = past[-1]
            Fe = F.at[e] if e in F.index else np.nan
            iv_before = _atm_iv(grid, F, e - pd.Timedelta(minutes=1), close_t)
            ev = {"min_since_event": (t - e).total_seconds() / 60.0,
                  "move_since_event": (abs(math.log(Ft / Fe)) * 1e4
                                       if np.isfinite(Fe) and Fe > 0 else np.nan),
                  "ivchg_since_event": (iv_rm / iv_before - 1.0
                                        if np.isfinite(iv_before) and iv_before > 0 else np.nan)}
        else:
            ev = {"min_since_event": np.nan, "move_since_event": np.nan,
                  "ivchg_since_event": np.nan}
        ev["is_event_row"] = float(triggers[hhmm] != "grid")

        move = abs(settle - K)
        intrinsic = min(move, W) if np.isfinite(W) else float("nan")
        fly_ok = np.isfinite(W) and credit > 0 and debit > 0
        fees4 = 4 * fee

        # --- execution: rest at mid for 5 minutes, else cross at t+5 --------
        # A resting sell at the mid credit L counts as filled only if, at a
        # later minute, the fly's touch credit reaches L + 0.05: the market
        # came through our price, not merely to it. Same for buys. Fills
        # therefore happen when price moved our way, which is the realistic
        # (adverse-selection) bias rather than an optimistic one.
        ex = {"fill_mid_short": False, "fill_mid_long": False,
              "credit_t5": np.nan, "debit_t5": np.nan,
              "cap_short": np.nan, "cap_long": np.nan}
        if fly_ok:
            legs = (("C", K), ("P", K), ("C", K + W), ("P", K - W))
            for k in range(1, 6):
                tk = t + pd.Timedelta(minutes=k)
                b = [_leg(grid, "bid", tk, r_, k_) for r_, k_ in legs]
                a = [_leg(grid, "ask", tk, r_, k_) for r_, k_ in legs]
                if not all(np.isfinite(b + a)):
                    continue
                c_touch = b[0] + b[1] - a[2] - a[3]
                d_touch = a[0] + a[1] - b[2] - b[3]
                if c_touch >= credit_mid + 0.05:
                    ex["fill_mid_short"] = True
                if d_touch <= credit_mid - 0.05:
                    ex["fill_mid_long"] = True
                if k == 5:
                    ex["credit_t5"], ex["debit_t5"] = c_touch, d_touch
            if "bid_size" in grid:
                bs = [_leg(grid, "bid_size", t, r_, k_) for r_, k_ in legs]
                as_ = [_leg(grid, "ask_size", t, r_, k_) for r_, k_ in legs]
                ex["cap_short"] = float(np.nanmin([bs[0], bs[1], as_[2], as_[3]]))
                ex["cap_long"] = float(np.nanmin([as_[0], as_[1], bs[2], bs[3]]))

        # --- iron condor: shorts at +/-0.75 x straddle, wings 1.0 x beyond ---
        cd = {"pnl_short_condor": np.nan, "pnl_long_condor": np.nan,
              "risk_short_condor": np.nan, "risk_long_condor": np.nan}
        kcs = _nearest(cb.index[cb.index >= K + 0.75 * straddle_mid], K + 0.75 * straddle_mid)
        kps = _nearest(pb.index[pb.index <= K - 0.75 * straddle_mid], K - 0.75 * straddle_mid)
        if kcs is not None and kps is not None:
            kcl = _nearest(cb.index[cb.index > kcs], kcs + straddle_mid)
            kpl = _nearest(pb.index[pb.index < kps], kps - straddle_mid)
            if kcl is not None and kpl is not None:
                c_cr = cb[kcs] + pb[kps] - ca[kcl] - pa[kpl]
                c_db = ca[kcs] + pa[kps] - cb[kcl] - pb[kpl]
                wmax = max(kcl - kcs, kps - kpl)
                intr = (min(max(settle - kcs, 0.0), kcl - kcs)
                        + min(max(kps - settle, 0.0), kps - kpl))
                if c_cr > 0 and c_db > 0:
                    cd = {"pnl_short_condor": (c_cr - intr) * MULT - fees4,
                          "pnl_long_condor": (intr - c_db) * MULT - fees4,
                          "risk_short_condor": (wmax - c_cr) * MULT + fees4,
                          "risk_long_condor": c_db * MULT + fees4}
        rows.append({
            "day": day, "t": t, "hhmm": hhmm, "F": Ft, "K": K, "W": W,
            "settle": settle, "trigger": triggers[hhmm],
            "on6": hhmm in DECISION_TIMES, **ev,
            # features (known at t)
            "implied_move": implied_move,
            "minutes_to_close": mtc,
            "iv_per_rootmin": iv_rm,
            "rv_so_far": rv,
            "rv_last30": rv30,
            "iv_over_rv": iv_rm / rv if rv and np.isfinite(rv) and rv > 0 else float("nan"),
            "move_so_far": math.log(Ft / F_open),
            "range_so_far": float((path.max() - path.min()) / F_open),
            "gap": (math.log(F_open / prev_settle)
                    if np.isfinite(prev_settle) and prev_settle > 0 else float("nan")),
            "skew": skew,
            "prev_ratio": prev_ratio,
            "dow": float(day.dayofweek),
            # costs at t
            "straddle_mid": straddle_mid,
            "straddle_spread": straddle_ask - straddle_bid,
            "fly_credit": credit, "fly_credit_mid": credit_mid,
            "fly_debit": debit,
            # outcome (known only after the close)
            "ratio": move / straddle_mid,
            "pnl_short_fly": (credit - intrinsic) * MULT - fees4 if fly_ok else float("nan"),
            "pnl_long_fly": (intrinsic - debit) * MULT - fees4 if fly_ok else float("nan"),
            "pnl_short_fly_mid": (credit_mid - intrinsic) * MULT if fly_ok else float("nan"),
            "risk_short_fly": (W - credit) * MULT + fees4 if fly_ok else float("nan"),
            "risk_long_fly": debit * MULT + fees4 if fly_ok else float("nan"),
            "pnl_short_straddle": (straddle_bid - move) * MULT - 2 * fee,
            # execution variants (fly): mid if the market came through, else t+5 touch
            "pnl_short_fly_midx": (credit_mid - intrinsic) * MULT - fees4 if fly_ok else np.nan,
            "pnl_long_fly_midx": (intrinsic - credit_mid) * MULT - fees4 if fly_ok else np.nan,
            "risk_short_fly_midx": (W - credit_mid) * MULT + fees4 if fly_ok else np.nan,
            "risk_long_fly_midx": credit_mid * MULT + fees4 if fly_ok else np.nan,
            "pnl_short_fly_t5": (ex["credit_t5"] - intrinsic) * MULT - fees4 if fly_ok else np.nan,
            "pnl_long_fly_t5": (intrinsic - ex["debit_t5"]) * MULT - fees4 if fly_ok else np.nan,
            "risk_short_fly_t5": (W - ex["credit_t5"]) * MULT + fees4 if fly_ok else np.nan,
            "risk_long_fly_t5": ex["debit_t5"] * MULT + fees4 if fly_ok else np.nan,
            **ex, **cd,
            **_exit_cols(grid, t, close_t, K, W, credit, debit, intrinsic, fee, fees4, fly_ok),
        })
    return rows


def _shock_marks(F: pd.Series, thr: float, day: pd.Timestamp) -> list:
    """Minutes where the 5-minute move first reaches `thr` (log units).

    Searched 09:35..14:55, at most MAX_SHOCKS a day, SHOCK_GAP_MIN apart. A
    shock at minute m uses F up to m only; its decision row is at m + 5.
    """
    f = F.dropna()
    r5 = (np.log(f) - np.log(f.shift(5))).abs()
    lo, hi = day + pd.Timedelta(hours=9, minutes=35), day + pd.Timedelta(hours=14, minutes=55)
    hits = r5[(r5.index >= lo) & (r5.index <= hi) & (r5 >= thr)]
    marks = []
    for m in hits.index:
        if not marks or (m - marks[-1]) >= pd.Timedelta(minutes=SHOCK_GAP_MIN):
            marks.append(m)
        if len(marks) >= MAX_SHOCKS:
            break
    return marks


def load_early_closes() -> dict:
    p = ROOT / "configs" / "calendar" / "early_closes.csv"
    if not p.exists():
        return {}
    e = pd.read_csv(p, dtype=str)
    return {pd.Timestamp(d): t for d, t in zip(e["date"], e["close_et"])}


def build_dataset(chains: dict, settlements: dict, times=DECISION_TIMES,
                  event_times=None, shocks: bool = False, early_closes=None,
                  **kw) -> pd.DataFrame:
    """Decision rows across sessions, in date order.

    `chains` maps session date -> that session's quotes; `settlements` maps
    date -> SPX official close. A session with no settlement falls back to
    the last implied forward of the day and is flagged.

    `times` is the regular grid (DECISION_TIMES or GRID15). `event_times`
    maps session date -> list of scheduled intraday event times ("HH:MM");
    each adds decisions EVENT_OFFSETS_MIN later. `shocks=True` adds a
    decision 5 minutes after each market shock (threshold: SHOCK_QUANTILE of
    prior sessions' 5-minute moves, prior days only).
    """
    rows = []
    prev_settle, prev_ratio = float("nan"), float("nan")
    hist5 = []                                   # |5-min moves| of prior sessions
    for day in sorted(chains):
        day = pd.Timestamp(day).normalize()
        grid = minute_grid(chains[day])
        F = implied_forward(grid)
        settle = settlements.get(day)
        src = "spx_close"
        if settle is None or not np.isfinite(settle):
            settle = float(F.dropna().iloc[-1]) if F.notna().any() else float("nan")
            src = "last_forward"
        triggers = {h: "grid" for h in times}
        marks = []
        first, last = day + pd.Timedelta(hours=10), day + pd.Timedelta(hours=15)
        for hh in (event_times or {}).get(day, []):
            et = day + pd.Timedelta(hours=int(hh[:2]), minutes=int(hh[3:]))
            marks.append(et)
            for off in EVENT_OFFSETS_MIN:
                tt = et + pd.Timedelta(minutes=off)
                if first <= tt <= last:
                    triggers[tt.strftime("%H:%M")] = "event"
        if shocks and len(hist5) >= 5:
            thr = float(np.quantile(np.concatenate(hist5[-SHOCK_LOOKBACK:]), SHOCK_QUANTILE))
            ec = (early_closes or {}).get(day)
            for m in _shock_marks(F if not ec else F[F.index < day + pd.Timedelta(hours=int(ec[:2]))], thr, day):
                marks.append(m)
                tt = m + pd.Timedelta(minutes=5)
                if first <= tt <= last:
                    triggers[tt.strftime("%H:%M")] = "shock"
        close_hhmm = (early_closes or {}).get(day)
        if close_hhmm:
            marks = [m for m in marks if m.strftime("%H:%M") < close_hhmm]
        day_rows = session_rows(grid, F, settle, prev_settle, prev_ratio,
                                triggers=triggers, event_marks=marks,
                                close_hhmm=close_hhmm, **kw)
        for r in day_rows:
            r["settle_source"] = src
        rows.extend(day_rows)
        if day_rows:
            ten = [r for r in day_rows if r["hhmm"] == "10:00"]
            prev_ratio = (ten or day_rows)[0]["ratio"]
        prev_settle = settle
        f = F.dropna()
        ec = (early_closes or {}).get(day)
        if ec:                               # stale post-bell quotes are not moves
            f = f[f.index < day + pd.Timedelta(hours=int(ec[:2]), minutes=int(ec[3:]))]
        r5 = (np.log(f) - np.log(f.shift(5))).abs().dropna()
        r5 = r5[r5.index >= day + pd.Timedelta(hours=9, minutes=35)].to_numpy()   # same window as detection
        if len(r5):
            hist5.append(r5)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Models and the walk-forward
# ---------------------------------------------------------------------------

def _make_model(name: str):
    if name == "ml_lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=200, learning_rate=0.03,
                             num_leaves=4, min_child_samples=8,
                             subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=0, verbose=-1)
    if name == "ml_ridge":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(name)


def _rule(pred_ratio: np.ndarray, margin: float) -> np.ndarray:
    return np.where(pred_ratio < 1 - margin, 1,
                    np.where(pred_ratio > 1 + margin, -1, 0))


REFIT_EVERY = 1   # sessions between refits; the 4-year run uses 5 (weekly)


def walk_forward(ds: pd.DataFrame, model: str, min_train_days: int = 8,
                 margin: float = 0.15, shuffle_labels_seed=None, features=None,
                 collect_gain=None):
    """Predictions and actions for every row on a day with enough history.

    Rows on the first `min_train_days` sessions get NaN (not tested). With
    `shuffle_labels_seed` set, training targets are permuted -- a placebo
    that must not beat the baselines.
    """
    days = np.array(sorted(ds["day"].unique()))
    pred = np.full(len(ds), np.nan)
    rng = np.random.default_rng(shuffle_labels_seed)
    y_all = ds["ratio"].clip(upper=4.0).to_numpy()
    X_all = ds[list(features or FEATURES)].to_numpy(dtype=float)
    dvals = ds["day"].to_numpy()
    m = None
    for i, d in enumerate(days):
        if i < min_train_days:
            continue
        te = dvals == d
        if m is None or (i - min_train_days) % REFIT_EVERY == 0:
            # Refit on everything strictly before day d; between refits the
            # last model is reused, so every prediction is still out-of-sample.
            tr = (dvals < d) & np.isfinite(y_all)
            y = y_all[tr]
            if shuffle_labels_seed is not None:
                y = rng.permutation(y)
            m = _make_model(model)
            m.fit(X_all[tr], y)
            if collect_gain is not None and model == "ml_lgbm":
                g = m.booster_.feature_importance("gain").astype(float)
                collect_gain.append(g / g.sum() if g.sum() > 0 else g)
        pred[te] = m.predict(X_all[te])
    act = np.where(np.isfinite(pred), _rule(np.nan_to_num(pred, nan=1.0), margin), 0)
    return pred, act


def _make_classifier(name: str):
    if name == "danger_lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=4,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
    if name == "danger_logit":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=2000))
    raise ValueError(name)


def walk_forward_danger(ds: pd.DataFrame, model: str, min_train_days: int = 8,
                        features=None):
    """P(danger) per row and the training base rate in force that day.

    Same walk-forward discipline as `walk_forward`: the model for day d is
    fit only on days before d.
    """
    days = np.array(sorted(ds["day"].unique()), dtype="datetime64[ns]")
    dvals = ds["day"].to_numpy(dtype="datetime64[ns]")
    y_all = (ds["ratio"].to_numpy() >= DANGER_RATIO).astype(int)
    X_all = ds[list(features or FEATURES)].to_numpy(dtype=float)
    p = np.full(len(ds), np.nan)
    base = np.full(len(ds), np.nan)
    m, rate = None, np.nan
    for i, d in enumerate(days):
        if i < min_train_days:
            continue
        te = dvals == d
        if m is None or (i - min_train_days) % REFIT_EVERY == 0:
            tr = dvals < d
            y = y_all[tr]
            if y.min() == y.max():
                continue
            m = _make_classifier(model)
            m.fit(X_all[tr], y)
            rate = y.mean()
        p[te] = m.predict_proba(X_all[te])[:, 1]
        base[te] = rate
    return p, base


def danger_actions(p, base, mask, tested, flip: bool) -> np.ndarray:
    ok = mask & tested & np.isfinite(p) & np.isfinite(base)
    act = np.where(ok, 1, 0)                                   # default: sell the fly
    act = np.where(ok & (p > SKIP_X * base), 0, act)           # danger: sit out
    if flip:
        act = np.where(ok & (p > FLIP_X * base), -1, act)      # big danger: buy the fly
    return act


def baseline_actions(ds: pd.DataFrame, tested: np.ndarray, seed: int = 0) -> dict:
    """Simple strategies the model has to beat, on the same rows."""
    n = len(ds)
    rng = np.random.default_rng(seed)
    out = {
        "flat": np.zeros(n, dtype=int),
        "always_short_fly": np.ones(n, dtype=int),
        "always_long_fly": -np.ones(n, dtype=int),
        "random": rng.choice([-1, 0, 1], size=n),
    }
    # Non-ML rule: sell when implied/realized is above its median so far.
    rule = np.zeros(n, dtype=int)
    days = np.array(sorted(ds["day"].unique()))
    x = ds["iv_over_rv"].to_numpy(dtype=float)
    dvals = ds["day"].to_numpy()
    for d in days:
        hist = x[(dvals < d) & np.isfinite(x)]
        if len(hist) < 10:
            continue
        te = dvals == d
        rule[te] = np.where(x[te] > np.median(hist), 1, 0)
    out["rule_iv_over_rv"] = rule
    return {k: np.where(tested, v, 0) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _row_pnl(ds: pd.DataFrame, act: np.ndarray, risk_budget=RISK_BUDGET, exits: bool = False):
    """Per-row P&L. With `risk_budget`, every trade is sized so its max loss
    equals the budget (fractional lots -- a research convention), which keeps
    a high-vol day from dominating the sample just because its fly is wider.
    With `risk_budget=None`, every trade is one lot."""
    s = ds["pnl_short_fly_x" if exits else "pnl_short_fly"].to_numpy()
    l = ds["pnl_long_fly_x" if exits else "pnl_long_fly"].to_numpy()
    rs = ds["risk_short_fly"].to_numpy()
    rl = ds["risk_long_fly"].to_numpy()
    tradable = np.isfinite(s) & np.isfinite(l)
    a = np.where(tradable, act, 0)
    pnl = np.where(a == 1, s, np.where(a == -1, l, 0.0))
    risk = np.where(a == 1, rs, np.where(a == -1, rl, np.nan))
    if risk_budget:
        with np.errstate(invalid="ignore", divide="ignore"):
            pnl = np.where(a != 0, pnl / risk * risk_budget, 0.0)
        risk = np.where(a != 0, risk_budget, np.nan)
    return pnl, risk, a


def summarize(ds: pd.DataFrame, act: np.ndarray, tested: np.ndarray,
              risk_budget=RISK_BUDGET, exits: bool = False) -> dict:
    pnl, risk, a = _row_pnl(ds, act, risk_budget, exits)
    sub = pd.DataFrame({"day": ds["day"], "pnl": pnl, "traded": a != 0})[tested]
    daily = sub.groupby("day")["pnl"].sum()
    trades = sub[sub["traded"]]
    boot = day_block_bootstrap(daily.to_numpy())
    cum = daily.cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        ror = pnl[(a != 0) & tested] / risk[(a != 0) & tested]
    return {
        "n_test_days": int(len(daily)),
        "n_trades": int(len(trades)),
        "n_short": int(((a == 1) & tested).sum()),
        "n_long": int(((a == -1) & tested).sum()),
        "total_pnl": float(daily.sum()),
        "mean_daily_pnl": float(daily.mean()) if len(daily) else float("nan"),
        "mean_pnl_per_trade": float(trades["pnl"].mean()) if len(trades) else 0.0,
        "mean_return_on_risk": float(np.nanmean(ror)) if len(ror) else 0.0,
        "hit_rate": float((trades["pnl"] > 0).mean()) if len(trades) else float("nan"),
        "worst_trade": float(trades["pnl"].min()) if len(trades) else 0.0,
        "worst_day": float(daily.min()) if len(daily) else float("nan"),
        "max_drawdown": float((cum - cum.cummax()).min()) if len(cum) else float("nan"),
        "sharpe": boot["sharpe"],
        "sharpe_ci": [boot["ci_low"], boot["ci_high"]],
        "p_sharpe_le_0": boot["p_sharpe_le_0"],
        "_daily": daily,
    }


def paired_diff(daily_a: pd.Series, daily_b: pd.Series, n_boot: int = 4000,
                seed: int = 0) -> dict:
    """Mean daily P&L of A minus B, with a day-resampled 95% CI."""
    d = (daily_a - daily_b.reindex(daily_a.index)).dropna().to_numpy()
    if len(d) < 3:
        return {"mean": float("nan"), "ci": [float("nan")] * 2, "p_le_0": float("nan")}
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"mean": float(d.mean()),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_le_0": float(np.mean(bs <= 0))}


def _first_per_day(ds: pd.DataFrame, act: np.ndarray) -> np.ndarray:
    """Keep only each day's first non-zero action (rows sorted by time)."""
    out = np.zeros_like(act)
    seen = set()
    for i, (d, a) in enumerate(zip(ds["day"].to_numpy(), act)):
        if a != 0 and d not in seen:
            out[i] = a
            seen.add(d)
    return out


def _mode_masks(ds: pd.DataFrame) -> dict:
    trig = ds["trigger"] if "trigger" in ds else pd.Series("grid", index=ds.index)
    on6 = ds["on6"] if "on6" in ds else pd.Series(True, index=ds.index)
    masks = {"fixed6": on6.to_numpy(bool),
             "every15": trig.eq("grid").to_numpy(),
             "events": trig.isin(["event", "shock"]).to_numpy()}
    # every15 only differs from fixed6 if the grid is finer than 6 checks.
    if masks["every15"].sum() <= masks["fixed6"].sum():
        masks.pop("every15")
    return {k: v for k, v in masks.items() if v.any()}


def _mode_actions(mode: str, ds, pred, mask, tested, margin) -> np.ndarray:
    ok = np.isfinite(pred) & mask & tested
    if mode == "every15":
        act = np.where(ok, _rule(np.nan_to_num(pred, nan=1.0), STRONG_MARGIN), 0)
        return _first_per_day(ds, act)
    return np.where(ok, _rule(np.nan_to_num(pred, nan=1.0), margin), 0)


def _mode_baselines(mode: str, ds, mask, tested, seed) -> dict:
    m = mask & tested
    if mode == "every15":
        # One trade a day, like the strategy it is compared with.
        rng = np.random.default_rng(seed)
        first = _first_per_day(ds, m.astype(int))
        rnd = np.zeros(len(ds), dtype=int)
        for d in np.unique(ds["day"].to_numpy()[m]):
            idx = np.flatnonzero(m & (ds["day"].to_numpy() == d))
            rnd[rng.choice(idx)] = rng.choice([-1, 1])
        return {"flat": np.zeros(len(ds), dtype=int),
                "short_once_at_10": first, "long_once_at_10": -first,
                "random_once": rnd}
    b = baseline_actions(ds, tested, seed=seed)
    return {k: np.where(mask, v, 0) for k, v in b.items()}


def run_study(ds: pd.DataFrame, min_train_days: int = 8, margin: float = 0.15,
              min_days_for_verdict: int = 40, seed: int = 0,
              risk_budget=RISK_BUDGET, feature_sets=None, extra_preds=None,
              layers=None, report_window=None, danger_sets=None) -> dict:
    """Train once, score three trading versions from the same predictions.

    Models are fit walk-forward on every decision row (grid, event, shock).
    `feature_sets` maps a name to a feature list; "base" (FEATURES) always
    runs and keeps the pre-registered names ml_ridge / ml_lgbm.

    Versions ("modes"):
      fixed6   the 6 pre-registered times, margin; decides the VERDICT
      every15  every 15 minutes, first STRONG_MARGIN signal, 1 trade/day
      events   decisions after scheduled events and market shocks
    Top-level keys describe fixed6 (unchanged interface); rep["modes"] has all.
    """
    ds = ds.sort_values(["day", "t"]).reset_index(drop=True)
    days = sorted(ds["day"].unique())
    tested = ds["day"].isin(days[min_train_days:]).to_numpy()
    if report_window:
        # Score only inside the window (models still train on all earlier days).
        lo, hi = (pd.Timestamp(x) if x else None for x in report_window)
        inwin = np.ones(len(ds), bool)
        if lo is not None:
            inwin &= (ds["day"] >= lo).to_numpy()
        if hi is not None:
            inwin &= (ds["day"] <= hi).to_numpy()
        tested = tested & inwin
    feature_sets = dict(feature_sets or {})
    feature_sets["base"] = list(FEATURES)

    preds = {}
    for fs_name, cols in feature_sets.items():
        for m in ("ml_ridge", "ml_lgbm"):
            label = m if fs_name == "base" else f"{m}{fs_name}"
            preds[label], _ = walk_forward(ds, m, min_train_days, margin, features=cols)
    for label, p in (extra_preds or {}).items():
        preds[label] = np.asarray(p, dtype=float)
    widest = max(feature_sets.values(), key=len)
    placebo, _ = walk_forward(ds, PRIMARY_MODEL, min_train_days, margin,
                              shuffle_labels_seed=seed + 7, features=widest)
    model_preds = dict(preds, placebo_shuffled_labels=placebo)

    # Objective B: danger detector (classifiers), on base and widest features.
    danger = {}
    d_sets = danger_sets or {"": list(FEATURES), "+all": widest}
    for fs_name, cols in d_sets.items():
        if fs_name and len(cols) == len(FEATURES) and fs_name == "+all":
            continue
        for m in ("danger_logit", "danger_lgbm"):
            danger[f"{m}{fs_name}"] = walk_forward_danger(ds, m, min_train_days, features=cols)

    modes = {}
    for mode, mask in _mode_masks(ds).items():
        strategies = {k: _mode_actions(mode, ds, p, mask, tested, margin)
                      for k, p in model_preds.items()}
        if mode != "every15":
            for k, (pd_, base_) in danger.items():
                strategies[f"{k} skip"] = danger_actions(pd_, base_, mask, tested, flip=False)
                strategies[f"{k} skip+flip"] = danger_actions(pd_, base_, mask, tested, flip=True)
        baselines = _mode_baselines(mode, ds, mask, tested, seed)
        strategies.update(baselines)
        # Items 2-3: A combined with B, and no shorts right after events.
        combos = {}
        a_key = next((k for k in ("ml_ridge+cal+ctx", PRIMARY_MODEL) if k in strategies), None)
        b_name = next((k for k in ("danger_logit+cal+ctx", "danger_logit") if k in danger), None)
        post = ((ds["min_since_event"] <= POST_EVENT_MIN).to_numpy()
                if "min_since_event" in ds else np.zeros(len(ds), bool))
        no_post = lambda a: np.where((a == 1) & post, 0, a)
        ref_short = "short_once_at_10" if mode == "every15" else "always_short_fly"
        if a_key:
            combos[f"{a_key} -postevt"] = no_post(strategies[a_key])
        combos[f"{ref_short} -postevt"] = no_post(strategies[ref_short])
        if a_key and b_name:
            pb, bb = danger[b_name]
            flag = np.isfinite(pb) & np.isfinite(bb) & (pb > SKIP_X * bb)
            axb = np.where((strategies[a_key] == 1) & flag, 0, strategies[a_key])
            combos[f"{a_key} xB"] = axb
            combos[f"{a_key} xB -postevt"] = no_post(axb)
            if f"{b_name} skip" in strategies:
                combos[f"{b_name} skip -postevt"] = no_post(strategies[f"{b_name} skip"])
        strategies.update(combos)
        res = {k: summarize(ds, v, tested, risk_budget) for k, v in strategies.items()}
        # Item 1: the same strategies with take-profit / stop-loss exits.
        exit_keys = [k for k in [ref_short, a_key, f"{b_name} skip" if b_name else None, *combos]
                     if k and k in strategies]
        for k in exit_keys:
            res[f"{k} +exits"] = summarize(ds, strategies[k], tested, risk_budget, exits=True)
        best_base = max((b for b in baselines if b != "flat"),
                        key=lambda b: res[b]["total_pnl"])
        ref = "short_once_at_10" if mode == "every15" else "always_short_fly"
        comparisons = {}
        extra_keys = [k for k in res if k.startswith("danger_") or k in combos or k.endswith("+exits")]
        for m in list(model_preds) + extra_keys:
            targets = [ref, best_base]
            if m.endswith("+exits") and f"{ref} +exits" in res:
                targets.append(f"{ref} +exits")
            base_m = m.split("+")[0]
            if m.startswith("seq_transformer"):
                base_m = "ml_lgbm" + ("" if m == "seq_transformer" else m[len("seq_transformer"):])
            if mode == "fixed6" and base_m != m and base_m in res:
                targets.append(base_m)        # does the extra data help this model?
            comparisons[m] = {"vs_" + b: paired_diff(res[m]["_daily"], res[b]["_daily"], seed=seed)
                              for b in dict.fromkeys(targets)}
        rows = mask & tested
        forecast_ic = {m: spearman_ic(p[rows], ds.loc[rows, "ratio"].to_numpy())
                       for m, p in preds.items()}
        # Danger detectors: how well P(danger) ranks actual blow-ups (AUC).
        from sklearn.metrics import roc_auc_score
        y_d = (ds["ratio"].to_numpy() >= DANGER_RATIO).astype(int)
        danger_auc = {}
        for k, (pd_, _) in danger.items():
            r = rows & np.isfinite(pd_)
            if r.sum() > 20 and 0 < y_d[r].mean() < 1:
                danger_auc[k] = float(roc_auc_score(y_d[r], pd_[r]))
        from .portfolio import money_view
        ml_names = [k for k in preds if k != PRIMARY_MODEL]
        best_ml = max(ml_names, key=lambda k: res[k]["total_pnl"]) if ml_names else None
        pick = {PRIMARY_MODEL: strategies[PRIMARY_MODEL], ref: strategies[ref]}
        if best_ml:
            pick[f"{best_ml} (best variant, picked after the fact)"] = strategies[best_ml]
        pred_for = {PRIMARY_MODEL: preds[PRIMARY_MODEL]}
        if best_ml:
            pred_for[f"{best_ml} (best variant, picked after the fact)"] = preds[best_ml]
        money = money_view(ds, pick, pred_for, tested)
        keep = [k for k in [ref, a_key, f"{a_key} -postevt" if a_key else None,
                            f"{ref} -postevt", f"{b_name} skip" if b_name else None]
                if k and k in res]
        daily_series = {k: {str(pd.Timestamp(d).date()): float(v)
                            for d, v in res[k]["_daily"].items()} for k in keep}
        for v in res.values():
            v.pop("_daily", None)
        modes[mode] = {"strategies": res, "comparisons": comparisons, "daily": daily_series,
                       "items_1_3": [k for k in res if k in combos or k.endswith("+exits")
                                     or k in (ref, a_key, f"{b_name} skip" if b_name else "")],
                       "best_baseline": best_base, "reference": ref,
                       "forecast_ic": forecast_ic, "n_rows_tested": int(rows.sum()),
                       "danger_auc": danger_auc,
                       "danger_base_rate": float(y_d[rows].mean()) if rows.any() else float("nan"),
                       "money": money}

    importance = None
    if layers:
        # (1) drop one layer from the full LightGBM model, (2) gain shares.
        full_cols = list(widest)
        m6 = _mode_masks(ds)["fixed6"]
        rows6 = m6 & tested
        gains = []
        p_full, _ = walk_forward(ds, "ml_lgbm", min_train_days, margin,
                                 features=full_cols, collect_gain=gains)
        a_full = _mode_actions("fixed6", ds, p_full, m6, tested, margin)
        s_full = summarize(ds, a_full, tested, risk_budget)
        ic_full = spearman_ic(p_full[rows6], ds.loc[rows6, "ratio"].to_numpy())
        drop = {}
        for layer, cols in layers.items():
            keep = [c for c in full_cols if c not in set(cols)]
            if len(keep) == len(full_cols) or not keep:
                continue
            p_wo, _ = walk_forward(ds, "ml_lgbm", min_train_days, margin, features=keep)
            a_wo = _mode_actions("fixed6", ds, p_wo, m6, tested, margin)
            s_wo = summarize(ds, a_wo, tested, risk_budget)
            drop[layer] = {
                "ic_without": spearman_ic(p_wo[rows6], ds.loc[rows6, "ratio"].to_numpy()),
                "ic_drop": ic_full - spearman_ic(p_wo[rows6], ds.loc[rows6, "ratio"].to_numpy()),
                "pnl_lost_per_day": paired_diff(s_full["_daily"], s_wo["_daily"], seed=seed),
                "n_features": len(cols)}
        g = np.mean(gains, axis=0) if gains else np.zeros(len(full_cols))
        gain = dict(sorted(zip(full_cols, map(float, g)), key=lambda kv: -kv[1]))
        owner = {c: L for L, cols in layers.items() for c in cols}
        by_layer = {}
        for c, v in gain.items():
            by_layer[owner.get(c, "base")] = by_layer.get(owner.get(c, "base"), 0.0) + v
        importance = {"full_ic": ic_full, "drop_one": drop, "gain_top": dict(list(gain.items())[:20]),
                      "gain_by_layer": dict(sorted(by_layer.items(), key=lambda kv: -kv[1]))}

    main = modes["fixed6"]
    prim = main["strategies"][PRIMARY_MODEL]
    n_test_days = int(prim["n_test_days"])
    beat_best = main["comparisons"][PRIMARY_MODEL]["vs_" + main["best_baseline"]]
    if n_test_days < min_days_for_verdict:
        verdict = (f"INCONCLUSIVE: {n_test_days} test days "
                   f"(< {min_days_for_verdict}). Directional read only.")
    elif (beat_best["ci"][0] > 0 and prim["p_sharpe_le_0"] < 0.05):
        verdict = "PASS: beats the best baseline and Sharpe > 0 at 95%."
    else:
        verdict = "FAIL: no edge over the baselines that survives the CI."

    m6 = _mode_masks(ds)["fixed6"]
    summary = {
        "n_days": len(days), "n_test_days": n_test_days,
        "n_rows": int(len(ds)), "n_rows_tested": int(tested.sum()),
        "rows_by_trigger": (ds["trigger"].value_counts().to_dict() if "trigger" in ds else {}),
        "settle_sources": ds["settle_source"].value_counts().to_dict()
        if "settle_source" in ds else {},
        "mean_ratio_realized_over_implied": float(ds.loc[m6, "ratio"].mean()),
        "median_ratio": float(ds.loc[m6, "ratio"].median()),
        "median_straddle_spread_pct": float(
            (ds["straddle_spread"] / ds["straddle_mid"]).median()),
        "median_fly_credit_$": float(ds["fly_credit"].median() * MULT),
        "fee_per_contract_$": FEE_PER_CONTRACT,
        "sizing": (f"each trade sized to ${risk_budget:.0f} max loss"
                   if risk_budget else "1 lot per trade"),
    }
    return {"summary": summary, "strategies": main["strategies"],
            "comparisons": main["comparisons"], "best_baseline": main["best_baseline"],
            "forecast_ic": main["forecast_ic"], "primary_model": PRIMARY_MODEL,
            "verdict": verdict, "modes": modes, "importance": importance,
            "feature_sets": {k: len(v) for k, v in feature_sets.items()},
            "params": {"min_train_days": min_train_days, "margin": margin,
                       "refit_every": REFIT_EVERY, "report_window": report_window,
                       "strong_margin": STRONG_MARGIN,
                       "event_offsets_min": list(EVENT_OFFSETS_MIN),
                       "shock_quantile": SHOCK_QUANTILE,
                       "decision_times": list(DECISION_TIMES)}}


# ---------------------------------------------------------------------------
# Loading real data
# ---------------------------------------------------------------------------

def load_sessions(data_dir: Path) -> tuple:
    chains = {}
    for p in sorted(Path(data_dir).glob("20??-??-??.parquet")):
        chains[pd.Timestamp(p.stem)] = pd.read_parquet(p)
    settlements = {}
    sp = Path(data_dir) / "settlements.csv"
    if sp.exists():
        s = pd.read_csv(sp, parse_dates=["date"])
        settlements = {pd.Timestamp(d).normalize(): float(v)
                       for d, v in zip(s["date"], s["spx_close"])}
    return chains, settlements


# ---------------------------------------------------------------------------
# Synthetic sessions, for validating the instrument
# ---------------------------------------------------------------------------

def _bs(S, K, sig, tau, right):
    from scipy.special import ndtr
    st = sig * np.sqrt(tau)
    d1 = (np.log(S / K) + 0.5 * st ** 2) / st
    d2 = d1 - st
    if right == "C":
        return S * ndtr(d1) - K * ndtr(d2)
    return K * ndtr(-d2) - S * ndtr(-d1)


def _quote(val: np.ndarray):
    val = np.maximum(val, 0.0)
    tick = np.where(val < 3.0, 0.05, 0.10)
    hs = np.maximum(0.025, 0.004 * val)
    bid = np.floor((val - hs) / tick) * tick
    ask = np.ceil((val + hs) / tick) * tick
    return np.maximum(bid, 0.0), np.maximum(ask, tick)


def synthetic_session(day: pd.Timestamp, S_open: float, sigma_true: float,
                      sigma_mkt: float, rng, strike_step: float = 5.0,
                      path_override=None):
    """One session of SPXW-like 0DTE quotes.

    The index follows GBM at `sigma_true` (annualized on a 252 x 390-minute
    clock). Options are quoted by Black-Scholes at `sigma_mkt`, the market's
    belief. With sigma_mkt == sigma_true the market is efficient and no
    feature should help; when it under-reacts to the day's real volatility,
    realized-vol-so-far carries a genuine, detectable edge.
    """
    n = 390
    dt = 1.0 / (252 * 390)
    if path_override is None:
        z = rng.normal(0.0, sigma_true * math.sqrt(dt), size=n)
        S = S_open * np.exp(np.concatenate([[0.0], np.cumsum(z)]))
    else:
        S = np.asarray(path_override, dtype=float)
    times = day + pd.Timedelta(hours=9, minutes=30) + pd.to_timedelta(np.arange(n + 1), unit="min")
    dec = {day + pd.Timedelta(hours=int(h[:2]), minutes=int(h[3:])) for h in DECISION_TIMES}
    m_idx, ks = [], []
    for m in range(n):          # no quotes at 16:00 itself
        tau = (n - m) * dt
        base = round(S[m] / strike_step) * strike_step
        if times[m] in dec:
            span = max(4 * 0.8 * S[m] * sigma_mkt * math.sqrt(tau), 40.0)
            span = math.ceil(span / strike_step) * strike_step
            k = np.arange(base - span, base + span + strike_step / 2, strike_step)
        else:
            k = base + strike_step * np.arange(-2, 3)
        m_idx.append(np.full(len(k), m))
        ks.append(k)
    m_all = np.concatenate(m_idx)
    k_all = np.concatenate(ks)
    tau_all = (n - m_all) * dt
    S_all = S[m_all]
    frames = []
    for right in ("C", "P"):
        b, a = _quote(_bs(S_all, k_all, sigma_mkt, tau_all, right))
        frames.append(pd.DataFrame({
            "quote_datetime": times[m_all], "root": "SPXW", "expiration": day,
            "strike": k_all, "option_type": right, "bid": b, "ask": a,
            "bid_size": 10.0, "ask_size": 10.0}))
    return pd.concat(frames, ignore_index=True), float(S[-1]), S


def synthetic_world(n_days: int = 60, informed: bool = True, seed: int = 0,
                    S0: float = 6000.0):
    """Many sessions. `informed=True`: the market prices each day at a blend
    of that day's true vol and the long-run level, plus a 10% premium.
    `informed=False`: the market prices exactly the true vol, no premium."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-06", periods=n_days)
    chains, settles = {}, {}
    S_open = S0
    sig_bar = 0.15
    for d in days:
        sig = sig_bar * math.exp(rng.normal(0.0, 0.5))
        sig_mkt = (math.sqrt(sig * sig_bar) * 1.10) if informed else sig
        chain, close, _ = synthetic_session(d, S_open, sig, sig_mkt, rng)
        chains[d] = chain
        settles[d] = close
        S_open = close * math.exp(rng.normal(0.0, 0.003))
    return chains, settles


def selftest(verbose: bool = True) -> bool:
    checks = []

    # 1. Leakage: changing what happens after t must not change features at t.
    rng = np.random.default_rng(11)
    day = pd.Timestamp("2025-03-03")
    chain, close, S = synthetic_session(day, 6000.0, 0.2, 0.2, rng)
    S2 = S.copy()
    S2[200:] = S2[200:] * 1.01          # alter the path after 12:50
    rng2 = np.random.default_rng(11)
    chain2, close2, _ = synthetic_session(day, 6000.0, 0.2, 0.2, rng2, path_override=S2)
    r1 = session_rows(minute_grid(chain), implied_forward(minute_grid(chain)), close)
    r2 = session_rows(minute_grid(chain2), implied_forward(minute_grid(chain2)), close2)
    early = [h for h in DECISION_TIMES if h <= "12:00"]
    same = all(
        np.allclose([a[f] for f in FEATURES], [b[f] for f in FEATURES], equal_nan=True)
        for a, b in zip(r1, r2) if a["hhmm"] in early)
    later_differs = any(a["ratio"] != b["ratio"] for a, b in zip(r1, r2))
    checks.append(("features at t ignore the future", same and later_differs,
                   f"compared {len(early)} early decision times"))

    # 2. Parity forward recovers the true index level.
    F = implied_forward(minute_grid(chain))
    err = np.nanmax(np.abs(F.to_numpy()[:-1] - S[:-1]))
    checks.append(("parity forward tracks the index", err < 0.5,
                   f"max |F - S| = {err:.3f} pts"))

    # 3. Costs are charged: short + long fly on the same row must lose money.
    ds_eff = build_dataset(*synthetic_world(n_days=50, informed=False, seed=3))
    both = (ds_eff["pnl_short_fly"] + ds_eff["pnl_long_fly"]).dropna()
    checks.append(("round trip through the spread always costs", bool((both < 0).all()),
                   f"max(short+long) = {both.max():.2f} $"))

    # 4. Efficient market: the model must not manufacture an edge.
    rep_eff = run_study(ds_eff)
    eff = rep_eff["strategies"][PRIMARY_MODEL]
    ok4 = not (eff["total_pnl"] > 0 and eff["p_sharpe_le_0"] < 0.05)
    checks.append(("no false edge when the market is efficient", ok4,
                   f"{PRIMARY_MODEL} total=${eff['total_pnl']:.0f} "
                   f"p(Sharpe<=0)={eff['p_sharpe_le_0']:.2f}"))

    # 5. Mispriced market: the model must find the planted signal and make
    #    money with it. (Proving it beats always-short-fly is a much harder
    #    bar -- see POWER NOTE in the module docstring -- so it is reported,
    #    not required.)
    ds_inf = build_dataset(*synthetic_world(n_days=120, informed=True, seed=4))
    rep_inf = run_study(ds_inf)
    prim = rep_inf["strategies"][PRIMARY_MODEL]
    cmp = rep_inf["comparisons"][PRIMARY_MODEL]["vs_always_short_fly"]
    ok5 = (rep_inf["forecast_ic"][PRIMARY_MODEL] > 0.1
           and prim["p_sharpe_le_0"] < 0.05)
    checks.append(("planted edge is found and is profitable", ok5,
                   f"forecast IC={rep_inf['forecast_ic'][PRIMARY_MODEL]:+.2f}, "
                   f"Sharpe {prim['sharpe']:+.2f} p(<=0)={prim['p_sharpe_le_0']:.3f}; "
                   f"vs always_short {cmp['mean']:+.0f}/day "
                   f"CI [{cmp['ci'][0]:+.0f}, {cmp['ci'][1]:+.0f}] (reported only)"))

    n_ok = sum(ok for _, ok, _ in checks)
    if verbose:
        print("\n=== expiry_edge self-test ===")
        for name, ok, detail in checks:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}\n         {detail}")
        print(f"\n{'PASS' if n_ok == len(checks) else 'FAIL'} ({n_ok}/{len(checks)})")
    return n_ok == len(checks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_report(rep: dict) -> None:
    s = rep["summary"]
    print("\n=== SPXW 0DTE hold-to-expiry study ===")
    print(f"sessions            : {s['n_days']} ({s['n_test_days']} tested walk-forward)")
    print(f"decisions           : {s['n_rows']} ({s['n_rows_tested']} tested)")
    print(f"settlement source   : {s['settle_sources']}")
    print(f"realized / implied  : mean {s['mean_ratio_realized_over_implied']:.2f}, "
          f"median {s['median_ratio']:.2f}   (<1 = options overpriced the rest of the day)")
    print(f"straddle spread     : {s['median_straddle_spread_pct']*100:.2f}% of mid (median)")
    print(f"fly credit (median) : ${s['median_fly_credit_$']:.0f} per 1-lot; "
          f"fees ${s['fee_per_contract_$']:.2f}/leg x 4")
    print(f"sizing              : {s['sizing']}")
    print(f"\n{'strategy':24s} {'trades':>6s} {'total $':>9s} {'$/day':>8s} "
          f"{'$/trade':>8s} {'ret/risk':>8s} {'hit':>5s} {'worst day':>10s} "
          f"{'max DD':>9s}  sharpe [95% CI]")
    for name, r in rep["strategies"].items():
        ci = r["sharpe_ci"]
        print(f"{name:24s} {r['n_trades']:6d} {r['total_pnl']:9.0f} "
              f"{r['mean_daily_pnl']:8.1f} {r['mean_pnl_per_trade']:8.1f} "
              f"{r['mean_return_on_risk']*100:7.1f}% "
              f"{(r['hit_rate'] if np.isfinite(r['hit_rate']) else 0)*100:4.0f}% "
              f"{r['worst_day']:10.0f} {r['max_drawdown']:9.0f}  "
              f"{r['sharpe']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    print(f"\nforecast IC (pred vs realized/implied): "
          + ", ".join(f"{k} {v:+.3f}" for k, v in rep["forecast_ic"].items()))
    print(f"best non-ML baseline: {rep['best_baseline']}")
    for m, c in rep["comparisons"].items():
        for k, v in c.items():
            print(f"  {m:24s} {k:24s} {v['mean']:+8.1f} $/day  "
                  f"CI [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")
    extra = {m: c for m, c in rep["comparisons"].items() if "+" in m}
    if extra:
        print("\ndoes the extra data help? (each model vs the same model on base features)")
        for m, c in extra.items():
            k = "vs_" + m.split("+")[0]
            if k in c:
                v = c[k]
                print(f"  {m:28s} {v['mean']:+8.1f} $/day  CI [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]")
    md = rep.get("modes", {}).get("fixed6", {})
    dk = [k for k in md.get("strategies", {}) if k.startswith("danger_")]
    if dk:
        print(f"\n=== OBJECTIVE B - danger detector (version 1 times; danger = move >= "
              f"{DANGER_RATIO:.2f}x priced, base rate {md.get('danger_base_rate', float('nan')):.0%}) ===")
        print("  AUC (0.5 = coin flip): " + ", ".join(f"{k} {v:.3f}" for k, v in md.get("danger_auc", {}).items()))
        ref = md["strategies"]["always_short_fly"]
        print(f"  {'always_short_fly (benchmark)':38s} total {ref['total_pnl']:9.0f}  worst day {ref['worst_day']:7.0f}  "
              f"max DD {ref['max_drawdown']:9.0f}  sharpe {ref['sharpe']:+.2f}")
        for k in dk:
            r = md["strategies"][k]
            c = md["comparisons"].get(k, {}).get("vs_always_short_fly", {})
            print(f"  {k:38s} total {r['total_pnl']:9.0f}  worst day {r['worst_day']:7.0f}  "
                  f"max DD {r['max_drawdown']:9.0f}  sharpe {r['sharpe']:+.2f}  "
                  f"vs always-sell {c.get('mean', float('nan')):+7.1f}/day")
    for mode, md_ in rep.get("modes", {}).items():
        keys = md_.get("items_1_3") or []
        if not keys:
            continue
        print(f"\n=== ITEMS 1-3 ({mode}): exits (TP {EXIT_TP:.0%}/SL {EXIT_SL:.0%}), A x B, "
              f"no shorts {POST_EVENT_MIN} min after events ===")
        for k in keys:
            r = md_["strategies"][k]
            c = md_["comparisons"].get(k, {})
            vs = c.get("vs_" + md_["reference"], {})
            print(f"  {k:44s} total {r['total_pnl']:9.0f}  worst day {r['worst_day']:7.0f}  "
                  f"max DD {r['max_drawdown']:9.0f}  sharpe {r['sharpe']:+.2f} "
                  f"[{r['sharpe_ci'][0]:+.2f},{r['sharpe_ci'][1]:+.2f}]  "
                  f"vs {md_['reference']} {vs.get('mean', float('nan')):+7.1f}/day "
                  f"[{vs.get('ci', [np.nan, np.nan])[0]:+.0f},{vs.get('ci', [np.nan, np.nan])[1]:+.0f}]")
    titles = {"every15": "VERSION 2 - check every 15 min, 1 trade/day on a strong signal",
              "events": "VERSION 3 - decide after scheduled events and market shocks"}
    for mode, title in titles.items():
        md = rep.get("modes", {}).get(mode)
        if not md:
            continue
        print(f"\n=== {title} ({md['n_rows_tested']} tested decision points) ===")
        for name, r in md["strategies"].items():
            ci = r["sharpe_ci"]
            print(f"{name:34s} {r['n_trades']:5d} trades  total {r['total_pnl']:9.0f}  "
                  f"$/day {r['mean_daily_pnl']:7.1f}  hit {(r['hit_rate'] if np.isfinite(r['hit_rate']) else 0)*100:3.0f}%  "
                  f"worst day {r['worst_day']:7.0f}  sharpe {r['sharpe']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        print(f"  vs {md['reference']} / best baseline ({md['best_baseline']}):")
        for m, c in md["comparisons"].items():
            parts = [f"{k[3:]} {v['mean']:+.1f} [{v['ci'][0]:+.0f},{v['ci'][1]:+.0f}]" for k, v in c.items()]
            print(f"  {m:34s} " + " | ".join(parts))
    imp = rep.get("importance")
    if imp:
        print(f"\n=== WHAT HELPS - layer by layer (full LightGBM, version 1; full IC {imp['full_ic']:+.3f}) ===")
        print("  drop one layer from the full model (positive = the layer was helping):")
        for L, v in imp["drop_one"].items():
            pl = v["pnl_lost_per_day"]
            print(f"    {L:10s} {v['n_features']:3d} feats   IC drop {v['ic_drop']:+.3f}   "
                  f"$/day lost {pl['mean']:+8.1f}  CI [{pl['ci'][0]:+.0f}, {pl['ci'][1]:+.0f}]")
        print("  share of the model's splits (LightGBM gain), by layer:")
        print("    " + "  ".join(f"{L} {v:.0%}" for L, v in imp["gain_by_layer"].items()))
        print("  top inputs: " + ", ".join(f"{k} {v:.1%}" for k, v in list(imp["gain_top"].items())[:12]))
    from .portfolio import print_money
    for mode, md in rep.get("modes", {}).items():
        if md.get("money"):
            print_money(md["money"], {"fixed6": "version 1 (6 fixed times)",
                                      "every15": "version 2 (every 15 min)",
                                      "events": "version 3 (events)"}[mode])
    n_var = sum(1 for k in rep["strategies"] if k.startswith(("ml_", "seq_")))
    if n_var > 2:
        print(f"\ncaution: {n_var} model variants were tried. At 95% confidence about "
              f"{0.05 * n_var:.1f} of their CIs\n  would exclude zero by luck alone, so a lone "
              f"'significant' variant is exploratory.\n  Only the pre-registered "
              f"{rep['primary_model']} decides the verdict; a variant needs to repeat on new data.")
    print(f"\nVERDICT ({rep['primary_model']}, pre-registered): {rep['verdict']}")


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--min-train-days", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--base-only", action="store_true",
                    help="skip calendar / context / social / nlp features")
    ap.add_argument("--seq", action="store_true",
                    help="also run the minute-sequence transformer (slower)")
    ap.add_argument("--fixed6-only", action="store_true",
                    help="only the 6 fixed times (no 15-min grid, events or shocks)")
    ap.add_argument("--holdout-from", default=None,
                    help="drop every session on/after this date (locked final exam)")
    ap.add_argument("--report-from", default=None)
    ap.add_argument("--report-to", default=None)
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--sets", nargs="+", default=None,
                    help="keep only these feature sets (names as printed, e.g. +cal +cal+ctx)")
    ap.add_argument("--no-importance", action="store_true")
    ap.add_argument("--tag", default=None, help="write study-TAG.json / decisions-TAG.parquet")
    a = ap.parse_args()
    global REFIT_EVERY
    REFIT_EVERY = max(1, a.refit_every)
    if a.selftest:
        return 0 if selftest() else 1
    chains, settles = load_sessions(Path(a.data_dir))
    if a.holdout_from:
        cut = pd.Timestamp(a.holdout_from)
        chains = {d: c for d, c in chains.items() if d < cut}
        print(f"[holdout] {cut:%Y-%m-%d} onward is locked away; {len(chains)} sessions used")
    if not chains:
        print(f"no sessions in {a.data_dir}; run odte.data.fetch_0dte first")
        return 1
    if a.fixed6_only:
        ds = build_dataset(chains, settles, early_closes=load_early_closes())
    else:
        event_times = {}
        try:
            from odte.data import calendar as cal
            ev = cal.load()
            timed = ev[ev["session_time"].notna()]
            for d, g in timed.groupby("session"):
                event_times[pd.Timestamp(d)] = sorted(set(g["session_time"]))
        except FileNotFoundError:
            print("[events] no calendar built; scheduled-event decisions skipped")
        ds = build_dataset(chains, settles, times=GRID15, event_times=event_times,
                           shocks=True, early_closes=load_early_closes())
        print(f"[rows] {ds['trigger'].value_counts().to_dict()}")
    sets, skipped = {}, {}
    if not a.base_only:
        from .extra_features import attach
        ds, sets, skipped = attach(ds)
        sets.pop("base", None)
        for k, v in skipped.items():
            print(f"[features] {k}: {v}")
    if "min_since_event" in ds.columns and ds["min_since_event"].notna().any():
        # Post-event features as the last cumulative layer.
        top = max(sets, key=lambda k: len(sets[k])) if sets else None
        sets[(top or "") + "+evt"] = (sets[top] if top else list(FEATURES)) + EVENT_FEATURES
    extra = {}
    if a.seq:
        from .seq_model import minute_paths, walk_forward_seq
        print("[seq] building minute paths ...", flush=True)
        paths = {pd.Timestamp(d): minute_paths(c) for d, c in chains.items()}
        ds = ds.sort_values(["day", "t"]).reset_index(drop=True)
        widest_name = max(sets, key=lambda k: len(sets[k])) if sets else None
        static = sets[widest_name] if widest_name else list(FEATURES)
        label = "seq_transformer" + (widest_name or "")
        extra[label] = walk_forward_seq(ds, paths, static, min_train_days=a.min_train_days)
    if a.sets:
        sets = {k: v for k, v in sets.items() if k in set(a.sets)}
        print(f"[sets] {list(sets)}")
    layers = {}
    ordered = sorted(sets.items(), key=lambda kv: len(kv[1]))
    prev = list(FEATURES)
    for name, cols in ordered:
        new_cols = [c for c in cols if c not in set(prev)]
        if new_cols:
            layers[name.split("+")[-1]] = new_cols
        prev = cols
    rw = (a.report_from, a.report_to) if (a.report_from or a.report_to) else None
    dsets = None
    if a.sets:
        dsets = {"": list(FEATURES)}
        dsets.update({k: v for k, v in sets.items()})
    rep = run_study(ds, min_train_days=a.min_train_days, margin=a.margin,
                    feature_sets=sets, extra_preds=extra,
                    layers=(None if a.no_importance else (layers or None)),
                    report_window=rw, danger_sets=dsets)
    rep["skipped_features"] = skipped
    print_report(rep)
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        name = f"study-{a.tag}.json" if a.tag else "study.json"
        (OUT / name).write_text(json.dumps(rep, indent=2, default=str))
        ds.to_parquet(OUT / (f"decisions-{a.tag}.parquet" if a.tag else "decisions.parquet"))
        print(f"\nwrote {OUT/name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
