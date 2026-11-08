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
    for side in ("bid", "ask"):
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


def session_rows(grid: dict, F: pd.Series, settle: float,
                 prev_settle: float = float("nan"),
                 prev_ratio: float = float("nan"),
                 times=DECISION_TIMES, wing_mult: float = 1.5,
                 fee: float = FEE_PER_CONTRACT) -> list:
    """One row per decision time: causal features, then settlement P&L."""
    day = grid["bid"].index[0].normalize()
    close_t = day + pd.Timedelta(hours=16)
    valid_F = F.dropna()
    if valid_F.empty:
        return []
    F_open = float(valid_F.iloc[0])
    rows = []
    for hhmm in times:
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

        move = abs(settle - K)
        intrinsic = min(move, W) if np.isfinite(W) else float("nan")
        fly_ok = np.isfinite(W) and credit > 0 and debit > 0
        fees4 = 4 * fee
        rows.append({
            "day": day, "t": t, "hhmm": hhmm, "F": Ft, "K": K, "W": W,
            "settle": settle,
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
        })
    return rows


def build_dataset(chains: dict, settlements: dict, **kw) -> pd.DataFrame:
    """Decision rows across sessions, in date order.

    `chains` maps session date -> that session's quotes; `settlements` maps
    date -> SPX official close. A session with no settlement falls back to
    the last implied forward of the day and is flagged.
    """
    rows = []
    prev_settle, prev_ratio = float("nan"), float("nan")
    for day in sorted(chains):
        grid = minute_grid(chains[day])
        F = implied_forward(grid)
        settle = settlements.get(pd.Timestamp(day).normalize())
        src = "spx_close"
        if settle is None or not np.isfinite(settle):
            settle = float(F.dropna().iloc[-1]) if F.notna().any() else float("nan")
            src = "last_forward"
        day_rows = session_rows(grid, F, settle, prev_settle, prev_ratio, **kw)
        for r in day_rows:
            r["settle_source"] = src
        rows.extend(day_rows)
        if day_rows:
            prev_ratio = day_rows[0]["ratio"]
        prev_settle = settle
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


def walk_forward(ds: pd.DataFrame, model: str, min_train_days: int = 8,
                 margin: float = 0.15, shuffle_labels_seed=None):
    """Predictions and actions for every row on a day with enough history.

    Rows on the first `min_train_days` sessions get NaN (not tested). With
    `shuffle_labels_seed` set, training targets are permuted -- a placebo
    that must not beat the baselines.
    """
    days = np.array(sorted(ds["day"].unique()))
    pred = np.full(len(ds), np.nan)
    rng = np.random.default_rng(shuffle_labels_seed)
    y_all = ds["ratio"].clip(upper=4.0).to_numpy()
    X_all = ds[FEATURES].to_numpy(dtype=float)
    dvals = ds["day"].to_numpy()
    for i, d in enumerate(days):
        if i < min_train_days:
            continue
        tr = (dvals < d) & np.isfinite(y_all)
        te = dvals == d
        y = y_all[tr]
        if shuffle_labels_seed is not None:
            y = rng.permutation(y)
        m = _make_model(model)
        m.fit(X_all[tr], y)
        pred[te] = m.predict(X_all[te])
    act = np.where(np.isfinite(pred), _rule(np.nan_to_num(pred, nan=1.0), margin), 0)
    return pred, act


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

def _row_pnl(ds: pd.DataFrame, act: np.ndarray, risk_budget=RISK_BUDGET):
    """Per-row P&L. With `risk_budget`, every trade is sized so its max loss
    equals the budget (fractional lots -- a research convention), which keeps
    a high-vol day from dominating the sample just because its fly is wider.
    With `risk_budget=None`, every trade is one lot."""
    s = ds["pnl_short_fly"].to_numpy()
    l = ds["pnl_long_fly"].to_numpy()
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
              risk_budget=RISK_BUDGET) -> dict:
    pnl, risk, a = _row_pnl(ds, act, risk_budget)
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


def run_study(ds: pd.DataFrame, min_train_days: int = 8, margin: float = 0.15,
              min_days_for_verdict: int = 40, seed: int = 0,
              risk_budget=RISK_BUDGET) -> dict:
    ds = ds.sort_values(["day", "t"]).reset_index(drop=True)
    days = sorted(ds["day"].unique())
    tested = ds["day"].isin(days[min_train_days:]).to_numpy()

    strategies = {}
    preds = {}
    for m in ("ml_ridge", "ml_lgbm"):
        preds[m], strategies[m] = walk_forward(ds, m, min_train_days, margin)
    _, strategies["placebo_shuffled_labels"] = walk_forward(
        ds, PRIMARY_MODEL, min_train_days, margin, shuffle_labels_seed=seed + 7)
    baselines = baseline_actions(ds, tested, seed=seed)
    strategies.update(baselines)

    res = {k: summarize(ds, v, tested, risk_budget) for k, v in strategies.items()}
    base_names = list(baselines)
    best_base = max((b for b in base_names if b != "flat"),
                    key=lambda b: res[b]["total_pnl"])
    comparisons = {}
    for m in ("ml_ridge", "ml_lgbm", "placebo_shuffled_labels"):
        comparisons[m] = {
            "vs_" + b: paired_diff(res[m]["_daily"], res[b]["_daily"], seed=seed)
            for b in ("always_short_fly", best_base)
        }

    t_rows = ds[tested]
    forecast_ic = {m: spearman_ic(preds[m][tested], t_rows["ratio"].to_numpy())
                   for m in preds}
    n_test_days = int(res[PRIMARY_MODEL]["n_test_days"])
    prim = res[PRIMARY_MODEL]
    beat_best = comparisons[PRIMARY_MODEL]["vs_" + best_base]
    if n_test_days < min_days_for_verdict:
        verdict = (f"INCONCLUSIVE: {n_test_days} test days "
                   f"(< {min_days_for_verdict}). Directional read only.")
    elif (beat_best["ci"][0] > 0 and prim["p_sharpe_le_0"] < 0.05):
        verdict = "PASS: beats the best baseline and Sharpe > 0 at 95%."
    else:
        verdict = "FAIL: no edge over the baselines that survives the CI."

    summary = {
        "n_days": len(days), "n_test_days": n_test_days,
        "n_rows": int(len(ds)), "n_rows_tested": int(tested.sum()),
        "settle_sources": ds["settle_source"].value_counts().to_dict()
        if "settle_source" in ds else {},
        "mean_ratio_realized_over_implied": float(ds["ratio"].mean()),
        "median_ratio": float(ds["ratio"].median()),
        "median_straddle_spread_pct": float(
            (ds["straddle_spread"] / ds["straddle_mid"]).median()),
        "median_fly_credit_$": float(ds["fly_credit"].median() * MULT),
        "fee_per_contract_$": FEE_PER_CONTRACT,
        "sizing": (f"each trade sized to ${risk_budget:.0f} max loss"
                   if risk_budget else "1 lot per trade"),
    }
    for v in res.values():
        v.pop("_daily", None)
    return {"summary": summary, "strategies": res, "comparisons": comparisons,
            "best_baseline": best_base, "forecast_ic": forecast_ic,
            "primary_model": PRIMARY_MODEL, "verdict": verdict,
            "params": {"min_train_days": min_train_days, "margin": margin,
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
    print(f"\nVERDICT ({rep['primary_model']}): {rep['verdict']}")


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--min-train-days", type=int, default=8)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    chains, settles = load_sessions(Path(a.data_dir))
    if not chains:
        print(f"no sessions in {a.data_dir}; run odte.data.fetch_0dte first")
        return 1
    ds = build_dataset(chains, settles)
    rep = run_study(ds, min_train_days=a.min_train_days, margin=a.margin)
    print_report(rep)
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "study.json").write_text(json.dumps(rep, indent=2, default=str))
        ds.to_parquet(OUT / "decisions.parquet")
        print(f"\nwrote {OUT/'study.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
