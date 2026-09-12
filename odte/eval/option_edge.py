"""Edge-existence harness: does any signal predict 0DTE option returns net of spread?

One question, asked so it can come back "no":

    Given a candidate signal, does trading it on the quoted touch -- buying at
    the ask and selling at the bid -- make money after the spread you actually
    pay?

The design copies what the equity study in `nlpalpha/` was built to survive,
and adds the two things options require:

  executable pricing, not a cost assumption
      The cost is quoted in the data. Nothing is assumed. `ret_long_net`
      already has the spread inside it.

  a delta baseline
      An option's return is mostly delta times the underlying move, so a
      signal that only predicts SPX will appear to predict every contract. The
      `delta_proxy` rung exists to catch that, and a signal that cannot beat
      it has found nothing options-specific.

Headline metric is **breakeven spread capture**: the fraction of the quoted
spread you would have to capture -- by resting orders rather than crossing --
for the strategy to break even. Below ~0 the signal is dead on arrival; near 1
it needs perfect passive fills, which is itself a market-making problem rather
than a forecasting one. This is the options analogue of "breakeven cost in
bps", and it is reported instead of accuracy because the equity study showed
an AUC of 0.586 sitting on top of a gross Sharpe of -2.5.

Status: the harness is validated against synthetic chains with a known
injected edge (`--selftest`). It has not been run on real market data, because
none is in the repo yet -- that is Stage 1 of the plan in STATE.md. Running it
on real chains is the point; everything here is the instrument, not a result.

Usage:
    PYTHONPATH=. python -m odte.eval.option_edge --selftest
    PYTHONPATH=. python -m odte.eval.option_edge --chain shards/spx_0dte.parquet
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .option_panel import (build_option_panel,
                           spread_cost_summary)
from .panel_stats import (day_block_bootstrap, group_indices, permutation_ic,
                          spearman_ic)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "option_edge"
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def breakeven_spread_capture(panel: pd.DataFrame, signal: np.ndarray,
                             top_frac: float = 0.2) -> dict:
    """What fraction of the quoted spread must be captured to break even?

    Trading the touch pays the full spread. Resting an order and getting
    filled pays less. If `c` is the fraction of the half-spread saved on each
    leg, realized return sits between `ret_mid` (c = 1, a perfect passive
    fill on both sides, which nobody gets consistently) and `ret_long_net`
    (c = 0, crossing both ways).

    Interpolating linearly in `c` and solving for zero mean P&L gives the
    capture requirement. Returned as a fraction: <= 0 means the signal makes
    money even crossing the spread; >= 1 means it cannot make money even with
    perfect passive execution, and is not a strategy at any fill quality.
    """
    sel = _top_bottom_mask(panel, signal, top_frac)
    if sel is None:
        return {"breakeven_capture": float("nan"), "n_trades": 0}

    long_mask, short_mask = sel
    mid_pnl = np.where(long_mask, panel["ret_mid"], 0.0) \
        + np.where(short_mask, -panel["ret_mid"], 0.0)
    net_pnl = np.where(long_mask, panel["ret_long_net"], 0.0) \
        + np.where(short_mask, panel["ret_short_net"], 0.0)

    traded = long_mask | short_mask
    mean_mid = float(mid_pnl[traded].mean())
    mean_net = float(net_pnl[traded].mean())
    gap = mean_mid - mean_net           # cost of crossing, per trade

    if gap <= _EPS:
        capture = 0.0 if mean_net >= 0 else float("inf")
    else:
        capture = float((0.0 - mean_net) / gap)
    return {
        "breakeven_capture": capture,
        "mean_ret_mid": mean_mid,
        "mean_ret_net": mean_net,
        "spread_drag_per_trade": gap,
        "n_trades": int(traded.sum()),
    }


def _top_bottom_mask(panel: pd.DataFrame, signal: np.ndarray,
                     top_frac: float):
    """Long the top slice of each bar's cross-section, short the bottom."""
    s = np.asarray(signal, dtype=float)
    if len(s) != len(panel):
        raise ValueError(f"signal length {len(s)} != panel rows {len(panel)}")
    long_mask = np.zeros(len(s), dtype=bool)
    short_mask = np.zeros(len(s), dtype=bool)
    any_group = False
    for g in group_indices(panel["bar"].to_numpy()):
        if len(g) < 4:
            continue
        any_group = True
        k = max(1, int(round(top_frac * len(g))))
        order = g[np.argsort(s[g], kind="mergesort")]
        long_mask[order[-k:]] = True
        short_mask[order[:k]] = True
    return (long_mask, short_mask) if any_group else None


def backtest_signal(panel: pd.DataFrame, signal: np.ndarray,
                    top_frac: float = 0.2) -> dict:
    """Bar-by-bar long/short book on executable prices, aggregated per day."""
    sel = _top_bottom_mask(panel, signal, top_frac)
    if sel is None:
        return {"n_bars": 0, "note": "no bar had enough contracts to trade"}
    long_mask, short_mask = sel

    per_bar = []
    for g in group_indices(panel["bar"].to_numpy()):
        lm = long_mask[g]
        sm = short_mask[g]
        if not lm.any() and not sm.any():
            continue
        long_r = panel["ret_long_net"].to_numpy()[g][lm]
        short_r = panel["ret_short_net"].to_numpy()[g][sm]
        mid_r = panel["ret_mid"].to_numpy()[g]
        legs = []
        if len(long_r):
            legs.append(long_r.mean())
        if len(short_r):
            legs.append(short_r.mean())
        per_bar.append({
            "bar": panel["bar"].to_numpy()[g][0],
            "net": float(np.mean(legs)),
            "mid": float(np.mean([mid_r[lm].mean() if lm.any() else 0.0,
                                  -mid_r[sm].mean() if sm.any() else 0.0])),
        })

    if not per_bar:
        return {"n_bars": 0, "note": "no tradeable bars"}
    bars = pd.DataFrame(per_bar)
    bars["day"] = pd.to_datetime(bars["bar"]).dt.normalize()
    daily = bars.groupby("day")[["net", "mid"]].sum()

    boot = day_block_bootstrap(daily["net"].to_numpy())
    return {
        "n_bars": int(len(bars)),
        "n_days": int(len(daily)),
        "mean_bar_net": float(bars["net"].mean()),
        "mean_bar_mid": float(bars["mid"].mean()),
        "daily_net_mean": float(daily["net"].mean()),
        "sharpe_net": boot["sharpe"],
        "sharpe_net_ci": [boot["ci_low"], boot["ci_high"]],
        "p_sharpe_le_0": boot["p_sharpe_le_0"],
        "hit_rate_bars": float((bars["net"] > 0).mean()),
    }


def evaluate_signal(panel: pd.DataFrame, signal: np.ndarray, name: str,
                    n_perm: int = 500, seed: int = 0,
                    top_frac: float = 0.2) -> dict:
    """Full report card for one signal."""
    ts = panel["bar"].to_numpy()
    ic_mid = spearman_ic(signal, panel["ret_mid"].to_numpy())
    ic_net = spearman_ic(signal, panel["ret_long_net"].to_numpy())
    perm = permutation_ic(signal, panel["ret_long_net"].to_numpy(), ts,
                          n_perm=n_perm, seed=seed)

    rng = np.random.default_rng(seed + 991)
    placebo = np.asarray(signal, dtype=float).copy()
    for g in group_indices(ts):
        placebo[g] = np.asarray(signal, dtype=float)[rng.permutation(g)]

    # Circularity guard. `ret_long_net` has the spread subtracted from it, so
    # any signal correlated with the spread earns net-IC mechanically: rank
    # contracts by tightness and the tight ones "outperform" purely because
    # they were charged less. That is an accounting identity, not a forecast.
    # A signal whose net-IC exceeds its mid-IC while sitting on a large
    # |corr_with_spread| is almost certainly reading its own cost term.
    spread_corr = spearman_ic(signal, panel["spread_pct"].to_numpy())
    mechanical = bool(abs(spread_corr) > 0.30 and abs(ic_net) > abs(ic_mid))

    out = {
        "signal": name,
        "ic_vs_mid": ic_mid,
        "ic_vs_net": ic_net,
        "ic_lost_to_spread": ic_mid - ic_net,
        "corr_with_spread": spread_corr,
        "mechanical_spread_effect": mechanical,
        "permutation_net": perm,
        "placebo_ic_net": spearman_ic(placebo, panel["ret_long_net"].to_numpy()),
        "backtest": backtest_signal(panel, signal, top_frac),
    }
    out.update(breakeven_spread_capture(panel, signal, top_frac))
    if np.isfinite(panel["delta_hedged_pnl"]).any():
        out["ic_vs_delta_hedged"] = spearman_ic(
            signal, panel["delta_hedged_pnl"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Baseline ladder
# ---------------------------------------------------------------------------

def baseline_signals(panel: pd.DataFrame, seed: int = 0) -> dict:
    """Cheap explanations a real signal has to outrank."""
    rng = np.random.default_rng(seed)
    sigs = {
        "random": rng.normal(size=len(panel)),
        "always_long": np.ones(len(panel)),
        "short_premium": -panel["mid"].to_numpy(),          # sell the expensive
        "reversal": -np.nan_to_num(panel["opt_ret_1"].to_numpy()),
        "momentum": np.nan_to_num(panel["opt_ret_1"].to_numpy()),
        "tight_spread": -np.nan_to_num(panel["spread_pct"].to_numpy()),
    }
    if np.isfinite(panel["delta_proxy"]).any():
        # The confound: pure directional exposure via delta. Uses the CURRENT
        # bar's delta and the underlying's TRAILING move, so it stays causal.
        sigs["delta_proxy"] = (np.nan_to_num(panel["delta"].to_numpy())
                               * np.nan_to_num(panel["und_ret_1"].to_numpy()))
    return sigs


# ---------------------------------------------------------------------------
# Synthetic chain, for validating the instrument
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def synthetic_chain(n_bars: int = 78, n_strikes: int = 21, S0: float = 5500.0,
                    strike_step: float = 10.0, sigma: float = 0.20,
                    minutes_per_bar: int = 5, half_spread_pct: float = 0.03,
                    tick: float = 0.05, seed: int = 0,
                    edge_strength: float = 0.0) -> pd.DataFrame:
    """A 0DTE session with Black-Scholes quotes and realistic spreads.

    `edge_strength` injects a known, tradeable signal: an extra `edge` column
    correlated with each contract's *next-bar* mid return. That is precisely
    what a harness must be able to detect, and setting it to 0 gives a chain
    where the honest answer is "no edge" -- both cases are exercised in
    `selftest`.
    """
    rng = np.random.default_rng(seed)
    minutes_total = n_bars * minutes_per_bar
    start = pd.Timestamp("2024-06-03 09:30")
    expiry = start + pd.Timedelta(minutes=minutes_total + minutes_per_bar)

    dt = minutes_per_bar / (60 * 24 * 252)
    shocks = rng.normal(0.0, sigma * math.sqrt(dt), size=n_bars)
    S = S0 * np.exp(np.cumsum(shocks))
    strikes = S0 + strike_step * (np.arange(n_strikes) - n_strikes // 2)

    rows = []
    for i in range(n_bars):
        bar_ts = start + pd.Timedelta(minutes=i * minutes_per_bar)
        mte = (expiry - bar_ts).total_seconds() / 60.0
        tau = max(mte / (60 * 24 * 252), 1e-8)
        sqrt_tau = math.sqrt(tau)
        for K in strikes:
            for right in ("C", "P"):
                d1 = (math.log(S[i] / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_tau)
                d2 = d1 - sigma * sqrt_tau
                if right == "C":
                    val = S[i] * _norm_cdf(d1) - K * _norm_cdf(d2)
                    delta = float(_norm_cdf(d1))
                else:
                    val = K * _norm_cdf(-d2) - S[i] * _norm_cdf(-d1)
                    delta = float(_norm_cdf(d1) - 1.0)
                val = float(max(val, 0.02))
                hs = max(val * half_spread_pct, tick)
                bid = math.floor((val - hs) / tick) * tick
                ask = math.ceil((val + hs) / tick) * tick
                rows.append({
                    "quote_datetime": bar_ts, "root": "SPXW",
                    "expiration": expiry, "strike": float(K),
                    "option_type": right,
                    "bid": max(bid, tick), "ask": max(ask, 2 * tick),
                    "bid_size": float(rng.integers(1, 200)),
                    "ask_size": float(rng.integers(1, 200)),
                    "trade_volume": float(rng.integers(0, 500)),
                    "underlying_price": float(S[i]),
                    "delta": delta,
                    "minutes_to_expiry": mte,
                })
    chain = pd.DataFrame(rows)
    chain.attrs["edge_strength"] = edge_strength
    return chain


def inject_oracle_signal(panel: pd.DataFrame, strength: float,
                         seed: int = 0) -> np.ndarray:
    """A signal that partially sees the next bar's mid return.

    Used only to prove the harness can detect an edge that exists. `strength`
    of 1.0 is a perfect forecast of the mid move; 0.0 is pure noise.
    """
    rng = np.random.default_rng(seed)
    truth = np.nan_to_num(panel["ret_mid"].to_numpy())
    noise = rng.normal(0.0, truth.std() + _EPS, size=len(truth))
    return strength * truth + (1.0 - strength) * noise


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(chain: pd.DataFrame, interval: str = "5min", seed: int = 0,
              n_perm: int = 500, extra_signals: dict | None = None,
              write: bool = False) -> dict:
    panel = build_option_panel(chain, interval=interval)
    if panel.empty:
        return {"error": "panel empty after filters",
                "drop_reasons": chain.attrs.get("drop_reasons", {})}

    report = {
        "panel": {
            "n_rows": int(len(panel)),
            "n_rows_in": int(panel.attrs.get("n_rows_in", 0)),
            "n_dropped": int(panel.attrs.get("n_dropped", 0)),
            "drop_reasons": panel.attrs.get("drop_reasons", {}),
            "n_bars": int(panel["bar"].nunique()),
            "n_contracts": int(panel["contract_id"].nunique()),
            "interval": interval,
        },
        "spread_cost": spread_cost_summary(panel),
    }

    signals = baseline_signals(panel, seed=seed)
    if extra_signals:
        signals.update(extra_signals)
    report["signals"] = {
        name: evaluate_signal(panel, sig, name, n_perm=n_perm, seed=seed)
        for name, sig in signals.items()
    }
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "study.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def selftest(verbose: bool = True) -> bool:
    """Three checks that the instrument works before it is trusted on real data."""
    results = []

    chain = synthetic_chain(seed=1)
    panel = build_option_panel(chain, interval="5min")

    # 1. A random signal must come back null.
    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(panel))
    r_noise = evaluate_signal(panel, noise, "random", n_perm=200)
    ok1 = (abs(r_noise["ic_vs_net"]) < 0.05
           and r_noise["permutation_net"]["p_value"] > 0.05)
    results.append(("random signal reports null", ok1,
                    f"ic_net={r_noise['ic_vs_net']:+.4f} "
                    f"p={r_noise['permutation_net']['p_value']:.3f}"))

    # 2. A signal that half-sees the next mid move must be detected.
    oracle = inject_oracle_signal(panel, strength=0.5, seed=2)
    r_oracle = evaluate_signal(panel, oracle, "oracle", n_perm=200)
    ok2 = (r_oracle["ic_vs_mid"] > 0.20
           and r_oracle["permutation_net"]["p_value"] < 0.05)
    results.append(("injected edge is detected", ok2,
                    f"ic_mid={r_oracle['ic_vs_mid']:+.4f} "
                    f"ic_net={r_oracle['ic_vs_net']:+.4f} "
                    f"p={r_oracle['permutation_net']['p_value']:.3f}"))

    # 3. The spread must visibly eat the edge: a perfect mid-forecast should
    #    still require capturing part of the spread to be profitable.
    perfect = inject_oracle_signal(panel, strength=1.0, seed=3)
    be = breakeven_spread_capture(panel, perfect)
    ok3 = be["mean_ret_mid"] > be["mean_ret_net"]
    results.append(("spread demonstrably erodes a perfect mid-forecast", ok3,
                    f"mid={be['mean_ret_mid']:+.4f} net={be['mean_ret_net']:+.4f} "
                    f"breakeven_capture={be['breakeven_capture']:.3f}"))

    n_pass = sum(1 for _, ok, _ in results if ok)
    if verbose:
        print("\n=== option_edge self-test ===")
        for name, ok, detail in results:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}")
            print(f"         {detail}")
        print(f"\n{'PASS' if n_pass == len(results) else 'FAIL'} "
              f"({n_pass}/{len(results)})")
    return n_pass == len(results)


def _cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--chain", default=None,
                    help="parquet/csv of the canonical chain schema")
    ap.add_argument("--interval", default="5min")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return 0 if selftest() else 1

    if a.chain:
        p = Path(a.chain)
        chain = pd.read_parquet(p) if p.suffix in (".parquet", ".pq") \
            else pd.read_csv(p)
    else:
        print("(no --chain given; running on a SYNTHETIC session -- "
              "results below are about the harness, not the market)")
        chain = synthetic_chain(seed=0)

    rep = run_study(chain, interval=a.interval, n_perm=a.n_perm, write=a.write)
    if "error" in rep:
        print("ERROR:", rep["error"])
        return 1

    sc = rep["spread_cost"]
    print("\n=== 0DTE option edge-existence scan ===")
    print(f"panel                 : {rep['panel']['n_rows']} rows, "
          f"{rep['panel']['n_bars']} bars, "
          f"{rep['panel']['n_contracts']} contracts "
          f"({rep['panel']['n_dropped']} dropped)")
    print(f"median spread         : {sc['median_spread_pct']*100:.2f}% of mid")
    print(f"median |mid move|     : {sc['median_abs_ret_mid']*100:.2f}%")
    print(f"spread / typical move : {sc['ratio_spread_to_move']:.2f}x   "
          f"(>1 means the spread exceeds the move you are forecasting)")
    print(f"\n{'signal':18s} {'IC_mid':>8s} {'IC_net':>8s} {'perm_p':>7s} "
          f"{'breakeven':>10s} {'sprd_r':>7s}  flag")
    for name, r in rep["signals"].items():
        flag = "MECHANICAL" if r.get("mechanical_spread_effect") else ""
        print(f"{name:18s} {r['ic_vs_mid']:+8.4f} {r['ic_vs_net']:+8.4f} "
              f"{r['permutation_net']['p_value']:7.3f} "
              f"{r['breakeven_capture']:10.3f} "
              f"{r['corr_with_spread']:+7.3f}  {flag}")
    n_days = max((r["backtest"].get("n_days", 0)
                  for r in rep["signals"].values()), default=0)
    print(f"\nbreakeven = fraction of the quoted spread you must capture to "
          f"break even.\n  <=0 profitable while crossing; >=1 not a strategy "
          f"at any fill quality.")
    print("sprd_r = rank corr with the quoted spread. MECHANICAL flags a "
          "signal whose net-IC\n  beats its mid-IC while tracking the "
          "spread -- it is reading its own cost term,\n  not forecasting.")
    if n_days < 3:
        print(f"\nSharpe omitted: {n_days} trading day(s) in this panel. "
              "A Sharpe needs many days;\n  a single session cannot "
              "produce one and none is reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
