"""Real-data signal-presence probe — does a tree see next-bar directional signal?

This is the locally-runnable cousin of infra/modal/dir_baseline.py. That
diagnostic needs paid OPRA shards on a Modal volume; this one runs the *same
question* on free intraday equity bars (yfinance) so the methodology can be
exercised end-to-end on real market data today.

It is NOT a substitute for the OPRA test — equities 1-minute bars are a
different instrument class and a coarser microstructure than 0DTE options. What
it does honestly establish:
  1. the corrected per-instrument feature pipeline runs on real data,
  2. a leakage-free LightGBM directional probe with a proper time-split, and
  3. an honest signal / no-signal verdict with a z-score and baselines.

Method (mirrors dir_baseline, made leakage-safe):
  - per ticker, build features from PAST bars only (lagged returns, realized
    vol, volume z-score, range, minute-of-day);
  - target = sign of the NEXT bar's log return (per ticker);
  - split by TIME (earliest 80% train, latest 20% eval) — never shuffle across
    the boundary;
  - report held-out accuracy vs 50% with a z-score, against two baselines
    (always-up, last-bar-momentum). Liquid equities at 1-minute horizon are
    near-efficient, so ~50% is the honest expected outcome.

Usage:
    PYTHONPATH=. python -m odte.eval.signal_probe
    PYTHONPATH=. python -m odte.eval.signal_probe --tickers AAPL MSFT NVDA --days 7
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "signal_probe"

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX",
    "JPM", "BAC", "XOM", "WMT", "INTC", "QCOM", "CSCO", "ORCL", "CRM",
    "SPY", "QQQ", "IWM", "DIA", "AVGO", "MU", "BABA",
]


def _per_ticker_features(close: pd.Series, vol: pd.Series,
                         high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """Features at bar t from info <= t; target = sign(return t -> t+1)."""
    logp = np.log(close.clip(lower=1e-9))
    r1 = logp.diff()
    feats = pd.DataFrame(index=close.index)
    feats["r1"] = r1
    feats["r5"] = logp.diff(5)
    feats["r15"] = logp.diff(15)
    feats["rvol15"] = r1.rolling(15).std()
    feats["vol_z"] = (vol - vol.rolling(30).mean()) / (vol.rolling(30).std() + 1e-9)
    feats["range"] = (high - low) / close.clip(lower=1e-9)
    feats["mom_sign"] = np.sign(r1.rolling(5).sum())
    feats["minute"] = (feats.index.view("int64") // 60_000_000_000) % 390
    # Target: next-bar direction (strictly future; this row's label uses t->t+1).
    feats["y"] = (r1.shift(-1) > 0).astype(float)
    feats["t"] = feats.index.view("int64")
    return feats.dropna()


def build_dataset(tickers, days: int):
    import yfinance as yf
    period = f"{min(days, 7)}d"
    raw = yf.download(tickers, period=period, interval="1m",
                      progress=False, auto_adjust=True, group_by="ticker")
    frames = []
    kept = []
    for tk in tickers:
        try:
            sub = raw[tk] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub.dropna()
            if len(sub) < 120:
                continue
            f = _per_ticker_features(sub["Close"], sub["Volume"],
                                     sub["High"], sub["Low"])
            f["ticker"] = tk
            frames.append(f)
            kept.append(tk)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("no usable ticker data fetched")
    df = pd.concat(frames, ignore_index=True)
    return df, kept


def run(tickers=None, days: int = 7, seed: int = 0) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    t0 = time.time()
    df, kept = build_dataset(tickers, days)
    feat_cols = ["r1", "r5", "r15", "rvol15", "vol_z", "range", "mom_sign", "minute"]

    # Time split: earliest 80% train, latest 20% eval (global time boundary so
    # no eval bar precedes any train bar -> no look-ahead).
    cut = np.quantile(df["t"].to_numpy(), 0.80)
    tr = df[df["t"] <= cut]
    ev = df[df["t"] > cut]
    X_tr, y_tr = tr[feat_cols].to_numpy(), tr["y"].to_numpy()
    X_ev, y_ev = ev[feat_cols].to_numpy(), ev["y"].to_numpy()

    import lightgbm as lgb
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                               num_leaves=31, min_child_samples=200,
                               reg_alpha=0.1, reg_lambda=0.1, subsample=0.8,
                               colsample_bytree=0.8, n_jobs=-1, random_state=seed,
                               verbose=-1)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_ev)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(np.mean(pred == y_ev))
    base_rate = float(np.mean(y_ev))
    n = len(y_ev)
    se = float(np.sqrt(0.25 / max(n, 1)))
    z = (acc - 0.5) / max(se, 1e-12)

    # Baselines.
    always_up = float(np.mean(y_ev == 1))
    mom = ev["mom_sign"].to_numpy()
    mom_pred = (mom > 0).astype(int)
    mom_acc = float(np.mean(mom_pred == y_ev))

    verdict = ("SIGNAL (>=53%)" if acc >= 0.53 else
               "marginal (51-53%)" if acc >= 0.51 else
               "NO extractable signal (~50%)")
    imp = dict(sorted(zip(feat_cols, model.feature_importances_.tolist()),
                      key=lambda kv: -kv[1]))

    result = {
        "tickers": kept, "n_tickers": len(kept), "days": days,
        "n_train": int(len(y_tr)), "n_eval": int(n),
        "eval_base_rate_up": round(base_rate, 4),
        "lgbm_accuracy": round(acc, 4),
        "z_score_vs_50": round(z, 2),
        "baseline_always_up_acc": round(always_up, 4),
        "baseline_momentum_acc": round(mom_acc, 4),
        "verdict": verdict,
        "feature_importance": imp,
        "wall_sec": round(time.time() - t0, 1),
        "note": ("Statistical predictability only; NOT tradeable edge — 1-min "
                 "equity moves below this magnitude vanish under spread+fees. "
                 "Equities proxy for the OPRA methodology, not the OPRA result."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--days", type=int, default=7)
    a = ap.parse_args()
    r = run(tickers=a.tickers, days=a.days)
    print("\n=== Signal-presence probe (real intraday equities) ===")
    print(f"tickers={r['n_tickers']}  train={r['n_train']:,}  eval={r['n_eval']:,}  "
          f"({r['wall_sec']}s)")
    print(f"eval base rate (up)     : {r['eval_base_rate_up']*100:.2f}%")
    print(f"LightGBM accuracy       : {r['lgbm_accuracy']*100:.2f}%   "
          f"(z = {r['z_score_vs_50']:+.2f} vs 50%)")
    print(f"baseline always-up      : {r['baseline_always_up_acc']*100:.2f}%")
    print(f"baseline momentum       : {r['baseline_momentum_acc']*100:.2f}%")
    print(f"VERDICT                 : {r['verdict']}")
    print(f"top features            : "
          f"{list(r['feature_importance'])[:4]}")
    print(f"\n{r['note']}")


if __name__ == "__main__":
    _cli()
