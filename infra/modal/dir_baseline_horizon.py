"""Multi-horizon directional baseline — does the signal survive at trading
horizons?

Sister to dir_baseline.py. Same LightGBM + 16-row × 7-feature history setup,
but instead of predicting the next single tick's direction, predicts the
*majority direction* over the next N rows for several N values:

    horizon = 1     → next-tick (matches dir_baseline.py — sanity check)
    horizon = 10    → ~half-second of OPRA observations
    horizon = 100   → ~5 seconds
    horizon = 1000  → ~50 seconds

Each row in the parquet is one OPRA observation snapshot (7 feature tokens).
Reading "horizon" as rows-ahead, not raw tokens-ahead.

The 70% next-tick result from dir_baseline.py is closer to data compression
than tradeable edge. Real spread cost on SPX 0DTE eats anything below ~52%.
This script tells us where on the horizon curve the edge collapses, which is
what determines whether a 524M directional-head retrain is worth $30-50.

Usage:
    modal run infra/modal/dir_baseline_horizon.py::run_horizon_baseline
"""
from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "tradefm-dir-baseline-horizon"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    .pip_install(
        "pyarrow>=15.0",
        "pandas>=2.2",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "lightgbm>=4.0",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

shard_volume = modal.Volume.from_name("tradefm-smoke-shards", create_if_missing=False)

app = modal.App(APP_NAME, image=image)

DEFAULT_EVAL_JOB_ID = "OPRA-20260424-DLKDHYSC6M"  # 2026-04-16
DEFAULT_HORIZONS = [1, 10, 100, 1000]


@app.function(
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/shards": shard_volume},
)
def run_horizon_baseline(
    eval_job_id: str = DEFAULT_EVAL_JOB_ID,
    history_rows: int = 16,
    max_train_shards: int = 50,
    max_eval_shards: int = 50,
    n_features: int = 7,
    horizons: list[int] = DEFAULT_HORIZONS,
):
    """Sweep LightGBM directional accuracy across N-row horizons.

    Returns a dict mapping horizon → (acc, z-score, n_eval). Decision rule:
    if any horizon ≥ 10 stays above 53%, signal survives at trading-relevant
    timescales and the 524M directional head retrain is justified.
    """
    import time
    import glob as _glob
    import numpy as np
    import pandas as pd

    base = Path("/shards/databento_reuse_packed")
    assert base.exists(), f"no packed shards at {base}"

    train_paths: list[Path] = []
    eval_paths: list[Path] = []
    for job_dir in sorted(base.iterdir()):
        if not job_dir.is_dir():
            continue
        shards = sorted(_glob.glob(f"{job_dir}/opra_*.parquet"))
        if not shards:
            continue
        if job_dir.name == eval_job_id:
            eval_paths.extend(shards[:max_eval_shards])
        else:
            train_paths.extend(shards[:max_train_shards // 4])
    print(f"[horizon] train shards: {len(train_paths)}  eval shards: {len(eval_paths)}",
          flush=True)
    print(f"[horizon] history_rows={history_rows}  horizons={horizons}",
          flush=True)

    def _load_rows(paths: list[Path], label: str) -> np.ndarray:
        all_arr: list[np.ndarray] = []
        for shard in paths:
            df = pd.read_parquet(shard, columns=["tokens"])
            for row_tokens in df["tokens"]:
                arr = np.asarray(row_tokens, dtype=np.int16)
                if arr.shape[0] == n_features:
                    all_arr.append(arr)
        if not all_arr:
            return np.zeros((0, n_features), dtype=np.int16)
        rows = np.stack(all_arr, axis=0)
        print(f"[horizon] {label}: {rows.shape[0]:,} rows loaded", flush=True)
        return rows

    print("[horizon] loading train rows…", flush=True)
    t0 = time.time()
    train_rows = _load_rows(train_paths, "train")
    print(f"[horizon] train load: {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    eval_rows = _load_rows(eval_paths, "eval")
    print(f"[horizon] eval load:  {time.time()-t0:.1f}s", flush=True)
    if len(train_rows) == 0 or len(eval_rows) == 0:
        raise RuntimeError("not enough rows for training/eval")

    # Single shared threshold = train ret-token median. Apply to both splits
    # so the per-horizon "up tick" definition is consistent.
    train_ret = train_rows[:, 0]
    threshold = float(np.median(train_ret))
    print(f"[horizon] shared ret-token threshold (train median) = {threshold:.2f}",
          flush=True)

    def _build_xy(rows: np.ndarray, horizon: int
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) for a given horizon.

        X = flattened (history_rows × n_features) tokens at [t, t+history_rows).
        y = 1 if majority of ret tokens in [t+history_rows, t+history_rows+horizon)
            are above `threshold`, else 0.

        For horizon=1 this reduces exactly to dir_baseline.py's next-tick target.
        For horizon>1 it is a path-direction proxy (count of up-ticks in window).
        """
        n_rows = rows.shape[0]
        n_samples = n_rows - history_rows - horizon + 1
        if n_samples <= 0:
            return np.zeros((0, history_rows * n_features), dtype=np.int16), np.zeros(0, dtype=np.int8)
        ret_tokens = rows[:, 0]
        above = (ret_tokens > threshold).astype(np.int32)
        # Cumulative sum lets us compute window sums in O(1).
        cum = np.concatenate([[0], np.cumsum(above)])
        starts = history_rows + np.arange(n_samples)
        ends = starts + horizon
        window_sums = cum[ends] - cum[starts]
        y = (window_sums * 2 > horizon).astype(np.int8)
        # Build X via vectorized stride trick (avoids 500k-iter Python loop).
        # rows has shape (n_rows, n_features). Sliding windows of length
        # history_rows over the row-axis.
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(rows, (history_rows, n_features))[:, 0, :, :]
        X = windows[:n_samples].reshape(n_samples, history_rows * n_features).astype(np.int16)
        return X, y

    # Subsample cap for training — 500k is plenty for diagnostic.
    cap = 500_000
    rng = np.random.RandomState(0)

    import lightgbm as lgb
    results: dict[int, dict] = {}

    for horizon in horizons:
        print()
        print(f"[horizon] ===== horizon={horizon} =====", flush=True)
        t0 = time.time()
        X_tr, y_tr = _build_xy(train_rows, horizon)
        X_ev, y_ev = _build_xy(eval_rows, horizon)
        print(f"[horizon] h={horizon}: train={X_tr.shape} eval={X_ev.shape} "
              f"(train pos rate={y_tr.mean()*100:.2f}%, eval pos rate={y_ev.mean()*100:.2f}%) "
              f"in {time.time()-t0:.1f}s", flush=True)
        if len(X_tr) == 0 or len(X_ev) == 0:
            print(f"[horizon] h={horizon}: insufficient data, skipping", flush=True)
            continue

        if len(X_tr) > cap:
            idx = rng.choice(len(X_tr), cap, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            print(f"[horizon] h={horizon}: subsampled train to {cap:,}", flush=True)

        t0 = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)],
                  callbacks=[lgb.log_evaluation(0)])
        fit_s = time.time() - t0

        pred = (model.predict_proba(X_ev)[:, 1] >= 0.5).astype(np.int8)
        acc = float(np.mean(pred == y_ev))
        base_rate = float(np.mean(y_ev))
        se = float(np.sqrt(0.5 * 0.5 / max(len(y_ev), 1)))
        z = (acc - 0.5) / max(se, 1e-9)
        print(f"[horizon] h={horizon}: acc={acc*100:.2f}%  z={z:+.1f}σ  "
              f"base={base_rate*100:.2f}%  n_eval={len(y_ev):,}  "
              f"fit={fit_s:.1f}s", flush=True)

        results[horizon] = {
            "accuracy": acc, "z_score": z, "n_eval": len(y_ev),
            "base_rate": base_rate,
        }

    # ---- Summary table ----
    print()
    print("[horizon] ============ SUMMARY ============", flush=True)
    print(f"[horizon] {'horizon':>8} | {'accuracy':>9} | {'z-score':>9} | "
          f"{'n_eval':>12} | verdict", flush=True)
    print(f"[horizon] {'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*12}-+-{'-'*30}", flush=True)
    for h in horizons:
        if h not in results:
            continue
        r = results[h]
        if r["accuracy"] >= 0.53:
            verdict = "TRADEABLE — fund retrain"
        elif r["accuracy"] >= 0.51:
            verdict = "marginal — borderline"
        else:
            verdict = "no edge — signal decayed"
        print(f"[horizon] {h:>8} | {r['accuracy']*100:>8.2f}% | "
              f"{r['z_score']:>+8.1f}σ | {r['n_eval']:>12,} | {verdict}",
              flush=True)
    print()

    survives = any(r["accuracy"] >= 0.53 for h, r in results.items() if h >= 10)
    print(f"[horizon] decision: {'GO' if survives else 'STOP'} on 524M "
          f"directional retrain — signal "
          f"{'survives' if survives else 'decays'} at trading horizons",
          flush=True)

    return {
        "results": {h: results[h] for h in results},
        "survives_trading_horizon": survives,
        "threshold": threshold,
    }


@app.local_entrypoint()
def main(history_rows: int = 16,
         max_train_shards: int = 50,
         max_eval_shards: int = 50):
    """Run the multi-horizon baseline. ~10 min wall-clock, ~$0.30-0.50 on
    Modal CPU. Add --horizons via Python list-literal if you want to
    sweep a different set."""
    run_horizon_baseline.remote(
        history_rows=history_rows,
        max_train_shards=max_train_shards,
        max_eval_shards=max_eval_shards,
    )
