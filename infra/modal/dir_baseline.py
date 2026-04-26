"""Tiny directional-baseline diagnostic — does any model see signal?

Before spending $20 on a 524M retrain with a directional head, run this
$0 (CPU-only) sanity check: a small LightGBM classifier trained on
raw tokenized OPRA features predicting next-return direction.

If LightGBM gets ≥ 53% directional accuracy on the same held-out eval
day (2026-04-16) the 524M sees, the signal is *in the data* and our
524M's cross-entropy LM objective is the bottleneck → worth retraining
the 524M with a directional head. If LightGBM can't beat ~50%, the
signal isn't there at this token granularity / horizon and the 524M
retrain would be a waste.

Reads the same packed parquet shards used for 524M training. Runs on
Modal CPU (no GPU needed).

Usage:
    modal run infra/modal/dir_baseline.py::run_baseline
"""
from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "tradefm-dir-baseline"

# Reuse the existing image layer — has pandas, numpy, pyarrow already.
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


@app.function(
    cpu=8.0,
    memory=32768,
    timeout=3600,
    volumes={"/shards": shard_volume},
)
def run_baseline(
    eval_job_id: str = DEFAULT_EVAL_JOB_ID,
    history_rows: int = 16,
    max_train_shards: int = 50,
    max_eval_shards: int = 50,
    n_features: int = 7,
):
    """Train a LightGBM directional classifier on (history × 7 features)
    → next-return-direction. Reports held-out directional accuracy."""
    import time
    import glob as _glob
    import numpy as np
    import pandas as pd

    base = Path("/shards/databento_reuse_packed")
    assert base.exists(), f"no packed shards at {base}"

    # Identify train and eval shard paths the same way the 524M trainer
    # does — every job_id directory except the eval one is train.
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
    print(f"[baseline] train shards: {len(train_paths)}  eval shards: {len(eval_paths)}",
          flush=True)
    print(f"[baseline] history_rows={history_rows}  n_features={n_features}",
          flush=True)

    def _shards_to_xy(paths: list[Path], label: str) -> tuple[np.ndarray, np.ndarray]:
        """Load parquet shards into (X, y). X = flattened (history × n_features)
        token IDs. y = 1 if the next row's return token is in the upper half of
        return tokens (relative to the train-set median), else 0.

        Returns (X, y). The threshold is computed inside this fn from the
        return-token distribution and returned indirectly via y.
        """
        all_token_arrays: list[np.ndarray] = []
        for shard in paths:
            df = pd.read_parquet(shard, columns=["tokens"])
            for row_tokens in df["tokens"]:
                arr = np.asarray(row_tokens, dtype=np.int16)
                if arr.shape[0] != n_features:
                    continue
                all_token_arrays.append(arr)
        if not all_token_arrays:
            return np.zeros((0, history_rows * n_features)), np.zeros(0)
        # Stack into (n_rows, n_features) — feature 0 = ret.
        rows = np.stack(all_token_arrays, axis=0)
        n_rows = rows.shape[0]
        print(f"[baseline] {label}: {n_rows:,} rows loaded", flush=True)

        # Build sliding-window features: row[t-history_rows : t] flattened
        # → predict sign(row[t][0] - median).
        if n_rows <= history_rows + 1:
            return np.zeros((0, history_rows * n_features)), np.zeros(0)
        ret_tokens = rows[:, 0]  # feature 0 = return
        threshold = float(np.median(ret_tokens))
        print(f"[baseline] {label}: ret-token median = {threshold:.2f}",
              flush=True)
        # Targets: did row t's ret token land above threshold?
        y = (ret_tokens[history_rows:] > threshold).astype(np.int8)
        # Features: previous history_rows × n_features tokens, flattened.
        n_samples = n_rows - history_rows
        X = np.zeros((n_samples, history_rows * n_features), dtype=np.int16)
        for t in range(n_samples):
            X[t] = rows[t : t + history_rows].reshape(-1)
        return X, y

    print("[baseline] building train features…", flush=True)
    t0 = time.time()
    X_tr, y_tr = _shards_to_xy(train_paths, "train")
    print(f"[baseline] train shape: X={X_tr.shape} y={y_tr.shape} "
          f"(positive rate={y_tr.mean()*100:.2f}%) in {time.time()-t0:.1f}s",
          flush=True)
    print("[baseline] building eval features…", flush=True)
    t0 = time.time()
    X_ev, y_ev = _shards_to_xy(eval_paths, "eval")
    print(f"[baseline] eval shape : X={X_ev.shape} y={y_ev.shape} "
          f"(positive rate={y_ev.mean()*100:.2f}%) in {time.time()-t0:.1f}s",
          flush=True)
    if len(X_tr) == 0 or len(X_ev) == 0:
        raise RuntimeError("not enough rows for training/eval")

    # Subsample if huge — LightGBM at 10M+ rows takes a while on CPU; the
    # signal/noise ratio doesn't improve enough above ~500k rows for a
    # diagnostic.
    cap = 500_000
    if len(X_tr) > cap:
        idx = np.random.RandomState(0).choice(len(X_tr), cap, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"[baseline] subsampled train to {cap:,} rows", flush=True)

    # ----- LightGBM -----
    print("[baseline] training LightGBM (binary, 200 rounds)…", flush=True)
    t0 = time.time()
    import lightgbm as lgb
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
    model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], callbacks=[lgb.log_evaluation(50)])
    print(f"[baseline] LightGBM fit in {time.time()-t0:.1f}s", flush=True)

    # ----- Eval -----
    pred_proba = model.predict_proba(X_ev)[:, 1]
    pred = (pred_proba >= 0.5).astype(np.int8)
    acc = float(np.mean(pred == y_ev))
    base_rate = float(np.mean(y_ev))
    se = float(np.sqrt(0.5 * 0.5 / max(len(y_ev), 1)))
    z = (acc - 0.5) / max(se, 1e-9)
    verdict = ("SIGNAL — retrain 524M with directional head" if acc >= 0.53
               else ("marginal — borderline, lean conservative"
                     if acc >= 0.51 else
                     "NO signal extractable — pivot to Phase 6/7 alpha factors"))

    print()
    print(f"[baseline] ===== HELD-OUT DIRECTIONAL ON {eval_job_id} =====")
    print(f"[baseline] eval n              : {len(y_ev):,}")
    print(f"[baseline] base rate (y=1)     : {base_rate*100:.2f}%  "
          f"(closer to 50% = balanced)")
    print(f"[baseline] LightGBM accuracy   : {acc*100:.2f}%")
    print(f"[baseline] z-score vs 50%      : {z:+.2f}σ")
    print(f"[baseline] threshold (53%)     : {'PASS' if acc >= 0.53 else 'FAIL'}")
    print(f"[baseline] verdict             : {verdict}")

    # Top features by importance — useful diagnostic.
    importances = model.feature_importances_
    top_k = np.argsort(importances)[::-1][:10]
    print()
    print(f"[baseline] top 10 feature indices by gain (out of "
          f"{history_rows * n_features}):")
    for rank, fi in enumerate(top_k):
        row_offset = fi // n_features
        feat_idx = fi % n_features
        feat_names = ["ret", "mid", "spread", "bid_sz", "ask_sz", "last_sz", "ia_ms"]
        feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"f{feat_idx}"
        print(f"   {rank+1:>2}. row[t-{history_rows-row_offset}].{feat_name}  "
              f"importance={importances[fi]:.0f}")

    return {
        "accuracy": acc,
        "n_eval": len(y_ev),
        "base_rate": base_rate,
        "z_score": z,
        "passes_53_threshold": bool(acc >= 0.53),
    }


@app.local_entrypoint()
def main(history_rows: int = 16,
         max_train_shards: int = 50,
         max_eval_shards: int = 50):
    """Run the directional baseline. Defaults are calibrated for ~5 min wall-clock
    on Modal CPU + ~$0.10. Bump --max-train-shards / --max-eval-shards for
    a more thorough run."""
    run_baseline.remote(
        history_rows=history_rows,
        max_train_shards=max_train_shards,
        max_eval_shards=max_eval_shards,
    )
