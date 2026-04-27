"""h=10 capacity probe — is model size the bottleneck?

Multi-day balanced eval on horizon=10 only, but with a much larger LightGBM
config (history_rows=64, num_leaves=255, n_estimators=500). Multi-day run
showed h=10 sitting at AUC 0.64 across all 6 days at the small config. The
question: does *more capacity at the same data scale* push that toward 0.70?

  - If h=10 AUC moves to ≥0.68 → capacity IS the bottleneck. The 524M with a
    directional head has a real target — fund the retrain.
  - If h=10 AUC plateaus at ~0.64 → capacity is NOT the bottleneck. Feature
    expressiveness or data are the limit. The 524M retrain would be wasted;
    redirect to richer features / longer history / Phase 6 alpha factors.

Same train/eval temporal split + train-set median threshold + per-day metrics
as dir_baseline_multiday.py. Eval is cross-day so balanced metrics aren't
corrupted by single-day skew.

Usage:
    modal run infra/modal/dir_baseline_capacity.py::main
"""
from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "tradefm-dir-baseline-capacity"

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    .pip_install(
        "pyarrow>=15.0", "pandas>=2.2", "numpy>=1.26",
        "scikit-learn>=1.4", "lightgbm>=4.0",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)

shard_volume = modal.Volume.from_name("tradefm-smoke-shards", create_if_missing=False)
app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=8.0,
    memory=65536,
    timeout=7200,
    volumes={"/shards": shard_volume},
)
def run_capacity_probe(
    horizon: int = 10,
    history_rows: int = 64,
    num_leaves: int = 255,
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    max_shards_per_day: int = 50,
    n_features: int = 7,
    train_frac: float = 0.80,
    train_subsample: int = 1_500_000,
):
    """High-capacity LightGBM on h=10. Reports per-day + aggregate AUC and
    balanced accuracy. Compare against multiday h=10 baseline (AUC 0.64)."""
    import time
    import glob as _glob
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from numpy.lib.stride_tricks import sliding_window_view

    base = Path("/shards/databento_reuse_packed")
    assert base.exists(), f"no packed shards at {base}"

    day_paths: dict[str, list[Path]] = {}
    for job_dir in sorted(base.iterdir()):
        if not job_dir.is_dir():
            continue
        shards = sorted(_glob.glob(f"{job_dir}/opra_*.parquet"))
        if shards:
            day_paths[job_dir.name] = shards[:max_shards_per_day]
    print(f"[capacity] days: {list(day_paths.keys())}", flush=True)
    print(f"[capacity] horizon={horizon}  history_rows={history_rows}  "
          f"num_leaves={num_leaves}  n_estimators={n_estimators}  "
          f"lr={learning_rate}", flush=True)

    def _load_day(paths: list[Path], label: str) -> np.ndarray:
        all_arr: list[np.ndarray] = []
        for shard in paths:
            df = pd.read_parquet(shard, columns=["tokens"])
            for row_tokens in df["tokens"]:
                arr = np.asarray(row_tokens, dtype=np.int16)
                if arr.shape[0] == n_features:
                    all_arr.append(arr)
        rows = np.stack(all_arr, axis=0) if all_arr else np.zeros((0, n_features), dtype=np.int16)
        print(f"[capacity] {label}: {rows.shape[0]:,} rows", flush=True)
        return rows

    print("[capacity] loading rows per day…", flush=True)
    t0 = time.time()
    day_rows: dict[str, np.ndarray] = {d: _load_day(p, d) for d, p in day_paths.items()}
    print(f"[capacity] all loads: {time.time()-t0:.1f}s", flush=True)

    def _split(rows, h):
        n = rows.shape[0]
        train_end = int(n * train_frac)
        eval_start = train_end + h  # gap = horizon
        if eval_start >= n - history_rows - h:
            return rows[:train_end], np.zeros((0, n_features), dtype=rows.dtype)
        return rows[:train_end], rows[eval_start:]

    def _build_xy(rows, h, threshold):
        n = rows.shape[0]
        n_samples = n - history_rows - h + 1
        if n_samples <= 0:
            return (np.zeros((0, history_rows * n_features), dtype=np.int16),
                    np.zeros(0, dtype=np.int8))
        ret_tokens = rows[:, 0]
        above = (ret_tokens > threshold).astype(np.int32)
        cum = np.concatenate([[0], np.cumsum(above)])
        starts = history_rows + np.arange(n_samples)
        ends = starts + h
        window_sums = cum[ends] - cum[starts]
        y = (window_sums * 2 > h).astype(np.int8)
        windows = sliding_window_view(rows, (history_rows, n_features))[:, 0, :, :]
        X = windows[:n_samples].reshape(n_samples, history_rows * n_features).astype(np.int16)
        return X, y

    # Per-day train/eval split, concatenate.
    train_rows_all = []
    eval_per_day: dict[str, np.ndarray] = {}
    for day, rows in day_rows.items():
        tr, ev = _split(rows, horizon)
        train_rows_all.append(tr)
        eval_per_day[day] = ev

    train_rows = np.concatenate(train_rows_all, axis=0)
    threshold = float(np.median(train_rows[:, 0]))
    print(f"[capacity] train rows={len(train_rows):,}  threshold={threshold:.2f}",
          flush=True)

    t0 = time.time()
    X_tr, y_tr = _build_xy(train_rows, horizon, threshold)
    print(f"[capacity] train shape={X_tr.shape}  pos rate={y_tr.mean()*100:.2f}%  "
          f"build={time.time()-t0:.1f}s", flush=True)

    rng = np.random.RandomState(0)
    if len(X_tr) > train_subsample:
        idx = rng.choice(len(X_tr), train_subsample, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"[capacity] subsampled train to {train_subsample:,}", flush=True)

    eval_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for day, ev_rows in eval_per_day.items():
        if ev_rows.shape[0] >= history_rows + horizon + 1:
            eval_xy[day] = _build_xy(ev_rows, horizon, threshold)

    ev_X, ev_y = [], []
    per_day_index: dict[str, tuple[int, int]] = {}
    cur = 0
    for day, (Xd, yd) in eval_xy.items():
        ev_X.append(Xd); ev_y.append(yd)
        per_day_index[day] = (cur, cur + len(Xd)); cur += len(Xd)
    X_ev = np.concatenate(ev_X, axis=0)
    y_ev = np.concatenate(ev_y, axis=0)
    print(f"[capacity] agg eval={X_ev.shape}  pos rate={y_ev.mean()*100:.2f}%",
          flush=True)

    # ---- Train high-capacity LightGBM ----
    print(f"[capacity] training LightGBM "
          f"(num_leaves={num_leaves}, n_estimators={n_estimators}, lr={learning_rate})…",
          flush=True)
    t0 = time.time()
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=200,
        max_depth=-1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)],
              callbacks=[lgb.log_evaluation(50)])
    print(f"[capacity] fit in {time.time()-t0:.1f}s", flush=True)

    proba = model.predict_proba(X_ev)[:, 1]
    pred = (proba >= 0.5).astype(np.int8)
    agg_auc = float(roc_auc_score(y_ev, proba))
    agg_bal = float(balanced_accuracy_score(y_ev, pred))
    agg_acc = float(np.mean(pred == y_ev))
    agg_base = float(np.mean(y_ev))
    print(f"[capacity] AGG: AUC={agg_auc:.4f}  bal={agg_bal*100:.2f}%  "
          f"raw={agg_acc*100:.2f}%  base={agg_base*100:.2f}%", flush=True)

    print()
    print("[capacity] per-day:", flush=True)
    per_day = {}
    for day, (lo, hi) in per_day_index.items():
        if hi <= lo:
            continue
        yd, pd_proba, pd_pred = y_ev[lo:hi], proba[lo:hi], pred[lo:hi]
        if len(set(yd)) > 1:
            d_auc = float(roc_auc_score(yd, pd_proba))
            d_bal = float(balanced_accuracy_score(yd, pd_pred))
        else:
            d_auc = float("nan"); d_bal = float("nan")
        d_acc = float(np.mean(pd_pred == yd)); d_base = float(np.mean(yd))
        per_day[day] = {"auc": d_auc, "bal": d_bal, "raw": d_acc,
                        "base": d_base, "n": int(hi - lo)}
        print(f"[capacity]   {day}  AUC={d_auc:.4f}  bal={d_bal*100:.2f}%  "
              f"raw={d_acc*100:.2f}%  base={d_base*100:.2f}%  n={hi-lo:,}",
              flush=True)

    # ---- Verdict ----
    BASELINE_AUC = 0.64
    delta = agg_auc - BASELINE_AUC
    print()
    print(f"[capacity] ============ VERDICT ============", flush=True)
    print(f"[capacity] baseline (small) AUC: {BASELINE_AUC:.4f}", flush=True)
    print(f"[capacity] capacity AUC        : {agg_auc:.4f}", flush=True)
    print(f"[capacity] Δ                   : {delta:+.4f}", flush=True)
    if agg_auc >= 0.68:
        verdict = ("CAPACITY IS THE BOTTLENECK — fund 524M directional "
                   "retrain (target: AUC ≥ 0.68 with directional head)")
    elif agg_auc >= 0.66:
        verdict = ("partial lift — capacity helps marginally; proceed with "
                   "directional retrain only if combined with longer context")
    else:
        verdict = ("CAPACITY IS NOT THE BOTTLENECK — feature expressiveness "
                   "or data scale is the limit. Skip the 524M retrain. "
                   "Redirect to richer features / Phase 6 alpha factors.")
    print(f"[capacity] verdict             : {verdict}", flush=True)

    result = {
        "agg_auc": agg_auc, "agg_bal": agg_bal,
        "agg_raw": agg_acc, "agg_base": agg_base,
        "per_day": per_day, "delta_vs_baseline": delta,
        "verdict": verdict,
        "config": {
            "horizon": horizon, "history_rows": history_rows,
            "num_leaves": num_leaves, "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        },
    }

    # Persist to the shard volume — `modal app logs` doesn't backfill on
    # stopped detached apps, so we always need a way to retrieve the
    # number after the fact. Auto-commits when the function returns.
    import json
    results_dir = Path("/shards/_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"capacity_h{horizon}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[capacity] persisted result to {out_path}", flush=True)
    return result


@app.function(volumes={"/shards": shard_volume}, cpu=0.5, memory=1024,
              timeout=120)
def read_capacity_result(horizon: int = 10) -> dict:
    """Tiny CPU helper to read the persisted JSON. Use after a detached
    run to retrieve results without scraping logs."""
    import json
    p = Path(f"/shards/_results/capacity_h{horizon}.json")
    if not p.exists():
        return {"error": f"no result at {p}"}
    return json.loads(p.read_text())


@app.local_entrypoint()
def fetch_result(horizon: int = 10):
    """Fetch and print the persisted capacity-probe result."""
    import json
    r = read_capacity_result.remote(horizon=horizon)
    print(json.dumps(r, indent=2, default=str))


@app.local_entrypoint()
def main(history_rows: int = 64, num_leaves: int = 255,
         n_estimators: int = 500):
    """~15-25 min wall-clock, ~$0.50 on Modal CPU."""
    run_capacity_probe.remote(
        history_rows=history_rows, num_leaves=num_leaves,
        n_estimators=n_estimators,
    )
