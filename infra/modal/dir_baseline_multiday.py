"""Multi-day balanced directional baseline — does signal survive *across*
days when we don't let single-day class imbalance corrupt the metric?

Sister to dir_baseline_horizon.py. Same horizons sweep, but:

  1. Eval is cross-day, not single-day. Each of the 5 OPRA days contributes
     its last 20% of rows to a unified eval pool (with a `horizon`-row gap
     between train and eval to prevent label leakage). Train is the first
     80% of every day. This kills the "expiry-skewed single-day eval"
     artifact that made dir_baseline_horizon.py's h≥10 metrics worse than a
     constant predictor on 2026-04-16.

  2. Primary metrics: ROC-AUC + balanced accuracy (not raw accuracy).
     Both are insensitive to class imbalance, which was exactly what
     corrupted the previous run.

  3. Per-day eval breakdown — if one day carries all the signal we want to
     see that, not have it averaged into a happy aggregate.

Decision rule:
    GO on 524M directional retrain only if:
      - balanced_acc ≥ 53% on aggregate eval
      - AUC ≥ 0.55 on aggregate eval
      - and at least 3 of 5 days individually pass balanced_acc ≥ 52%
    at any horizon ≥ 10. Otherwise: STOP, signal is next-tick-only.

Usage:
    modal run infra/modal/dir_baseline_multiday.py::main
"""
from __future__ import annotations

import modal
from pathlib import Path

APP_NAME = "tradefm-dir-baseline-multiday"

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

DEFAULT_HORIZONS = [1, 10, 100, 1000]


@app.function(
    cpu=8.0,
    memory=65536,
    timeout=5400,
    volumes={"/shards": shard_volume},
)
def run_multiday_baseline(
    history_rows: int = 16,
    max_shards_per_day: int = 50,
    n_features: int = 7,
    horizons: list[int] = DEFAULT_HORIZONS,
    train_frac: float = 0.80,
):
    """Train LightGBM per-horizon on cross-day eval, reporting AUC + balanced
    accuracy + per-day breakdown."""
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
    print(f"[multiday] days found: {list(day_paths.keys())}", flush=True)
    print(f"[multiday] history_rows={history_rows} horizons={horizons} "
          f"train_frac={train_frac}", flush=True)

    def _load_day_rows(paths: list[Path], label: str) -> np.ndarray:
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
        print(f"[multiday] {label}: {rows.shape[0]:,} rows", flush=True)
        return rows

    # Load each day's rows separately so we can do a temporal split per day.
    print("[multiday] loading rows per day…", flush=True)
    t0 = time.time()
    day_rows: dict[str, np.ndarray] = {}
    for day, paths in day_paths.items():
        day_rows[day] = _load_day_rows(paths, day)
    print(f"[multiday] all loads: {time.time()-t0:.1f}s", flush=True)

    def _split_day(rows: np.ndarray, horizon: int
                   ) -> tuple[np.ndarray, np.ndarray]:
        """Temporal split with a `horizon`-row gap to prevent label leakage
        between train and eval (the target window for a train sample at the
        boundary would otherwise overlap eval feature rows)."""
        n = rows.shape[0]
        gap = horizon
        train_end = int(n * train_frac)
        eval_start = train_end + gap
        if eval_start >= n - history_rows - horizon:
            return rows[:train_end], np.zeros((0, n_features), dtype=rows.dtype)
        return rows[:train_end], rows[eval_start:]

    def _build_xy(rows: np.ndarray, horizon: int, threshold: float
                  ) -> tuple[np.ndarray, np.ndarray]:
        """X = flattened (history_rows × n_features) tokens.
        y = 1 if majority of ret tokens in the next `horizon` rows are
            above `threshold`."""
        n = rows.shape[0]
        n_samples = n - history_rows - horizon + 1
        if n_samples <= 0:
            return (np.zeros((0, history_rows * n_features), dtype=np.int16),
                    np.zeros(0, dtype=np.int8))
        ret_tokens = rows[:, 0]
        above = (ret_tokens > threshold).astype(np.int32)
        cum = np.concatenate([[0], np.cumsum(above)])
        starts = history_rows + np.arange(n_samples)
        ends = starts + horizon
        window_sums = cum[ends] - cum[starts]
        y = (window_sums * 2 > horizon).astype(np.int8)
        windows = sliding_window_view(rows, (history_rows, n_features))[:, 0, :, :]
        X = windows[:n_samples].reshape(n_samples, history_rows * n_features).astype(np.int16)
        return X, y

    cap = 600_000
    rng = np.random.RandomState(0)

    import lightgbm as lgb
    horizon_results: dict[int, dict] = {}

    for horizon in horizons:
        print()
        print(f"[multiday] ===== horizon={horizon} =====", flush=True)
        t0 = time.time()

        # Per-day temporal splits, then concatenate.
        train_rows_all: list[np.ndarray] = []
        eval_per_day: dict[str, np.ndarray] = {}
        for day, rows in day_rows.items():
            tr, ev = _split_day(rows, horizon)
            train_rows_all.append(tr)
            eval_per_day[day] = ev

        train_rows = np.concatenate(train_rows_all, axis=0)
        threshold = float(np.median(train_rows[:, 0]))
        print(f"[multiday] h={horizon}: train rows={len(train_rows):,}  "
              f"threshold={threshold:.2f}", flush=True)

        # Build train (X, y).
        X_tr, y_tr = _build_xy(train_rows, horizon, threshold)
        print(f"[multiday] h={horizon}: train shape={X_tr.shape}  "
              f"pos rate={y_tr.mean()*100:.2f}%", flush=True)
        if len(X_tr) == 0:
            print(f"[multiday] h={horizon}: skip — empty train", flush=True)
            continue

        if len(X_tr) > cap:
            idx = rng.choice(len(X_tr), cap, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            print(f"[multiday] h={horizon}: subsampled train to {cap:,}",
                  flush=True)

        # Build per-day eval (X, y) using the SAME train threshold.
        eval_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for day, ev_rows in eval_per_day.items():
            if ev_rows.shape[0] >= history_rows + horizon + 1:
                eval_xy[day] = _build_xy(ev_rows, horizon, threshold)

        # Concat eval across days, while keeping per-day slice indices.
        ev_pieces_X = []
        ev_pieces_y = []
        per_day_index: dict[str, tuple[int, int]] = {}
        cursor = 0
        for day, (Xd, yd) in eval_xy.items():
            ev_pieces_X.append(Xd)
            ev_pieces_y.append(yd)
            per_day_index[day] = (cursor, cursor + len(Xd))
            cursor += len(Xd)
        if not ev_pieces_X:
            print(f"[multiday] h={horizon}: no eval data, skip", flush=True)
            continue
        X_ev = np.concatenate(ev_pieces_X, axis=0)
        y_ev = np.concatenate(ev_pieces_y, axis=0)
        print(f"[multiday] h={horizon}: agg eval={X_ev.shape}  "
              f"pos rate={y_ev.mean()*100:.2f}%", flush=True)

        # ----- Train -----
        t1 = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=200, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=-1, random_state=42,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)],
                  callbacks=[lgb.log_evaluation(0)])
        fit_s = time.time() - t1

        # ----- Aggregate eval metrics -----
        proba = model.predict_proba(X_ev)[:, 1]
        pred = (proba >= 0.5).astype(np.int8)
        agg_acc = float(np.mean(pred == y_ev))
        agg_base = float(np.mean(y_ev))
        # AUC and balanced acc are imbalance-resistant.
        try:
            agg_auc = float(roc_auc_score(y_ev, proba))
        except ValueError:
            agg_auc = float("nan")
        try:
            agg_bal = float(balanced_accuracy_score(y_ev, pred))
        except ValueError:
            agg_bal = float("nan")
        print(f"[multiday] h={horizon}: agg  AUC={agg_auc:.4f}  "
              f"bal_acc={agg_bal*100:.2f}%  raw={agg_acc*100:.2f}%  "
              f"base={agg_base*100:.2f}%  fit={fit_s:.1f}s",
              flush=True)

        # ----- Per-day eval metrics -----
        per_day_metrics: dict[str, dict] = {}
        for day, (lo, hi) in per_day_index.items():
            if hi <= lo:
                continue
            yd = y_ev[lo:hi]
            pd_proba = proba[lo:hi]
            pd_pred = pred[lo:hi]
            try:
                d_auc = float(roc_auc_score(yd, pd_proba)) if len(set(yd)) > 1 else float("nan")
            except ValueError:
                d_auc = float("nan")
            try:
                d_bal = float(balanced_accuracy_score(yd, pd_pred)) if len(set(yd)) > 1 else float("nan")
            except ValueError:
                d_bal = float("nan")
            d_acc = float(np.mean(pd_pred == yd))
            d_base = float(np.mean(yd))
            per_day_metrics[day] = {
                "auc": d_auc, "balanced_acc": d_bal,
                "raw_acc": d_acc, "base_rate": d_base, "n": int(hi - lo),
            }
            print(f"[multiday] h={horizon}:   {day}  AUC={d_auc:.4f}  "
                  f"bal={d_bal*100:.2f}%  raw={d_acc*100:.2f}%  "
                  f"base={d_base*100:.2f}%  n={hi-lo:,}",
                  flush=True)

        horizon_results[horizon] = {
            "auc": agg_auc, "balanced_acc": agg_bal, "raw_acc": agg_acc,
            "base_rate": agg_base, "n_eval": int(len(y_ev)),
            "per_day": per_day_metrics,
        }

    # ---- Summary ----
    print()
    print("[multiday] ============ SUMMARY ============", flush=True)
    print(f"[multiday] {'h':>5} | {'AUC':>6} | {'bal_acc':>7} | "
          f"{'days≥52%bal':>11} | verdict", flush=True)
    print(f"[multiday] {'-'*5}-+-{'-'*6}-+-{'-'*7}-+-{'-'*11}-+-{'-'*40}",
          flush=True)
    survives_any = False
    for h, r in horizon_results.items():
        n_days_pass = sum(1 for d in r["per_day"].values()
                          if not np.isnan(d["balanced_acc"])
                          and d["balanced_acc"] >= 0.52)
        n_days_total = len(r["per_day"])
        if r["balanced_acc"] >= 0.53 and r["auc"] >= 0.55 and n_days_pass >= 3:
            verdict = "TRADEABLE — fund 524M directional retrain"
            if h >= 10:
                survives_any = True
        elif r["balanced_acc"] >= 0.51 and r["auc"] >= 0.52:
            verdict = "marginal — borderline, needs more days"
        else:
            verdict = "no edge — signal does not survive"
        print(f"[multiday] {h:>5} | {r['auc']:>5.3f} | "
              f"{r['balanced_acc']*100:>6.2f}% | "
              f"{n_days_pass:>3}/{n_days_total:<7} | {verdict}",
              flush=True)
    print()
    print(f"[multiday] decision: "
          f"{'GO' if survives_any else 'STOP'} on 524M directional retrain "
          f"(any horizon ≥10 passing balanced+AUC+per-day gates)",
          flush=True)

    return {
        "horizon_results": horizon_results,
        "survives_trading_horizon": survives_any,
    }


@app.local_entrypoint()
def main(history_rows: int = 16,
         max_shards_per_day: int = 50,
         train_frac: float = 0.80):
    """~10-15 min wall-clock, ~$0.40-0.60 on Modal CPU."""
    run_multiday_baseline.remote(
        history_rows=history_rows,
        max_shards_per_day=max_shards_per_day,
        train_frac=train_frac,
    )
