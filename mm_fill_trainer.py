"""Hourly fill-probability trainer.

Pipeline:
  1. Read last 24h of persisted quotes + trades (mm/persistence.py).
  2. Reconcile to produce bid/ask fill labels (mm/reconcile.py).
  3. Reshape into per-side training rows (features + label).
  4. Fit calibrated FillProbabilityModel.
  5. Save checkpoint to checkpoints/fill_model.pkl + metrics report.

Meant to be invoked by the existing hourly LaunchAgent.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm.persistence import read_recent
from mm.reconcile import build_fill_labels, to_training_frame
from mm.fill_model import FillProbabilityModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("fill_trainer")

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "checkpoints" / "fill_model.pkl"
REPORT_DIR = ROOT / "reports"


def main(hours: int = 24, horizon_ms: int = 2000, min_rows: int = 500) -> int:
    quotes = read_recent("quotes", hours=hours)
    trades = read_recent("trades", hours=hours)
    log.info("quotes=%d  trades=%d  (last %dh)", len(quotes), len(trades), hours)

    if len(quotes) < min_rows:
        log.warning("Not enough quotes yet (%d < %d); skipping training.",
                    len(quotes), min_rows)
        return 1
    if len(trades) < 50:
        log.warning("Not enough trades yet (%d < 50); skipping training.", len(trades))
        return 1

    labeled = build_fill_labels(quotes, trades, horizon_ms=horizon_ms)
    n_bid_fills = int(labeled["bid_fill"].sum())
    n_ask_fills = int(labeled["ask_fill"].sum())
    log.info("bid fills=%d  ask fills=%d  quote rows=%d",
             n_bid_fills, n_ask_fills, len(labeled))

    if n_bid_fills + n_ask_fills < 20:
        log.warning("Too few fills (%d) to train a meaningful fill model.",
                    n_bid_fills + n_ask_fills)
        return 1

    training = to_training_frame(labeled)
    training = training.dropna()
    X = training.drop(columns=["label", "side"])
    y = training["label"]

    # Simple holdout for AUC
    n = len(training)
    split = int(n * 0.8)
    model = FillProbabilityModel(horizon_steps=int(horizon_ms / 100))
    model.fit(X.iloc[:split], y.iloc[:split])
    from sklearn.metrics import roc_auc_score, brier_score_loss
    p = model.predict_proba(X.iloc[split:])
    auc = float(roc_auc_score(y.iloc[split:], p)) if len(set(y.iloc[split:])) > 1 else float("nan")
    brier = float(brier_score_loss(y.iloc[split:], p))
    log.info("holdout  AUC=%.3f  Brier=%.4f  (n=%d)", auc, brier, n - split)

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    model.save(CKPT)

    report = REPORT_DIR / f"fill_model_report_{time.strftime('%Y%m%dT%H%M%S')}.md"
    report.write_text(f"""# Fill-probability model retrain — {time.strftime('%Y-%m-%d %H:%M')}

- hours of data: {hours}
- quotes: {len(quotes)}   trades: {len(trades)}
- bid fills: {n_bid_fills}   ask fills: {n_ask_fills}
- training rows: {n}  (80/20 split)
- holdout AUC:   {auc:.3f}
- holdout Brier: {brier:.4f}
- checkpoint:    `{CKPT.relative_to(ROOT)}`

Interpretation:
  AUC ≤ 0.55 = effectively noise. 0.60–0.70 = weak but real. >0.75 = decent.
  Brier compares to class baseline: lower is better.
""")
    log.info("report → %s", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
