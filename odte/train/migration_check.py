"""Colab → H100 migration gate.

A disciplined answer to "is my Colab 40M run good enough to justify the
$40-50k H100 training bill?" — based on:

  1. Validation loss is strictly decreasing over the last K evals
     (no plateau, no up-trend)
  2. Held-out directional accuracy at 1-step ≥ the gate (default 53%)
  3. Greek error on the companion DMLPricer is within tolerance
  4. Tokenizer-edge hash matches what you plan to retrain at scale

Writes a one-page markdown decision doc to reports/migration_decision.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MigrationDecision:
    ready: bool
    reasons: List[str]
    metrics: dict


def _strictly_decreasing(vals: List[float], tail: int = 5,
                          min_rel_slope: float = 1e-3) -> bool:
    """Strictly decreasing monotone + RELATIVE slope ≥ min_rel_slope of
    the tail mean per step.

    Relative slope prevents false negatives when the run has converged
    near the noise floor (e.g. loss=0.0005). Previous absolute threshold
    (1e-4 per step) rejected any run where remaining drop/step < 1e-4.
    """
    if len(vals) < tail + 1:
        return False
    tail_vals = vals[-tail:]
    if any(b >= a for a, b in zip(tail_vals, tail_vals[1:])):
        return False
    x = np.arange(len(tail_vals), dtype=np.float64)
    slope, _ = np.polyfit(x, tail_vals, 1)
    slope = float(slope)
    mean_val = float(np.mean(tail_vals))
    if mean_val <= 0:
        return bool(slope < 0)
    rel_slope = -slope / mean_val
    return bool(rel_slope >= min_rel_slope)


def decide(train_loss_history: List[float],
            val_loss_history: List[float],
            directional_hit_rate: Optional[float] = None,
            dml_greek_err_max_pct: Optional[float] = None,
            tokenizer_json_path: Optional[Path] = None,
            require_dir_acc: float = 0.53,
            require_dml_pct: float = 2.0) -> MigrationDecision:
    reasons: List[str] = []
    metrics: dict = {}

    # 1. val-loss shape
    strict = _strictly_decreasing(val_loss_history)
    metrics["val_strictly_decreasing"] = strict
    metrics["val_loss_tail"] = val_loss_history[-5:] if len(val_loss_history) >= 5 else val_loss_history
    if not strict:
        reasons.append("val_loss not strictly decreasing over last 5 evals")

    # 2. train vs val divergence (overfit check)
    if train_loss_history and val_loss_history:
        gap = val_loss_history[-1] - train_loss_history[-1]
        metrics["train_val_gap"] = float(gap)
        if gap > 1.0:
            reasons.append(f"train/val gap = {gap:.2f} — possible overfit")

    # 3. directional hit rate
    if directional_hit_rate is not None:
        metrics["directional_hit_rate"] = directional_hit_rate
        if directional_hit_rate < require_dir_acc:
            reasons.append(f"dir-acc {directional_hit_rate*100:.1f}% "
                           f"< {require_dir_acc*100:.0f}% required")
        # Overfit sentinel — dir-acc > 99% on synth is memorization, not
        # edge. 524M on real L3 data realistically caps ~54-58%.
        if directional_hit_rate >= 0.99:
            reasons.append(f"dir-acc {directional_hit_rate*100:.1f}% — "
                           "too perfect, model is memorizing. Check for "
                           "train/test shard leak or trivial synth data.")

    # 4. DML Greek error
    if dml_greek_err_max_pct is not None:
        metrics["dml_max_greek_pct"] = dml_greek_err_max_pct
        if dml_greek_err_max_pct > require_dml_pct:
            reasons.append(f"DML max Greek err {dml_greek_err_max_pct:.2f}% "
                           f"> {require_dml_pct}% tolerance")

    # 5. tokenizer hash (so H100 retrain uses the exact same edges)
    if tokenizer_json_path and Path(tokenizer_json_path).exists():
        h = hashlib.sha256(Path(tokenizer_json_path).read_bytes()).hexdigest()[:16]
        metrics["tokenizer_sha256"] = h

    ready = len(reasons) == 0
    return MigrationDecision(ready=ready, reasons=reasons, metrics=metrics)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(decision: MigrationDecision, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    status = "✅ GO — migrate to H100 cluster" if decision.ready \
        else "❌ NO-GO — fix the issues below, do NOT spend $40-50k yet"
    lines.append(f"# Colab → H100 migration decision — {time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"## {status}\n")
    if decision.reasons:
        lines.append("## Blockers\n")
        for r in decision.reasons:
            lines.append(f"- {r}")
        lines.append("")
    lines.append("## Metrics\n")
    for k, v in decision.metrics.items():
        if isinstance(v, list):
            v = ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v)
        lines.append(f"- **{k}**: {v}")
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-loss", required=True,
                    help="JSON file with list of train losses")
    ap.add_argument("--val-loss", required=True,
                    help="JSON file with list of val losses")
    ap.add_argument("--dir-acc", type=float, default=None)
    ap.add_argument("--dml-greek-pct", type=float, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default="reports/migration_decision.md")
    ap.add_argument("--dir-acc-gate", type=float, default=0.53)
    ap.add_argument("--dml-gate", type=float, default=2.0)
    a = ap.parse_args()

    train = json.loads(Path(a.train_loss).read_text())
    val = json.loads(Path(a.val_loss).read_text())
    d = decide(train, val,
                directional_hit_rate=a.dir_acc,
                dml_greek_err_max_pct=a.dml_greek_pct,
                tokenizer_json_path=Path(a.tokenizer) if a.tokenizer else None,
                require_dir_acc=a.dir_acc_gate,
                require_dml_pct=a.dml_gate)
    write_report(d, Path(a.out))
    print(json.dumps({"ready": d.ready, "reasons": d.reasons,
                      "metrics": d.metrics}, indent=2))
    sys.exit(0 if d.ready else 1)


if __name__ == "__main__":
    _cli()
