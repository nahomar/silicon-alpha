"""Learning-rate finder (Smith 2017, exponential ramp).

Run 200-ish steps with LR exponentially increasing from `lr_min` to `lr_max`
on the real training data. Record loss at each step. The best LR is the
point of steepest descent before the loss explodes — we report:
  - suggested_lr: LR at min smoothed-gradient (one order of magnitude
                  below the value where loss is minimum)
  - min_loss_lr: LR at the absolute minimum of the smoothed loss curve
  - plot_path:   PNG of loss vs log-LR (if matplotlib available)

Use this BEFORE the full 100B-token run so cfg.lr is data-calibrated.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from odte.transformer_tradefm import TradeFM
from odte.train.pretrain_tradefm import ShardTokenDataset, load_config

log = logging.getLogger(__name__)


def _device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and hasattr(torch.backends, "mps") \
            and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _smooth(values: np.ndarray, beta: float = 0.98) -> np.ndarray:
    """Exponential moving average; bias-corrected."""
    avg = np.zeros_like(values)
    running = 0.0
    for i, v in enumerate(values):
        running = beta * running + (1 - beta) * v
        avg[i] = running / (1 - beta ** (i + 1))
    return avg


def find_lr(model: TradeFM, shard_paths: List[Path], device: torch.device,
            lr_min: float = 1e-7, lr_max: float = 1e-1,
            steps: int = 200, batch: int = 8,
            diverge_factor: float = 4.0) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_min)
    lrs = np.exp(np.linspace(math.log(lr_min), math.log(lr_max), steps))
    losses: list[float] = []
    ds = ShardTokenDataset(shard_paths, ctx_len=model.cfg.ctx_len, seed=0)
    loader = DataLoader(ds, batch_size=batch, num_workers=0)
    it = iter(loader)

    best = float("inf")
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = float(lrs[step])
        try:
            batch_tok = next(it)
        except StopIteration:
            it = iter(loader)
            batch_tok = next(it)
        batch_tok = batch_tok.to(device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(batch_tok)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        best = min(best, losses[-1])
        if step % max(1, steps // 20) == 0:
            log.info("lr_finder step %d  lr=%.2e  loss=%.4f", step,
                     lrs[step], losses[-1])
        if not math.isfinite(losses[-1]) or losses[-1] > diverge_factor * best:
            log.info("diverged at step %d (loss=%.4f, best=%.4f); truncating",
                     step, losses[-1], best)
            lrs = lrs[: step + 1]
            losses_arr = np.array(losses)
            break
    else:
        losses_arr = np.array(losses)

    smooth = _smooth(losses_arr)
    # Steepest descent point: argmin of gradient of smoothed curve
    grad = np.gradient(smooth, np.log(lrs))
    # Cap the search to the portion before the minimum (after min, loss rises)
    min_idx = int(np.argmin(smooth))
    search = grad[: max(1, min_idx)]
    steep_idx = int(np.argmin(search)) if len(search) else min_idx
    suggested = float(lrs[steep_idx])
    min_loss_lr = float(lrs[min_idx])

    result = {
        "suggested_lr": suggested,
        "min_loss_lr": min_loss_lr,
        "min_loss": float(smooth[min_idx]),
        "lrs": lrs.tolist(),
        "losses": losses_arr.tolist(),
        "smooth": smooth.tolist(),
        "steep_idx": steep_idx,
        "min_idx": min_idx,
    }
    return result


def maybe_plot(result: dict, out_png: Path) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable; skipping plot")
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["lrs"], result["losses"], alpha=0.35, label="raw")
    ax.plot(result["lrs"], result["smooth"], linewidth=2, label="smoothed")
    ax.axvline(result["suggested_lr"], color="red", linestyle="--",
               label=f"suggested {result['suggested_lr']:.1e}")
    ax.axvline(result["min_loss_lr"], color="green", linestyle=":",
               label=f"min-loss {result['min_loss_lr']:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss")
    ax.set_title("LR finder")
    ax.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _cli():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--lr-max", type=float, default=1e-1)
    ap.add_argument("--out", default="reports/lr_finder.json")
    ap.add_argument("--plot", default="reports/lr_finder.png")
    a = ap.parse_args()

    cfg = load_config(a.config)
    dev = _device(a.device)
    shards = sorted(Path(p) for p in _glob.glob(a.shards))
    if not shards:
        raise RuntimeError(f"no shards matched {a.shards!r}")
    model = TradeFM(cfg)
    result = find_lr(model, shards, dev,
                     lr_min=a.lr_min, lr_max=a.lr_max,
                     steps=a.steps, batch=a.batch)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if a.plot:
        maybe_plot(result, Path(a.plot))
    print(json.dumps({k: result[k] for k in
                      ("suggested_lr", "min_loss_lr", "min_loss",
                       "steep_idx", "min_idx")}, indent=2))


if __name__ == "__main__":
    _cli()
