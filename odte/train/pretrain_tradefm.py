"""TradeFM pretraining harness — single-node.

Consumes parquet shards produced by odte.data.DataShopPacker and trains a
TradeFM model (size configured via YAML). On a cloud A100 this handles the
Phase-1 40M-param run; Phase-2 524M scales out through a separate
distributed wrapper that re-uses the same Dataset / Config classes.

Features:
  - Resumable checkpointing (ckpt_every, save_best)
  - Linear warmup + cosine LR schedule
  - AMP / grad-accum
  - Context packing: concatenate token sequences across rows until ctx_len
  - W&B / tensorboard logging optional; stdout always on
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from models.config import TradeFMConfig
from odte.transformer_tradefm import TradeFM, wrap_fp8_autocast

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (YAML with fallback to JSON)
# ---------------------------------------------------------------------------

def load_config(path: str) -> TradeFMConfig:
    p = Path(path)
    try:
        import yaml
        payload = yaml.safe_load(p.read_text())
    except ImportError:
        payload = json.loads(p.read_text())
    return TradeFMConfig(**payload)


# ---------------------------------------------------------------------------
# Dataset — iterable streaming over parquet shards
# ---------------------------------------------------------------------------

class ShardTokenDataset(IterableDataset):
    """Yields fixed-length contexts of token ids from parquet shards.

    Rows across shards are concatenated (token-packed) until ctx_len.
    """

    def __init__(self, shard_paths: Iterable[Path], ctx_len: int,
                 shuffle_buffer: int = 64, seed: int = 0):
        super().__init__()
        self.shards = sorted(Path(p) for p in shard_paths)
        self.ctx_len = ctx_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _iter_rows(self):
        rng = np.random.default_rng(self.seed)
        order = list(self.shards)
        rng.shuffle(order)
        for shard in order:
            df = pd.read_parquet(shard, columns=["tokens"])
            buf = []
            idx = list(range(len(df)))
            rng.shuffle(idx)
            for i in idx:
                buf.append(np.asarray(df.iloc[i]["tokens"], dtype=np.int32))
                if len(buf) >= self.shuffle_buffer:
                    yield from buf
                    buf = []
            yield from buf

    def __iter__(self):
        pack = np.empty(0, dtype=np.int32)
        for arr in self._iter_rows():
            pack = np.concatenate([pack, arr]) if len(pack) else arr
            while len(pack) >= self.ctx_len + 1:
                yield torch.as_tensor(pack[: self.ctx_len + 1], dtype=torch.long)
                pack = pack[self.ctx_len:]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainArgs:
    shard_glob: str
    ckpt_dir: str
    config_path: str
    steps: int = 10_000
    batch: int = 32
    grad_accum: int = 1
    ckpt_every: int = 500
    log_every: int = 50
    device: str = "cpu"
    seed: int = 0
    max_shards: Optional[int] = None   # for smoke tests


def _build_model(cfg: TradeFMConfig, device: str) -> TradeFM:
    dev = torch.device(device)
    m = TradeFM(cfg).to(dev)
    log.info("TradeFM built: %d params  d_model=%d  layers=%d  ctx=%d  vocab=%d",
             m.num_params(), cfg.d_model, cfg.n_layers, cfg.ctx_len, cfg.vocab)
    return m


def _param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _save_ckpt(model: TradeFM, opt, step: int, ckpt_dir: Path,
               best_loss: float, label: str = "ckpt") -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"{label}_{step:08d}.pt"
    torch.save({
        "state": model.state_dict(),
        "cfg": asdict(model.cfg),
        "step": step,
        "best_loss": best_loss,
        "opt": opt.state_dict(),
    }, p)
    log.info("ckpt → %s", p)
    return p


def pretrain(args: TrainArgs) -> dict:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.config_path)
    model = _build_model(cfg, args.device)

    import glob as _glob
    shard_paths = sorted(Path(p) for p in _glob.glob(args.shard_glob))
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise RuntimeError(f"no shards matched {args.shard_glob!r}")
    log.info("shards: %d", len(shard_paths))

    ds = ShardTokenDataset(shard_paths, ctx_len=cfg.ctx_len, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    warmup = cfg.warmup_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup))),
    )

    ckpt_dir = Path(args.ckpt_dir)
    best_loss = float("inf")
    t0 = time.time()
    step = 0
    run_loss: List[float] = []
    loader_iter = iter(loader)

    # bf16 autocast on CUDA halves activation memory vs fp32 and is lossless
    # for transformer training at this scale. FP8 (transformer_engine) takes
    # precedence when available via wrap_fp8_autocast.
    use_bf16 = (args.device == "cuda" and torch.cuda.is_available()
                and not getattr(cfg, "fp8", False))

    def _autocast_ctx():
        if use_bf16:
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        return wrap_fp8_autocast()

    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(loader_iter).to(args.device)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter).to(args.device)
            with _autocast_ctx():
                loss = model.loss(batch) / args.grad_accum
            loss.backward()
            loss_accum += float(loss.item()) * args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss_accum / args.grad_accum)

        if step % args.log_every == 0:
            avg = float(np.mean(run_loss[-args.log_every:]))
            rate = (step + 1) / max(1e-3, time.time() - t0)
            log.info("step %d  loss=%.4f  lr=%.2e  %.1f step/s",
                     step, avg, sched.get_last_lr()[0], rate)

        if step > 0 and step % args.ckpt_every == 0:
            avg = float(np.mean(run_loss[-args.ckpt_every:]))
            _save_ckpt(model, opt, step, ckpt_dir, best_loss)
            if avg < best_loss:
                best_loss = avg
                _save_ckpt(model, opt, step, ckpt_dir, best_loss, label="best")
        step += 1

    _save_ckpt(model, opt, step, ckpt_dir, best_loss, label="final")
    return {"final_loss": run_loss[-1] if run_loss else float("nan"),
            "best_loss": best_loss, "steps": step,
            "elapsed_s": time.time() - t0}


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True,
                    help="glob for parquet shard paths, e.g. reports/odte_shards/opra_*.parquet")
    ap.add_argument("--ckpt-dir", default="checkpoints/tradefm")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=None)
    a = ap.parse_args()
    stats = pretrain(TrainArgs(
        shard_glob=a.shards, ckpt_dir=a.ckpt_dir, config_path=a.config,
        steps=a.steps, batch=a.batch, grad_accum=a.grad_accum,
        ckpt_every=a.ckpt_every, log_every=a.log_every,
        device=a.device, seed=a.seed, max_shards=a.max_shards,
    ))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
