"""Phase-2 distributed TradeFM trainer.

Multi-node, multi-GPU training with:
  - FSDP (ZeRO-3-equivalent sharding)
  - Mixed precision: bf16 compute + fp32 master weights
  - Optional FP8 via transformer-engine (cfg.fp8, H100 only)
  - FlashAttention-3 via flash_attn (cfg.use_flash_attn, H100 only)
  - Activation checkpointing for memory savings at 524M
  - Sharded checkpointing via odte.train.checkpoint.CheckpointManager
  - W&B logging from rank 0

Launch:
    torchrun --nproc_per_node=8 --nnodes=$N --node_rank=$R \
        --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
        -m odte.train.distributed \
        --config configs/tradefm_524m.yml \
        --shards 's3://my-bucket/opra/opra_*.parquet' \
        --ckpt-store s3://my-bucket/ckpts \
        --steps 200000 --batch 16 --grad-accum 4

Single-node dry-run on Mac / any CPU is supported with nproc_per_node=1.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from odte.transformer_tradefm import TradeFM, TransformerBlock, wrap_fp8_autocast
from odte.train.pretrain_tradefm import ShardTokenDataset, load_config
from odte.train.checkpoint import CheckpointManager
from odte.train.eval_loop import evaluate, load_shards

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rank-aware sharding helper
# ---------------------------------------------------------------------------

class ShardedTokenDataset(ShardTokenDataset):
    """Rank-partitioned over shards: each rank reads only its assigned shards.

    Previous implementation had every rank read every shard and filter by
    sample index mod world_size. That's correct but wastes N-fold S3 egress
    and adds a cross-rank correlation risk when two ranks happen to align on
    the same packed window.

    This implementation:
      1. Shuffles the shard *order* using `seed` (consistent across ranks)
         so all ranks agree on the global shard ordering.
      2. Assigns each rank a disjoint slice of shards at positions
         {rank, rank+W, rank+2W, ...} of the shuffled list.
      3. The parent class then shuffles rows *within* each shard using
         `seed + rank`, giving each rank an independent within-shard order.

    Per-epoch each rank sees disjoint data; together all ranks cover the
    corpus. I/O is balanced at O(N/W) shards per rank instead of O(N).
    """

    def __init__(self, shard_paths, ctx_len, rank: int, world_size: int,
                 shuffle_buffer: int = 64, seed: int = 0):
        all_shards = sorted(Path(p) for p in shard_paths)
        # Shared-seed shuffle so every rank agrees on the global shard order.
        perm_rng = np.random.default_rng(seed)
        order = np.arange(len(all_shards))
        perm_rng.shuffle(order)
        my_shards = [all_shards[order[i]]
                     for i in range(rank, len(order), world_size)]
        # Parent uses seed+rank for per-rank within-shard row shuffle.
        super().__init__(my_shards, ctx_len, shuffle_buffer=shuffle_buffer,
                         seed=seed + rank)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        yield from super().__iter__()


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def _init_dist() -> tuple[int, int, int]:
    """Initialize c10d and return (rank, world_size, local_rank)."""
    if dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count() if torch.cuda.is_available() else 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def _device_for(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------

def _wrap_fsdp(model: TradeFM, device: torch.device, world_size: int) -> torch.nn.Module:
    """FSDP wrap when distributed, else return model as-is."""
    if world_size <= 1 or not torch.cuda.is_available():
        return model.to(device)
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision, ShardingStrategy, CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    import functools

    mp = MixedPrecision(param_dtype=torch.bfloat16,
                        reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={TransformerBlock})
    wrapped = FSDP(model.to(device),
                   auto_wrap_policy=policy,
                   mixed_precision=mp,
                   sharding_strategy=ShardingStrategy.FULL_SHARD,
                   device_id=device,
                   limit_all_gathers=True,
                   use_orig_params=True)
    # Activation checkpointing for memory at 524M.
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper, apply_activation_checkpointing, CheckpointImpl,
        )
        apply_activation_checkpointing(
            wrapped,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, TransformerBlock),
        )
    except Exception as e:
        log.warning("activation checkpointing not applied: %s", e)
    return wrapped


# ---------------------------------------------------------------------------
# Logging (rank 0 only)
# ---------------------------------------------------------------------------

@dataclass
class RankLogger:
    rank: int
    wandb_project: Optional[str] = None
    _wandb: Optional[object] = None

    def __post_init__(self):
        if self.rank == 0 and self.wandb_project:
            try:
                import wandb
                self._wandb = wandb.init(project=self.wandb_project,
                                         config={"launched_at": time.time()})
            except Exception as e:
                log.warning("wandb init failed: %s", e)

    def scalar(self, **kvs) -> None:
        if self.rank != 0:
            return
        if self._wandb is not None:
            try:
                self._wandb.log(kvs)
            except Exception:
                pass
        log.info("  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in kvs.items()))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> dict:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Backfill defaults so direct train(...) calls (not via CLI) don't crash
    for attr, default in (("eval_shards", None), ("eval_every", 0),
                          ("eval_max_batches", 50)):
        if not hasattr(args, attr):
            setattr(args, attr, default)
    rank, world, local_rank = _init_dist()
    device = _device_for(local_rank)
    log.info("[rank %d/%d] device=%s", rank, world, device)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    cfg = load_config(args.config)
    model = TradeFM(cfg)
    model = _wrap_fsdp(model, device, world)
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        log.info("TradeFM wrapped  params≈%d  cfg=%s", n_params, asdict(cfg))

    # Optimizer AFTER FSDP wrap (AdamW; parameters are flattened per-shard)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    warmup = cfg.warmup_steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warmup) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, args.steps - warmup))),
    )

    # Checkpoint manager (stores one shard per rank)
    ckpt = CheckpointManager(store_url=args.ckpt_store,
                             prefix=args.ckpt_prefix, rank=rank, world_size=world)
    # Pass current_cfg so CheckpointManager enforces cfg_hash parity and
    # refuses to resume a ckpt that was saved with a different config.
    meta = (ckpt.load(model, opt, current_cfg=asdict(cfg))
            if args.resume else {"step": 0, "best_loss": float("inf")})
    start_step = int(meta["step"]); best_loss = float(meta["best_loss"])

    # ---- Eval-only short-circuit ----------------------------------------
    # Loaded ckpt; instead of running the training loop, run one evaluate()
    # pass on --eval-shards and exit. Used by the post-training Modal
    # eval_524m function so we get the held-out metrics without re-running
    # all of training.
    if args.eval_only:
        if not args.eval_shards:
            raise RuntimeError("--eval-only requires --eval-shards")
        from odte.train.eval_loop import evaluate, load_shards
        eval_paths = load_shards(args.eval_shards)
        if not eval_paths:
            raise RuntimeError(f"no eval shards for {args.eval_shards!r}")
        if rank == 0:
            log.info("[eval-only] %d eval shards from %s",
                     len(eval_paths), args.eval_shards)
        device_for_eval = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        result = evaluate(
            model, eval_paths, ctx_len=cfg.ctx_len, vocab=cfg.vocab,
            device=device_for_eval, batch=args.batch,
            max_batches=args.eval_max_batches,
        )
        if world > 1:
            # Aggregate per-rank counts via all_reduce so rank 0 reports the
            # global metrics (ranks see disjoint shard slices).
            t = torch.tensor([
                result.loss * result.n_tokens, float(result.n_tokens),
                result.top1_acc * result.n_tokens,
                result.top5_acc * result.n_tokens,
                result.directional_acc * result.n_tokens,
            ], device=device_for_eval, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            n = max(float(t[1].item()), 1.0)
            agg_loss = float(t[0].item() / n)
            agg_top1 = float(t[2].item() / n)
            agg_top5 = float(t[3].item() / n)
            agg_dir = float(t[4].item() / n)
        else:
            agg_loss = result.loss; agg_top1 = result.top1_acc
            agg_top5 = result.top5_acc; agg_dir = result.directional_acc
        if rank == 0:
            ppl = math.exp(min(agg_loss, 50.0))
            verdict = "SIGNAL" if agg_dir >= 0.53 else "sub-threshold"
            print()
            print(f"[eval-only] ===== HELD-OUT METRICS =====")
            print(f"[eval-only] eval shards   : {len(eval_paths)}")
            print(f"[eval-only] step loaded   : {start_step}")
            print(f"[eval-only] eval loss     : {agg_loss:.4f}")
            print(f"[eval-only] perplexity    : {ppl:.2f}")
            print(f"[eval-only] top-1 acc     : {agg_top1*100:.2f}%")
            print(f"[eval-only] top-5 acc     : {agg_top5*100:.2f}%")
            print(f"[eval-only] directional   : {agg_dir*100:.2f}%  (>= 53% threshold)")
            print(f"[eval-only] verdict       : {verdict}")
        return {"loss": agg_loss, "top1": agg_top1, "directional": agg_dir,
                "step": start_step}

    # Data — support comma-separated glob patterns so multi-day corpora can
    # be passed as one arg (e.g. "/shards/day1/*.parquet,/shards/day2/*.parquet").
    # A single pattern with no commas behaves exactly as before.
    import glob as _glob
    patterns = [p.strip() for p in args.shards.split(",") if p.strip()]
    matched: set[Path] = set()
    for pat in patterns:
        for p in _glob.glob(pat):
            matched.add(Path(p))
    shard_paths = sorted(matched)
    if args.max_shards:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise RuntimeError(f"no shards for {args.shards!r}")
    ds = ShardedTokenDataset(shard_paths, ctx_len=cfg.ctx_len,
                             rank=rank, world_size=world, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    rlog = RankLogger(rank=rank, wandb_project=args.wandb)

    loader_iter = iter(loader)
    t0 = time.time()
    step = start_step
    run_loss: list[float] = []
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = batch.to(device, non_blocking=True)
            with wrap_fp8_autocast():
                # Route through the FSDP-wrapped __call__ (not model.module.loss)
                # so the pre-forward all-gather hook fires and unshards the
                # root flat-param. Calling `model.module.loss(batch)` would
                # bypass FSDP.__call__ entirely — nn.Embedding.forward then
                # sees tok_emb.weight as a 1-D shard view and F.embedding
                # raises "weight must be 2-D". Caught by the 8-GPU smoke.
                logits = model(batch[:, :-1])
                target = batch[:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                )
            (loss / args.grad_accum).backward()
            loss_accum += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        # All-reduce loss for logging
        if world > 1:
            t = torch.tensor(loss_accum / args.grad_accum, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            step_loss = float(t.item())
        else:
            step_loss = loss_accum / args.grad_accum
        run_loss.append(step_loss)

        if step % args.log_every == 0:
            avg = float(np.mean(run_loss[-args.log_every:]))
            rlog.scalar(step=step, loss=avg, lr=sched.get_last_lr()[0],
                        step_per_s=(step - start_step + 1) / max(1e-3, time.time() - t0))

        if step > 0 and step % args.ckpt_every == 0:
            ckpt.save(model, opt, step, best_loss, asdict(cfg), label="ckpt")
            avg = float(np.mean(run_loss[-args.ckpt_every:]))
            if avg < best_loss:
                best_loss = avg
                ckpt.save(model, opt, step, best_loss, asdict(cfg), label="best")

        # Diagnostic: print whether the eval condition is being checked at
        # all. Once per ckpt boundary on rank 0 — proves the eval-gate is
        # being reached even if it doesn't fire (e.g., args.eval_shards is
        # None). Cheap; helps when Modal logs age out.
        if (rank == 0 and args.ckpt_every > 0
                and step > 0 and step % args.ckpt_every == 0):
            print(f"[eval-gate@{step}] eval_shards={bool(args.eval_shards)} "
                  f"eval_every={args.eval_every} "
                  f"step%eval_every={step % max(args.eval_every, 1)}",
                  flush=True)

        # Held-out eval — all ranks evaluate (distributed eval). See the
        # 2026-04-25 analysis in the FSDP load saga for why this is the
        # primary metric path; the post-hoc CheckpointManager.load route
        # has a non-isomorphic FSDP-tree problem we can't bridge cheaply.
        if (args.eval_shards and step > 0 and args.eval_every > 0
                and step % args.eval_every == 0):
            try:
                if rank == 0:
                    print(f"[eval-fire@{step}] starting eval pass...",
                          flush=True)
                ev_paths = load_shards(args.eval_shards)
                ev = evaluate(model, ev_paths, ctx_len=cfg.ctx_len,
                              vocab=cfg.vocab, device=device,
                              batch=args.batch, max_batches=args.eval_max_batches)
                # Aggregate across ranks so rank 0 reports the global metric.
                if world > 1:
                    t = torch.tensor([
                        ev.loss * ev.n_tokens, float(ev.n_tokens),
                        ev.top1_acc * ev.n_tokens,
                        ev.top5_acc * ev.n_tokens,
                        ev.directional_acc * ev.n_tokens,
                    ], device=device, dtype=torch.float64)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    n = max(float(t[1].item()), 1.0)
                    agg_loss = float(t[0].item() / n)
                    agg_top1 = float(t[2].item() / n)
                    agg_top5 = float(t[3].item() / n)
                    agg_dir = float(t[4].item() / n)
                else:
                    agg_loss = ev.loss; agg_top1 = ev.top1_acc
                    agg_top5 = ev.top5_acc; agg_dir = ev.directional_acc
                if rank == 0:
                    ppl = math.exp(min(agg_loss, 50.0))
                    verdict = "SIGNAL" if agg_dir >= 0.53 else "sub-thresh"
                    print(f"[eval@{step:>6d}] loss={agg_loss:.4f} ppl={ppl:.2f} "
                          f"top1={agg_top1*100:.2f}% top5={agg_top5*100:.2f}% "
                          f"dir={agg_dir*100:.2f}% [{verdict}]", flush=True)
                    rlog.scalar(step=step, eval_loss=agg_loss,
                                eval_top1=agg_top1, eval_top5=agg_top5,
                                eval_dir_acc=agg_dir, eval_ppl=ppl)
                    # File-based persistence — durable on the Modal volume
                    # even after log retention ages out. Append JSON-line
                    # so each eval call adds an entry without overwriting.
                    if args.ckpt_store:
                        eval_log_path = Path(args.ckpt_store) / "eval_log.jsonl"
                        eval_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(eval_log_path, "a") as fh:
                            import json as _json
                            fh.write(_json.dumps({
                                "step": step,
                                "eval_loss": agg_loss,
                                "eval_ppl": ppl,
                                "eval_top1": agg_top1,
                                "eval_top5": agg_top5,
                                "eval_dir_acc": agg_dir,
                                "verdict": verdict,
                            }) + "\n")
                        print(f"[eval@{step}] appended to {eval_log_path}",
                              flush=True)
            except Exception as e:
                if rank == 0:
                    print(f"[eval@{step}] FAILED: {type(e).__name__}: {e}",
                          flush=True)
                log.warning("eval skipped: %s", e)
        step += 1

    # Final save
    ckpt.save(model, opt, step, best_loss, asdict(cfg), label="final")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return {"final_loss": run_loss[-1] if run_loss else float("nan"),
            "best_loss": best_loss, "steps": step,
            "elapsed_s": time.time() - t0, "rank": rank}


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--ckpt-store", default="checkpoints/tradefm_dist")
    ap.add_argument("--ckpt-prefix", default="tradefm")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", default=None, help="W&B project name (rank 0 only)")
    # Held-out eval (rank 0 runs it; NOT synchronized across ranks)
    ap.add_argument("--eval-shards", default=None,
                    help="glob of reserved eval shards; if set, eval every --eval-every steps")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-max-batches", type=int, default=50)
    # Eval-only mode: skip training, load latest ckpt via --resume, run one
    # full evaluate() pass on --eval-shards, print metrics, exit.
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training; load ckpt via --resume and run one eval pass")
    # LR-finder short-circuit: if set, skip normal training and run the sweep
    ap.add_argument("--find-lr", action="store_true",
                    help="run the LR finder instead of training")
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--lr-max", type=float, default=1e-1)
    ap.add_argument("--lr-steps", type=int, default=200)
    a = ap.parse_args()

    if a.find_lr:
        from odte.train.lr_finder import find_lr, maybe_plot
        from pathlib import Path as _P
        cfg = load_config(a.config)
        model = TradeFM(cfg)
        dev = _device_for(0)
        import glob as _g
        shards = sorted(_P(p) for p in _g.glob(a.shards))
        if not shards:
            raise RuntimeError(f"no shards for {a.shards!r}")
        result = find_lr(model, shards, dev, lr_min=a.lr_min, lr_max=a.lr_max,
                         steps=a.lr_steps, batch=a.batch)
        out_dir = _P("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "lr_finder.json").write_text(json.dumps(result, indent=2))
        maybe_plot(result, out_dir / "lr_finder.png")
        print(json.dumps({k: result[k] for k in
                          ("suggested_lr", "min_loss_lr", "min_loss")}, indent=2))
        return
    stats = train(a)
    if stats["rank"] == 0:
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
