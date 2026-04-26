"""Held-out evaluation for TradeFM pretraining.

Reports four numbers on a reserved shard set:
  - loss            standard CE (same as training objective)
  - top1_acc        P(argmax_pred == target)
  - top5_acc        P(target ∈ top-5)
  - directional_acc P(sign(decoded_pred) == sign(decoded_target))

Directional accuracy uses the fact that HybridBinTokenizer produces
quantile-ordered buckets for price/return features — so "bucket > vocab/2"
is an up move and "bucket < vocab/2" is a down move. Tokens in the median
bucket are treated as neutral and excluded from the denominator.

Called from odte.train.distributed via --eval-shards / --eval-every.
"""
from __future__ import annotations

import glob as _glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from odte.train.pretrain_tradefm import ShardTokenDataset

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    loss: float
    top1_acc: float
    top5_acc: float
    directional_acc: float
    n_tokens: int

    def to_dict(self) -> Dict[str, float]:
        return {"eval_loss": self.loss, "eval_top1": self.top1_acc,
                "eval_top5": self.top5_acc, "eval_dir_acc": self.directional_acc,
                "eval_n_tokens": float(self.n_tokens)}


def _directional(preds: torch.Tensor, targets: torch.Tensor, vocab: int
                 ) -> tuple[int, int]:
    """Return (hits, total) for directional accuracy.

    'Directional' here = "did we predict a token in the same half of the
    actual target distribution as the true target?" This is a tractable
    proxy for price direction when feature tokens are bin indices —
    above-median bin = "high" half, below = "low" half.

    NOTE on the threshold: we use the per-batch *target median* as the
    cutoff, NOT `vocab // 2`. The HybridBinTokenizer with 7 features ×
    64 buckets uses ~448 token IDs out of vocab=4096, so almost every
    token lands in [0, 448) and `vocab // 2 = 2048` would always
    classify everything as 'low' → spurious 100% agreement. Per-batch
    target-median makes the threshold data-relative and meaningful.
    """
    targets_f = targets.float()
    preds_f = preds.float()
    median = targets_f.median()
    p_sign = torch.sign(preds_f - median)
    t_sign = torch.sign(targets_f - median)
    valid = (p_sign != 0) & (t_sign != 0)
    hits = int(((p_sign == t_sign) & valid).sum().item())
    total = int(valid.sum().item())
    return hits, total


@torch.no_grad()
def evaluate(model: torch.nn.Module, shard_paths: List[Path],
             ctx_len: int, vocab: int, device: torch.device,
             batch: int = 16, max_batches: int | None = 200) -> EvalResult:
    model.eval()
    ds = ShardTokenDataset(shard_paths, ctx_len=ctx_len, seed=0)
    loader = DataLoader(ds, batch_size=batch, num_workers=0)

    loss_sum = 0.0
    tokens_sum = 0
    top1_hits = 0; top5_hits = 0; tok_cnt = 0
    dir_hits = 0; dir_total = 0
    for i, batch_tok in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch_tok = batch_tok.to(device)
        logits = _forward_logits(model, batch_tok[:, :-1])  # (B, T-1, V)
        target = batch_tok[:, 1:]                            # (B, T-1)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_target = target.reshape(-1)

        loss = F.cross_entropy(flat_logits, flat_target, reduction="sum")
        loss_sum += float(loss.item())
        tokens_sum += flat_target.numel()

        preds = flat_logits.argmax(dim=-1)
        top1_hits += int((preds == flat_target).sum().item())
        top5 = flat_logits.topk(5, dim=-1).indices
        top5_hits += int(top5.eq(flat_target.unsqueeze(-1)).any(-1).sum().item())
        tok_cnt += flat_target.numel()

        dh, dt = _directional(preds, flat_target, vocab)
        dir_hits += dh; dir_total += dt

    model.train()
    return EvalResult(
        loss=(loss_sum / max(1, tokens_sum)),
        top1_acc=(top1_hits / max(1, tok_cnt)),
        top5_acc=(top5_hits / max(1, tok_cnt)),
        directional_acc=(dir_hits / max(1, dir_total)) if dir_total else float("nan"),
        n_tokens=tok_cnt,
    )


def _forward_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Call the model's wrapped __call__ so FSDP's pre-forward all-gather
    fires and unshards the root flat-param. Calling model.module(tokens)
    bypasses the wrapper and reads sharded 1-D weight views, which breaks
    nn.Embedding the same way it broke training (caught in 11103cc)."""
    return model(tokens)


def load_shards(glob_pattern: str) -> List[Path]:
    paths = sorted(Path(p) for p in _glob.glob(glob_pattern))
    if not paths:
        raise RuntimeError(f"no eval shards matched {glob_pattern!r}")
    return paths
