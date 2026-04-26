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
    # New (Phase-2 eval debugging): directional restricted to *return-feature*
    # token positions only, with the threshold computed from the actual return
    # token distribution. The cross-feature `directional_acc` above is
    # corrupted by feature-rotation patterns (the model learns the every-7-
    # tokens feature-cycle and that dominates the metric). `directional_ret`
    # is the trading-relevant number: did we predict the price-direction half
    # correctly on positions where the *next token is a return*?
    directional_ret_acc: float = float("nan")
    directional_ret_n: int = 0
    n_tokens: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {"eval_loss": self.loss, "eval_top1": self.top1_acc,
                "eval_top5": self.top5_acc, "eval_dir_acc": self.directional_acc,
                "eval_dir_ret_acc": self.directional_ret_acc,
                "eval_dir_ret_n": float(self.directional_ret_n),
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
             batch: int = 16, max_batches: int | None = 200,
             n_features: int = 7) -> EvalResult:
    """Held-out eval. Computes loss + top-1 + top-5 + two directional
    metrics:
      - directional_acc: legacy cross-feature, kept for back-compat. Inflated
        by the model learning the every-N-token feature-rotation pattern.
      - directional_ret_acc (NEW): restricted to return-feature positions
        only, using the return-token distribution's median as threshold.
        This is the trading-relevant number.

    Return positions are identified using the dataset's feature_offset
    metadata: at window position p with feature_offset f, the feature
    index is `(p + f) % n_features`. Return = feature index 0 (per
    DataShopPacker.feature_spec, which lists `ret` first).
    """
    model.eval()
    ds = ShardTokenDataset(shard_paths, ctx_len=ctx_len, seed=0,
                           n_features=n_features, with_feature_offset=True)
    # batch_size > 1 mixes feature_offsets in the same batch — handle in loop.
    loader = DataLoader(ds, batch_size=batch, num_workers=0,
                        collate_fn=_collate_with_feature_offset)

    loss_sum = 0.0
    tokens_sum = 0
    top1_hits = 0; top5_hits = 0; tok_cnt = 0
    dir_hits = 0; dir_total = 0
    # Return-only directional accumulators
    ret_preds: list[torch.Tensor] = []
    ret_targets: list[torch.Tensor] = []

    for i, (batch_tok, feat_offsets) in enumerate(loader):
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

        # Identify return-feature positions per row in the batch.
        # Window position p has feature index (p + offset) % n_features.
        # Targets correspond to position p+1 in the original window (since
        # we shifted by 1 for next-token prediction). So the target at
        # batch_tok[:, p+1] has feature index (p + 1 + offset) % n_features.
        B, Tm1 = target.shape
        pred_2d = preds.view(B, Tm1)
        for b in range(B):
            off = int(feat_offsets[b])
            # Target position p (within shifted target) corresponds to
            # original window position p+1; feature idx = (p+1+off) % N.
            positions = torch.arange(Tm1, device=device)
            feat_idx = (positions + 1 + off) % n_features
            mask = (feat_idx == 0)  # feature 0 = return
            if mask.any():
                ret_preds.append(pred_2d[b, mask].cpu())
                ret_targets.append(target[b, mask].cpu())

    # Compute return-only directional from the accumulated return-position
    # predictions. Threshold = median of all collected return-token targets.
    ret_dir_acc = float("nan")
    ret_n = 0
    if ret_preds:
        rp = torch.cat(ret_preds).float()
        rt = torch.cat(ret_targets).float()
        median = rt.median()
        p_above = rp > median
        t_above = rt > median
        valid = (rp != median) & (rt != median)
        if int(valid.sum().item()) > 0:
            ret_dir_acc = float(((p_above == t_above) & valid).float().mean().item())
            ret_n = int(valid.sum().item())

    model.train()
    return EvalResult(
        loss=(loss_sum / max(1, tokens_sum)),
        top1_acc=(top1_hits / max(1, tok_cnt)),
        top5_acc=(top5_hits / max(1, tok_cnt)),
        directional_acc=(dir_hits / max(1, dir_total)) if dir_total else float("nan"),
        directional_ret_acc=ret_dir_acc,
        directional_ret_n=ret_n,
        n_tokens=tok_cnt,
    )


def _collate_with_feature_offset(batch):
    """DataLoader collate for ShardTokenDataset with with_feature_offset=True.
    Each item is (tokens_tensor, feat_offset_int). Stack tokens; collect
    offsets in a CPU LongTensor."""
    tokens = torch.stack([b[0] for b in batch], dim=0)
    offsets = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return tokens, offsets


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
