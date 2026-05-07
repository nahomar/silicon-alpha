"""Held-out evaluation of a trained TradeFM checkpoint with directional head.

Loads a single-file checkpoint (from CheckpointManager.save), runs the
forward pass on a held-out time slice of multimodal shards, and reports:

  - Held-out LM cross-entropy (the trained loss)
  - Top-1 / Top-5 token accuracy
  - **Directional AUC at the configured horizon** — the trading-relevant
    metric. Compares the directional head's logits against the same
    "majority direction over next H rows of return tokens" target the
    LightGBM baseline used. AUC > 0.64 = beat the LightGBM ceiling.
  - Per-modality directional AUC — does the head help more on OPRA, ES,
    or SPY individually?

Held-out split: takes the LAST `eval_frac` (default 20%) of each shard
file, leaving the first 80% as training data. This matches the
multiday-baseline split convention so AUC is comparable.

Usage:
    python -m odte.eval.multimodal_eval \\
        --ckpt /scratch/.../checkpoints/.../final_*.pt \\
        --shards "/scratch/.../packed/multimodal/shard_*.parquet" \\
        --max-shards 100
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.config import TradeFMConfig
from odte.transformer_tradefm import TradeFM

log = logging.getLogger(__name__)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> TradeFM:
    """Load a CheckpointManager-style ckpt: dict with state, cfg, step.
    Returns a TradeFM in eval mode on `device`.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = raw["cfg"]
    cfg = TradeFMConfig(**cfg_dict)
    log.info("ckpt cfg: d_model=%d n_layers=%d ctx_len=%d "
             "modality_vocab=%d dir_head=%s step=%d",
             cfg.d_model, cfg.n_layers, cfg.ctx_len,
             cfg.modality_vocab,
             getattr(cfg, "dir_head_enabled", False),
             raw.get("step", -1))
    model = TradeFM(cfg)
    state = raw["state"]
    # Strip FSDP prefixes if any.
    cleaned = {}
    for k, v in state.items():
        new_k = k
        for prefix in ("_fsdp_wrapped_module.", "module.",
                       "_checkpoint_wrapped_module."):
            new_k = new_k.replace(prefix, "")
        cleaned[new_k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        log.warning("missing keys (first 5): %s", missing[:5])
    if unexpected:
        log.warning("unexpected keys (first 5): %s", unexpected[:5])
    model.to(device).eval()
    return model


def _build_eval_windows(shard_paths: list[Path], ctx_len: int,
                       eval_frac: float = 0.2,
                       n_features: int = 7,
                       max_windows: int | None = None
                       ) -> Iterable[tuple[torch.Tensor, torch.Tensor, int]]:
    """Yield (tokens, modality_ids, feature_offset) windows from the last
    `eval_frac` of each shard. Each window is exactly ctx_len+1 tokens
    long (model input + 1 next-token target).
    """
    n_yielded = 0
    for shard in shard_paths:
        df = pd.read_parquet(shard,
                             columns=["tokens", "modality_id"])
        n_rows = len(df)
        eval_start = int(n_rows * (1 - eval_frac))
        sub = df.iloc[eval_start:]
        if len(sub) == 0:
            continue
        # Flatten rows into a single token + modality stream.
        tok_flat = np.concatenate([
            np.asarray(t, dtype=np.int32) for t in sub["tokens"].values
        ])
        mod_per_row = sub["modality_id"].values.astype(np.int8)
        # Each row contributes n_features tokens, all with the same modality.
        mod_flat = np.repeat(mod_per_row, n_features).astype(np.int8)
        if len(tok_flat) != len(mod_flat):
            # Defensive: ragged shard. Skip.
            log.warning("shape mismatch in %s, skipping", shard)
            continue
        # Cut into ctx_len+1 windows. The first row's feature_offset is 0
        # (since we preserved row alignment by using n_features-token
        # multiples).
        consumed = 0
        L = ctx_len + 1
        while len(tok_flat) - consumed >= L:
            tokens = torch.as_tensor(tok_flat[consumed:consumed + L],
                                     dtype=torch.long)
            mids = torch.as_tensor(mod_flat[consumed:consumed + L],
                                   dtype=torch.long)
            feat_off = consumed % n_features
            yield tokens, mids, feat_off
            consumed += ctx_len  # overlap by 1 to maintain teacher-forcing
            n_yielded += 1
            if max_windows and n_yielded >= max_windows:
                return


@torch.no_grad()
def evaluate(ckpt_path: Path, shard_glob: str, eval_frac: float = 0.2,
             max_shards: int | None = None,
             max_windows: int | None = 200,
             batch: int = 4) -> dict:
    """Run held-out eval and return a result dict."""
    import glob as _glob
    shard_paths = sorted(Path(p) for p in _glob.glob(shard_glob))
    if max_shards:
        shard_paths = shard_paths[:max_shards]
    if not shard_paths:
        raise RuntimeError(f"no shards matched {shard_glob!r}")
    log.info("eval shards: %d (taking last %.0f%% of each)",
             len(shard_paths), eval_frac * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(ckpt_path, device)
    cfg = model.cfg
    n_features = int(getattr(cfg, "dir_n_features", 7))

    # Accumulators
    loss_sum = 0.0; n_tokens = 0
    top1_hits = 0; top5_hits = 0
    dir_logits_all: list[torch.Tensor] = []
    dir_targets_all: list[torch.Tensor] = []
    dir_modality_all: list[torch.Tensor] = []  # for per-modality split
    dir_target_tokens_all: list[torch.Tensor] = []  # for magnitude P&L

    win_iter = _build_eval_windows(shard_paths, cfg.ctx_len,
                                   eval_frac=eval_frac,
                                   n_features=n_features,
                                   max_windows=max_windows)
    # Batch the iterator manually.
    while True:
        batch_tokens: list[torch.Tensor] = []
        batch_mids: list[torch.Tensor] = []
        batch_feat_off: list[int] = []
        for _ in range(batch):
            try:
                t, m, fo = next(win_iter)
            except StopIteration:
                break
            batch_tokens.append(t); batch_mids.append(m); batch_feat_off.append(fo)
        if not batch_tokens:
            break

        tok = torch.stack(batch_tokens, dim=0).to(device)
        mid = torch.stack(batch_mids, dim=0).to(device)
        feat_off = torch.tensor(batch_feat_off, dtype=torch.long, device=device)

        # Forward through input window with modality ids.
        lm_logits, dir_logits = model(
            tok[:, :-1], modality_ids=mid[:, :-1], return_aux=True)
        target = tok[:, 1:]
        flat_logits = lm_logits.reshape(-1, lm_logits.size(-1))
        flat_target = target.reshape(-1)

        loss = F.cross_entropy(flat_logits, flat_target, reduction="sum")
        loss_sum += float(loss.item())
        n_tokens += flat_target.numel()

        preds = flat_logits.argmax(dim=-1)
        top1_hits += int((preds == flat_target).sum().item())
        top5 = flat_logits.topk(5, dim=-1).indices
        top5_hits += int(top5.eq(flat_target.unsqueeze(-1)).any(-1).sum().item())

        # Directional targets — same logic the model trains against.
        dir_tgt, dir_mask = model._build_dir_targets(
            tok, feature_offset=feat_off)
        if int(dir_mask.sum().item()) == 0:
            continue
        # Take the in-window modality_ids for masked positions.
        # mid[:, :-1] aligns with dir_logits / dir_tgt / dir_mask.
        mid_in = mid[:, :-1]
        # Raw target tokens at the same masked positions — needed for the
        # magnitude-weighted P&L backtest (bin-index proxy for return size).
        # `target` is shape (B, T-1); dir_mask is shape (B, T-1).
        dir_logits_all.append(dir_logits[dir_mask].float().cpu())
        dir_targets_all.append(dir_tgt[dir_mask].float().cpu())
        dir_modality_all.append(mid_in[dir_mask].cpu())
        dir_target_tokens_all.append(target[dir_mask].cpu())

    # ---- Aggregate metrics ----
    out: dict = {
        "loss": loss_sum / max(1, n_tokens),
        "ppl": float(np.exp(min(loss_sum / max(1, n_tokens), 50.0))),
        "top1_acc": top1_hits / max(1, n_tokens),
        "top5_acc": top5_hits / max(1, n_tokens),
        "n_tokens": n_tokens,
    }
    if dir_logits_all:
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        logits = torch.cat(dir_logits_all).numpy()
        proba = 1.0 / (1.0 + np.exp(-logits))
        targets = torch.cat(dir_targets_all).numpy().astype(np.int8)
        modalities = torch.cat(dir_modality_all).numpy().astype(np.int8)
        target_tokens = (torch.cat(dir_target_tokens_all).numpy().astype(np.int32)
                         if dir_target_tokens_all else None)
        preds = (proba >= 0.5).astype(np.int8)

        try:
            out["dir_auc"] = float(roc_auc_score(targets, proba))
        except ValueError:
            out["dir_auc"] = float("nan")
        try:
            out["dir_bal_acc"] = float(balanced_accuracy_score(targets, preds))
        except ValueError:
            out["dir_bal_acc"] = float("nan")
        out["dir_n"] = int(len(targets))
        out["dir_pos_rate"] = float(targets.mean())

        # ---- Magnitude-weighted P&L backtest (synthetic) ----
        # We don't have the raw prices in packed shards, so we use the bin
        # index as a proxy for return magnitude:
        #   centered_token = target_token - median(target_token)
        #   magnitude ∝ |centered_token|
        # This is "bin-units" P&L — directionally correct, magnitude-aware,
        # but NOT in dollars. For real dollar P&L we'd need to re-pack with
        # the tokenizer's edges saved so we can inverse-bin back to log-return.
        if target_tokens is not None and len(target_tokens) > 0:
            tok_med = float(np.median(target_tokens))
            centered = (target_tokens - tok_med).astype(np.float64)
            # Normalize: divide by the scale so per-trade P&L is in [-1, 1].
            scale = float(np.abs(centered).max() + 1e-9)
            ret_proxy = centered / scale  # signed magnitude in [-1, 1]

            # Trade rule: confident long if proba > thr_long, short if < thr_short.
            thr_long = 0.55
            thr_short = 0.45
            # round-trip cost per trade in units of `ret_proxy` (default 0.05
            # = 5% of max bin range — rough proxy for retail-level spread+fee).
            cost = 0.05

            position = np.where(proba > thr_long, 1.0,
                                np.where(proba < thr_short, -1.0, 0.0))
            traded = position != 0
            pnl_per_trade = position * ret_proxy - traded * cost
            cum_pnl = np.cumsum(pnl_per_trade)

            n_trades = int(traded.sum())
            n_wins = int(((position * ret_proxy) > cost)[traded].sum()) if n_trades else 0
            mean_pnl = float(pnl_per_trade.mean()) if len(pnl_per_trade) else 0.0
            std_pnl = float(pnl_per_trade.std() + 1e-12)
            sharpe = mean_pnl / std_pnl * np.sqrt(252)  # annualized assuming daily samples (rough)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdown = (cum_pnl - running_max).min() if len(cum_pnl) else 0.0

            # Magnitude split: average win vs average loss
            wins_mag = pnl_per_trade[traded & (pnl_per_trade > 0)]
            losses_mag = pnl_per_trade[traded & (pnl_per_trade < 0)]

            out["bt"] = {
                "thr_long": thr_long, "thr_short": thr_short,
                "cost_per_round_trip": cost,
                "n_predictions": int(len(proba)),
                "n_trades": n_trades,
                "trade_rate": float(n_trades / max(1, len(proba))),
                "n_wins": n_wins,
                "win_rate": float(n_wins / max(1, n_trades)),
                "total_pnl_proxy": float(cum_pnl[-1]) if len(cum_pnl) else 0.0,
                "mean_pnl_per_obs": mean_pnl,
                "sharpe_annualized": float(sharpe),
                "max_drawdown_proxy": float(drawdown),
                "avg_win_size": float(wins_mag.mean()) if len(wins_mag) else 0.0,
                "avg_loss_size": float(losses_mag.mean()) if len(losses_mag) else 0.0,
                "win_loss_ratio": (float(abs(wins_mag.mean() / losses_mag.mean()))
                                   if len(wins_mag) and len(losses_mag) and losses_mag.mean() != 0
                                   else float("nan")),
            }

        # Per-modality AUC.
        per_mod: dict[int, dict] = {}
        for m in np.unique(modalities):
            mask = modalities == m
            if mask.sum() < 10 or len(set(targets[mask])) < 2:
                continue
            try:
                m_auc = float(roc_auc_score(targets[mask], proba[mask]))
                m_bal = float(balanced_accuracy_score(targets[mask], preds[mask]))
            except ValueError:
                continue
            per_mod[int(m)] = {"auc": m_auc, "bal_acc": m_bal,
                               "n": int(mask.sum()),
                               "pos_rate": float(targets[mask].mean())}
        out["per_modality"] = per_mod

    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--shards", required=True,
                    help="glob for packed multimodal shards")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    res = evaluate(args.ckpt, args.shards,
                   eval_frac=args.eval_frac,
                   max_shards=args.max_shards,
                   max_windows=args.max_windows,
                   batch=args.batch)
    print()
    print("===== HELD-OUT MULTIMODAL EVAL =====")
    print(f"loss              : {res['loss']:.4f}")
    print(f"perplexity        : {res['ppl']:.2f}")
    print(f"top-1 token acc   : {res['top1_acc']*100:.2f}%")
    print(f"top-5 token acc   : {res['top5_acc']*100:.2f}%")
    print(f"n_tokens scored   : {res['n_tokens']:,}")
    if "dir_auc" in res:
        print()
        print("===== DIRECTIONAL HEAD (h=10 majority direction) =====")
        print(f"AUC               : {res['dir_auc']:.4f}  "
              f"(LightGBM baseline = 0.64)")
        print(f"balanced acc      : {res['dir_bal_acc']*100:.2f}%")
        print(f"positive rate     : {res['dir_pos_rate']*100:.2f}%")
        print(f"n labeled         : {res['dir_n']:,}")
        if res.get("per_modality"):
            print()
            print("Per-modality AUC:")
            mod_names = {0: "OPRA", 1: "ES", 2: "SPY"}
            for m, mres in sorted(res["per_modality"].items()):
                name = mod_names.get(m, f"mod{m}")
                print(f"  {name:6s}  AUC={mres['auc']:.4f}  "
                      f"bal={mres['bal_acc']*100:.2f}%  "
                      f"n={mres['n']:,}  pos={mres['pos_rate']*100:.2f}%")
        print()
        verdict = ("BREAK" if res['dir_auc'] >= 0.66 else
                   ("MARGINAL" if res['dir_auc'] >= 0.64 else
                    "FAIL"))
        print(f"VERDICT vs LightGBM 0.64 ceiling: {verdict}")

        # Magnitude-weighted backtest (bin-units, NOT dollars)
        bt = res.get("bt")
        if bt:
            print()
            print("===== MAGNITUDE-WEIGHTED BACKTEST (bin-units, synthetic) =====")
            print(f"trade rule        : LONG if proba > {bt['thr_long']}, "
                  f"SHORT if proba < {bt['thr_short']}, else FLAT")
            print(f"cost/round-trip   : {bt['cost_per_round_trip']:.3f} "
                  f"(in [-1,1] bin-units)")
            print(f"predictions       : {bt['n_predictions']:,}")
            print(f"trades taken      : {bt['n_trades']:,}  "
                  f"({bt['trade_rate']*100:.2f}%)")
            print(f"wins              : {bt['n_wins']:,}  "
                  f"({bt['win_rate']*100:.2f}% of trades)")
            print(f"avg win size      : {bt['avg_win_size']:+.4f}")
            print(f"avg loss size     : {bt['avg_loss_size']:+.4f}")
            print(f"win/loss ratio    : {bt['win_loss_ratio']:.3f}")
            print(f"total P&L (proxy) : {bt['total_pnl_proxy']:+.4f}")
            print(f"Sharpe (annualized, rough proxy): {bt['sharpe_annualized']:+.3f}")
            print(f"max drawdown      : {bt['max_drawdown_proxy']:+.4f}")
            print()
            print("Caveats:")
            print("  - P&L is in arbitrary bin-units, NOT dollars. To get")
            print("    dollar P&L we'd need to inverse-tokenize using the")
            print("    saved tokenizer edges, which our packer doesn't yet")
            print("    persist. Action item: re-pack with tokenizer save")
            print("    + add price reconstruction → real backtest in V2.")
            print("  - Sharpe is annualized assuming daily-frequency samples;")
            print("    the actual sample frequency depends on the modality")
            print("    mix, so this number is rough comparative-only.")
            print("  - Cost = 0.05 of [-1,1] bin range is a guess at retail")
            print("    spread+fee; tune to match the actual bid-ask of your")
            print("    target instrument before drawing any conclusions.")
