"""End-to-end training orchestration.

Pipeline:
    1. build_panel()                     — monolithic dataset
    2. fit tokenizer
    3. train transformer next-token model
    4. train diffusion model on continuous returns
    5. (optional) use world_model rollouts to augment RL training data
    6. train PPO agent on real returns + features
    7. write checkpoints + a training report

CPU defaults are TINY so everything smoke-tests fast. Scale via
`Config(d_model=..., n_layers=..., batch_size=..., etc.)`.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .unified_dataset import build_panel
from .tokenizer import QuantileTokenizer, build_tokenizer
from .transformer import MarketTransformer
from .diffusion import DDPM
from .world_model import WorldModel
from .rl_env import PortfolioEnv
from .rl_agent import train_ppo

log = logging.getLogger("models.train")


def _windowize(arr: np.ndarray, window: int):
    """Return (num_windows, window, *tail_shape)."""
    T = arr.shape[0]
    if T <= window:
        return arr[None]
    windows = [arr[i: i + window] for i in range(T - window)]
    return np.stack(windows)


def train_transformer(tokens: np.ndarray, sent: np.ndarray, cfg: Config,
                      epochs: int = 1) -> MarketTransformer:
    device = torch.device(cfg.device)
    n_tickers = tokens.shape[1]
    model = MarketTransformer(n_tickers, cfg.n_return_buckets, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    W = cfg.context_len
    X = _windowize(tokens, W)         # (B, W, N)
    S = _windowize(sent, W)           # (B, W)
    Xt = torch.tensor(X, dtype=torch.long, device=device)
    St = torch.tensor(S, dtype=torch.float32, device=device)
    B = cfg.batch_size
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        losses = []
        for i in range(0, len(Xt), B):
            idx = perm[i: i + B]
            loss = model.loss(Xt[idx], St[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        log.info("transformer epoch %d  loss=%.4f", ep, float(np.mean(losses)))
    return model


def train_diffusion(ret: np.ndarray, cfg: Config, epochs: int = 1) -> DDPM:
    device = torch.device(cfg.device)
    n_ch = ret.shape[1]
    ddpm = DDPM(n_ch, cfg)
    opt = torch.optim.AdamW(ddpm.model.parameters(), lr=cfg.lr)
    L = cfg.synth_seq_len
    windows = _windowize(ret, L)            # (B, L, N)
    X = torch.tensor(windows.transpose(0, 2, 1), dtype=torch.float32, device=device)  # (B,N,L)
    B = cfg.batch_size
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        losses = []
        for i in range(0, len(X), B):
            idx = perm[i: i + B]
            loss = ddpm.loss(X[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        log.info("diffusion epoch %d  loss=%.4f", ep, float(np.mean(losses)))
    return ddpm


def run(epochs_tx: int = 1, epochs_diff: int = 1, ppo_steps: int = 1500) -> Path:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = Config(); cfg.ensure_dirs()

    log.info("Building unified panel…")
    prices, features, _ = build_panel(cfg)
    ret_cols = [c for c in features.columns if c.startswith("ret_")]
    ret_df = features[ret_cols]
    sent = features["sentiment"].values.astype(np.float32)

    log.info("Fitting tokenizer…")
    tok = build_tokenizer(ret_df, cfg)
    tok.save(cfg.checkpoints_dir / "tokenizer.json")
    tokens = tok.encode(ret_df).values.astype(np.int64)
    log.info("Tokens shape %s  sentiment shape %s", tokens.shape, sent.shape)

    log.info("Training transformer…")
    tx = train_transformer(tokens, sent, cfg, epochs=epochs_tx)
    torch.save({"state": tx.state_dict(),
                "n_tickers": tokens.shape[1],
                "vocab": cfg.n_return_buckets},
               cfg.checkpoints_dir / "transformer.pt")

    log.info("Training diffusion…")
    ret_np = ret_df.values.astype(np.float32)
    ddpm = train_diffusion(ret_np, cfg, epochs=epochs_diff)
    torch.save(ddpm.model.state_dict(), cfg.checkpoints_dir / "ddpm.pt")

    log.info("Sampling synthetic paths…")
    synth = ddpm.sample(n=8, seq_len=cfg.synth_seq_len, n_channels=ret_np.shape[1])
    np.save(cfg.synthetic_dir / f"synth_{int(time.time())}.npy", synth.cpu().numpy())

    log.info("Building RL env + training PPO…")
    # Real returns (simple, not log) for the env
    simple_ret = np.exp(ret_np) - 1.0
    feats = features.values.astype(np.float32)
    # Align: drop rows where any feature is nan
    valid = ~np.isnan(feats).any(axis=1)
    simple_ret = simple_ret[valid]
    feats = feats[valid]
    env = PortfolioEnv(returns=simple_ret, features=feats, cfg=cfg)
    ac = train_ppo(env, cfg, total_steps=ppo_steps, rollout_len=min(256, ppo_steps))
    torch.save(ac.state_dict(), cfg.checkpoints_dir / "ppo_actor_critic.pt")

    report = cfg.reports_dir / f"train_report_{time.strftime('%Y%m%dT%H%M%S')}.md"
    report.write_text(f"""# Training report — {time.strftime('%Y-%m-%d %H:%M')}

- Panel: {features.shape[0]} rows × {features.shape[1]} features
- Tokenizer: {cfg.n_return_buckets} buckets, saved to `tokenizer.json`
- Transformer: d_model={cfg.d_model}, heads={cfg.n_heads}, layers={cfg.n_layers}
- Diffusion: {cfg.diffusion_steps} steps, hidden={cfg.diffusion_hidden}
- PPO: {ppo_steps} env steps, commission={cfg.commission_bps} bps

Artifacts in `checkpoints/`:
  tokenizer.json, transformer.pt, ddpm.pt, ppo_actor_critic.pt

Synthetic paths in `synthetic/`.

Caveat: short training run intended as a smoke test. To get real signal, scale
epochs, context_len, d_model, and use walk-forward validation.
""")
    log.info("Report → %s", report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx-epochs", type=int, default=1)
    ap.add_argument("--diff-epochs", type=int, default=1)
    ap.add_argument("--ppo-steps", type=int, default=1500)
    a = ap.parse_args()
    run(epochs_tx=a.tx_epochs, epochs_diff=a.diff_epochs, ppo_steps=a.ppo_steps)
