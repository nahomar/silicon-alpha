"""1-D DDPM for synthetic return paths.

Trained on continuous log-return sequences (shape (B, L, N_tickers)), this
generates plausible synthetic return windows used for:
  - data augmentation for the transformer
  - stress-testing the RL agent on paths it has never seen
  - exploring counterfactual scenarios

Honest framing: diffusion models learn the training distribution; synthetic
paths can overfit and should not be treated as independent market realities.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def _beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, steps)


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        a = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, ch: int, t_dim: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)
        self.t_mlp = nn.Linear(t_dim, ch)

    def forward(self, x, t_emb):
        h = F.silu(self.n1(self.c1(x)))
        h = h + self.t_mlp(t_emb)[..., None]
        h = F.silu(self.n2(self.c2(h)))
        return x + h


class UNet1D(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.t_dim = hidden
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbed(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.in_proj = nn.Conv1d(channels, hidden, 1)
        self.b1 = ResBlock1D(hidden, hidden)
        self.b2 = ResBlock1D(hidden, hidden)
        self.b3 = ResBlock1D(hidden, hidden)
        self.out_proj = nn.Conv1d(hidden, channels, 1)

    def forward(self, x, t):
        t_emb = self.t_embed(t)
        h = self.in_proj(x)
        h = self.b1(h, t_emb)
        h = self.b2(h, t_emb)
        h = self.b3(h, t_emb)
        return self.out_proj(h)


class DDPM:
    def __init__(self, n_channels: int, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.T = cfg.diffusion_steps
        self.model = UNet1D(n_channels, cfg.diffusion_hidden).to(self.device)
        betas = _beta_schedule(self.T).to(self.device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cum = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cum)
        self.sqrt_1macp = torch.sqrt(1 - self.alphas_cum)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        xt = self.sqrt_acp[t][:, None, None] * x0 + self.sqrt_1macp[t][:, None, None] * noise
        return xt, noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        xt, eps = self.q_sample(x0, t)
        pred = self.model(xt, t)
        return F.mse_loss(pred, eps)

    @torch.no_grad()
    def sample(self, n: int, seq_len: int, n_channels: int) -> torch.Tensor:
        x = torch.randn(n, n_channels, seq_len, device=self.device)
        for t in reversed(range(self.T)):
            tb = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.model(x, tb)
            a = self.alphas_cum[t]
            a_prev = self.alphas_cum[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            beta = self.betas[t]
            coef = (1 - a_prev) / (1 - a) * beta
            mu = (1 / torch.sqrt(1 - beta)) * (x - beta / self.sqrt_1macp[t] * eps)
            if t > 0:
                x = mu + torch.sqrt(coef) * torch.randn_like(x)
            else:
                x = mu
        return x  # (N, C, L) where C = n_tickers
