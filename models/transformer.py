"""Decoder-only transformer for next-token-return prediction.

Each ticker is treated as an independent sequence over the same time axis;
at each step we emit the categorical distribution over the return codebook.
Multi-ticker dependencies are captured by stacking tickers as separate
channels in the embedding.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def _causal_mask(T: int, device) -> torch.Tensor:
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class MarketTransformer(nn.Module):
    """Inputs:
        tokens: (B, T, N) long tensor  — token id per ticker
        sent:   (B, T)     float tensor — market sentiment feature
    Output:
        logits: (B, T, N, V)            — per-ticker next-token distribution
    """

    def __init__(self, n_tickers: int, vocab: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_tickers = n_tickers
        self.vocab = vocab
        self.tok_emb = nn.Embedding(vocab, cfg.d_model)
        self.ticker_emb = nn.Embedding(n_tickers, cfg.d_model)
        self.sent_proj = nn.Linear(1, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.d_model, vocab)

    def forward(self, tokens: torch.Tensor, sent: torch.Tensor) -> torch.Tensor:
        B, T, N = tokens.shape
        # Average over tickers at each timestep, then add per-ticker projection
        # at the output head. This keeps sequence length at T (not T*N) so
        # training scales on CPU.
        tok_h = self.tok_emb(tokens).mean(dim=2)                      # (B,T,D)
        sent_h = self.sent_proj(sent.unsqueeze(-1))                   # (B,T,D)
        h = self.pos(tok_h + sent_h)
        mask = _causal_mask(T, h.device)
        h = self.blocks(h, mask=mask)                                 # (B,T,D)
        # Broadcast shared temporal state to per-ticker logits via ticker emb
        ticker_ids = torch.arange(N, device=h.device)
        ticker_h = self.ticker_emb(ticker_ids)                        # (N,D)
        merged = h.unsqueeze(2) + ticker_h.view(1, 1, N, -1)          # (B,T,N,D)
        return self.head(merged)                                      # (B,T,N,V)

    def loss(self, tokens: torch.Tensor, sent: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens[:, :-1], sent[:, :-1])
        target = tokens[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab),
            target.reshape(-1),
        )
