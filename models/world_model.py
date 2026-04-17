"""Transformer-based market world model.

Given a history window of token sequences + sentiment, predict the NEXT
state's tokens. Used by the RL env to roll out imagined trajectories
("world simulation") for policy improvement between live-data runs.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .transformer import MarketTransformer


class WorldModel:
    def __init__(self, n_tickers: int, vocab: int, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = MarketTransformer(n_tickers, vocab, cfg).to(self.device)
        self.n_tickers = n_tickers
        self.vocab = vocab

    def step(self, tokens: torch.Tensor, sent: torch.Tensor, temperature: float = 1.0
             ) -> torch.Tensor:
        """Sample the next step of tokens given a history.

        tokens: (B, T, N) long
        sent:   (B, T)    float
        returns next_tokens: (B, N) long
        """
        self.net.eval()
        with torch.no_grad():
            logits = self.net(tokens, sent)[:, -1]                     # (B, N, V)
            if temperature <= 0:
                return logits.argmax(dim=-1)
            probs = torch.softmax(logits / temperature, dim=-1)
            B, N, V = probs.shape
            flat = probs.reshape(B * N, V)
            idx = torch.multinomial(flat, 1).reshape(B, N)
            return idx

    def rollout(self, start_tokens: torch.Tensor, start_sent: torch.Tensor,
                steps: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressively generate a trajectory of length `steps`.

        start_tokens: (1, T, N)
        start_sent:   (1, T)
        returns (1, T + steps, N) tokens
        """
        tokens = start_tokens.clone()
        sent = start_sent.clone()
        for _ in range(steps):
            next_tok = self.step(tokens[:, -self.cfg.context_len:],
                                 sent[:, -self.cfg.context_len:], temperature)
            tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)
            # carry sentiment forward (no prediction target for sentiment here)
            sent = torch.cat([sent, sent[:, -1:]], dim=1)
        return tokens

    def save(self, path) -> None:
        torch.save({"state": self.net.state_dict(),
                    "n_tickers": self.n_tickers,
                    "vocab": self.vocab}, path)

    @classmethod
    def load(cls, path, cfg: Config) -> "WorldModel":
        blob = torch.load(path, map_location=cfg.device)
        wm = cls(blob["n_tickers"], blob["vocab"], cfg)
        wm.net.load_state_dict(blob["state"])
        return wm
