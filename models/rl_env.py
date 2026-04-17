"""Gymnasium-style portfolio RL environment.

State  : flattened feature window over last context_len days.
Action : portfolio weights over N tickers (continuous, sum-to-one softmax).
Reward : next-day log portfolio return minus transaction cost and variance
         penalty. Long-only, fully invested; plug in short/leverage later.

Works on either (a) real historical data or (b) a trained WorldModel's
imagined rollouts.

Honest framing: PPO + transformer on daily bars is a toy. Real trading
needs intraday data, slippage models, regime-aware risk controls, and
walk-forward validation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


class PortfolioEnv:
    def __init__(self, returns: np.ndarray, features: np.ndarray,
                 cfg: Optional[Config] = None):
        """returns: (T, N) simple returns. features: (T, F) state matrix."""
        self.cfg = cfg or Config()
        self.returns = returns.astype(np.float32)
        self.features = features.astype(np.float32)
        self.N = returns.shape[1]
        self.F = features.shape[1]
        self.T = returns.shape[0]
        self.context = self.cfg.context_len
        self.cost = self.cfg.commission_bps / 1e4
        self._reset_state()

    def _reset_state(self, start: Optional[int] = None) -> None:
        self.prev_w = np.ones(self.N, dtype=np.float32) / self.N
        max_start = self.T - self.cfg.episode_len - 1
        if start is None:
            start = int(np.random.randint(self.context, max(self.context + 1, max_start)))
        self.t0 = start
        self.t = start
        self.episode_return = 0.0

    def reset(self, seed: Optional[int] = None, start: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self._reset_state(start)
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        return self.features[self.t - self.context: self.t].reshape(-1)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # softmax to enforce simplex
        a = np.exp(action - action.max())
        w = a / a.sum()
        turnover = np.abs(w - self.prev_w).sum()
        r_next = float((w * self.returns[self.t]).sum())
        reward = np.log(1 + r_next) - self.cost * turnover
        # risk-penalty term on realized portfolio volatility over context
        recent = (self.returns[self.t - self.context: self.t] * w).sum(axis=1)
        reward -= 0.1 * float(np.std(recent))

        self.prev_w = w
        self.t += 1
        self.episode_return += r_next
        done = (self.t - self.t0) >= self.cfg.episode_len or self.t >= self.T - 1
        info = {"return": r_next, "turnover": float(turnover),
                "cum_return": self.episode_return}
        return self._obs(), float(reward), bool(done), False, info

    @property
    def obs_dim(self) -> int:
        return self.context * self.F

    @property
    def act_dim(self) -> int:
        return self.N
