"""Market-making simulation / RL env.

Action space (4-d continuous):
  [bid_offset_ticks, ask_offset_ticks, bid_size, ask_size]
Offsets are added to Avellaneda-Stoikov baseline quotes; the agent learns
deviations, not absolute prices. That keeps training stable.

Reward components:
  + spread earned on each round-trip
  − markout adverse selection at short horizon
  − inventory penalty (running variance of q)
  − turnover / cancellation cost if configured

Ideal: train policy with PPO/SAC using a transformer encoder over a rolling
window of microprice features. Here we expose a clean obs/action interface so
models.rl_agent.train_ppo can be reused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .avellaneda_stoikov import ASParams, as_quotes


@dataclass
class MMConfig:
    context_len: int = 32
    markout_horizon: int = 50
    max_inventory: int = 50
    inventory_penalty: float = 1e-4
    turnover_cost: float = 0.0
    use_predictor: bool = True
    as_params: ASParams = field(default_factory=ASParams)


class MarketMakingEnv:
    def __init__(self, tob_feats: pd.DataFrame, tox: pd.DataFrame,
                 predictor=None, cfg: Optional[MMConfig] = None):
        self.cfg = cfg or MMConfig()
        self.feats = tob_feats.copy().reset_index(drop=True)
        self.tox = tox.reindex(tob_feats.index).ffill().fillna(0.0).reset_index(drop=True)
        self.predictor = predictor
        self.T = len(self.feats)
        self.inv = 0
        self.cash = 0.0
        self.t = self.cfg.context_len

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.inv = 0
        self.cash = 0.0
        self.t = self.cfg.context_len
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        window = self.feats.iloc[self.t - self.cfg.context_len: self.t]
        tox_window = self.tox.iloc[self.t - self.cfg.context_len: self.t]
        x = np.concatenate([window.values.flatten(), tox_window.values.flatten(),
                            np.array([self.inv / max(self.cfg.max_inventory, 1)])])
        return x.astype(np.float32)

    def _predicted_drift(self) -> float:
        if self.predictor is None:
            return 0.0
        row = self.feats.iloc[[self.t]]
        tx = self.tox.iloc[[self.t]]
        mu, _ = self.predictor.predict(self.predictor.build_features(row, tx))
        mid = float(self.feats["mid"].iat[self.t])
        # predicted log-return → expected price drift
        return float(mid * mu[0])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        bid_off, ask_off, bsize, asize = action
        row = self.feats.iloc[self.t]
        S = float(row["micro"])
        tau_t = self.t / self.T
        vpin = float(self.tox["vpin"].iat[self.t]) if "vpin" in self.tox.columns else 0.0

        bid_as, ask_as, _ = as_quotes(
            s=S, q=self.inv, t=tau_t, params=self.cfg.as_params,
            predicted_drift=self._predicted_drift(),
            toxicity=min(max(vpin, 0.0), 1.0),
        )
        tick = self.cfg.as_params.tick_size
        bid = bid_as - bid_off * tick
        ask = ask_as + ask_off * tick

        # Simulate fills using next-bar trade print
        next_row = self.feats.iloc[self.t + 1] if self.t + 1 < self.T else row
        fill_bid = next_row.get("last_px", np.nan) if hasattr(next_row, "get") else np.nan
        # Approximation: if a sell print at/below our bid → we bought; symmetric.
        fill_qty_b = 0; fill_qty_a = 0
        last_px = next_row["last_px"] if "last_px" in next_row else np.nan
        last_side = next_row["last_side"] if "last_side" in next_row else 0
        if not np.isnan(last_px):
            if last_side < 0 and last_px <= bid and self.inv < self.cfg.max_inventory:
                fill_qty_b = max(1, int(bsize))
            if last_side > 0 and last_px >= ask and self.inv > -self.cfg.max_inventory:
                fill_qty_a = max(1, int(asize))

        self.cash -= fill_qty_b * bid
        self.cash += fill_qty_a * ask
        self.inv += fill_qty_b - fill_qty_a

        # Reward = mark-to-market delta of (cash + inv*mid) minus penalties
        next_mid = float(next_row["mid"]) if "mid" in next_row else S
        pnl = self.cash + self.inv * next_mid
        inv_pen = self.cfg.inventory_penalty * (self.inv ** 2)
        reward = pnl - inv_pen  # cumulative; per-step = diff
        reward_step = reward - getattr(self, "_prev_pnl_pen", 0.0)
        self._prev_pnl_pen = reward

        self.t += 1
        done = self.t >= self.T - 1
        info = {
            "pnl": pnl, "inv": self.inv, "bid": bid, "ask": ask,
            "fill_b": fill_qty_b, "fill_a": fill_qty_a,
            "as_bid": bid_as, "as_ask": ask_as,
        }
        return self._obs(), float(reward_step), bool(done), False, info

    @property
    def obs_dim(self) -> int:
        return self._obs().shape[0]

    @property
    def act_dim(self) -> int:
        return 4
