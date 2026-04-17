"""PPO agents for adversarial market simulation.

Each agent chooses Hawkes intensity parameters (baseline, self_excite, decay,
bias) every N ticks, conditional on the current world state. Rewards come
from the difference in PnL that the market-maker pays out to each agent —
informed agents want to find "toxic" configurations that extract spread,
uninformed agents want to break even.

Kept deliberately small: tiny MLP actor + critic, GAE(λ)-PPO. On Mac CPU
this trains in minutes for 10k total environment steps. Scales up by
swapping hidden_dim / batch / epochs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    obs_dim: int = 4                # [mid_chg, spread, microprice_dev, toxicity]
    n_actions: int = 4              # [baseline, self_excite, decay, informed_bias]
    action_low: Tuple[float, ...] = (0.0, 0.0, 0.1, -0.5)
    action_high: Tuple[float, ...] = (1.0, 0.5, 5.0, 0.5)
    hidden_dim: int = 64
    lr: float = 3e-4
    gamma: float = 0.98
    lam: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    rollout_len: int = 256
    batch_size: int = 64
    entropy_coef: float = 1e-3
    value_coef: float = 0.5


# ---------------------------------------------------------------------------
# Actor-critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Diagonal-Gaussian continuous policy + scalar value head."""

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.body = nn.Sequential(
            nn.Linear(cfg.obs_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
        )
        self.mu = nn.Linear(h, cfg.n_actions)
        self.log_std = nn.Parameter(torch.full((cfg.n_actions,), -1.0))
        self.v = nn.Linear(h, 1)
        self.cfg = cfg
        self.register_buffer("a_lo", torch.tensor(cfg.action_low))
        self.register_buffer("a_hi", torch.tensor(cfg.action_high))

    def _squash(self, raw: torch.Tensor) -> torch.Tensor:
        """tanh-squash + affine map into [lo, hi]."""
        sq = torch.tanh(raw)
        return self.a_lo + 0.5 * (sq + 1.0) * (self.a_hi - self.a_lo)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        mu_raw = self.mu(h)
        value = self.v(h).squeeze(-1)
        return mu_raw, self.log_std.exp(), value

    def act(self, obs: torch.Tensor):
        mu_raw, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu_raw, std)
        raw = dist.rsample()
        logp = dist.log_prob(raw).sum(-1)
        action = self._squash(raw)
        return action.detach(), logp.detach(), v.detach(), raw.detach()

    def evaluate(self, obs, raw_actions):
        mu_raw, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu_raw, std)
        logp = dist.log_prob(raw_actions).sum(-1)
        ent = dist.entropy().sum(-1)
        return logp, ent, v


# ---------------------------------------------------------------------------
# Agent wrappers
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    baseline: float
    self_excite: float
    decay: float
    informed_bias: float
    raw_action: np.ndarray
    logp: float
    value: float


class BaseAgent:
    def __init__(self, cfg: AgentConfig, informed: bool, device: str = "cpu"):
        self.cfg = cfg
        self.informed = informed
        self.device = torch.device(device)
        self.net = ActorCritic(cfg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.buf: list[dict] = []

    def act(self, obs: np.ndarray) -> AgentOutput:
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp, value, raw = self.net.act(t)
        a = action.squeeze(0).cpu().numpy()
        return AgentOutput(
            baseline=float(a[0]), self_excite=float(a[1]),
            decay=float(a[2]),
            informed_bias=float(a[3]) if self.informed else 0.0,
            raw_action=raw.squeeze(0).cpu().numpy(),
            logp=float(logp.item()), value=float(value.item()),
        )

    def record(self, obs, raw_action, logp, value, reward, done):
        self.buf.append({"obs": obs, "raw": raw_action, "logp": logp,
                         "value": value, "reward": float(reward),
                         "done": bool(done)})

    def ready(self) -> bool:
        return len(self.buf) >= self.cfg.rollout_len

    def _gae(self, last_value: float):
        T = len(self.buf)
        adv = np.zeros(T, dtype=np.float32); ret = np.zeros(T, dtype=np.float32)
        run = 0.0
        for t in reversed(range(T)):
            nxt_v = last_value if t == T - 1 else self.buf[t + 1]["value"]
            delta = self.buf[t]["reward"] + self.cfg.gamma * nxt_v * (1 - self.buf[t]["done"]) \
                - self.buf[t]["value"]
            run = delta + self.cfg.gamma * self.cfg.lam * (1 - self.buf[t]["done"]) * run
            adv[t] = run
            ret[t] = adv[t] + self.buf[t]["value"]
        # standardize advantages
        if adv.std() > 1e-9:
            adv = (adv - adv.mean()) / adv.std()
        return adv, ret

    def update(self, last_value: float) -> dict:
        adv, ret = self._gae(last_value)
        obs = torch.as_tensor(np.stack([b["obs"] for b in self.buf]),
                              dtype=torch.float32, device=self.device)
        raw = torch.as_tensor(np.stack([b["raw"] for b in self.buf]),
                              dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([b["logp"] for b in self.buf],
                                   dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        N = len(self.buf)
        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        for _ in range(self.cfg.ppo_epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, self.cfg.batch_size):
                sl = idx[start: start + self.cfg.batch_size]
                logp, ent, v = self.net.evaluate(obs[sl], raw[sl])
                ratio = (logp - old_logp[sl]).exp()
                unclipped = ratio * adv_t[sl]
                clipped = torch.clamp(ratio, 1 - self.cfg.ppo_clip,
                                      1 + self.cfg.ppo_clip) * adv_t[sl]
                pl = -torch.min(unclipped, clipped).mean()
                vl = F.mse_loss(v, ret_t[sl])
                el = ent.mean()
                loss = pl + self.cfg.value_coef * vl - self.cfg.entropy_coef * el
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                losses["policy"] += float(pl.item())
                losses["value"] += float(vl.item())
                losses["entropy"] += float(el.item())
        self.buf.clear()
        return losses


class InformedAgent(BaseAgent):
    def __init__(self, cfg: AgentConfig | None = None, device: str = "cpu"):
        super().__init__(cfg or AgentConfig(), informed=True, device=device)


class UninformedAgent(BaseAgent):
    def __init__(self, cfg: AgentConfig | None = None, device: str = "cpu"):
        super().__init__(cfg or AgentConfig(), informed=False, device=device)


def train_agents(agents: List[BaseAgent], final_values: List[float]) -> List[dict]:
    return [a.update(v) for a, v in zip(agents, final_values)]
