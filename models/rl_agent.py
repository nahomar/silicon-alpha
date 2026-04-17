"""Minimal PPO agent — actor / critic with shared MLP trunk.

Pure PyTorch, no stable-baselines dependency. Tuned for CPU-sized runs.
Replace with a transformer actor for sequence-native policies once you
have GPU budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .rl_env import PortfolioEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.mu(h), self.log_std.expand_as(self.mu(h)), self.v(h).squeeze(-1)

    def act(self, obs: torch.Tensor):
        mu, log_std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, log_std.exp())
        a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a, logp, v


@dataclass
class Rollout:
    obs: List[np.ndarray]
    actions: List[np.ndarray]
    logps: List[float]
    rewards: List[float]
    values: List[float]
    dones: List[bool]


def _gae(rewards, values, dones, last_v, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nxt = last_v if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nxt * (1 - dones[t]) - values[t]
        last = delta + gamma * lam * (1 - dones[t]) * last
        adv[t] = last
    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


def train_ppo(env: PortfolioEnv, cfg: Config, total_steps: int = 2000,
              rollout_len: int = 256, log_every: int = 1) -> ActorCritic:
    device = torch.device(cfg.device)
    ac = ActorCritic(env.obs_dim, env.act_dim).to(device)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    obs, _ = env.reset()

    step = 0
    ep_returns: List[float] = []
    while step < total_steps:
        roll = Rollout([], [], [], [], [], [])
        for _ in range(rollout_len):
            o = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a, logp, v = ac.act(o)
            a_np = a.squeeze(0).cpu().numpy()
            new_obs, reward, done, _, info = env.step(a_np)
            roll.obs.append(obs); roll.actions.append(a_np)
            roll.logps.append(float(logp.item())); roll.rewards.append(reward)
            roll.values.append(float(v.item())); roll.dones.append(done)
            obs = new_obs
            step += 1
            if done:
                ep_returns.append(info["cum_return"])
                obs, _ = env.reset()
                if step >= total_steps:
                    break

        with torch.no_grad():
            o = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            _, _, last_v = ac.forward(o)
        adv, ret = _gae(np.array(roll.rewards), np.array(roll.values),
                        np.array(roll.dones, dtype=np.float32),
                        float(last_v.item()), cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(roll.obs), device=device, dtype=torch.float32)
        act_t = torch.tensor(np.array(roll.actions), device=device, dtype=torch.float32)
        logp_t = torch.tensor(roll.logps, device=device, dtype=torch.float32)
        adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
        ret_t = torch.tensor(ret, device=device, dtype=torch.float32)

        for _ in range(cfg.ppo_epochs):
            mu, log_std, v = ac.forward(obs_t)
            dist = torch.distributions.Normal(mu, log_std.exp())
            new_logp = dist.log_prob(act_t).sum(-1)
            ratio = (new_logp - logp_t).exp()
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(v, ret_t)
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            opt.zero_grad(); loss.backward(); opt.step()

        if log_every and ep_returns:
            print(f"[ppo] step={step}  avg_cum_return={np.mean(ep_returns[-10:]):+.3%}  "
                  f"policy_loss={policy_loss.item():+.4f}  value_loss={value_loss.item():.4f}")

    return ac
