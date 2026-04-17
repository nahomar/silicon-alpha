"""Adversarial RL loop inside the 0DTE world simulation.

Loop per environment step:
  1. Informed + uninformed agents observe state, propose Hawkes params.
  2. WorldSim rolls forward one tick of microstructure.
  3. A market-maker policy (Avellaneda-Stoikov via mm/avellaneda_stoikov.py)
     posts bid/ask; we simulate fills against the aggregated flow.
  4. Reward for each agent = realized PnL extracted from the MM minus a
     per-trade impact cost (keeps flow honest, prevents degenerate policies).
  5. MM earns (or loses) the spread - markout.

The MM itself is NOT RL-trained here — in Phase 5 it gets wired through
DeterministicExecutor. For now, we hold the MM as the "environment" and
train the agents to exploit whatever weaknesses the fixed quoter has, so
Phase 5 learns to close those holes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from mm.avellaneda_stoikov import ASParams, as_quotes
from mm.synthetic_book import simulate_book
from .agents import InformedAgent, UninformedAgent, AgentConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _obs(mid_chg: float, spread: float, micro_dev: float, toxicity: float
         ) -> np.ndarray:
    return np.array([mid_chg, spread, micro_dev, toxicity], dtype=np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class AdversarialTrainer:
    n_informed: int = 2
    n_uninformed: int = 4
    steps_per_epoch: int = 1024
    epochs: int = 4
    ctx_ticks: int = 2048
    tick_ms: int = 100
    mm_gamma: float = 0.02
    mm_kappa: float = 500.0
    mm_sigma: float = 0.0005
    mm_tick_size: float = 0.01
    seed: int = 42
    device: str = "cpu"
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def run(self) -> dict:
        informed = [InformedAgent(AgentConfig(), device=self.device)
                    for _ in range(self.n_informed)]
        uninformed = [UninformedAgent(AgentConfig(), device=self.device)
                      for _ in range(self.n_uninformed)]

        # Seed a synthetic book; we consume the mid series as the exogenous
        # fair-value drift that informed agents partially observe.
        book = simulate_book(n_steps=self.ctx_ticks, seed=self.seed)
        mids = ((book["bid_px"] + book["ask_px"]) / 2).values

        as_params = ASParams(gamma=self.mm_gamma, sigma=self.mm_sigma,
                              kappa=self.mm_kappa, horizon=1.0,
                              inv_limit=50, tick_size=self.mm_tick_size)

        history: list[dict] = []
        mm_inv = 0
        mm_cash = 0.0
        mm_pnl_log: list[float] = []

        total_steps = self.steps_per_epoch * self.epochs
        for step in range(total_steps):
            # Wrap around the book if we go past its length.
            i = step % (len(mids) - 2)
            mid = float(mids[i])
            nxt = float(mids[i + 1])
            mid_chg = (nxt - mid) / max(mid, 1e-9)

            # Aggregate informed/uninformed flow using each agent's policy.
            buy_sz = 0.0; sell_sz = 0.0
            informed_sz = 0.0
            outs = []
            for ag in informed + uninformed:
                obs = _obs(mid_chg=mid_chg, spread=1e-4, micro_dev=0.0,
                            toxicity=0.0)    # toxicity computed below iteratively
                out = ag.act(obs)
                outs.append((ag, obs, out))
                # Hawkes-lite: baseline * max(0, bias * sign(mid_chg)+1) trades
                n_trades = int(max(0, self._rng.poisson(out.baseline * 5)))
                bias_prob = 0.5 + (out.informed_bias * np.tanh(mid_chg * 1e3) if ag.informed else 0.0)
                bias_prob = float(np.clip(bias_prob, 0.02, 0.98))
                for _ in range(n_trades):
                    side = +1 if self._rng.random() < bias_prob else -1
                    size = float(np.exp(self._rng.normal(0.0, 0.9)))
                    if side > 0: buy_sz += size
                    else: sell_sz += size
                    if ag.informed: informed_sz += size
            total_sz = buy_sz + sell_sz + 1e-9
            toxicity = informed_sz / total_sz
            # Recompute toxicity in each observation for next iteration
            obs_updated = _obs(mid_chg, spread=1e-4, micro_dev=0.0,
                                toxicity=toxicity)

            # Market-maker quotes using calibrated A-S plus toxicity widening.
            bid, ask, _info = as_quotes(s=mid, q=mm_inv, t=0.0,
                                        params=as_params,
                                        predicted_drift=0.0,
                                        toxicity=min(toxicity, 1.0))

            # Fill simulation: one buy-aggressor fills ask if ask<=mid_chg'd price
            fills_b = 0; fills_a = 0
            if self._rng.random() < min(buy_sz / 10, 0.9):
                fills_a += 1
            if self._rng.random() < min(sell_sz / 10, 0.9):
                fills_b += 1
            mm_cash -= fills_b * bid
            mm_cash += fills_a * ask
            mm_inv += fills_b - fills_a
            mm_pnl = mm_cash + mm_inv * nxt
            mm_pnl_log.append(mm_pnl)

            # Rewards: informed gets +share of (mm markout * informed_share);
            # uninformed gets small positive for participating (to keep the
            # Hawkes rate non-zero) minus fill-impact penalty.
            mkout = (nxt - mid) * (fills_b - fills_a)
            for ag, obs_old, out in outs:
                if ag.informed:
                    reward = 0.5 * mkout
                else:
                    reward = -0.01 * (buy_sz + sell_sz) / 10.0
                ag.record(obs_updated, out.raw_action, out.logp, out.value,
                          reward, done=(step == total_steps - 1))

            # Periodically update every agent that has a full rollout.
            for ag in informed + uninformed:
                if ag.ready():
                    # Bootstrap using zero value (last-step terminal).
                    loss = ag.update(last_value=0.0)
                    history.append({"agent": ("informed" if ag.informed
                                              else "uninformed"),
                                    **loss, "step": step})

        mm_terminal = mm_cash + mm_inv * float(mids[total_steps % (len(mids) - 1)])
        log.info("MM terminal PnL=%.4f  inv=%d  pnl_mean=%.4f  pnl_std=%.4f",
                 mm_terminal, mm_inv,
                 float(np.mean(mm_pnl_log)), float(np.std(mm_pnl_log)))
        return {
            "mm_terminal_pnl": float(mm_terminal),
            "mm_inv": int(mm_inv),
            "mm_pnl_mean": float(np.mean(mm_pnl_log)),
            "mm_pnl_std": float(np.std(mm_pnl_log)),
            "agent_updates": history,
            "n_informed": self.n_informed,
            "n_uninformed": self.n_uninformed,
            "total_steps": total_steps,
        }


def adversarial_train(**kwargs) -> dict:
    return AdversarialTrainer(**kwargs).run()
