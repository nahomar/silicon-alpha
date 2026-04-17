"""WorldSim — digital-twin market simulator for 0DTE.

Two roles combine:
  - Forward model: frozen TradeFM predicts the next-token distribution over
    microstructure events. Sampling from it generates counterfactual tapes.
  - Oracle:        DMLPricer values every option at every step so agents can
    compute realized PnL.

Hawkes agents fight over the simulated book:
  - InformedFlow:   knows a small forward drift (next-Δprice bias)
  - UninformedFlow: pure self-exciting Poisson

Interactions train the executor to widen quotes against toxic flow and
tighten them against benign flow. This is the "adversarial participants"
phase of the roadmap.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from models.config import WorldSimConfig
from mm.synthetic_book import simulate_book
from mm.microprice import microprice

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hawkes agents
# ---------------------------------------------------------------------------

@dataclass
class HawkesAgent:
    name: str
    baseline: float = 0.1
    self_excite: float = 0.2
    decay: float = 2.0
    informed: bool = False
    edge_bps: float = 0.0
    lam: float = field(default=0.0, init=False)

    def step(self, dt: float, next_dp_hint: float,
             rng: np.random.Generator) -> List[dict]:
        """Return a list of order dicts this tick generated."""
        self.lam = self.baseline + (self.lam - self.baseline) * math.exp(-self.decay * dt)
        n_events = rng.poisson(self.lam * dt)
        out: List[dict] = []
        for _ in range(n_events):
            if self.informed and self.edge_bps > 0:
                bias = math.tanh(next_dp_hint / (self.edge_bps / 1e4 + 1e-9))
                p_buy = 0.5 + 0.5 * float(np.clip(bias, -0.95, 0.95))
            else:
                p_buy = 0.5 + 0.02 * rng.standard_normal()
            side = +1 if rng.random() < p_buy else -1
            size = float(np.exp(rng.normal(0.0, 1.2)))
            out.append({"agent": self.name, "side": int(side), "size": size})
            self.lam += self.self_excite
        return out


class InformedFlow(HawkesAgent):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "informed")
        kwargs.setdefault("informed", True)
        kwargs.setdefault("edge_bps", 2.0)
        kwargs.setdefault("baseline", 0.05)
        super().__init__(**kwargs)


class UninformedFlow(HawkesAgent):
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "uninformed")
        kwargs.setdefault("informed", False)
        kwargs.setdefault("baseline", 0.3)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# WorldSim
# ---------------------------------------------------------------------------

@dataclass
class WorldSim:
    cfg: WorldSimConfig = field(default_factory=WorldSimConfig)
    forward_model: Optional[Callable] = None    # e.g. TradeFM.forward (frozen)
    oracle: Optional[Callable] = None           # e.g. DMLPricer.price
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def run(self, steps: int | None = None) -> pd.DataFrame:
        """Run the simulator; return a wide DataFrame of events per tick."""
        steps = steps or self.cfg.n_ticks
        book = simulate_book(n_steps=steps, seed=self.seed)
        mids = ((book["bid_px"] + book["ask_px"]) / 2).values
        informed = [InformedFlow() for _ in range(self.cfg.n_informed_agents)]
        uninformed = [UninformedFlow() for _ in range(self.cfg.n_uninformed_agents)]

        rows: List[dict] = []
        dt = self.cfg.tick_ms / 1000.0
        for t in range(steps - 1):
            hint = mids[t + 1] - mids[t]
            buy_sz = 0.0; sell_sz = 0.0
            n_informed = 0; n_uninformed = 0
            for ag in informed:
                for ev in ag.step(dt, hint, self.rng):
                    if ev["side"] > 0: buy_sz += ev["size"]
                    else: sell_sz += ev["size"]
                    n_informed += 1
            for ag in uninformed:
                for ev in ag.step(dt, hint, self.rng):
                    if ev["side"] > 0: buy_sz += ev["size"]
                    else: sell_sz += ev["size"]
                    n_uninformed += 1
            mid = float(mids[t])
            mp = float(microprice(
                [book["bid_px"].iloc[t]], [book["ask_px"].iloc[t]],
                [book["bid_sz"].iloc[t]], [book["ask_sz"].iloc[t]])[0])
            rows.append({
                "t": t, "mid": mid, "micro": mp,
                "buy_sz": buy_sz, "sell_sz": sell_sz,
                "n_informed": n_informed, "n_uninformed": n_uninformed,
                "toxicity_proxy": abs(buy_sz - sell_sz) / (buy_sz + sell_sz + 1e-9),
            })
        return pd.DataFrame(rows)
