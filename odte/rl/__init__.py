"""Adversarial RL for the 0DTE world simulation.

Agents drive Hawkes flow inside odte.world_sim.WorldSim:
  - InformedAgent (knows future price drift, can be "toxic")
  - UninformedAgent (has no edge; pure noise trader)

The MARKET MAKER (our strategy) then has to learn to widen quotes when
informed pressure is high and tighten them when flow is benign. That's
the adversarial loop implemented in train_world_sim.py.
"""
from .agents import (
    InformedAgent, UninformedAgent, AgentConfig, AgentOutput, train_agents,
)
from .train_world_sim import AdversarialTrainer, adversarial_train
