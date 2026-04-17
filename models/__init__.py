"""Advanced modeling stack.

Layers (each independently runnable):
  unified_dataset → tokenizer → transformer / diffusion / world_model → rl_env → rl_agent

Honest framing: these are well-known research building blocks. They do NOT
constitute a trading edge by themselves. Treat every result as a hypothesis
to be validated on held-out data and paper-traded before any capital risk.
"""
from .config import Config
