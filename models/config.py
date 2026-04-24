"""Central hyperparameter config. Override by mutating Config() or env vars."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # Universe
    tickers: List[str] = field(default_factory=lambda: [
        "^GSPC", "^NDX", "^DJI", "^VIX",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    ])
    history_period: str = "5y"
    interval: str = "1d"

    # Tokenizer
    n_return_buckets: int = 64   # codebook size per ticker
    context_len: int = 128

    # Transformer
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    # Diffusion
    diffusion_steps: int = 200
    diffusion_hidden: int = 128
    synth_seq_len: int = 64

    # RL
    episode_len: int = 252          # one trading year
    commission_bps: float = 1.0     # 1 bp round-trip
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    ppo_clip: float = 0.2

    # Compute
    device: str = os.getenv("DEVICE", "cpu")      # "cuda" / "mps" / "cpu"
    batch_size: int = int(os.getenv("BATCH", "64"))
    lr: float = 3e-4
    mixed_precision: bool = False                  # set True w/ cuda

    # Paths
    checkpoints_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")
    synthetic_dir: Path = field(default_factory=lambda: ROOT / "synthetic")
    reports_dir: Path = field(default_factory=lambda: ROOT / "reports")

    def ensure_dirs(self) -> None:
        for p in (self.checkpoints_dir, self.synthetic_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 0DTE-specific configs (Phase 0 of odte roadmap)
# ---------------------------------------------------------------------------

@dataclass
class TradeFMConfig:
    """Decoder-only generative transformer for market microstructure tokens."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    vocab: int = 1024
    ctx_len: int = 512
    dropout: float = 0.1
    fp8: bool = False                  # set True on H100 with transformer-engine
    rotary: bool = True
    use_flash_attn: bool = False       # set True when flash_attn available
    grad_checkpointing: bool = False   # trade ~20% compute for ~60% activation mem
    # Cross-asset fusion (Phase-2.5 scaffold, off by default).
    # 0 = single-modality (current behavior, back-compat). >0 = enable a
    # modality-embedding channel that gets added to tok_emb so the attention
    # layers can learn cross-asset lead/lag. Typical values:
    #   2 = OPRA + ES
    #   3 = OPRA + ES + ETF-arb
    # Tokens arrive with an aligned modality_ids tensor of the same shape.
    # See docs/cross_asset_fusion.md for the token-interleaving schema.
    modality_vocab: int = 0
    # Training
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200

    @classmethod
    def mini(cls) -> "TradeFMConfig":
        """~10M params — Mac/MPS friendly."""
        return cls(d_model=256, n_heads=4, n_layers=6, vocab=1024, ctx_len=512)

    @classmethod
    def tradefm_524m(cls) -> "TradeFMConfig":
        """~524M params — 8xH100 target."""
        return cls(d_model=2048, n_heads=16, n_layers=24, vocab=4096,
                   ctx_len=4096, fp8=True, use_flash_attn=True)


@dataclass
class DMLConfig:
    """Differential-ML option pricer with maturity-gated variance."""
    hidden_dim: int = 128
    n_layers: int = 4
    sigma_floor: float = 1e-4          # minimum effective sigma (avoid div/0 near τ=0)
    grad_loss_weight: float = 1.0      # λ on AAD-greek MSE term
    lr: float = 1e-3
    batch_size: int = 1024
    pretrain_steps: int = 2000


@dataclass
class WorldSimConfig:
    """Digital-twin market simulator."""
    n_ticks: int = 10_000
    tick_ms: int = 100
    n_informed_agents: int = 2
    n_uninformed_agents: int = 6
    hawkes_baseline: float = 0.05
    hawkes_self_excite: float = 0.15
    adversarial_reward_scale: float = 1.0
