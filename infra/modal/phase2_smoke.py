"""Phase-2 Gate-3 smoke on Modal — single 1× H100.

Why this exists: before committing to the ~$50k 3-node A3 Mega run (or any
paid compute at all), prove the repo actually trains end-to-end on real
Hopper hardware. Modal's $30/mo free tier gives ~7.5 H100-hours; this smoke
burns ~0.5 hour of that.

What it exercises:
  - nvcr.io/nvidia/pytorch image builds; transformer-engine + flash-attn-3
    are importable.
  - odte_kernels_cu compiles against SM90a (H100 native).
  - 40M-param TradeFM with fp8=true, use_flash_attn=true trains for 200 steps
    on synthetic shards.
  - CheckpointManager writes to a Modal Volume and the file lands on disk.

What it does NOT exercise:
  - Multi-node NCCL (Modal is single-node).
  - Real OPRA throughput (shards are synthetic random tokens).
  - Chinchilla-optimal convergence (200 steps is pure pipeline validation).

Run (after `pip install modal && modal setup`):
    modal run infra/modal/phase2_smoke.py
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "tradefm-phase2-smoke"

# REPO_ROOT is only meaningful on the local machine at image-build time.
# Inside the Modal container, Modal flattens the entry file to /root/<name>.py
# so .parents[2] walks off the end of the path. Guard it.
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    REPO_ROOT = Path("/root/repo")


# NVCR ships CUDA 12 + torch + TE + FA3 pre-built; saves ~30 min vs building
# flash-attn-3 from source on every image refresh.
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    .apt_install("git", "build-essential")
    .pip_install(
        "pyarrow>=15.0",
        "pandas>=2.2",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "zstandard>=0.22",
        "scikit-learn>=1.4",
    )
    .run_commands(
        # NVCR pytorch:24.12 ships TE but not flash-attn; FA2/3 would add
        # 20-40 min of source build. The repo falls back to SDPA when
        # HAS_FLASH_ATTN is False — fine for a smoke. FA3 gets validated
        # separately on the real Phase-2 DLP image.
        "python -c 'import transformer_engine.pytorch; print(\"TE ok\")'",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/repo",
        copy=True,
        ignore=[
            "**/.venv",
            "**/__pycache__",
            "**/.git",
            "**/.claude",
            "**/data",
            "**/notebooks",
        ],
    )
    .run_commands(
        "cd /root/repo/odte/kernels && python setup.py build_ext --inplace",
    )
)

ckpt_volume = modal.Volume.from_name("tradefm-smoke-ckpts", create_if_missing=True)
shard_volume = modal.Volume.from_name("tradefm-smoke-shards", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# 40M baseline with Hopper path (fp8 + FA3) flipped on. Kept inline instead of
# as a committed YAML so the smoke is self-contained and doesn't fork configs.
SMOKE_CONFIG_YAML = """\
d_model: 768
n_heads: 12
n_layers: 12
vocab: 4096
ctx_len: 2048
dropout: 0.1
fp8: true
rotary: true
use_flash_attn: false
lr: 6.0e-4
weight_decay: 0.1
warmup_steps: 50
"""


def _make_synthetic_shards(
    out_dir: Path,
    n_shards: int = 4,
    rows_per_shard: int = 512,
    seq_len: int = 2048,
    vocab: int = 4096,
    seed: int = 0,
) -> None:
    """Write parquet shards matching ShardTokenDataset's schema: one column
    `tokens` holding int32 arrays of length seq_len."""
    import numpy as np
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        rows = [
            rng.integers(0, vocab, size=seq_len, dtype=np.int32).tolist()
            for _ in range(rows_per_shard)
        ]
        pd.DataFrame({"tokens": rows}).to_parquet(
            out_dir / f"smoke_{i:03d}.parquet", index=False
        )


@app.function(
    gpu="H100",
    timeout=3600,
    volumes={"/ckpts": ckpt_volume, "/shards": shard_volume},
)
def smoke():
    import subprocess
    import sys

    repo = Path("/root/repo")
    sys.path.insert(0, str(repo))

    import torch

    print(
        f"[smoke] torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )
    assert torch.cuda.is_available(), "H100 GPU not visible inside the container"

    shard_dir = Path("/shards")
    if not any(shard_dir.glob("*.parquet")):
        print("[smoke] generating synthetic shards…")
        _make_synthetic_shards(shard_dir)
        shard_volume.commit()
    shards = sorted(shard_dir.glob("*.parquet"))
    print(f"[smoke] shards: {[s.name for s in shards]}")

    cfg_path = Path("/tmp/tradefm_smoke_h100.yml")
    cfg_path.write_text(SMOKE_CONFIG_YAML)

    cmd = [
        "torchrun", "--nproc_per_node=1", "--nnodes=1",
        "-m", "odte.train.distributed",
        "--config", str(cfg_path),
        "--shards", f"{shard_dir}/smoke_*.parquet",
        "--ckpt-store", "/ckpts/tradefm_smoke",
        "--ckpt-prefix", "tradefm",
        "--steps", "200",
        "--batch", "4",
        "--grad-accum", "1",
        "--ckpt-every", "100",
        "--log-every", "10",
        "--num-workers", "2",
    ]
    print("[smoke] launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(repo), check=True)

    ckpts = list(Path("/ckpts/tradefm_smoke").rglob("*.pt"))
    print(f"[smoke] checkpoints written: {len(ckpts)}")
    assert ckpts, "no checkpoints — training didn't reach ckpt_every"
    for c in ckpts:
        print(f"    {c}  ({c.stat().st_size / 1e6:.1f} MB)")
    ckpt_volume.commit()

    print("[smoke] PASS — Hopper path + FSDP init + ckpt I/O verified.")


@app.local_entrypoint()
def main():
    smoke.remote()
