"""Colab Pro+ bootstrap for Phase 1 (40M TradeFM).

Paste this into a fresh Colab Pro+ notebook. Colab Pro+ gives priority
access to A100 (sometimes 40GB, sometimes 80GB) and ~24h continuous
runtime — enough for a 40M smoke run but NOT the full 524M.

Usage (in a Colab cell):
    !wget -q https://raw.githubusercontent.com/<you>/silicon-alpha/main/infra/gcp/colab_phase1_bootstrap.py -O /content/bootstrap.py
    %run /content/bootstrap.py

Then in subsequent cells run:
    !cd /content/silicon-alpha && python -m odte.train.pretrain_tradefm \
        --config configs/tradefm_40m.yml \
        --shards '/content/silicon-alpha/reports/odte_shards/opra_*.parquet' \
        --steps 5000 --batch 32 --grad-accum 2 --device cuda

Honest caveats:
  - Colab often assigns A100 40GB (not 80GB). If ctx_len=2048 OOMs, drop
    to 1024 and grad-accum=4.
  - "Priority access" is not guaranteed availability. Always snapshot
    the checkpoint back to GCS before the session expires.
  - For serious Phase 1 runs use infra/gcp/phase1_a100.sh instead —
    you own the VM and control termination.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def sh(cmd: str, check: bool = True) -> str:
    print(f"$ {cmd}")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
    if out.stdout:
        print(out.stdout)
    if out.stderr and out.returncode != 0:
        print(out.stderr, file=sys.stderr)
    return out.stdout


def main():
    if not any("colab" in k.lower() for k in os.environ):
        print("WARN: this script is intended for Google Colab.")
    # 1. Confirm A100
    sh("nvidia-smi -L")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

    # 2. Pull repo
    repo_url = os.environ.get("REPO_URL",
                              "https://github.com/YOURUSER/silicon-alpha")
    dst = Path("/content/silicon-alpha")
    if not dst.exists():
        sh(f"git clone --depth=1 {repo_url} {dst}")
    os.chdir(dst)

    # 3. Deps
    sh("pip install -q -r requirements.txt")
    sh("pip install -q wandb pyarrow google-cloud-storage gcsfs")

    # 4. Mount Drive if available (for checkpoint persistence)
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        os.makedirs("/content/drive/MyDrive/tradefm_ckpts", exist_ok=True)
        os.symlink("/content/drive/MyDrive/tradefm_ckpts",
                    "/content/silicon-alpha/checkpoints", target_is_directory=True)
        print("Checkpoints will land in /content/drive/MyDrive/tradefm_ckpts")
    except ImportError:
        print("Not on Colab — skipping Drive mount")

    # 5. Verify torch + CUDA
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device:", torch.cuda.get_device_name(0))
        print("  memory:", torch.cuda.get_device_properties(0).total_memory // (1024 ** 3), "GB")

    # 6. Smoke run on synth shards (proves the path is live before a long run)
    sh("python odte_phase1_smoke.py --device cuda --days 2 --steps 100",
       check=False)
    print("\nPhase-1 Colab bootstrap complete. Next: real training cell.")


if __name__ == "__main__":
    main()
