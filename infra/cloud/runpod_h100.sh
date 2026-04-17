#!/usr/bin/env bash
# Provision an H100 pod on RunPod for TradeFM Phase-2.
# Assumes the `runpodctl` CLI is installed + `RUNPOD_API_KEY` exported.
#
# Usage:
#   export RUNPOD_API_KEY=...
#   export RUNPOD_POD_TYPE="H100 PCIe"        # or "H100 SXM"
#   export REPO_URL="git@github.com:REPLACE_ME/market-pattern-bot.git"
#   ./infra/cloud/runpod_h100.sh

set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
POD_NAME="${POD_NAME:-tradefm-524m}"
POD_TYPE="${RUNPOD_POD_TYPE:-H100 PCIe}"
REPO_URL="${REPO_URL:?set REPO_URL}"
IMAGE="${RUNPOD_IMAGE:-nvcr.io/nvidia/pytorch:24.03-py3}"
VOL_GB="${RUNPOD_VOL_GB:-500}"

echo "[runpod] creating pod $POD_NAME ($POD_TYPE)"
POD_ID="$(runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$POD_TYPE" \
  --gpuCount 8 \
  --imageName "$IMAGE" \
  --volumeInGb "$VOL_GB" \
  --containerDiskInGb 200 \
  --env DEVICE=cuda \
  --output id)"

echo "[runpod] pod=$POD_ID  waiting for ssh"
runpodctl wait pod "$POD_ID" --state RUNNING
SSH_TARGET="$(runpodctl describe pod "$POD_ID" --output ssh)"

ssh -o StrictHostKeyChecking=no "$SSH_TARGET" <<BOOTSTRAP
set -euo pipefail
apt-get update -y
apt-get install -y git python3-venv zstd
git clone --depth=1 "$REPO_URL" /root/market-pattern-bot
cd /root/market-pattern-bot
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pyarrow pyyaml wandb deepspeed accelerate
# Hopper-only deps
pip install --no-build-isolation transformer-engine[pytorch] \
    || echo "[runpod] transformer-engine install failed; disable fp8 in cfg"
pip install flash-attn --no-build-isolation \
    || echo "[runpod] flash-attn install failed; cfg.use_flash_attn=false"

echo "[runpod] ready. Run:"
echo "  torchrun --nproc_per_node=8 -m odte.train.pretrain_tradefm \\
      --config configs/tradefm_524m.yml \\
      --shards 'reports/odte_shards/opra_*.parquet' \\
      --steps 200000 --batch 16 --grad-accum 4"
BOOTSTRAP

echo "[runpod] pod=$POD_ID ssh=$SSH_TARGET"
echo "         terminate with:  runpodctl remove pod $POD_ID"
