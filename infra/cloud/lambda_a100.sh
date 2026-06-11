#!/usr/bin/env bash
# Provision a single-GPU Lambda Labs A100 instance for TradeFM Phase-1.
# Assumes the Lambda Cloud CLI is installed and authenticated.
#
# Usage:
#   export LAMBDA_INSTANCE_TYPE=gpu_1x_a100
#   export LAMBDA_REGION=us-west-1
#   export LAMBDA_SSH_KEY=~/.ssh/id_rsa
#   ./infra/cloud/lambda_a100.sh

set -euo pipefail

INSTANCE_TYPE="${LAMBDA_INSTANCE_TYPE:-gpu_1x_a100}"
REGION="${LAMBDA_REGION:-us-west-1}"
SSH_KEY="${LAMBDA_SSH_KEY:-$HOME/.ssh/id_rsa}"
REPO_URL="${REPO_URL:-https://github.com/REPLACE_ME/silicon-alpha.git}"

echo "[lambda] launching $INSTANCE_TYPE in $REGION"
INSTANCE_ID="$(lambda-cloud instances launch \
  --type "$INSTANCE_TYPE" \
  --region "$REGION" \
  --ssh-key "$SSH_KEY" \
  --name tradefm-40m \
  --output-id)"

echo "[lambda] instance=$INSTANCE_ID — waiting for ready"
lambda-cloud instances wait-until-running "$INSTANCE_ID"
IP="$(lambda-cloud instances describe "$INSTANCE_ID" --output ip)"

echo "[lambda] ip=$IP — bootstrap"
ssh -o StrictHostKeyChecking=no -i "$SSH_KEY" ubuntu@"$IP" <<BOOTSTRAP
set -euo pipefail
sudo apt-get update -y
sudo apt-get install -y git python3-venv zstd htop
git clone --depth=1 "$REPO_URL" ~/silicon-alpha
cd ~/silicon-alpha
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==2.8.0 pyarrow pyyaml wandb
# NOTE: transformer-engine and flash-attn only matter for H100; skip on A100.
echo "[lambda] ready — TradeFM pretrain can run via:"
echo "  DEVICE=cuda python -m odte.train.pretrain_tradefm --config configs/tradefm_40m.yml --shards 'reports/odte_shards/opra_*.parquet' --steps 20000 --batch 64"
BOOTSTRAP

echo "[lambda] done. SSH:  ssh -i $SSH_KEY ubuntu@$IP"
echo "         instance:  $INSTANCE_ID"
echo "   remember to terminate:  lambda-cloud instances terminate $INSTANCE_ID"
