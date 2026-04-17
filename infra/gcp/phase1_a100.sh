#!/usr/bin/env bash
# Phase 1 — single A100 80GB on Google Cloud for the 40M TradeFM
# pretrain + streaming quantile fit. Roughly $3-4/hr; full run ~$150-250.
#
# Requires: gcloud CLI authenticated, a project with `Compute Engine API`
# + `Cloud Storage` enabled, and the `a2-highgpu-1g` quota approved
# (request via Quotas console if first time).

set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE="${GCP_INSTANCE:-tradefm-phase1}"
MACHINE_TYPE="${GCP_MACHINE:-a2-ultragpu-1g}"   # single A100 80GB
DISK_GB="${GCP_DISK_GB:-500}"
IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-4-cu124-py310}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
BUCKET="${GCP_BUCKET:?set GCP_BUCKET (gs://... for checkpoints/shards)}"
REPO_URL="${REPO_URL:?set REPO_URL}"
SERVICE_ACCOUNT="${GCP_SERVICE_ACCOUNT:-}"      # optional; uses compute default if empty

echo "[gcp-phase1] launching $INSTANCE  ($MACHINE_TYPE) in $ZONE"

gcloud compute instances create "$INSTANCE" \
    --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=nvidia-a100-80gb,count=1" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-ssd \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --metadata="install-nvidia-driver=True" \
    ${SERVICE_ACCOUNT:+--service-account=$SERVICE_ACCOUNT} \
    --scopes=cloud-platform \
    --preemptible   # omit this flag for a non-preemptible run

echo "[gcp-phase1] waiting for SSH ..."
gcloud compute instances describe "$INSTANCE" --zone "$ZONE" \
    --project "$PROJECT" --format="value(status)" | grep -q RUNNING
sleep 30

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" -- \
"set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y git-lfs zstd

if [[ ! -d ~/market-pattern-bot ]]; then
    git clone --depth=1 '$REPO_URL' ~/market-pattern-bot
fi
cd ~/market-pattern-bot
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch pyyaml wandb google-cloud-storage gcsfs pyarrow

# Mount the GCS bucket (checkpoints + shards) via gcsfuse for convenient IO
if ! command -v gcsfuse >/dev/null; then
    sudo apt-get install -y fuse
    wget -q https://github.com/GoogleCloudPlatform/gcsfuse/releases/latest/download/gcsfuse_Linux_x86_64.deb
    sudo dpkg -i gcsfuse_*.deb
fi
mkdir -p ~/gcs
gcsfuse --implicit-dirs ${BUCKET#gs://} ~/gcs || true

echo '[gcp-phase1] bootstrap complete'
echo '[gcp-phase1] run: DEVICE=cuda python -m odte.train.pretrain_tradefm \\
    --config configs/tradefm_40m.yml \\
    --shards ~/gcs/opra_shards/opra_*.parquet \\
    --ckpt-dir ~/gcs/ckpts/tradefm_40m \\
    --steps 20000 --batch 64 --grad-accum 2'
"

PUBLIC_IP="$(gcloud compute instances describe "$INSTANCE" \
    --zone "$ZONE" --project "$PROJECT" \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)')"

cat <<EOF

[gcp-phase1] DONE
    gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT
    scp from:   gcloud compute scp $INSTANCE:~/gcs/ckpts/tradefm_40m/best_*.pt ./
    terminate:  gcloud compute instances delete $INSTANCE --zone $ZONE --project $PROJECT

NOTE: --preemptible saves ~70% but GCP can reclaim the VM any time. Use
'--max-run-duration=24h' for a non-preemptible upper-bound instead.
EOF
