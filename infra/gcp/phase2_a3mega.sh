#!/usr/bin/env bash
# Phase 2 — 3× A3 Mega (8× H100 80GB each) = 24× H100 SXM for the full
# 524M TradeFM run on Google Cloud.
#
# HRT chose GCP A3 in their July 2024 partnership for precisely this tier.
# The A3 Mega node has 1600 Gbps of GPUDirect-TCPX bandwidth across 8 NICs
# which is comparable to Mellanox ConnectX-7 IB for all-reduce throughput
# at this scale.
#
# Cost estimate (us-central1, 2026 on-demand):
#   a3-megagpu-8g  ≈ $95-105/hr per node × 3 nodes  →  ~$290/hr cluster
#   100B tokens at ~210k tok/s per node = ~130h  →  ~$38k for one run
#   Add 30% for restart/ckpt overhead → plan $50k per full run.
#
# Budget alternative: use 2× A3 High (16× H100 total) for ~$160/hr, takes
# ~50% longer. a3-highgpu-8g is ~$80-90/hr vs $95-105/hr for Mega.

set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${GCP_ZONE:-us-central1-a}"
N_NODES="${GCP_N_NODES:-3}"                   # 3× 8GPU = 24 GPUs
CLUSTER="${GCP_CLUSTER:-tradefm-524m}"
MACHINE_TYPE="${GCP_MACHINE:-a3-megagpu-8g}"  # or a3-highgpu-8g for budget
DISK_GB="${GCP_DISK_GB:-2000}"
IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-4-cu124-py310-transformerengine-flashattn}"
IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
BUCKET="${GCP_BUCKET:?set GCP_BUCKET}"
REPO_URL="${REPO_URL:?set REPO_URL}"
SERVICE_ACCOUNT="${GCP_SERVICE_ACCOUNT:-}"

# --- 1. Reserve the nodes as a compact placement group for low latency --
echo "[gcp-phase2] creating compact placement policy for $N_NODES nodes"
POLICY="${CLUSTER}-pp"
gcloud compute resource-policies create group-placement "$POLICY" \
    --collocation=COLLOCATED --project="$PROJECT" --region="${ZONE%-*}" 2>/dev/null || true

# --- 2. Launch N nodes -------------------------------------------------
for i in $(seq 0 $((N_NODES-1))); do
    NAME="${CLUSTER}-n${i}"
    echo "[gcp-phase2] launching $NAME"
    gcloud compute instances create "$NAME" \
        --project="$PROJECT" --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --resource-policies="$POLICY" \
        --maintenance-policy=TERMINATE \
        --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-ssd \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --metadata="install-nvidia-driver=True,ROLE=$i" \
        ${SERVICE_ACCOUNT:+--service-account=$SERVICE_ACCOUNT} \
        --scopes=cloud-platform \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC" \
        --network-interface="network=default,nic-type=GVNIC"
done

# --- 3. Wait for SSH then bootstrap each node -------------------------
echo "[gcp-phase2] waiting for SSH on all nodes"
sleep 45

for i in $(seq 0 $((N_NODES-1))); do
    NAME="${CLUSTER}-n${i}"
    gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" -- "
set -euo pipefail
# git + gcsfuse + repo
sudo apt-get update -qq
sudo apt-get install -y git git-lfs fuse zstd
if ! command -v gcsfuse >/dev/null; then
    wget -q https://github.com/GoogleCloudPlatform/gcsfuse/releases/latest/download/gcsfuse_Linux_x86_64.deb
    sudo dpkg -i gcsfuse_*.deb
fi
mkdir -p ~/gcs
gcsfuse --implicit-dirs ${BUCKET#gs://} ~/gcs || true

if [[ ! -d ~/market-pattern-bot ]]; then
    git clone --depth=1 '$REPO_URL' ~/market-pattern-bot
fi
cd ~/market-pattern-bot
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch pyyaml wandb google-cloud-storage gcsfs pyarrow
# The DLP image already ships transformer-engine + flash-attn; verify:
python -c 'import transformer_engine.pytorch; import flash_attn; print(\"TE+FA3 ok\")'

# Build the Hopper-native kernels in-place
cd odte/kernels
python setup.py build_ext --inplace
python -c 'import odte_kernels_cu; print(\"kernels ok\")'
" &
done
wait
echo "[gcp-phase2] bootstrap complete on all nodes"

HEAD="${CLUSTER}-n0"
HEAD_IP="$(gcloud compute instances describe "$HEAD" \
    --zone "$ZONE" --project "$PROJECT" \
    --format='value(networkInterfaces[0].networkIP)')"

cat <<EOF

[gcp-phase2] CLUSTER READY
    head node:  $HEAD  ($HEAD_IP)
    ssh:        gcloud compute ssh $HEAD --zone $ZONE --project $PROJECT

NEXT STEP — launch training:
    (on every node, from your laptop:)
    ./infra/gcp/launch_torchrun_524m.sh

Terminate cluster when done:
    for i in \$(seq 0 $((N_NODES-1))); do
        gcloud compute instances delete ${CLUSTER}-n\$i --zone $ZONE --project $PROJECT --quiet
    done
    gcloud compute resource-policies delete $POLICY --region=${ZONE%-*} --project $PROJECT
EOF
