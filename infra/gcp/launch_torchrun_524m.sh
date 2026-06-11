#!/usr/bin/env bash
# Launch the distributed 524M TradeFM training across N A3 Mega nodes.
#
# Must be run from your local laptop (needs gcloud CLI). It SSHes into
# each node and starts a torchrun process with the right --node_rank.
#
# Environment required (same variables as phase2_a3mega.sh):
#   GCP_PROJECT GCP_ZONE GCP_N_NODES GCP_CLUSTER GCP_BUCKET

set -euo pipefail

PROJECT="${GCP_PROJECT:?set GCP_PROJECT}"
ZONE="${GCP_ZONE:-us-central1-a}"
N_NODES="${GCP_N_NODES:-3}"
CLUSTER="${GCP_CLUSTER:-tradefm-524m}"
BUCKET="${GCP_BUCKET:?set GCP_BUCKET}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CKPT_PREFIX="${CKPT_PREFIX:-tradefm_524m_$(date -u +%Y%m%d)}"

HEAD="${CLUSTER}-n0"
HEAD_IP="$(gcloud compute instances describe "$HEAD" \
    --zone "$ZONE" --project "$PROJECT" \
    --format='value(networkInterfaces[0].networkIP)')"
echo "[launch] head=$HEAD  ip=$HEAD_IP"

for i in $(seq 0 $((N_NODES-1))); do
    NAME="${CLUSTER}-n${i}"
    echo "[launch] starting node_rank=$i on $NAME"
    gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" -- "
set -euo pipefail
cd ~/silicon-alpha
. .venv/bin/activate
source infra/gcp/tcpx_nccl_env.sh
nohup torchrun \
    --nnodes=$N_NODES --node_rank=$i --nproc_per_node=8 \
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_IP:29500 \
    -m odte.train.distributed \
    --config configs/tradefm_524m.yml \
    --shards 'file:///home/\$USER/gcs/opra_shards/opra_*.parquet' \
    --ckpt-store gs://${BUCKET#gs://}/ckpts/$CKPT_PREFIX \
    --ckpt-prefix tradefm \
    --steps $STEPS --batch $BATCH --grad-accum $GRAD_ACCUM \
    --ckpt-every 1000 --log-every 50 \
    --num-workers 4 \
    --eval-shards 'file:///home/\$USER/gcs/opra_shards_eval/opra_*.parquet' \
    --eval-every 2000 \
    --wandb tradefm-524m \
    --resume \
    > ~/train.log 2>&1 &
echo \"rank $i launched PID=\$!\"
" &
done
wait

echo
echo "[launch] All ranks launched. Monitor:"
echo "    gcloud compute ssh $HEAD --zone $ZONE --project $PROJECT -- tail -f ~/train.log"
echo "    or W&B project: tradefm-524m"
echo
echo "Checkpoints:  gs://${BUCKET#gs://}/ckpts/$CKPT_PREFIX/"
