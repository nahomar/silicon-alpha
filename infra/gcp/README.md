# Google Cloud Platform — TradeFM training playbook

HRT's July-2024 partnership with Google Cloud put their compute-heavy
simulations on GCP A3 clusters. We target the same stack:

- **Phase 1 (40M TradeFM):** `a2-ultragpu-1g` — single A100 80GB, ~$3-4/hr
- **Phase 2 (524M TradeFM):** `a3-megagpu-8g` × 3 nodes = 24× H100 SXM
- **Colab Pro+ alternative for Phase 1:** priority A100 access, ~$50/mo

## Why GCP instead of RunPod / Lambda for this stack

1. **A3 Mega's 1600 Gbps GPUDirect-TCPX** matches IB all-reduce throughput
   at this scale without requiring Mellanox ConnectX-7 drivers or the
   `nv_peer_memory` kernel module — everything is pre-configured in the
   `pytorch-cu124-transformerengine-flashattn` Deep Learning image.
2. **GCS as a first-class checkpoint store.** `odte/train/checkpoint.py`'s
   `FsspecStore` already accepts `gs://` URLs; no code change needed.
3. **Ephemeral, quota-priced** — if a run dies you delete the instances;
   no monthly bill like a colo.

## Quota prep (one-time, 24–48 h ahead)

Request in IAM → Quotas:

| Quota | Region | Amount |
|---|---|---|
| NVIDIA A100 80GB GPUs | `us-central1` | 1 (Phase 1) |
| NVIDIA H100 80GB GPUs | `us-central1` | **24** (Phase 2, 3 nodes × 8) |
| CPUs (all regions)   | global | 144 |
| SSD Persistent Disk  | `us-central1` | 6000 GB |

A100 approval is usually instant. H100 approval takes 24–48 h; open the
request at least 2 business days before the intended training window.

## Buckets + IAM

```bash
# one-time setup
gcloud storage buckets create gs://my-tradefm \
    --project=$GCP_PROJECT --location=us-central1 --uniform-bucket-level-access

# training service account (compute default works; this is cleaner)
gcloud iam service-accounts create tradefm-trainer --project=$GCP_PROJECT
gcloud storage buckets add-iam-policy-binding gs://my-tradefm \
    --member=serviceAccount:tradefm-trainer@$GCP_PROJECT.iam.gserviceaccount.com \
    --role=roles/storage.objectAdmin
export GCP_SERVICE_ACCOUNT=tradefm-trainer@$GCP_PROJECT.iam.gserviceaccount.com
```

## Phase 1 — 40M run

```bash
export GCP_PROJECT=myproject
export GCP_ZONE=us-central1-a
export GCP_BUCKET=gs://my-tradefm
export REPO_URL=https://github.com/you/market-pattern-bot

./infra/gcp/phase1_a100.sh

# SSH in:
gcloud compute ssh tradefm-phase1 --zone us-central1-a --project $GCP_PROJECT
cd ~/market-pattern-bot && . .venv/bin/activate

# Pack DataShop csv.zst → parquet shards on the mounted bucket:
python -c "from odte.data import pack_folder; pack_folder('~/gcs/cboe_csv', '~/gcs/opra_shards')"

# Train:
DEVICE=cuda python -m odte.train.pretrain_tradefm \
    --config configs/tradefm_40m.yml \
    --shards '~/gcs/opra_shards/opra_*.parquet' \
    --ckpt-dir ~/gcs/ckpts/tradefm_40m \
    --steps 20000 --batch 64 --grad-accum 2
```

## Phase 2 — 24× H100 run

```bash
export GCP_PROJECT=myproject
export GCP_ZONE=us-central1-a          # A3 Mega available in us-central1, europe-west4, asia-east1
export GCP_N_NODES=3
export GCP_CLUSTER=tradefm-524m
export GCP_BUCKET=gs://my-tradefm
export REPO_URL=https://github.com/you/market-pattern-bot

./infra/gcp/phase2_a3mega.sh           # provisions 3× a3-megagpu-8g
./infra/gcp/launch_torchrun_524m.sh    # spawns torchrun on every node

# Monitor
gcloud compute ssh tradefm-524m-n0 --zone $GCP_ZONE -- tail -f ~/train.log
# or W&B: tradefm-524m
```

## TCPX vs InfiniBand — source the right NCCL env

- `deploy/nccl_env.sh` → **InfiniBand** (for Mellanox ConnectX-7 boxes:
  on-prem, RunPod bare-metal, some Lambda reservations).
- `infra/gcp/tcpx_nccl_env.sh` → **GCP GPUDirect-TCPX**
  (A3 High / A3 Mega). Sets `NCCL_NET_PLUGIN=/usr/local/gib/lib64/libnccl-net.so`,
  unsets `NCCL_IB_HCA`, uses `eth0..eth7` instead of `ens1`.

**Never source both in the same shell** — the IB-specific vars will
confuse NCCL's GCP TCPX plugin and init will hang indefinitely.

## Checkpoint strategy

`odte/train/checkpoint.py::CheckpointManager` with `store_url=gs://…/ckpts/tradefm`
writes one sharded file per rank per step directly to GCS. This lets you:

- Kill a node mid-run without losing progress
- Resume on any number of nodes (as long as world_size matches)
- Copy best.pt to a different region with `gcloud storage cp` for
  downstream DML fine-tune / RL / live paper

```bash
# download the best ckpt back to your Mac:
gcloud storage cp gs://my-tradefm/ckpts/tradefm/step_00020000/rank_0.pt \
    ./checkpoints/tradefm_524m.pt
```

## Cost estimates (2026 on-demand, us-central1)

| Phase | Instance | $/hr | Full run | Notes |
|---|---|---|---|---|
| 1 | a2-ultragpu-1g (1×A100-80G) | ~$4 | ~$150-250 | 40M, 20k steps, 40-60h |
| 1 | Colab Pro+ priority A100 | ~$50/mo | capped 24h sessions | Good for iterating; not for production |
| 2 | a3-highgpu-8g × 3 (24× H100) | ~$255 | ~$35-45k | budget tier — slower all-reduce |
| 2 | a3-megagpu-8g × 3 (24× H100) | ~$300 | ~$40-50k | recommended — 2× network |
| 2 | a3-ultragpu-8g × 3 (24× H200) | ~$380 | ~$50-60k | future-proofed, 141 GB HBM3e |

Preemptible saves ~50–70 % but GCP can reclaim any time; only useful
with frequent ckpt + `--resume`. `odte/train/checkpoint.py` handles this
correctly — it discovers the latest step at resume.

## Teardown

```bash
# Phase 2
for i in 0 1 2; do
    gcloud compute instances delete tradefm-524m-n$i --zone $GCP_ZONE --project $GCP_PROJECT --quiet
done
gcloud compute resource-policies delete tradefm-524m-pp \
    --region ${GCP_ZONE%-*} --project $GCP_PROJECT --quiet

# Phase 1
gcloud compute instances delete tradefm-phase1 --zone $GCP_ZONE --project $GCP_PROJECT --quiet
```

Always tear down when a run ends — A3 Mega is ~$7,200/day if you forget.

## Integration points with the deploy/ runbook

After training, the produced `gs://bucket/ckpts/tradefm/step_*/rank_0.pt`
is what `deploy/monday_go_live.sh` expects as the 524M checkpoint. Copy
it into `checkpoints/tradefm_524m.pt` before running the Monday preflight:

```bash
gcloud storage cp \
    gs://my-tradefm/ckpts/tradefm_524m_20260418/best/rank_0.pt \
    checkpoints/tradefm_524m.pt
```

## Honest caveats specific to GCP

1. **A3 Mega availability is variable.** Some zones never see capacity;
   `us-central1-a/b/c` and `europe-west4-b` are the most reliable. The
   provisioner will error loudly if the zone is out of quota.
2. **Networking quirks.** The 8 GVE NICs are not fully redundant — losing
   one causes NCCL to fall back to TCP over the primary interface and
   throughput drops ~8×. Monitor with `nvidia-smi dmon -s pucvmet`.
3. **No IB fabric.** If you want the on-prem HRT-style 400 Gbps IB, you
   need Cloud TPU (different architecture) or a different provider (CoreWeave,
   Lambda reserved). GCP A3's TCPX is close but not identical.
