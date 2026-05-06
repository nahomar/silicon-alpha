#!/bin/bash
# Multi-node multimodal 524M — 2 nodes × 2 A100 = 4 A100 total via
# InfiniBand. Uses srun + torchrun with c10d rendezvous (matches the
# yesterday's NCCL smoke that already passed).
#
# Effective FSDP shard size: 524M / 4 ≈ 131M params per rank.
# Plenty of memory headroom on 80GB A100s for ctx_len=4096.
#SBATCH --account=grp_jadriazo
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --job-name=sa-train-mm-2n
#SBATCH --output=/scratch/%u/sol_logs/train_mm_2n_%j.out
#SBATCH --error=/scratch/%u/sol_logs/train_mm_2n_%j.err

set -e
cd /scratch/$USER/silicon-alpha
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate tradefm

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

CKPT_DIR=/scratch/$USER/checkpoints/tradefm_524m_mm_2n
mkdir -p "$CKPT_DIR" /scratch/$USER/sol_logs

srun torchrun --nproc_per_node=2 --nnodes=2 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m odte.train.distributed \
  --config configs/tradefm_524m_multimodal.yml \
  --shards "/scratch/$USER/data/packed/multimodal/shard_*.parquet" \
  --ckpt-store "$CKPT_DIR" \
  --ckpt-prefix tradefm_524m_mm \
  --steps 8000 \
  --batch 2 \
  --grad-accum 4 \
  --ckpt-every 500 \
  --log-every 25 \
  --num-workers 4

echo "training done at $(date)"
