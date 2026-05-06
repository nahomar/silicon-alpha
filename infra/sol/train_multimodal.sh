#!/bin/bash
# Phase 2.5 multimodal 524M training on Sol.
#
# Single-node × 2 A100-80GB (NVLink intra-node). Phase-2 NCCL smoke
# already retired; this is the actual training.
#
# Time-bound: 4 hours wall-clock for ~5k steps at d_model=2048,
# ctx_len=4096, batch=2, grad_accum=4 → effective batch=16. Adjust steps
# for your token budget (~83M rows × 7 tokens × 4 ctx-overlap ≈ 2.3B
# training tokens, so 5k steps × 16 × 4096 ≈ 327M tokens consumed —
# ~14% of corpus, sufficient for a meaningful smoke).
#
# Usage:
#   sbatch infra/sol/train_multimodal.sh
#SBATCH --account=grp_jadriazo
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=sa-train-mm
#SBATCH --output=/scratch/%u/sol_logs/train_mm_%j.out
#SBATCH --error=/scratch/%u/sol_logs/train_mm_%j.err

set -e
cd /scratch/$USER/silicon-alpha
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate tradefm

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN  # quieter than INFO; flip to INFO if NCCL acts up

CKPT_DIR=/scratch/$USER/checkpoints/tradefm_524m_mm
mkdir -p "$CKPT_DIR" /scratch/$USER/sol_logs

torchrun --nproc_per_node=2 --nnodes=1 \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m odte.train.distributed \
  --config configs/tradefm_524m_multimodal.yml \
  --shards "/scratch/$USER/data/packed/multimodal/shard_*.parquet" \
  --ckpt-store "$CKPT_DIR" \
  --ckpt-prefix tradefm_524m_mm \
  --steps 5000 \
  --batch 2 \
  --grad-accum 4 \
  --ckpt-every 500 \
  --log-every 25 \
  --num-workers 4

echo "training done at $(date)"
