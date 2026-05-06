#!/bin/bash
# H100 PROBE — try training on Hopper. May fail if grp_jadriazo lacks
# H100 allocation; the error message will say so explicitly.
#
# H100 advantages over A100:
#   - 2-3× compute throughput on bf16
#   - FP8 via transformer_engine (set fp8=true in config to test)
#   - Larger HBM (80GB SXM same, but 3 TB/s vs 2 TB/s bandwidth)
#
# If this gets "no nodes available" or "qos invalid", fall back to
# train_multimodal_2node.sh (A100 multi-node) which we know works.
#SBATCH --account=grp_jadriazo
#SBATCH --partition=general
#SBATCH --qos=private
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=sa-train-mm-h100
#SBATCH --output=/scratch/%u/sol_logs/train_mm_h100_%j.out
#SBATCH --error=/scratch/%u/sol_logs/train_mm_h100_%j.err

set -e
cd /scratch/$USER/silicon-alpha
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate tradefm

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

CKPT_DIR=/scratch/$USER/checkpoints/tradefm_524m_mm_h100
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
