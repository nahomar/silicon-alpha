#!/bin/bash
# V2 multimodal training — single A100, 5-modality real-ts corpus.
#
# Compute estimate: ctx_len 2048 × batch 2 × grad_accum 8 = effective 32k
# tokens/step. 10k steps × 32k = 320M tokens consumed. At ~0.5 step/s on
# A100 with bf16 (V1 baseline), 10k steps = ~5.5 hours. Adds buffer.
#SBATCH --account=grp_jadriazo
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=sa-train-v2
#SBATCH --output=/scratch/%u/sol_logs/train_v2_%j.out
#SBATCH --error=/scratch/%u/sol_logs/train_v2_%j.err

set -e
cd /scratch/$USER/silicon-alpha
export PATH=/scratch/$USER/miniconda3/envs/tradefm/bin:$PATH
export PYTHONUNBUFFERED=1

CKPT_DIR=/scratch/$USER/checkpoints/tradefm_524m_mm_v2
mkdir -p "$CKPT_DIR" /scratch/$USER/sol_logs

which python
python --version
echo "--- starting V2 training ---"

python -m odte.train.distributed \
  --config configs/tradefm_524m_multimodal_v2.yml \
  --shards "/scratch/$USER/data/packed/multimodal_v2/shard_*.parquet" \
  --ckpt-store "$CKPT_DIR" \
  --ckpt-prefix tradefm_524m_mm_v2 \
  --steps 10000 \
  --batch 2 \
  --grad-accum 8 \
  --ckpt-every 1000 \
  --log-every 50 \
  --num-workers 4

echo "training done at $(date)"
