#!/bin/bash
# N2 sequence model (odte/research/seq.py): one seed per array task, one GPU each.
#   sbatch infra/sol/seq.sbatch      -> reports/meta/seq/preds_seed{0,1,2}.parquet
#SBATCH -J meta-seq
#SBATCH -p public
#SBATCH -q public
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-04:00
#SBATCH --array=0-2
#SBATCH -o logs/meta_seq_%A_%a.out
PY="$HOME/.conda/envs/odte/bin/python"
cd "$HOME/silicon-alpha"
export PYTHONPATH="$PWD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -W ignore -u -m odte.research.seq run "$SLURM_ARRAY_TASK_ID"
