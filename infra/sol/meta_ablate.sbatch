#!/bin/bash
# Phase 3 group ablation on Sol: one array task per dropped feature group (index 0 = nothing dropped).
#   sbatch --export=ALL,MODEL=ml_ridge infra/asu/meta_ablate.sbatch
#SBATCH -J meta-ablate
#SBATCH -p public
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH --array=0-10
#SBATCH -o logs/meta_ablate_%A_%a.out
GROUP_LIST=(none session calendar context gex term flow strategy cross factory identity)
PY="$HOME/.conda/envs/odte/bin/python"
cd "$HOME/silicon-alpha"
export PYTHONPATH="$PWD"
"$PY" -W ignore -m odte.research.meta ablate "${MODEL:-ml_ridge}" "${GROUP_LIST[$SLURM_ARRAY_TASK_ID]}"
