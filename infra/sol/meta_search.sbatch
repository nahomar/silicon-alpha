#!/bin/bash
# Phase-3 search: every subset of the 9 searchable feature groups x 4 ridge strengths (2,048 points).
# Each array task runs 16 points (128 tasks).   sbatch infra/sol/meta_search.sbatch
#SBATCH -J meta-search
#SBATCH -p public
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 0-04:00
#SBATCH --array=0-127%64
#SBATCH -o logs/meta_search_%A_%a.out
PY="$HOME/.conda/envs/odte/bin/python"
cd "$HOME/silicon-alpha"
export PYTHONPATH="$PWD"
START=$(( SLURM_ARRAY_TASK_ID * 16 ))
"$PY" -W ignore -m odte.research.meta search $(seq $START $(( START + 15 )))
