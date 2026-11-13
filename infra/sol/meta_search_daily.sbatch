#!/bin/bash
# T2: the Phase-3 grid again, saving each configuration's daily P&L (reports/meta/search_daily/).
# Each array task runs 16 points (128 tasks).   sbatch infra/sol/meta_search_daily.sbatch
#SBATCH -J meta-sdaily
#SBATCH -p public
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH -t 0-04:00
#SBATCH --array=0-127%64
#SBATCH -o logs/meta_sdaily_%A_%a.out
PY="$HOME/.conda/envs/odte/bin/python"
cd "$HOME/silicon-alpha"
export PYTHONPATH="$PWD"
START=$(( SLURM_ARRAY_TASK_ID * 16 ))
"$PY" -W ignore -m odte.research.meta search_daily $(seq $START $(( START + 15 )))
