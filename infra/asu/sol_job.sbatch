#!/bin/bash
# GPU job template for ASU Sol (Slurm). Submit from the repo root on Sol:
#   sbatch -A <your account> infra/asu/sol_job.sbatch
# (the account is passed on the command line so no personal id lives in git)
# Checked 2026-11-08 with infra/asu/check_sol.sh: partitions public (7 d)
# and general (14 d); QOS public; GPUs include a100, h100, h200, l40, a30.
#SBATCH --job-name=odte-study
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
module load mamba/latest
source activate odte                          # env created once: see infra/asu/README.md
nvidia-smi || true
# The data folders are not in git: copy them first (rsync, see README).
PYTHONPATH=. python -m odte.eval.expiry_edge --seq --min-train-days 40 --write
