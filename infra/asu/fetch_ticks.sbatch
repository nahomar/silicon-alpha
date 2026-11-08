#!/bin/bash -l
# Download the TradeFM tick days (configs/tradefm/tick_days.csv) to Sol scratch.
#   sbatch -A <your account> infra/asu/fetch_ticks.sbatch
# Needs ~/.config/silicon-alpha/databento.env (owner-only) containing
#   export DATABENTO_API_KEY=...
# The key is copied there by the user, never committed and never written to logs.
#SBATCH --job-name=fetch-ticks
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
set +x
. "$HOME/.config/silicon-alpha/databento.env"
PY="$HOME/.conda/envs/odte/bin/python"
OUT="/scratch/$USER/odte_ticks"
mkdir -p "$OUT"
PYTHONPATH=. "$PY" -W ignore -u -m odte.data.fetch_0dte --fast --keep-raw \
    --schema cmbp-1 --band 0.03 --days-file configs/tradefm/tick_days.csv \
    --start 2025-11-01 --end 2026-11-06 --max-usd 60 --workers 4 --out "$OUT"
ls -la "$OUT"
