#!/bin/bash
# GDELT Africa live collector (CHOKEPOINT stage 2), run by launchd every 15 minutes.
# Catches up from the last stored batch, so a sleeping laptop loses nothing (up to 3 days;
# older gaps are filled by `python3 -m chokepoint.data.gdelt backfill`).
set -uo pipefail
REPO="${REPO:-/Users/nahom/projects/silicon-alpha}"
PY="${PY:-$REPO/.venv-live/bin/python}"
LOG_DIR="$REPO/data/africa/logs"
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1
PYTHONPATH="$REPO" "$PY" -W ignore -m chokepoint.data.gdelt live >> "$LOG_DIR/gdelt-live-$(date +%Y-%m).log" 2>&1
