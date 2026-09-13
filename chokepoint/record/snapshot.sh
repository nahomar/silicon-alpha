#!/bin/bash
# Wrapper for scheduled snapshots.
#
# launchd runs jobs with a minimal environment and no working directory, so
# every path here is absolute and nothing is inherited from an interactive
# shell. If you move the repo, edit REPO below and reload the agent.
set -uo pipefail   # NOT -e: one failing ticker must not kill the whole run

REPO="/Users/nahom/projects/silicon-alpha"
LOG_DIR="$REPO/data/market/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/snapshot-$(date +%Y-%m).log"

cd "$REPO" || { echo "$(date -u +%FT%TZ) FATAL: repo missing at $REPO" >> "$LOG"; exit 1; }

echo "--- $(date -u +%FT%TZ) snapshot start" >> "$LOG"
PYTHONPATH="$REPO" /usr/bin/python3 -m chokepoint.record.recorder snapshot \
  --pause 0.2 >> "$LOG" 2>&1
echo "--- $(date -u +%FT%TZ) snapshot exit=$?" >> "$LOG"
