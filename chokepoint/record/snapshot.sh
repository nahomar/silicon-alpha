#!/bin/bash
# Scheduled recorder: prices every run, supply sources once a day.
#
# launchd runs jobs with a minimal environment and no working directory, so
# every path here is absolute and nothing is inherited from an interactive
# shell. If you move the repo, edit REPO below and reload the agent.
#
# NOTE: the repo must NOT live under ~/Downloads, ~/Documents or ~/Desktop.
# Those are TCC-protected and a launchd agent cannot execute or read there --
# it registers cleanly, reports exit 126, and never runs.
set -uo pipefail   # NOT -e: one failing ticker must not kill the whole run

REPO="/Users/nahom/projects/silicon-alpha"
LOG_DIR="$REPO/data/market/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/snapshot-$(date +%Y-%m).log"

cd "$REPO" || { echo "$(date -u +%FT%TZ) FATAL: repo missing at $REPO" >> "$LOG"; exit 1; }

# --- prices: every run -------------------------------------------------------
echo "--- $(date -u +%FT%TZ) snapshot start" >> "$LOG"
PYTHONPATH="$REPO" /usr/bin/python3 -m chokepoint.record.recorder snapshot \
  --pause 0.2 >> "$LOG" 2>&1
echo "--- $(date -u +%FT%TZ) snapshot exit=$?" >> "$LOG"

# --- supply: once a day ------------------------------------------------------
# Customs and bulletin data are MONTHLY. Polling them twelve times a day would
# write nothing eleven times and risk a rate-limit ban for the privilege --
# ComexStat answers 429 rather than throttling, and a 429 is silent data loss.
# A date stamp keeps it to one attempt per calendar day.
STAMP="$REPO/data/supply/.last_poll"
TODAY="$(date -u +%F)"
if [ "$(cat "$STAMP" 2>/dev/null)" != "$TODAY" ]; then
    echo "--- $(date -u +%FT%TZ) supply poll start" >> "$LOG"
    PYTHONPATH="$REPO" /usr/bin/python3 -m chokepoint.record.supply poll \
      >> "$LOG" 2>&1
    rc=$?
    echo "--- $(date -u +%FT%TZ) supply poll exit=$rc" >> "$LOG"
    # Stamp only on success, so a failed poll retries on the next scheduled run
    # rather than being skipped until tomorrow.
    if [ $rc -eq 0 ]; then
        mkdir -p "$(dirname "$STAMP")"
        echo "$TODAY" > "$STAMP"
    fi
fi
