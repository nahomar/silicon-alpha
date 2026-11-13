#!/bin/bash
# Scheduled Reddit catch-up: fetch everything since the newest saved post.
#
# launchd gives no working directory or PATH, so every path is absolute. If
# the repo moves, edit REPO and reload the agent. The repo must not live
# under ~/Downloads, ~/Documents or ~/Desktop (TCC blocks launchd there).
# A run that finds another sync holding the lock exits immediately, so this
# is safe to schedule while a long backfill is still going.
set -uo pipefail

REPO="/Users/nahom/projects/silicon-alpha"
LOG_DIR="$REPO/data/social/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/reddit-$(date +%Y-%m).log"

cd "$REPO" || { echo "$(date -u +%FT%TZ) FATAL: repo missing at $REPO" >> "$LOG"; exit 1; }
echo "--- $(date -u +%FT%TZ) reddit sync start" >> "$LOG"
# --start makes every run also fill any history still missing since that
# date (newest-first, so an interrupted fill never leaves a hole).
PYTHONPATH="$REPO" /usr/bin/python3 -W ignore -m social.reddit sync --start 2026-10-01 >> "$LOG" 2>&1
echo "--- $(date -u +%FT%TZ) reddit sync exit=$?" >> "$LOG"

# Stocktwits has its own lock, so it runs even while a Reddit backfill
# holds the Reddit one (and vice versa).
echo "--- $(date -u +%FT%TZ) stocktwits sync start" >> "$LOG"
# Index streams back to Nov 2025 so the 12-month 0DTE study has Stocktwits
# features. SPY is live-only: at ~6,400 messages a day and ~200 calls/hour,
# even one month of SPY history would take ~40 hours to page through.
PYTHONPATH="$REPO" /usr/bin/python3 -W ignore -m social.stocktwits sync --start 2025-11-01 --symbols SPX ES_F VIX >> "$LOG" 2>&1
PYTHONPATH="$REPO" /usr/bin/python3 -W ignore -m social.stocktwits sync --symbols SPY >> "$LOG" 2>&1
echo "--- $(date -u +%FT%TZ) stocktwits sync exit=$?" >> "$LOG"
