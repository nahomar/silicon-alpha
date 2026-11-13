#!/bin/bash
# Copy what the live jobs need from the Mac to the server (about 30 MB). Safe to re-run.
#   bash deploy/sync_data.sh <server-ip>
# Secrets are NOT copied here: the owner copies them (deploy/README.md step 4).
set -euo pipefail
IP="${1:?usage: sync_data.sh <server-ip>}"
KEY="${KEY:-$HOME/.ssh/silicon_server_ed25519}"
DEST="silicon@$IP:/home/silicon/projects/silicon-alpha"
cd "$(dirname "$0")/.."
RS=(rsync -az --mkpath -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new")
"${RS[@]}" --exclude 'chains/*replay*' --exclude 'journal/*replay*' data/live/ "$DEST/data/live/"
"${RS[@]}" data/calendar/ "$DEST/data/calendar/"
"${RS[@]}" data/context/ "$DEST/data/context/"
"${RS[@]}" data/supply/ "$DEST/data/supply/" --exclude aci/
"${RS[@]}" data/market/ "$DEST/data/market/"
for d in odte_0dte odte_0dte_ndxp odte_0dte_rutw; do
  "${RS[@]}" "data/$d/settlements.csv" "$DEST/data/$d/"
done
"${RS[@]}" reports/expiry_edge/decisions-sealed-c1.parquet reports/expiry_edge/decisions-dev-gex-es.parquet \
  "$DEST/reports/expiry_edge/"
ssh -i "$KEY" "silicon@$IP" "mkdir -p /home/silicon/projects/silicon-alpha/data/live/logs /home/silicon/projects/silicon-alpha/data/africa/logs"
echo "synced to $IP"
