#!/bin/bash
# Read-only check that ASU Sol is reachable and usable. Changes nothing on Sol.
#
# Sol's login host has no public DNS: connect the ASU VPN first
# (Cisco Secure Client -> sslvpn.asu.edu, ASURITE + Duo). sol.asu.edu is a
# public website, not the SSH host.
set -u
# Personal settings live in infra/asu/sol.env (gitignored); see sol.env.example.
ENV_FILE="$(dirname "$0")/sol.env"
[ -f "$ENV_FILE" ] && . "$ENV_FILE"
HOST="${SOL_HOST:-login.sol.rc.asu.edu}"
USER_="${SOL_USER:?set SOL_USER in infra/asu/sol.env}"
KEY="${SOL_KEY:-$HOME/.ssh/id_ed25519}"

echo "1. DNS for $HOST"
if ! dig +short "$HOST" | grep -q .; then
  echo "   FAIL: $HOST does not resolve -> the ASU VPN is not connected."
  echo "   Open Cisco Secure Client, connect to sslvpn.asu.edu, sign in, then re-run."
  exit 1
fi
echo "   ok: $(dig +short "$HOST" | head -1)"

echo "2. SSH with key $KEY (no password prompts)"
if ! ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$USER_@$HOST" true 2>/tmp/sol_ssh_err; then
  echo "   FAIL: $(tail -1 /tmp/sol_ssh_err)"
  echo "   If it says 'Permission denied (publickey)': the key is not on Sol yet."
  echo "   Log in once interactively (ssh $USER_@$HOST) and append"
  echo "   $(basename "$KEY").pub to ~/.ssh/authorized_keys there."
  exit 2
fi
echo "   ok"

echo "3. What Sol offers this account (read-only)"
ssh -i "$KEY" -o BatchMode=yes "$USER_@$HOST" '
  echo "   host: $(hostname)"
  echo "   partitions:"; sinfo -s 2>/dev/null | sed "s/^/     /"
  echo "   GPU types:"; sinfo -h -o "%G" 2>/dev/null | tr "," "\n" | grep -i gpu | sort -u | sed "s/^/     /"
  echo "   my accounts/QOS:"; sacctmgr -n show assoc user=$USER format=account,qos 2>/dev/null | sed "s/^/     /"
  echo "   storage:"; (mydisk 2>/dev/null || df -h ~ 2>/dev/null) | sed "s/^/     /"
  echo "   python/cuda modules:"; (module -t avail python 2>&1; module -t avail cuda 2>&1) | grep -i -E "python|cuda" | head -8 | sed "s/^/     /"
'
echo "Done. Next: infra/asu/sol_job.sbatch (adjust partition/QOS to what step 3 printed)."
