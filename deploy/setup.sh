#!/bin/bash
# One-time (idempotent) setup of the always-on bot server. Ubuntu 24.04, run as root:
#   curl -fsSL https://raw.githubusercontent.com/nahomar/silicon-alpha/claude/calendar-features/deploy/setup.sh | bash
# or, after cloning: sudo bash deploy/setup.sh
# Creates user 'silicon', the repo at /home/silicon/projects/silicon-alpha, the bot's Python env,
# Docker for IB Gateway, the systemd timers (stopped until the owner finishes deploy/README.md),
# and a firewall that allows SSH only.
set -euo pipefail
BRANCH="${BRANCH:-claude/calendar-features}"
U=silicon
HOME_U=/home/$U
REPO=$HOME_U/projects/silicon-alpha

echo "== packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3-venv python3-dev build-essential docker.io docker-compose-v2 ufw rsync curl jq >/dev/null
systemctl enable --now docker

echo "== user"
id -u $U >/dev/null 2>&1 || useradd -m -s /bin/bash $U
mkdir -p $HOME_U/projects $HOME_U/.config/silicon-alpha $HOME_U/.ssh
chmod 700 $HOME_U/.config/silicon-alpha $HOME_U/.ssh
if [ -f /root/.ssh/authorized_keys ] && [ ! -f $HOME_U/.ssh/authorized_keys ]; then
  cp /root/.ssh/authorized_keys $HOME_U/.ssh/authorized_keys
fi
chown -R $U:$U $HOME_U

echo "== code"
if [ ! -d $REPO/.git ]; then
  sudo -u $U git clone -q --branch "$BRANCH" https://github.com/nahomar/silicon-alpha.git $REPO
else
  sudo -u $U git -C $REPO pull -q --ff-only origin "$BRANCH"
fi

echo "== python env (same pinned versions as the Mac bot env)"
if [ ! -x $REPO/.venv-live/bin/python ]; then
  sudo -u $U python3 -m venv $REPO/.venv-live
fi
sudo -u $U $REPO/.venv-live/bin/pip install -q --upgrade pip
sudo -u $U $REPO/.venv-live/bin/pip install -q -r $REPO/deploy/requirements-live.txt PyYAML

echo "== records repo deploy key (owner adds the public key to nahomar/silicon-alpha-records with write access)"
if [ ! -f $HOME_U/.ssh/records_deploy ]; then
  sudo -u $U ssh-keygen -q -t ed25519 -N "" -C "silicon-bot-server records" -f $HOME_U/.ssh/records_deploy
fi
cat > $HOME_U/.ssh/config <<EOF
Host github-records
  HostName github.com
  User git
  IdentityFile $HOME_U/.ssh/records_deploy
  IdentitiesOnly yes
EOF
chown $U:$U $HOME_U/.ssh/config && chmod 600 $HOME_U/.ssh/config

echo "== IB Gateway image"
cd $REPO/deploy
[ -f ib.env ] || { cp ib.env.example ib.env; chmod 600 ib.env; chown $U:$U ib.env; }
docker compose pull -q || IB_IMAGE=ghcr.io/gnzsnz/ib-gateway:stable docker compose pull -q
docker image inspect --format '{{index .RepoDigests 0}}' $(docker compose config --images | head -1) || true

echo "== systemd timers (installed, NOT started: deploy/README.md step 6 starts them)"
cp $REPO/deploy/systemd/*.service $REPO/deploy/systemd/*.timer /etc/systemd/system/
systemctl daemon-reload

echo "== firewall: SSH only (the IB API stays on 127.0.0.1)"
ufw allow OpenSSH >/dev/null
ufw --force enable >/dev/null

echo "== time"
timedatectl set-ntp true || true

echo
echo "DONE. Next (deploy/README.md):"
echo "  - owner: fill $REPO/deploy/ib.env (IBKR paper login), copy the Databento key and ntfy topic"
echo "  - owner: add this deploy key to nahomar/silicon-alpha-records (Settings > Deploy keys, allow write):"
cat $HOME_U/.ssh/records_deploy.pub
