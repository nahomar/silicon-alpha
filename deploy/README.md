# Always-on bot server (DigitalOcean, New York, next to IBKR)

Why: on a laptop the bot stops whenever the lid closes (2026-11-12: no trading, Gateway not
running, downloads stalled). The live parts run here; research stays on the Mac / Sol.

| Runs on the server | Schedule (systemd timers, Eastern time) |
|---|---|
| IB Gateway (Docker + IBC: logs in by itself, restarts nightly) | always |
| health check (alerts the phone if not ready) | weekdays 08:40, 08:55 |
| trading bot (paper) | weekdays 09:00 (= 06:00 PT) |
| evening watchdog + private records backup | weekdays 16:30 |
| weekly loop | Saturday 11:00 |
| Africa GDELT collector | every 15 min |
| CHOKEPOINT recorder | every 2 h |

Alerts go to the owner's phone through **ntfy** (app, no account) and to data/live/logs/alerts.log.

## Steps

**1. Owner: create the server.** DigitalOcean → Create → Droplets:
- Region: **New York (NYC3)**. IBKR's main US data centres are in the NY/NJ area.
- Image: **Ubuntu 24.04 (LTS) x64**.
- Size: Basic, Regular, **2 vCPU / 4 GB / 80 GB** (about $24/month; check the price shown).
- Authentication: **SSH key**. Paste the Mac's public key from
  `cat ~/.ssh/silicon_server_ed25519.pub`.
- Hostname: `silicon-bot`. Create, then send Claude the IPv4 address.

**2. Claude: set up** (from the Mac, using that key):
`ssh -i ~/.ssh/silicon_server_ed25519 root@IP 'bash -s' < deploy/setup.sh`

**3. Owner: GitHub deploy key** for the private records backup.
- setup prints a public key. In github.com/nahomar/silicon-alpha-records → Settings → Deploy keys
  → Add, tick "Allow write access", and paste it.
- Claude then clones the records repo on the server.

**4. Owner: secrets** (typed or copied by the owner only):
- IBKR paper login: `ssh -i ~/.ssh/silicon_server_ed25519 silicon@IP`, then
  `nano ~/projects/silicon-alpha/deploy/ib.env` (TWS_USERID, TWS_PASSWORD), save, and run
  `chmod 600 ~/projects/silicon-alpha/deploy/ib.env`.
- Databento key and ntfy topic, from the Mac:
  `scp -i ~/.ssh/silicon_server_ed25519 ~/.config/silicon-alpha/databento.env ~/.config/silicon-alpha/ntfy_topic silicon@IP:.config/silicon-alpha/`
- Phone: install **ntfy** and subscribe to the topic in `~/.config/silicon-alpha/ntfy_topic` (on
  the Mac: `cat ~/.config/silicon-alpha/ntfy_topic`).

**5. Claude: data + Gateway.**
- `bash deploy/sync_data.sh IP` (about 30 MB).
- Start the Gateway: `cd deploy && docker compose up -d` (as root).
- `python -m odte.live.health` must report a paper account (D…).

**6. Claude: switch over.**
- Enable the timers: `systemctl enable --now silicon-{health,bot,evening,weekly,gdelt,chokepoint}.timer`.
- Run one full trading day in parallel. Then the owner turns the Mac's bot jobs off:
  `launchctl bootout gui/$UID/com.siliconalpha.bot.daily` (and .evening, .weekly, .africa.gdelt,
  com.nahomar.chokepoint.snapshot).
- **The Mac and the server must never both trade.** Both use the same paper account; only one
  bot may run on a given day.

## Notes
- **One IBKR market-data session per username:** don't stay logged into TWS / the IBKR app with
  the same live username during market hours, or the paper feed can drop.
- IBKR occasionally forces a re-login, and paper accounts normally need no phone 2FA. If it ever
  asks, the health check alerts the phone at 08:40 ET.
- The API listens on 127.0.0.1 only; the firewall allows SSH only.
- Update code: `sudo -u silicon git -C ~silicon/projects/silicon-alpha pull --ff-only`.
