# Autonomous Training Runbook

This playbook collects everything you need to keep the ML training loop running on its own and know exactly when to jump in.

---

## 1. Prerequisites

- Python 3.10+
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- Populate `.env` with at least:
  ```dotenv
  DISCORD_WEBHOOK_URL=...
  TRAINING_WEBHOOK_URL=...  # optional (falls back to DISCORD_WEBHOOK_URL)
  DB_NAME=trading_bot
  DB_USER=postgres
  DB_PASS=...
  DB_HOST=localhost
  DB_PORT=5432
  APCA_API_KEY_ID=...
  APCA_API_SECRET_KEY=...
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
  ```
- PostgreSQL running (see `docker-compose.yml`) and seeded tickers (`python scripts/seed_tickers.py`).

---

## 2. One-Time Sanity Checks

1. Pull a fresh model once to make sure dependencies work:
   ```bash
   source .venv/bin/activate
   python scripts/train_backtest.py
   ```
2. Confirm output:
   - `data/backtests/latest.json` (performance payload)
   - `logs/progress.jsonl` / `logs/progress_latest.json` (training breadcrumbs)
   - Discord webhook message with training metrics

---

## 3. Start the Autonomous Training Daemon

The daemon wraps `scripts/train_backtest.py` in a loop with jitter, exponential backoff on failures, and Discord alerts.

```bash
./scripts/start_training.sh
```

What it does:
- sources `.env`
- picks an available Python (`.venv/bin/python` preferred)
- writes streaming output to `logs/training_daemon.log`
- records PID in `logs/training_daemon.pid`
- emits breadcrumb events to `logs/progress.jsonl`

**Multiple training profiles.** Pass a label (or export `TRAINING_LABEL`) to spin up parallel daemons with isolated PID/log files and distinct Discord prefixes:
```bash
./scripts/start_training.sh classic   # classic rules-only profile
./scripts/start_training.sh neural    # ML/neural-blend profile
```
Each label maps to `logs/training_daemon_<label>.log` and `logs/training_daemon_<label>.pid`. The status/stop scripts accept the same label argument.

Environment overrides (either in `.env` or inline before the command):

| Variable | Default | Notes |
|----------|---------|-------|
| `TRAINING_INTERVAL_MINUTES` | `360` | Average spacing between runs |
| `TRAINING_JITTER_MINUTES` | `10` | ± jitter to avoid cron collisions |
| `TRAINING_SYMBOL` | `BTC-USD` | Ticker the daemon downloads/backtests |
| `TRAINING_BAR_INTERVAL` | `5m` | yfinance interval |
| `TRAINING_PERIOD` | `60d` | Data window |
| `TRAINING_LABEL` | _(blank)_ | Optional tag applied to logs/webhooks. Also accepted as first CLI arg to the helper scripts. |
| `TRAINING_MODE` | `auto` | `classic`, `neural`, or `auto`. Defaults to the label when obvious. |

Check status / tail logs:
```bash
./scripts/status_training.sh [label]
```
Stop the daemon:
```bash
./scripts/stop_training.sh [label]
```

---

## 4. Keep It Running Hands-Off

### Option A – Cron watchdog (lightweight)
Add to `crontab -e` (adjust paths as needed):
```cron
@reboot /bin/bash -lc 'cd /home/clawdbot/.openclaw/workspace/trading_bot && ./scripts/start_training.sh >> logs/training_watchdog.log 2>&1'
*/30 * * * * /bin/bash -lc 'cd /home/clawdbot/.openclaw/workspace/trading_bot && ./scripts/start_training.sh >> logs/training_watchdog.log 2>&1'
```
`start_training.sh` is idempotent—it exits early if the loop is already alive. The cron job simply restarts it when the PID disappears (e.g., crash, reboot).

### Option B – systemd service (persistent)
Create `/etc/systemd/system/training-daemon.service`:
```ini
[Unit]
Description=Trading Bot Training Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=clawdbot
WorkingDirectory=/home/clawdbot/.openclaw/workspace/trading_bot
Environment=PYTHONUNBUFFERED=1
ExecStart=/bin/bash -lc './scripts/start_training.sh'
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now training-daemon.service
```

Both approaches keep the daemon alive without manual babysitting.

---

## 5. Failure Signals & When to Step In

The daemon already emits several warnings—here’s where to look:

1. **Discord alerts** (`send_training_update`)
   - First, third, sixth consecutive failures (and every tenth after) trigger a ⚠️ message
   - Sample alert: `⚠️ training daemon warning x3: RuntimeError: download failed...`

2. **Breadcrumbs** (`logs/progress.jsonl` / `logs/progress_latest.json`)
   - `kind` field highlights `training_failure`, `training_alert`, `training_sleep`, etc.
   - Quick glance: `cat logs/progress_latest.json`

3. **Daemon log** (`logs/training_daemon.log`)
   - Full stack traces plus sleep/backoff timing

Act when:
- `status_training.sh` reports `NOT_RUNNING`
- `logs/progress_latest.json` hasn’t moved in longer than your expected cadence (default: >6h)
- Discord emits repeated warnings (>=3 consecutive failures)

To recover:
```bash
./scripts/stop_training.sh  # optional if PID file still present
./scripts/start_training.sh
```
Fix underlying issue (network, API credentials, disk, etc.) before restarting if failures persist.

---

## 6. Optional: Heartbeat/monitoring hooks

- Add the daemon checks to your existing supervisor dashboards by reading `logs/progress_latest.json`.
- `services/progress.record_state()` can mirror any extra metadata you want under `logs/progress_state.json`—populate it from `train_backtest.py` if you decide to surface additional metrics (open TODO in Project ledger).

---

## 7. Quick Reference

| Task | Command |
|------|---------|
| Start daemon | `./scripts/start_training.sh` |
| Status + tail | `./scripts/status_training.sh` |
| Stop daemon | `./scripts/stop_training.sh` |
| Manual training | `python scripts/train_backtest.py` |
| Tail breadcrumb log | `tail -f logs/progress.jsonl` |
| Latest breadcrumb | `cat logs/progress_latest.json` |

---

With this in place the training loop runs itself, escalates when it’s unhappy, and you only need to step in when the alerts fire.
