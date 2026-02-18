#!/usr/bin/env bash
set -euo pipefail
cd /home/clawdbot/.openclaw/workspace/trading_bot

PYTHON_BIN=""
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No python runtime found" >&2
  exit 1
fi

# Prevent duplicate runners
if [ -f logs/autonomous.pid ] && kill -0 "$(cat logs/autonomous.pid)" 2>/dev/null; then
  echo "autonomous runner already active (pid $(cat logs/autonomous.pid))"
  exit 0
fi

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u cli/main.py --loop --interval 60 --status-every 5 --execute >> logs/autonomous.log 2>&1 &
echo $! > logs/autonomous.pid
sleep 1
if kill -0 "$(cat logs/autonomous.pid)" 2>/dev/null; then
  echo "started autonomous runner pid $(cat logs/autonomous.pid)"
else
  echo "failed to start autonomous runner" >&2
  exit 1
fi
