#!/usr/bin/env bash
set -euo pipefail
cd /home/clawdbot/.openclaw/workspace/trading_bot
if [ -f logs/autonomous.pid ] && kill -0 "$(cat logs/autonomous.pid)" 2>/dev/null; then
  echo "RUNNING pid $(cat logs/autonomous.pid)"
  tail -n 20 logs/autonomous.log || true
else
  echo "NOT_RUNNING"
  tail -n 40 logs/autonomous.log || true
fi
