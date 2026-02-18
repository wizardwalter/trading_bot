#!/usr/bin/env bash
set -euo pipefail
cd /home/clawdbot/.openclaw/workspace/trading_bot
if [ -f logs/autonomous.pid ]; then
  PID="$(cat logs/autonomous.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" || true
    fi
    echo "stopped $PID"
  else
    echo "pid file exists but process not running"
  fi
  rm -f logs/autonomous.pid
else
  echo "no pid file"
fi
