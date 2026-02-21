#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/clawdbot/.openclaw/workspace/trading_bot"
cd "$ROOT"

slugify() {
  local value="$1"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  value="$(printf '%s' "$value" | tr -cs 'a-z0-9' '-')"
  value="$(printf '%s' "$value" | sed 's/^-//; s/-$//; s/-\{2,\}/-/g')"
  printf '%s' "$value"
}

LABEL_ARG="${1:-}"
LABEL_VALUE="${LABEL_ARG:-${TRAINING_LABEL:-${TRAINING_VARIANT:-}}}"
LABEL_VALUE="$(printf '%s' "$LABEL_VALUE" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"

LABEL_SLUG=""
if [ -n "$LABEL_VALUE" ]; then
  LABEL_SLUG="$(slugify "$LABEL_VALUE")"
fi
if [ -z "$LABEL_SLUG" ]; then
  PID_FILE="logs/training_daemon.pid"
  LABEL_DISPLAY="default"
else
  PID_FILE="logs/training_daemon_${LABEL_SLUG}.pid"
  LABEL_DISPLAY="$LABEL_VALUE"
fi

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      kill -9 "$PID" || true
    fi
    echo "stopped $PID (label=${LABEL_DISPLAY})"
  else
    echo "pid file exists but process not running (label=${LABEL_DISPLAY})"
  fi
  rm -f "$PID_FILE"
else
  echo "no pid file for label=${LABEL_DISPLAY}"
fi
