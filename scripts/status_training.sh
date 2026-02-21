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
  LOG_FILE="logs/training_daemon.log"
  LABEL_DISPLAY="default"
else
  PID_FILE="logs/training_daemon_${LABEL_SLUG}.pid"
  LOG_FILE="logs/training_daemon_${LABEL_SLUG}.log"
  LABEL_DISPLAY="$LABEL_VALUE"
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "RUNNING (label=${LABEL_DISPLAY}) pid $(cat "$PID_FILE")"
  tail -n 20 "$LOG_FILE" || true
else
  echo "NOT_RUNNING (label=${LABEL_DISPLAY})"
  tail -n 40 "$LOG_FILE" || true
fi
