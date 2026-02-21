#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/clawdbot/.openclaw/workspace/trading_bot"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

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
else
  PID_FILE="logs/training_daemon_${LABEL_SLUG}.pid"
  LOG_FILE="logs/training_daemon_${LABEL_SLUG}.log"
fi

mkdir -p logs

PYTHON_BIN=""
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "No python runtime found" >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "training daemon already active for label '${LABEL_VALUE:-default}' (pid $(cat "$PID_FILE"))"
  exit 0
fi

INTERVAL_MINUTES="${TRAINING_INTERVAL_MINUTES:-360}"
JITTER_MINUTES="${TRAINING_JITTER_MINUTES:-10}"
SYMBOL="${TRAINING_SYMBOL:-BTC-USD}"
BAR_INTERVAL="${TRAINING_BAR_INTERVAL:-5m}"
PERIOD="${TRAINING_PERIOD:-60d}"

MODE_ENV="${TRAINING_MODE:-}"
if [ -z "$MODE_ENV" ] && [ -n "$LABEL_VALUE" ]; then
  case "$(printf '%s' "$LABEL_VALUE" | tr '[:upper:]' '[:lower:]')" in
    classic)
      MODE_ENV="classic"
      ;;
    neural)
      MODE_ENV="neural"
      ;;
  esac
fi

CMD_ENV=(env PYTHONUNBUFFERED=1)
if [ -n "$LABEL_VALUE" ]; then
  CMD_ENV+=(TRAINING_LABEL="$LABEL_VALUE")
fi
if [ -n "$MODE_ENV" ]; then
  CMD_ENV+=(TRAINING_MODE="$MODE_ENV")
fi

nohup "${CMD_ENV[@]}" "$PYTHON_BIN" -m services.training_daemon \
  --interval-minutes "$INTERVAL_MINUTES" \
  --jitter-minutes "$JITTER_MINUTES" \
  --symbol "$SYMBOL" \
  --bar-interval "$BAR_INTERVAL" \
  --period "$PERIOD" \
  >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
sleep 1
if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "started training daemon pid $(cat "$PID_FILE") (label=${LABEL_VALUE:-default})"
  echo "log: $LOG_FILE"
else
  echo "failed to start training daemon" >&2
  exit 1
fi
