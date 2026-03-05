#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/clawdbot/.openclaw/workspace/trading_bot"
BOT_MEMORY_ROOT="/home/clawdbot/.openclaw/workspace/bot-memory"
cd "$ROOT"

set -a
source "$ROOT/.env"
set +a

BACKUP_REMOTE="${BACKUP_REMOTE:-gdrive:clawbot-backups}"
HEALTH_WEBHOOK_URL="${HEALTH_WEBHOOK_URL:-}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
Y="$(date -u +%Y)"
M="$(date -u +%m)"
D="$(date -u +%d)"
REMOTE_DIR="$BACKUP_REMOTE/daily/$Y/$M/$D/backup-$STAMP"

TMP_DIR="$ROOT/data/backups/$STAMP"
mkdir -p "$TMP_DIR"

DB_FILE="$TMP_DIR/db_trading_bot.sql.gz"
TB_REPO_BUNDLE="$TMP_DIR/trading_bot.bundle"
BM_REPO_BUNDLE="$TMP_DIR/bot_memory.bundle"
MANIFEST="$TMP_DIR/manifest.txt"

# 1) Core trading data (DB) backup
PGPASSWORD="$DB_PASS" pg_dump -h "${DB_HOST:-localhost}" -U "$DB_USER" -d "$DB_NAME" | gzip > "$DB_FILE"

# 2) Repo backup snapshots (portable git bundles)
git -C "$ROOT" bundle create "$TB_REPO_BUNDLE" --all
if [[ -d "$BOT_MEMORY_ROOT/.git" ]]; then
  git -C "$BOT_MEMORY_ROOT" bundle create "$BM_REPO_BUNDLE" --all
fi

{
  echo "timestamp_utc=$STAMP"
  echo "remote_dir=$REMOTE_DIR"
  echo "db_name=$DB_NAME"
  echo "db_host=${DB_HOST:-localhost}"
  echo "files="
  ls -lh "$TMP_DIR"
} > "$MANIFEST"

rclone mkdir "$REMOTE_DIR"
rclone copy "$TMP_DIR" "$REMOTE_DIR" --checksum

if [[ -n "$HEALTH_WEBHOOK_URL" ]]; then
  curl -s -H 'Content-Type: application/json' \
    -d "{\"content\":\"🩺 Backup complete: backup-$STAMP uploaded to $REMOTE_DIR\"}" \
    "$HEALTH_WEBHOOK_URL" >/dev/null || true
fi

echo "Backup complete: backup-$STAMP"
