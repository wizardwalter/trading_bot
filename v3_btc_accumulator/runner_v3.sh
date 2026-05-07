#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
while true; do
  python v3_btc_accumulator/train_v3.py >> /tmp/v3_btc_accumulator.log 2>&1 || true
  sleep 600
done
