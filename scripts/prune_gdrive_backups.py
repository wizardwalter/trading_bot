#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timedelta, timezone

ROOT = "/home/clawdbot/.openclaw/workspace/trading_bot"
ENV_PATH = f"{ROOT}/.env"

if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)

remote = os.getenv("BACKUP_REMOTE", "gdrive:clawbot-backups")
base = f"{remote}/daily"
min_keep = int(os.getenv("BACKUP_MIN_KEEP", "10"))
retention_days = int(os.getenv("BACKUP_RETENTION_DAYS", "60"))
cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

ls = subprocess.run(["rclone", "lsf", base, "--dirs-only", "--recursive"], capture_output=True, text=True)
if ls.returncode != 0:
    raise SystemExit(ls.stderr.strip() or "rclone lsf failed")

pat = re.compile(r"backup-(\d{8}T\d{6}Z)/$")
entries: list[tuple[datetime, str]] = []
for rel in ls.stdout.splitlines():
    rel = rel.strip()
    if not rel:
        continue
    m = pat.search(rel)
    if not m:
        continue
    ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    entries.append((ts, f"{base}/{rel.rstrip('/') }"))

entries.sort(key=lambda x: x[0])
count = len(entries)
if count <= min_keep:
    print(f"Skip prune: count={count} <= min_keep={min_keep}")
    raise SystemExit(0)

old = [(ts, path) for ts, path in entries if ts < cutoff]
max_delete = max(0, count - min_keep)
old = old[:max_delete]

for _, path in old:
    subprocess.run(["rclone", "purge", path], check=False)
    print(f"Deleted {path}")

print(f"Prune done: deleted={len(old)} count_before={count} min_keep={min_keep}")
