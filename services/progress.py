from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# Shared progress ledger so we can recover context after crashes.
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_LOG = LOG_DIR / "progress.jsonl"
LATEST_EVENT = LOG_DIR / "progress_latest.json"

_LOCK = threading.Lock()


def record_event(kind: str, message: str | None = None, data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Persist a structured progress event for later review.

    Events are appended to ``logs/progress.jsonl`` and the most recent event is
    mirrored in ``logs/progress_latest.json`` so supervisors (or a fresh model)
    can quickly see what the system was doing before an interruption.
    """

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "kind": kind,
        "message": message,
        "data": dict(data or {}),
    }

    text = json.dumps(payload, ensure_ascii=False)

    with _LOCK:
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(text + "\n")
        with LATEST_EVENT.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    return payload


def record_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror an arbitrary state snapshot next to the progress log."""

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "state": dict(state),
    }

    with _LOCK:
        path = LOG_DIR / "progress_state.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    return payload
