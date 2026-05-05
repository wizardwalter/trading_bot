from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.scalper_engine import DailyRiskState, ManagedTrade


STATE_PATH = Path("data/state/scalper_state.json")
_LOCK = threading.Lock()


def _default_state() -> dict[str, Any]:
    return {"symbols": {}}


def load_state(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return _default_state()
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload.setdefault("symbols", {})
            return payload
    except Exception:
        pass
    return _default_state()


def save_state(payload: dict[str, Any], path: Path = STATE_PATH) -> None:
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def get_symbol_state(symbol: str, path: Path = STATE_PATH) -> dict[str, Any]:
    payload = load_state(path)
    return dict(payload.get("symbols", {}).get(symbol, {}))


def put_symbol_state(symbol: str, symbol_state: dict[str, Any], path: Path = STATE_PATH) -> None:
    payload = load_state(path)
    payload.setdefault("symbols", {})
    payload["symbols"][symbol] = symbol_state
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_state(payload, path)


def remove_symbol_state(symbol: str, path: Path = STATE_PATH) -> None:
    payload = load_state(path)
    symbols = payload.setdefault("symbols", {})
    if symbol in symbols:
        del symbols[symbol]
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_state(payload, path)


def serialize_trade(trade: ManagedTrade | None) -> dict[str, Any] | None:
    if trade is None:
        return None
    return asdict(trade)


def deserialize_trade(payload: dict[str, Any] | None) -> ManagedTrade | None:
    if not isinstance(payload, dict):
        return None
    try:
        return ManagedTrade(**payload)
    except Exception:
        return None


def serialize_daily_state(daily: DailyRiskState | None) -> dict[str, Any] | None:
    if daily is None:
        return None
    return asdict(daily)


def deserialize_daily_state(payload: dict[str, Any] | None) -> DailyRiskState | None:
    if not isinstance(payload, dict):
        return None
    try:
        return DailyRiskState(**payload)
    except Exception:
        return None
