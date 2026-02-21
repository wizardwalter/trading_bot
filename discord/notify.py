import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Ensure .env is loaded even in isolated cron runs.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TRAINING_WEBHOOK_URL = os.getenv("TRAINING_WEBHOOK_URL")
TRAINING_LABEL = (os.getenv("TRAINING_LABEL") or os.getenv("TRAINING_VARIANT") or "").strip()


def _format_label(label: str | None) -> str:
    label = (label or "").strip()
    if not label:
        return ""
    return label.upper()


def _send_to(url: str | None, content: str):
    if not url:
        return False

    for attempt in range(3):
        try:
            r = requests.post(url, json={"content": content}, timeout=10)
            if 200 <= r.status_code < 300:
                return True
            # Retry on rate limits/server errors.
            if r.status_code not in (429, 500, 502, 503, 504):
                return False
        except Exception:
            pass

        if attempt < 2:
            time.sleep(0.6 * (attempt + 1))
    return False


def _send(content: str):
    _send_to(DISCORD_WEBHOOK_URL, content)


def send_trade_alert(ticker, action, price, quantity, confidence, reason, paper=True):
    trade_type = "📄 Paper Trade" if paper else "💸 LIVE TRADE"
    icon = "🟢" if action == "buy" else "🔴"
    _send(
        f"{trade_type} {icon} **{action.upper()} {ticker}** | "
        f"qty={quantity} @ **${price:.2f}** | conf={confidence:.2f}\n"
        f"`{reason}`"
    )


def send_status_update(message: str):
    _send(f"📊 {message}")


def send_training_update(message: str, label: str | None = None) -> bool:
    prefix = _format_label(label if label is not None else TRAINING_LABEL)
    if prefix:
        message = f"[{prefix}] {message}"
    # Prefer dedicated training channel; fallback to default webhook if unset.
    return _send_to(TRAINING_WEBHOOK_URL or DISCORD_WEBHOOK_URL, f"🧠 {message}")
