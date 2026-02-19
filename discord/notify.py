import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Ensure .env is loaded even in isolated cron runs.
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
TRAINING_WEBHOOK_URL = os.getenv("TRAINING_WEBHOOK_URL")


def _send_to(url: str | None, content: str):
    if not url:
        return
    try:
        requests.post(url, json={"content": content}, timeout=10)
    except Exception:
        pass


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


def send_training_update(message: str):
    # Prefer dedicated training channel; fallback to default webhook if unset.
    _send_to(TRAINING_WEBHOOK_URL or DISCORD_WEBHOOK_URL, f"🧠 {message}")
