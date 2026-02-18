import os
import requests

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def _send(content: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=10)
    except Exception:
        pass


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
