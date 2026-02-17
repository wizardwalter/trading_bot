import os
import requests

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


def send_trade_alert(ticker, action, price, quantity, confidence, reason, paper=True):
    if not DISCORD_WEBHOOK_URL:
        return

    trade_type = "📄 Paper Trade" if paper else "💸 LIVE TRADE"
    icon = "🟢" if action == "buy" else "🔴"

    data = {
        "content": (
            f"{trade_type} {icon} **{action.upper()} {ticker}** | "
            f"qty={quantity} @ **${price:.2f}** | conf={confidence:.2f}\n"
            f"`{reason}`"
        )
    }

    try:
        requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=10)
    except Exception:
        pass
