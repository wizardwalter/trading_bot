import requests
import os

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_trade_alert(ticker, price, paper):
    if not DISCORD_WEBHOOK_URL:
        return

    trade_type = "📄 Paper Trade" if paper else "💸 LIVE TRADE"
    data = {
        "content": f"{trade_type} triggered on **{ticker}** at **${price:.2f}**!"
    }
    requests.post(DISCORD_WEBHOOK_URL, json=data)