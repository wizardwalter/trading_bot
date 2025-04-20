from datetime import datetime
from core.strategy import should_enter_trade
from discord.notify import send_trade_alert
from data.database import log_trade, get_all_tickers

def run_bot(paper_mode=True):
    print(f"[{datetime.now()}] 🚀 Trading bot started... Paper mode: {paper_mode}")

    tickers = get_all_tickers()

    for ticker in tickers:
        print(f"[{datetime.now()}] 🔍 Analyzing {ticker}...")
        decision = should_enter_trade(ticker)

        if decision["enter"]:
            print(f"[{datetime.now()}] ✅ Entering trade on {ticker} at {decision['price']}")

            if paper_mode:
                log_trade(ticker, decision['price'], paper=True)
            else:
                # insert real trading API logic here
                pass

            send_trade_alert(ticker, decision['price'], paper_mode)

        else:
            print(f"[{datetime.now()}] ⏭️ No trade on {ticker}")

    print(f"[{datetime.now()}] ✅ Trading session complete.")