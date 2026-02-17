from __future__ import annotations

from datetime import datetime
from typing import Dict

from core.strategy import should_enter_trade, position_size
from discord.notify import send_trade_alert
from data.database import (
    get_all_tickers,
    get_position_qty,
    get_portfolio_equity,
    log_trade,
)


def run_bot(paper_mode: bool = True):
    print(f"[{datetime.utcnow().isoformat()}] 🚀 Trading bot started | paper_mode={paper_mode}")

    tickers = get_all_tickers()
    if not tickers:
        print("No tickers configured. Seed tickers first.")
        return

    latest_prices: Dict[str, float] = {}

    for ticker in tickers:
        print(f"[{datetime.utcnow().isoformat()}] 🔍 Analyzing {ticker}...")
        decision = should_enter_trade(ticker)
        latest_prices[ticker] = decision["price"]

        action = decision["action"]
        if action == "hold":
            print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {decision['reason']}")
            continue

        equity = get_portfolio_equity(latest_prices)
        qty = position_size(
            equity=equity,
            price=decision["price"],
            volatility=decision["volatility"],
            max_risk_per_trade=0.01,
        )

        current_qty = get_position_qty(ticker)
        if action == "sell" and current_qty <= 0:
            print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip SELL {ticker} (no long position)")
            continue

        if action == "sell":
            qty = min(qty, current_qty)

        print(
            f"[{datetime.utcnow().isoformat()}] ✅ {action.upper()} {ticker} qty={qty} @ {decision['price']:.2f} "
            f"| conf={decision['confidence']:.2f}"
        )

        if paper_mode:
            log_trade(
                ticker=ticker,
                action=action,
                price=decision["price"],
                quantity=qty,
                signal_strength=decision["confidence"],
                reason=decision["reason"],
            )

        send_trade_alert(
            ticker=ticker,
            action=action,
            price=decision["price"],
            quantity=qty,
            confidence=decision["confidence"],
            reason=decision["reason"],
            paper=paper_mode,
        )

    final_equity = get_portfolio_equity(latest_prices)
    print(f"[{datetime.utcnow().isoformat()}] ✅ Trading session complete | est_equity=${final_equity:.2f}")
