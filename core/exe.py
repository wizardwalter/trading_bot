from __future__ import annotations

import os
from datetime import datetime
from typing import Dict

from core.market_hours import is_trade_window_open
from core.risk import MAX_RISK_PER_TRADE, drawdown_exceeded, exceeds_portfolio_exposure
from core.strategy import position_size, should_enter_trade
from data.database import get_all_tickers, log_trade
from discord.notify import send_trade_alert
from services.alpaca_broker import AlpacaBroker


MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.12"))


def run_bot(paper_mode: bool = True, execute_orders: bool = False):
    print(
        f"[{datetime.utcnow().isoformat()}] 🚀 Trading bot started | paper_mode={paper_mode} | execute_orders={execute_orders}"
    )

    broker = AlpacaBroker()
    account = broker.get_account()
    start_equity = float(account.get("equity", 0.0))

    tickers = get_all_tickers()
    if not tickers:
        print("No tickers configured. Seed tickers first.")
        return

    latest_prices: Dict[str, float] = {}

    for ticker in tickers:
        try:
            if not is_trade_window_open(ticker):
                print(f"[{datetime.utcnow().isoformat()}] 🕒 Market closed for {ticker}, skipping")
                continue

            account_now = broker.get_account()
            current_equity = float(account_now.get("equity", 0.0))
            if drawdown_exceeded(start_equity, current_equity):
                print("🛑 Daily drawdown limit reached. Halting trading loop.")
                break

            print(f"[{datetime.utcnow().isoformat()}] 🔍 Analyzing {ticker}...")
            decision = should_enter_trade(ticker)
            latest_prices[ticker] = decision["price"]

            action = decision["action"]
            if action == "hold":
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {decision['reason']}")
                continue

            qty = position_size(
                equity=current_equity,
                price=decision["price"],
                volatility=decision["volatility"],
                max_risk_per_trade=MAX_RISK_PER_TRADE,
            )

            current_qty = broker.get_position_qty(ticker)
            if action == "sell" and current_qty <= 0:
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip SELL {ticker} (no position)")
                continue

            if decision["confidence"] < MIN_SIGNAL_CONFIDENCE and action == "buy":
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                    f"(low confidence={decision['confidence']:.2f} < {MIN_SIGNAL_CONFIDENCE:.2f})"
                )
                continue

            if action == "sell":
                qty = min(qty, int(current_qty))

            if action == "buy":
                positions = broker.get_positions()
                current_exposure = sum(max(float(p.get("market_value", 0.0)), 0.0) for p in positions)
                trade_notional = float(decision["price"]) * max(int(qty), 0)
                if exceeds_portfolio_exposure(
                    current_exposure=current_exposure,
                    trade_notional=trade_notional,
                    equity=current_equity,
                ):
                    print(
                        f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                        f"(exposure cap) | projected={(current_exposure + trade_notional):.2f} equity={current_equity:.2f}"
                    )
                    continue

            if execute_orders:
                order = broker.submit_market_order(symbol=ticker, side=action, qty=qty)
                order_id = order.get("id", "n/a")
                reason = f"{decision['reason']} | order_id={order_id}"
            else:
                reason = f"{decision['reason']} | dry_run=true"

            print(
                f"[{datetime.utcnow().isoformat()}] ✅ {action.upper()} {ticker} qty={qty} @ {decision['price']:.2f} "
                f"| conf={decision['confidence']:.2f}"
            )

            log_trade(
                ticker=ticker,
                action=action,
                price=decision["price"],
                quantity=qty,
                signal_strength=decision["confidence"],
                reason=reason,
            )

            send_trade_alert(
                ticker=ticker,
                action=action,
                price=decision["price"],
                quantity=qty,
                confidence=decision["confidence"],
                reason=reason,
                paper=paper_mode,
            )
        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] ⚠️ Error processing {ticker}: {e}")
            continue

    print(f"[{datetime.utcnow().isoformat()}] ✅ Trading session complete")
