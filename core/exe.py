from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict

from core.market_hours import is_trade_window_open
from core.risk import (
    MAX_PORTFOLIO_EXPOSURE,
    MAX_RISK_PER_TRADE,
    POSITION_FRACTION,
    drawdown_exceeded,
    exceeds_portfolio_exposure,
)
from core.strategy import position_size, should_enter_trade
from data.database import get_all_tickers, log_trade
from discord.notify import send_trade_alert
from services.alpaca_broker import AlpacaBroker


MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.12"))
MAX_SYMBOL_EXPOSURE = float(os.getenv("MAX_SYMBOL_EXPOSURE", "0.18"))
MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "25"))
MIN_CASH_BUFFER = float(os.getenv("MIN_CASH_BUFFER", "250"))
CRYPTO_QTY_PRECISION = int(os.getenv("CRYPTO_QTY_PRECISION", "6"))
ALLOW_PYRAMIDING = os.getenv("ALLOW_PYRAMIDING", "0") == "1"
ACCOUNT_REFRESH_SECONDS = float(os.getenv("ACCOUNT_REFRESH_SECONDS", "45"))
MAX_STALE_ACCOUNT_SECONDS = float(os.getenv("MAX_STALE_ACCOUNT_SECONDS", "180"))


def _is_crypto_symbol(symbol: str) -> bool:
    return "-" in (symbol or "")


def _normalize_qty(symbol: str, qty: float) -> float:
    q = max(float(qty), 0.0)
    if _is_crypto_symbol(symbol):
        return round(q, CRYPTO_QTY_PRECISION)
    return float(int(q))


def run_bot(paper_mode: bool = True, execute_orders: bool = False):
    print(
        f"[{datetime.utcnow().isoformat()}] 🚀 Trading bot started | paper_mode={paper_mode} | execute_orders={execute_orders}"
    )

    try:
        broker = AlpacaBroker()
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] ⚠️ Broker init failed: {e}")
        return

    try:
        account = broker.get_account()
    except Exception as e:
        print(f"[{datetime.utcnow().isoformat()}] ⚠️ Unable to fetch account snapshot: {e}")
        return

    if account.get("_stale"):
        age = account.get("_cache_age_seconds")
        age_desc = f"{age:.0f}s" if isinstance(age, (int, float)) else "unknown"
        warning = account.get("_cache_warning")
        warning_suffix = f" | {warning}" if warning else ""
        print(
            f"[{datetime.utcnow().isoformat()}] ⚠️ Using cached Alpaca account snapshot age={age_desc}{warning_suffix}"
        )

    start_equity = float(account.get("equity", 0.0))
    last_account_snapshot = account
    if account.get("_stale"):
        last_account_refresh_ts = time.time() - ACCOUNT_REFRESH_SECONDS - 1
    else:
        last_account_refresh_ts = time.time()

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

            now_ts = time.time()
            need_refresh = (now_ts - last_account_refresh_ts) >= ACCOUNT_REFRESH_SECONDS

            if need_refresh:
                try:
                    account_candidate = broker.get_account()
                    if account_candidate.get("_stale"):
                        age = account_candidate.get("_cache_age_seconds")
                        age_desc = f"{age:.0f}s" if isinstance(age, (int, float)) else "unknown"
                        warning = account_candidate.get("_cache_warning")
                        warning_suffix = f" | {warning}" if warning else ""
                        print(
                            f"[{datetime.utcnow().isoformat()}] ⚠️ Using cached Alpaca account snapshot age={age_desc} for {ticker}{warning_suffix}"
                        )
                        last_account_refresh_ts = now_ts - (ACCOUNT_REFRESH_SECONDS * 0.5)
                    else:
                        last_account_refresh_ts = now_ts
                    last_account_snapshot = account_candidate
                    account_now = account_candidate
                except Exception as e:
                    account_now = last_account_snapshot
                    print(
                        f"[{datetime.utcnow().isoformat()}] ⚠️ Account fetch failed for {ticker}; "
                        f"using stale snapshot: {e}"
                    )
            else:
                account_now = last_account_snapshot

            current_equity = float(account_now.get("equity", 0.0))
            if drawdown_exceeded(start_equity, current_equity):
                print("🛑 Daily drawdown limit reached. Halting trading loop.")
                break

            print(f"[{datetime.utcnow().isoformat()}] 🔍 Analyzing {ticker}...")
            current_qty = broker.get_position_qty(ticker)
            decision = should_enter_trade(ticker, has_position=(current_qty > 0))
            latest_prices[ticker] = decision["price"]

            action = decision["action"]
            if action == "hold":
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {decision['reason']}")
                continue

            stale_age_s = time.time() - last_account_refresh_ts
            if action == "buy" and stale_age_s > MAX_STALE_ACCOUNT_SECONDS:
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                    f"(account snapshot stale {stale_age_s:.0f}s > {MAX_STALE_ACCOUNT_SECONDS:.0f}s)"
                )
                continue

            qty_risk = position_size(
                equity=current_equity,
                price=decision["price"],
                volatility=decision["volatility"],
                max_risk_per_trade=MAX_RISK_PER_TRADE,
            )
            buying_power = float(account_now.get("buying_power") or account_now.get("equity") or 0.0)
            qty_bp_raw = (buying_power * POSITION_FRACTION) / max(decision["price"], 0.01)
            qty_bp = _normalize_qty(ticker, qty_bp_raw)
            # Use the tighter of risk sizing and buying-power sizing.
            qty = min(float(qty_risk), float(qty_bp))

            if action == "sell" and current_qty <= 0:
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip SELL {ticker} (no position)")
                continue

            if action == "buy" and current_qty > 0 and not ALLOW_PYRAMIDING:
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                    f"(already in position qty={current_qty}, pyramiding disabled)"
                )
                continue

            if decision["confidence"] < MIN_SIGNAL_CONFIDENCE and action == "buy":
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                    f"(low confidence={decision['confidence']:.2f} < {MIN_SIGNAL_CONFIDENCE:.2f})"
                )
                continue

            if action == "sell":
                # Exit the full position on sell signal; avoids partial dribble exits.
                qty = _normalize_qty(ticker, float(current_qty))

            if action == "buy" and qty <= 0:
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                    f"(size=0 | qty_risk={qty_risk} qty_bp={qty_bp} bp={buying_power:.2f})"
                )
                continue

            if action == "buy":
                try:
                    positions = broker.get_positions()
                except Exception as e:
                    print(f"[{datetime.utcnow().isoformat()}] ⚠️ Positions fetch failed for {ticker}: {e}")
                    continue
                current_exposure = sum(max(float(p.get("market_value", 0.0)), 0.0) for p in positions)

                symbol_position = next((p for p in positions if p.get("symbol", "").upper() == ticker.replace("-", "").upper()), None)
                current_symbol_exposure = max(float(symbol_position.get("market_value", 0.0)), 0.0) if symbol_position else 0.0

                # Compute all notional caps first, then derive qty from bounded notional.
                max_allowed_exposure = max(current_equity * MAX_PORTFOLIO_EXPOSURE, 0.0)
                remaining_exposure = max(max_allowed_exposure - current_exposure, 0.0)

                max_symbol_exposure = max(current_equity * MAX_SYMBOL_EXPOSURE, 0.0)
                remaining_symbol_exposure = max(max_symbol_exposure - current_symbol_exposure, 0.0)

                max_bp_notional = max(buying_power - MIN_CASH_BUFFER, 0.0) * POSITION_FRACTION
                risk_notional = float(qty) * float(decision["price"])

                allowed_notional = min(
                    remaining_exposure,
                    remaining_symbol_exposure,
                    max_bp_notional,
                    risk_notional,
                )

                qty = _normalize_qty(
                    ticker,
                    allowed_notional / max(float(decision["price"]), 0.01),
                )

                if qty <= 0:
                    print(
                        f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                        f"(cap/no room) | portfolio={current_exposure:.2f}/{max_allowed_exposure:.2f} "
                        f"symbol={current_symbol_exposure:.2f}/{max_symbol_exposure:.2f} "
                        f"bp_room={max_bp_notional:.2f}"
                    )
                    continue

                trade_notional = float(decision["price"]) * max(float(qty), 0.0)
                if trade_notional < MIN_ORDER_NOTIONAL:
                    print(
                        f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                        f"(min notional ${MIN_ORDER_NOTIONAL:.2f})"
                    )
                    continue

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
                reason = (
                    f"{decision['reason']} | order_id={order_id} "
                    f"| qty_risk={qty_risk} qty_bp={qty_bp} bp={buying_power:.2f}"
                )
            else:
                reason = (
                    f"{decision['reason']} | dry_run=true "
                    f"| qty_risk={qty_risk} qty_bp={qty_bp} bp={buying_power:.2f}"
                )

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
