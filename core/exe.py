from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from core.market_hours import is_trade_window_open
from core.risk import (
    MAX_PORTFOLIO_EXPOSURE,
    MAX_RISK_PER_TRADE,
    POSITION_FRACTION,
    drawdown_exceeded,
    exceeds_portfolio_exposure,
)
from core.scalper_engine import (
    ManagedTrade,
    apply_exit_fill,
    can_open_new_trade,
    cooldown_for_exit_reason,
    DailyRiskState,
    finalize_trade_return,
    load_btc_scalper_config,
    manage_open_trade,
    roll_daily_state,
    update_daily_after_closed_trade,
)
from core.scalper_state import (
    deserialize_daily_state,
    deserialize_trade,
    put_symbol_state,
    serialize_daily_state,
    serialize_trade,
)
from core.strategy import build_market_snapshot, position_size
from data.database import get_all_tickers, log_trade
from discord.notify import send_trade_alert
from services.alpaca_broker import AlpacaBroker
from services.progress import record_event, record_state


MIN_SIGNAL_CONFIDENCE = float(os.getenv("MIN_SIGNAL_CONFIDENCE", "0.12"))
MAX_SYMBOL_EXPOSURE = float(os.getenv("MAX_SYMBOL_EXPOSURE", "0.18"))
MIN_ORDER_NOTIONAL = float(os.getenv("MIN_ORDER_NOTIONAL", "25"))
MIN_CASH_BUFFER = float(os.getenv("MIN_CASH_BUFFER", "250"))
CRYPTO_QTY_PRECISION = int(os.getenv("CRYPTO_QTY_PRECISION", "6"))
ACCOUNT_REFRESH_SECONDS = float(os.getenv("ACCOUNT_REFRESH_SECONDS", "45"))
MAX_STALE_ACCOUNT_SECONDS = float(os.getenv("MAX_STALE_ACCOUNT_SECONDS", "180"))
PRIMARY_SCALPER_SYMBOL = os.getenv("PRIMARY_SCALPER_SYMBOL", "BTC-USD").upper()
ALLOW_NON_PRIMARY_SYMBOLS = os.getenv("ALLOW_NON_PRIMARY_SYMBOLS", "0") == "1"
LIVE_ROUND_TRIP_FRICTION_PCT = float(os.getenv("LIVE_ROUND_TRIP_FRICTION_PCT", "0.0075"))
LIVE_SIDE_FRICTION_PCT = LIVE_ROUND_TRIP_FRICTION_PCT / 2.0


def _is_crypto_symbol(symbol: str) -> bool:
    return "-" in (symbol or "")


def _normalize_qty(symbol: str, qty: float) -> float:
    q = max(float(qty), 0.0)
    if _is_crypto_symbol(symbol):
        return round(q, CRYPTO_QTY_PRECISION)
    return float(int(q))


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_primary_symbol(symbol: str) -> bool:
    return symbol.upper() == PRIMARY_SCALPER_SYMBOL


def _load_symbol_state(symbol: str) -> tuple[ManagedTrade | None, DailyRiskState | None, datetime | None]:
    from core.scalper_state import get_symbol_state

    payload = get_symbol_state(symbol)
    open_trade = deserialize_trade(payload.get("open_trade"))
    daily = deserialize_daily_state(payload.get("daily"))
    cooldown_until = _parse_dt(payload.get("cooldown_until"))
    return open_trade, daily, cooldown_until


def _persist_symbol_state(
    symbol: str,
    *,
    open_trade: ManagedTrade | None,
    daily: DailyRiskState | None,
    cooldown_until: datetime | None,
    last_reason: str,
) -> None:
    put_symbol_state(
        symbol,
        {
            "open_trade": serialize_trade(open_trade),
            "daily": serialize_daily_state(daily),
            "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
            "last_reason": last_reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _submit_order(
    *,
    broker: AlpacaBroker,
    ticker: str,
    side: str,
    qty: float,
    execute_orders: bool,
) -> tuple[float, str]:
    normalized_qty = _normalize_qty(ticker, qty)
    if normalized_qty <= 0:
        return 0.0, "qty=0"

    if execute_orders:
        order = broker.submit_market_order(symbol=ticker, side=side, qty=normalized_qty)
        return normalized_qty, f"order_id={order.get('id', 'n/a')}"
    return normalized_qty, "dry_run=true"


def run_bot(paper_mode: bool = True, execute_orders: bool = False):
    print(
        f"[{datetime.utcnow().isoformat()}] 🚀 Trading bot started | paper_mode={paper_mode} | execute_orders={execute_orders}"
    )

    try:
        broker = AlpacaBroker()
    except Exception as exc:
        print(f"[{datetime.utcnow().isoformat()}] ⚠️ Broker init failed: {exc}")
        return

    try:
        account = broker.get_account()
    except Exception as exc:
        print(f"[{datetime.utcnow().isoformat()}] ⚠️ Unable to fetch account snapshot: {exc}")
        return

    start_equity = float(account.get("equity", 0.0))
    last_account_snapshot = account
    last_account_refresh_ts = time.time()

    tickers = get_all_tickers()
    if not tickers:
        print("No tickers configured. Seed tickers first.")
        return

    state_snapshot: dict[str, dict] = {}

    for ticker in tickers:
        try:
            if not ALLOW_NON_PRIMARY_SYMBOLS and not _is_primary_symbol(ticker):
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip {ticker} "
                    f"(runtime tuned for {PRIMARY_SCALPER_SYMBOL}; set ALLOW_NON_PRIMARY_SYMBOLS=1 to override)"
                )
                continue

            if not is_trade_window_open(ticker):
                print(f"[{datetime.utcnow().isoformat()}] 🕒 Market closed for {ticker}, skipping")
                continue

            now_ts = time.time()
            need_refresh = (now_ts - last_account_refresh_ts) >= ACCOUNT_REFRESH_SECONDS
            if need_refresh:
                try:
                    last_account_snapshot = broker.get_account()
                    last_account_refresh_ts = now_ts
                except Exception as exc:
                    print(
                        f"[{datetime.utcnow().isoformat()}] ⚠️ Account fetch failed for {ticker}; "
                        f"using stale snapshot: {exc}"
                    )
            account_now = last_account_snapshot

            current_equity = float(account_now.get("equity", 0.0))
            if drawdown_exceeded(start_equity, current_equity):
                print("🛑 Daily drawdown limit reached. Halting trading loop.")
                break

            print(f"[{datetime.utcnow().isoformat()}] 🔍 Evaluating {ticker}...")
            snapshot = build_market_snapshot(ticker)
            market_ts = snapshot.df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            config = load_btc_scalper_config(snapshot.threshold)

            position = broker.get_position(ticker)
            current_qty = float(position.get("qty", 0.0)) if position else 0.0
            avg_entry_price = float(position.get("avg_entry_price", 0.0)) if position else 0.0

            open_trade, daily, cooldown_until = _load_symbol_state(ticker)
            daily = roll_daily_state(daily, market_ts)

            if current_qty <= 0 and open_trade is not None:
                print(f"[{datetime.utcnow().isoformat()}] ℹ️ {ticker} state sync: broker flat, clearing local open trade")
                open_trade = None

            if current_qty > 0 and open_trade is None:
                reconstructed_entry = avg_entry_price if avg_entry_price > 0 else snapshot.price
                open_trade = ManagedTrade.new(
                    symbol=ticker,
                    entry_time=market_ts,
                    entry_price=reconstructed_entry,
                    threshold=snapshot.threshold,
                    setup="reconstructed_broker_position",
                    setup_score=snapshot.entry_signal.setup_score,
                    confidence=snapshot.entry_signal.confidence,
                    qty=current_qty,
                    config=config,
                    entry_cost_pct=LIVE_SIDE_FRICTION_PCT,
                )
                print(
                    f"[{datetime.utcnow().isoformat()}] ℹ️ {ticker} reconstructed open trade from broker "
                    f"qty={current_qty} avg_entry={reconstructed_entry:.2f}"
                )

            if open_trade is not None:
                open_trade.remaining_qty = current_qty
                action = manage_open_trade(snapshot.df, open_trade, config)
                managed_trade = action.updated_trade or open_trade
                managed_trade.remaining_qty = current_qty

                if action.action == "hold":
                    _persist_symbol_state(
                        ticker,
                        open_trade=managed_trade,
                        daily=daily,
                        cooldown_until=cooldown_until,
                        last_reason=action.reason,
                    )
                    state_snapshot[ticker] = {
                        "mode": "manage_open_trade",
                        "action": "hold",
                        "reason": action.reason,
                    }
                    print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {action.reason}")
                    continue

                sell_fraction = 1.0 if action.action == "sell" else min(max(action.fraction, 0.0), 1.0)
                qty_to_sell = current_qty if action.action == "sell" else (current_qty * sell_fraction)
                qty_to_sell = _normalize_qty(ticker, qty_to_sell)

                # If the partial would be too small to execute cleanly, flatten the remainder.
                if action.action == "sell_partial" and (qty_to_sell <= 0 or (qty_to_sell * snapshot.price) < MIN_ORDER_NOTIONAL):
                    action.action = "sell"
                    sell_fraction = 1.0
                    qty_to_sell = _normalize_qty(ticker, current_qty)

                if qty_to_sell <= 0:
                    print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip exit {ticker} (qty=0)")
                    continue

                filled_qty, order_note = _submit_order(
                    broker=broker,
                    ticker=ticker,
                    side="sell",
                    qty=qty_to_sell,
                    execute_orders=execute_orders,
                )
                if filled_qty <= 0:
                    continue

                remaining_qty = max(current_qty - filled_qty, 0.0)
                filled_fraction = 1.0 if current_qty <= 0 else min(max(filled_qty / max(current_qty, 1e-9), 0.0), 1.0)
                managed_trade = apply_exit_fill(
                    managed_trade,
                    price=snapshot.price,
                    fraction=filled_fraction,
                    exit_cost_pct=LIVE_SIDE_FRICTION_PCT,
                    remaining_qty=remaining_qty,
                )

                reason = (
                    f"{action.reason} | qty={filled_qty} | state_setup={managed_trade.setup} | "
                    f"score={snapshot.score:+.3f} | conf={snapshot.entry_signal.confidence:.3f} | {order_note}"
                )
                log_trade(
                    ticker=ticker,
                    action="sell",
                    price=snapshot.price,
                    quantity=filled_qty,
                    signal_strength=snapshot.entry_signal.confidence,
                    reason=reason,
                )
                send_trade_alert(
                    ticker=ticker,
                    action="sell",
                    price=snapshot.price,
                    quantity=filled_qty,
                    confidence=snapshot.entry_signal.confidence,
                    reason=reason,
                    paper=paper_mode,
                )

                if action.action == "sell_partial":
                    managed_trade.remaining_qty = remaining_qty
                    managed_trade.remaining_fraction = max(open_trade.remaining_fraction - filled_fraction, 0.0)
                    record_event(
                        "trade_partial_exit",
                        message=reason,
                        data={"ticker": ticker, "qty": filled_qty, "price": snapshot.price},
                    )
                    _persist_symbol_state(
                        ticker,
                        open_trade=managed_trade,
                        daily=daily,
                        cooldown_until=cooldown_until,
                        last_reason=reason,
                    )
                    state_snapshot[ticker] = {
                        "mode": "manage_open_trade",
                        "action": "sell_partial",
                        "reason": reason,
                    }
                    print(f"[{datetime.utcnow().isoformat()}] ✅ PARTIAL SELL {ticker} qty={filled_qty} @ {snapshot.price:.2f}")
                    continue

                total_trade_return = finalize_trade_return(managed_trade)
                exit_reason = managed_trade.exit_reason or "managed_exit"
                daily = update_daily_after_closed_trade(daily, total_trade_return, exit_reason, config)
                cooldown_until = market_ts + cooldown_for_exit_reason(exit_reason, config)
                record_event(
                    "trade_exit",
                    message=reason,
                    data={
                        "ticker": ticker,
                        "qty": filled_qty,
                        "price": snapshot.price,
                        "exit_reason": exit_reason,
                        "trade_return": total_trade_return,
                    },
                )
                _persist_symbol_state(
                    ticker,
                    open_trade=None,
                    daily=daily,
                    cooldown_until=cooldown_until,
                    last_reason=reason,
                )
                state_snapshot[ticker] = {
                    "mode": "manage_open_trade",
                    "action": "sell",
                    "reason": reason,
                    "trade_return": total_trade_return,
                }
                print(
                    f"[{datetime.utcnow().isoformat()}] ✅ SELL {ticker} qty={filled_qty} @ {snapshot.price:.2f} "
                    f"| ret={total_trade_return:+.2%} | exit={exit_reason}"
                )
                continue

            allowed, deny_reason = can_open_new_trade(daily, cooldown_until, market_ts, config)
            if not allowed:
                reason = f"entry_blocked:{deny_reason}"
                _persist_symbol_state(
                    ticker,
                    open_trade=None,
                    daily=daily,
                    cooldown_until=cooldown_until,
                    last_reason=reason,
                )
                state_snapshot[ticker] = {"mode": "flat", "action": "hold", "reason": reason}
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {reason}")
                continue

            if not snapshot.entry_signal.enter:
                reason = f"no_entry:{snapshot.reason}"
                _persist_symbol_state(
                    ticker,
                    open_trade=None,
                    daily=daily,
                    cooldown_until=cooldown_until,
                    last_reason=reason,
                )
                state_snapshot[ticker] = {"mode": "flat", "action": "hold", "reason": reason}
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {snapshot.reason}")
                continue

            if snapshot.entry_signal.confidence < max(MIN_SIGNAL_CONFIDENCE, config.min_entry_confidence):
                reason = (
                    f"low_confidence={snapshot.entry_signal.confidence:.3f} "
                    f"< {max(MIN_SIGNAL_CONFIDENCE, config.min_entry_confidence):.3f}"
                )
                _persist_symbol_state(
                    ticker,
                    open_trade=None,
                    daily=daily,
                    cooldown_until=cooldown_until,
                    last_reason=reason,
                )
                state_snapshot[ticker] = {"mode": "flat", "action": "hold", "reason": reason}
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ HOLD {ticker} | {reason}")
                continue

            stale_age_s = time.time() - last_account_refresh_ts
            if stale_age_s > MAX_STALE_ACCOUNT_SECONDS:
                reason = f"stale_account_snapshot={stale_age_s:.0f}s"
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} ({reason})")
                continue

            qty_risk = position_size(
                equity=current_equity,
                price=snapshot.price,
                volatility=snapshot.volatility,
                max_risk_per_trade=MAX_RISK_PER_TRADE,
                stop_loss_pct=config.stop_loss_pct,
                friction_buffer_pct=config.breakeven_buffer_pct,
            )
            buying_power = float(account_now.get("buying_power") or account_now.get("equity") or 0.0)
            qty_bp_raw = (buying_power * POSITION_FRACTION) / max(snapshot.price, 0.01)
            qty_bp = _normalize_qty(ticker, qty_bp_raw)
            qty = min(float(qty_risk), float(qty_bp))

            try:
                positions = broker.get_positions()
            except Exception as exc:
                print(f"[{datetime.utcnow().isoformat()}] ⚠️ Positions fetch failed for {ticker}: {exc}")
                continue

            current_exposure = sum(max(float(p.get("market_value", 0.0)), 0.0) for p in positions)
            symbol_position = next(
                (p for p in positions if p.get("symbol", "").upper() == ticker.replace("-", "").upper()),
                None,
            )
            current_symbol_exposure = max(float(symbol_position.get("market_value", 0.0)), 0.0) if symbol_position else 0.0
            max_allowed_exposure = max(current_equity * MAX_PORTFOLIO_EXPOSURE, 0.0)
            remaining_exposure = max(max_allowed_exposure - current_exposure, 0.0)
            max_symbol_exposure = max(current_equity * MAX_SYMBOL_EXPOSURE, 0.0)
            remaining_symbol_exposure = max(max_symbol_exposure - current_symbol_exposure, 0.0)
            max_bp_notional = max(buying_power - MIN_CASH_BUFFER, 0.0) * POSITION_FRACTION
            risk_notional = float(qty) * float(snapshot.price)
            allowed_notional = min(remaining_exposure, remaining_symbol_exposure, max_bp_notional, risk_notional)
            qty = _normalize_qty(ticker, allowed_notional / max(float(snapshot.price), 0.01))

            if qty <= 0:
                print(
                    f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} "
                    f"(size=0 | qty_risk={qty_risk:.6f} qty_bp={qty_bp:.6f} bp={buying_power:.2f})"
                )
                continue

            trade_notional = float(snapshot.price) * qty
            if trade_notional < MIN_ORDER_NOTIONAL:
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} (min notional)")
                continue

            if exceeds_portfolio_exposure(
                current_exposure=current_exposure,
                trade_notional=trade_notional,
                equity=current_equity,
            ):
                print(f"[{datetime.utcnow().isoformat()}] ⏭️ Skip BUY {ticker} (exposure cap)")
                continue

            filled_qty, order_note = _submit_order(
                broker=broker,
                ticker=ticker,
                side="buy",
                qty=qty,
                execute_orders=execute_orders,
            )
            if filled_qty <= 0:
                continue

            daily.trades_opened += 1
            open_trade = ManagedTrade.new(
                symbol=ticker,
                entry_time=market_ts,
                entry_price=snapshot.price,
                threshold=snapshot.threshold,
                setup=snapshot.entry_signal.setup,
                setup_score=snapshot.entry_signal.setup_score,
                confidence=snapshot.entry_signal.confidence,
                qty=filled_qty,
                config=config,
                entry_cost_pct=LIVE_SIDE_FRICTION_PCT,
            )

            reason = (
                f"{snapshot.reason} | setup={snapshot.entry_signal.setup} | qty={filled_qty} | "
                f"risk_qty={qty_risk:.6f} | bp_qty={qty_bp:.6f} | {order_note}"
            )
            record_event(
                "trade_entry",
                message=reason,
                data={
                    "ticker": ticker,
                    "qty": filled_qty,
                    "price": snapshot.price,
                    "setup": snapshot.entry_signal.setup,
                    "threshold": snapshot.threshold,
                },
            )
            log_trade(
                ticker=ticker,
                action="buy",
                price=snapshot.price,
                quantity=filled_qty,
                signal_strength=snapshot.entry_signal.confidence,
                reason=reason,
            )
            send_trade_alert(
                ticker=ticker,
                action="buy",
                price=snapshot.price,
                quantity=filled_qty,
                confidence=snapshot.entry_signal.confidence,
                reason=reason,
                paper=paper_mode,
            )
            _persist_symbol_state(
                ticker,
                open_trade=open_trade,
                daily=daily,
                cooldown_until=cooldown_until,
                last_reason=reason,
            )
            state_snapshot[ticker] = {
                "mode": "opened_trade",
                "action": "buy",
                "reason": reason,
                "setup": snapshot.entry_signal.setup,
            }
            print(
                f"[{datetime.utcnow().isoformat()}] ✅ BUY {ticker} qty={filled_qty} @ {snapshot.price:.2f} "
                f"| setup={snapshot.entry_signal.setup} | conf={snapshot.entry_signal.confidence:.2f}"
            )
        except Exception as exc:
            print(f"[{datetime.utcnow().isoformat()}] ⚠️ Error processing {ticker}: {exc}")
            record_event("trade_loop_error", message=str(exc), data={"ticker": ticker})
            continue

    record_state({"symbols": state_snapshot, "paper_mode": paper_mode, "execute_orders": execute_orders})
    print(f"[{datetime.utcnow().isoformat()}] ✅ Trading session complete")
