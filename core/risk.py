from __future__ import annotations

import os

MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.0075"))
MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "0.03"))
MAX_PORTFOLIO_EXPOSURE = float(os.getenv("MAX_PORTFOLIO_EXPOSURE", "0.35"))


def drawdown_exceeded(day_start_equity: float, current_equity: float) -> bool:
    if day_start_equity <= 0:
        return False
    dd = (day_start_equity - current_equity) / day_start_equity
    return dd >= MAX_DAILY_DRAWDOWN


def exceeds_portfolio_exposure(current_exposure: float, trade_notional: float, equity: float) -> bool:
    if equity <= 0:
        return False
    projected = current_exposure + max(trade_notional, 0.0)
    return (projected / equity) > MAX_PORTFOLIO_EXPOSURE
