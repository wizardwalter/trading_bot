from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def is_24_7_symbol(symbol: str) -> bool:
    return symbol.upper() in {"BTC-USD", "ETH-USD"}


def is_us_equity_symbol(symbol: str) -> bool:
    return symbol.upper() in {"SPY", "QQQ"}


def is_trade_window_open(symbol: str, now_utc: datetime | None = None) -> bool:
    if now_utc is None:
        now_utc = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

    if is_24_7_symbol(symbol):
        return True

    if is_us_equity_symbol(symbol):
        et_now = now_utc.astimezone(EASTERN)
        if et_now.weekday() >= 5:
            return False
        start = time(9, 30)
        end = time(16, 0)
        return start <= et_now.time() <= end

    return False
