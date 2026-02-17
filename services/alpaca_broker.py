from __future__ import annotations

import requests

from config.settings import ALPACA_BASE_URL, APCA_API_KEY_ID, APCA_API_SECRET_KEY


def _to_alpaca_symbol(symbol: str) -> str:
    mapping = {
        "BTC-USD": "BTCUSD",
        "ETH-USD": "ETHUSD",
    }
    return mapping.get(symbol.upper(), symbol.upper())


class AlpacaBroker:
    def __init__(self):
        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
            raise RuntimeError("Alpaca API credentials are missing")
        self.base_url = ALPACA_BASE_URL.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": APCA_API_KEY_ID,
                "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
                "Content-Type": "application/json",
            }
        )

    def get_account(self) -> dict:
        r = self.session.get(f"{self.base_url}/v2/account", timeout=15)
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> list[dict]:
        r = self.session.get(f"{self.base_url}/v2/positions", timeout=15)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

    def get_position_qty(self, symbol: str) -> float:
        symbol = _to_alpaca_symbol(symbol)
        try:
            r = self.session.get(f"{self.base_url}/v2/positions/{symbol}", timeout=15)
            if r.status_code == 404:
                return 0.0
            r.raise_for_status()
            return float(r.json().get("qty", 0.0))
        except Exception:
            return 0.0

    def submit_market_order(self, symbol: str, side: str, qty: int) -> dict:
        symbol = _to_alpaca_symbol(symbol)
        payload = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        r = self.session.post(f"{self.base_url}/v2/orders", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
