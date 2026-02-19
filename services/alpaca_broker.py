from __future__ import annotations

import time

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
        base = ALPACA_BASE_URL.rstrip("/")
        # Accept either base URL style:
        # - https://paper-api.alpaca.markets
        # - https://paper-api.alpaca.markets/v2
        if base.endswith("/v2"):
            base = base[:-3]
        self.base_url = base
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": APCA_API_KEY_ID,
                "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, timeout: int = 15, retries: int = 3, **kwargs):
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = self.session.request(method, f"{self.base_url}{path}", timeout=timeout, **kwargs)
                return response
            except requests.RequestException as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.5 * attempt)
        raise RuntimeError(f"Alpaca {method} {path} failed after {retries} attempts: {last_err}")

    def get_account(self) -> dict:
        r = self._request("GET", "/v2/account")
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> list[dict]:
        r = self._request("GET", "/v2/positions")
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json()

    def get_buying_power(self) -> float:
        acct = self.get_account()
        return float(acct.get("buying_power") or acct.get("equity") or 0.0)

    def get_position_qty(self, symbol: str) -> float:
        symbol = _to_alpaca_symbol(symbol)
        try:
            r = self._request("GET", f"/v2/positions/{symbol}")
            if r.status_code == 404:
                return 0.0
            r.raise_for_status()
            return float(r.json().get("qty", 0.0))
        except Exception:
            return 0.0

    def submit_market_order(self, symbol: str, side: str, qty: float) -> dict:
        raw_symbol = symbol
        symbol = _to_alpaca_symbol(symbol)
        is_crypto = "-" in (raw_symbol or "")

        if is_crypto:
            qty_str = f"{float(qty):.6f}".rstrip("0").rstrip(".")
            tif = "gtc"
        else:
            qty_str = str(int(float(qty)))
            tif = "day"

        payload = {
            "symbol": symbol,
            "qty": qty_str,
            "side": side,
            "type": "market",
            "time_in_force": tif,
        }
        r = self._request("POST", "/v2/orders", json=payload)
        r.raise_for_status()
        return r.json()
