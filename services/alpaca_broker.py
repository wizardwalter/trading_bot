from __future__ import annotations

import os
import random
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

    def _request(self, method: str, path: str, timeout: int | tuple[int, int] = 15, retries: int = 4, **kwargs):
        last_err: Exception | None = None
        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}

        # Allow env tuning without code changes.
        connect_timeout = int(os.getenv("ALPACA_CONNECT_TIMEOUT", "8"))
        read_timeout = int(os.getenv("ALPACA_READ_TIMEOUT", "20"))
        request_timeout = timeout if isinstance(timeout, tuple) else (connect_timeout, read_timeout)
        total_retries = max(int(os.getenv("ALPACA_REQUEST_RETRIES", str(retries))), 1)

        for attempt in range(1, total_retries + 1):
            try:
                response = self.session.request(method, f"{self.base_url}{path}", timeout=request_timeout, **kwargs)

                if response.status_code in retryable_statuses and attempt < total_retries:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            sleep_s = max(float(retry_after), 0.25)
                        except ValueError:
                            sleep_s = min(0.6 * (2 ** (attempt - 1)), 6.0)
                    else:
                        sleep_s = min(0.6 * (2 ** (attempt - 1)), 6.0)
                    sleep_s += random.uniform(0.0, 0.2)
                    time.sleep(sleep_s)
                    continue

                return response
            except requests.RequestException as e:
                last_err = e
                if attempt < total_retries:
                    sleep_s = min(0.6 * (2 ** (attempt - 1)), 6.0) + random.uniform(0.0, 0.2)
                    time.sleep(sleep_s)

        raise RuntimeError(f"Alpaca {method} {path} failed after {total_retries} attempts: {last_err}")

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
