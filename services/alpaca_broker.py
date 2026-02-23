from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

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
        cache_path = os.getenv("ALPACA_ACCOUNT_CACHE", "data/cache/alpaca_account.json")
        self._account_cache_path = Path(cache_path) if cache_path else None
        self._account_cache_max_age = max(int(os.getenv("ALPACA_ACCOUNT_CACHE_MAX_AGE_SECONDS", "420")), 0)
        self._allow_stale_account = os.getenv("ALPACA_ALLOW_STALE_ACCOUNT", "1") == "1"
        self._allow_synth_account = os.getenv("ALPACA_ALLOW_SYNTH_ACCOUNT", "0") == "1"
        self._synth_bp_multiplier = max(float(os.getenv("ALPACA_SYNTH_BP_MULT", "1.4")), 1.0)
        self._synth_equity = os.getenv("ALPACA_SYNTH_EQUITY")

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

    def _write_account_cache(self, payload: dict) -> None:
        if not self._account_cache_path:
            return
        try:
            to_store = dict(payload)
            to_store["_cached_at"] = datetime.now(timezone.utc).isoformat()
            self._account_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._account_cache_path.write_text(json.dumps(to_store, indent=2, sort_keys=True))
        except Exception:
            # Cache best-effort only.
            return

    def _load_account_cache(self) -> dict | None:
        if not self._allow_stale_account or not self._account_cache_path or not self._account_cache_path.exists():
            return None
        try:
            cached_payload = json.loads(self._account_cache_path.read_text())
        except Exception:
            return None

        try:
            cached_at_raw = cached_payload.get("_cached_at")
            if cached_at_raw:
                cached_at = datetime.fromisoformat(cached_at_raw)
                if cached_at.tzinfo is None:
                    cached_at = cached_at.replace(tzinfo=timezone.utc)
            else:
                cached_at = datetime.fromtimestamp(self._account_cache_path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            cached_at = datetime.fromtimestamp(self._account_cache_path.stat().st_mtime, tz=timezone.utc)

        age_seconds = max((datetime.now(timezone.utc) - cached_at).total_seconds(), 0.0)
        if self._account_cache_max_age and age_seconds > self._account_cache_max_age:
            return None

        payload = dict(cached_payload)
        payload["_cache_age_seconds"] = age_seconds
        payload["_stale"] = True
        return payload

    def _synthetic_account_payload(self, error: Exception | str | None = None) -> dict | None:
        if not self._allow_synth_account:
            return None
        try:
            base_equity = float(self._synth_equity or os.getenv("PAPER_STARTING_CASH", "100000"))
        except Exception:
            base_equity = 100000.0
        bp = base_equity * self._synth_bp_multiplier
        warning = "Synthetic Alpaca account snapshot"
        if error:
            warning = f"{warning} ({error})"
        return {
            "equity": base_equity,
            "cash": base_equity,
            "buying_power": bp,
            "long_market_value": 0.0,
            "short_market_value": 0.0,
            "portfolio_value": base_equity,
            "status": "SYNTHETIC",
            "_stale": True,
            "_cache_age_seconds": float("inf"),
            "_cache_warning": warning,
            "_synthetic": True,
        }

    def get_account(self) -> dict:
        try:
            r = self._request("GET", "/v2/account")
            r.raise_for_status()
            payload = r.json()
            self._write_account_cache(payload)
            payload["_stale"] = False
            payload["_cache_age_seconds"] = 0.0
            return payload
        except Exception as exc:
            cached = self._load_account_cache()
            if cached:
                cached["_cache_warning"] = f"Alpaca account fallback after error: {exc}"
                return cached
            synth = self._synthetic_account_payload(exc)
            if synth:
                return synth
            raise RuntimeError(f"Alpaca account fetch failed with no cache available: {exc}") from exc

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
