from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests


ALPACA_DATA_BASE_URL = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/")
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_crypto_bars(symbol: str = "BTC/USD", timeframe: str = "5Min", lookback_days: int = 365, limit: int = 10000) -> pd.DataFrame:
    """Fetch crypto bars from Alpaca market data API.

    Returns DataFrame with columns: Open, High, Low, Close, Volume indexed by timestamp.
    Raises RuntimeError on failures.
    """
    if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
        raise RuntimeError("Alpaca credentials missing for market data")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(int(lookback_days), 1))

    url = f"{ALPACA_DATA_BASE_URL}/v1beta3/crypto/us/bars"
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": _to_iso(start),
        "end": _to_iso(end),
        "limit": max(1, min(int(limit), 10000)),
        "sort": "asc",
    }
    headers = {
        "APCA-API-KEY-ID": APCA_API_KEY_ID,
        "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY,
    }

    rows = []
    page_token: Optional[str] = None

    for _ in range(100):
        if page_token:
            params["page_token"] = page_token
        else:
            params.pop("page_token", None)

        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca data request failed ({r.status_code}): {r.text[:200]}")

        payload = r.json()
        symbol_rows = payload.get("bars", {}).get(symbol, [])
        rows.extend(symbol_rows)

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not rows:
        raise RuntimeError(f"No Alpaca bars returned for {symbol}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"Empty Alpaca bars returned for {symbol}")

    # Alpaca fields: t,o,h,l,c,v
    df["timestamp"] = pd.to_datetime(df["t"], utc=True)
    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(df["o"], errors="coerce").to_numpy(),
            "High": pd.to_numeric(df["h"], errors="coerce").to_numpy(),
            "Low": pd.to_numeric(df["l"], errors="coerce").to_numpy(),
            "Close": pd.to_numeric(df["c"], errors="coerce").to_numpy(),
            "Volume": pd.to_numeric(df["v"], errors="coerce").to_numpy(),
        },
        index=df["timestamp"].to_numpy(),
    ).dropna()

    if out.empty:
        raise RuntimeError(f"All Alpaca bars invalid after parsing for {symbol}")

    return out
