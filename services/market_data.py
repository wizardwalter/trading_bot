from __future__ import annotations

import os
import time

import pandas as pd
import yfinance as yf

from data.database import get_all_candles
from services.alpaca_candles import fetch_crypto_bars


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def period_to_days(period: str) -> int:
    p = str(period).strip().lower()
    if p.endswith("d"):
        return max(int(p[:-1]), 1)
    if p.endswith("mo"):
        return max(int(p[:-2]) * 30, 30)
    if p.endswith("y"):
        return max(int(p[:-1]) * 365, 365)
    return 60


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce provider-specific candle frames into a sorted OHLCV DataFrame."""
    if df.empty:
        raise RuntimeError("market data frame is empty")

    out = df.copy()
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    for src, dst in rename_map.items():
        if src in out.columns and dst not in out.columns:
            out.rename(columns={src: dst}, inplace=True)

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    elif isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[~out.index.isna()].sort_index()
    else:
        raise RuntimeError("market data is missing a timestamp index/column")

    missing = [col for col in OHLCV_COLUMNS if col not in out.columns]
    if missing:
        raise RuntimeError(f"market data missing columns: {missing}")

    out = out[OHLCV_COLUMNS].copy()
    for col in OHLCV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna()
    if out.empty:
        raise RuntimeError("market data is empty after OHLCV normalization")
    return out


def download_from_db(
    symbol: str,
    interval: str,
    period: str,
    *,
    max_stale_hours: float | None = None,
) -> pd.DataFrame:
    df = get_all_candles(symbol, interval)
    if df.empty:
        raise RuntimeError(f"DB has no candles for {symbol} {interval}")

    out = normalize_ohlcv_frame(df)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=period_to_days(period))

    if max_stale_hours is None:
        max_stale_hours = float(os.getenv("MARKET_DATA_DB_MAX_STALE_HOURS", "36"))

    latest_ts = out.index.max()
    if pd.notna(latest_ts):
        stale_cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=max_stale_hours)
        if latest_ts < stale_cutoff:
            raise RuntimeError(
                f"DB candles stale for {symbol} {interval}: latest={latest_ts}, "
                f"max_stale_hours={max_stale_hours}"
            )

    out = out[out.index >= cutoff]
    if out.empty:
        raise RuntimeError(f"DB candles empty for {symbol} {interval} after period filter")
    return out


def download_market_data(
    symbol: str = "BTC-USD",
    interval: str = "5m",
    period: str = "60d",
    retries: int = 4,
    *,
    source_pref: str | None = None,
) -> pd.DataFrame:
    """Fetch candles using the same provider priority for training and live runs."""
    resolved_source = (source_pref or os.getenv("MARKET_DATA_SOURCE") or "db-first").strip().lower()

    if resolved_source in {"db", "db-first", "auto"}:
        try:
            return download_from_db(symbol=symbol, interval=interval, period=period)
        except Exception as exc:
            if resolved_source == "db":
                raise
            print(f"DB market data unavailable for {symbol} {interval}, falling back: {exc}")

    use_alpaca_first = symbol.upper() in {"BTC-USD", "BTC/USD", "BTCUSD"}
    if use_alpaca_first:
        tf_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
        timeframe = tf_map.get(interval, "5Min")
        lookback_days = period_to_days(period)
        min_lookback = int(os.getenv("MARKET_DATA_MIN_LOOKBACK_DAYS", "180"))
        lookback_days = max(lookback_days, min_lookback)

        try:
            alpaca_df = fetch_crypto_bars(
                symbol="BTC/USD",
                timeframe=timeframe,
                lookback_days=lookback_days,
            )
            return normalize_ohlcv_frame(alpaca_df)
        except Exception as exc:
            print(f"Alpaca market data unavailable for {symbol}, falling back: {exc}")

    yf_period = period
    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
    if interval in intraday_intervals and str(period).endswith("d"):
        try:
            if int(str(period)[:-1]) > 60:
                yf_period = "60d"
        except Exception:
            yf_period = "60d"

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=yf_period,
                progress=False,
                threads=False,
                auto_adjust=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            return normalize_ohlcv_frame(df)
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(0.6 * attempt)

    raise RuntimeError(f"download failed for {symbol} after {retries} attempts: {last_err}")
