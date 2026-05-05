from __future__ import annotations

import os
import sys

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import macd, macd_signal
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.database import get_connection
from ml.label_generator import generate_labels


def build_labeled_dataset(symbol: str = "SPY") -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM candles_1m
            WHERE symbol = %s
            ORDER BY timestamp ASC
            """,
            conn,
            params=(symbol,),
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    df_10m = df.resample("10T").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()

    df_10m["RSI"] = RSIIndicator(close=df_10m["close"], window=14).rsi()
    df_10m["MACD"] = macd(df_10m["close"], window_slow=26, window_fast=12)
    df_10m["MACD_signal"] = macd_signal(df_10m["close"], window_slow=26, window_fast=12, window_sign=9)
    bb = BollingerBands(close=df_10m["close"], window=20, window_dev=2)
    df_10m["bb_upper"] = bb.bollinger_hband()
    df_10m["bb_lower"] = bb.bollinger_lband()
    vwap = VolumeWeightedAveragePrice(
        high=df_10m["high"],
        low=df_10m["low"],
        close=df_10m["close"],
        volume=df_10m["volume"],
    )
    df_10m["VWAP"] = vwap.volume_weighted_average_price()
    df_10m.dropna(inplace=True)
    return generate_labels(df_10m)


def main() -> None:
    symbol = os.getenv("LEGACY_TRAIN_SYMBOL", "SPY")
    df_labeled = build_labeled_dataset(symbol=symbol)
    os.makedirs("data", exist_ok=True)
    df_labeled.to_csv("data/labeled_10m.csv", index=True)
    print("Labeled 10m data saved to: data/labeled_10m.csv")


if __name__ == "__main__":
    main()
