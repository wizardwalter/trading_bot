from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import joblib
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import macd, macd_signal
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.database import get_connection
from discord.notify import send_trade_alert


MODEL_PATH = os.getenv("LEGACY_MODEL_PATH", "models/latest_model.pkl")
SYMBOL = os.getenv("LEGACY_LIVE_SYMBOL", "SPY")
LOOP_INTERVAL_SECONDS = int(os.getenv("LEGACY_LIVE_INTERVAL_SECONDS", "60"))
WINDOW_ROWS = int(os.getenv("LEGACY_LIVE_WINDOW_ROWS", "60"))
LABELS = {0: "SELL", 1: "HOLD", 2: "BUY"}


def fetch_latest_1m_candles(symbol: str = SYMBOL) -> pd.DataFrame:
    with get_connection() as conn:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles_1m
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(symbol, WINDOW_ROWS))
    return df.sort_values("timestamp")


def build_features(df: pd.DataFrame) -> pd.DataFrame | None:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    df_10m = df.resample("10min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    if df_10m.empty:
        return None

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
    return df_10m


def _prepare_model_row(model, features_df: pd.DataFrame) -> pd.DataFrame:
    latest = features_df.iloc[[-1]].copy()
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("Loaded model is missing feature_names_in_; cannot align live features safely.")

    missing = [col for col in feature_names if col not in latest.columns]
    if missing:
        raise ValueError(f"Live feature row is missing model columns: {missing}")

    return latest[feature_names].astype("float32")


def predict_live() -> None:
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    while True:
        try:
            df_raw = fetch_latest_1m_candles(SYMBOL)
            df_features = build_features(df_raw)

            if df_features is None or df_features.empty:
                print(f"[{datetime.now()}] Not enough data to make a prediction.")
            else:
                latest = _prepare_model_row(model, df_features)
                probs = model.predict_proba(latest)[0]
                pred = int(model.predict(latest)[0])
                label_text = LABELS.get(pred, f"UNKNOWN({pred})")
                price = float(df_features["close"].iloc[-1])

                print(f"[{datetime.now()}] Prediction: {label_text} | Probabilities: {probs.round(2)}")
                if label_text in {"BUY", "SELL"}:
                    send_trade_alert(
                        ticker=SYMBOL,
                        action=label_text.lower(),
                        price=price,
                        quantity=0,
                        confidence=float(max(probs)),
                        reason=f"legacy_model={os.path.basename(MODEL_PATH)} probs={probs.round(3).tolist()}",
                        paper=True,
                    )
        except Exception as exc:
            print(f"[{datetime.now()}] Error: {exc}")

        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    predict_live()
