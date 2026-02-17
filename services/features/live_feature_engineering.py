import pandas as pd
from data.database import get_latest_candles
from services.features.feature_engineering import generate_features


def get_realtime_features(symbol: str, interval: str = '1m', lookback: int = 100):
    candles_df = get_latest_candles(symbol, interval, lookback)
    if candles_df.empty:
        return pd.DataFrame()
    return generate_features(candles_df)
