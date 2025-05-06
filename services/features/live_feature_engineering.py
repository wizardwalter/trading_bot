# services/features/live_feature_engineering.py
import pandas as pd
from data.database import get_latest_candles
from features.feature_engineering import generate_features

def get_realtime_features(symbol: str, interval: str = '1m', lookback: int = 100):
    candles_df = get_latest_candles(symbol, interval, lookback)
    if candles_df.empty:
        return pd.DataFrame()
    features_df = generate_features(candles_df)
    return features_df