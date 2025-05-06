# services/features/store_features_cron.py
import pandas as pd
from data.database import get_all_candles, insert_features_to_db
from features.feature_engineering import generate_features
from scripts.seed_tickers import get_tracked_tickers

def store_eod_features():
    for symbol in get_tracked_tickers():
        for interval in ['1m', '5m', '15m']:
            candles_df = get_all_candles(symbol, interval)
            if candles_df.empty:
                continue
            features_df = generate_features(candles_df)
            insert_features_to_db(symbol, interval, features_df)

if __name__ == "__main__":
    store_eod_features()