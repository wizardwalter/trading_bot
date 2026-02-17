import os
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def log_trade(ticker, price, paper=True):
    print(f"📦 Logging trade: {ticker} @ ${price:.2f} [paper={paper}]")


def _candles_to_df(rows):
    cols = ["id", "symbol", "timestamp", "Open", "High", "Low", "Close", "Volume"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
    return df


def get_latest_candles(symbol, interval, limit=100):
    table_name = f"candles_{interval}"
    query = f"""
        SELECT * FROM {table_name}
        WHERE symbol = %s
        ORDER BY timestamp DESC
        LIMIT %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, limit))
            rows = cur.fetchall()
            return _candles_to_df(list(reversed(rows)))


def get_all_candles(symbol, interval):
    table_name = f"candles_{interval}"
    query = f"""
        SELECT * FROM {table_name}
        WHERE symbol = %s
        ORDER BY timestamp ASC
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            return _candles_to_df(cur.fetchall())


def get_all_tickers():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM tickers ORDER BY symbol")
            return [r[0] for r in cur.fetchall()]


def insert_features_to_db(symbol, interval, features_df):
    if features_df.empty:
        return

    rows = []
    for _, row in features_df.iterrows():
        rows.append((
            symbol,
            interval,
            row['timestamp'].to_pydatetime() if hasattr(row['timestamp'], 'to_pydatetime') else datetime.utcnow(),
            float(row.get('RSI')) if pd.notna(row.get('RSI')) else None,
            float(row.get('MACD')) if pd.notna(row.get('MACD')) else None,
            float(row.get('MACD_signal')) if pd.notna(row.get('MACD_signal')) else None,
            float(row.get('MACD_hist')) if pd.notna(row.get('MACD_hist')) else None,
            float(row.get('EMA_12')) if pd.notna(row.get('EMA_12')) else None,
            float(row.get('EMA_26')) if pd.notna(row.get('EMA_26')) else None,
            float(row.get('BB_upper')) if pd.notna(row.get('BB_upper')) else None,
            float(row.get('BB_lower')) if pd.notna(row.get('BB_lower')) else None,
            float(row.get('OBV')) if pd.notna(row.get('OBV')) else None,
            float(row.get('VWAP')) if pd.notna(row.get('VWAP')) else None,
        ))

    query = """
        INSERT INTO features (
            symbol, interval, timestamp, rsi, macd, macd_signal, macd_hist,
            ema_20, ema_50, bollinger_upper, bollinger_lower, obv, vwap
        ) VALUES %s
        ON CONFLICT (symbol, interval, timestamp) DO NOTHING
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)
