import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def log_trade(ticker, price, paper=True):
    # TODO: Save into PostgreSQL
    print(f"📦 Logging trade: {ticker} @ ${price:.2f} [paper={paper}]")

import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

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
            return cur.fetchall()

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
            return cur.fetchall()

def insert_features_to_db(symbol, interval, features, timestamp):
    rows = [
        (symbol, interval, feature_name, float(value), timestamp)
        for feature_name, value in features.items()
    ]
    query = """
        INSERT INTO indicators (symbol, interval, feature_name, value, timestamp)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, query, rows)