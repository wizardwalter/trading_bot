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

STARTING_CASH = float(os.getenv("PAPER_STARTING_CASH", "100000"))


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def log_trade(ticker, action, price, quantity, signal_strength=0.0, reason=""):
    qty = float(quantity)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades (ticker, action, price, quantity, signal_strength, reason, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (ticker, action, float(price), qty, float(signal_strength), reason),
                )
    except Exception:
        # Backward-compatibility for older schemas where quantity is INTEGER.
        # Preserve non-zero intent instead of truncating tiny crypto fills to 0.
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO trades (ticker, action, price, quantity, signal_strength, reason, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, NOW())
                        """,
                        (ticker, action, float(price), int(round(qty)), float(signal_strength), reason),
                    )
        except Exception:
            # DB optional during early bring-up; trading loop should still run.
            return


def get_position_qty(ticker: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(CASE WHEN action='buy' THEN quantity ELSE -quantity END), 0)
                FROM trades
                WHERE ticker=%s
                """,
                (ticker,),
            )
            return int(cur.fetchone()[0] or 0)


def get_portfolio_equity(latest_prices: dict[str, float]) -> float:
    cash = STARTING_CASH

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker, action, price, quantity FROM trades ORDER BY timestamp ASC")
            rows = cur.fetchall()

    positions: dict[str, int] = {}

    for ticker, action, price, quantity in rows:
        qty = int(quantity)
        px = float(price)
        if action == "buy":
            cash -= px * qty
            positions[ticker] = positions.get(ticker, 0) + qty
        elif action == "sell":
            cash += px * qty
            positions[ticker] = positions.get(ticker, 0) - qty

    mtm = 0.0
    for ticker, qty in positions.items():
        if qty <= 0:
            continue
        px = latest_prices.get(ticker)
        if px is None:
            continue
        mtm += qty * float(px)

    return cash + mtm


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
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM tickers ORDER BY symbol")
                out = [r[0] for r in cur.fetchall()]
                if out:
                    return out
    except Exception:
        pass

    env_list = os.getenv("DEFAULT_TICKERS", "SPY,QQQ,BTC-USD")
    return [s.strip() for s in env_list.split(",") if s.strip()]


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
