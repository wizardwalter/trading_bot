import os
import psycopg2
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}

TIMEFRAMES = {
    "1m": "candles_1m",
    "5m": "candles_5m",
    "15m": "candles_15m",
    "1h": "candles_1h",
    "1d": "candles_daily",
}


def _scalar(v):
    try:
        return float(v.iloc[0])
    except Exception:
        return float(v)


def fetch_and_store_candles():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT symbol FROM tickers")
        tickers = [row[0] for row in cur.fetchall()]

        for symbol in tickers:
            for interval, table in TIMEFRAMES.items():
                print(f"📈 Fetching {interval} candles for {symbol}...")
                data = yf.download(tickers=symbol, interval=interval, period="2d", progress=False)

                if data.empty:
                    continue

                for timestamp, row in data.iterrows():
                    cur.execute(
                        f"""
                        INSERT INTO {table} (symbol, timestamp, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (symbol, timestamp) DO NOTHING
                    """,
                        (
                            symbol,
                            timestamp.to_pydatetime(),
                            _scalar(row["Open"]),
                            _scalar(row["High"]),
                            _scalar(row["Low"]),
                            _scalar(row["Close"]),
                            _scalar(row["Volume"]),
                        ),
                    )

                conn.commit()
                print(f"✅ Inserted into {table}")

        cur.close()
        conn.close()
        print("🎉 Done storing all candles!")

    except Exception as e:
        print("❌ Error during candle fetch/store:", e)


if __name__ == "__main__":
    fetch_and_store_candles()
