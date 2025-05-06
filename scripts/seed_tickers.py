import psycopg2
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

# Only seed QQQ and SPY
tickers = [
    "QQQ",
    "SPY"
]

def insert_tickers():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        for ticker in tickers:
            cur.execute("INSERT INTO tickers (symbol) VALUES (%s) ON CONFLICT (symbol) DO NOTHING;", (ticker,))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Tickers successfully seeded.")
    except Exception as e:
        print("❌ Error seeding tickers:", e)

def get_tracked_tickers():
    return tickers

if __name__ == "__main__":
    insert_tickers()