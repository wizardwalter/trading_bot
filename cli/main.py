import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


DB_CONFIG = {
    "dbname": "trading_bot",
    "user": "walter",
    "password": os.getenv("DB_PASS"),
    "host": "localhost",
    "port": 5432
}

def test_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT * FROM tickers;")
        rows = cur.fetchall()
        print("✅ Connected to DB. Tickers found:")
        for row in rows:
            print(row)
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ Failed to connect to DB:", e)

if __name__ == "__main__":
    test_connection()