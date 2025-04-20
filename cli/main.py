import os
import psycopg2
from dotenv import load_dotenv
import subprocess

load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")


DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

def ensure_postgres_running():
    try:
        result = subprocess.run(
            ["pg_isready"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "accepting connections" not in result.stdout:
            print("🔁 Starting Postgres service...")
            subprocess.run(["brew", "services", "start", "postgresql"], check=True)
        else:
            print("✅ Postgres is already running.")
    except Exception as e:
        print("❌ Error checking or starting Postgres:", e)

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
    ensure_postgres_running()
    test_connection()