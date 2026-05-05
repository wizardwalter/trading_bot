import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.database import get_connection

load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

def get_latest_timestamp():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(timestamp) FROM candles_1m WHERE symbol = 'SPY'")
        return cur.fetchone()[0]

def fetch_polygon_data(start_date, end_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/{start_date}/{end_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print("❌ Polygon API Error:", r.status_code, r.text)
        return []

    return r.json().get("results", [])

def polygon_to_dataframe(results):
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(columns={
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume"
    }, inplace=True)
    df["symbol"] = "SPY"
    return df[["symbol", "timestamp", "open", "high", "low", "close", "volume"]]

def backfill_polygon_spy():
    print("🔍 Checking last timestamp in DB...")
    latest = get_latest_timestamp()

    if latest is None:
        print("❌ No existing data. Set a manual start date.")
        return

    start_date = (latest + timedelta(minutes=1)).date()
    end_date = datetime.now().date()

    print(f"📈 Backfilling from {start_date} to {end_date}")

    while start_date <= end_date:
        iso_date = start_date.isoformat()
        print(f"⏳ Fetching {iso_date}...")
        results = fetch_polygon_data(iso_date, iso_date)

        df = polygon_to_dataframe(results)
        if df.empty:
            print(f"⚠️ No data for {iso_date}. Skipping.")
            start_date += timedelta(days=1)
            continue

        print(f"✅ Inserting {len(df)} rows for {iso_date}")
        with get_connection() as conn:
            cur = conn.cursor()
            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO candles_1m (symbol, timestamp, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (
                    row["symbol"], row["timestamp"],
                    row["open"], row["high"], row["low"],
                    row["close"], row["volume"]
                ))
            conn.commit()

        start_date += timedelta(days=1)
        time.sleep(60)

    print("🎉 Polygon backfill complete.")

if __name__ == "__main__":
    backfill_polygon_spy()