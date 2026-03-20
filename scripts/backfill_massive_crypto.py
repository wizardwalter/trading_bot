from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import psycopg2
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')

API_KEY = os.getenv('MASSIVE_API_KEY') or os.getenv('POLYGON_API_KEY')
BASE_URL = (os.getenv('MASSIVE_BASE_URL') or 'https://api.polygon.io').rstrip('/')

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASS'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT'),
}

TIMEFRAMES = [
    ('1m', 'candles_1m', 1, 'minute'),
    ('5m', 'candles_5m', 5, 'minute'),
    ('15m', 'candles_15m', 15, 'minute'),
    ('1h', 'candles_1h', 1, 'hour'),
]


def fetch_range(ticker: str, mult: int, span: str, start: date, end: date) -> list[dict]:
    # Massive/Polygon aggs endpoint
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start}/{end}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': API_KEY,
    }

    all_rows: list[dict] = []
    next_url = url
    while next_url:
        if next_url.startswith('http'):
            req_url = next_url
            req_params = None
        else:
            req_url = next_url
            req_params = params

        r = requests.get(req_url, params=req_params, timeout=40)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get('results', [])
        all_rows.extend(rows)

        nxt = payload.get('next_url')
        if nxt:
            sep = '&' if '?' in nxt else '?'
            next_url = f"{nxt}{sep}apiKey={API_KEY}"
            time.sleep(0.08)
        else:
            next_url = ''

    return all_rows


def insert_rows(conn, table: str, symbol: str, rows: list[dict]):
    if not rows:
        return 0
    tuples = []
    for r in rows:
        ts = datetime.fromtimestamp(r['t'] / 1000, tz=timezone.utc)
        tuples.append((
            symbol,
            ts,
            float(r.get('o', 0.0)),
            float(r.get('h', 0.0)),
            float(r.get('l', 0.0)),
            float(r.get('c', 0.0)),
            float(r.get('v', 0.0)),
        ))

    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO {table} (symbol, timestamp, open, high, low, close, volume)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (symbol, timestamp) DO UPDATE
            SET open=EXCLUDED.open,
                high=EXCLUDED.high,
                low=EXCLUDED.low,
                close=EXCLUDED.close,
                volume=EXCLUDED.volume
            """,
            tuples,
        )
    conn.commit()
    return len(tuples)


def main():
    if not API_KEY:
        raise RuntimeError('Missing MASSIVE_API_KEY/POLYGON_API_KEY in .env')

    symbol = 'BTC-USD'
    massive_ticker = 'X:BTCUSD'
    start = date(2018, 1, 1)
    end = datetime.now(timezone.utc).date()

    conn = psycopg2.connect(**DB_CONFIG)
    total_inserted = {tf[0]: 0 for tf in TIMEFRAMES}

    try:
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=29), end)
            print(f"Backfilling chunk {chunk_start} -> {chunk_end}")

            for tf_name, table, mult, span in TIMEFRAMES:
                rows = fetch_range(massive_ticker, mult, span, chunk_start, chunk_end)
                inserted = insert_rows(conn, table, symbol, rows)
                total_inserted[tf_name] += inserted
                print(f"  {tf_name}: fetched={len(rows)} upserted={inserted}")

            chunk_start = chunk_end + timedelta(days=1)
            time.sleep(0.2)

    finally:
        conn.close()

    print('Backfill complete:')
    for k, v in total_inserted.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
