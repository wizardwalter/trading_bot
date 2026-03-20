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

# "Meaningful" training timeframes we use (plus daily context).
TIMEFRAMES = [
    ('1m', 'candles_1m', 1, 'minute'),
    ('5m', 'candles_5m', 5, 'minute'),
    ('15m', 'candles_15m', 15, 'minute'),
    ('1h', 'candles_1h', 1, 'hour'),
    ('1d', 'candles_daily', 1, 'day'),
]


def fetch_range(ticker: str, mult: int, span: str, start: date, end: date) -> list[dict]:
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{start}/{end}"
    params = {
        'adjusted': 'true',
        'sort': 'asc',
        'limit': 50000,
        'apiKey': API_KEY,
    }

    all_rows: list[dict] = []
    next_url = url
    first_page = True

    while next_url:
        req_url = next_url
        req_params = params if first_page else None
        first_page = False

        last_err: Exception | None = None
        r = None
        for attempt in range(1, 7):
            try:
                r = requests.get(req_url, params=req_params, timeout=45)
                if r.status_code >= 400:
                    snippet = r.text[:240]
                    raise RuntimeError(f"HTTP {r.status_code}: {snippet}")
                break
            except Exception as e:
                last_err = e
                if attempt < 6:
                    time.sleep(min(0.6 * (2 ** (attempt - 1)), 8.0))
                else:
                    raise RuntimeError(f"Massive request failed for {req_url}: {last_err}")

        payload = r.json() if r is not None else {}
        rows = payload.get('results', [])
        all_rows.extend(rows)

        nxt = payload.get('next_url')
        if nxt:
            sep = '&' if '?' in nxt else '?'
            next_url = f"{nxt}{sep}apiKey={API_KEY}"
            time.sleep(0.05)
        else:
            next_url = ''

    return all_rows


def insert_rows(conn, table: str, symbol: str, rows: list[dict]) -> int:
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


def max_ts(conn, table: str, symbol: str):
    with conn.cursor() as cur:
        cur.execute(f"SELECT max(timestamp) FROM {table} WHERE symbol=%s", (symbol,))
        return cur.fetchone()[0]


def tf_step_days(tf_name: str) -> int:
    # Keep request sizes sane while preserving throughput.
    if tf_name == '1m':
        return 30
    if tf_name == '5m':
        return 90
    if tf_name == '15m':
        return 180
    if tf_name == '1h':
        return 365
    return 365 * 2


def main():
    if not API_KEY:
        raise RuntimeError('Missing MASSIVE_API_KEY/POLYGON_API_KEY in .env')

    symbol = os.getenv('BACKFILL_SYMBOL', 'BTC-USD')
    massive_ticker = os.getenv('BACKFILL_MASSIVE_TICKER', 'X:BTCUSD')
    years = int(os.getenv('BACKFILL_YEARS', '10'))
    end = datetime.now(timezone.utc).date()
    start_global = end - timedelta(days=max(years, 1) * 365)

    conn = psycopg2.connect(**DB_CONFIG)
    totals = {tf[0]: 0 for tf in TIMEFRAMES}

    try:
        print(f"Starting Massive backfill: symbol={symbol} ticker={massive_ticker} years={years} start={start_global} end={end}")

        for tf_name, table, mult, span in TIMEFRAMES:
            existing_max = max_ts(conn, table, symbol)
            if existing_max is not None:
                start = max(start_global, existing_max.date() - timedelta(days=2))
            else:
                start = start_global

            step_days = tf_step_days(tf_name)
            chunk_start = start
            print(f"[{tf_name}] start={start} step_days={step_days}")

            while chunk_start <= end:
                chunk_end = min(chunk_start + timedelta(days=step_days - 1), end)
                rows = fetch_range(massive_ticker, mult, span, chunk_start, chunk_end)
                inserted = insert_rows(conn, table, symbol, rows)
                totals[tf_name] += inserted
                print(f"[{tf_name}] {chunk_start} -> {chunk_end} fetched={len(rows)} upserted={inserted}")
                chunk_start = chunk_end + timedelta(days=1)
                time.sleep(0.06)

    finally:
        conn.close()

    print('Backfill complete:')
    for tf, n in totals.items():
        print(f"  {tf}: {n}")


if __name__ == '__main__':
    main()
