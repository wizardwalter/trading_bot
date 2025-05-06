-- import os
-- import psycopg2
-- from dotenv import load_dotenv

-- load_dotenv()

-- DB_CONFIG = {
--     "dbname": os.getenv("DB_NAME"),
--     "user": os.getenv("DB_USER"),
--     "password": os.getenv("DB_PASS"),
--     "host": os.getenv("DB_HOST"),
--     "port": os.getenv("DB_PORT")
-- }

-- conn = psycopg2.connect(
--     dbname=DB_CONFIG["dbname"],
--     user=DB_CONFIG["user"],
--     password=DB_CONFIG["password"],
--     host=DB_CONFIG["host"],
--     port=DB_CONFIG["port"]
-- )
-- conn.autocommit = True
-- cur = conn.cursor()

-- # Tables to clear
-- candle_tables = [
--     "candles_1m",
--     "candles_5m",
--     "candles_15m",
--     "candles_1h",
--     "candles_daily"
-- ]

-- # Clear candle data
-- for table in candle_tables:
--     cur.execute(f"DELETE FROM {table};")

-- # Only keep QQQ and SPY in tickers table
-- cur.execute("DELETE FROM tickers WHERE symbol NOT IN ('QQQ', 'SPY');")

-- # Confirm it worked
-- cur.execute("SELECT symbol FROM tickers;")
-- print("Remaining tickers:", cur.fetchall())

-- cur.close()
-- conn.close()