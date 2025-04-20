-- schema.sql
-- docker exec -it trading-postgres psql -U walter -d trading_bot
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    action VARCHAR(10),  -- buy or sell
    price DECIMAL(10, 2),
    quantity INTEGER,
    signal_strength FLOAT,
    reason TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS indicators (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    indicator_name VARCHAR(50),
    value FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tickers (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL UNIQUE
);

