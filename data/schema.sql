-- schema.sql
-- docker exec -it trading-postgres psql -U walter -d trading_bot
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    action VARCHAR(10),  -- buy or sell
    price DECIMAL(10, 2),
    quantity INTEGER,
    signal_strength FLOAT,
    reason TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS indicators (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES tickers(symbol),
    indicator_name VARCHAR(50),
    value FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tickers (
  id SERIAL PRIMARY KEY,
  symbol VARCHAR(10) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS candles_1m (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_candles_1m_symbol_ts ON candles_1m(symbol, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS candles_1m_unique_idx ON candles_1m (symbol, timestamp);

CREATE TABLE IF NOT EXISTS candles_5m (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_candles_5m_symbol_ts ON candles_5m(symbol, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS candles_5m_unique_idx ON candles_5m (symbol, timestamp);

CREATE TABLE IF NOT EXISTS candles_daily (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_candles_daily_symbol_ts ON candles_daily(symbol, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS candles_daily_unique_idx ON candles_daily (symbol, timestamp);

CREATE TABLE IF NOT EXISTS candles_15m (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_candles_15m_symbol_ts ON candles_15m(symbol, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS candles_15m_unique_idx ON candles_15m (symbol, timestamp);

CREATE TABLE IF NOT EXISTS candles_1h (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES tickers(symbol),
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_candles_1h_symbol_ts ON candles_1h(symbol, timestamp);
CREATE UNIQUE INDEX IF NOT EXISTS candles_1h_unique_idx ON candles_1h (symbol, timestamp);
-- Repeat for 5m, 15m, 1h, daily

CREATE TABLE IF NOT EXISTS shadow_trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL REFERENCES tickers(symbol),
    timeframe VARCHAR(10),
    decision VARCHAR(10), -- buy, sell, hold
    confidence FLOAT,
    reason TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    outcome FLOAT -- % move after a set time, to evaluate false positives/negatives
);


CREATE TABLE IF NOT EXISTS features (
    symbol TEXT NOT NULL REFERENCES tickers(symbol),
    interval TEXT NOT NULL CHECK (interval IN ('1m', '5m', '15m', '1h', 'daily')),
    timestamp TIMESTAMPTZ NOT NULL,
    rsi FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_hist FLOAT,
    ema_20 FLOAT,
    ema_50 FLOAT,
    bollinger_upper FLOAT,
    bollinger_lower FLOAT,
    obv FLOAT,
    vwap FLOAT,
    PRIMARY KEY (symbol, interval, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp);
