-- schema.sql
-- docker exec -it trading-postgres psql -U walter -d trading_bot
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    action VARCHAR(10),  -- buy or sell
    price DECIMAL(10, 2),
    quantity INTEGER,
    signal_strength FLOAT,
    reason TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE indicators (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    indicator_name VARCHAR(50),
    value FLOAT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

