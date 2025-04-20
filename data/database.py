def log_trade(ticker, price, paper=True):
    # TODO: Save into PostgreSQL
    print(f"📦 Logging trade: {ticker} @ ${price:.2f} [paper={paper}]")