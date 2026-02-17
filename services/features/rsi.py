import pandas as pd


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    if 'close' in df.columns:
        close = df['close']
    elif 'Close' in df.columns:
        close = df['Close']
    else:
        raise ValueError("DataFrame must contain a 'close' or 'Close' column")

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss

    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return calculate_rsi(df.copy(), period=period)['RSI']
