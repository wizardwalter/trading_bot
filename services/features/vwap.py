import pandas as pd


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    if not {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close', and 'Volume' columns.")

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_volume = df['Volume'].cumsum()
    cumulative_vp = (typical_price * df['Volume']).cumsum()

    df['VWAP'] = cumulative_vp / cumulative_volume
    return df


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    return calculate_vwap(df.copy())['VWAP']
