import pandas as pd

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        df (pd.DataFrame): DataFrame with 'Close', 'High', 'Low', and 'Volume' columns.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'VWAP' column.
    """
    if not {'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', 'Close', and 'Volume' columns.")

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_volume = df['Volume'].cumsum()
    cumulative_vp = (typical_price * df['Volume']).cumsum()

    df['VWAP'] = cumulative_vp / cumulative_volume
    return df