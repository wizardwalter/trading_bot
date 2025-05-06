import pandas as pd

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: float = 2):
    """
    Calculates Bollinger Bands and appends them to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with at least a 'Close' column.
        window (int): Period for moving average and std dev.
        num_std_dev (float): Number of standard deviations for the bands.

    Returns:
        pd.DataFrame: Original DataFrame with 'MA20', 'Upper_Band', 'Lower_Band' columns added.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    df['MA20'] = df['Close'].rolling(window=window).mean()
    df['STD20'] = df['Close'].rolling(window=window).std()
    df['Upper_Band'] = df['MA20'] + (num_std_dev * df['STD20'])
    df['Lower_Band'] = df['MA20'] - (num_std_dev * df['STD20'])

    return df