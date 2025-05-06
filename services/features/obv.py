import pandas as pd

def calculate_obv(df: pd.DataFrame, column_close: str = 'Close', column_volume: str = 'Volume') -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        df (pd.DataFrame): DataFrame with 'Close' and 'Volume' columns.
        column_close (str): Name of the close price column.
        column_volume (str): Name of the volume column.

    Returns:
        pd.DataFrame: DataFrame with an additional 'OBV' column.
    """
    if column_close not in df.columns or column_volume not in df.columns:
        raise ValueError("Required columns not found in DataFrame.")

    obv = [0]  # Start OBV at 0
    for i in range(1, len(df)):
        if df[column_close].iloc[i] > df[column_close].iloc[i - 1]:
            obv.append(obv[-1] + df[column_volume].iloc[i])
        elif df[column_close].iloc[i] < df[column_close].iloc[i - 1]:
            obv.append(obv[-1] - df[column_volume].iloc[i])
        else:
            obv.append(obv[-1])

    df['OBV'] = obv
    return df