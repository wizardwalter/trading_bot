import pandas as pd

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close') -> pd.DataFrame:
    """
    Calculate MACD and Signal Line for the given DataFrame.
    
    Args:
        df (pd.DataFrame): Input data with a 'Close' column (or another specified).
        fast_period (int): EMA fast period (default 12).
        slow_period (int): EMA slow period (default 26).
        signal_period (int): Signal line EMA period (default 9).
        column (str): Column to use for calculation (default 'Close').

    Returns:
        pd.DataFrame: Modified DataFrame with 'MACD' and 'MACD_Signal' columns.
    """
    if column not in df.columns:
        raise ValueError(f"'{column}' column not found in DataFrame.")

    exp1 = df[column].ewm(span=fast_period, adjust=False).mean()
    exp2 = df[column].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df