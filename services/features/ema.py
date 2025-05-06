import pandas as pd

def calculate_ema(df: pd.DataFrame, span: int = 20, column: str = 'Close') -> pd.DataFrame:
    """
    Calculates the Exponential Moving Average (EMA) for a given column.

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        span (int): Span period for EMA calculation.
        column (str): Column to calculate EMA on (default is 'Close').

    Returns:
        pd.DataFrame: DataFrame with an additional 'EMA_{span}' column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df[f'EMA_{span}'] = df[column].ewm(span=span, adjust=False).mean()
    return df