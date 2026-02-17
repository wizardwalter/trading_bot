import pandas as pd


def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close') -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"'{column}' column not found in DataFrame.")

    exp1 = df[column].ewm(span=fast_period, adjust=False).mean()
    exp2 = df[column].ewm(span=slow_period, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df


def compute_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'Close'):
    out = calculate_macd(df.copy(), fast_period, slow_period, signal_period, column)
    return out['MACD'], out['MACD_signal'], out['MACD_hist']
