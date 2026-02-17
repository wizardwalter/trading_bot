import pandas as pd


def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: float = 2):
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['BB_middle'] = ma
    df['BB_upper'] = ma + (num_std_dev * std)
    df['BB_lower'] = ma - (num_std_dev * std)
    return df


def compute_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std_dev: float = 2):
    out = calculate_bollinger_bands(df.copy(), window=window, num_std_dev=num_std_dev)
    return out['BB_upper'], out['BB_middle'], out['BB_lower']
