import pandas as pd


def calculate_obv(df: pd.DataFrame, column_close: str = 'Close', column_volume: str = 'Volume') -> pd.DataFrame:
    if column_close not in df.columns or column_volume not in df.columns:
        raise ValueError("Required columns not found in DataFrame.")

    obv = [0]
    for i in range(1, len(df)):
        if df[column_close].iloc[i] > df[column_close].iloc[i - 1]:
            obv.append(obv[-1] + df[column_volume].iloc[i])
        elif df[column_close].iloc[i] < df[column_close].iloc[i - 1]:
            obv.append(obv[-1] - df[column_volume].iloc[i])
        else:
            obv.append(obv[-1])

    df['OBV'] = obv
    return df


def compute_obv(df: pd.DataFrame, column_close: str = 'Close', column_volume: str = 'Volume') -> pd.Series:
    return calculate_obv(df.copy(), column_close=column_close, column_volume=column_volume)['OBV']
