import pandas as pd


def calculate_ema(df: pd.DataFrame, span: int = 20, column: str = 'Close') -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    df[f'EMA_{span}'] = df[column].ewm(span=span, adjust=False).mean()
    return df


def compute_ema(df: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
    return calculate_ema(df.copy(), span=period, column=column)[f'EMA_{period}']
