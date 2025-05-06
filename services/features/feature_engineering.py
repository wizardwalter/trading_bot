from features.rsi import compute_rsi
from features.vwap import compute_vwap
from features.macd import compute_macd
from features.ema import compute_ema
from features.obv import compute_obv
from features.bollinger import compute_bollinger_bands

def generate_features(df):
    """
    Takes a pandas DataFrame with candle data and returns a DataFrame with all indicators added as features.
    Expects columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    df = df.copy()
    
    # Price-based indicators
    df['RSI'] = compute_rsi(df)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df)
    df['EMA_12'] = compute_ema(df, period=12)
    df['EMA_26'] = compute_ema(df, period=26)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = compute_bollinger_bands(df)

    # Volume-based indicators
    df['OBV'] = compute_obv(df)
    df['VWAP'] = compute_vwap(df)

    # Drop NaNs caused by rolling window operations
    df.dropna(inplace=True)

    return df