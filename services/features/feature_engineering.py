from services.features.rsi import compute_rsi
from services.features.vwap import compute_vwap
from services.features.macd import compute_macd
from services.features.ema import compute_ema
from services.features.obv import compute_obv
from services.features.bollinger import compute_bollinger_bands


def generate_features(df):
    df = df.copy()

    df['RSI'] = compute_rsi(df)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df)
    df['EMA_12'] = compute_ema(df, period=12)
    df['EMA_26'] = compute_ema(df, period=26)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = compute_bollinger_bands(df)
    df['OBV'] = compute_obv(df)
    df['VWAP'] = compute_vwap(df)

    df.dropna(inplace=True)
    return df
