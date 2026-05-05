

import pandas as pd

def generate_labels(df, target_col='close', threshold=0.025, future_window=10):
    """
    Adds a 'label' column to the DataFrame:
    1 = Buy (price increases >= threshold within future_window)
    -1 = Sell (price drops >= threshold within future_window)
    0 = Hold (otherwise)

    Parameters:
    - df: DataFrame with at least a 'close' column
    - target_col: which price column to base movement on
    - threshold: percentage change to qualify as signal (e.g., 0.025 = 2.5%)
    - future_window: number of rows (candles) to look ahead
    """
    df = df.copy()
    labels = []

    for i in range(len(df) - future_window):
        current_price = df.iloc[i][target_col]
        future_prices = df.iloc[i+1:i+1+future_window][target_col]

        max_future = future_prices.max()
        min_future = future_prices.min()

        if (max_future - current_price) / current_price >= threshold:
            labels.append(1)  # Buy
        elif (current_price - min_future) / current_price >= threshold:
            labels.append(-1)  # Sell
        else:
            labels.append(0)  # Hold

    labels += [None] * future_window
    df['label'] = labels
    df.dropna(inplace=True)

    return df