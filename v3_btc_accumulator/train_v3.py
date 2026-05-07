from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from config import V3Config

OUT = Path('v3_btc_accumulator/out')
OUT.mkdir(parents=True, exist_ok=True)

@dataclass
class Metrics:
    bars: int
    trades: int
    trims: int
    buys: int
    start_btc: float
    end_btc: float
    delta_btc: float
    btc_accum_pct: float
    core_btc: float


def _load_data() -> pd.DataFrame:
    p = Path('data/labeled_10m.csv')
    if p.exists():
        df = pd.read_csv(p)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.sort_values('timestamp').set_index('timestamp')
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
            df = df.sort_values('time').set_index('time')
        return df
    raise FileNotFoundError('No v3 source data found (expected data/labeled_10m.csv)')


def _series(df: pd.DataFrame, names: list[str], default=0.0):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def run_once(cfg: V3Config) -> Metrics:
    df = _load_data().tail(6000).copy()
    close = _series(df, ['close', 'Close'])
    high = _series(df, ['high', 'High'], default=np.nan).fillna(close)
    low = _series(df, ['low', 'Low'], default=np.nan).fillna(close)
    vol = _series(df, ['volume', 'Volume'], default=0.0)

    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=80, adjust=False).mean()
    ema_slope = ema_slow.pct_change(6).fillna(0.0)
    rsi = _series(df, ['rsi'], default=50.0)
    atr = (high - low).rolling(14).mean().fillna((high - low).mean())
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vwap = (close * vol).rolling(96).sum() / vol.rolling(96).sum().replace(0, np.nan)
    vwap = vwap.fillna(close)

    lookback = cfg.support_resistance_lookback
    support = low.rolling(lookback).min().shift(1)
    resistance = high.rolling(lookback).max().shift(1)

    start_btc = 1.0
    start_usd = close.iloc[0] * 0.2
    core = start_btc * cfg.core_floor_pct
    btc = start_btc
    usd = start_usd

    trims = buys = trades = 0
    fees = cfg.friction_round_trip_pct / 2.0

    for i in range(lookback + 5, len(df)):
        p = float(close.iloc[i])
        sup = float(support.iloc[i]) if pd.notna(support.iloc[i]) else p
        res = float(resistance.iloc[i]) if pd.notna(resistance.iloc[i]) else p
        if p <= 0:
            continue

        trend_up = ema_fast.iloc[i] > ema_slow.iloc[i] and ema_slope.iloc[i] > -0.0005
        breakout = (p > res * 1.002) and (vol.iloc[i] > vol.iloc[max(0, i-96):i].mean() * 1.8) and trend_up
        breakdown = (p < sup * 0.998) and (vol.iloc[i] > vol.iloc[max(0, i-96):i].mean() * 1.8)

        near_res = p >= res * 0.998
        near_sup = p <= sup * 1.002
        abnormal_vol = atr_pct.iloc[i] > max(0.03, atr_pct.iloc[max(0, i-400):i].quantile(0.99))
        conflict = (rsi.iloc[i] > 70 and near_sup) or (rsi.iloc[i] < 30 and near_res)
        if abnormal_vol or conflict:
            continue

        scalp_cap_btc = max(0.0, btc - core)

        if near_res and scalp_cap_btc > 0 and not breakout:
            sell_btc = min(scalp_cap_btc, btc * 0.05)
            if sell_btc > 0:
                btc -= sell_btc
                usd += sell_btc * p * (1 - fees)
                trims += 1
                trades += 1
                continue

        if near_sup and usd > (start_usd * 0.02) and trend_up:
            buy_usd = min(usd * 0.35, start_usd * 0.35)
            got_btc = (buy_usd * (1 - fees)) / p
            btc += got_btc
            usd -= buy_usd
            buys += 1
            trades += 1
            continue

        if breakdown and scalp_cap_btc > 0:
            sell_btc = min(scalp_cap_btc, btc * 0.03)
            btc -= sell_btc
            usd += sell_btc * p * (1 - fees)
            trims += 1
            trades += 1

    end_btc = btc + (usd / float(close.iloc[-1]))
    delta = end_btc - start_btc
    m = Metrics(
        bars=len(df), trades=trades, trims=trims, buys=buys,
        start_btc=start_btc, end_btc=end_btc, delta_btc=delta,
        btc_accum_pct=(delta / start_btc) * 100.0, core_btc=core,
    )
    OUT.joinpath('latest_v3.json').write_text(json.dumps(asdict(m), indent=2))
    print(json.dumps(asdict(m), indent=2))
    return m

if __name__ == '__main__':
    cfg = V3Config()
    run_once(cfg)
