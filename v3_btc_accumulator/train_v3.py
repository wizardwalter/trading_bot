from __future__ import annotations
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from config import V3Config

OUT = Path('v3_btc_accumulator/out')
OUT.mkdir(parents=True, exist_ok=True)
RUNS_JSONL = OUT / 'training_runs.jsonl'
LATEST_JSON = OUT / 'latest_v3.json'

@dataclass
class Metrics:
    run_id: str
    started_at: str
    finished_at: str
    timeframe: str
    data_window_start: str
    data_window_end: str
    bars: int
    start_btc: float
    end_btc: float
    delta_btc: float
    btc_accum_pct: float
    trades: int
    trims: int
    buys: int
    win_rate: float
    max_btc_drawdown: float
    fees_paid_btc: float
    slippage_est_btc: float
    core_btc_floor: float
    scalp_tranche_pct: float
    params: dict


def _load_data() -> pd.DataFrame:
    p = Path('data/labeled_10m.csv')
    if not p.exists():
        raise FileNotFoundError('No v3 source data found (expected data/labeled_10m.csv)')
    df = pd.read_csv(p)
    ts_col = 'timestamp' if 'timestamp' in df.columns else ('time' if 'time' in df.columns else None)
    if ts_col is None:
        raise RuntimeError('No timestamp/time column in source data')
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    return df


def _series(df: pd.DataFrame, names: list[str], default=0.0):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def _simulate_window(df: pd.DataFrame, cfg: V3Config, timeframe: str) -> dict:
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
    start_usd = close.iloc[0] * cfg.max_scalp_tranche_pct
    core = start_btc * cfg.core_floor_pct
    btc = start_btc
    usd = start_usd

    trims = buys = trades = wins = 0
    fees_paid_btc = 0.0
    slippage_est_btc = 0.0
    fees = cfg.friction_round_trip_pct / 2.0
    slip = 0.00125
    equity_curve = []

    for i in range(lookback + 5, len(df)):
        p = float(close.iloc[i]);
        sup = float(support.iloc[i]) if pd.notna(support.iloc[i]) else p
        res = float(resistance.iloc[i]) if pd.notna(resistance.iloc[i]) else p
        if p <= 0:
            continue

        trend_up = ema_fast.iloc[i] > ema_slow.iloc[i] and ema_slope.iloc[i] > -0.0005 and p >= vwap.iloc[i]*0.998
        vol_mean = float(vol.iloc[max(0, i-96):i].mean() or 0.0)
        breakout = (p > res * 1.002) and (vol.iloc[i] > vol_mean * 1.8 if vol_mean > 0 else False) and trend_up
        breakdown = (p < sup * 0.998) and (vol.iloc[i] > vol_mean * 1.8 if vol_mean > 0 else False)

        near_res = p >= res * 0.998
        near_sup = p <= sup * 1.002
        abnormal_vol = atr_pct.iloc[i] > max(0.03, atr_pct.iloc[max(0, i-400):i].quantile(0.99))
        conflict = (rsi.iloc[i] > 70 and near_sup) or (rsi.iloc[i] < 30 and near_res)
        if abnormal_vol or conflict:
            equity_curve.append(btc + usd / p)
            continue

        scalp_cap_btc = max(0.0, btc - core)

        if near_res and scalp_cap_btc > 0 and not breakout:
            sell_btc = min(scalp_cap_btc, btc * 0.05)
            if sell_btc > 0:
                exec_price = p * (1 - slip)
                btc -= sell_btc
                gross_usd = sell_btc * exec_price
                fee_usd = gross_usd * fees
                usd += (gross_usd - fee_usd)
                fees_paid_btc += fee_usd / p
                slippage_est_btc += (sell_btc * (p - exec_price)) / p
                trims += 1; trades += 1
                wins += 1
                equity_curve.append(btc + usd / p)
                continue

        if near_sup and usd > (start_usd * 0.02) and trend_up:
            buy_usd = min(usd * 0.35, start_usd * 0.35)
            exec_price = p * (1 + slip)
            fee_usd = buy_usd * fees
            got_btc = (buy_usd - fee_usd) / exec_price
            btc += got_btc
            usd -= buy_usd
            fees_paid_btc += fee_usd / p
            slippage_est_btc += ((exec_price - p) * got_btc) / p
            buys += 1; trades += 1
            equity_curve.append(btc + usd / p)
            continue

        if breakdown and scalp_cap_btc > 0:
            sell_btc = min(scalp_cap_btc, btc * 0.03)
            exec_price = p * (1 - slip)
            btc -= sell_btc
            gross_usd = sell_btc * exec_price
            fee_usd = gross_usd * fees
            usd += (gross_usd - fee_usd)
            fees_paid_btc += fee_usd / p
            slippage_est_btc += (sell_btc * (p - exec_price)) / p
            trims += 1; trades += 1
            equity_curve.append(btc + usd / p)
            continue

        equity_curve.append(btc + usd / p)

    end_btc = btc + (usd / float(close.iloc[-1]))
    delta = end_btc - start_btc
    eq = pd.Series(equity_curve if equity_curve else [start_btc, end_btc])
    roll_max = eq.cummax()
    dd = ((eq - roll_max) / roll_max.replace(0, np.nan)).fillna(0.0)
    max_dd = float(dd.min())

    return {
        'timeframe': timeframe,
        'data_window_start': str(df.index[0]),
        'data_window_end': str(df.index[-1]),
        'bars': int(len(df)),
        'start_btc': start_btc,
        'end_btc': float(end_btc),
        'delta_btc': float(delta),
        'btc_accum_pct': float((delta / start_btc) * 100.0),
        'trades': int(trades),
        'trims': int(trims),
        'buys': int(buys),
        'win_rate': float((wins / trades) if trades else 0.0),
        'max_btc_drawdown': max_dd,
        'fees_paid_btc': float(fees_paid_btc),
        'slippage_est_btc': float(slippage_est_btc),
        'core_btc_floor': float(cfg.core_floor_pct),
        'scalp_tranche_pct': float(cfg.max_scalp_tranche_pct),
    }


def run_once(cfg: V3Config) -> dict:
    started = datetime.now(timezone.utc)
    base = _load_data()

    window_specs = [
        ('1m', 30), ('5m', 60), ('15m', 60), ('1h', 90)
    ]

    runs = []
    for tf, days in window_specs:
        if tf == '1m':
            d = base.resample('1min').last().dropna().tail(days * 24 * 60)
        elif tf == '5m':
            d = base.resample('5min').last().dropna().tail(days * 24 * 12)
        elif tf == '15m':
            d = base.resample('15min').last().dropna().tail(days * 24 * 4)
        else:
            d = base.resample('1h').last().dropna().tail(days * 24)
        if len(d) < 200:
            continue
        runs.append(_simulate_window(d, cfg, tf))

    if not runs:
        raise RuntimeError('No sufficient windows for v3 run')

    agg_end = float(np.mean([r['end_btc'] for r in runs]))
    agg_delta = agg_end - 1.0
    agg = dict(runs[-1])
    agg.update({
        'timeframe': 'aggregate',
        'bars': int(sum(r['bars'] for r in runs)),
        'end_btc': agg_end,
        'delta_btc': agg_delta,
        'btc_accum_pct': agg_delta * 100.0,
        'trades': int(sum(r['trades'] for r in runs)),
        'trims': int(sum(r['trims'] for r in runs)),
        'buys': int(sum(r['buys'] for r in runs)),
        'win_rate': float(np.mean([r['win_rate'] for r in runs])),
        'max_btc_drawdown': float(min(r['max_btc_drawdown'] for r in runs)),
        'fees_paid_btc': float(sum(r['fees_paid_btc'] for r in runs)),
        'slippage_est_btc': float(sum(r['slippage_est_btc'] for r in runs)),
        'windows': runs,
    })

    finished = datetime.now(timezone.utc)
    out = Metrics(
        run_id=str(uuid.uuid4()),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        timeframe=agg['timeframe'],
        data_window_start=agg['data_window_start'],
        data_window_end=agg['data_window_end'],
        bars=agg['bars'],
        start_btc=1.0,
        end_btc=agg['end_btc'],
        delta_btc=agg['delta_btc'],
        btc_accum_pct=agg['btc_accum_pct'],
        trades=agg['trades'],
        trims=agg['trims'],
        buys=agg['buys'],
        win_rate=agg['win_rate'],
        max_btc_drawdown=agg['max_btc_drawdown'],
        fees_paid_btc=agg['fees_paid_btc'],
        slippage_est_btc=agg['slippage_est_btc'],
        core_btc_floor=agg['core_btc_floor'],
        scalp_tranche_pct=agg['scalp_tranche_pct'],
        params=asdict(cfg),
    )
    payload = asdict(out)
    payload['windows'] = runs

    with RUNS_JSONL.open('a') as f:
        f.write(json.dumps(payload) + '\n')
    LATEST_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload

if __name__ == '__main__':
    run_once(V3Config())
