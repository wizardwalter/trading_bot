from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from services.alpaca_candles import fetch_crypto_bars
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from discord.notify import send_training_update
from data.database import get_all_candles

OUT_DIR = Path("data/backtests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_MODE = (os.getenv("TRAINING_MODE") or "auto").strip().lower()
if TRAINING_MODE not in {"auto", "classic", "neural"}:
    TRAINING_MODE = "auto"
TRAINING_LABEL = (os.getenv("TRAINING_LABEL") or os.getenv("TRAINING_VARIANT") or "").strip()


def _period_to_days(period: str) -> int:
    p = str(period).strip().lower()
    if p.endswith("d"):
        return max(int(p[:-1]), 1)
    if p.endswith("mo"):
        return max(int(p[:-2]) * 30, 30)
    if p.endswith("y"):
        return max(int(p[:-1]) * 365, 365)
    return 60


def _download_from_db(symbol: str, interval: str, period: str) -> pd.DataFrame:
    df = get_all_candles(symbol, interval)
    if df.empty:
        raise RuntimeError(f"DB has no candles for {symbol} {interval}")

    # Normalize schema from DB helper to expected OHLCV names.
    out = df.copy()
    if "open" in out.columns:
        out.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

    # Keep recent window requested by training period.
    days = _period_to_days(period)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out[out["timestamp"] >= cutoff]
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out[out.index >= cutoff]

    cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise RuntimeError(f"DB candles missing columns: {missing}")

    out = out[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna().copy()
    if out.empty:
        raise RuntimeError(f"DB candles empty for {symbol} {interval} after period filter")
    return out


def download(symbol: str = "BTC-USD", interval: str = "5m", period: str = "60d", retries: int = 4) -> pd.DataFrame:
    """Fetch training market data.

    Priority:
    1) DB candles (Massive/Polygon historical backfill)
    2) Alpaca crypto data for BTC (execution-aligned)
    3) yfinance fallback for robustness
    """
    source_pref = (os.getenv("TRAINING_DATA_SOURCE") or "db-first").strip().lower()

    if source_pref in {"db", "db-first", "auto"}:
        try:
            db_df = _download_from_db(symbol=symbol, interval=interval, period=period)
            print(f"Using DB market data for training: symbol={symbol}, interval={interval}, rows={len(db_df)}")
            return db_df
        except Exception as e:
            print(f"DB market data unavailable for training, falling back: {e}")
            if source_pref == "db":
                raise

    # Prefer Alpaca for BTC training so data distribution better matches execution.
    use_alpaca_first = symbol.upper() in {"BTC-USD", "BTC/USD", "BTCUSD"}
    if use_alpaca_first:
        tf_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour"}
        timeframe = tf_map.get(interval, "5Min")
        lookback_days = 60
        if period.endswith("d"):
            try:
                lookback_days = int(period[:-1])
            except Exception:
                lookback_days = 60
        lookback_days = max(lookback_days, int(os.getenv("TRAINING_MIN_LOOKBACK_DAYS", "180")))

        try:
            alpaca_symbol = "BTC/USD"
            df = fetch_crypto_bars(symbol=alpaca_symbol, timeframe=timeframe, lookback_days=lookback_days)
            cleaned = df.dropna().copy()
            if not cleaned.empty:
                print(f"Using Alpaca market data for training: symbol={alpaca_symbol}, timeframe={timeframe}, rows={len(cleaned)}")
                return cleaned
        except Exception as e:
            print(f"Alpaca market data unavailable, falling back to yfinance: {e}")

    # yfinance intraday windows are capped (e.g., 5m ~= last 60 days), so clamp fallback period.
    yf_period = period
    intraday_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}
    if interval in intraday_intervals and str(period).endswith("d"):
        try:
            if int(str(period)[:-1]) > 60:
                yf_period = "60d"
        except Exception:
            yf_period = "60d"

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbol, interval=interval, period=yf_period, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            cleaned = df.dropna().copy()
            if cleaned.empty:
                raise ValueError(f"No market data returned for {symbol}")
            if yf_period != period:
                print(f"Using yfinance fallback period={yf_period} for interval={interval} (requested {period})")
            return cleaned
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.6 * attempt)

    raise RuntimeError(f"download failed for {symbol} after {retries} attempts: {last_err}")


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Base features
    out["ema_fast"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=26, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    out["rsi"] = 100 - (100 / (1 + rs))

    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_20"] = out["Close"].pct_change(20)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["volatility"] = out["hl_spread"].rolling(20).mean().fillna(0.002)

    # Additional features: volume, volume changes, multi-timeframe
    out["ema_ratio"] = ((out["Close"] / out["ema_slow"]) - 1).clip(-0.2, 0.2)
    out["macd_hist"] = ((out["ema_fast"] - out["ema_slow"]) / out["Close"]).clip(-0.2, 0.2)
    out["price_momentum"] = out["Close"].pct_change().rolling(6).mean().clip(-0.05, 0.05)

    vol_growth = out["Volume"].pct_change(36).replace([np.inf, -np.inf], np.nan)
    out["volume_trend"] = vol_growth.ewm(span=24, adjust=False).mean().clip(-4, 4).fillna(0.0)

    volume = out["Volume"].ffill()
    vol_mean = volume.rolling(96).mean()
    vol_std = volume.rolling(96).std().replace(0, np.nan)
    volume_z = ((volume - vol_mean) / vol_std).clip(-3, 3).fillna(0.0)
    out["volume_z"] = volume_z

    high_low = out["High"] - out["Low"]
    prev_close = out["Close"].shift(1)
    true_range = pd.concat([
        high_low,
        (out["High"] - prev_close).abs(),
        (out["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(48).mean()
    out["atr_pct"] = (atr / out["Close"]).fillna((high_low / out["Close"]).rolling(12).mean()).fillna(0.004)

    intraday_position = ((out["Close"] - out["Low"]) / (high_low.replace(0, np.nan))).clip(0, 1) - 0.5
    out["range_score"] = intraday_position.rolling(12).mean().fillna(0.0)

    # Multi-timeframe signals: add moving averages with longer windows
    out["ema_fast_2"] = out["Close"].ewm(span=24, adjust=False).mean()
    out["ema_slow_2"] = out["Close"].ewm(span=52, adjust=False).mean()

    # Multi-timeframe context proxies (derived from 5m stream)
    out["ret_1h"] = out["Close"].pct_change(12)
    out["ret_4h"] = out["Close"].pct_change(48)
    out["mtf_trend_1h"] = (out["ret_1h"] * 8.0).clip(-1, 1)
    out["mtf_trend_4h"] = (out["ret_4h"] * 4.0).clip(-1, 1)

    trend = ((out["ema_fast"] - out["ema_slow"]) / out["Close"]).clip(-1, 1) * 220
    trend = trend.clip(-1, 1)
    m20 = (out["ret_20"] * 25).clip(-1, 1)
    m3 = (out["ret_3"] * 35).clip(-1, 1)

    rsi_comp = np.where(out["rsi"] < 33, 0.8, np.where(out["rsi"] > 70, -0.8, 0.0))
    volume_bias = np.tanh(out["volume_z"].clip(-3, 3) / 1.8)
    range_component = out["range_score"].clip(-1, 1)

    # Regime/time features
    if isinstance(out.index, pd.DatetimeIndex):
        hour = out.index.hour
        dow = out.index.dayofweek
    else:
        hour = pd.Series(0, index=out.index)
        dow = pd.Series(0, index=out.index)
    out["hour_sin"] = np.sin((2 * np.pi * hour) / 24.0)
    out["hour_cos"] = np.cos((2 * np.pi * hour) / 24.0)
    out["dow_sin"] = np.sin((2 * np.pi * dow) / 7.0)
    out["dow_cos"] = np.cos((2 * np.pi * dow) / 7.0)

    trend_abs = trend.abs()
    out["regime_trend"] = (trend_abs > trend_abs.quantile(0.70)).astype(float)
    out["regime_chop"] = (trend_abs < trend_abs.quantile(0.35)).astype(float)
    out["regime_high_vol"] = (out["atr_pct"] > out["atr_pct"].quantile(0.80)).astype(float)

    out["score"] = (
        0.28 * trend
        + 0.12 * m20
        + 0.05 * m3
        + 0.32 * rsi_comp
        + 0.10 * out["mtf_trend_1h"].fillna(0.0)
        + 0.08 * out["mtf_trend_4h"].fillna(0.0)
    )
    out["score_raw"] = out["score"]
    out["trend"] = trend
    out["m3"] = m3
    out["m20"] = m20
    out["volume_bias"] = volume_bias
    out["range_score"] = range_component

    out = out.dropna().copy()
    return out



@dataclass
class Metrics:
    threshold: float
    trades: int
    win_rate: float
    win_loss_ratio: float
    profit_factor: float
    expectancy: float
    total_return: float
    sharpe_like: float
    max_drawdown: float
    consistency_score: float
    do_not_trade: bool


def _target_position(df: pd.DataFrame, threshold: float) -> np.ndarray:
    score = df["score"].values
    rsi = df["rsi"].values
    m3 = df["m3"].values
    m20 = df["m20"].values
    trend = df["trend"].values
    vol = df["volatility"].values
    atr_pct = df["atr_pct"].values
    volume_bias = df["volume_bias"].values
    range_score = df["range_score"].values

    if "score_ml" in df.columns:
        score_ml = df["score_ml"].values
    else:
        score_ml = score
    if "score_ml_raw" in df.columns:
        score_ml_raw = df["score_ml_raw"].values
    else:
        score_ml_raw = score_ml

    exit_cooldown_bars = 3
    flip_cooldown_bars = 1

    buy_threshold = threshold + np.clip((vol - 0.01) * 8.0, 0.0, 0.10)
    sell_threshold = -threshold - np.clip((vol - 0.01) * 8.0, 0.0, 0.10)

    atr_rel = np.maximum(0.0, atr_pct - np.nanpercentile(atr_pct, 70))
    atr_boost = np.clip(atr_rel * 12.0, 0.0, 0.12)
    buy_threshold = buy_threshold + atr_boost
    sell_threshold = sell_threshold - atr_boost

    # Regime-adaptive thresholding: tighter in chop/high-vol, looser in strong trend.
    trend_abs = np.abs(trend)
    chop_regime = trend_abs < np.nanpercentile(trend_abs, 35)
    strong_trend = trend_abs > np.nanpercentile(trend_abs, 75)
    extreme_vol = atr_pct > np.nanpercentile(atr_pct, 97)

    buy_threshold = buy_threshold + np.where(chop_regime, 0.02, 0.0) + np.where(extreme_vol, 0.03, 0.0)
    sell_threshold = sell_threshold - np.where(chop_regime, 0.02, 0.0) - np.where(extreme_vol, 0.03, 0.0)
    buy_threshold = buy_threshold - np.where(strong_trend & (trend > 0), 0.015, 0.0)
    sell_threshold = sell_threshold + np.where(strong_trend & (trend < 0), 0.015, 0.0)

    long_relief = (
        np.clip(volume_bias - 0.2, 0.0, 1.0) * 0.02
        + np.clip(range_score, 0.0, 0.4) * 0.02
    )
    short_relief = (
        np.clip(-0.2 - volume_bias, 0.0, 1.0) * 0.02
        + np.clip(-range_score, 0.0, 0.4) * 0.02
    )
    buy_threshold = buy_threshold - long_relief
    sell_threshold = sell_threshold + short_relief

    # Tuned thresholds: slightly later overbought filtering and deeper oversold allowance.
    overbought = rsi > 71
    oversold = rsi < 19

    # Reduce entries during volatile chop unless directional conviction is strong.
    vol_guard = vol <= np.nanpercentile(vol, 85)
    high_vol_regime = atr_pct >= np.nanpercentile(atr_pct, 94)
    high_vol_penalty_long = high_vol_regime & (volume_bias < 0.15)
    high_vol_penalty_short = high_vol_regime & (volume_bias > -0.15)

    range_ok_long = range_score > -0.05
    range_ok_short = range_score < 0.02

    ml_bias = max(0.02, float(threshold) * 0.22)
    ml_relief = np.clip(volume_bias * 0.01, -0.02, 0.02)
    long_ml_gate = score_ml > (ml_bias - ml_relief)
    short_ml_gate = score_ml < (-ml_bias - ml_relief)
    long_override = score > (buy_threshold + 0.05)
    short_override = score < (sell_threshold - 0.05)

    mtf_1h = df["mtf_trend_1h"].values if "mtf_trend_1h" in df.columns else np.zeros(len(df))
    mtf_4h = df["mtf_trend_4h"].values if "mtf_trend_4h" in df.columns else np.zeros(len(df))

    bullish_confirmation = (trend > -0.01) & (m20 > -0.05) & (m3 > -0.13) & (mtf_1h > -0.15) & (mtf_4h > -0.20)
    bearish_confirmation = (trend < 0.01) & (m20 < 0.05) & (m3 < 0.13) & (mtf_1h < 0.15) & (mtf_4h < 0.20)

    ml_raw_abs = np.abs(score_ml_raw)
    low_confidence = ml_raw_abs < np.maximum(0.06, threshold * 0.20)
    weak_trend = np.abs(trend) < 0.05
    volatility_block = atr_pct > np.nanpercentile(atr_pct, 97)

    if "meta_take_prob" in df.columns:
        meta_take_prob = np.clip(df["meta_take_prob"].values, 0.001, 0.999)
    else:
        meta_take_prob = np.clip((score_ml + 1.0) * 0.5, 0.001, 0.999)
    min_take_prob = np.where(high_vol_regime, 0.62, 0.55)
    min_take_prob = np.where(np.abs(trend) > 0.25, min_take_prob - 0.03, min_take_prob)
    meta_skip = meta_take_prob < min_take_prob

    do_not_trade_filter = low_confidence | (weak_trend & (~vol_guard)) | volatility_block | meta_skip

    long_entry = (
        (score > buy_threshold)
        & bullish_confirmation
        & (~overbought)
        & (vol_guard | (trend > 0.20) | (volume_bias > 0.35))
        & range_ok_long
        & (~high_vol_penalty_long)
        & (long_ml_gate | long_override)
        & (~do_not_trade_filter)
    )
    short_entry = (
        (score < sell_threshold)
        & bearish_confirmation
        & (~oversold)
        & (vol_guard | (trend < -0.20) | (volume_bias < -0.35))
        & range_ok_short
        & (~high_vol_penalty_short)
        & (short_ml_gate | short_override)
        & (~do_not_trade_filter)
    )

    long_exit = (
        (score < -0.011)
        | (overbought & (m3 < 0.08))
        | ((trend < -0.08) & (m3 < -0.15))
        | (high_vol_regime & (range_score < -0.02))
        | (range_score < -0.18)
    )
    short_exit = (
        (score > 0.008)
        | (oversold & (m3 > -0.08))
        | ((trend > 0.08) & (m3 > 0.15))
        | (high_vol_regime & (range_score > 0.02))
        | (range_score > 0.18)
    )

    position = np.zeros(len(df), dtype=np.int8)
    state = 0
    cooldown = 0

    for i in range(len(df)):
        if cooldown > 0:
            cooldown -= 1

        if state == 0:
            if cooldown == 0 and long_entry[i]:
                state = 1
            elif cooldown == 0 and short_entry[i]:
                state = -1
        elif state == 1:
            if long_exit[i]:
                state = 0
                cooldown = exit_cooldown_bars + (2 if high_vol_regime[i] else 0)
            elif short_entry[i]:
                state = -1
                cooldown = flip_cooldown_bars
        elif state == -1:
            if short_exit[i]:
                state = 0
                cooldown = exit_cooldown_bars + (2 if high_vol_regime[i] else 0)
            elif long_entry[i]:
                state = 1
                cooldown = flip_cooldown_bars

        position[i] = state

    return position


def simulate(
    df: pd.DataFrame,
    threshold: float,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
    latency_ms: float = 150.0,
) -> Metrics:
    if df.empty:
        return Metrics(
            threshold=float(threshold),
            trades=0,
            win_rate=0.0,
            win_loss_ratio=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            total_return=0.0,
            sharpe_like=0.0,
            max_drawdown=0.0,
            consistency_score=0.0,
            do_not_trade=True,
        )

    fee = fee_bps / 10_000
    slippage = slippage_bps / 10_000

    position = _target_position(df, threshold).astype(float)

    # Position applies from the next bar onward, plus latency delay approximation.
    position = pd.Series(position).shift(1).fillna(0)
    delay_bars = max(0, int(np.ceil(float(latency_ms) / (5 * 60 * 1000))))
    if delay_bars > 0:
        position = position.shift(delay_bars).fillna(0)
    position = position.values

    rets = df["Close"].pct_change().fillna(0).values
    strat = position * rets

    turns = np.abs(np.diff(np.r_[0, position]))
    costs = turns * (fee + slippage)
    strat = strat - costs

    eq = (1 + pd.Series(strat)).cumprod()
    total_return = float(eq.iloc[-1] - 1)

    # Compute trade-level PnL over contiguous non-zero position regimes.
    trade_rets: list[float] = []
    active = False
    active_side = 0.0
    acc = 0.0
    for i in range(len(position)):
        side = position[i]
        r = float(strat[i])

        if not active and side != 0:
            active = True
            active_side = side
            acc = r
            continue

        if active:
            if side == 0:
                acc += r
                trade_rets.append(acc)
                active = False
                active_side = 0.0
                acc = 0.0
            elif side != active_side:
                # Side flip: close prior trade and start the new one on the same bar.
                trade_rets.append(acc)
                active_side = side
                acc = r
            else:
                acc += r

    if active:
        trade_rets.append(acc)

    trades = len(trade_rets)
    wins = [x for x in trade_rets if x > 0]
    losses = [x for x in trade_rets if x < 0]
    win_rate = float(np.mean(np.array(trade_rets) > 0)) if trades else 0.0
    expectancy = float(np.mean(trade_rets)) if trades else 0.0

    gross_profit = float(np.sum(wins)) if wins else 0.0
    gross_loss = abs(float(np.sum(losses))) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (9.99 if gross_profit > 0 else 0.0)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = abs(float(np.mean(losses))) if losses else 0.0
    win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (9.99 if avg_win > 0 else 0.0)

    vol = float(pd.Series(strat).std())
    sharpe_like = float((pd.Series(strat).mean() / vol) * np.sqrt(252 * 24 * 12)) if vol > 0 else 0.0

    rolling_max = eq.cummax()
    dd = (eq / rolling_max) - 1
    max_drawdown = float(dd.min())

    # consistency across time buckets (higher is better)
    buckets = np.array_split(np.array(strat, dtype=float), 6)
    bucket_means = [float(np.mean(b)) for b in buckets if len(b) > 20]
    if bucket_means:
        consistency_score = float(np.mean(bucket_means) - (np.std(bucket_means) * 0.5))
    else:
        consistency_score = 0.0

    do_not_trade = bool(
        trades < 8
        or max_drawdown < -0.08
        or profit_factor < 1.02
        or consistency_score < -0.0002
    )

    return Metrics(
        threshold=float(threshold),
        trades=trades,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        expectancy=expectancy,
        total_return=total_return,
        sharpe_like=sharpe_like,
        max_drawdown=max_drawdown,
        consistency_score=consistency_score,
        do_not_trade=do_not_trade,
    )


def _score_metrics(m: Metrics) -> float:
    trades_penalty = 0.0
    if m.trades < 8:
        trades_penalty = 0.35
    elif m.trades < 12:
        trades_penalty = 0.18
    elif m.trades > 36:
        trades_penalty = 0.20

    risk_penalty = 0.12 if m.max_drawdown < -0.11 else (0.05 if m.max_drawdown < -0.08 else 0.0)
    expectancy_penalty = 0.12 if (m.expectancy < 0 and m.win_rate < 0.4) else 0.0
    dnt_penalty = 0.45 if m.do_not_trade else 0.0

    return (
        (m.total_return * 3.0)
        + (m.expectancy * 12.0)
        + (m.sharpe_like * 0.05)
        + (m.win_rate * 0.20)
        + (m.profit_factor * 0.12)
        + (m.win_loss_ratio * 0.08)
        + (m.consistency_score * 40.0)
        + (m.max_drawdown * 0.55)
        - trades_penalty
        - risk_penalty
        - expectancy_penalty
        - dnt_penalty
    )


def _regime_slices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty or "atr_pct" not in df.columns or "trend" not in df.columns:
        return {}

    atr = df["atr_pct"].fillna(df["atr_pct"].median())
    trend_abs = df["trend"].abs().fillna(0)

    low_q = float(atr.quantile(0.35))
    high_q = float(atr.quantile(0.75))

    return {
        "low_vol": df[atr <= low_q],
        "mid_vol": df[(atr > low_q) & (atr < high_q)],
        "high_vol": df[atr >= high_q],
        "trend": df[trend_abs >= 0.22],
        "chop": df[trend_abs <= 0.08],
    }


def _regime_penalty(df: pd.DataFrame, threshold: float) -> float:
    slices = _regime_slices(df)
    if not slices:
        return 0.0

    penalties = 0.0
    covered = 0
    for name, sdf in slices.items():
        if len(sdf) < 180:
            continue
        m = simulate(sdf, threshold)
        covered += 1
        # Penalize regime fragility (especially negative expectancy and deep DD).
        if m.expectancy < 0:
            penalties += abs(m.expectancy) * 10.0
        if m.total_return < 0:
            penalties += abs(m.total_return) * 1.2
        if m.max_drawdown < -0.08:
            penalties += abs(m.max_drawdown + 0.08) * 0.8
        if m.trades < 3:
            penalties += 0.03

    if covered == 0:
        return 0.0
    return penalties / covered


def _neighbor_instability_penalty(df: pd.DataFrame, threshold: float, base_return: float) -> float:
    deltas = (-0.01, -0.005, 0.005, 0.01)
    neighbor_returns: list[float] = []
    for delta in deltas:
        neighbor_th = float(threshold + delta)
        if neighbor_th <= 0.02 or neighbor_th >= 0.90:
            continue
        neighbor_returns.append(simulate(df, neighbor_th).total_return)

    if not neighbor_returns:
        return 0.0

    mean_diff = float(np.mean([abs(base_return - r) for r in neighbor_returns]))
    spread = float(np.std(neighbor_returns))
    return (mean_diff * 1.6) + (max(spread - 0.003, 0.0) * 1.1)


def _walk_forward_summary(df: pd.DataFrame, threshold: float, train_ratio: float = 0.8) -> dict[str, float]:
    n = len(df)
    if n < 1200:
        return {"folds": 0.0, "pass_ratio": 0.0, "avg_return": 0.0, "avg_pf": 0.0, "penalty": 0.25}

    train_len = int(n * train_ratio)
    test_len = max(240, int(n * 0.10))
    step = max(120, int(test_len * 0.5))

    start = 0
    fold_metrics: list[Metrics] = []
    while (start + train_len + test_len) <= n:
        test_slice = df.iloc[start + train_len : start + train_len + test_len]
        fold_metrics.append(simulate(test_slice, threshold))
        start += step
        if len(fold_metrics) >= 8:
            break

    if not fold_metrics:
        return {"folds": 0.0, "pass_ratio": 0.0, "avg_return": 0.0, "avg_pf": 0.0, "penalty": 0.25}

    passes = [1 for m in fold_metrics if (m.total_return > 0 and m.profit_factor >= 1.0 and not m.do_not_trade)]
    pass_ratio = float(sum(passes) / len(fold_metrics))
    avg_return = float(np.mean([m.total_return for m in fold_metrics]))
    avg_pf = float(np.mean([m.profit_factor for m in fold_metrics]))
    penalty = max(0.0, (0.65 - pass_ratio)) + (0.15 if avg_pf < 1.0 else 0.0)

    return {
        "folds": float(len(fold_metrics)),
        "pass_ratio": pass_ratio,
        "avg_return": avg_return,
        "avg_pf": avg_pf,
        "penalty": float(penalty),
    }


def pick_best(train_df: pd.DataFrame) -> Metrics:
    n = len(train_df)
    fold_start = int(n * 0.30)
    fold_ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]
    recent_start = int(n * 0.55)
    recent_window = train_df.iloc[recent_start:] if (n - recent_start) >= 240 else None

    def evaluate_candidates(candidates: np.ndarray) -> list[tuple[float, Metrics, dict[str, float]]]:
        scored_local: list[tuple[float, Metrics, dict[str, float]]] = []
        for th in candidates:
            full_m = simulate(train_df, float(th))
            instability_penalty = _neighbor_instability_penalty(train_df, float(th), full_m.total_return)
            trade_scarcity_penalty = max(0.0, (6 - min(full_m.trades, 6)) * 0.03)

            fold_metrics: list[Metrics] = []
            prev = fold_start
            for end in fold_ends:
                fold = train_df.iloc[prev:end]
                if len(fold) < 200:
                    continue
                fold_metrics.append(simulate(fold, float(th)))
                prev = end

            fold_scores = [_score_metrics(m) for m in fold_metrics]
            if fold_scores:
                base_weights = np.array([1.0, 1.2, 1.5, 1.8], dtype=float)
                w = base_weights[-len(fold_scores) :]
                cv_score = float(np.average(fold_scores, weights=w))
            else:
                cv_score = -1.0

            fold_returns = [m.total_return for m in fold_metrics]
            fold_trade_counts = [m.trades for m in fold_metrics]
            fold_positive = sum(1 for r in fold_returns if r > 0)
            fold_count = len(fold_metrics)
            median_fold_return = float(np.median(fold_returns)) if fold_returns else 0.0
            worst_fold_return = float(min(fold_returns)) if fold_returns else 0.0
            ret_std = float(np.std(fold_returns)) if fold_returns else 0.0
            pessimistic_return = median_fold_return - (abs(min(worst_fold_return, 0.0)) * 0.55) - (ret_std * 0.65)

            recent_penalty = 0.0
            recent_bonus = 0.0
            recent_last_return = 0.0
            if recent_window is not None and len(recent_window) >= 200:
                recent_m = simulate(recent_window, float(th))
                recent_last_return = recent_m.total_return
                if recent_m.total_return < 0:
                    recent_penalty += abs(recent_m.total_return) * 0.9
                else:
                    recent_bonus += min(recent_m.total_return * 0.6, 0.10)
                if recent_m.max_drawdown < -0.09:
                    recent_penalty += abs(recent_m.max_drawdown + 0.055) * 0.6
                if recent_m.trades < 4:
                    recent_penalty += 0.04

            if fold_metrics:
                worst_fold_dd = float(min(m.max_drawdown for m in fold_metrics))
                recent_fold_return = float(fold_metrics[-1].total_return)
                avg_fold_trades = float(np.mean(fold_trade_counts))
                stability_penalty = (
                    (ret_std * 1.8)
                    + (abs(min(worst_fold_return, 0.0)) * 0.7)
                    + (abs(min(median_fold_return, 0.0)) * 0.5)
                    + (abs(min(worst_fold_dd + 0.15, 0.0)) * 0.3)
                    + (abs(min(recent_fold_return, 0.0)) * 0.6)
                    + (0.08 if avg_fold_trades < 2.0 else 0.0)
                )
            else:
                avg_fold_trades = 0.0
                stability_penalty = 0.45 + recent_penalty - recent_bonus
                recent_fold_return = 0.0

            regime_penalty = _regime_penalty(train_df, float(th))
            wfv = _walk_forward_summary(train_df, float(th))

            score = (
                (_score_metrics(full_m) * 0.30)
                + (cv_score * 0.55)
                + (wfv["avg_return"] * 2.2)
                + (wfv["avg_pf"] * 0.08)
                - stability_penalty
                - (instability_penalty * 0.75)
                - (regime_penalty * 0.85)
                - (wfv["penalty"] * 0.9)
                - recent_penalty
                + recent_bonus
                - trade_scarcity_penalty
            )
            extras = {
                "fold_positive": float(fold_positive),
                "fold_count": float(fold_count),
                "median_fold_return": float(median_fold_return),
                "worst_fold_return": float(worst_fold_return),
                "pessimistic_return": float(pessimistic_return),
                "avg_fold_trades": float(avg_fold_trades),
                "recent_last_return": float(recent_last_return),
                "wf_folds": float(wfv["folds"]),
                "wf_pass_ratio": float(wfv["pass_ratio"]),
                "wf_avg_return": float(wfv["avg_return"]),
                "wf_avg_pf": float(wfv["avg_pf"]),
            }
            scored_local.append((score, full_m, extras))

        return scored_local

    coarse = np.arange(0.05, 0.71, 0.01)
    scored = evaluate_candidates(coarse)

    top_seed = sorted(scored, key=lambda x: x[0], reverse=True)[:4]
    centers = [m.threshold for _, m, _ in top_seed]
    refined_vals: set[float] = set()
    for c in centers:
        for th in np.arange(max(0.03, c - 0.03), min(0.80, c + 0.03) + 1e-9, 0.002):
            refined_vals.add(round(float(th), 3))
    refined = np.array(sorted(refined_vals), dtype=float)
    if len(refined) > 0:
        scored.extend(evaluate_candidates(refined))

    best_by_th: dict[float, tuple[float, Metrics, dict[str, float]]] = {}
    for score, m, extras in scored:
        th = round(m.threshold, 3)
        prev = best_by_th.get(th)
        if prev is None or score > prev[0]:
            best_by_th[th] = (score, m, extras)
    deduped = list(best_by_th.values())

    def _robust_key(item: tuple[float, Metrics, dict[str, float]]) -> float:
        score, m, extras = item
        return (
            score
            + (extras["pessimistic_return"] * 5.0)
            + (extras["median_fold_return"] * 3.0)
            - (abs(m.max_drawdown + 0.07) * 0.45)
            - (abs(m.trades - 16) * 0.03)
        )

    viable = [
        (score, m, extras)
        for score, m, extras in deduped
        if (
            m.total_return > 0
            and m.max_drawdown >= -0.11
            and m.trades >= 8
            and m.profit_factor >= 1.03
            and (not m.do_not_trade)
            and extras["fold_count"] >= 3
            and extras["fold_positive"] >= max(2.0, extras["fold_count"] - 1)
            and extras["median_fold_return"] > 0
            and extras["pessimistic_return"] > -0.005
            and extras.get("wf_folds", 0.0) >= 3
            and extras.get("wf_pass_ratio", 0.0) >= 0.55
        )
    ]

    if viable:
        return max(viable, key=_robust_key)[1]

    trade_balanced = [
        (score, m, extras)
        for score, m, extras in deduped
        if (
            m.total_return > 0
            and m.trades >= 9
            and m.max_drawdown >= -0.17
            and m.profit_factor >= 1.0
            and extras["median_fold_return"] > -0.01
        )
    ]
    if trade_balanced:
        sweet_spot = 18
        return max(
            trade_balanced,
            key=lambda x: (
                x[0],
                x[1].total_return * 1.1,
                -abs(x[1].trades - sweet_spot),
                x[1].win_rate,
                -abs(x[1].max_drawdown + 0.08),
                x[2]["median_fold_return"],
            ),
        )[1]

    activity_guard = [
        (score, m, extras)
        for score, m, extras in deduped
        if m.total_return > 0 and m.trades >= 4
    ]
    if activity_guard:
        return max(
            activity_guard,
            key=lambda x: (
                x[0],
                -abs(x[1].trades - 12),
                x[1].total_return,
                x[1].sharpe_like,
                -abs(x[1].max_drawdown + 0.1),
            ),
        )[1]

    return max(
        deduped,
        key=lambda x: (x[0], x[1].total_return, x[1].max_drawdown, x[1].sharpe_like, -x[1].threshold),
    )[1]


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.2%}"
    except Exception:
        return "n/a"


def _build_webhook_metrics(symbol: str, interval: str, period: str, mode: str, train_return: float | None, test_return: float | None, test_win_rate: float | None, test_max_drawdown: float | None, error: str | None = None) -> str:
    msg = (
        f"{symbol} {interval} {period} {mode} | "
        f"train_return={_fmt_pct(train_return)} | "
        f"test_return={_fmt_pct(test_return)} | "
        f"test_win_rate={_fmt_pct(test_win_rate)} | "
        f"test_max_drawdown={_fmt_pct(test_max_drawdown)}"
    )
    if error:
        msg += f" | error={error[:120]}"
    return msg


from ml.neural_model import NeuralSequenceModel, SequenceDataset, train_neural_model, neural_inference
from torch.utils.data import DataLoader

def _fit_ml_candidate(df: pd.DataFrame, train_rows: int, horizon: int, feature_cols: list[str]) -> dict[str, Any] | None:
    if len(df) <= horizon or train_rows <= horizon:
        return None

    target = df["Close"].pct_change(horizon).shift(-horizon)
    label = (target > 0).astype(int)
    valid_mask = label.notna().to_numpy().copy()
    valid_mask[-horizon:] = False

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[:train_rows] = True
    train_mask &= valid_mask

    train_idx = np.flatnonzero(train_mask)
    if train_idx.size < 320:
        return None

    val_size = max(int(train_idx.size * 0.2), 120)
    if val_size >= train_idx.size:
        return None

    core_idx = train_idx[:-val_size]
    val_idx = train_idx[-val_size:]

    if core_idx.size < 200 or val_idx.size < 80:
        return None

    feat_df = df[feature_cols]
    label_series = label

    X_core = feat_df.iloc[core_idx]
    y_core = label_series.iloc[core_idx]

    X_val = feat_df.iloc[val_idx]
    y_val = label_series.iloc[val_idx]

    if y_core.nunique() < 2 or y_val.nunique() < 2:
        return None

    model_builders = [
        (
            "logit",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.8,
                            max_iter=900,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            ),
        ),
        (
            "hgb",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=0.08,
                            max_iter=400,
                            max_depth=3,
                            min_samples_leaf=80,
                            l2_regularization=0.4,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]

    evaluated: list[dict[str, Any]] = []
    for name, pipeline in model_builders:
        try:
            pipeline.fit(X_core, y_core)
        except Exception as err:
            print(f"Hybrid ML signal (h={horizon}, model={name}) skipped during fit: {err}")
            continue

        try:
            val_probs = pipeline.predict_proba(X_val)[:, 1]
        except Exception as err:
            print(f"Hybrid ML signal (h={horizon}, model={name}) validation inference failed: {err}")
            continue

        try:
            auc = roc_auc_score(y_val, val_probs)
        except Exception as err:
            print(f"Hybrid ML signal (h={horizon}, model={name}) AUC failed: {err}")
            continue

        pred_labels = (val_probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, pred_labels)

        calibrator = None
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_probs, y_val.astype(float).to_numpy())
            val_probs_cal = calibrator.predict(val_probs)
            auc_cal = roc_auc_score(y_val, val_probs_cal)
            if np.isfinite(auc_cal) and auc_cal >= auc:
                val_probs = val_probs_cal
                auc = float(auc_cal)
        except Exception:
            calibrator = None

        evaluated.append(
            {
                "name": name,
                "pipeline": pipeline,
                "auc": float(auc),
                "acc": float(acc),
                "calibrator": calibrator,
            }
        )

    # Neural candidate training and evaluation
    try:
        neural_model = NeuralSequenceModel(input_dim=len(feature_cols))
        sequence_length = 12

        target = df["Close"].pct_change(horizon).shift(-horizon)
        label = (target > 0).astype(int)
        valid_mask = label.notna().to_numpy().copy()
        valid_mask[-horizon:] = False

        train_mask = np.zeros(len(df), dtype=bool)
        train_mask[:train_rows] = True
        train_mask &= valid_mask

        train_idx = np.flatnonzero(train_mask)
        if train_idx.size < 320:
            raise ValueError("Insufficient training data for neural model")

        val_size = max(int(train_idx.size * 0.2), 120)
        if val_size >= train_idx.size:
            raise ValueError("Insufficient validation data for neural model")

        core_idx = train_idx[:-val_size]
        val_idx = train_idx[-val_size:]

        feat_df = df[feature_cols]
        label_series = label

        X_core = feat_df.iloc[core_idx]
        y_core = label_series.iloc[core_idx]

        X_val = feat_df.iloc[val_idx]
        y_val = label_series.iloc[val_idx]

        train_dataset = SequenceDataset(X_core, y_core, sequence_length=sequence_length)
        val_dataset = SequenceDataset(X_val, y_val, sequence_length=sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        trained_model = train_neural_model(
            neural_model, train_loader, val_loader, epochs=10, lr=0.001, device=device
        )

        val_features = feat_df.iloc[val_idx].values
        preds = neural_inference(trained_model, val_features, sequence_length=sequence_length, device=device)

        y_val_seq = y_val[sequence_length:]
        auc = roc_auc_score(y_val_seq, preds)
        pred_labels = (preds >= 0.5).astype(int)
        acc = accuracy_score(y_val_seq, pred_labels)

        calibrator = None
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(preds, np.asarray(y_val_seq, dtype=float))
            preds_cal = calibrator.predict(preds)
            auc_cal = roc_auc_score(y_val_seq, preds_cal)
            if np.isfinite(auc_cal) and auc_cal >= auc:
                auc = float(auc_cal)
        except Exception:
            calibrator = None

        trained_model = trained_model.to("cpu")
        evaluated.append(
            {
                "name": "neural_sequence",
                "pipeline": trained_model,
                "auc": float(auc),
                "acc": float(acc),
                "sequence_length": sequence_length,
                "device": "cpu",
                "calibrator": calibrator,
            }
        )
    except Exception as err:
        print(f"Neural candidate training failed: {err}")



    if not evaluated:
        return None

    best_model = max(evaluated, key=lambda item: (item["auc"], item["acc"]))
    auc = best_model["auc"]
    acc = best_model["acc"]
    model_name = best_model["name"]

    min_auc = 0.512
    min_acc = 0.51
    if (not np.isfinite(auc)) or auc < min_auc or acc < min_acc:
        safe_auc = float(auc) if np.isfinite(auc) else float("nan")
        print(
            "Hybrid ML signal skipped due to weak validation: "
            f"h={horizon}, model={model_name}, auc={safe_auc:.3f}, acc={acc:.3f}, n_val={len(y_val)}"
        )
        return None

    full_train_df = feat_df.iloc[train_idx]
    full_train_y = label_series.iloc[train_idx]
    best_pipeline = best_model["pipeline"]
    model_name = best_model["name"]

    if model_name == "neural_sequence":
        sequence_length = int(best_model.get("sequence_length", 12))
        device = best_model.get("device", "cpu")
        try:
            full_probs = np.full(len(df), 0.5, dtype=float)
            neural_preds = neural_inference(
                best_pipeline,
                feat_df.values,
                sequence_length=sequence_length,
                device=device,
            )
            if len(neural_preds) > 0:
                full_probs[sequence_length:] = neural_preds
        except Exception as err:
            print(f"Hybrid ML signal (h={horizon}, model={model_name}) inference failed: {err}")
            return None
    else:
        try:
            best_pipeline.fit(full_train_df, full_train_y)
        except Exception as err:
            print(f"Hybrid ML signal (h={horizon}, model={model_name}) refit failed: {err}")
            return None

        try:
            full_probs = best_pipeline.predict_proba(feat_df)[:, 1]
        except Exception as err:
            print(f"Hybrid ML signal (h={horizon}, model={model_name}) inference failed: {err}")
            return None

    calibrator = best_model.get("calibrator")
    if calibrator is not None:
        try:
            full_probs = calibrator.predict(np.asarray(full_probs, dtype=float))
        except Exception:
            pass

    full_probs = np.clip(np.asarray(full_probs, dtype=float), 0.001, 0.999)

    lift = max(0.0, float(auc) - 0.5)
    ml_weight = min(0.65, 0.22 + (lift * 1.8))
    ml_component = pd.Series(((full_probs - 0.5) * 2.0).clip(-1.0, 1.0), index=df.index)
    ml_smoothed = ml_component.ewm(span=18, adjust=False).mean().clip(-1.0, 1.0)

    return {
        "horizon": horizon,
        "model": model_name,
        "auc": float(auc),
        "acc": float(acc),
        "weight": ml_weight,
        "raw": ml_component,
        "smoothed": ml_smoothed,
        "prob": pd.Series(full_probs, index=df.index),
        "val_size": len(y_val),
    }


def _blend_with_ml_signal(df: pd.DataFrame, train_rows: int, horizons: tuple[int, ...] = (3, 6, 12)) -> None:
    feature_cols = [
        "score",
        "trend",
        "m3",
        "m20",
        "volume_bias",
        "range_score",
        "ret_3",
        "ret_20",
        "volatility",
        "atr_pct",
        "volume_z",
        "ema_ratio",
        "macd_hist",
        "price_momentum",
        "volume_trend",
        "mtf_trend_1h",
        "mtf_trend_4h",
        "regime_trend",
        "regime_chop",
        "regime_high_vol",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return

    base_score = df["score"].clip(-1.0, 1.0)
    candidates: list[dict[str, Any]] = []

    for horizon in horizons:
        candidate = _fit_ml_candidate(df, train_rows, horizon, feature_cols)
        if candidate is None:
            continue
        candidates.append(candidate)

    if not candidates:
        print("Hybrid ML signal skipped: no horizon met validation criteria")
        return

    candidates.sort(key=lambda item: (item["auc"], item["acc"]), reverse=True)
    primary = candidates[0]

    blend_group: list[dict[str, Any]] = [primary]
    for candidate in candidates[1:]:
        auc_close = candidate["auc"] >= primary["auc"] - 0.02
        acc_close = candidate["acc"] >= primary["acc"] - 0.03
        if auc_close and acc_close:
            blend_group.append(candidate)
        if len(blend_group) >= 3:
            break

    weights: list[float] = []
    sum_weights = 0.0
    raw_combo = None
    smooth_combo = None
    prob_combo = None

    for idx, candidate in enumerate(blend_group):
        decay = max(0.55, 1.0 - (0.18 * idx))
        weight = candidate["weight"] * decay
        weights.append(weight)
        sum_weights += weight
        contrib_raw = candidate["raw"] * weight
        contrib_smooth = candidate["smoothed"] * weight
        contrib_prob = candidate.get("prob", (candidate["raw"] * 0.5 + 0.5)) * weight
        raw_combo = contrib_raw if raw_combo is None else raw_combo + contrib_raw
        smooth_combo = contrib_smooth if smooth_combo is None else smooth_combo + contrib_smooth
        prob_combo = contrib_prob if prob_combo is None else prob_combo + contrib_prob

    if sum_weights <= 0 or raw_combo is None or smooth_combo is None or prob_combo is None:
        print("Hybrid ML signal skipped: blend weights collapsed to zero")
        return

    combined_raw = (raw_combo / sum_weights).clip(-1.0, 1.0)
    combined_smoothed = (smooth_combo / sum_weights).clip(-1.0, 1.0)
    combined_prob = (prob_combo / sum_weights).clip(0.001, 0.999)

    base_std = float(base_score.std()) if hasattr(base_score, "std") else 0.0
    ml_std = float(combined_smoothed.std()) if hasattr(combined_smoothed, "std") else 0.0
    if ml_std > 0 and base_std > 0:
        scale = float(np.clip(base_std / ml_std, 0.6, 1.6))
        combined_raw = (combined_raw * scale).clip(-1.0, 1.0)
        combined_smoothed = (combined_smoothed * scale).clip(-1.0, 1.0)

    max_weight = 0.38 if len(blend_group) == 1 else 0.45
    effective_weight = min(max_weight, sum_weights)

    df["score_ml_raw"] = combined_raw
    df["score_ml"] = combined_smoothed
    df["meta_take_prob"] = combined_prob.ewm(span=16, adjust=False).mean().clip(0.001, 0.999)
    df["score"] = (effective_weight * combined_smoothed) + ((1.0 - effective_weight) * base_score)

    blend_summary = ", ".join(
        f"h={c['horizon']}, model={c['model']}, auc={c['auc']:.3f}, acc={c['acc']:.3f}, w={w:.2f}"
        for c, w in zip(blend_group, weights)
    )
    print(
        "Hybrid ML signal applied ("
        f"{len(blend_group)} models, effective_weight={effective_weight:.2f}; "
        f"{blend_summary}"
        ")"
    )


from ml.model_orchestrator import ModelOrchestrator


SHADOW_SCORE_PATH = Path("data/backtests/shadow_score.json")


def _update_shadow_score(profile: str, variant: str, test: Metrics) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if SHADOW_SCORE_PATH.exists():
        try:
            payload = json.loads(SHADOW_SCORE_PATH.read_text())
        except Exception:
            payload = {}

    hist = payload.get("history")
    if not isinstance(hist, list):
        hist = []

    entry = {
        "ts": datetime.utcnow().isoformat(),
        "profile": profile,
        "variant": variant,
        "ret": float(test.total_return),
        "dd": float(test.max_drawdown),
        "trades": int(test.trades),
        "win": float(test.win_rate),
    }
    hist.append(entry)
    hist = hist[-200:]

    recent = [x for x in hist if x.get("profile") == profile][-12:]
    if recent:
        avg_ret = float(np.mean([x.get("ret", 0.0) for x in recent]))
        avg_dd = float(np.mean([x.get("dd", 0.0) for x in recent]))
        positive_runs = sum(1 for x in recent if x.get("ret", 0.0) > 0)
    else:
        avg_ret, avg_dd, positive_runs = 0.0, 0.0, 0

    stability = {
        "window": len(recent),
        "avg_ret": avg_ret,
        "avg_dd": avg_dd,
        "positive_runs": positive_runs,
        "pass": (len(recent) >= 6 and avg_ret >= 0 and avg_dd >= -0.04 and positive_runs >= max(3, len(recent) // 2)),
    }

    payload["history"] = hist
    payload["latest"] = entry
    payload["stability"] = stability
    SHADOW_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SHADOW_SCORE_PATH.write_text(json.dumps(payload, indent=2))
    return stability


def run(symbol: str = "BTC-USD", interval: str = "5m", period: str = "60d", training_mode: str | None = None):
    mode_setting = (training_mode or TRAINING_MODE or "auto").strip().lower()
    if mode_setting not in {"auto", "classic", "neural"}:
        mode_setting = "auto"

    raw = download(symbol=symbol, interval=interval, period=period)
    baseline_df = features(raw)
    if len(baseline_df) < 200:
        raise RuntimeError(f"insufficient feature rows for backtest: {len(baseline_df)}")

    orchestrator = ModelOrchestrator()

    split = int(len(baseline_df) * 0.7)

    baseline_variant_name = "classic_baseline" if mode_setting == "classic" else "baseline"
    variant_candidates: list[tuple[str, pd.DataFrame]] = [(baseline_variant_name, baseline_df.copy())]

    ml_applied = False
    if mode_setting != "classic":
        ml_df = baseline_df.copy()
        _blend_with_ml_signal(ml_df, split)
        ml_applied = "score_ml" in ml_df.columns
        if ml_applied:
            blend_name = "neural_blend" if mode_setting == "neural" else "ml_blend"
            variant_candidates.append((blend_name, ml_df))
            ml_signal_df = baseline_df.copy()
            ml_signal_df["score"] = ml_df["score_ml"].copy()
            ml_signal_df["score_ml"] = ml_df["score_ml"].copy()
            if "score_ml_raw" in ml_df.columns:
                ml_signal_df["score_ml_raw"] = ml_df["score_ml_raw"].copy()
            signal_name = "neural_signal_only" if mode_setting == "neural" else "ml_signal_only"
            variant_candidates.append((signal_name, ml_signal_df))
        elif mode_setting == "neural":
            raise RuntimeError(
                "Neural training mode requires a validated ML signal, but blend generation failed."
            )

    start_time = time.time()
    variant_results: list[tuple[str, pd.DataFrame, Metrics, Metrics]] = []
    for variant_name, variant_df in variant_candidates:
        train_df_variant = variant_df.iloc[:split]
        test_df_variant = variant_df.iloc[split:]
        best_variant = pick_best(train_df_variant)
        test_variant = simulate(test_df_variant, best_variant.threshold)
        variant_results.append((variant_name, variant_df, best_variant, test_variant))
    elapsed_s = time.time() - start_time

    def _variant_key(item: tuple[str, pd.DataFrame, Metrics, Metrics]) -> tuple[float, float, float, float, float]:
        _, _, _, test_metrics = item
        return (
            -1.0 if test_metrics.do_not_trade else 0.0,
            test_metrics.total_return,
            test_metrics.profit_factor,
            -abs(test_metrics.max_drawdown),
            test_metrics.win_rate,
        )

    if not variant_results:
        raise RuntimeError("No training variants were evaluated")

    if mode_setting == "classic":
        selected_variant = next(item for item in variant_results if item[0] == baseline_variant_name)
    elif mode_setting == "neural":
        neural_variants = [item for item in variant_results if item[0] != baseline_variant_name]
        if not neural_variants:
            raise RuntimeError("Neural training mode did not produce any ML-backed variants")

        best_neural = max(neural_variants, key=_variant_key)
        baseline_candidate = next(item for item in variant_results if item[0] == baseline_variant_name)

        allow_baseline_fallback = os.getenv("NEURAL_ALLOW_BASELINE_FALLBACK", "1") == "1"
        _, _, _, neural_test = best_neural
        _, _, _, baseline_test = baseline_candidate
        neural_degraded = bool(neural_test.do_not_trade or neural_test.total_return < 0)

        if allow_baseline_fallback and neural_degraded and _variant_key(baseline_candidate) > _variant_key(best_neural):
            selected_variant = baseline_candidate
            print(
                "[ORCHESTRATION] Neural mode fallback activated: "
                f"selected baseline variant '{baseline_variant_name}' over degraded neural candidate."
            )
        else:
            selected_variant = best_neural
    else:
        selected_variant = max(variant_results, key=_variant_key)

    selected_name, selected_df, best, test = selected_variant

    if mode_setting == "classic":
        resolved_profile = "classic"
    elif mode_setting == "neural":
        resolved_profile = "neural"
    else:
        resolved_profile = "neural" if selected_name != baseline_variant_name else "classic"

    baseline_entry = next(item for item in variant_results if item[0] == baseline_variant_name)
    _, _, baseline_best, baseline_test = baseline_entry

    improvement = test.total_return - baseline_test.total_return
    drawdown_delta = test.max_drawdown - baseline_test.max_drawdown

    variants_payload = {
        name: {
            "train": asdict(train_metrics),
            "test": asdict(test_metrics),
        }
        for name, _, train_metrics, test_metrics in variant_results
    }

    summary_lines = ["Variant comparison:"]
    for name, _, train_metrics, test_metrics in variant_results:
        marker = "*" if name == selected_name else "-"
        summary_lines.append(
            f" {marker} {name}: train_ret={train_metrics.total_return:.2%}, "
            f"test_ret={test_metrics.total_return:.2%}, "
            f"test_dd={test_metrics.max_drawdown:.2%}, trades={test_metrics.trades}, "
            f"threshold={train_metrics.threshold:.3f}"
        )
    print("\n".join(summary_lines))

    train_df = selected_df.iloc[:split]
    test_df = selected_df.iloc[split:]

    # Check for champion promotion
    champion_model, champion_metrics = orchestrator.get_champion()
    challenger_metrics = asdict(test)
    challenger_metrics["threshold"] = best.threshold
    challenger_metrics["variant"] = selected_name
    challenger_metrics["training_profile"] = resolved_profile

    shadow_stability = _update_shadow_score(resolved_profile, selected_name, test)

    promoted = False
    candidate_model_name = f"{resolved_profile}:{selected_name}"
    if shadow_stability.get("pass") and orchestrator.should_promote(challenger_metrics):
        orchestrator.promote(candidate_model_name, challenger_metrics)
        promoted = True
        print(
            f"[ORCHESTRATION] Promoted new champion model: {candidate_model_name} "
            f"(return={challenger_metrics.get('total_return', 0.0):.2%}, "
            f"dd={challenger_metrics.get('max_drawdown', 0.0):.2%}, "
            f"trades={challenger_metrics.get('trades', 0)})"
        )
    else:
        print("[ORCHESTRATION] Candidate rejected by promotion/shadow stability gates.")

    result = {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "rows": len(selected_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "best_train": asdict(best),
        "test": asdict(test),
        "elapsed_time_s": elapsed_s,
        "variant": selected_name,
        "variants": variants_payload,
        "variant_improvement": {
            "test_return_delta_vs_baseline": improvement,
            "test_max_drawdown_delta_vs_baseline": drawdown_delta,
        },
        "orchestration": {
            "candidate_model": candidate_model_name,
            "promoted": promoted,
            "previous_champion": champion_model,
            "shadow_stability": shadow_stability,
        },
        "training_profile": resolved_profile,
        "training_profile_requested": mode_setting,
        "training_label": TRAINING_LABEL or resolved_profile,
    }

    out_file = OUT_DIR / "latest.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_file}")

    mode = "defensive-standby" if (best.trades == 0 or test.trades == 0) else "active-trading"
    if selected_name == baseline_variant_name:
        if ml_applied and mode_setting != "classic":
            variant_note = f"{selected_name} (ML blend rejected)"
        else:
            variant_note = selected_name
    elif selected_name in {"ml_blend", "neural_blend", "ml_signal_only", "neural_signal_only"}:
        variant_note = (
            f"{selected_name} (Δ test return vs {baseline_variant_name}: {improvement:+.2%}, "
            f"Δ max DD: {drawdown_delta:+.2%})"
        )
    else:
        variant_note = selected_name

    label_used = TRAINING_LABEL or resolved_profile
    label_line = f"\nLabel: {label_used}"
    detailed_msg = (
        f"\n🏁 Training iteration completed in {elapsed_s:.1f}s. Mode: {mode}"
        f"\nTraining profile: {resolved_profile} (requested: {mode_setting})"
        f"\nSymbol: {symbol} | Interval: {interval} | Period: {period}"
        f"\nTrain return: {best.total_return:.2%} | Test return: {test.total_return:.2%}"
        f"\nTest win rate: {test.win_rate:.2%} | Max drawdown: {test.max_drawdown:.2%}"
        f"\nProfit factor: {test.profit_factor:.2f} | Win/Loss ratio: {test.win_loss_ratio:.2f} | Consistency: {test.consistency_score:.5f}"
        f"\nThreshold: {best.threshold:.3f} | Trades: {test.trades} | DoNotTrade: {test.do_not_trade}"
        f"\nVariant: {variant_note}"
        f"\nChampion: {champion_model}"
        f"{label_line}"
    )
    print(detailed_msg)

    webhook_msg = _build_webhook_metrics(
        symbol=symbol,
        interval=interval,
        period=period,
        mode=mode,
        train_return=best.total_return,
        test_return=test.total_return,
        test_win_rate=test.win_rate,
        test_max_drawdown=test.max_drawdown,
    )
    webhook_msg = f"{webhook_msg} | training_profile={resolved_profile} | variant={selected_name}"
    print(f"Webhook metrics: {webhook_msg}")
    try:
        delivered = send_training_update(webhook_msg, label=label_used)
        if not delivered:
            print("Webhook update failed: no webhook configured or delivery failed after retries")
    except Exception as e:
        # Never fail the training/backtest run just because the webhook endpoint is unavailable.
        print(f"Webhook update failed: {e}")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        # Ensure each training run still emits a concise metrics update, even on failure.
        err_msg = _build_webhook_metrics(
            symbol="BTC-USD",
            interval="5m",
            period="60d",
            mode="failed",
            train_return=None,
            test_return=None,
            test_win_rate=None,
            test_max_drawdown=None,
            error=str(e),
        )
        requested_profile = TRAINING_MODE or "auto"
        label_used = TRAINING_LABEL or requested_profile
        err_msg = f"{err_msg} | training_profile={requested_profile}"
        print(f"Webhook metrics: {err_msg}")
        try:
            send_training_update(err_msg, label=label_used)
        except Exception:
            pass
        raise
