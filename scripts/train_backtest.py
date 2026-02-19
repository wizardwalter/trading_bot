from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf

from discord.notify import send_training_update

OUT_DIR = Path("data/backtests")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download(symbol: str = "BTC-USD", interval: str = "5m", period: str = "60d", retries: int = 4) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(symbol, interval=interval, period=period, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            cleaned = df.dropna().copy()
            if cleaned.empty:
                raise ValueError(f"No market data returned for {symbol}")
            return cleaned
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.6 * attempt)

    raise RuntimeError(f"download failed for {symbol} after {retries} attempts: {last_err}")


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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

    out = out.dropna().copy()

    trend = ((out["ema_fast"] - out["ema_slow"]) / out["Close"]).clip(-1, 1) * 220
    trend = trend.clip(-1, 1)
    m20 = (out["ret_20"] * 25).clip(-1, 1)
    m3 = (out["ret_3"] * 35).clip(-1, 1)

    rsi_comp = np.where(out["rsi"] < 35, 0.8, np.where(out["rsi"] > 68, -0.8, 0.0))
    out["score"] = 0.44 * trend + 0.22 * m20 + 0.20 * m3 + 0.14 * rsi_comp
    out["trend"] = trend
    out["m3"] = m3
    out["m20"] = m20
    return out


@dataclass
class Metrics:
    threshold: float
    trades: int
    win_rate: float
    expectancy: float
    total_return: float
    sharpe_like: float
    max_drawdown: float


def _target_position(df: pd.DataFrame, threshold: float) -> np.ndarray:
    score = df["score"].values
    rsi = df["rsi"].values
    m3 = df["m3"].values
    m20 = df["m20"].values
    trend = df["trend"].values
    vol = df["volatility"].values

    buy_threshold = threshold + np.clip((vol - 0.01) * 8.0, 0.0, 0.10)
    sell_threshold = -threshold - np.clip((vol - 0.01) * 8.0, 0.0, 0.10)

    overbought = rsi > 74
    oversold = rsi < 28

    bullish_confirmation = (trend > -0.01) & (m20 > -0.05)
    bearish_confirmation = (trend < 0.01) & (m20 < 0.05)

    long_entry = (score > buy_threshold) & bullish_confirmation & (~overbought)
    short_entry = (score < sell_threshold) & bearish_confirmation & (~oversold)

    long_exit = (score < -0.05) | (overbought & (m3 < 0.06))
    short_exit = (score > 0.05) | (oversold & (m3 > -0.06))

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
                cooldown = 2
            elif short_entry[i]:
                state = -1
                cooldown = 1
        elif state == -1:
            if short_exit[i]:
                state = 0
                cooldown = 2
            elif long_entry[i]:
                state = 1
                cooldown = 1

        position[i] = state

    return position


def simulate(df: pd.DataFrame, threshold: float, fee_bps: float = 4.0, slippage_bps: float = 2.0) -> Metrics:
    if df.empty:
        return Metrics(
            threshold=float(threshold),
            trades=0,
            win_rate=0.0,
            expectancy=0.0,
            total_return=0.0,
            sharpe_like=0.0,
            max_drawdown=0.0,
        )

    fee = fee_bps / 10_000
    slippage = slippage_bps / 10_000

    position = _target_position(df, threshold).astype(float)

    # Position applies from the next bar onward.
    position = pd.Series(position).shift(1).fillna(0).values

    rets = df["Close"].pct_change().fillna(0).values
    strat = position * rets

    turns = np.abs(np.diff(np.r_[0, position]))
    costs = turns * (fee + slippage)
    strat = strat - costs

    eq = (1 + pd.Series(strat)).cumprod()
    total_return = float(eq.iloc[-1] - 1)

    trade_mask = turns > 0
    trade_rets = pd.Series(strat)[trade_mask]
    trades = int((turns > 0).sum())
    win_rate = float((trade_rets > 0).mean()) if len(trade_rets) else 0.0
    expectancy = float(trade_rets.mean()) if len(trade_rets) else 0.0

    vol = float(pd.Series(strat).std())
    sharpe_like = float((pd.Series(strat).mean() / vol) * np.sqrt(252 * 24 * 12)) if vol > 0 else 0.0

    rolling_max = eq.cummax()
    dd = (eq / rolling_max) - 1
    max_drawdown = float(dd.min())

    return Metrics(
        threshold=float(threshold),
        trades=trades,
        win_rate=win_rate,
        expectancy=expectancy,
        total_return=total_return,
        sharpe_like=sharpe_like,
        max_drawdown=max_drawdown,
    )


def pick_best(train_df: pd.DataFrame) -> Metrics:
    candidates = np.arange(0.05, 0.71, 0.01)

    # Use a simple internal walk-forward split so threshold tuning is less likely
    # to overfit to the earliest segment of the train window.
    inner_split = int(len(train_df) * 0.7)
    inner_split = max(120, min(len(train_df) - 80, inner_split))
    fit_df = train_df.iloc[:inner_split]
    val_df = train_df.iloc[inner_split:]

    if len(fit_df) < 80 or len(val_df) < 40:
        scored = [simulate(train_df, th) for th in candidates]
        return max(
            scored,
            key=lambda m: (
                m.total_return,
                m.max_drawdown,
                m.sharpe_like,
                -abs(m.trades - 30),
                -m.threshold,
            ),
        )

    ranked: list[tuple[float, Metrics, Metrics]] = []
    for th in candidates:
        fit = simulate(fit_df, th)
        val = simulate(val_df, th)

        # Score favors validation robustness, while penalizing deep drawdowns and
        # strongly negative fit behavior.
        score = (
            (val.total_return * 2.8)
            + (val.expectancy * 180.0)
            + (val.sharpe_like * 0.06)
            + (val.max_drawdown * 0.22)
            - (0.55 if fit.max_drawdown < -0.18 else 0.0)
            - (0.45 if fit.total_return < -0.06 else 0.0)
            - (0.20 if val.trades < 4 else 0.0)
        )
        ranked.append((score, fit, val))

    # Prefer active-but-defensive settings when available.
    viable = [
        (score, fit, val)
        for score, fit, val in ranked
        if val.max_drawdown >= -0.10 and fit.max_drawdown >= -0.15
    ]

    if viable:
        chosen = max(
            viable,
            key=lambda x: (
                x[0],
                x[2].total_return,
                x[2].expectancy,
                x[2].sharpe_like,
                -abs(x[2].trades - 24),
                -x[2].threshold,
            ),
        )
        return simulate(train_df, chosen[2].threshold)

    # Fallback: most defensive threshold over the full training slice.
    scored = [simulate(train_df, th) for th in candidates]
    return max(
        scored,
        key=lambda m: (
            m.total_return,
            m.max_drawdown,
            m.sharpe_like,
            -abs(m.trades - 20),
            -m.threshold,
        ),
    )


def run(symbol: str = "BTC-USD", interval: str = "5m", period: str = "60d"):
    raw = download(symbol=symbol, interval=interval, period=period)
    df = features(raw)
    if len(df) < 200:
        raise RuntimeError(f"insufficient feature rows for backtest: {len(df)}")

    split = int(len(df) * 0.7)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    best = pick_best(train_df)
    test = simulate(test_df, best.threshold)

    result = {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "best_train": asdict(best),
        "test": asdict(test),
    }

    out_file = OUT_DIR / "latest.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {out_file}")

    mode = "defensive-standby" if (best.trades == 0 or test.trades == 0) else "active-trading"
    webhook_msg = (
        f"{symbol} {interval} {period} | mode={mode} | "
        f"thr={best.threshold:.2f} | train trades={best.trades}, test trades={test.trades} | "
        f"train return {best.total_return:.2%}, "
        f"test return {test.total_return:.2%}, "
        f"test win rate {test.win_rate:.1%}, "
        f"test max drawdown {test.max_drawdown:.2%}"
    )
    try:
        send_training_update(webhook_msg)
    except Exception as e:
        # Never fail the training/backtest run just because the webhook endpoint is unavailable.
        print(f"Webhook update failed: {e}")


if __name__ == "__main__":
    run()
