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

    rsi_comp = np.where(out["rsi"] < 33, 0.8, np.where(out["rsi"] > 70, -0.8, 0.0))
    # Bias toward RSI mean-reversion while preserving trend context for fewer whipsaws.
    out["score"] = 0.35 * trend + 0.15 * m20 + 0.05 * m3 + 0.45 * rsi_comp
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

    # Slightly earlier overbought filtering and deeper oversold allowance improved OOS behavior.
    overbought = rsi > 70
    oversold = rsi < 22

    # Reduce entries during volatile chop unless directional conviction is strong.
    vol_guard = vol <= np.nanpercentile(vol, 85)

    bullish_confirmation = (trend > -0.01) & (m20 > -0.05) & (m3 > -0.12)
    bearish_confirmation = (trend < 0.01) & (m20 < 0.05) & (m3 < 0.12)

    long_entry = (score > buy_threshold) & bullish_confirmation & (~overbought) & (vol_guard | (trend > 0.25))
    short_entry = (score < sell_threshold) & bearish_confirmation & (~oversold) & (vol_guard | (trend < -0.25))

    long_exit = (score < -0.01) | (overbought & (m3 < 0.08)) | ((trend < -0.08) & (m3 < -0.15))
    short_exit = (score > 0.01) | (oversold & (m3 > -0.08)) | ((trend > 0.08) & (m3 > 0.15))

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
                cooldown = 0
            elif short_entry[i]:
                state = -1
                cooldown = 0
        elif state == -1:
            if short_exit[i]:
                state = 0
                cooldown = 0
            elif long_entry[i]:
                state = 1
                cooldown = 0

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
    win_rate = float(np.mean(np.array(trade_rets) > 0)) if trades else 0.0
    expectancy = float(np.mean(trade_rets)) if trades else 0.0

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


def _score_metrics(m: Metrics) -> float:
    return (
        (m.total_return * 2.8)
        + (m.sharpe_like * 0.045)
        + (m.win_rate * 0.20)
        + (m.max_drawdown * 0.30)
        - (0.25 if m.trades < 6 else 0.0)
        - (0.15 if m.max_drawdown < -0.12 else 0.0)
        - (0.08 if (m.expectancy < 0 and m.win_rate < 0.30) else 0.0)
    )


def pick_best(train_df: pd.DataFrame) -> Metrics:
    candidates = np.arange(0.05, 0.71, 0.01)

    # Walk-forward validation within the training window to reduce threshold overfitting.
    n = len(train_df)
    fold_start = int(n * 0.30)
    fold_ends = [int(n * 0.50), int(n * 0.65), int(n * 0.80), n]

    scored: list[tuple[float, Metrics]] = []
    for th in candidates:
        full_m = simulate(train_df, th)

        fold_metrics: list[Metrics] = []
        prev = fold_start
        for end in fold_ends:
            fold = train_df.iloc[prev:end]
            if len(fold) < 200:
                continue
            fold_metrics.append(simulate(fold, th))
            prev = end

        fold_scores = [_score_metrics(m) for m in fold_metrics]
        cv_score = float(np.mean(fold_scores)) if fold_scores else -1.0

        # Prefer thresholds that are consistent across folds, not just high average.
        if fold_metrics:
            ret_std = float(np.std([m.total_return for m in fold_metrics]))
            worst_fold_return = float(min(m.total_return for m in fold_metrics))
            worst_fold_dd = float(min(m.max_drawdown for m in fold_metrics))
            stability_penalty = (ret_std * 1.8) + (abs(min(worst_fold_return, 0.0)) * 0.7) + (abs(min(worst_fold_dd + 0.15, 0.0)) * 0.3)
        else:
            stability_penalty = 0.45

        score = (_score_metrics(full_m) * 0.40) + (cv_score * 0.60) - stability_penalty
        scored.append((score, full_m))

    # Favor thresholds with positive train behavior, acceptable risk, and enough activity.
    viable = [
        (score, m)
        for score, m in scored
        if m.total_return > 0 and m.max_drawdown >= -0.12 and m.trades >= 6
    ]

    if viable:
        return max(
            viable,
            key=lambda x: (x[0], x[1].total_return, x[1].sharpe_like, -abs(x[1].trades - 16), -x[1].threshold),
        )[1]

    # Fallback: least-bad threshold by blended return/risk score.
    return max(
        scored,
        key=lambda x: (x[0], x[1].total_return, x[1].max_drawdown, x[1].sharpe_like, -x[1].threshold),
    )[1]


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
        f"{symbol} {interval} {period} {mode} "
        f"train_return={best.total_return:.2%} "
        f"test_return={test.total_return:.2%} "
        f"test_win_rate={test.win_rate:.1%} "
        f"test_max_drawdown={test.max_drawdown:.2%}"
    )
    try:
        delivered = send_training_update(webhook_msg)
        if not delivered:
            print("Webhook update failed: no webhook configured or delivery failed after retries")
    except Exception as e:
        # Never fail the training/backtest run just because the webhook endpoint is unavailable.
        print(f"Webhook update failed: {e}")


if __name__ == "__main__":
    run()
