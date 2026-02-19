from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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


def _signals_long_only(df: pd.DataFrame, threshold: float) -> np.ndarray:
    score = df["score"].values
    rsi = df["rsi"].values
    m3 = df["m3"].values
    m20 = df["m20"].values
    trend = df["trend"].values
    vol = df["volatility"].values

    buy_threshold = threshold + np.clip((vol - 0.01) * 8.0, 0.0, 0.10)
    sell_threshold = -threshold - np.clip((vol - 0.01) * 8.0, 0.0, 0.10)

    overbought_exit = (rsi > 74) & (m3 < 0.08)
    bearish_confirmation = (trend < -0.03) & (m3 < 0.0)
    risk_off_regime = ((trend < -0.06) & (m20 < -0.08)) | (vol > 0.02)

    oversold_rebound = (rsi < 30) & (m3 > (m20 + 0.10))
    extreme_oversold_reversal = (rsi < 27) & (m3 > -0.15)
    overbought_exhaustion = (rsi > 78) & (m3 > 0.10)

    want_buy = (score > buy_threshold) | oversold_rebound | extreme_oversold_reversal
    want_buy = want_buy & (~overbought_exhaustion) & (~risk_off_regime)

    want_sell = overbought_exit | ((score < sell_threshold) & bearish_confirmation) | risk_off_regime

    signal = np.zeros(len(df), dtype=np.int8)
    in_pos = False
    cooldown = 0

    for i in range(len(df)):
        if cooldown > 0:
            cooldown -= 1

        if not in_pos:
            if cooldown == 0 and want_buy[i]:
                signal[i] = 1
                in_pos = True
        else:
            if want_sell[i]:
                signal[i] = -1
                in_pos = False
                cooldown = 2  # avoid immediate churn in noisy bars

    return signal


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

    signal = _signals_long_only(df, threshold)

    # Convert event signal (buy/sell actions) to position state.
    position = np.zeros(len(signal), dtype=float)
    in_pos = 0.0
    for i, sig in enumerate(signal):
        if sig == 1:
            in_pos = 1.0
        elif sig == -1:
            in_pos = 0.0
        position[i] = in_pos

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
    trades = int((signal == 1).sum())
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
    candidates = np.arange(0.06, 0.26, 0.01)
    scored = [simulate(train_df, th) for th in candidates]

    # Objective: prioritize robust equity growth, then risk-adjusted return, then drawdown control.
    scored.sort(
        key=lambda m: (
            m.total_return,
            m.sharpe_like,
            m.expectancy * 1000,
            m.max_drawdown,
            -m.trades,
        ),
        reverse=True,
    )
    return scored[0]


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


if __name__ == "__main__":
    run()
