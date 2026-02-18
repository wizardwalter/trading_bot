from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import yfinance as yf


@dataclass
class Signal:
    symbol: str
    action: str  # buy|sell|hold
    price: float
    confidence: float
    score: float
    reason: str
    volatility: float


def _download(symbol: str, interval: str = "5m", period: str = "5d", retries: int = 3) -> pd.DataFrame:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                tickers=symbol,
                interval=interval,
                period=period,
                progress=False,
                threads=False,
                auto_adjust=False,
            )
            if df.empty:
                raise ValueError(f"No market data returned for {symbol}")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    raise ValueError(f"Missing {col} in market data for {symbol}")

            cleaned = df.dropna().copy()
            if cleaned.empty:
                raise ValueError(f"Only NaN market data returned for {symbol}")
            return cleaned
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.6 * attempt)

    raise ValueError(f"Market data fetch failed for {symbol}: {last_err}")


def _features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ema_fast"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=26, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    out["rsi"] = 100 - (100 / (1 + rs))

    out["ret_20"] = out["Close"].pct_change(20)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["volatility"] = out["hl_spread"].rolling(20).mean()

    return out.dropna()


def build_signal(symbol: str) -> Signal:
    df = _features(_download(symbol))
    row = df.iloc[-1]

    price = float(row["Close"])
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    ret_20 = float(row["ret_20"])
    vol = max(float(row["volatility"]), 0.002)

    trend_component = (ema_fast - ema_slow) / price
    trend_component = max(min(trend_component * 220, 1.0), -1.0)

    if rsi < 35:
        rsi_component = 0.8
    elif rsi > 68:
        rsi_component = -0.8
    else:
        rsi_component = 0.0

    momentum_component = max(min(ret_20 * 25, 1.0), -1.0)

    score = (0.52 * trend_component) + (0.28 * momentum_component) + (0.20 * rsi_component)

    # Dynamic thresholds: in noisier conditions ask for stronger edge.
    vol_penalty = min(max((vol - 0.01) * 8.0, 0.0), 0.10)
    buy_threshold = 0.18 + vol_penalty
    sell_threshold = -0.26 - vol_penalty

    # Oversold bounce bias avoids a perpetual HOLD state in mild downtrends.
    oversold_rebound = (rsi < 32 and momentum_component > -0.20)

    if score > buy_threshold or oversold_rebound:
        action = "buy"
    elif score < sell_threshold or rsi > 74:
        action = "sell"
    else:
        action = "hold"

    confidence = min(max((abs(score) - 0.05) / 0.65, 0.0), 1.0)
    reason = (
        f"trend={trend_component:+.2f}, momentum={momentum_component:+.2f}, "
        f"rsi={rsi:.1f}, vol={vol:.4f}, score={score:+.2f}, "
        f"thr=[{sell_threshold:+.2f},{buy_threshold:+.2f}]"
    )

    return Signal(
        symbol=symbol,
        action=action,
        price=price,
        confidence=confidence,
        score=score,
        reason=reason,
        volatility=vol,
    )


def position_size(equity: float, price: float, volatility: float, max_risk_per_trade: float = 0.01) -> int:
    risk_budget = max(equity, 1.0) * max_risk_per_trade
    risk_per_unit = price * max(volatility, 0.005)
    qty = int(risk_budget / risk_per_unit)
    return max(qty, 1)


def should_enter_trade(ticker: str) -> Dict:
    signal = build_signal(ticker)
    return {
        "enter": signal.action in ("buy", "sell"),
        "action": signal.action,
        "price": signal.price,
        "confidence": signal.confidence,
        "score": signal.score,
        "reason": signal.reason,
        "volatility": signal.volatility,
    }
