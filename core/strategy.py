from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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

    out["ret_3"] = out["Close"].pct_change(3)
    out["ret_20"] = out["Close"].pct_change(20)
    out["hl_spread"] = (out["High"] - out["Low"]) / out["Close"]
    out["volatility"] = out["hl_spread"].rolling(20).mean()
    out["range_high"] = out["Close"].rolling(48, min_periods=30).max()
    out["range_low"] = out["Close"].rolling(48, min_periods=30).min()

    return out.dropna()


def _load_backtest_threshold(symbol: str, fallback: float, expected_interval: str | None = None) -> float:
    if os.getenv("USE_BACKTEST_THRESHOLD", "1") != "1":
        return fallback

    path = Path(os.getenv("BACKTEST_LATEST_PATH", "data/backtests/latest.json"))
    if not path.exists():
        return fallback

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return fallback

    if str(payload.get("symbol", "")).upper() != symbol.upper():
        return fallback

    # Guard against applying a threshold calibrated for a different bar interval.
    if expected_interval and str(payload.get("interval", "")).lower() != str(expected_interval).lower():
        return fallback

    # Guard against stale calibrations lingering for too long.
    max_age_hours = float(os.getenv("BACKTEST_MAX_AGE_HOURS", "72"))
    try:
        age_s = time.time() - path.stat().st_mtime
        if max_age_hours > 0 and age_s > (max_age_hours * 3600):
            return fallback
    except Exception:
        return fallback

    test_block = payload.get("test", {}) if isinstance(payload.get("test", {}), dict) else {}
    # Reject calibrations that explicitly fail execution gates.
    if bool(test_block.get("do_not_trade", False)):
        return fallback

    shadow_stability = payload.get("orchestration", {}).get("shadow_stability", {})
    if isinstance(shadow_stability, dict) and shadow_stability.get("pass") is False:
        return fallback

    candidate = test_block.get("threshold")
    if candidate is None:
        candidate = payload.get("best_train", {}).get("threshold")
    if candidate is None:
        return fallback

    # Keep thresholds in a sane range so stale/overfit backtests cannot brick execution.
    return float(min(max(float(candidate), 0.08), 0.60))


def _shadow_drift_penalty(symbol: str) -> float:
    """Return an additive threshold penalty when recent shadow drift is deteriorating."""
    path = Path(os.getenv("BACKTEST_SHADOW_PATH", "data/backtests/shadow_score.json"))
    if not path.exists():
        return 0.0

    try:
        payload = json.loads(path.read_text())
    except Exception:
        return 0.0

    history = payload.get("history", []) if isinstance(payload, dict) else []
    if not isinstance(history, list) or not history:
        return 0.0

    rows = []
    for row in history:
        if not isinstance(row, dict):
            continue
        if str(row.get("profile", "")).lower() != "neural":
            continue
        if str(row.get("variant", "")).lower() == "baseline":
            continue
        ts_raw = row.get("ts")
        try:
            ts = datetime.fromisoformat(str(ts_raw))
        except Exception:
            continue
        rows.append({"ts": ts, "ret": float(row.get("ret", 0.0)), "dd": float(row.get("dd", 0.0))})

    if len(rows) < 24:
        return 0.0

    now = max(item["ts"] for item in rows)

    def _window(hours: int) -> list[dict]:
        cutoff_s = hours * 3600
        return [item for item in rows if (now - item["ts"]).total_seconds() <= cutoff_s]

    window_24h = _window(24)
    window_72h = _window(72)
    if len(window_24h) < 12 or len(window_72h) < 24:
        return 0.0

    avg_ret_24h = sum(item["ret"] for item in window_24h) / len(window_24h)
    avg_ret_72h = sum(item["ret"] for item in window_72h) / len(window_72h)
    avg_dd_24h = sum(item["dd"] for item in window_24h) / len(window_24h)
    avg_dd_72h = sum(item["dd"] for item in window_72h) / len(window_72h)

    drift_negative = avg_ret_24h < 0 and avg_ret_24h < avg_ret_72h
    drawdown_worse = avg_dd_24h < avg_dd_72h

    if drift_negative and drawdown_worse:
        # Keep the response measured: add a modest +2% score threshold until
        # rolling metrics stabilize.
        return float(os.getenv("BTC_DRIFT_THRESHOLD_PENALTY", "0.02"))

    return 0.0


def _symbol_profile(symbol: str) -> dict:
    s = symbol.upper()
    if s == "BTC-USD":
        base = 0.15  # slightly stricter BTC threshold after negative drift to curb marginal entries
        interval = "1m"
        adaptive_base = min(base + _shadow_drift_penalty(s), 0.22)
        return {
            "interval": interval,
            "period": "2d",
            # BTC live execution runs on 1m bars while backtests currently publish
            # 5m calibrations; allow fresh backtest thresholds instead of always
            # falling back to the static base.
            "entry_threshold": _load_backtest_threshold(s, adaptive_base, expected_interval=None),
        }
    # default day-trading profile for equities
    return {
        "interval": "5m",
        "period": "5d",
        "entry_threshold": 0.12,
    }


def _latest_bar_age_seconds(df: pd.DataFrame) -> float:
    if df.empty:
        return float("inf")
    ts = df.index[-1]
    try:
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max((now - ts).total_seconds(), 0.0)
    except Exception:
        return float("inf")


def build_signal(symbol: str, has_position: bool = False) -> Signal:
    profile = _symbol_profile(symbol)
    df = _features(_download(symbol, interval=profile["interval"], period=profile["period"]))

    max_staleness_s = float(os.getenv("MAX_MARKET_DATA_STALENESS_SECONDS", "240"))
    latest_age_s = _latest_bar_age_seconds(df)
    if latest_age_s > max_staleness_s:
        raise ValueError(
            f"Stale market data for {symbol}: last bar age {latest_age_s:.0f}s > {max_staleness_s:.0f}s"
        )

    row = df.iloc[-1]

    price = float(row["Close"])
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    ret_3 = float(row["ret_3"])
    ret_20 = float(row["ret_20"])
    vol = max(float(row["volatility"]), 0.002)
    range_high = float(row["range_high"])
    range_low = float(row["range_low"])
    range_width = max(range_high - range_low, max(price * 0.0005, 1e-6))
    range_pos = max(min((price - range_low) / range_width, 1.2), -0.2)

    trend_component = (ema_fast - ema_slow) / price
    trend_component = max(min(trend_component * 220, 1.0), -1.0)

    if rsi < 35:
        rsi_component = 0.8
    elif rsi > 68:
        rsi_component = -0.8
    else:
        rsi_component = 0.0

    momentum_component = max(min(ret_20 * 25, 1.0), -1.0)
    short_momentum_component = max(min(ret_3 * 35, 1.0), -1.0)
    range_drift_component = max(min(((range_pos - 0.5) * 1.8), 1.0), -1.0)
    breakout_component = max(min(((price - range_high) / range_width) * 2.0, 1.0), -1.0)

    score = (
        (0.40 * trend_component)
        + (0.20 * momentum_component)
        + (0.18 * short_momentum_component)
        + (0.12 * rsi_component)
        + (0.06 * range_drift_component)
        + (0.04 * breakout_component)
    )

    # Dynamic thresholds: per-symbol baseline + volatility penalty.
    base_threshold = float(profile["entry_threshold"])
    vol_penalty = min(max((vol - 0.01) * 8.0, 0.0), 0.10)
    buy_threshold = base_threshold + vol_penalty
    sell_threshold = -base_threshold - vol_penalty

    # Oversold rebound bias: let BTC enter a little earlier on deep pullbacks
    # when momentum is no longer strongly deteriorating.
    oversold_rebound = (rsi < 30 and short_momentum_component > momentum_component + 0.10)
    extreme_oversold_reversal = (rsi < 27 and short_momentum_component > -0.15)

    # Avoid fresh long entries during euphoric spikes.
    overbought_exhaustion = rsi > 78 and short_momentum_component > 0.10

    # Exit logic: trim risk on clear downside, and also de-risk overbought conditions
    # once short-term momentum stops accelerating higher.
    overbought_exit = rsi > 74 and short_momentum_component < 0.08

    bearish_confirmation = (trend_component < -0.03 and short_momentum_component < 0.0)

    # Regime filter: avoid new entries in high-volatility downside chop where the
    # current model historically overtrades and bleeds on fees/slippage.
    risk_off_regime = (trend_component < -0.06 and momentum_component < -0.08) or vol > 0.02

    # Require at least one supportive regime signal for non-rebound buys.
    bullish_alignment = (trend_component > 0.0) or (momentum_component > 0.0 and short_momentum_component > -0.05)

    # Selective counter-trend allowance for washed-out reversals to avoid getting
    # stuck in perpetual HOLD during deep but stabilizing pullbacks.
    allow_countertrend_reversal = (
        extreme_oversold_reversal
        and vol < 0.03
        and trend_component > -0.30
    )

    breakout_ready = price >= range_high * 0.998
    breakdown_risk = price <= range_low * 1.002
    compression_regime = (range_width / price) < 0.012
    range_stop_trigger = has_position and (range_pos < 0.18 and short_momentum_component < -0.05)

    if (
        (score > buy_threshold or oversold_rebound or extreme_oversold_reversal)
        and not overbought_exhaustion
        and (not risk_off_regime or allow_countertrend_reversal)
        and (bullish_alignment or oversold_rebound or extreme_oversold_reversal)
        and (breakout_ready or range_pos > 0.45 or oversold_rebound or extreme_oversold_reversal)
    ):
        action = "buy"
    elif (
        overbought_exit
        or (score < sell_threshold and bearish_confirmation)
        or risk_off_regime
        or range_stop_trigger
        or (has_position and breakdown_risk and short_momentum_component < -0.02)
    ):
        action = "sell" if has_position else "hold"
    else:
        action = "hold"

    # Suppress low-conviction long entries around neutral conditions, but do not mask
    # explicit risk-off exits (sell threshold / overbought exit).
    if action == "buy" and abs(score) < 0.06 and not (oversold_rebound or extreme_oversold_reversal):
        action = "hold"

    if action == "buy" and not (oversold_rebound or extreme_oversold_reversal):
        if range_pos < 0.20 or (compression_regime and not breakout_ready and short_momentum_component < 0.04):
            action = "hold"
        if range_pos > 0.85 and short_momentum_component < 0.08 and not breakout_ready:
            action = "hold"

    base_confidence = min(max((abs(score) - 0.04) / 0.56, 0.0), 1.0)
    setup_confidence_boost = 0.0
    if oversold_rebound:
        setup_confidence_boost = max(setup_confidence_boost, 0.07)
    if extreme_oversold_reversal:
        setup_confidence_boost = max(setup_confidence_boost, 0.10)
    if overbought_exhaustion and action == "hold":
        setup_confidence_boost = max(setup_confidence_boost, 0.05)

    # Damp confidence under elevated volatility so execution/risk filters reject
    # more borderline entries during noisy regimes.
    vol_conf_penalty = min(max((vol - 0.015) * 15.0, 0.0), 0.20)
    confidence = min(max(base_confidence + setup_confidence_boost - vol_conf_penalty, 0.0), 1.0)
    reason = (
        f"trend={trend_component:+.2f}, momentum20={momentum_component:+.2f}, "
        f"momentum3={short_momentum_component:+.2f}, rsi={rsi:.1f}, vol={vol:.4f}, score={score:+.2f}, "
        f"thr=[{sell_threshold:+.2f},{buy_threshold:+.2f}], rng={range_pos:.2f}, conf={confidence:.2f}, age_s={latest_age_s:.0f}"
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


def position_size(equity: float, price: float, volatility: float, max_risk_per_trade: float = 0.01) -> float:
    vol_clamped = min(max(float(volatility), 0.0045), 0.06)

    if vol_clamped >= 0.030:
        risk_factor = 0.25
    elif vol_clamped >= 0.022:
        risk_factor = 0.40
    elif vol_clamped >= 0.017:
        risk_factor = 0.60
    elif vol_clamped <= 0.007:
        risk_factor = 1.15
    else:
        risk_factor = 1.0

    dynamic_risk = max_risk_per_trade * risk_factor
    dynamic_risk = min(max(dynamic_risk, max_risk_per_trade * 0.25), max_risk_per_trade * 1.20)

    risk_budget = max(equity, 0.0) * dynamic_risk
    risk_per_unit = price * max(vol_clamped, 0.0055)
    if risk_per_unit <= 0:
        return 0.0

    qty = risk_budget / risk_per_unit
    return max(float(qty), 0.0)


def should_enter_trade(ticker: str, has_position: bool = False) -> Dict:
    signal = build_signal(ticker, has_position=has_position)
    return {
        "enter": signal.action in ("buy", "sell"),
        "action": signal.action,
        "price": signal.price,
        "confidence": signal.confidence,
        "score": signal.score,
        "reason": signal.reason,
        "volatility": signal.volatility,
    }
