from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from ml.signal_engine import apply_signal_artifact, compute_market_features, derive_live_decision
from services.market_data import download_market_data


LIVE_SIGNAL_ARTIFACT_PATH = Path(os.getenv("LIVE_SIGNAL_ARTIFACT_PATH", "models/live_signal_artifact.pkl"))
_LIVE_SIGNAL_CACHE: dict[str, object | None] = {"mtime": None, "artifact": None}


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
    return download_market_data(
        symbol=symbol,
        interval=interval,
        period=period,
        retries=retries,
        source_pref=os.getenv("LIVE_DATA_SOURCE") or os.getenv("MARKET_DATA_SOURCE"),
    )


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return compute_market_features(df)


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

    if expected_interval and str(payload.get("interval", "")).lower() != str(expected_interval).lower():
        return fallback

    max_age_hours = float(os.getenv("BACKTEST_MAX_AGE_HOURS", "72"))
    try:
        age_s = time.time() - path.stat().st_mtime
        if max_age_hours > 0 and age_s > (max_age_hours * 3600):
            return fallback
    except Exception:
        return fallback

    test_block = payload.get("test", {}) if isinstance(payload.get("test", {}), dict) else {}
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
    window_7d = _window(24 * 7)
    if len(window_24h) < 12 or len(window_72h) < 24 or len(window_7d) < 48:
        return 0.0

    avg_ret_24h = sum(item["ret"] for item in window_24h) / len(window_24h)
    avg_ret_72h = sum(item["ret"] for item in window_72h) / len(window_72h)
    avg_ret_7d = sum(item["ret"] for item in window_7d) / len(window_7d)
    avg_dd_24h = sum(item["dd"] for item in window_24h) / len(window_24h)
    avg_dd_72h = sum(item["dd"] for item in window_72h) / len(window_72h)
    avg_dd_7d = sum(item["dd"] for item in window_7d) / len(window_7d)

    drift_negative = avg_ret_24h < 0 and avg_ret_24h < avg_ret_72h
    drawdown_worse = avg_dd_24h < avg_dd_72h
    broad_deterioration = avg_ret_72h < avg_ret_7d and avg_dd_72h < avg_dd_7d

    if drift_negative and drawdown_worse:
        default_penalty = 0.04 if broad_deterioration else 0.02
        ret_gap = max(avg_ret_72h - avg_ret_24h, 0.0)
        dd_gap = max(avg_dd_72h - avg_dd_24h, 0.0)
        severe_decay = (ret_gap >= 0.0025) and (dd_gap >= 0.0025)
        sustained_stress = broad_deterioration and (avg_ret_24h <= -0.02) and (avg_dd_24h <= -0.04)
        penalty = default_penalty + (0.015 if severe_decay else 0.0) + (0.01 if sustained_stress else 0.0)

        env_penalty = float(os.getenv("BTC_DRIFT_THRESHOLD_PENALTY", str(penalty)))
        return min(max(env_penalty, 0.0), 0.06)

    return 0.0


def _symbol_profile(symbol: str) -> dict:
    s = symbol.upper()
    if s == "BTC-USD":
        interval = os.getenv("BTC_LIVE_INTERVAL", os.getenv("TRAINING_BAR_INTERVAL", "5m"))
        period = os.getenv("BTC_LIVE_PERIOD", "10d" if interval != "1m" else "3d")
        base = float(os.getenv("BTC_BASE_ENTRY_THRESHOLD", "0.16"))
        drift_penalty = _shadow_drift_penalty(s)
        adaptive_base = min(base + drift_penalty, base + 0.06)
        return {
            "interval": interval,
            "period": period,
            "drift_penalty": drift_penalty,
            "entry_threshold": _load_backtest_threshold(s, adaptive_base, expected_interval=interval),
        }

    interval = os.getenv("EQUITY_LIVE_INTERVAL", "5m")
    return {
        "interval": interval,
        "period": os.getenv("EQUITY_LIVE_PERIOD", "5d"),
        "entry_threshold": _load_backtest_threshold(s, 0.12, expected_interval=interval),
        "drift_penalty": 0.0,
    }


def _interval_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "60m": 3600,
        "90m": 5400,
        "1h": 3600,
        "1d": 86400,
        "daily": 86400,
    }
    return mapping.get(str(interval).lower(), 300)


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


def _load_live_signal_artifact(symbol: str, interval: str) -> dict | None:
    if os.getenv("USE_LIVE_SIGNAL_ARTIFACT", "1") != "1":
        return None
    if not LIVE_SIGNAL_ARTIFACT_PATH.exists():
        return None

    try:
        mtime = LIVE_SIGNAL_ARTIFACT_PATH.stat().st_mtime
    except Exception:
        return None

    cached_mtime = _LIVE_SIGNAL_CACHE.get("mtime")
    artifact = _LIVE_SIGNAL_CACHE.get("artifact")
    if artifact is None or cached_mtime != mtime:
        try:
            artifact = joblib.load(LIVE_SIGNAL_ARTIFACT_PATH)
        except Exception:
            return None
        _LIVE_SIGNAL_CACHE["mtime"] = mtime
        _LIVE_SIGNAL_CACHE["artifact"] = artifact

    if not isinstance(artifact, dict):
        return None
    if str(artifact.get("symbol", "")).upper() not in {"", symbol.upper()}:
        return None
    if str(artifact.get("interval", "")).lower() not in {"", str(interval).lower()}:
        return None
    return artifact


def build_signal(symbol: str, has_position: bool = False) -> Signal:
    profile = _symbol_profile(symbol)
    df = _features(_download(symbol, interval=profile["interval"], period=profile["period"]))

    artifact = _load_live_signal_artifact(symbol, profile["interval"])
    artifact_note = ""
    if artifact is not None:
        try:
            df = apply_signal_artifact(df, artifact)
            artifact_note = f", artifact={artifact.get('variant', 'ml')}"
        except Exception as exc:
            artifact_note = f", artifact_error={exc}"

    bar_seconds = _interval_seconds(profile["interval"])
    default_staleness = max(240, int(bar_seconds * 2.5))
    max_staleness_s = float(os.getenv("MAX_MARKET_DATA_STALENESS_SECONDS", str(default_staleness)))
    latest_age_s = _latest_bar_age_seconds(df)
    if latest_age_s > max_staleness_s:
        raise ValueError(
            f"Stale market data for {symbol}: last bar age {latest_age_s:.0f}s > {max_staleness_s:.0f}s"
        )

    row = df.iloc[-1]
    price = float(row["Close"])
    decision = derive_live_decision(df, threshold=float(profile["entry_threshold"]), has_position=has_position)
    reason = (
        f"{decision['reason']}, interval={profile['interval']}, age_s={latest_age_s:.0f}, "
        f"profile_thr={float(profile['entry_threshold']):.3f}{artifact_note}"
    )

    return Signal(
        symbol=symbol,
        action=str(decision["action"]),
        price=price,
        confidence=float(decision["confidence"]),
        score=float(decision["score"]),
        reason=reason,
        volatility=float(decision["volatility"]),
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
