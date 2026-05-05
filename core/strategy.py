from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from core.scalper_engine import EntrySignal, evaluate_entry_signal, load_btc_scalper_config
from ml.signal_engine import apply_signal_artifact, compute_market_features
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


@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    threshold: float
    entry_signal: EntrySignal
    score: float
    volatility: float
    reason: str
    df: pd.DataFrame


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
    if len(window_24h) < 12 or len(window_72h) < 24:
        return 0.0

    avg_ret_24h = sum(item["ret"] for item in window_24h) / len(window_24h)
    avg_ret_72h = sum(item["ret"] for item in window_72h) / len(window_72h)
    avg_dd_24h = sum(item["dd"] for item in window_24h) / len(window_24h)
    avg_dd_72h = sum(item["dd"] for item in window_72h) / len(window_72h)

    if avg_ret_24h < 0 and avg_ret_24h < avg_ret_72h and avg_dd_24h < avg_dd_72h:
        return min(max(float(os.getenv("BTC_DRIFT_THRESHOLD_PENALTY", "0.02")), 0.0), 0.06)

    return 0.0


def _symbol_profile(symbol: str) -> dict[str, Any]:
    s = symbol.upper()
    if s == "BTC-USD":
        interval = os.getenv("BTC_LIVE_INTERVAL", os.getenv("TRAINING_BAR_INTERVAL", "5m"))
        period = os.getenv("BTC_LIVE_PERIOD", "10d" if interval != "1m" else "3d")
        # Default fallback should be tradeable even before the latest backtest
        # threshold is available. The trainer still overwrites this via
        # `data/backtests/latest.json` once a proper run completes.
        base = float(os.getenv("BTC_BASE_ENTRY_THRESHOLD", "0.12"))
        drift_penalty = _shadow_drift_penalty(s)
        adaptive_base = min(base + drift_penalty, base + 0.06)
        return {
            "interval": interval,
            "period": period,
            "entry_threshold": _load_backtest_threshold(s, adaptive_base, expected_interval=interval),
        }
    return {
        "interval": "5m",
        "period": "5d",
        "entry_threshold": _load_backtest_threshold(s, 0.12, expected_interval="5m"),
    }


def _interval_seconds(interval: str) -> int:
    mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
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


def build_market_snapshot(symbol: str) -> MarketSnapshot:
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
    max_staleness_s = float(os.getenv("MAX_MARKET_DATA_STALENESS_SECONDS", str(max(240, int(bar_seconds * 2.5)))))
    latest_age_s = _latest_bar_age_seconds(df)
    if latest_age_s > max_staleness_s:
        raise ValueError(
            f"Stale market data for {symbol}: last bar age {latest_age_s:.0f}s > {max_staleness_s:.0f}s"
        )

    threshold = float(profile["entry_threshold"])
    config = load_btc_scalper_config(threshold)
    entry_signal = evaluate_entry_signal(df, config)
    row = df.iloc[-1]
    price = float(row["Close"])
    volatility = float(row.get("volatility", 0.0))
    score = float(row.get("score", 0.0))
    reason = (
        f"{entry_signal.reason}, interval={profile['interval']}, age_s={latest_age_s:.0f}, "
        f"profile_thr={threshold:.3f}{artifact_note}"
    )
    return MarketSnapshot(
        symbol=symbol,
        price=price,
        threshold=threshold,
        entry_signal=entry_signal,
        score=score,
        volatility=volatility,
        reason=reason,
        df=df,
    )


def build_signal(symbol: str, has_position: bool = False) -> Signal:
    snapshot = build_market_snapshot(symbol)
    action = "buy" if (snapshot.entry_signal.enter and not has_position) else "hold"
    return Signal(
        symbol=snapshot.symbol,
        action=action,
        price=snapshot.price,
        confidence=snapshot.entry_signal.confidence,
        score=snapshot.score,
        reason=snapshot.reason,
        volatility=snapshot.volatility,
    )


def position_size(
    equity: float,
    price: float,
    volatility: float,
    max_risk_per_trade: float = 0.01,
    stop_loss_pct: float | None = None,
    friction_buffer_pct: float | None = None,
) -> float:
    stop_loss_pct = float(stop_loss_pct if stop_loss_pct is not None else os.getenv("BTC_SCALPER_STOP_LOSS_PCT", "0.01"))
    friction_buffer_pct = float(
        friction_buffer_pct if friction_buffer_pct is not None else os.getenv("BTC_SCALPER_BREAKEVEN_BUFFER_PCT", "0.0075")
    )

    vol_clamped = min(max(float(volatility), 0.0035), 0.04)
    risk_per_unit_pct = max(stop_loss_pct + (friction_buffer_pct * 0.50), vol_clamped * 1.15, 0.0085)
    risk_budget = max(equity, 0.0) * max(float(max_risk_per_trade), 0.0)
    risk_per_unit = max(price, 0.01) * risk_per_unit_pct
    if risk_per_unit <= 0:
        return 0.0
    return max(float(risk_budget / risk_per_unit), 0.0)


def should_enter_trade(ticker: str, has_position: bool = False) -> Dict:
    signal = build_signal(ticker, has_position=has_position)
    return {
        "enter": signal.action == "buy",
        "action": signal.action,
        "price": signal.price,
        "confidence": signal.confidence,
        "score": signal.score,
        "reason": signal.reason,
        "volatility": signal.volatility,
    }
