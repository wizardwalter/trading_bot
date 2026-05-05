from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from services.market_data import normalize_ohlcv_frame


ARTIFACT_VERSION = 1
DEFAULT_SIGNAL_FEATURE_COLUMNS = [
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


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact feature frame used by both backtests and live execution."""
    out = normalize_ohlcv_frame(df)

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

    out["ema_ratio"] = ((out["Close"] / out["ema_slow"]) - 1).clip(-0.2, 0.2)
    out["macd_hist"] = ((out["ema_fast"] - out["ema_slow"]) / out["Close"]).clip(-0.2, 0.2)
    out["price_momentum"] = out["Close"].pct_change().rolling(6).mean().clip(-0.05, 0.05)

    vol_growth = out["Volume"].pct_change(36).replace([np.inf, -np.inf], np.nan)
    out["volume_trend"] = vol_growth.ewm(span=24, adjust=False).mean().clip(-4, 4).fillna(0.0)

    volume = out["Volume"].ffill()
    vol_mean = volume.rolling(96).mean()
    vol_std = volume.rolling(96).std().replace(0, np.nan)
    out["volume_z"] = ((volume - vol_mean) / vol_std).clip(-3, 3).fillna(0.0)

    high_low = out["High"] - out["Low"]
    prev_close = out["Close"].shift(1)
    true_range = pd.concat(
        [
            high_low,
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(48).mean()
    out["atr_pct"] = (
        (atr / out["Close"])
        .fillna((high_low / out["Close"]).rolling(12).mean())
        .fillna(0.004)
    )

    intraday_position = ((out["Close"] - out["Low"]) / (high_low.replace(0, np.nan))).clip(0, 1) - 0.5
    out["range_score"] = intraday_position.rolling(12).mean().fillna(0.0)

    out["ema_fast_2"] = out["Close"].ewm(span=24, adjust=False).mean()
    out["ema_slow_2"] = out["Close"].ewm(span=52, adjust=False).mean()
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

    hour = out.index.hour
    dow = out.index.dayofweek
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

    return out.dropna().copy()


def apply_signal_artifact(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Replay the saved ML blend on fresh live features."""
    if not artifact or int(artifact.get("version", 0)) != ARTIFACT_VERSION:
        return df

    feature_cols = artifact.get("feature_cols") or DEFAULT_SIGNAL_FEATURE_COLUMNS
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"ML artifact cannot be applied; missing feature columns: {missing_cols}")

    base_score = df["score"].clip(-1.0, 1.0)
    raw_combo = None
    smooth_combo = None
    prob_combo = None
    sum_weights = 0.0

    for candidate in artifact.get("candidates", []):
        pipeline = candidate.get("pipeline")
        if pipeline is None:
            continue

        probs = pipeline.predict_proba(df[feature_cols])[:, 1]
        calibrator = candidate.get("calibrator")
        if calibrator is not None:
            try:
                probs = calibrator.predict(np.asarray(probs, dtype=float))
            except Exception:
                pass

        prob_series = pd.Series(np.clip(np.asarray(probs, dtype=float), 0.001, 0.999), index=df.index)
        raw = ((prob_series - 0.5) * 2.0).clip(-1.0, 1.0)
        smooth_span = int(candidate.get("smooth_span", 18))
        smooth = raw.ewm(span=smooth_span, adjust=False).mean().clip(-1.0, 1.0)

        weight = float(candidate.get("blend_weight", 0.0))
        if weight <= 0:
            continue

        sum_weights += weight
        raw_combo = raw * weight if raw_combo is None else raw_combo + (raw * weight)
        smooth_combo = smooth * weight if smooth_combo is None else smooth_combo + (smooth * weight)
        prob_combo = prob_series * weight if prob_combo is None else prob_combo + (prob_series * weight)

    if sum_weights <= 0 or raw_combo is None or smooth_combo is None or prob_combo is None:
        return df

    combined_raw = (raw_combo / sum_weights).clip(-1.0, 1.0)
    combined_smoothed = (smooth_combo / sum_weights).clip(-1.0, 1.0)
    combined_prob = (prob_combo / sum_weights).clip(0.001, 0.999)

    base_std = float(base_score.std()) if hasattr(base_score, "std") else 0.0
    ml_std = float(combined_smoothed.std()) if hasattr(combined_smoothed, "std") else 0.0
    if ml_std > 0 and base_std > 0:
        scale = float(np.clip(base_std / ml_std, 0.6, 1.6))
        combined_raw = (combined_raw * scale).clip(-1.0, 1.0)
        combined_smoothed = (combined_smoothed * scale).clip(-1.0, 1.0)

    effective_weight = float(artifact.get("effective_weight", min(0.45, sum_weights)))
    out = df.copy()
    out["score_ml_raw"] = combined_raw
    out["score_ml"] = combined_smoothed
    out["meta_take_prob"] = combined_prob.ewm(span=16, adjust=False).mean().clip(0.001, 0.999)
    out["score"] = (effective_weight * combined_smoothed) + ((1.0 - effective_weight) * base_score)
    return out


def compute_target_position(df: pd.DataFrame, threshold: float, long_only_mode: bool | None = None) -> np.ndarray:
    if long_only_mode is None:
        long_only_mode = str(os.getenv("TRAINING_LONG_ONLY", "1")).strip().lower() in {"1", "true", "yes", "on"}

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

    overbought = rsi > 71
    oversold = rsi < 19

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

    bullish_confirmation = (
        (trend > -0.01)
        & (m20 > -0.05)
        & (m3 > -0.13)
        & (mtf_1h > -0.15)
        & (mtf_4h > -0.20)
    )
    bearish_confirmation = (
        (trend < 0.01)
        & (m20 < 0.05)
        & (m3 < 0.13)
        & (mtf_1h < 0.15)
        & (mtf_4h < 0.20)
    )

    ml_raw_abs = np.abs(score_ml_raw)
    low_confidence = ml_raw_abs < np.maximum(0.06, threshold * 0.20)
    weak_trend = np.abs(trend) < 0.05
    volatility_block = atr_pct > np.nanpercentile(atr_pct, 97)

    if "meta_take_prob" in df.columns:
        meta_take_prob = np.clip(df["meta_take_prob"].values, 0.001, 0.999)
    else:
        meta_take_prob = np.clip((score_ml + 1.0) * 0.5, 0.001, 0.999)
    min_take_prob = np.where(high_vol_regime, 0.60, 0.53)
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
    if long_only_mode:
        short_entry = np.zeros_like(short_entry, dtype=bool)

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


def derive_live_decision(df: pd.DataFrame, threshold: float, has_position: bool) -> dict[str, Any]:
    if df.empty:
        return {
            "action": "hold",
            "confidence": 0.0,
            "score": 0.0,
            "reason": "feature frame empty",
            "volatility": 0.0,
        }

    desired = compute_target_position(df, threshold)
    latest_desired = int(desired[-1]) if len(desired) else 0
    prev_desired = int(desired[-2]) if len(desired) > 1 else 0
    row = df.iloc[-1]

    score = float(row.get("score", 0.0))
    score_ml = float(row.get("score_ml", score))
    take_prob = float(row.get("meta_take_prob", np.clip((score_ml + 1.0) * 0.5, 0.001, 0.999)))
    volatility = float(row.get("volatility", 0.0))

    if latest_desired > 0 and not has_position:
        action = "buy"
    elif latest_desired <= 0 and has_position:
        action = "sell"
    else:
        action = "hold"

    confidence = float(
        np.clip(
            0.40 * min(abs(score) / max(threshold, 0.05), 1.5)
            + 0.35 * min(abs(score_ml) / max(threshold * 0.8, 0.04), 1.5)
            + 0.25 * abs(take_prob - 0.5) * 2.0,
            0.0,
            1.0,
        )
    )
    reason = (
        f"desired={latest_desired}, prev={prev_desired}, score={score:+.3f}, "
        f"score_ml={score_ml:+.3f}, take_prob={take_prob:.3f}, threshold={threshold:.3f}, "
        f"vol={volatility:.4f}"
    )

    return {
        "action": action,
        "confidence": confidence,
        "score": score,
        "reason": reason,
        "volatility": volatility,
    }
