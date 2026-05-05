from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), low), high))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def infer_bar_seconds(df: pd.DataFrame, fallback: int = 300) -> int:
    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 2:
        return fallback
    deltas = pd.Series(df.index).diff().dropna().dt.total_seconds()
    if deltas.empty:
        return fallback
    bar_seconds = int(deltas.median())
    return max(bar_seconds, 1)


@dataclass(frozen=True)
class BTCScalperConfig:
    # Entry threshold is the main knob the trainer searches over. Every other
    # setting below is intentionally explicit so the lower-cost model can tune
    # one concept at a time without unraveling the rest of the system.
    entry_threshold: float

    # Entry quality gates. Raising these makes the bot more selective; lowering
    # them increases trade count at the cost of accepting weaker setups.
    min_entry_confidence: float = 0.18
    min_take_prob: float = 0.53

    # Core trade management. These are the live rules that the backtest must
    # model exactly, otherwise training/live parity breaks again.
    stop_loss_pct: float = 0.0100
    take_profit1_pct: float = 0.0150
    take_profit1_fraction: float = 0.50
    breakeven_buffer_pct: float = 0.0075
    trailing_stop_pct: float = 0.0080
    hard_target_pct: float = 0.0300
    time_stop_hours: float = 8.0

    # Cooldowns and kill-switches. These are here to stop revenge-trading after
    # bad exits and to enforce the low-frequency BTC scalper mandate.
    cooldown_after_stop_hours: float = 4.0
    cooldown_after_loss_hours: float = 2.0
    cooldown_after_time_stop_hours: float = 1.0
    daily_loss_cap_pct: float = 0.0200
    max_trades_per_day: int = 2
    max_stopouts_per_day: int = 2

    # Regime filters. These are the safest places to adjust when the model is
    # overtrading chop or missing too many legitimate continuations.
    no_trade_atr_pct: float = 0.0120
    hard_risk_off_trend: float = -0.22
    hard_risk_off_mtf_4h: float = -0.20
    overbought_rsi: float = 72.0
    washed_rsi: float = 33.0

    # Entry setup weights. Comment each weight so future edits are deliberate.
    # `setup_score` is the main ranking score for whether a candle deserves an
    # entry. If you want to bias harder toward trend or ML confirmation, edit
    # the relevant weight below instead of adding ad hoc conditions elsewhere.
    weight_score: float = 0.32
    weight_score_ml: float = 0.20
    weight_trend: float = 0.12
    weight_mtf_1h: float = 0.10
    weight_mtf_4h: float = 0.06
    weight_volume: float = 0.08
    weight_momentum_short: float = 0.06
    weight_momentum_medium: float = 0.04
    weight_range_reclaim: float = 0.02


def load_btc_scalper_config(entry_threshold: float) -> BTCScalperConfig:
    return BTCScalperConfig(
        entry_threshold=float(entry_threshold),
        min_entry_confidence=float(os.getenv("BTC_SCALPER_MIN_ENTRY_CONFIDENCE", "0.18")),
        min_take_prob=float(os.getenv("BTC_SCALPER_MIN_TAKE_PROB", "0.53")),
        stop_loss_pct=float(os.getenv("BTC_SCALPER_STOP_LOSS_PCT", "0.01")),
        take_profit1_pct=float(os.getenv("BTC_SCALPER_TP1_PCT", "0.015")),
        take_profit1_fraction=float(os.getenv("BTC_SCALPER_TP1_FRACTION", "0.50")),
        breakeven_buffer_pct=float(os.getenv("BTC_SCALPER_BREAKEVEN_BUFFER_PCT", "0.0075")),
        trailing_stop_pct=float(os.getenv("BTC_SCALPER_TRAILING_STOP_PCT", "0.008")),
        hard_target_pct=float(os.getenv("BTC_SCALPER_HARD_TARGET_PCT", "0.03")),
        time_stop_hours=float(os.getenv("BTC_SCALPER_TIME_STOP_HOURS", "8")),
        cooldown_after_stop_hours=float(os.getenv("BTC_SCALPER_COOLDOWN_STOP_HOURS", "4")),
        cooldown_after_loss_hours=float(os.getenv("BTC_SCALPER_COOLDOWN_LOSS_HOURS", "2")),
        cooldown_after_time_stop_hours=float(os.getenv("BTC_SCALPER_COOLDOWN_TIME_STOP_HOURS", "1")),
        daily_loss_cap_pct=float(os.getenv("BTC_SCALPER_DAILY_LOSS_CAP_PCT", "0.02")),
        max_trades_per_day=int(os.getenv("BTC_SCALPER_MAX_TRADES_PER_DAY", "2")),
        max_stopouts_per_day=int(os.getenv("BTC_SCALPER_MAX_STOPOUTS_PER_DAY", "2")),
        no_trade_atr_pct=float(os.getenv("BTC_SCALPER_NO_TRADE_ATR_PCT", "0.012")),
        hard_risk_off_trend=float(os.getenv("BTC_SCALPER_HARD_RISK_OFF_TREND", "-0.22")),
        hard_risk_off_mtf_4h=float(os.getenv("BTC_SCALPER_HARD_RISK_OFF_MTF_4H", "-0.20")),
        overbought_rsi=float(os.getenv("BTC_SCALPER_OVERBOUGHT_RSI", "72")),
        washed_rsi=float(os.getenv("BTC_SCALPER_WASHED_RSI", "33")),
        weight_score=float(os.getenv("BTC_SCALPER_WEIGHT_SCORE", "0.32")),
        weight_score_ml=float(os.getenv("BTC_SCALPER_WEIGHT_SCORE_ML", "0.20")),
        weight_trend=float(os.getenv("BTC_SCALPER_WEIGHT_TREND", "0.12")),
        weight_mtf_1h=float(os.getenv("BTC_SCALPER_WEIGHT_MTF_1H", "0.10")),
        weight_mtf_4h=float(os.getenv("BTC_SCALPER_WEIGHT_MTF_4H", "0.06")),
        weight_volume=float(os.getenv("BTC_SCALPER_WEIGHT_VOLUME", "0.08")),
        weight_momentum_short=float(os.getenv("BTC_SCALPER_WEIGHT_MOMENTUM_SHORT", "0.06")),
        weight_momentum_medium=float(os.getenv("BTC_SCALPER_WEIGHT_MOMENTUM_MEDIUM", "0.04")),
        weight_range_reclaim=float(os.getenv("BTC_SCALPER_WEIGHT_RANGE_RECLAIM", "0.02")),
    )


@dataclass
class EntrySignal:
    enter: bool
    setup: str
    setup_score: float
    confidence: float
    take_prob: float
    reason: str
    do_not_trade: bool = False


@dataclass
class ManagedTrade:
    symbol: str
    entry_time: str
    entry_price: float
    threshold: float
    setup: str
    setup_score: float
    confidence: float
    entry_cost_pct: float = 0.0
    initial_qty: float = 0.0
    remaining_qty: float = 0.0
    remaining_fraction: float = 1.0
    peak_price: float = 0.0
    tp1_hit: bool = False
    tp1_price: float | None = None
    stop_price: float | None = None
    exit_reason: str | None = None
    realized_return: float = 0.0
    max_return_seen: float = 0.0
    min_return_seen: float = 0.0
    bars_held: int = 0

    @classmethod
    def new(
        cls,
        symbol: str,
        entry_time: datetime,
        entry_price: float,
        threshold: float,
        setup: str,
        setup_score: float,
        confidence: float,
        qty: float,
        config: BTCScalperConfig,
        entry_cost_pct: float = 0.0,
    ) -> "ManagedTrade":
        return cls(
            symbol=symbol,
            entry_time=entry_time.astimezone(timezone.utc).isoformat(),
            entry_price=float(entry_price),
            threshold=float(threshold),
            setup=setup,
            setup_score=float(setup_score),
            confidence=float(confidence),
            entry_cost_pct=float(entry_cost_pct),
            initial_qty=float(qty),
            remaining_qty=float(qty),
            remaining_fraction=1.0,
            peak_price=float(entry_price),
            tp1_hit=False,
            stop_price=float(entry_price) * (1.0 - config.stop_loss_pct),
        )

    def entry_dt(self) -> datetime:
        return datetime.fromisoformat(self.entry_time)


@dataclass
class DailyRiskState:
    day: str
    realized_return: float = 0.0
    trades_opened: int = 0
    stopouts: int = 0
    paused: bool = False
    pause_reason: str | None = None


@dataclass
class TradeAction:
    action: str
    fraction: float
    reason: str
    confidence: float
    setup_score: float
    setup: str = ""
    do_not_trade: bool = False
    updated_trade: ManagedTrade | None = None


@dataclass
class CompletedTrade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    return_pct: float
    max_return_seen: float
    min_return_seen: float
    bars_held: int
    exit_reason: str
    tp1_hit: bool
    setup: str


@dataclass
class BacktestResult:
    total_return: float
    trade_returns: list[float]
    completed_trades: list[CompletedTrade]
    equity_curve: list[float]
    daily_states: list[DailyRiskState] = field(default_factory=list)


def config_as_dict(config: BTCScalperConfig) -> dict[str, Any]:
    return asdict(config)


def _entry_take_prob(row: pd.Series, score_ml: float) -> float:
    if "meta_take_prob" in row:
        return _clip(_safe_float(row["meta_take_prob"], 0.5), 0.001, 0.999)
    return _clip((score_ml + 1.0) * 0.5, 0.001, 0.999)


def evaluate_entry_signal(df: pd.DataFrame, config: BTCScalperConfig) -> EntrySignal:
    if df.empty or len(df) < 3:
        return EntrySignal(
            enter=False,
            setup="insufficient_data",
            setup_score=0.0,
            confidence=0.0,
            take_prob=0.5,
            reason="insufficient feature rows",
            do_not_trade=True,
        )

    row = df.iloc[-1]
    prev = df.iloc[-2]

    score = _safe_float(row.get("score"), 0.0)
    score_ml = _safe_float(row.get("score_ml"), score)
    take_prob = _entry_take_prob(row, score_ml)
    trend = _safe_float(row.get("trend"), 0.0)
    mtf_1h = _safe_float(row.get("mtf_trend_1h"), 0.0)
    mtf_4h = _safe_float(row.get("mtf_trend_4h"), 0.0)
    volume_bias = _safe_float(row.get("volume_bias"), 0.0)
    m3 = _safe_float(row.get("m3"), 0.0)
    m20 = _safe_float(row.get("m20"), 0.0)
    range_score = _safe_float(row.get("range_score"), 0.0)
    atr_pct = _safe_float(row.get("atr_pct"), 0.0)
    rsi = _safe_float(row.get("rsi"), 50.0)
    score_delta = score - _safe_float(prev.get("score"), 0.0)
    trend_delta = trend - _safe_float(prev.get("trend"), 0.0)

    high_vol = bool(atr_pct >= config.no_trade_atr_pct or _safe_float(row.get("regime_high_vol"), 0.0) > 0.5)
    hard_risk_off = bool(trend <= config.hard_risk_off_trend or mtf_4h <= config.hard_risk_off_mtf_4h)
    overbought = bool(rsi >= config.overbought_rsi and m3 > 0.0)

    # Three explicit long-only BTC setups:
    # 1) trend_pullback: strongest setup for a controlled scalper.
    trend_pullback = bool(
        trend > 0.10
        and mtf_1h > -0.02
        and mtf_4h > -0.06
        and -0.26 <= range_score <= 0.08
        and m3 > -0.10
        and 36 <= rsi <= 62
    )
    # 2) breakout_continuation: only allowed with decent volume and trend.
    breakout_continuation = bool(
        trend > 0.16
        and mtf_1h >= 0.02
        and range_score > -0.04
        and volume_bias > -0.05
        and m3 > -0.03
    )
    # 3) washed_reversal: allowed, but only when the higher timeframe is not
    # collapsing. This keeps “catch the knife” behavior constrained.
    washed_reversal = bool(
        rsi <= config.washed_rsi
        and score_delta > 0.02
        and trend > -0.15
        and mtf_4h > -0.10
        and volume_bias > -0.15
        and m3 > -0.08
    )

    setup = "none"
    if trend_pullback:
        setup = "trend_pullback"
    elif breakout_continuation:
        setup = "breakout_continuation"
    elif washed_reversal:
        setup = "washed_reversal"

    # Setup score drives threshold optimization in training.
    setup_score = (
        (config.weight_score * score)
        + (config.weight_score_ml * score_ml)
        + (config.weight_trend * trend)
        + (config.weight_mtf_1h * mtf_1h)
        + (config.weight_mtf_4h * mtf_4h)
        + (config.weight_volume * volume_bias)
        + (config.weight_momentum_short * m3)
        + (config.weight_momentum_medium * m20)
        + (config.weight_range_reclaim * max(-range_score, 0.0))
    )
    if breakout_continuation:
        setup_score += 0.02
    if washed_reversal:
        setup_score -= 0.015

    confidence = (
        0.40 * max((setup_score - config.entry_threshold) / 0.22, 0.0)
        + 0.22 * max((take_prob - 0.5) * 2.0, 0.0)
        + 0.16 * max(trend, 0.0)
        + 0.10 * max(volume_bias, 0.0)
        + 0.07 * max(score_delta * 4.0, 0.0)
        + 0.05 * max(trend_delta * 4.0, 0.0)
    )
    if high_vol:
        confidence -= 0.14
    if overbought:
        confidence -= 0.10
    if hard_risk_off:
        confidence -= 0.25
    confidence = _clip(confidence, 0.0, 1.0)

    do_not_trade = bool(
        setup == "none"
        or take_prob < config.min_take_prob
        or confidence < config.min_entry_confidence
        or high_vol
        or hard_risk_off
        or overbought
    )

    enter = bool((setup_score >= config.entry_threshold) and (not do_not_trade))
    reason = (
        f"setup={setup}, score={setup_score:+.3f}, thr={config.entry_threshold:.3f}, "
        f"conf={confidence:.3f}, take_prob={take_prob:.3f}, trend={trend:+.3f}, "
        f"mtf1h={mtf_1h:+.3f}, mtf4h={mtf_4h:+.3f}, m3={m3:+.3f}, m20={m20:+.3f}, "
        f"range={range_score:+.3f}, vol_bias={volume_bias:+.3f}, atr={atr_pct:.4f}"
    )
    return EntrySignal(
        enter=enter,
        setup=setup,
        setup_score=float(setup_score),
        confidence=float(confidence),
        take_prob=float(take_prob),
        reason=reason,
        do_not_trade=do_not_trade,
    )


def _trade_return_pct(trade: ManagedTrade, price: float) -> float:
    return (float(price) / max(trade.entry_price, 1e-9)) - 1.0


def apply_exit_fill(
    trade: ManagedTrade,
    *,
    price: float,
    fraction: float,
    exit_cost_pct: float,
    remaining_qty: float | None = None,
) -> ManagedTrade:
    updated = ManagedTrade(**asdict(trade))
    clipped_fraction = _clip(fraction, 0.0, max(updated.remaining_fraction, 0.0))
    updated.realized_return += clipped_fraction * (_trade_return_pct(updated, price) - exit_cost_pct)
    updated.remaining_fraction = max(updated.remaining_fraction - clipped_fraction, 0.0)
    if remaining_qty is not None:
        updated.remaining_qty = max(float(remaining_qty), 0.0)
    return updated


def finalize_trade_return(trade: ManagedTrade) -> float:
    return float(trade.realized_return - trade.entry_cost_pct)


def _breakeven_stop_price(trade: ManagedTrade, config: BTCScalperConfig) -> float:
    return trade.entry_price * (1.0 + config.breakeven_buffer_pct)


def _trailing_stop_price(trade: ManagedTrade, config: BTCScalperConfig) -> float:
    peak = max(trade.peak_price, trade.entry_price)
    return peak * (1.0 - config.trailing_stop_pct)


def manage_open_trade(
    df: pd.DataFrame,
    trade: ManagedTrade,
    config: BTCScalperConfig,
) -> TradeAction:
    row = df.iloc[-1]
    now = df.index[-1].to_pydatetime()
    price = _safe_float(row.get("Close"), trade.entry_price)
    score = _safe_float(row.get("score"), 0.0)
    trend = _safe_float(row.get("trend"), 0.0)
    m3 = _safe_float(row.get("m3"), 0.0)
    volume_bias = _safe_float(row.get("volume_bias"), 0.0)
    atr_pct = _safe_float(row.get("atr_pct"), 0.0)

    updated = ManagedTrade(**asdict(trade))
    updated.peak_price = max(updated.peak_price, price)
    updated.bars_held += 1

    current_return = _trade_return_pct(updated, price)
    updated.max_return_seen = max(updated.max_return_seen, current_return)
    updated.min_return_seen = min(updated.min_return_seen, current_return)

    hours_held = max((now.replace(tzinfo=timezone.utc) - updated.entry_dt().astimezone(timezone.utc)).total_seconds() / 3600.0, 0.0)

    if current_return <= -config.stop_loss_pct:
        updated.exit_reason = "stop_loss"
        return TradeAction(
            action="sell",
            fraction=1.0,
            reason=f"stop_loss ret={current_return:+.2%}",
            confidence=0.0,
            setup_score=updated.setup_score,
            setup=updated.setup,
            updated_trade=updated,
        )

    early_invalidation = bool((score < -0.06) and (trend < -0.10) and (m3 < -0.08))
    if (not updated.tp1_hit) and early_invalidation and current_return > -config.stop_loss_pct:
        updated.exit_reason = "signal_invalidation"
        return TradeAction(
            action="sell",
            fraction=1.0,
            reason=f"signal_invalidation ret={current_return:+.2%}",
            confidence=0.0,
            setup_score=updated.setup_score,
            setup=updated.setup,
            updated_trade=updated,
        )

    if (not updated.tp1_hit) and current_return >= config.take_profit1_pct:
        updated.tp1_hit = True
        updated.tp1_price = price
        updated.remaining_fraction = max(1.0 - config.take_profit1_fraction, 0.0)
        updated.stop_price = max(_breakeven_stop_price(updated, config), _trailing_stop_price(updated, config))
        fraction = min(config.take_profit1_fraction, 1.0)
        return TradeAction(
            action="sell_partial",
            fraction=fraction,
            reason=f"tp1 ret={current_return:+.2%}",
            confidence=0.0,
            setup_score=updated.setup_score,
            setup=updated.setup,
            updated_trade=updated,
        )

    if current_return >= config.hard_target_pct:
        updated.exit_reason = "hard_target"
        return TradeAction(
            action="sell",
            fraction=1.0,
            reason=f"hard_target ret={current_return:+.2%}",
            confidence=0.0,
            setup_score=updated.setup_score,
            setup=updated.setup,
            updated_trade=updated,
        )

    if updated.tp1_hit:
        trail_stop = max(_breakeven_stop_price(updated, config), _trailing_stop_price(updated, config))
        updated.stop_price = trail_stop
        if price <= trail_stop:
            updated.exit_reason = "trail_or_breakeven"
            return TradeAction(
                action="sell",
                fraction=1.0,
                reason=f"trail_or_breakeven ret={current_return:+.2%}",
                confidence=0.0,
                setup_score=updated.setup_score,
                setup=updated.setup,
                updated_trade=updated,
            )
        if (score < -0.01) and (trend < -0.04) and (volume_bias < -0.08):
            updated.exit_reason = "post_tp1_signal_break"
            return TradeAction(
                action="sell",
                fraction=1.0,
                reason=f"post_tp1_signal_break ret={current_return:+.2%}",
                confidence=0.0,
                setup_score=updated.setup_score,
                setup=updated.setup,
                updated_trade=updated,
            )

    if hours_held >= config.time_stop_hours:
        updated.exit_reason = "time_stop"
        return TradeAction(
            action="sell",
            fraction=1.0,
            reason=f"time_stop hours={hours_held:.1f} ret={current_return:+.2%}",
            confidence=0.0,
            setup_score=updated.setup_score,
            setup=updated.setup,
            updated_trade=updated,
        )

    hold_reason = (
        f"manage hold ret={current_return:+.2%}, peak={updated.peak_price:.2f}, "
        f"tp1_hit={updated.tp1_hit}, atr={atr_pct:.4f}, score={score:+.3f}, trend={trend:+.3f}"
    )
    return TradeAction(
        action="hold",
        fraction=0.0,
        reason=hold_reason,
        confidence=0.0,
        setup_score=updated.setup_score,
        setup=updated.setup,
        updated_trade=updated,
    )


def _new_day_state(ts: datetime) -> DailyRiskState:
    return DailyRiskState(day=ts.astimezone(timezone.utc).date().isoformat())


def roll_daily_state(
    current: DailyRiskState | None,
    ts: datetime,
) -> DailyRiskState:
    day = ts.astimezone(timezone.utc).date().isoformat()
    if current is None or current.day != day:
        return _new_day_state(ts)
    return current


def can_open_new_trade(
    daily: DailyRiskState,
    cooldown_until: datetime | None,
    now: datetime,
    config: BTCScalperConfig,
) -> tuple[bool, str]:
    if daily.paused:
        return False, f"daily_paused:{daily.pause_reason}"
    if daily.realized_return <= -abs(config.daily_loss_cap_pct):
        return False, "daily_loss_cap"
    if daily.trades_opened >= config.max_trades_per_day:
        return False, "trade_cap"
    if daily.stopouts >= config.max_stopouts_per_day:
        return False, "stopout_cap"
    if cooldown_until and now < cooldown_until:
        return False, f"cooldown_until={cooldown_until.isoformat()}"
    return True, ""


def update_daily_after_closed_trade(
    daily: DailyRiskState,
    trade_return: float,
    exit_reason: str,
    config: BTCScalperConfig,
) -> DailyRiskState:
    daily.realized_return += float(trade_return)
    if exit_reason == "stop_loss":
        daily.stopouts += 1
    if daily.realized_return <= -abs(config.daily_loss_cap_pct):
        daily.paused = True
        daily.pause_reason = "daily_loss_cap"
    elif daily.stopouts >= config.max_stopouts_per_day:
        daily.paused = True
        daily.pause_reason = "stopout_cap"
    return daily


def cooldown_for_exit_reason(
    exit_reason: str,
    config: BTCScalperConfig,
) -> timedelta:
    if exit_reason == "stop_loss":
        return timedelta(hours=config.cooldown_after_stop_hours)
    if exit_reason in {"signal_invalidation", "time_stop"}:
        return timedelta(hours=config.cooldown_after_time_stop_hours)
    if exit_reason in {"trail_or_breakeven", "post_tp1_signal_break"}:
        return timedelta(hours=max(config.cooldown_after_time_stop_hours * 0.5, 0.25))
    return timedelta(hours=config.cooldown_after_loss_hours if "loss" in exit_reason else 0.0)


def run_backtest(
    df: pd.DataFrame,
    *,
    symbol: str,
    threshold: float,
    fee_bps: float = 4.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    if df.empty:
        return BacktestResult(total_return=0.0, trade_returns=[], completed_trades=[], equity_curve=[1.0])

    config = load_btc_scalper_config(entry_threshold=threshold)
    friction = (float(fee_bps) + float(slippage_bps)) / 10_000.0

    equity = 1.0
    equity_curve = [equity]
    position_fraction = 0.0
    trade: ManagedTrade | None = None
    cooldown_until: datetime | None = None
    daily: DailyRiskState | None = None
    daily_snapshots: list[DailyRiskState] = []
    trade_returns: list[float] = []
    completed: list[CompletedTrade] = []

    for i in range(1, len(df)):
        ts = df.index[i].to_pydatetime().replace(tzinfo=timezone.utc)
        daily = roll_daily_state(daily, ts)

        prev_close = _safe_float(df["Close"].iloc[i - 1], 0.0)
        close = _safe_float(df["Close"].iloc[i], prev_close)
        if position_fraction > 0 and prev_close > 0:
            equity *= 1.0 + (position_fraction * ((close / prev_close) - 1.0))

        window = df.iloc[: i + 1]

        if trade is not None:
            action = manage_open_trade(window, trade, config)
            trade = action.updated_trade or trade
            if action.action == "sell_partial":
                delta = min(action.fraction, position_fraction)
                if delta > 0:
                    equity *= 1.0 - (delta * friction)
                    trade = apply_exit_fill(
                        trade,
                        price=close,
                        fraction=delta,
                        exit_cost_pct=friction,
                        remaining_qty=trade.initial_qty * max(position_fraction - delta, 0.0),
                    )
                    position_fraction = max(position_fraction - delta, 0.0)
                    trade.remaining_fraction = position_fraction
            elif action.action == "sell":
                delta = position_fraction
                if delta > 0:
                    equity *= 1.0 - (delta * friction)
                    trade = apply_exit_fill(
                        trade,
                        price=close,
                        fraction=delta,
                        exit_cost_pct=friction,
                        remaining_qty=0.0,
                    )
                total_trade_return = finalize_trade_return(trade)
                position_fraction = 0.0
                daily = update_daily_after_closed_trade(daily, total_trade_return, trade.exit_reason or "sell", config)
                if trade.exit_reason:
                    cooldown_until = ts + cooldown_for_exit_reason(trade.exit_reason, config)
                trade_returns.append(total_trade_return)
                completed.append(
                    CompletedTrade(
                        entry_time=trade.entry_time,
                        exit_time=ts.isoformat(),
                        entry_price=trade.entry_price,
                        exit_price=close,
                        return_pct=float(total_trade_return),
                        max_return_seen=float(trade.max_return_seen),
                        min_return_seen=float(trade.min_return_seen),
                        bars_held=int(trade.bars_held),
                        exit_reason=str(trade.exit_reason or "sell"),
                        tp1_hit=bool(trade.tp1_hit),
                        setup=str(trade.setup),
                    )
                )
                trade = None
        else:
            allowed, deny_reason = can_open_new_trade(daily, cooldown_until, ts, config)
            if allowed:
                signal = evaluate_entry_signal(window, config)
                if signal.enter:
                    trade = ManagedTrade.new(
                        symbol=symbol,
                        entry_time=ts,
                        entry_price=close,
                        threshold=config.entry_threshold,
                        setup=signal.setup,
                        setup_score=signal.setup_score,
                        confidence=signal.confidence,
                        qty=1.0,
                        config=config,
                        entry_cost_pct=friction,
                    )
                    daily.trades_opened += 1
                    position_fraction = 1.0
                    equity *= 1.0 - friction
            else:
                if deny_reason in {"daily_loss_cap", "trade_cap", "stopout_cap"}:
                    daily.paused = True
                    daily.pause_reason = deny_reason

        equity_curve.append(equity)
        daily_snapshots.append(DailyRiskState(**asdict(daily)))

    return BacktestResult(
        total_return=float(equity_curve[-1] - 1.0),
        trade_returns=trade_returns,
        completed_trades=completed,
        equity_curve=equity_curve,
        daily_states=daily_snapshots,
    )
