from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Ensure project root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discord.notify import send_training_update  # noqa: E402
from scripts.train_backtest import run as run_training  # noqa: E402
from services.progress import record_event  # noqa: E402


DEFAULT_INTERVAL_MINUTES = int(os.getenv("TRAINING_INTERVAL_MINUTES", "360"))
DEFAULT_JITTER_MINUTES = float(os.getenv("TRAINING_JITTER_MINUTES", "10"))
DEFAULT_SYMBOL = os.getenv("TRAINING_SYMBOL", "BTC-USD")
DEFAULT_INTERVAL = os.getenv("TRAINING_BAR_INTERVAL", "5m")
DEFAULT_PERIOD = os.getenv("TRAINING_PERIOD", "60d")
TRAINING_LABEL = (os.getenv("TRAINING_LABEL") or os.getenv("TRAINING_VARIANT") or "").strip()
TRAINING_MODE = (os.getenv("TRAINING_MODE") or "").strip()
LABEL_PREFIX = f"[{TRAINING_LABEL}] " if TRAINING_LABEL else ""
NOTIFICATION_LABEL = TRAINING_LABEL or TRAINING_MODE or ""


def _clamp_error_message(message: str, limit: int = 320) -> str:
    return message if len(message) <= limit else f"{message[: limit - 1]}…"


def run_loop(
    interval_minutes: float = DEFAULT_INTERVAL_MINUTES,
    jitter_minutes: float = DEFAULT_JITTER_MINUTES,
    symbol: str = DEFAULT_SYMBOL,
    bar_interval: str = DEFAULT_INTERVAL,
    period: str = DEFAULT_PERIOD,
    oneshot: bool = False,
):
    interval_minutes = max(float(interval_minutes), 1.0)
    jitter_minutes = max(float(jitter_minutes), 0.0)

    loop = 0
    consecutive_failures = 0

    while True:
        loop += 1
        started = datetime.utcnow().isoformat()
        print(
            f"{LABEL_PREFIX}[{started}] 🧠 Training daemon loop {loop} start | symbol={symbol} interval={bar_interval} period={period}"
        )

        event_data = {
            "loop": loop,
            "symbol": symbol,
            "interval": bar_interval,
            "period": period,
            "oneshot": oneshot,
        }
        if NOTIFICATION_LABEL:
            event_data["label"] = NOTIFICATION_LABEL

        record_event(
            "training_start",
            message=f"training loop {loop} begin",
            data=event_data,
        )

        try:
            run_training(symbol=symbol, interval=bar_interval, period=period)
            success_data = {
                "loop": loop,
                "symbol": symbol,
                "interval": bar_interval,
                "period": period,
            }
            if NOTIFICATION_LABEL:
                success_data["label"] = NOTIFICATION_LABEL

            record_event(
                "training_success",
                message=f"training loop {loop} completed",
                data=success_data,
            )
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            consecutive_failures += 1
            now = datetime.utcnow().isoformat()
            err_message = f"{exc.__class__.__name__}: {exc}"
            print(f"{LABEL_PREFIX}[{now}] ⚠️ Training run failed: {err_message}")
            traceback.print_exc()

            clamped = _clamp_error_message(err_message)
            failure_data = {
                "loop": loop,
                "symbol": symbol,
                "interval": bar_interval,
                "period": period,
                "consecutive_failures": consecutive_failures,
            }
            if NOTIFICATION_LABEL:
                failure_data["label"] = NOTIFICATION_LABEL

            record_event(
                "training_failure",
                message=clamped,
                data=failure_data,
            )

            if consecutive_failures in {1, 3, 6} or consecutive_failures % 10 == 0:
                note = _clamp_error_message(
                    f"training daemon warning x{consecutive_failures}: {err_message}"
                )
                try:
                    send_training_update(f"⚠️ {note}", label=NOTIFICATION_LABEL or None)
                except Exception as notify_err:  # noqa: BLE001
                    print(
                        f"{LABEL_PREFIX}[{datetime.utcnow().isoformat()}] ⚠️ Failed to send training warning webhook: {notify_err}"
                    )
                    alert_error_data = {
                        "loop": loop,
                        "symbol": symbol,
                        "interval": bar_interval,
                        "period": period,
                        "consecutive_failures": consecutive_failures,
                    }
                    if NOTIFICATION_LABEL:
                        alert_error_data["label"] = NOTIFICATION_LABEL

                    record_event(
                        "training_alert_error",
                        message=f"failed to send training alert: {notify_err}",
                        data=alert_error_data,
                    )
                else:
                    alert_data = {
                        "loop": loop,
                        "symbol": symbol,
                        "interval": bar_interval,
                        "period": period,
                        "consecutive_failures": consecutive_failures,
                    }
                    if NOTIFICATION_LABEL:
                        alert_data["label"] = NOTIFICATION_LABEL

                    record_event(
                        "training_alert",
                        message=note,
                        data=alert_data,
                    )
        else:
            print(f"{LABEL_PREFIX}[{datetime.utcnow().isoformat()}] ✅ Training run completed")

        if oneshot:
            break

        sleep_seconds = max(interval_minutes * 60.0, 60.0)
        if consecutive_failures > 0:
            # Exponential backoff to avoid hammering upstream services during outages.
            backoff_factor = min(consecutive_failures, 6)
            backoff_seconds = min(3600.0 * 6, (2**backoff_factor - 1) * 120.0)
            sleep_seconds += backoff_seconds
        else:
            backoff_seconds = 0.0
        jitter_seconds = 0.0
        if jitter_minutes > 0:
            jitter_seconds = random.uniform(-jitter_minutes * 60.0, jitter_minutes * 60.0)
        total_sleep = max(60.0, sleep_seconds + jitter_seconds)
        print(
            f"{LABEL_PREFIX}[{datetime.utcnow().isoformat()}] 💤 Sleeping {total_sleep / 60:.1f} minutes "
            f"(failures={consecutive_failures})"
        )
        sleep_data = {
            "loop": loop,
            "symbol": symbol,
            "interval": bar_interval,
            "period": period,
            "sleep_seconds": total_sleep,
            "backoff_seconds": backoff_seconds,
            "jitter_seconds": jitter_seconds,
            "consecutive_failures": consecutive_failures,
        }
        if NOTIFICATION_LABEL:
            sleep_data["label"] = NOTIFICATION_LABEL

        record_event(
            "training_sleep",
            message=f"training loop {loop} sleep",
            data=sleep_data,
        )
        time.sleep(total_sleep)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous training/backtest loop")
    parser.add_argument("--interval-minutes", type=float, default=DEFAULT_INTERVAL_MINUTES)
    parser.add_argument("--jitter-minutes", type=float, default=DEFAULT_JITTER_MINUTES)
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--bar-interval", type=str, default=DEFAULT_INTERVAL)
    parser.add_argument("--period", type=str, default=DEFAULT_PERIOD)
    parser.add_argument("--oneshot", action="store_true", help="Run a single iteration then exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    run_loop(
        interval_minutes=args.interval_minutes,
        jitter_minutes=args.jitter_minutes,
        symbol=args.symbol,
        bar_interval=args.bar_interval,
        period=args.period,
        oneshot=args.oneshot,
    )


if __name__ == "__main__":
    main()
