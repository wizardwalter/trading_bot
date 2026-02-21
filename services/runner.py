from __future__ import annotations

import os
import time
from datetime import datetime

from core.exe import run_bot
from core.performance import compute_performance
from discord.notify import send_status_update
from services.progress import record_event


def run_loop(interval_seconds: int = 60, execute_orders: bool = False, status_every_loops: int = 5):
    loop_num = 0
    consecutive_failures = 0

    while True:
        loop_num += 1
        started = datetime.utcnow().isoformat()
        print(f"[{started}] 🔁 Loop {loop_num} start")

        record_event(
            "loop_start",
            message=f"loop {loop_num} begin",
            data={
                "loop": loop_num,
                "execute_orders": execute_orders,
                "interval_seconds": interval_seconds,
            },
        )

        try:
            run_bot(paper_mode=True, execute_orders=execute_orders)
            if consecutive_failures >= 3:
                recovery_message = f"loop recovered after {consecutive_failures} failures"
                send_status_update(recovery_message)
                record_event(
                    "loop_recovered",
                    message=recovery_message,
                    data={"loop": loop_num, "recoveries": consecutive_failures},
                )
            record_event(
                "loop_success",
                message=f"loop {loop_num} completed",
                data={
                    "loop": loop_num,
                    "execute_orders": execute_orders,
                },
            )
            consecutive_failures = 0
        except Exception as e:  # noqa: BLE001
            consecutive_failures += 1
            err_message = f"{e.__class__.__name__}: {e}"
            print(f"[{datetime.utcnow().isoformat()}] ⚠️ loop error: {err_message}")

            record_event(
                "loop_failure",
                message=err_message,
                data={
                    "loop": loop_num,
                    "execute_orders": execute_orders,
                    "consecutive_failures": consecutive_failures,
                },
            )

            # Rate-limit noisy warning pushes during extended upstream outages.
            if consecutive_failures in {1, 3, 10} or consecutive_failures % 25 == 0:
                warning = (
                    f"loop warning x{consecutive_failures}: {e.__class__.__name__} (will retry with backoff)"
                )
                send_status_update(warning)
                record_event(
                    "loop_alert",
                    message=warning,
                    data={
                        "loop": loop_num,
                        "execute_orders": execute_orders,
                        "consecutive_failures": consecutive_failures,
                    },
                )

        status_message = None
        if loop_num % max(status_every_loops, 1) == 0:
            use_db_perf = os.getenv("ENABLE_DB_PERF", "0") == "1"
            if use_db_perf:
                try:
                    perf = compute_performance()
                    status_message = (
                        f"Loop {loop_num}: trades={perf.trade_count}, win_rate={perf.win_rate:.1%}, "
                        f"gross_pnl=${perf.gross_pnl:.2f}"
                    )
                    send_status_update(status_message)
                    record_event(
                        "loop_status",
                        message=status_message,
                        data={
                            "loop": loop_num,
                            "source": "db",
                            "trade_count": perf.trade_count,
                            "win_rate": float(perf.win_rate),
                            "gross_pnl": float(perf.gross_pnl),
                        },
                    )
                except Exception as perf_err:  # noqa: BLE001
                    status_message = f"Loop {loop_num}: running (DB perf unavailable)"
                    send_status_update(status_message)
                    record_event(
                        "loop_status",
                        message=status_message,
                        data={
                            "loop": loop_num,
                            "source": "db_error",
                            "error": str(perf_err),
                        },
                    )
            else:
                status_message = f"Loop {loop_num}: running"
                send_status_update(status_message)
                record_event(
                    "loop_status",
                    message=status_message,
                    data={
                        "loop": loop_num,
                        "source": "heartbeat",
                    },
                )

        backoff_s = 0
        if consecutive_failures > 0:
            # Exponential backoff capped to keep heartbeat alive while reducing API thrash.
            backoff_s = min(300, (2 ** min(consecutive_failures, 6)) - 1)

        sleep_s = interval_seconds + backoff_s
        print(f"[{datetime.utcnow().isoformat()}] 💤 sleeping {sleep_s}s")
        record_event(
            "loop_sleep",
            message=f"loop {loop_num} sleep",
            data={
                "loop": loop_num,
                "sleep_seconds": sleep_s,
                "consecutive_failures": consecutive_failures,
            },
        )
        time.sleep(sleep_s)
