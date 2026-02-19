from __future__ import annotations

import os
import time
from datetime import datetime

from core.exe import run_bot
from core.performance import compute_performance
from discord.notify import send_status_update


def run_loop(interval_seconds: int = 60, execute_orders: bool = False, status_every_loops: int = 5):
    loop_num = 0
    consecutive_failures = 0

    while True:
        loop_num += 1
        started = datetime.utcnow().isoformat()
        print(f"[{started}] 🔁 Loop {loop_num} start")

        try:
            run_bot(paper_mode=True, execute_orders=execute_orders)
            if consecutive_failures >= 3:
                send_status_update(f"loop recovered after {consecutive_failures} failures")
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"[{datetime.utcnow().isoformat()}] ⚠️ loop error: {e.__class__.__name__}: {e}")

            # Rate-limit noisy warning pushes during extended upstream outages.
            if consecutive_failures in {1, 3, 10} or consecutive_failures % 25 == 0:
                send_status_update(
                    f"loop warning x{consecutive_failures}: {e.__class__.__name__} (will retry with backoff)"
                )

        if loop_num % max(status_every_loops, 1) == 0:
            use_db_perf = os.getenv("ENABLE_DB_PERF", "0") == "1"
            if use_db_perf:
                try:
                    perf = compute_performance()
                    send_status_update(
                        f"Loop {loop_num}: trades={perf.trade_count}, win_rate={perf.win_rate:.1%}, gross_pnl=${perf.gross_pnl:.2f}"
                    )
                except Exception:
                    send_status_update(f"Loop {loop_num}: running (DB perf unavailable)")
            else:
                send_status_update(f"Loop {loop_num}: running")

        backoff_s = 0
        if consecutive_failures > 0:
            # Exponential backoff capped to keep heartbeat alive while reducing API thrash.
            backoff_s = min(300, (2 ** min(consecutive_failures, 6)) - 1)

        sleep_s = interval_seconds + backoff_s
        print(f"[{datetime.utcnow().isoformat()}] 💤 sleeping {sleep_s}s")
        time.sleep(sleep_s)
