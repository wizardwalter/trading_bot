from __future__ import annotations

import time
from datetime import datetime

from core.exe import run_bot
from core.performance import compute_performance
from discord.notify import send_status_update


def run_loop(interval_seconds: int = 60, execute_orders: bool = False, status_every_loops: int = 5):
    loop_num = 0
    while True:
        loop_num += 1
        started = datetime.utcnow().isoformat()
        print(f"[{started}] 🔁 Loop {loop_num} start")

        run_bot(paper_mode=True, execute_orders=execute_orders)

        if loop_num % max(status_every_loops, 1) == 0:
            try:
                perf = compute_performance()
                send_status_update(
                    f"Loop {loop_num}: trades={perf.trade_count}, win_rate={perf.win_rate:.1%}, gross_pnl=${perf.gross_pnl:.2f}"
                )
            except Exception as e:
                send_status_update(f"Loop {loop_num}: running (performance DB unavailable: {e.__class__.__name__})")

        print(f"[{datetime.utcnow().isoformat()}] 💤 sleeping {interval_seconds}s")
        time.sleep(interval_seconds)
