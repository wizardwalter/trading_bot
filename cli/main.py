import argparse
import os
import sys

# allow running as: python cli/main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.exe import run_bot
from services.runner import run_loop


def main():
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval seconds (default: 60)")
    parser.add_argument("--status-every", type=int, default=5, help="Status update every N loops")
    parser.add_argument("--execute", action="store_true", help="Actually place Alpaca paper orders")
    args = parser.parse_args()

    if args.loop:
        run_loop(
            interval_seconds=max(10, args.interval),
            execute_orders=args.execute,
            status_every_loops=max(1, args.status_every),
        )
    else:
        run_bot(paper_mode=True, execute_orders=args.execute)


if __name__ == "__main__":
    main()
