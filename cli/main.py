import argparse

from core.exe import run_bot


def main():
    parser = argparse.ArgumentParser(description="Paper trading runner")
    parser.add_argument("--live", action="store_true", help="Enable live mode (paper mode disabled)")
    args = parser.parse_args()

    run_bot(paper_mode=not args.live)


if __name__ == "__main__":
    main()
