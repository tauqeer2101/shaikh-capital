"""Atlas2 CLI: python -m atlas2 <command>

Commands:
  universe            rebuild index-constituent lists
  scan [us de in]     run the full scan (default: all markets)
  serve               start the dashboard at http://127.0.0.1:8899
  backtest [market]   backtest the pattern strategy on cached history
"""
from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "serve"
    if cmd == "universe":
        from pathlib import Path

        from .universe_sync import build_universe

        print("Universe built:", build_universe(Path(__file__).resolve().parent.parent))
    elif cmd == "scan":
        from .comeback import scan_comeback
        from .scan import run_scan

        markets = [a for a in args[1:] if a in ("us", "de", "in")] or None
        run_scan(markets)
        scan_comeback()
    elif cmd == "comeback":
        from .comeback import scan_comeback

        scan_comeback()
    elif cmd == "comeback-test":
        from .comeback import backtest_comeback

        backtest_comeback()
    elif cmd == "backtest":
        from .backtest import run_backtest

        markets = [a for a in args[1:] if a in ("us", "de", "in")] or ["us", "de", "in"]
        run_backtest(markets)
    elif cmd == "portfolio":
        from .portfolio_sim import run_portfolio_report

        run_portfolio_report()
    elif cmd == "monitor":
        from .monitor import run_monitor

        run_monitor(force="--force" in args)
    elif cmd == "serve":
        import uvicorn

        from .server import app

        # 0.0.0.0 = reachable from your other devices (via Tailscale or home WiFi),
        # not from the internet. Use "127.0.0.1" to restrict to this Mac only.
        uvicorn.run(app, host="0.0.0.0", port=8899, log_level="warning")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
