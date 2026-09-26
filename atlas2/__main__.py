"""Atlas2 CLI: python -m atlas2 <command>

Commands:
  universe            rebuild index-constituent lists
  scan [us de in]     run the full scan (default: all markets)
  serve               start the dashboard at http://127.0.0.1:8899
  backtest [us de in] [--raw] [--limit N] [--workers N] [--no-portfolio]
           [--legacy-grid] [--entry-mode buy_stop|close_confirm]
           [--keep-others | --replace]
                      replay the live scanner on cached history (after fees,
                      vs holding the market); --limit N = quick smoke run on the
                      first N tickers per market (writes data/backtest_smoke.json).
                      Naming some markets re-runs only those and keeps the other
                      markets' previous results in backtest_latest.json /
                      backtest_trades.json (--keep-others keeps them even if they
                      were computed with other settings, --replace drops them;
                      results in the old v1 format cannot be kept: the run stops
                      unless --replace is given).
                      --legacy-grid (every 5 days, 15-day buy-stop) and
                      --entry-mode write labelled files, never the headline.
  portfolio           replay the backtest trades as one account
"""
from __future__ import annotations

import sys


def _parse_backtest_opts(opts: list[str]) -> dict:
    """Strict parsing of the backtest options: an unknown flag, a missing or
    non-positive number never starts a (full) run. Raises ValueError."""
    markets: list[str] = []
    out = {"raw": False, "portfolio": True, "legacy_grid": False, "entry_mode": None,
           "keep_others": False, "replace_others": False}
    ints = {"--limit": None, "--workers": None}
    k = 0
    while k < len(opts):
        a = opts[k]
        if a in ("us", "de", "in"):
            if a not in markets:
                markets.append(a)
        elif a == "--raw":
            out["raw"] = True
        elif a == "--no-portfolio":
            out["portfolio"] = False
        elif a == "--legacy-grid":
            out["legacy_grid"] = True
        elif a == "--keep-others":
            out["keep_others"] = True
        elif a == "--replace":
            out["replace_others"] = True
        else:
            name, eq, val = a.partition("=")
            if name == "--entry-mode":
                if not eq:
                    if k + 1 >= len(opts):
                        raise ValueError("--entry-mode needs buy_stop or close_confirm")
                    k += 1
                    val = opts[k]
                if val not in ("buy_stop", "close_confirm"):
                    raise ValueError(f"--entry-mode must be buy_stop or close_confirm, got {val!r}")
                out["entry_mode"] = val
            elif name in ints:
                if not eq:
                    if k + 1 >= len(opts):
                        raise ValueError(f"{name} needs a number")
                    k += 1
                    val = opts[k]
                try:
                    n = int(val)
                except ValueError:
                    raise ValueError(f"{name} needs a whole number, got {val!r}") from None
                if n <= 0:
                    raise ValueError(f"{name} must be >= 1, got {n}")
                ints[name] = n
            else:
                raise ValueError(f"unknown option: {a}")
        k += 1
    if out["keep_others"] and out["replace_others"]:
        raise ValueError("use either --keep-others or --replace, not both")
    out["markets"] = markets or ["us", "de", "in"]
    out["limit"] = ints["--limit"]
    out["workers"] = ints["--workers"]
    return out


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
        try:
            o = _parse_backtest_opts(args[1:])
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            print(__doc__)
            sys.exit(1)
        from .backtest import run_backtest

        try:
            run_backtest(o["markets"], raw=o["raw"], limit=o["limit"], workers=o["workers"],
                         portfolio=o["portfolio"], legacy_grid=o["legacy_grid"],
                         entry_mode=o["entry_mode"], keep_others=o["keep_others"],
                         replace_others=o["replace_others"])
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
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
