"""Price-alert monitor: python -m atlas2 monitor

Checks (a) open journal positions against their stop and target, and
(b) A-grade 'forming' setups from the latest scans against their entry level.
Fires a macOS notification and appends to data/alerts.jsonl. Each alert fires
at most once per day per condition. Intended to run hourly via launchd; exits
quietly outside 09:00-22:30 local or on weekends.
"""
from __future__ import annotations

import json
import subprocess
from datetime import date, datetime
from pathlib import Path

import yfinance as yf

from .journal import Journal

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "alerts_state.json"
LOG_PATH = ROOT / "data" / "alerts.jsonl"
WATCH_GRADES = ("A",)
WATCH_TOP_N = 10


def _within_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    return 9 <= now.hour <= 22


def _latest_prices(tickers: list[str]) -> dict[str, float]:
    """Most recent intraday price per ticker (15m bars, ~delayed); daily fallback."""
    out: dict[str, float] = {}
    if not tickers:
        return out
    try:
        data = yf.download(tickers, period="2d", interval="15m", progress=False,
                           auto_adjust=True, group_by="ticker", threads=True)
    except Exception:
        data = None
    for t in tickers:
        price = None
        if data is not None and not data.empty:
            try:
                series = (data[t]["Close"] if len(tickers) > 1 else data["Close"]).dropna()
                if len(series):
                    price = float(series.iloc[-1])
            except Exception:
                price = None
        out[t] = price
    missing = [t for t, p in out.items() if p is None]
    if missing:
        try:
            daily = yf.download(missing, period="5d", interval="1d", progress=False,
                                auto_adjust=True, group_by="ticker", threads=True)
            for t in missing:
                try:
                    series = (daily[t]["Close"] if len(missing) > 1 else daily["Close"]).dropna()
                    if len(series):
                        out[t] = float(series.iloc[-1])
                except Exception:
                    pass
        except Exception:
            pass
    return {t: p for t, p in out.items() if p is not None}


def _notify(title: str, message: str) -> None:
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            if state.get("day") == date.today().isoformat():
                return state
        except Exception:
            pass
    return {"day": date.today().isoformat(), "fired": []}


def run_monitor(force: bool = False, progress=print) -> list[dict]:
    now = datetime.now()
    if not force and not _within_market_hours(now):
        progress("outside market hours - nothing to do")
        return []
    state = _load_state()
    fired: set[str] = set(state["fired"])
    alerts: list[dict] = []

    checks = []  # (key, ticker, kind, level, message-template data)
    journal = Journal(ROOT / "data" / "journal.sqlite3")
    open_trades = [t for t in journal.all_trades() if t["exit_price"] is None]
    journal.close()
    for t in open_trades:
        checks.append(("stop", t["ticker"], t["stop"],
                       f"{t['ticker']}: price at/below your stop {t['stop']:g} - review position"))
        if t.get("target"):
            checks.append(("target", t["ticker"], t["target"],
                           f"{t['ticker']}: target {t['target']:g} reached - consider taking profit"))

    for market in ("us", "de", "in", "comeback"):
        path = ROOT / "data" / "scans" / f"{market}_latest.json"
        if not path.exists():
            continue
        scan = json.loads(path.read_text())
        watch = [s for s in scan["setups"] if s["grade"] in WATCH_GRADES
                 and s["status"] == "forming"][:WATCH_TOP_N]
        for s in watch:
            pattern_label = s["pattern"].replace("_", " ")
            checks.append(("entry", s["ticker"], s["entry"],
                           f"{s['ticker']} ({pattern_label}): "
                           f"crossed entry {s['entry']:g} - setup triggered"))

    tickers = sorted({c[1] for c in checks})
    progress(f"checking {len(tickers)} tickers ({len(open_trades)} open positions, "
             f"{sum(1 for c in checks if c[0]=='entry')} watched entries)")
    prices = _latest_prices(tickers)

    for kind, ticker, level, message in checks:
        price = prices.get(ticker)
        if price is None:
            continue
        hit = price <= level if kind == "stop" else price >= level
        if not hit:
            continue
        key = f"{date.today().isoformat()}:{kind}:{ticker}:{level:g}"
        if key in fired:
            continue
        fired.add(key)
        alert = {"time": now.isoformat(timespec="seconds"), "kind": kind,
                 "ticker": ticker, "level": level, "price": round(price, 4),
                 "message": message}
        alerts.append(alert)
        _notify(f"Shaikh Capital — {kind.upper()}", message)
        with LOG_PATH.open("a") as fh:
            fh.write(json.dumps(alert) + "\n")
        progress(f"ALERT {kind}: {message} (now {price:g})")

    state["fired"] = sorted(fired)
    STATE_PATH.write_text(json.dumps(state, indent=1))
    if not alerts:
        progress("no alerts")
    return alerts
