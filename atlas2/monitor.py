"""Price-alert monitor: python -m atlas2 monitor

Checks (a) open journal positions against their stop and target - a hit stop
is an EXIT instruction, not a hint to review (the backtest's stops are hit about
half the time, so this has to be unmistakable), (b) the
time exit of open positions (a reminder on the morning of the last allowed
trading day - the backtest sells at that day's close if neither stop nor
target was hit - and every day after it while the position is still open), and
(c) A-grade 'forming' setups from the latest scans against their entry level
(the alert repeats the scan's buy-up-to price).
Fires a macOS notification and appends to data/alerts.jsonl. Each alert fires
at most once per day per condition. Intended to run hourly via launchd; exits
quietly outside 09:00-22:30 local or on weekends.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yfinance as yf

from .journal import Journal

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / "data" / "alerts_state.json"
PRICES_DB = ROOT / "data" / "cache" / "prices.sqlite3"
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


def _default_max_hold() -> int:
    """The pattern scanner's tested time exit (config backtest.max_hold)."""
    try:
        from .scan import load_config
        return int(load_config()["backtest"]["max_hold"])
    except Exception:
        from .scan import BACKTEST_DEFAULTS
        return int(BACKTEST_DEFAULTS["max_hold"])


def sessions_since_entry(bar_dates: list[str], entry_date: str, today: date) -> int:
    """Completed trading sessions after the entry day and before today.

    bar_dates: the stock's cached bar dates ('YYYY-MM-DD'); only those strictly
    after entry_date and strictly before today count (a partial intraday bar for
    today - e.g. India during the 07:30 refresh - is ignored). Weekdays after
    the last cached bar are added, so a stale cache does not undercount."""
    t = today.isoformat()
    done = [d for d in bar_dates if entry_date < d < t]
    last = max(done) if done else entry_date
    try:
        start = date.fromisoformat(last[:10]) + timedelta(days=1)
    except ValueError:
        return len(done)
    gap = int(np.busday_count(start, today)) if start < today else 0
    return len(done) + max(0, gap)


MIN_CALENDAR_TICKERS = 5   # a date counts as a session when this many same-exchange stocks traded


def _session_dates(conn: sqlite3.Connection, ticker: str, entry: str, today: date) -> list[str]:
    """Cached trading days after `entry` and before today for this stock: its own
    bars plus every day on which at least MIN_CALENDAR_TICKERS stocks of the same
    exchange (same ticker suffix) have a bar - the free data has single-day holes
    for some stocks, which would otherwise delay the reminder."""
    lo, hi = entry, today.isoformat()
    try:
        own = {str(r[0])[:10] for r in conn.execute(
            "SELECT date FROM bars WHERE ticker = ? AND date > ? AND date < ?", (ticker, lo, hi))}
        if "." in ticker:
            clause, args = "ticker LIKE ?", ["%." + ticker.rsplit(".", 1)[1]]
        else:
            clause, args = "ticker NOT LIKE '%.%' AND ticker NOT LIKE '^%'", []
        cal = {str(r[0])[:10] for r in conn.execute(
            f"SELECT date FROM bars WHERE date > ? AND date < ? AND {clause} "
            "GROUP BY date HAVING COUNT(*) >= ?", (lo, hi, *args, MIN_CALENDAR_TICKERS))}
    except sqlite3.Error:
        return []
    return sorted(own | cal)


def time_exit_checks(open_trades: list[dict], today: date, default_max_hold: int,
                     prices_db: Path | None = None) -> list[dict]:
    """Time-exit reminders for open journal trades, from the local price cache only.

    Each trade's limit is its max_hold_days (the strategy's rule when it was
    logged); NULL (older rows / older scan files) falls back to default_max_hold,
    0 means the strategy has no time exit. Today is session count+1 after the
    entry; on the last allowed day (and on any later day, if the position is
    still open) one reminder per trade. Weekends never fire."""
    if today.weekday() >= 5:
        return []
    conn = None
    p = Path(prices_db) if prices_db else PRICES_DB
    if p.exists():
        try:
            conn = sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error:
            conn = None
    out: list[dict] = []
    try:
        for t in open_trades:
            n = t.get("max_hold_days")
            n = default_max_hold if n is None else int(n)
            entry = str(t.get("entry_date") or "")[:10]
            if n <= 0 or not entry or entry >= today.isoformat():
                continue
            dates = _session_dates(conn, str(t["ticker"]), entry, today) if conn is not None else []
            day = sessions_since_entry(dates, entry, today) + 1
            if day < n:
                continue
            if day == n:
                msg = (f"{t['ticker']}: day {n} of {n} trading days since you bought ({entry}) "
                       "- time exit: sell at today's close unless the stop or target is hit first")
            else:
                msg = (f"{t['ticker']}: day {day} since you bought ({entry}) - past the {n}-day "
                       f"time exit (the backtest sold at the close of day {n}); review the position")
            out.append({"id": t.get("id"), "ticker": t["ticker"], "day": day,
                        "max_hold_days": n, "message": msg})
    finally:
        if conn is not None:
            conn.close()
    return out


def _minus(x: float, nd: int) -> str:
    """Signed percentage with a real minus sign, e.g. '−5.3%' / '+2.0%'."""
    return f"{round(float(x), nd) + 0.0:+.{nd}f}%".replace("-", "−")


def _num(x: float) -> str:
    """A price with up to four decimals and no trailing zeros. `:g` would round a
    six-figure price (MRF.NS trades above 100000) to whole units."""
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def stop_message(trade: dict, price: float) -> str:
    """The instruction for a position whose price is at or below its stop. The
    percentage is the stop's distance from the entry - the loss the plan accepted."""
    entry, stop = float(trade["entry_price"]), float(trade["stop"])
    price = round(float(price), 4)   # what the alert record carries; drops float32 noise
    shown_price, shown_stop = _num(price), _num(stop)
    where = "at" if shown_price == shown_stop else "below"   # the word matches the numbers
    planned = (f" ({_minus((stop / entry - 1) * 100, 1)} from entry {_num(entry)})"
               if entry > 0 else "")
    return (f"{trade['ticker']}: STOP HIT — sell now. Price {shown_price} is {where} your stop "
            f"{shown_stop}{planned}. The plan was to exit here; every day held past the stop is "
            "outside the plan.")


def days_past_stop(ticker: str, level: float, entry_date: str, today: date,
                   log_path: Path | None = None) -> int | None:
    """Calendar days since the FIRST stop alert for this position - the alerts in
    the log with the same ticker and stop level, dated on or after the entry.
    None when the log has no such alert yet (the position is new to the stop)."""
    path = Path(log_path) if log_path else LOG_PATH
    if not path.exists():
        return None
    entry = str(entry_date or "")[:10]
    first: str | None = None
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                a = json.loads(line)
            except ValueError:
                continue
            if a.get("kind") != "stop" or a.get("ticker") != ticker:
                continue
            lvl = a.get("level")
            if not isinstance(lvl, (int, float)) or abs(float(lvl) - float(level)) > 1e-6:
                continue
            day = str(a.get("time") or "")[:10]
            if len(day) != 10 or day < entry:
                continue
            if first is None or day < first:
                first = day
    except OSError:
        return None
    if first is None:
        return None
    try:
        return max(0, (today - date.fromisoformat(first)).days)
    except ValueError:
        return None


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

    checks = []  # (kind, ticker, level, message, trade) - the stop message is built
    #              once the price is known (trade is None for target / entry)
    journal = Journal(ROOT / "data" / "journal.sqlite3")
    open_trades = [t for t in journal.all_trades() if t["exit_price"] is None]
    journal.close()
    for t in open_trades:
        checks.append(("stop", t["ticker"], t["stop"], None, t))
        if t.get("target"):
            checks.append(("target", t["ticker"], t["target"],
                           f"{t['ticker']}: target {t['target']:g} reached - consider taking profit",
                           None))

    for market in ("us", "de", "in", "comeback"):
        path = ROOT / "data" / "scans" / f"{market}_latest.json"
        if not path.exists():
            continue
        scan = json.loads(path.read_text())
        # sector-capped setups (G2) neither alert nor take a watch slot; the
        # Comeback scan has no such key, so .get leaves it unchanged
        watch = [s for s in scan["setups"] if s["grade"] in WATCH_GRADES
                 and s["status"] == "forming" and not s.get("sector_capped")][:WATCH_TOP_N]
        for s in watch:
            pattern_label = s["pattern"].replace("_", " ")
            cap = s.get("max_buy")
            cap_note = (f"; the backtest only buys up to {float(cap):g}. Skip it if it "
                        "opened higher") if cap is not None else ""
            checks.append(("entry", s["ticker"], s["entry"],
                           f"{s['ticker']} ({pattern_label}): "
                           f"crossed entry {s['entry']:g} - setup triggered{cap_note}", None))

    tickers = sorted({c[1] for c in checks})
    progress(f"checking {len(tickers)} tickers ({len(open_trades)} open positions, "
             f"{sum(1 for c in checks if c[0]=='entry')} watched entries)")
    prices = _latest_prices(tickers)

    try:
        time_alerts = time_exit_checks(open_trades, date.today(), _default_max_hold())
    except Exception as exc:  # a reminder must never stop the price alerts
        progress(f"time-exit check failed: {exc.__class__.__name__}: {exc}")
        time_alerts = []
    for ta in time_alerts:
        key = f"{date.today().isoformat()}:time:{ta['ticker']}:{ta['id']}"
        if key in fired:
            continue
        fired.add(key)
        price = prices.get(ta["ticker"])
        alert = {"time": now.isoformat(timespec="seconds"), "kind": "time",
                 "ticker": ta["ticker"], "level": None, "day": ta["day"],
                 "max_hold_days": ta["max_hold_days"],
                 "price": round(price, 4) if price is not None else None,
                 "message": ta["message"]}
        alerts.append(alert)
        _notify("Shaikh Capital — TIME EXIT", ta["message"])
        with LOG_PATH.open("a") as fh:
            fh.write(json.dumps(alert) + "\n")
        progress(f"ALERT time: {ta['message']}")

    for kind, ticker, level, message, trade in checks:
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
                 "ticker": ticker, "level": level, "price": round(price, 4)}
        title = f"Shaikh Capital — {kind.upper()}"
        if kind == "stop":
            # an exit instruction; days_past_stop counts from the first stop alert
            # for this position (0 today), so the dashboard can say "hit 3 days ago"
            message = stop_message(trade, price)
            title = f"Shaikh Capital — EXIT: {ticker}"
            alert["days_past_stop"] = days_past_stop(
                ticker, level, trade.get("entry_date"), date.today()) or 0
            alert["trade_id"] = trade.get("id")
        alert["message"] = message
        alerts.append(alert)
        _notify(title, message)
        with LOG_PATH.open("a") as fh:
            fh.write(json.dumps(alert) + "\n")
        progress(f"ALERT {kind}: {message} (now {price:g})")

    state["fired"] = sorted(fired)
    STATE_PATH.write_text(json.dumps(state, indent=1))
    if not alerts:
        progress("no alerts")
    return alerts
