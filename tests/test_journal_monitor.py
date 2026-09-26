"""Tests for the journal's time-exit column and the monitor's time-exit / buy-up-to
alerts (temp dirs and synthetic price DBs only; never touches data/, no network)."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import pandas as pd

from atlas2 import monitor
from atlas2.journal import Journal

OLD_SCHEMA = """
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL, name TEXT DEFAULT '', market TEXT NOT NULL,
    currency TEXT NOT NULL, pattern TEXT DEFAULT '', grade TEXT DEFAULT '',
    entry_date TEXT NOT NULL, entry_price REAL NOT NULL, shares REAL NOT NULL,
    stop REAL NOT NULL, target REAL, exit_date TEXT, exit_price REAL,
    exit_reason TEXT, notes TEXT DEFAULT ''
);
INSERT INTO trades (ticker, market, currency, entry_date, entry_price, shares, stop)
VALUES ('OLD', 'us', 'USD', '2026-08-18', 10.0, 5.0, 9.0);
"""


def trade(**kw) -> dict:
    t = {"ticker": "AAA", "market": "us", "currency": "USD", "entry_price": 10.0,
         "shares": 5, "stop": 9.0, "target": 13.0, "entry_date": "2026-08-18"}
    t.update(kw)
    return t


def write_bars(path: Path, bars: dict[str, list[str]]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE bars (ticker TEXT NOT NULL, date TEXT NOT NULL, open REAL, "
                 "high REAL, low REAL, close REAL, volume REAL, PRIMARY KEY (ticker, date))")
    conn.executemany("INSERT INTO bars VALUES (?,?,1,1,1,1,1)",
                     [(t, d) for t, ds in bars.items() for d in ds])
    conn.commit()
    conn.close()
    return path


class JournalTimeExitTest(unittest.TestCase):
    def test_old_journal_gets_the_column_and_keeps_its_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "j.sqlite3"
            conn = sqlite3.connect(db)
            conn.executescript(OLD_SCHEMA)
            conn.close()
            j = Journal(db)
            try:
                rows = j.all_trades()
                self.assertEqual(len(rows), 1)
                self.assertIn("max_hold_days", rows[0])
                self.assertIsNone(rows[0]["max_hold_days"])
                j.add(trade(ticker="NEW", max_hold_days=40))
            finally:
                j.close()
            j = Journal(db)   # re-opening an upgraded file is a no-op
            try:
                got = {t["ticker"]: t["max_hold_days"] for t in j.all_trades()}
            finally:
                j.close()
        self.assertEqual(got, {"OLD": None, "NEW": 40})

    def test_add_stores_the_time_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = Journal(Path(tmp) / "j.sqlite3")
            try:
                for tk, v in (("A", 40), ("B", None), ("C", 0), ("D", "40"), ("E", "x"),
                              ("F", -3), ("G", True), ("H", float("nan"))):
                    j.add(trade(ticker=tk, max_hold_days=v))
                j.add(trade(ticker="I"))          # field missing (older dashboard)
                got = {t["ticker"]: t["max_hold_days"] for t in j.all_trades()}
            finally:
                j.close()
        self.assertEqual(got, {"A": 40, "B": None, "C": 0, "D": 40, "E": None, "F": None,
                               "G": None, "H": None, "I": None})


class SessionCountTest(unittest.TestCase):
    def test_counts_completed_sessions_only(self):
        dates = ["2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21", "2026-08-24"]
        # Mon 24th: its own (partial) bar is ignored; 19, 20, 21 are done -> 3
        self.assertEqual(monitor.sessions_since_entry(dates, "2026-08-18", date(2026, 8, 24)), 3)

    def test_stale_cache_adds_weekdays(self):
        # last cached bar Thu 20th, today Tue 25th: Fri 21 + Mon 24 are added
        n = monitor.sessions_since_entry(["2026-08-19", "2026-08-20"], "2026-08-18",
                                         date(2026, 8, 25))
        self.assertEqual(n, 4)
        self.assertEqual(monitor.sessions_since_entry([], "2026-08-18", date(2026, 8, 19)), 0)
        self.assertEqual(monitor.sessions_since_entry([], "2026-08-18", date(2026, 8, 18)), 0)

    def test_exchange_calendar_fills_single_stock_holes(self):
        days = list(pd.bdate_range("2026-08-19", "2026-08-28").strftime("%Y-%m-%d"))
        hole = days[3]
        bars = {f"P{k}": days for k in range(monitor.MIN_CALENDAR_TICKERS)}
        bars["AAA"] = [d for d in days if d != hole]
        bars["X.DE"] = days[:2]            # another exchange never counts for AAA
        with tempfile.TemporaryDirectory() as tmp:
            db = write_bars(Path(tmp) / "p.sqlite3", bars)
            conn = sqlite3.connect(db)
            try:
                got = monitor._session_dates(conn, "AAA", "2026-08-18", date(2026, 8, 31))
                de = monitor._session_dates(conn, "X.DE", "2026-08-18", date(2026, 8, 31))
            finally:
                conn.close()
        self.assertEqual(got, days)
        self.assertEqual(de, days[:2])     # too few .DE stocks for a calendar


class TimeExitChecksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        days = list(pd.bdate_range("2026-08-19", "2026-10-30").strftime("%Y-%m-%d"))
        self.days = days
        self.db = write_bars(Path(self.tmp.name) / "p.sqlite3", {"AAA": days, "BBB": days})

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, trades, today, default=40):
        return monitor.time_exit_checks(trades, today, default, self.db)

    def test_fires_on_the_last_allowed_day(self):
        # entry Tue 2026-08-18: day 40 is the 40th session after it
        day40 = date.fromisoformat(self.days[39])
        day39 = date.fromisoformat(self.days[38])
        t = trade(id=1, max_hold_days=None)
        self.assertEqual(self.check([t], day39), [])
        got = self.check([t], day40)
        self.assertEqual(len(got), 1)
        self.assertEqual((got[0]["day"], got[0]["max_hold_days"]), (40, 40))
        self.assertIn("day 40 of 40 trading days", got[0]["message"])
        self.assertIn("sell at today's close", got[0]["message"])
        later = self.check([t], date.fromisoformat(self.days[41]))
        self.assertEqual(later[0]["day"], 42)
        self.assertIn("past the 40-day time exit", later[0]["message"])

    def test_limit_per_trade_and_no_limit(self):
        today = date.fromisoformat(self.days[9])            # day 10
        trades = [trade(id=1, ticker="AAA", max_hold_days=10),
                  trade(id=2, ticker="BBB", max_hold_days=0),     # Comeback: no time exit
                  trade(id=3, ticker="AAA", max_hold_days=None)]  # falls back to 40
        got = self.check(trades, today)
        self.assertEqual([g["id"] for g in got], [1])
        self.assertEqual([g["id"] for g in self.check(trades, today, default=10)], [1, 3])

    def test_weekend_and_same_day_entries_never_fire(self):
        t = trade(id=1, max_hold_days=1)
        self.assertEqual(self.check([t], date(2026, 8, 22)), [])   # Saturday
        self.assertEqual(self.check([trade(id=2, max_hold_days=1, entry_date="2026-08-19")],
                                    date(2026, 8, 19)), [])
        self.assertEqual(len(self.check([trade(id=3, max_hold_days=1, entry_date="2026-08-19")],
                                        date(2026, 8, 20))), 1)

    def test_missing_cache_counts_weekdays(self):
        got = monitor.time_exit_checks([trade(id=1, max_hold_days=3)], date(2026, 8, 21), 40,
                                       Path(self.tmp.name) / "missing.sqlite3")
        self.assertEqual([g["day"] for g in got], [3])


class RunMonitorTest(unittest.TestCase):
    def test_entry_alert_repeats_the_buy_up_to_price(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "scans").mkdir(parents=True)
            Journal(root / "data" / "journal.sqlite3").close()
            setup = {"ticker": "AAA", "grade": "A", "status": "forming",
                     "pattern": "bull_flag", "entry": 10.0, "max_buy": 10.3}
            old = dict(setup, ticker="OLD")
            old.pop("max_buy")
            (root / "data" / "scans" / "us_latest.json").write_text(
                json.dumps({"setups": [setup, old]}))
            with mock.patch.object(monitor, "ROOT", root), \
                    mock.patch.object(monitor, "STATE_PATH", root / "data" / "state.json"), \
                    mock.patch.object(monitor, "LOG_PATH", root / "data" / "alerts.jsonl"), \
                    mock.patch.object(monitor, "PRICES_DB", root / "none.sqlite3"), \
                    mock.patch.object(monitor, "_latest_prices",
                                      return_value={"AAA": 10.1, "OLD": 10.1}), \
                    mock.patch.object(monitor, "_default_max_hold", return_value=40), \
                    mock.patch.object(monitor, "_notify"):
                alerts = monitor.run_monitor(force=True, progress=lambda *_: None)
        msgs = {a["ticker"]: a["message"] for a in alerts}
        self.assertIn("the backtest only buys up to 10.3. Skip it if it opened higher",
                      msgs["AAA"])
        self.assertTrue(msgs["OLD"].endswith("setup triggered"))


if __name__ == "__main__":
    unittest.main()
