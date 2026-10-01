"""Tests for the protective guards (README "Protective guards"): the stop alert as
an exit instruction with days_past_stop, the scanner's sector cap, the journal's
sector column and the sector-exposure endpoint. Temp dirs and synthetic data
only; never touches data/, no network."""
from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from atlas2 import monitor, scan, server
from atlas2.journal import (UNKNOWN_SECTOR, Journal, enrich_and_summarize,
                            sector_exposure)

try:
    from starlette.testclient import TestClient
    _TESTCLIENT_ERROR = None
except Exception as exc:  # starlette raises RuntimeError without httpx, not ImportError
    TestClient, _TESTCLIENT_ERROR = None, f"{exc.__class__.__name__}: {exc}"

# the journal schema of the previous release: max_hold_days present, no sector
PREVIOUS_SCHEMA = """
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL, name TEXT DEFAULT '', market TEXT NOT NULL,
    currency TEXT NOT NULL, pattern TEXT DEFAULT '', grade TEXT DEFAULT '',
    entry_date TEXT NOT NULL, entry_price REAL NOT NULL, shares REAL NOT NULL,
    stop REAL NOT NULL, target REAL, exit_date TEXT, exit_price REAL,
    exit_reason TEXT, notes TEXT DEFAULT '', max_hold_days INTEGER
);
INSERT INTO trades (ticker, market, currency, entry_date, entry_price, shares, stop, max_hold_days)
VALUES ('OLD', 'us', 'USD', '2026-08-18', 10.0, 5.0, 9.0, 40);
INSERT INTO trades (ticker, market, currency, entry_date, entry_price, shares, stop, exit_price)
VALUES ('DONE', 'us', 'USD', '2026-07-01', 10.0, 5.0, 9.0, 12.0);
"""
# the first release: neither column
FIRST_SCHEMA = PREVIOUS_SCHEMA.replace(", max_hold_days INTEGER", "").replace(
    ", stop, max_hold_days)\nVALUES ('OLD', 'us', 'USD', '2026-08-18', 10.0, 5.0, 9.0, 40)",
    ", stop)\nVALUES ('OLD', 'us', 'USD', '2026-08-18', 10.0, 5.0, 9.0)")

SYF_MESSAGE = ("SYF: STOP HIT — sell now. Price 71.55 is below your stop 76.51 (−5.3% from entry "
               "80.75). The plan was to exit here; every day held past the stop is outside the plan.")


def trade(**kw) -> dict:
    t = {"ticker": "AAA", "market": "us", "currency": "USD", "entry_price": 10.0,
         "shares": 5, "stop": 9.0, "target": 13.0, "entry_date": "2026-08-18"}
    t.update(kw)
    return t


def setup(ticker: str, sector, score: float, grade: str = "B", market: str | None = None) -> dict:
    s = {"ticker": ticker, "sector": sector, "composite_score": score, "grade": grade,
         "name": ticker, "pattern": "bull_flag"}
    if market:
        s["market"] = market
    return s


def write_alerts(path: Path, alerts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(a) + "\n" for a in alerts))


# ------------------------------------------------------------------ G1 ------
class StopMessageTest(unittest.TestCase):
    def test_matches_the_spec_example(self):
        t = {"ticker": "SYF", "entry_price": 80.75, "stop": 76.51}
        self.assertEqual(monitor.stop_message(t, 71.55), SYF_MESSAGE)

    def test_at_the_stop_and_trailed_stop(self):
        t = {"ticker": "SYF", "entry_price": 80.75, "stop": 76.51}
        self.assertIn("Price 76.51 is at your stop 76.51", monitor.stop_message(t, 76.51))
        up = {"ticker": "AAA", "entry_price": 10.0, "stop": 10.5}   # stop trailed above entry
        self.assertIn("(+5.0% from entry 10)", monitor.stop_message(up, 10.4))
        self.assertNotIn("-", monitor.stop_message(up, 10.4).split("entry")[0].replace("—", ""))

    def test_word_and_numbers_agree_and_six_figure_prices_keep_their_decimals(self):
        t = {"ticker": "SYF", "entry_price": 80.75, "stop": 76.51}
        # float(np.float32(76.51)): float32 noise from the price store must not read
        # "Price 76.51 is below your stop 76.51"
        self.assertIn("Price 76.51 is at your stop 76.51",
                      monitor.stop_message(t, 76.51000213623047))
        self.assertIn("Price 76.5099 is below your stop 76.51", monitor.stop_message(t, 76.5099))
        mrf = {"ticker": "MRF.NS", "entry_price": 125430.5, "stop": 119158.98}
        self.assertEqual(monitor.stop_message(mrf, 118950.25),
                         "MRF.NS: STOP HIT — sell now. Price 118950.25 is below your stop "
                         "119158.98 (−5.0% from entry 125430.5). The plan was to exit here; "
                         "every day held past the stop is outside the plan.")


class DaysPastStopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "alerts.jsonl"
        self.today = date(2026, 8, 23)

    def tearDown(self):
        self.tmp.cleanup()

    def days(self, ticker="SYF", level=76.51, entry="2026-08-18"):
        return monitor.days_past_stop(ticker, level, entry, self.today, self.log)

    def test_missing_or_empty_log_means_none(self):
        self.assertIsNone(self.days())
        self.log.write_text("")
        self.assertIsNone(self.days())

    def test_counts_from_the_first_matching_alert(self):
        write_alerts(self.log, [
            # an earlier trade in the same stock, same stop: before this entry -> ignored
            {"time": "2026-08-10T10:00:00", "kind": "stop", "ticker": "SYF", "level": 76.51},
            {"time": "2026-08-20T15:00:00", "kind": "stop", "ticker": "SYF", "level": 76.51},
            {"time": "2026-08-21T10:00:00", "kind": "stop", "ticker": "SYF", "level": 76.51},
            {"time": "2026-08-19T10:00:00", "kind": "stop", "ticker": "SYF", "level": 70.0},  # other level
            {"time": "2026-08-19T10:00:00", "kind": "target", "ticker": "SYF", "level": 76.51},
            {"time": "2026-08-19T10:00:00", "kind": "stop", "ticker": "XYZ", "level": 76.51},
            {"time": "2026-08-19T10:00:00", "kind": "entry", "ticker": "SYF", "level": None},
        ])
        self.assertEqual(self.days(), 3)                      # first on the 20th
        self.assertEqual(self.days(level=70.0), 4)            # the other level counts separately
        self.assertIsNone(self.days(ticker="QQQ"))
        self.assertIsNone(self.days(entry="2026-08-22"))      # all alerts predate this entry

    def test_tolerates_garbage_lines_and_json_float_round_trip(self):
        self.log.write_text('not json\n\n{"kind": "stop", "ticker": "SYF", "level": "x"}\n'
                            + json.dumps({"time": "2026-08-22T09:00:00", "kind": "stop",
                                          "ticker": "SYF", "level": 76.51}) + "\n")
        self.assertEqual(self.days(level=float("76.51")), 1)
        self.assertEqual(monitor.days_past_stop("SYF", 76.51, "2026-08-18", date(2026, 8, 22),
                                                self.log), 0)


class RunMonitorStopAlertTest(unittest.TestCase):
    """run_monitor end to end with mocked prices / notifications in a temp root."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data" / "scans").mkdir(parents=True)
        j = Journal(self.root / "data" / "journal.sqlite3")
        self.syf_id = j.add(trade(ticker="SYF", entry_price=80.75, stop=76.51, target=95.0,
                                  entry_date="2026-08-18"))
        j.add(trade(ticker="TGT", entry_price=10.0, stop=9.0, target=13.0))
        j.add(trade(ticker="FINE", entry_price=10.0, stop=9.0, target=13.0))
        j.close()
        self.today = date.today()
        self.notified: list[tuple[str, str]] = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_monitor(self, prices, time_alerts=()):
        with mock.patch.object(monitor, "ROOT", self.root), \
                mock.patch.object(monitor, "STATE_PATH", self.root / "data" / "state.json"), \
                mock.patch.object(monitor, "LOG_PATH", self.root / "data" / "alerts.jsonl"), \
                mock.patch.object(monitor, "PRICES_DB", self.root / "none.sqlite3"), \
                mock.patch.object(monitor, "_latest_prices", return_value=prices), \
                mock.patch.object(monitor, "_default_max_hold", return_value=40), \
                mock.patch.object(monitor, "time_exit_checks", return_value=list(time_alerts)), \
                mock.patch.object(monitor, "_notify",
                                  side_effect=lambda t, m: self.notified.append((t, m))):
            return monitor.run_monitor(force=True, progress=lambda *_: None)

    def test_stop_hit_is_an_exit_instruction_and_target_time_stay_as_they_were(self):
        time_alert = {"id": 3, "ticker": "FINE", "day": 40, "max_hold_days": 40,
                      "message": "FINE: day 40 of 40 trading days since you bought (2026-08-18) "
                                 "- time exit: sell at today's close unless the stop or target is hit first"}
        alerts = self.run_monitor({"SYF": 71.55, "TGT": 13.2, "FINE": 10.5}, [time_alert])
        by_kind = {a["kind"]: a for a in alerts}
        self.assertEqual(set(by_kind), {"stop", "target", "time"})
        stop = by_kind["stop"]
        self.assertEqual(stop["message"], SYF_MESSAGE)
        self.assertEqual((stop["ticker"], stop["level"], stop["price"]), ("SYF", 76.51, 71.55))
        self.assertEqual(stop["days_past_stop"], 0)        # first alert for this position
        self.assertEqual(stop["trade_id"], self.syf_id)
        self.assertEqual(by_kind["target"]["message"],
                         "TGT: target 13 reached - consider taking profit")
        self.assertNotIn("days_past_stop", by_kind["target"])
        self.assertEqual(by_kind["time"]["message"], time_alert["message"])
        titles = dict(self.notified)
        self.assertEqual(titles["Shaikh Capital — EXIT: SYF"], SYF_MESSAGE)
        self.assertIn("Shaikh Capital — TARGET", titles)
        self.assertIn("Shaikh Capital — TIME EXIT", titles)
        # the de-dup key keeps its format, so one alert per day per level
        state = json.loads((self.root / "data" / "state.json").read_text())
        self.assertIn(f"{self.today.isoformat()}:stop:SYF:76.51", state["fired"])
        logged = [json.loads(x) for x in
                  (self.root / "data" / "alerts.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(a["kind"] for a in logged), ["stop", "target", "time"])
        self.assertEqual(self.run_monitor({"SYF": 70.0, "TGT": 13.2}), [])

    def test_days_past_stop_counts_from_the_first_logged_stop_alert(self):
        first = (self.today - timedelta(days=3)).isoformat() + "T10:05:00"
        write_alerts(self.root / "data" / "alerts.jsonl", [
            {"time": first, "kind": "stop", "ticker": "SYF", "level": 76.51, "price": 75.0,
             "message": "old wording"},
            {"time": (self.today - timedelta(days=1)).isoformat() + "T10:05:00", "kind": "stop",
             "ticker": "SYF", "level": 76.51, "price": 74.0, "message": "old wording"},
        ])
        alerts = self.run_monitor({"SYF": 71.55})
        self.assertEqual([a["kind"] for a in alerts], ["stop"])
        self.assertEqual(alerts[0]["days_past_stop"], 3)
        self.assertEqual(alerts[0]["message"], SYF_MESSAGE)

    def test_no_stop_alert_above_the_stop(self):
        alerts = self.run_monitor({"SYF": 76.52, "TGT": 12.9, "FINE": 10.0})
        self.assertEqual(alerts, [])
        self.assertEqual(self.notified, [])

    def test_sector_capped_setups_neither_alert_nor_take_a_watch_slot(self):
        """G2 on the notification channel: a setup the dashboard greys out as 'sector
        full' must not say 'setup triggered', nor push an active setup off the list."""
        def forming(ticker: str, entry: float, capped: bool) -> dict:
            return {"ticker": ticker, "pattern": "bull_flag", "grade": "A", "status": "forming",
                    "entry": entry, "sector_capped": capped}
        capped = [forming(f"CAP{i}", 100.0, True) for i in range(monitor.WATCH_TOP_N)]
        setups = capped + [forming("ACTIVE", 50.0, False)]   # the active one ranks last
        (self.root / "data" / "scans" / "us_latest.json").write_text(
            json.dumps({"setups": setups}))
        prices = {s["ticker"]: s["entry"] + 1 for s in setups}    # every entry crossed
        prices.update({"SYF": 90.0, "TGT": 12.0, "FINE": 10.0})   # open positions: quiet
        alerts = self.run_monitor(prices)
        self.assertEqual([(a["kind"], a["ticker"]) for a in alerts], [("entry", "ACTIVE")])
        self.assertEqual(self.notified, [("Shaikh Capital — ENTRY",
                                          "ACTIVE (bull flag): crossed entry 50 - setup triggered")])


class EnrichStopHitTest(unittest.TestCase):
    def test_stop_hit_flag(self):
        trades = [dict(trade(ticker="A", id=1), exit_price=None, exit_date=None),
                  dict(trade(ticker="B", id=2), exit_price=None, exit_date=None),
                  dict(trade(ticker="C", id=3), exit_price=None, exit_date=None),
                  dict(trade(ticker="D", id=4), exit_price=12.0, exit_date="2026-08-20")]
        out = enrich_and_summarize(trades, {"A": 9.0, "B": 9.01})
        flags = {t["ticker"]: t["stop_hit"] for t in out["open"]}
        self.assertEqual(flags, {"A": True, "B": False, "C": False})   # C: no price known
        self.assertNotIn("stop_hit", out["closed"][0])


# ------------------------------------------------------------------ G2 ------
class SectorCapTest(unittest.TestCase):
    def setups(self):
        return [setup("F1", "Financial Services", 90, "A"), setup("T1", "Technology", 88, "A"),
                setup("F2", "Financial Services", 85, "A"), setup("U1", None, 84, "A"),
                setup("F3", "Financial Services", 80, "A"), setup("U2", None, 79),
                setup("T2", "Technology", 75), setup("F4", "Financial Services", 70),
                setup("T3", "Technology", 65, "A"), setup("U3", "", 60)]

    def test_flags_beyond_the_cap_keep_order_and_unknown_is_never_capped(self):
        ss = self.setups()
        counts = scan.apply_sector_cap(ss, 2, "us")
        self.assertEqual([s["ticker"] for s in ss],
                         ["F1", "T1", "F2", "U1", "F3", "U2", "T2", "F4", "T3", "U3"])
        self.assertEqual([s["ticker"] for s in ss if s["sector_capped"]], ["F3", "F4", "T3"])
        self.assertEqual(ss[4]["sector_cap_note"],
                         "already 2 Financial Services setups ranked higher today — skip unless "
                         "you sell one")
        self.assertIsNone(ss[0]["sector_cap_note"])
        self.assertIs(ss[0]["sector_capped"], False)
        self.assertEqual(counts, {"Financial Services": 4, "Technology": 3, UNKNOWN_SECTOR: 3})
        self.assertEqual(sum(1 for s in ss if s["sector_capped"]), 3)

    def test_zero_or_none_switches_the_cap_off(self):
        for off in (0, None, -1, "x", False):
            ss = self.setups()
            counts = scan.apply_sector_cap(ss, off, "us")
            self.assertEqual([s["sector_capped"] for s in ss], [False] * 10, off)
            self.assertEqual(counts["Financial Services"], 4)
        self.assertEqual(scan.sector_cap(None), 0)
        self.assertEqual(scan.sector_cap(0), 0)
        self.assertEqual(scan.sector_cap(-3), 0)
        self.assertEqual(scan.sector_cap("2"), 2)
        self.assertEqual(scan.sector_cap(2.9), 2)
        self.assertEqual(scan.sector_cap(True), 0)

    def test_cap_of_one_and_per_market_buckets(self):
        ss = self.setups()
        scan.apply_sector_cap(ss, 1, "us")
        self.assertEqual([s["ticker"] for s in ss if s["sector_capped"]], ["F2", "F3", "T2", "F4", "T3"])
        mixed = [setup("A", "Tech", 90, market="us"), setup("B", "Tech", 89, market="de"),
                 setup("C", "Tech", 88, market="us"), setup("D", "Tech", 87, market="de"),
                 setup("E", "Tech", 86, market="us")]
        scan.apply_sector_cap(mixed, 2)
        self.assertEqual([s["ticker"] for s in mixed if s["sector_capped"]], ["E"])

    def test_config_default(self):
        cfg = scan.load_config(Path(tempfile.gettempdir()) / "definitely-missing-config.json")
        self.assertEqual(cfg["max_per_sector"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps({"max_per_sector": 0}))
            self.assertEqual(scan.load_config(p)["max_per_sector"], 0)
            p.write_text(json.dumps({"max_per_sector": None}))
            self.assertIsNone(scan.load_config(p)["max_per_sector"])
            self.assertEqual(scan.sector_cap(scan.load_config(p)["max_per_sector"]), 0)

    def test_comeback_module_is_untouched(self):
        src = (Path(scan.__file__).parent / "comeback.py").read_text()
        self.assertNotIn("sector_cap", src)


class ScansOverviewTest(unittest.TestCase):
    def write_scan(self, root: Path, market: str, setups: list[dict]) -> None:
        (root / "data" / "scans").mkdir(parents=True, exist_ok=True)
        (root / "data" / "scans" / f"{market}_latest.json").write_text(json.dumps({
            "generated": "2026-10-01T07:30:00", "setups": setups, "warnings": [{"ticker": "W"}],
            "regime": {"status": "healthy"}}))

    def test_counts_only_active_setups_and_reports_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ss = [setup("F1", "Fin", 90, "A"), setup("F2", "Fin", 85, "A"),
                  setup("F3", "Fin", 80, "A"), setup("T1", "Tech", 70, "B"),
                  setup("F4", "Fin", 60, "B")]
            scan.apply_sector_cap(ss, 2, "us")
            self.write_scan(root, "us", ss)
            self.write_scan(root, "de", [setup("D1", "Fin", 90, "A")])   # old file: no flags
            self.write_scan(root, "in", [])
            with mock.patch.object(server, "ROOT", root):
                out = server.scans_available()
        self.assertEqual((out["us"]["setups"], out["us"]["a_grades"], out["us"]["capped"]), (3, 2, 2))
        self.assertEqual(out["us"]["top"]["ticker"], "F1")
        self.assertEqual((out["de"]["setups"], out["de"]["a_grades"], out["de"]["capped"]), (1, 1, 0))
        self.assertEqual((out["in"]["setups"], out["in"]["capped"], out["in"]["top"]), (0, 0, None))
        self.assertEqual(out["us"]["warnings"], 1)      # old fields kept


# ------------------------------------------------------------------ G3 ------
class JournalSectorTest(unittest.TestCase):
    def test_previous_release_db_gets_the_column_and_keeps_rows(self):
        for schema in (PREVIOUS_SCHEMA, FIRST_SCHEMA):
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "j.sqlite3"
                conn = sqlite3.connect(db)
                conn.executescript(schema)
                conn.close()
                j = Journal(db)
                try:
                    rows = {t["ticker"]: t for t in j.all_trades()}
                    self.assertEqual(set(rows), {"OLD", "DONE"})
                    self.assertIsNone(rows["OLD"]["sector"])
                    self.assertIn("max_hold_days", rows["OLD"])
                    self.assertEqual(rows["DONE"]["exit_price"], 12.0)
                    j.add(trade(ticker="NEW", sector="Financial Services"))
                finally:
                    j.close()
                j = Journal(db)    # re-opening an upgraded file is a no-op
                try:
                    got = {t["ticker"]: t["sector"] for t in j.all_trades()}
                finally:
                    j.close()
            self.assertEqual(got, {"OLD": None, "DONE": None, "NEW": "Financial Services"})

    def test_fresh_db_stores_and_sanitises_sector(self):
        with tempfile.TemporaryDirectory() as tmp:
            j = Journal(Path(tmp) / "j.sqlite3")
            try:
                for tk, v in (("A", "Technology"), ("B", None), ("C", ""), ("D", "  Energy "),
                              ("E", 12), ("F", "x" * 100)):
                    j.add(trade(ticker=tk, sector=v))
                j.add(trade(ticker="G"))        # field missing (older dashboard)
                got = {t["ticker"]: t["sector"] for t in j.all_trades()}
            finally:
                j.close()
        self.assertEqual(got, {"A": "Technology", "B": None, "C": None, "D": "Energy", "E": None,
                               "F": "x" * 80, "G": None})


class SectorExposureTest(unittest.TestCase):
    def test_groups_open_positions_and_marks_full_sectors(self):
        opens = [trade(ticker="SYF", sector="Financial Services"),
                 trade(ticker="STT", sector="Financial Services"),
                 trade(ticker="BK", sector="Financial Services"),
                 trade(ticker="NVDA", sector="Technology"),
                 trade(ticker="ZZZ", sector=None), trade(ticker="YYY"), trade(ticker="XXX", sector="")]
        ex = sector_exposure(opens, 2)
        self.assertEqual(list(ex["sectors"]),
                         ["Financial Services", UNKNOWN_SECTOR, "Technology"])
        self.assertEqual(ex["sectors"]["Financial Services"],
                         {"count": 3, "tickers": ["SYF", "STT", "BK"]})
        self.assertEqual(ex["sectors"][UNKNOWN_SECTOR]["count"], 3)
        self.assertEqual((ex["max_per_sector"], ex["over_cap"]), (2, ["Financial Services"]))
        off = sector_exposure(opens, 0)
        self.assertEqual((off["max_per_sector"], off["over_cap"]), (0, []))
        self.assertEqual(sector_exposure([], None), {"sectors": {}, "max_per_sector": 0, "over_cap": []})


def write_fundamentals(path: Path, sectors: dict[str, str | None]) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE fundamentals (ticker TEXT PRIMARY KEY, fetched TEXT NOT NULL, "
                 "payload TEXT NOT NULL)")
    conn.executemany("INSERT INTO fundamentals VALUES (?, '2026-09-01T00:00:00', ?)",
                     [(t, json.dumps({"sector": s, "shortName": t})) for t, s in sectors.items()])
    conn.commit()
    conn.close()
    return path


class ServerJournalTest(unittest.TestCase):
    """The journal routes called as plain functions, against a temp journal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(server, "JOURNAL_DB", self.root / "journal.sqlite3"),
            mock.patch.object(server, "FUNDAMENTALS_DB", self.root / "missing-fundamentals.sqlite3"),
            mock.patch.object(server, "_max_per_sector", return_value=2),
            mock.patch.object(monitor, "LOG_PATH", self.root / "alerts.jsonl"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_add_stores_sector_and_exposure_groups_open_positions(self):
        for tk, sec in (("SYF", "Financial Services"), ("STT", "Financial Services"),
                        ("NVDA", "Technology"), ("QQQ", None)):
            body = trade(ticker=tk)
            if sec is not None:
                body["sector"] = sec
            self.assertIn("id", server.journal_add(body))
        closed_id = server.journal_add(dict(trade(ticker="BK"), sector="Financial Services"))["id"]
        server.journal_close(closed_id, {"exit_price": 9.0, "exit_reason": "stop"})
        j = Journal(self.root / "journal.sqlite3")
        try:
            stored = {t["ticker"]: t["sector"] for t in j.all_trades()}
        finally:
            j.close()
        self.assertEqual(stored, {"SYF": "Financial Services", "STT": "Financial Services",
                                  "NVDA": "Technology", "QQQ": None, "BK": "Financial Services"})
        ex = server.journal_exposure()
        self.assertEqual(ex["max_per_sector"], 2)
        fin = ex["sectors"]["Financial Services"]
        # tickers follow the journal's order (newest first); BK is closed
        self.assertEqual((fin["count"], sorted(fin["tickers"])), (2, ["STT", "SYF"]))
        self.assertEqual(ex["sectors"][UNKNOWN_SECTOR]["tickers"], ["QQQ"])
        self.assertEqual(ex["over_cap"], ["Financial Services"])

    def test_null_sector_is_looked_up_read_only_in_the_fundamentals_cache(self):
        server.journal_add(trade(ticker="QQQ"))
        server.journal_add(dict(trade(ticker="SYF"), sector="Stored Sector"))
        server.journal_add(trade(ticker="NONE"))
        self.assertEqual(server.journal_exposure()["sectors"][UNKNOWN_SECTOR]["count"], 2)
        fdb = write_fundamentals(self.root / "fundamentals.sqlite3",
                                 {"QQQ": "Technology", "SYF": "Cache Sector", "NONE": None})
        with mock.patch.object(server, "FUNDAMENTALS_DB", fdb):
            ex = server.journal_exposure()
            self.assertEqual(ex["sectors"]["Technology"]["tickers"], ["QQQ"])
            self.assertEqual(ex["sectors"]["Stored Sector"]["tickers"], ["SYF"])   # stored wins
            self.assertEqual(ex["sectors"][UNKNOWN_SECTOR]["tickers"], ["NONE"])
            j = Journal(self.root / "journal.sqlite3")
            try:
                self.assertIsNone([t for t in j.all_trades() if t["ticker"] == "QQQ"][0]["sector"])
            finally:
                j.close()      # the row itself is not backfilled
        self.assertEqual(server._cached_sectors(["QQQ"], self.root / "nope.sqlite3"), {})
        broken = self.root / "broken.sqlite3"
        broken.write_text("not a database")
        self.assertEqual(server._cached_sectors(["QQQ"], broken), {})

    def test_journal_list_flags_stop_hits_with_days_and_exposure(self):
        server.journal_add(dict(trade(ticker="SYF", entry_price=80.75, stop=76.51,
                                      entry_date="2026-08-18"), sector="Financial Services"))
        server.journal_add(dict(trade(ticker="STT"), sector="Financial Services"))
        server.journal_add(dict(trade(ticker="NVDA"), sector="Technology"))
        today = date.today()
        write_alerts(self.root / "alerts.jsonl", [
            {"time": (today - timedelta(days=3)).isoformat() + "T10:00:00", "kind": "stop",
             "ticker": "SYF", "level": 76.51, "price": 75.0, "message": "x"}])
        with mock.patch.object(server, "_last_closes",
                               return_value={"SYF": 71.55, "STT": 10.5, "NVDA": 11.0}):
            out = server.journal_list()
        by = {t["ticker"]: t for t in out["open"]}
        self.assertTrue(by["SYF"]["stop_hit"])
        self.assertEqual(by["SYF"]["days_past_stop"], 3)
        self.assertFalse(by["STT"]["stop_hit"])
        self.assertIsNone(by["STT"]["days_past_stop"])
        self.assertEqual(by["SYF"]["sector"], "Financial Services")
        self.assertEqual(out["exposure"]["over_cap"], ["Financial Services"])
        self.assertEqual(out["stats"]["open_count"], 3)          # old fields kept
        # a stop hit the monitor has not alerted on yet: flagged, days unknown
        with mock.patch.object(server, "_last_closes", return_value={"STT": 8.0}):
            by = {t["ticker"]: t for t in server.journal_list()["open"]}
        self.assertTrue(by["STT"]["stop_hit"])
        self.assertIsNone(by["STT"]["days_past_stop"])


@unittest.skipIf(TestClient is None, f"starlette TestClient not importable ({_TESTCLIENT_ERROR})")
class ExposureEndpointTest(unittest.TestCase):
    def test_get_api_journal_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(server, "JOURNAL_DB", root / "journal.sqlite3"), \
                    mock.patch.object(server, "FUNDAMENTALS_DB", root / "none.sqlite3"), \
                    mock.patch.object(server, "_max_per_sector", return_value=2), \
                    mock.patch.object(server, "_dashboard_password", return_value=None):
                client = TestClient(server.app)
                r = client.post("/api/journal", json=dict(trade(ticker="SYF"), sector="Financial Services"))
                self.assertEqual(r.status_code, 200, r.text)
                r = client.post("/api/journal", json=dict(trade(ticker="STT"), sector="Financial Services"))
                self.assertEqual(r.status_code, 200, r.text)
                r = client.get("/api/journal/exposure")
                self.assertEqual(r.status_code, 200, r.text)
                ex = r.json()
        self.assertEqual(ex["max_per_sector"], 2)
        self.assertEqual(list(ex["sectors"]), ["Financial Services"])
        fin = ex["sectors"]["Financial Services"]
        self.assertEqual((fin["count"], sorted(fin["tickers"])), (2, ["STT", "SYF"]))
        self.assertEqual(ex["over_cap"], ["Financial Services"])


if __name__ == "__main__":
    unittest.main()
