"""Tests for the robustness layer: atlas2.ops, the price retry, the unfinished-bar
and stale-price guards, the monitor's unchecked-stop alert and the login limit.

Temp dirs only (ops.DATA is redirected); no network, no notifications, no launchd.
"""
from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

import pandas as pd

from atlas2 import data as D
from atlas2 import ops
from atlas2 import scan as S


class OpsCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "data"
        self.data.mkdir()
        self.patches = [mock.patch.object(ops, "DATA", self.data),
                        mock.patch.object(ops, "notify")]
        for p in self.patches:
            p.start()
        self.notify = ops.notify

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()


class LockTest(OpsCase):
    def test_second_holder_is_refused_and_released_after(self):
        with ops.run_lock("scan") as a:
            self.assertTrue(a)
            with ops.run_lock("scan") as b:
                self.assertFalse(b)
            self.assertTrue(ops.lock_is_held("scan"))
        self.assertFalse(ops.lock_is_held("scan"))
        with ops.run_lock("scan") as c:
            self.assertTrue(c)


class HealthTest(OpsCase):
    def test_job_records_ok_and_failure(self):
        with ops.job("scan:us"):
            pass
        j = ops.read_health()["jobs"]["scan:us"]
        self.assertEqual(j["last_status"], "ok")
        self.assertEqual(j["failures_in_a_row"], 0)
        for _ in range(2):
            with self.assertRaises(ValueError):
                with ops.job("scan:us"):
                    raise ValueError("boom")
        j = ops.read_health()["jobs"]["scan:us"]
        self.assertEqual((j["last_status"], j["failures_in_a_row"]), ("failed", 2))
        self.assertIn("ValueError: boom", j["last_error"])
        self.assertEqual(self.notify.call_count, 1)      # the 1st failure, not the 2nd
        self.assertTrue(ops.succeeded_on("scan:us", date.today()))

    def test_unreadable_health_file_starts_over(self):
        (self.data / "health.json").write_text("{broken")
        self.assertEqual(ops.read_health()["jobs"], {})
        ops.record("monitor", "ok")
        self.assertEqual(ops.read_health()["jobs"]["monitor"]["last_status"], "ok")


class DailyScanTest(OpsCase):
    def runners(self, calls, fail=()):
        def mk(name):
            def f(*a):
                calls.append(name if not a else f"{name}:{a[0]}")
                if (name if not a else f"{name}:{a[0]}") in fail:
                    raise RuntimeError(f"{name} broke")
            return f
        return {"market": mk("scan"), "comeback": mk("comeback"),
                "leverage_signal": mk("leverage_signal"), "backup": mk("backup"),
                "markets": lambda: ["us", "de", "in"]}

    def test_one_failure_never_stops_the_rest(self):
        calls: list[str] = []
        failed = ops.run_daily_scan(runners=self.runners(calls, fail={"scan:de"}),
                                    progress=lambda *a: None)
        self.assertEqual(failed, ["scan:de"])
        self.assertEqual(calls, ["backup", "scan:us", "scan:de", "scan:in", "comeback",
                                 "leverage_signal"])
        h = ops.read_health()["jobs"]
        self.assertEqual(h["scan"]["last_status"], "failed")
        self.assertEqual(h["scan:in"]["last_status"], "ok")
        self.assertFalse(ops.succeeded_on("scan", date.today(), ops.read_health()))
        self.notify.assert_called_once()

    def test_all_ok_and_markets_argument(self):
        calls: list[str] = []
        self.assertEqual(ops.run_daily_scan(["us"], runners=self.runners(calls),
                                            progress=lambda *a: None), [])
        self.assertEqual(calls, ["backup", "scan:us", "comeback", "leverage_signal"])
        self.assertTrue(ops.succeeded_on("scan", date.today()))

    def test_broken_config_still_runs_comeback_and_signal(self):
        calls: list[str] = []
        r = self.runners(calls)
        r["markets"] = lambda: (_ for _ in ()).throw(ValueError("bad json"))
        self.assertEqual(ops.run_daily_scan(runners=r, progress=lambda *a: None), ["config"])
        self.assertEqual(calls, ["backup", "comeback", "leverage_signal"])

    def test_refuses_to_run_twice(self):
        calls: list[str] = []
        with ops.run_lock("scan"):
            self.assertEqual(ops.run_daily_scan(runners=self.runners(calls),
                                                progress=lambda *a: None), ["already running"])
        self.assertEqual(calls, [])


class SelfHealTest(OpsCase):
    def test_rules(self):
        spawned = []
        spawn = lambda: spawned.append(1)
        heal = lambda now: ops.self_heal(now=now, spawn=spawn, progress=lambda *a: None)
        self.assertEqual(heal(datetime(2026, 11, 14, 12)), "not due")     # Saturday
        self.assertEqual(heal(datetime(2026, 11, 10, 9, 59)), "not due")  # before 10:00
        self.assertEqual(heal(datetime(2026, 11, 10, 10)), "started")
        self.assertEqual(heal(datetime(2026, 11, 10, 11)), "started")
        self.assertEqual(heal(datetime(2026, 11, 10, 12)), "gave up")     # twice a day at most
        self.assertEqual(len(spawned), 2)
        self.assertEqual(heal(datetime(2026, 11, 11, 10)), "started")     # a new day
        with ops.run_lock("scan"):
            self.assertEqual(heal(datetime(2026, 11, 12, 10)), "running")
        with mock.patch.object(ops, "_now", return_value=datetime(2026, 11, 12, 8)):
            ops.record("scan", "ok")
        self.assertEqual(heal(datetime(2026, 11, 12, 10)), "done today")
        self.assertEqual(len(spawned), 3)


class BackupTest(OpsCase):
    def make_journal(self):
        con = sqlite3.connect(self.data / "journal.sqlite3")
        con.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, ticker TEXT)")
        con.execute("INSERT INTO trades (ticker) VALUES ('HPQ')")
        con.commit()
        con.close()

    def test_daily_copy_prune_and_integrity(self):
        self.make_journal()
        (self.data / "alerts.jsonl").write_text('{"kind": "stop"}\n')
        r = ops.backup_daily(today=date(2026, 11, 10))
        self.assertEqual(r["status"], "made")
        copy = self.data / "backups" / "2026-11-10" / "journal.sqlite3"
        con = sqlite3.connect(copy)
        self.assertEqual(con.execute("SELECT ticker FROM trades").fetchall(), [("HPQ",)])
        con.close()
        self.assertTrue((copy.parent / "alerts.jsonl").exists())
        self.assertEqual(ops.backup_daily(today=date(2026, 11, 10))["status"], "exists")
        for k in range(1, 6):
            ops.backup_daily(today=date(2026, 11, 10 + k), keep=3)
        days = sorted(p.name for p in (self.data / "backups").iterdir())
        self.assertEqual(days, ["2026-11-13", "2026-11-14", "2026-11-15"])
        self.assertEqual(ops.latest_backup(), "2026-11-15")

    def test_corrupt_journal_is_not_copied(self):
        (self.data / "journal.sqlite3").write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)
        with self.assertRaises(Exception):
            ops.backup_daily(today=date(2026, 11, 10))
        self.assertFalse((self.data / "backups" / "2026-11-10" / "DONE").exists())

    def test_no_journal_yet(self):
        self.assertEqual(ops.backup_daily(today=date(2026, 11, 10))["status"], "made")


class LogTest(OpsCase):
    def test_rotate(self):
        p = self.data / "scan.log"
        p.write_text("x" * 50)
        self.assertFalse(ops.rotate_log(p, max_bytes=100))
        for k in range(4):
            p.write_text(str(k) * 200)
            self.assertTrue(ops.rotate_log(p, max_bytes=100, keep=3))
        self.assertFalse(p.exists())
        self.assertEqual((self.data / "scan.log.1").read_text()[0], "3")
        self.assertEqual((self.data / "scan.log.3").read_text()[0], "1")
        self.assertFalse((self.data / "scan.log.4").exists())

    def test_stamped_lines(self):
        buf = io.StringIO()
        w = ops._Stamped(buf)
        with mock.patch.object(ops, "_now", return_value=datetime(2026, 11, 10, 7, 30, 5)):
            w.write("one\ntw")
            w.write("o\nthree\n")
        self.assertEqual(buf.getvalue(), "[2026-11-10 07:30:05] one\n[2026-11-10 07:30:05] two\n"
                                         "[2026-11-10 07:30:05] three\n")


class DoctorTest(OpsCase):
    def scan_file(self, market, generated, **extra):
        d = self.data / "scans"
        d.mkdir(exist_ok=True)
        (d / f"{market}_latest.json").write_text(json.dumps({"generated": generated, **extra}))

    def test_scan_ages(self):
        now = datetime(2026, 11, 10, 12)                   # Tuesday
        self.scan_file("us", "2026-11-10T07:35:00")
        self.scan_file("de", "2026-11-09T07:35:00")
        self.scan_file("in", "2026-11-06T07:35:00", data_stale={"note": "old"})
        levels = {c: lvl for lvl, c, _ in ops.check_scans(["us", "de", "in"], now)}
        self.assertEqual(levels, {"scan us": "OK", "scan de": "WARN", "scan in": "FAIL",
                                  "scan comeback": "FAIL"})
        early = {c: lvl for lvl, c, _ in ops.check_scans(["de"], datetime(2026, 11, 10, 7))}
        self.assertEqual(early["scan de"], "OK")           # today's run is still to come
        self.scan_file("us", "2026-11-10T07:35:00", data_stale={"note": "old prices"})
        lvl, _, detail = ops.check_scans(["us"], now)[0]
        self.assertEqual(lvl, "WARN")
        self.assertIn("old prices", detail)

    def test_health_and_summary(self):
        with mock.patch.object(ops, "_now", return_value=datetime(2026, 11, 10, 9)):
            ops.record("scan:us", "failed", error="RuntimeError: x")
            ops.record("monitor", "ok")
        res = ops.check_health(datetime(2026, 11, 10, 12))
        d = {c: (lvl, det) for lvl, c, det in res}
        self.assertEqual(d["job scan:us"][0], "FAIL")
        self.assertEqual(d["monitor"][0], "WARN")          # 3 h since the last hourly run
        self.assertEqual(ops.summary(res)["status"], "fail")
        self.assertEqual(ops.summary([("OK", "a", "")])["status"], "ok")

    def test_quick_doctor_never_prints_the_config(self):
        root = Path(self.tmp.name)
        (root / "config.json").write_text(json.dumps({"dashboard_password": "s3cret-pw"}))
        out: list[str] = []
        with mock.patch.object(S, "ROOT", root):
            res = ops.doctor(quick=True, now=datetime(2026, 11, 10, 12), progress=out.append)
        self.assertIn(("OK", "config.json", "readable"), res)
        self.assertFalse(any("s3cret" in line for line in out))
        (root / "config.json").write_text("{broken")
        with mock.patch.object(S, "ROOT", root):
            res = ops.doctor(quick=True, progress=lambda *a: None)
        self.assertEqual([r[0] for r in res if r[1] == "config.json"], ["FAIL"])


class PriceRetryTest(unittest.TestCase):
    def frame(self, tickers, n=3):
        idx = pd.bdate_range("2026-11-02", periods=n)
        cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    def test_failed_batch_and_dropped_tickers_are_retried(self):
        with tempfile.TemporaryDirectory() as d:
            store = D.PriceStore(Path(d) / "p.sqlite3")
            calls = []

            def download(batch):
                calls.append(list(batch))
                if len(calls) == 1:
                    raise RuntimeError("rate limited")      # the whole first batch
                if len(calls) == 2:
                    return self.frame([t for t in batch if t != "C"])   # C dropped
                return self.frame([t for t in batch if t != "Z"])       # Z never comes
            with mock.patch("builtins.print"):
                res = D.refresh_prices(store, ["A", "B", "C", "D", "Z"], batch_size=2,
                                       retry_batch_size=10, retry_pause_s=0, download=download)
            store.close()
        self.assertEqual(calls[:3], [["A", "B"], ["C", "D"], ["Z"]])
        self.assertEqual(sorted(calls[3]), ["A", "B", "C", "Z"])
        self.assertEqual({t: n > 0 for t, n in res.items()},
                         {"A": True, "B": True, "C": True, "D": True, "Z": False})

    def test_flat_columns_are_not_given_to_every_ticker(self):
        flat = pd.DataFrame(1.0, index=pd.bdate_range("2026-11-02", periods=2),
                            columns=["Open", "High", "Low", "Close", "Volume"])
        self.assertIsNone(D._frame_for(flat, "A", ["A", "B"]))
        self.assertIsNotNone(D._frame_for(flat, "A", ["A"]))


class SessionGuardTest(unittest.TestCase):
    def bars(self, last: str, n=5):
        idx = pd.bdate_range(end=last, periods=n)
        return pd.DataFrame({"close": range(n)}, index=idx)

    def test_unfinished_bar_dropped(self):
        utc = lambda *a: datetime(*a, tzinfo=timezone.utc)
        df = self.bars("2026-11-10")
        # 07:30 CET = 06:30 UTC = 12:00 in Kolkata: India still trading
        self.assertEqual(len(S.drop_unfinished_bar(df, "in", utc(2026, 11, 10, 6, 30))), 4)
        # 11:00 UTC = 16:30 in Kolkata: closed (15:45 with the margin)
        self.assertEqual(len(S.drop_unfinished_bar(df, "in", utc(2026, 11, 10, 11))), 5)
        # Xetra at 14:00 UTC (15:00 Berlin) still open; New York at 21:30 UTC closed
        self.assertEqual(len(S.drop_unfinished_bar(df, "de", utc(2026, 11, 10, 14))), 4)
        self.assertEqual(len(S.drop_unfinished_bar(df, "us", utc(2026, 11, 10, 21, 30))), 5)
        # yesterday's bar is never touched
        self.assertEqual(len(S.drop_unfinished_bar(self.bars("2026-11-09"), "in",
                                                   utc(2026, 11, 10, 6, 30))), 5)

    def test_staleness(self):
        utc = lambda *a: datetime(*a, tzinfo=timezone.utc)
        now = utc(2026, 11, 10, 6, 30)        # Tue 07:30 CET: last US session = Mon 9 Nov
        self.assertEqual(S.last_finished_session("us", now), date(2026, 11, 9))
        self.assertIsNone(S.price_staleness(self.bars("2026-11-09"), "us", now))
        self.assertIsNone(S.price_staleness(self.bars("2026-11-06"), "us", now))  # a holiday
        st = S.price_staleness(self.bars("2026-11-05"), "us", now)
        self.assertEqual(st["sessions_behind"], 2)
        self.assertIn("2026-11-05", st["note"])
        self.assertIsNotNone(S.price_staleness(pd.DataFrame(), "us", now))


class MonitorUncheckedTest(unittest.TestCase):
    def test_open_position_without_price_alerts_once(self):
        from atlas2 import monitor as M

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data" / "scans").mkdir(parents=True)
            (root / "data" / "scans" / "us_latest.json").write_text("{broken")
            trade = {"id": 1, "ticker": "INFY.NS", "stop": 1400.0, "target": None,
                     "exit_price": None, "entry_date": "2026-11-02"}
            journal = mock.MagicMock()
            journal.all_trades.return_value = [trade]
            notes, msgs = [], []
            with mock.patch.object(M, "ROOT", root), \
                    mock.patch.object(M, "STATE_PATH", root / "data" / "state.json"), \
                    mock.patch.object(M, "LOG_PATH", root / "data" / "alerts.jsonl"), \
                    mock.patch.object(M, "Journal", return_value=journal), \
                    mock.patch.object(M, "_latest_prices", return_value={}), \
                    mock.patch.object(M, "time_exit_checks", return_value=[]), \
                    mock.patch.object(M, "_notify", side_effect=lambda t, m: notes.append(t)):
                first = M.run_monitor(force=True, progress=msgs.append)
                second = M.run_monitor(force=True, progress=msgs.append)
            self.assertEqual([a["kind"] for a in first], ["unchecked"])
            self.assertIn("1400", first[0]["message"])
            self.assertEqual(second, [])                       # once a day
            self.assertEqual(notes, ["Shaikh Capital — STOP NOT CHECKED"])
            self.assertTrue(any("us_latest.json unreadable" in m for m in msgs))


class LoginLimitTest(unittest.TestCase):
    def test_fails_expire(self):
        from atlas2 import server

        server._LOGIN_FAILS.clear()
        server._LOGIN_FAILS["1.2.3.4"] = [0.0] * 3 + [1000.0] * 2
        self.assertEqual(len(server._recent_fails("1.2.3.4", 1000.0 + 60)), 2)
        self.assertEqual(server._recent_fails("1.2.3.4", 1000.0 + server.LOGIN_WINDOW_S + 1), [])
        self.assertNotIn("1.2.3.4", server._LOGIN_FAILS)


try:
    from starlette.testclient import TestClient
except Exception:                       # needs httpx
    TestClient = None


@unittest.skipIf(TestClient is None, "starlette TestClient needs httpx")
class LoginEndpointTest(unittest.TestCase):
    def test_lockout_after_ten_wrong_passwords(self):
        from atlas2 import server

        server._LOGIN_FAILS.clear()
        with mock.patch.object(server, "_dashboard_password", return_value="right"):
            c = TestClient(server.app)
            for _ in range(server.LOGIN_MAX_FAILS):
                self.assertEqual(c.post("/login", data={"password": "x"},
                                        follow_redirects=False).status_code, 401)
            self.assertEqual(c.post("/login", data={"password": "right"},
                                    follow_redirects=False).status_code, 429)
        server._LOGIN_FAILS.clear()


if __name__ == "__main__":
    unittest.main()
