"""Tests for atlas2.leverage_signal (the dashboard's Nasdaq signal tab) and /api/leverage.

Synthetic series only, temp dirs only; the downloader and the notifier are stubs, so
nothing touches data/, config.json, the network or macOS notifications.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from atlas2 import leverage_signal as LS
from atlas2 import server
from atlas2 import strategy_lab as L1
from atlas2 import strategy_lab2 as L2

try:
    from starlette.testclient import TestClient
    _TESTCLIENT_ERROR = None
except Exception as exc:  # starlette raises RuntimeError without httpx, not ImportError
    TestClient, _TESTCLIENT_ERROR = None, f"{exc.__class__.__name__}: {exc}"


def make_frames(n: int = 700, seed: int = 3) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(n) % 200 < 100, 0.008, 0.022)       # calm and wild spells
    r = rng.normal(0.0005, vol)
    ndx = 10000.0 * np.cumprod(1 + r)
    tb = np.full(n, 4.0)
    tq = 50.0 * np.cumprod(1 + np.nan_to_num(L1.synthetic_fund_returns(ndx, tb, 3)))
    qqq = 300.0 * np.cumprod(1 + r)

    def f(c, a=None):
        return pd.DataFrame({"close": c, "adj_close": c if a is None else a}, index=dates)

    return {"^NDX": f(ndx), "^IRX": f(tb), "QQQ": f(qqq), "TQQQ": f(tq)}


def stub_fetch(frames):
    return lambda tickers: {t: frames[t] for t in tickers}


class Env:
    """Temp paths for one test, and a runner with stubbed I/O."""

    def __init__(self, d: str):
        self.root = Path(d)
        self.series = self.root / "series.sqlite3"
        self.state = self.root / "paper.json"
        self.out = self.root / "signal.json"
        self.notes: list[tuple[str, str]] = []

    def run(self, frames=None, fetch=None, **kw) -> dict:
        if fetch is None and frames is not None:
            fetch = stub_fetch(frames)
        return LS.run_leverage_signal(series_path=self.series, state_path=self.state,
                                      out_path=self.out, lab_path=self.root / "nolab.json",
                                      fetch=fetch, notifier=lambda t, m: self.notes.append((t, m)),
                                      progress=lambda *a: None, **kw)


def truncate(frames: dict, n: int) -> dict:
    return {t: f.iloc[:n] for t, f in frames.items()}


class SignalTest(unittest.TestCase):
    def test_signal_matches_the_lab_rule(self):
        frames = make_frames()
        with tempfile.TemporaryDirectory() as d:
            out = Env(d).run(frames)
        close = frames["^NDX"]["close"].to_numpy(float)
        sig = L2.vol_target_signal(close, 0.30, 60, "none")
        self.assertAlmostEqual(out["signal"]["new"], round(sig[-1], 3))
        self.assertAlmostEqual(out["signal"]["held"], round(sig[-2], 3))
        self.assertEqual(out["signal"]["trade_due"], bool(sig[-1] != sig[-2]))
        self.assertAlmostEqual(out["signal"]["fund_share_pct"], round(sig[-1] / 3 * 100, 1))
        self.assertEqual(out["as_of"], frames["^NDX"].index[-1])
        self.assertEqual(len(out["history"]), LS.HISTORY_SESSIONS)

    def test_trade_due_and_one_notification_per_change(self):
        frames = make_frames()
        sig = L2.vol_target_signal(frames["^NDX"]["close"].to_numpy(float), 0.30, 60, "none")
        k = int(np.nonzero(sig[300:] != sig[299:-1])[0][0]) + 300      # a close that moved it
        part = truncate(frames, k + 1)
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            out = env.run(part)
            self.assertTrue(out["signal"]["trade_due"])
            self.assertTrue(out["notified"])
            env.run(part)                                   # the same close again: no repeat
            self.assertEqual(len(env.notes), 1)
            self.assertIn(f"{sig[k]:.2f}x", env.notes[0][1])
            self.assertNotIn('"', env.notes[0][1])          # safe inside the AppleScript string
        quiet = int(np.nonzero(sig[300:] == sig[299:-1])[0][0]) + 300
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            self.assertFalse(env.run(truncate(frames, quiet + 1))["signal"]["trade_due"])
            env.run(part, notify=False)
            self.assertEqual(env.notes, [])

    def test_paper_start_is_fixed_and_replayed_with_real_fund_prices(self):
        frames = make_frames()
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            first = env.run(truncate(frames, 500))
            later = env.run(truncate(frames, 560))
            state = json.loads(env.state.read_text())
        start = frames["^NDX"].index[499]
        self.assertEqual(state["start"], start)
        self.assertEqual(first["paper"]["start"], start)
        self.assertEqual(later["paper"]["start"], start)
        self.assertEqual(later["paper"]["sessions"], 60)
        self.assertTrue(later["paper"]["trades"][0]["opening"])
        # the same account as lab 2's realistic account, on the TQQQ series
        part = truncate(frames, 560)
        close = part["^NDX"]["close"].to_numpy(float)
        tq = part["TQQQ"]["adj_close"].to_numpy(float)
        fund = np.full(len(tq), np.nan)
        fund[1:] = tq[1:] / tq[:-1] - 1
        sig = L2.vol_target_signal(close, 0.30, 60, "none")
        eq, h = L2.run_vol_account(fund, L1.cash_returns(part["^IRX"]["close"].to_numpy(float)),
                                   sig, 499, 559, return_holdings=True)
        self.assertAlmostEqual(later["paper"]["value_per_10000"], round(10000 * h.sum(), 2))
        self.assertAlmostEqual(later["paper"]["curve"][-1][1], round(eq[559], 5))
        moves = [d for d in range(500, 560) if sig[d - 1] != sig[d - 2]]
        self.assertEqual(len(later["paper"]["trades"]), 1 + len(moves))

    def test_download_failure_falls_back_to_the_cache(self):
        frames = make_frames()

        def boom(tickers):
            raise RuntimeError("no network")

        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            env.run(frames)
            out = env.run(fetch=boom)
        self.assertTrue(out["data"]["stale"])
        self.assertIn("no network", out["data"]["warning"])
        self.assertEqual(out["as_of"], frames["^NDX"].index[-1])

    def test_unreadable_state_is_kept_and_restarted(self):
        frames = make_frames()
        with tempfile.TemporaryDirectory() as d:
            env = Env(d)
            env.state.write_text("{not json")
            out = env.run(frames)
            self.assertTrue((env.root / "paper.json.bad").exists())
            self.assertIn("unreadable", out["data"]["note"])
            self.assertEqual(out["paper"]["start"], frames["^NDX"].index[-1])

    def test_lab_evidence_spans(self):
        lab = {"generated": "x", "vol_target_leverage": {
            "winner": {"name": "v"},
            "variants": {"v": {"periods": {
                "test": {"start": "2009-12-31", "end": "2026-11-02", "cagr_pct": 23.5},
                "full": {"start": "1986-09-30", "end": "2026-11-02", "cagr_pct": 19.7}}}},
            "benchmarks": {"test": {"QQQ": {"cagr_pct": 18.4}},
                           "full": {"nasdaq_1x_model": {"cagr_pct": 14.8}}}}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lab.json"
            p.write_text(json.dumps(lab))
            ev = LS.lab_evidence(p)
            self.assertIsNone(LS.lab_evidence(Path(d) / "missing.json"))
        self.assertEqual(ev["test"]["span"], "2010–2026")
        self.assertEqual(ev["full"]["span"], "1986–2026")
        self.assertEqual(ev["qqq_test"]["cagr_pct"], 18.4)


class LeverageHandlerTest(unittest.TestCase):
    """The route handler called directly (the HTTP test below needs httpx)."""

    def test_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signal.json"
            body = lambda: json.loads(server.leverage_signal().body)
            with mock.patch.object(server, "LEVERAGE_SIGNAL_PATH", path):
                self.assertEqual(body(), {"available": False})
                path.write_text(json.dumps({"available": True, "as_of": "2026-11-09"}))
                self.assertEqual(body()["as_of"], "2026-11-09")
                path.write_text("{broken")
                self.assertFalse(body()["available"])
                self.assertIn("error", body())


@unittest.skipIf(TestClient is None, f"starlette TestClient not importable ({_TESTCLIENT_ERROR})")
class LeverageEndpointTest(unittest.TestCase):
    def test_get_api_leverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signal.json"
            with mock.patch.object(server, "LEVERAGE_SIGNAL_PATH", path), \
                    mock.patch.object(server, "_dashboard_password", return_value=None):
                client = TestClient(server.app)
                self.assertEqual(client.get("/api/leverage").json(), {"available": False})
                path.write_text(json.dumps({"available": True, "as_of": "2026-11-09"}))
                self.assertEqual(client.get("/api/leverage").json()["as_of"], "2026-11-09")
                path.write_text("{broken")
                r = client.get("/api/leverage").json()
                self.assertFalse(r["available"])
                self.assertIn("error", r)


if __name__ == "__main__":
    unittest.main()
