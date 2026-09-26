"""Tests for atlas2.backtest (v2) and the backtest-driven tuning in atlas2.scan.

Synthetic data only; files go to temp dirs. Never touches data/ or config.json.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime as datetime_cls
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from atlas2 import backtest as bt
from atlas2 import scan
from atlas2.data import BENCHMARKS, PriceStore
from atlas2.patterns import PatternHit

REPO = Path(__file__).resolve().parent.parent
TUNING = {"auto_disable": True, "min_avg_r": 0.0, "min_trades": 30,
          "require_proven_edge": False, "disabled": {"us": [], "de": [], "in": []}}


def arr(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=float)


def random_frame(seed: int, n: int = 650, start: str = "2018-01-01", price: float = 50.0,
                 drift: float = 0.0006, vol: float = 0.018,
                 volume: float = 2e6) -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV bars (business days)."""
    rng = np.random.default_rng(seed)
    closes = price * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    prev = np.concatenate([[price], closes[:-1]])
    opens = prev * np.exp(0.004 * rng.standard_normal(n))
    highs = np.maximum(opens, closes) * (1 + 0.012 * np.abs(rng.standard_normal(n)))
    lows = np.minimum(opens, closes) * (1 - 0.012 * np.abs(rng.standard_normal(n)))
    idx = pd.bdate_range(start, periods=n, name="date")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": np.full(n, volume)}, index=idx)


def random_panel() -> bt.MarketPanel:
    """12 synthetic tickers; used in-process and by the PYTHONHASHSEED subprocess."""
    frames = {f"T{k:02d}": random_frame(100 + k, n=260) for k in range(12)}
    return bt.MarketPanel(bt.frames_to_long(frames))


def random_trade() -> dict:
    d = random_frame(100, n=260).index[120].strftime("%Y-%m-%d")
    return {"ticker": "T03", "entry_date": d, "pattern": "bull_flag",
            "entry_price": 50.0, "stop": 47.5, "target": 56.0}


def make_trade(**kw) -> dict:
    t = {"market": "us", "ticker": "AAA", "pattern": "bull_flag", "conf": 0.7,
         "entry_date": "2020-01-02", "exit_date": "2020-01-20", "outcome": "time",
         "hold_bars": 10, "r": 0.5, "r_net": 0.45, "ret_pct": 2.0, "ret_net_pct": 1.7,
         "index_ret_pct": 1.0, "universe_ret_pct": 1.2, "excess_vs_index_pct": 0.7,
         "excess_vs_universe_pct": 0.5, "rand_r_net": 0.1, "rand_ret_net_pct": 0.3}
    t.update(kw)
    return t


def quarter_dates(n: int, start_year: int = 2018) -> list[str]:
    """One date per quarter (and per month), n quarters."""
    out = []
    for k in range(n):
        y, q = start_year + k // 4, k % 4
        out.append(f"{y}-{3 * q + 1:02d}-15")
    return out


class CostMathTest(unittest.TestCase):
    def test_target_exit_costs(self):
        res = bt.trade_returns(100.0, 95.0, 110.0, "target", 0.15, 0.05)
        self.assertAlmostEqual(res["r"], 2.0)
        # entry 100.15 effective, exit 110 * 0.9985 = 109.835
        self.assertAlmostEqual(res["r_net"], (109.835 - 100.15) / 5)
        self.assertAlmostEqual(res["ret_pct"], 10.0)
        self.assertAlmostEqual(res["ret_net_pct"], (109.835 / 100.15 - 1) * 100)
        self.assertAlmostEqual(res["cost_pct_round_trip"], 0.30)

    def test_stop_exit_pays_slippage(self):
        res = bt.trade_returns(100.0, 95.0, 95.0, "stop", 0.15, 0.05)
        self.assertAlmostEqual(res["r"], -1.0)
        exit_eff = 95.0 * (1 - 0.0015 - 0.0005)  # 94.81
        self.assertAlmostEqual(res["r_net"], (exit_eff - 100.15) / 5)
        self.assertAlmostEqual(res["r_net"], -1.068)
        self.assertAlmostEqual(res["ret_net_pct"], (exit_eff / 100.15 - 1) * 100)
        self.assertAlmostEqual(res["cost_pct_round_trip"], 0.35)

    def test_time_exit_has_no_slippage(self):
        a = bt.trade_returns(100.0, 95.0, 101.0, "time", 0.10, 0.05)
        self.assertAlmostEqual(a["cost_pct_round_trip"], 0.20)
        self.assertAlmostEqual(a["r_net"], (101 * 0.999 - 100.1) / 5)

    def test_entry_slippage(self):
        a = bt.trade_returns(100.0, 95.0, 110.0, "target", 0.15, 0.05, entry_slip_pct=0.1)
        self.assertAlmostEqual(a["r"], 2.0)
        self.assertAlmostEqual(a["r_net"], (110 * 0.9985 - 100 * 1.0025) / 5)
        self.assertAlmostEqual(a["cost_pct_round_trip"], 0.40)
        b = bt.trade_returns(100.0, 95.0, 95.0, "stop", 0.15, 0.05, entry_slip_pct=0.1)
        self.assertAlmostEqual(b["cost_pct_round_trip"], 0.45)


class ExitRulesTest(unittest.TestCase):
    def test_fill_bar_stop_counts_as_stop(self):
        # fill bar touches both the stop and the target: stop assumed first
        o, h, l, c = arr(100, 101), arr(115, 102), arr(94, 100), arr(100, 101)
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 40),
                         (0, 95.0, "stop"))

    def test_fill_bar_target(self):
        o, h, l, c = arr(100, 101), arr(111, 102), arr(99, 100), arr(110, 101)
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 40),
                         (0, 110.0, "target"))

    def test_later_bar_gap_through_stop_exits_at_open(self):
        o, h, l, c = arr(100, 101, 90), arr(102, 103, 92), arr(99, 99, 88), arr(101, 102, 91)
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 40),
                         (2, 90.0, "stop"))

    def test_later_bar_intraday_stop_exits_at_stop(self):
        o, h, l, c = arr(100, 101, 98), arr(102, 103, 99), arr(99, 99, 94), arr(101, 102, 96)
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 40),
                         (2, 95.0, "stop"))

    def test_later_bar_stop_checked_before_target(self):
        o, h, l, c = arr(100, 101), arr(102, 112), arr(99, 94), arr(101, 100)
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 40),
                         (1, 95.0, "stop"))

    def test_later_bar_gap_above_target_exits_at_open(self):
        o, h, l, c = arr(100, 101, 113), arr(102, 103, 115), arr(99, 99, 112), arr(101, 102, 114)
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 40),
                         (2, 113.0, "target"))

    def test_time_exit_and_open_at_end(self):
        n = 10
        o = h = l = c = np.full(n, 100.0)
        c = c.copy()
        c[4] = 103.0
        # max_hold 4 from fill 0: time exit at the close of bar 4
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 4),
                         (4, 103.0, "time"))
        # data ends before max_hold: open_at_end at the last close
        self.assertEqual(bt.simulate_exit(o, h, l, c, 5, 100.0, 95.0, 110.0, 40),
                         (9, 100.0, "open_at_end"))

    def _flat(self, n=8):
        return [np.full(n, 100.0) for _ in range(4)]

    def test_stop_and_target_live_on_the_time_exit_bar(self):
        o, h, l, c = self._flat()
        l[4], c[4] = 90.0, 99.0
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 4), (4, 95.0, "stop"))
        o, h, l, c = self._flat()
        o[4], l[4] = 93.0, 92.0
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 4), (4, 93.0, "stop"))
        o, h, l, c = self._flat()
        h[4] = 111.0
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 4),
                         (4, 110.0, "target"))
        o, h, l, c = self._flat()
        l[5] = 90.0                                    # only after the time exit
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 4), (4, 100.0, "time"))
        o, h, l, c = self._flat(5)                     # data ends exactly at fill + max_hold
        l[4] = 90.0
        self.assertEqual(bt.simulate_exit(o, h, l, c, 0, 100.0, 95.0, 110.0, 4), (4, 95.0, "stop"))


class EntryRulesTest(unittest.TestCase):
    def test_triggered_fills_at_next_open(self):
        o = arr(100, 101.5, 99)
        self.assertEqual(bt.enter_triggered(o, 0, 100.0, 95.0, 3.0), (1, 101.5, None))

    def test_triggered_gap_too_far(self):
        o = arr(100, 103.5)
        self.assertEqual(bt.enter_triggered(o, 0, 100.0, 95.0, 3.0), (1, None, "gap_too_far"))

    def test_triggered_gap_below_stop(self):
        o = arr(100, 94.0)
        self.assertEqual(bt.enter_triggered(o, 0, 100.0, 95.0, 3.0), (1, None, "gap_below_stop"))

    def test_triggered_at_data_end(self):
        self.assertEqual(bt.enter_triggered(arr(100), 0, 100.0, 95.0, 3.0), (0, None, "no_breakout"))

    def test_forming_waits_for_breakout(self):
        o = arr(97, 97, 98, 99)
        h = arr(98, 98.5, 99, 100.5)
        l = arr(96, 96, 97, 98.5)
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0), (3, 100.0, None))

    def test_forming_gap_open_fills_at_open(self):
        o, h, l = arr(97, 101), arr(98, 102), arr(96, 100)
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0), (1, 101.0, None))

    def test_forming_gap_too_far(self):
        o, h, l = arr(97, 104), arr(98, 105), arr(96, 103)
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0),
                         (1, None, "gap_too_far"))

    def test_forming_abandons_on_pre_trigger_stop(self):
        o = arr(97, 97, 96, 99)
        h = arr(98, 98, 97, 101)   # breakout on bar 3 comes too late
        l = arr(96, 96, 94.5, 98)  # stop broken on bar 2
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0),
                         (2, None, "failed_before_trigger"))

    def test_forming_no_breakout_within_wait(self):
        o, h, l = np.full(30, 97.0), np.full(30, 98.0), np.full(30, 96.0)
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0),
                         (15, None, "no_breakout"))
        self.assertEqual(bt.enter_forming(o, h, l, 20, 100.0, 95.0, 15, 3.0),
                         (29, None, "no_breakout"))       # resolved at the data end

    def test_forming_same_bar_entry_and_stop_fills_then_stops(self):
        o, h, l, c = arr(97, 97), arr(98, 100.5), arr(96, 94.5), arr(97, 96)
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0), (1, 100.0, None))
        self.assertEqual(bt.simulate_exit(o, h, l, c, 1, 100.0, 95.0, 110.0, 40), (1, 95.0, "stop"))

    def test_forming_open_through_stop_abandons(self):
        o, h, l = arr(97, 94.8), arr(98, 100.5), arr(96, 94)
        self.assertEqual(bt.enter_forming(o, h, l, 0, 100.0, 95.0, 15, 3.0),
                         (1, None, "failed_before_trigger"))


class CloseConfirmTest(unittest.TestCase):
    def test_next_open_after_a_close_above_the_entry(self):
        o, l, c = arr(97, 98, 100.5, 101), arr(96, 97, 98, 100), arr(97, 99, 101, 102)
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0), (3, 101.0, None))

    def test_ran_away(self):
        o, l, c = arr(97, 98, 101), arr(96, 97, 100), arr(97, 104, 105)
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0),
                         (1, None, "ran_away"))

    def test_failed_before_trigger(self):
        o, l, c = arr(97, 97, 98), arr(96, 94, 97), arr(97, 101, 99)
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0),
                         (1, None, "failed_before_trigger"))

    def test_gaps_at_the_next_open(self):
        o, l, c = arr(97, 98, 104), arr(96, 97, 103), arr(97, 101, 104)
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0),
                         (2, None, "gap_too_far"))
        o, l, c = arr(97, 98, 94), arr(96, 97, 93), arr(97, 101, 94)
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0),
                         (2, None, "gap_below_stop"))

    def test_no_breakout_at_data_end(self):
        o, l, c = arr(97, 98), arr(96, 97), arr(97, 101)   # confirmed on the last bar
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0),
                         (1, None, "no_breakout"))
        o, l, c = np.full(5, 97.0), np.full(5, 96.0), np.full(5, 97.0)
        self.assertEqual(bt.enter_close_confirm(o, l, c, 0, 100.0, 95.0, 15, 3.0),
                         (4, None, "no_breakout"))


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])

    def rec(self, hit, fill, s, outcome="target", exit_price=110.0):
        return bt._record("us", "AAA", self.dates, hit, "healthy", 0, 1, fill, 3, exit_price,
                          outcome, s)

    def test_buy_stop_entries_pay_entry_slippage(self):
        s = bt.Settings()
        forming = bt.Hit("bull_flag", 0.8, 2.0, 100.0, 95.0, 110.0, "forming")
        for fill in (100.0, 101.0):                    # intrabar and gap-open buy-stop fills
            t = self.rec(forming, fill, s)
            want = bt.trade_returns(fill, 95.0, 110.0, "target", s.cost_pct("us"),
                                    s.stop_slip("us"), s.entry_slip("us"))
            self.assertEqual(t["entry_kind"], "buy_stop")
            self.assertEqual(t["r_net"], round(want["r_net"], 4))
            self.assertEqual(t["cost_pct_round_trip"], round(want["cost_pct_round_trip"], 3))
        t = self.rec(forming, 101.0, s)
        self.assertEqual(t["entry_level"], 100.0)
        self.assertEqual(t["risk_pct"], round(6 / 101 * 100, 3))
        self.assertEqual(t["risk_pct_plan"], 5.0)
        self.assertAlmostEqual(t["r_plan_net"], t["r_net"] * 6 / 5, places=3)

    def test_next_open_entries_pay_no_entry_slippage(self):
        s = bt.Settings()
        trig = bt.Hit("bull_flag", 0.8, 2.0, 100.0, 95.0, 110.0, "triggered")
        forming = bt.Hit("bull_flag", 0.8, 2.0, 100.0, 95.0, 110.0, "forming")
        confirm = bt.Settings(entry_mode="close_confirm")
        want = round(bt.trade_returns(101.0, 95.0, 110.0, "target", s.cost_pct("us"),
                                      s.stop_slip("us"))["r_net"], 4)
        for hit, st in ((trig, s), (forming, confirm)):
            t = self.rec(hit, 101.0, st)
            self.assertEqual(t["entry_kind"], "next_open")
            self.assertEqual(t["r_net"], want)

    def test_every_trade_carries_the_contract_keys(self):
        t = self.rec(bt.Hit("bull_flag", 0.8, 2.0, 100.0, 95.0, 110.0, "forming"), 100.0,
                     bt.Settings())
        for key in ("market", "ticker", "pattern", "conf", "status_at_signal", "regime",
                    "signal_date", "entry_date", "exit_date", "entry_price", "stop", "target",
                    "rr", "risk_pct", "outcome", "hold_bars", "r", "r_net", "ret_pct",
                    "ret_net_pct", "cost_pct_round_trip", "index_ret_pct", "universe_ret_pct",
                    "excess_vs_index_pct", "excess_vs_universe_pct", "rand_r_net",
                    "rand_ret_net_pct", "excess_vs_index_r", "excess_vs_universe_r"):
            self.assertIn(key, t)


class GradeProxyTest(unittest.TestCase):
    def test_pattern_score_cap(self):
        psr, comp, _ = bt.grade_proxy(80.0, 0.95, 4.0, False)
        self.assertEqual(psr, 100)
        self.assertAlmostEqual(comp, 0.55 * 80 + 0.45 * 100)

    def test_grade_thresholds_on_unrounded_composite(self):
        for comp_target, grade in ((72.0, "A"), (71.99, "B"), (58.0, "B"), (57.99, "C")):
            t = (comp_target - 0.45 * 80) / 0.55     # conf 0.8, rr 2 -> pattern score 80
            _, comp, g = bt.grade_proxy(t + 1e-9, 0.8, 2.0, False)
            self.assertEqual(g, grade, comp_target)
        t = (71.996 - 36) / 0.55                     # rounds to 72.0 but is below the line
        _, comp, g = bt.grade_proxy(t, 0.8, 2.0, False)
        self.assertEqual(round(comp, 1), 72.0)
        self.assertEqual(g, "B")

    def test_conflict_penalty(self):
        _, a, _ = bt.grade_proxy(80.0, 0.8, 2.0, False)
        _, b, _ = bt.grade_proxy(80.0, 0.8, 2.0, True)
        self.assertAlmostEqual(a - b, 10.0)

    def test_bearish_conflict_threshold(self):
        def bear(conf):
            return lambda win: [PatternHit("double_top", "bearish", 100.0, 105.0, 90.0, conf,
                                           "forming", 0, 0)]
        win = random_frame(1, n=50)
        for conf, want in ((0.55, True), (0.549, False)):
            with mock.patch.object(bt, "BEARISH_DETECTORS", [bear(conf)]):
                self.assertEqual(bt._bearish_conflict(win, bt.new_counters()), want)


class StatsTest(unittest.TestCase):
    def test_cluster_ci95_reference(self):
        lo, hi = bt.cluster_ci95([1, 2, 3, 4], ["a", "a", "b", "b"])
        # mean 2.5; cluster sums -2 and +2; SE = sqrt(2/1 * 8) / 4 = 1.0
        self.assertAlmostEqual(lo, 2.5 - 1.96)
        self.assertAlmostEqual(hi, 2.5 + 1.96)
        lo, hi = bt.cluster_ci95([1, 3, 2, 6], ["2020-01", "2020-02", "2020-02", "2020-03"])
        se = math.sqrt(1.5 * 14) / 4
        self.assertAlmostEqual(lo, 3 - 1.96 * se)
        self.assertAlmostEqual(hi, 3 + 1.96 * se)
        self.assertIsNone(bt.cluster_ci95([1, 2, 3], ["a", "a", "a"]))
        self.assertIsNone(bt.cluster_ci95([], []))

    def test_t_crit(self):
        for nu, want in ((9, 2.2622), (30, 2.0423), (38, 2.0244)):
            self.assertAlmostEqual(bt.t_crit(0.975, nu), want, delta=1e-3)

    def test_cluster_ci_singletons(self):
        xs = [0.5, -1.0, 2.0, 1.5, -0.3, 0.8, 1.1, -0.7, 0.2, 2.4, -1.2, 0.9]
        d = quarter_dates(12)
        lo, hi = bt.cluster_ci(xs, d)
        s = float(np.std(xs, ddof=1))
        half = s / math.sqrt(12) * bt.t_crit(0.975, 11)
        self.assertAlmostEqual((hi - lo) / 2, half)
        self.assertAlmostEqual((hi + lo) / 2, float(np.mean(xs)))

    def _paired(self, second_sign: int):
        xs, d = [], []
        for k, q in enumerate(quarter_dates(10)):
            v = 1.0 if k % 2 == 0 else -1.0
            y, m = q[:4], int(q[5:7])
            xs += [v, second_sign * v]
            d += [f"{y}-{m:02d}-10", f"{y}-{m + 1:02d}-10"]   # same quarter, different months
        return xs, d

    def test_quarter_se_used_when_larger(self):
        xs, d = self._paired(+1)
        se_m = math.sqrt(20 / 19 * 20) / 20
        se_q = math.sqrt(10 / 9 * 40) / 20
        self.assertGreater(se_q, se_m)
        lo, hi = bt.cluster_ci(xs, d)
        self.assertAlmostEqual(hi, 0 + bt.t_crit(0.975, 9) * se_q)
        self.assertAlmostEqual(lo, 0 - bt.t_crit(0.975, 9) * se_q)

    def test_month_se_used_when_quarter_se_smaller(self):
        xs, d = self._paired(-1)
        se_m = math.sqrt(20 / 19 * 20) / 20
        lo, hi = bt.cluster_ci(xs, d)
        self.assertAlmostEqual(hi, bt.t_crit(0.975, 9) * se_m)

    def test_cluster_ci_needs_ten_quarters(self):
        self.assertIsNone(bt.cluster_ci([1.0, 2.0] * 9, quarter_dates(9) * 2))
        self.assertIsNotNone(bt.cluster_ci([1.0, 2.0] * 5, quarter_dates(10)))
        self.assertIsNone(bt.cluster_ci([], []))
        self.assertIsNone(bt.cluster_ci([1.0, 2.0], ["2020-01-01"]))

    def test_pattern_stats_need_ten_quarters_for_a_range(self):
        """40 winning trades over 9 entry quarters: no range, so at best unproven."""
        dates = quarter_dates(9)
        ts = [make_trade(entry_date=dates[k % 9], r_net=0.4, r=0.45,
                         excess_vs_universe_pct=1.0 + 0.1 * (k % 5)) for k in range(40)]
        st = bt.pattern_stats(ts, TUNING)
        self.assertIsNone(st["excess_vs_universe_se"])
        self.assertIsNone(st["excess_vs_universe_ci95"])
        self.assertNotEqual(st["verdict"], "proven_edge")
        self.assertIn("too few quarters", st["verdict_reason"])
        dates = quarter_dates(10)                       # positive control
        st = bt.pattern_stats([dict(t, entry_date=dates[k % 10]) for k, t in enumerate(ts)], TUNING)
        self.assertIsNotNone(st["excess_vs_universe_se"])
        self.assertEqual(st["excess_vs_universe_df"], 9)

    def test_open_at_end_excluded_from_stats(self):
        trades = [make_trade(r_net=1.0, r=1.1), make_trade(r_net=-1.0, r=-0.9),
                  make_trade(outcome="open_at_end", r_net=5.0, r=5.0)]
        out = bt.summarize(trades, TUNING)
        self.assertEqual(out["total"]["trades"], 2)
        self.assertAlmostEqual(out["total"]["net_avg_r"], 0.0)
        self.assertEqual(out["per_pattern"]["bull_flag"]["trades"], 2)
        self.assertEqual(out["total"]["win_rate"], 50.0)

    def test_stats_fields_and_years(self):
        trades = [make_trade(entry_date=f"20{19 + k % 2}-0{1 + k % 3}-05",
                             excess_vs_universe_pct=1.0 if k % 2 else -0.5)
                  for k in range(30)]
        st = bt.summarize(trades, TUNING)["total"]
        for key in ("trades", "win_rate", "gross_win_rate", "avg_r", "net_avg_r", "avg_ret_pct",
                    "net_avg_ret_pct", "avg_hold_bars", "outcomes", "avg_index_ret_pct",
                    "avg_universe_ret_pct", "excess_vs_index_pct", "excess_vs_universe_pct",
                    "beat_index_rate", "random_entry_net_avg_r", "edge_vs_random_r",
                    "net_avg_r_ci95", "excess_vs_index_ci95", "excess_vs_universe_ci95",
                    "by_year", "positive_years", "years_counted", "verdict", "verdict_reason",
                    "excess_vs_universe_se", "edge_vs_random_r_ci95", "clearly_negative",
                    "gross_excess_vs_universe_pct", "avg_index_ret_price_pct"):
            self.assertIn(key, st)
        self.assertEqual(st["by_year"]["2019"]["trades"], 15)
        self.assertEqual(st["years_counted"], 2)
        self.assertEqual(st["positive_years"], 1)  # 2020 +1.0, 2019 -0.5
        self.assertAlmostEqual(st["edge_vs_random_r"], 0.45 - 0.1)

    def test_win_rate_is_after_fees(self):
        st = bt.summarize([make_trade(r=0.05, r_net=-0.02), make_trade(r=0.5, r_net=0.4)],
                          TUNING)["total"]
        self.assertEqual(st["win_rate"], 50.0)
        self.assertEqual(st["gross_win_rate"], 100.0)

    def test_positive_years_need_ten_trades(self):
        ts = ([make_trade(entry_date="2020-03-02", excess_vs_universe_pct=1.0)] * 12
              + [make_trade(entry_date="2021-03-02", excess_vs_universe_pct=1.0)] * 9)
        st = bt.summarize(ts, TUNING)["total"]
        self.assertEqual(st["years_counted"], 1)
        self.assertEqual(st["positive_years"], 1)

    def test_ranges_clustered_on_entry_dates(self):
        dates = quarter_dates(12)
        dates[3] = dates[2][:8] + "28"                  # two trades in one month
        xs = [1.0, 2.0, 3.0, -1.0, 4.0, 0.5, -2.0, 1.5, 2.5, -0.5, 3.5, 0.0]
        ts = [make_trade(entry_date=d, r_net=x, excess_vs_index_pct=x, excess_vs_universe_pct=x,
                         rand_r_net=0.0) for d, x in zip(dates, xs)]
        st = bt.summarize(ts, TUNING)["total"]
        lo, hi = bt.cluster_ci(xs, dates)
        self.assertEqual(st["net_avg_r_ci95"], [round(lo, 4), round(hi, 4)])
        self.assertEqual(st["excess_vs_index_ci95"], [round(lo, 3), round(hi, 3)])
        self.assertEqual(st["excess_vs_universe_ci95"], [round(lo, 3), round(hi, 3)])
        self.assertEqual(st["edge_vs_random_r_ci95"], [round(lo, 4), round(hi, 4)])

    def test_every_detector_gets_a_row(self):
        info = bt._new_pattern_info()
        info["double_bottom"].update({"detected": 40, "best": 12, "rr_rejected": 12, "max_rr": 0.97})
        out = bt.summarize([make_trade()], TUNING, info, 1.8)
        self.assertEqual(set(out["per_pattern"]), set(bt.detector_names()))
        db = out["per_pattern"]["double_bottom"]
        self.assertEqual(db["trades"], 0)
        self.assertEqual(db["verdict"], "insufficient_data")
        self.assertIn("risk:reward", db["verdict_reason"])
        self.assertIn("0.97", db["verdict_reason"])
        self.assertIsNone(db["net_avg_r"])

    def test_r_scaled_excess_is_informational(self):
        # 4% stop at -1%, 12% stop at +5%, market +1% each: % excess > 0, R excess < 0
        t1 = make_trade(r=-0.25, r_net=-0.25, ret_pct=-1.0, ret_net_pct=-1.0, universe_ret_pct=1.0,
                        excess_vs_universe_pct=-2.0, index_ret_pct=0.5, excess_vs_index_pct=-1.5,
                        excess_vs_universe_r=-0.25 - 1.0 / 4, excess_vs_index_r=-0.25 - 0.5 / 4)
        t2 = make_trade(r=5 / 12, r_net=5 / 12, ret_pct=5.0, ret_net_pct=5.0, universe_ret_pct=1.0,
                        excess_vs_universe_pct=4.0, index_ret_pct=0.5, excess_vs_index_pct=4.5,
                        excess_vs_universe_r=5 / 12 - 1.0 / 12, excess_vs_index_r=5 / 12 - 0.5 / 12)
        st = bt.pattern_stats([t1, t2], {**TUNING, "min_trades": 1})
        self.assertGreater(st["excess_vs_universe_pct"], 0)
        self.assertLess(st["excess_vs_universe_r"], 0)
        self.assertNotEqual(st["verdict"], "no_edge")    # still decided on the % metric


class VerdictTest(unittest.TestCase):
    base = {"trades": 100, "net_avg_r": 0.2, "excess_vs_index_pct": 0.5,
            "excess_vs_universe_pct": 0.6, "excess_vs_universe_ci95": [0.3, 0.9]}

    def test_insufficient_data(self):
        v, why = bt.pattern_verdict({**self.base, "trades": 12}, TUNING)
        self.assertEqual(v, "insufficient_data")
        self.assertIn("12", why)

    def test_no_edge_vs_index(self):
        v, why = bt.pattern_verdict({**self.base, "excess_vs_index_pct": -0.3}, TUNING)
        self.assertEqual(v, "no_edge")
        self.assertIn("0.30% per trade less than just holding the market", why)

    def test_no_edge_net_r(self):
        v, why = bt.pattern_verdict({**self.base, "net_avg_r": -0.05}, TUNING)
        self.assertEqual(v, "no_edge")
        self.assertIn("-0.05R", why)
        # min_avg_r applies to the net figure
        v, _ = bt.pattern_verdict(self.base, {**TUNING, "min_avg_r": 0.25})
        self.assertEqual(v, "no_edge")

    def test_no_edge_never_prints_negative_zero(self):
        v, why = bt.pattern_verdict({**self.base, "net_avg_r": -0.004}, TUNING)
        self.assertEqual(v, "no_edge")
        self.assertNotIn("-0.00", why)
        self.assertIn("0.00R", why)
        v, why = bt.pattern_verdict({**self.base, "excess_vs_universe_pct": -0.001}, TUNING)
        self.assertNotIn("0.00% per trade less", why)

    def test_no_edge_vs_universe(self):
        v, why = bt.pattern_verdict({**self.base, "excess_vs_universe_pct": 0.0}, TUNING)
        self.assertEqual(v, "no_edge")
        self.assertIn("equal-weight", why)

    def test_no_edge_reason_names_the_uncertainty(self):
        _, why = bt.pattern_verdict({**self.base, "excess_vs_universe_pct": -0.5,
                                     "excess_vs_universe_ci95": [-0.9, -0.1]}, TUNING)
        self.assertIn("95% range", why)
        self.assertIn("whole range is below zero", why)
        _, why = bt.pattern_verdict({**self.base, "excess_vs_universe_pct": -0.2,
                                     "excess_vs_universe_ci95": [-0.9, 0.5]}, TUNING)
        self.assertIn("not clearly worse", why)
        _, why = bt.pattern_verdict({**self.base, "excess_vs_universe_pct": -0.2,
                                     "gross_excess_vs_universe_pct": 0.1}, TUNING)
        self.assertIn("fees ate it", why)

    def test_no_edge_reason_when_it_beat_the_market(self):
        """Only the index test (or only net R) failed: the reason never claims the
        pattern lost to the equal-weight market or that fees ate its edge."""
        st = {"trades": 120, "net_avg_r": 0.08, "excess_vs_index_pct": -0.2,
              "excess_vs_universe_pct": 0.3, "excess_vs_universe_ci95": [-0.5, 1.1],
              "gross_excess_vs_universe_pct": 0.7, "excess_vs_universe_se": 0.4,
              "excess_vs_universe_df": 35, "excess_vs_index_ci95": [-0.6, 0.2]}
        v, why = bt.pattern_verdict(st, TUNING)
        self.assertEqual(v, "no_edge")
        self.assertNotIn("just not better on average", why)
        self.assertNotIn("fees ate it", why)
        self.assertNotIn("95% range vs the market", why)
        self.assertIn("less than just holding the market index", why)
        self.assertIn("95% range vs the index -0.60% to +0.20% per trade", why)
        self.assertIn("did beat the equal-weight market by +0.30% per trade", why)
        # only net R failed
        v, why = bt.pattern_verdict({**st, "net_avg_r": -0.03, "excess_vs_index_pct": 0.1,
                                     "excess_vs_universe_pct": 0.2,
                                     "net_avg_r_ci95": [-0.1, 0.04]}, TUNING)
        self.assertEqual(v, "no_edge")
        self.assertNotIn("fees ate it", why)
        self.assertNotIn("lost to the market", why)
        self.assertIn("95% range -0.10R to +0.04R per trade", why)
        self.assertIn("did beat the equal-weight market by +0.20%", why)
        # the market test failed as well: the market range and the fee clause are back
        _, why = bt.pattern_verdict({**st, "excess_vs_universe_pct": -0.1}, TUNING)
        self.assertIn("95% range vs the market", why)
        self.assertIn("fees ate it", why)
        self.assertNotIn("did beat", why)

    def test_proven_line_uses_t_with_few_quarters(self):
        """eu/se = 3.5 lies between z (2.991) and t with 9 df (4.069) for 18 tests."""
        st = {"trades": 120, "net_avg_r": 0.2, "excess_vs_index_pct": 1.0,
              "excess_vs_universe_pct": 1.4, "excess_vs_universe_se": 0.4,
              "excess_vs_universe_df": 9, "excess_vs_universe_ci95": [0.5, 2.3]}
        self.assertEqual(bt.pattern_verdict(st, TUNING)[0], "unproven")
        self.assertEqual(bt.pattern_verdict({**st, "excess_vs_universe_df": None}, TUNING)[0],
                         "proven_edge")
        self.assertAlmostEqual(bt._verdict_z(TUNING, 9), 4.069, places=2)

    def test_total_is_judged_on_three_more_tests(self):
        """The all-pattern total counts as n_tests + 3 (one per market)."""
        ts = []
        for q, d in enumerate(quarter_dates(12)):
            for k in range(4):
                ts.append(make_trade(entry_date=d, r_net=0.1, r=0.15,
                                     excess_vs_index_pct=0.2 + 0.1 * ((q + k) % 3),
                                     excess_vs_universe_pct=0.1 * ((q * 7 + k * 3) % 5) - 0.15))
        out = bt.summarize(ts, TUNING)
        total = out["total"]
        self.assertIn(total["verdict"], ("unproven", "proven_edge"))
        self.assertIn("21 pattern x market tests", total["verdict_reason"])
        self.assertIn("18 pattern x market tests", out["per_pattern"]["bull_flag"]["verdict_reason"])

    def test_unproven(self):
        v, why = bt.pattern_verdict({**self.base, "excess_vs_universe_ci95": [-0.2, 1.0]}, TUNING)
        self.assertEqual(v, "unproven")
        self.assertIn("luck", why)
        v, _ = bt.pattern_verdict({**self.base, "excess_vs_universe_ci95": None}, TUNING)
        self.assertEqual(v, "unproven")

    def test_proven_edge(self):
        v, why = bt.pattern_verdict(self.base, TUNING)
        self.assertEqual(v, "proven_edge")
        self.assertIn("18 pattern x market tests", why)

    def test_multiple_testing_adjustment(self):
        st = {**self.base, "excess_vs_universe_pct": 0.4, "excess_vs_universe_ci95": [0.05, 0.75]}
        self.assertEqual(bt.pattern_verdict(st, TUNING)[0], "unproven")
        self.assertEqual(bt.pattern_verdict(st, {**TUNING, "n_tests": 1})[0], "proven_edge")
        self.assertAlmostEqual(bt._verdict_z(TUNING), 2.991, places=3)
        self.assertAlmostEqual(bt._verdict_z({"n_tests": 1}), 1.96, places=2)

    def test_stored_se_takes_precedence(self):
        st = {**self.base, "excess_vs_universe_pct": 0.4, "excess_vs_universe_ci95": [0.05, 0.75],
              "excess_vs_universe_se": 0.05}
        self.assertEqual(bt.pattern_verdict(st, TUNING)[0], "proven_edge")

    def test_large_edge_still_proven(self):
        st = {**self.base, "excess_vs_universe_pct": 2.0, "excess_vs_universe_ci95": [1.5, 2.5]}
        self.assertEqual(bt.pattern_verdict(st, TUNING)[0], "proven_edge")


class DisabledPatternsTest(unittest.TestCase):
    per = {"a": {"trades": 100, "net_avg_r": -0.1, "excess_vs_index_pct": -0.5,
                 "excess_vs_universe_pct": -0.5},
           "b": {"trades": 100, "net_avg_r": 0.2, "excess_vs_index_pct": 0.5,
                 "excess_vs_universe_pct": 0.4, "excess_vs_universe_ci95": [-0.1, 0.9]},
           "c": {"trades": 5, "net_avg_r": -1.0}}

    def test_rule(self):
        self.assertEqual(bt.disabled_patterns(self.per, {}, "us"), {"a"})
        self.assertEqual(bt.disabled_patterns(self.per, {"require_proven_edge": True}, "us"), {"a", "b"})
        self.assertEqual(bt.disabled_patterns(self.per, {"auto_disable": False,
                                                         "disabled": {"us": ["c"]}}, "us"), {"c"})
        v1 = {"x": {"trades": 50, "avg_r": -0.1}, "y": {"trades": 50, "avg_r": 0.1}}
        self.assertEqual(bt.disabled_patterns(v1, {}, "us"), {"x"})


class BaselineTest(unittest.TestCase):
    def setUp(self):
        d = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        self.days = d
        rows = []
        # A trades every day; B has a holiday on 01-02; C lists on 01-03 (mid-window)
        for k, day in enumerate(d):
            rows.append(("A", day, 10 + k, 11 + k, 9 + k, 10.5 + k))
            if day != "2024-01-02":
                rows.append(("B", day, 20 + k, 21 + k, 19 + k, 20.5 + k))
            if day >= "2024-01-03":
                rows.append(("C", day, 30.0, 31.0, 29.0, 60.0))
        self.long = pd.DataFrame(rows, columns=["ticker", "date", "open", "high", "low", "close"])
        self.panel = bt.MarketPanel(self.long)

    def test_universe_with_holiday_and_listing(self):
        got = self.panel.universe_return("2024-01-02", "2024-01-04")
        a = 13.5 / 11 - 1           # A: open 01-02 -> close 01-04
        b = 23.5 / 22 - 1           # B: holiday on 01-02 -> open 01-03 -> close 01-04
        # C listed mid-window: excluded (it did not trade at entry)
        self.assertAlmostEqual(got, (a + b) / 2 * 100)

    def test_universe_exit_on_holiday_uses_last_bar_before(self):
        got = self.panel.universe_return("2024-01-01", "2024-01-02")
        a = 11.5 / 10 - 1           # A: close on 01-02
        b = 20.5 / 20 - 1           # B: last bar on/before 01-02 is 01-01
        self.assertAlmostEqual(got, (a + b) / 2 * 100)

    def test_universe_outside_data(self):
        self.assertIsNone(self.panel.universe_return("2025-01-01", "2025-02-01"))

    def test_index_with_holiday_misalignment(self):
        dates = np.array(["2024-01-01", "2024-01-03", "2024-01-04"])  # index shut on 01-02
        opens, closes = arr(100, 102, 104), arr(101, 103, 105)
        # entry on the index holiday: first index bar on/after = 01-03
        self.assertAlmostEqual(bt.index_return(dates, opens, closes, "2024-01-02", "2024-01-04"),
                               (105 / 102 - 1) * 100)
        # exit on the holiday: last bar on/before = 01-01
        self.assertAlmostEqual(bt.index_return(dates, opens, closes, "2024-01-01", "2024-01-02"),
                               (101 / 100 - 1) * 100)
        # a one-day trade on the holiday has no index bar at all
        self.assertIsNone(bt.index_return(dates, opens, closes, "2024-01-02", "2024-01-02"))

    def test_index_dividend_accrual(self):
        dates = np.array(["2020-01-02", "2021-01-04", "2024-01-02"])   # 1461 days = 4 x 365.25
        opens, closes = arr(100, 105, 120), arr(101, 106, 125)
        got = bt.index_return(dates, opens, closes, "2020-01-02", "2024-01-02", 1.5)
        self.assertAlmostEqual(got, (125 / 100 * 1.015 ** 4 - 1) * 100)
        # y = 0 is the price-only value, bit for bit
        self.assertEqual(bt.index_return(dates, opens, closes, "2020-01-02", "2024-01-02", 0.0),
                         (125 / 100 - 1) * 100)
        # same-day span: no accrual
        self.assertEqual(bt.index_return(dates, opens, closes, "2021-01-04", "2021-01-04", 1.5),
                         (106 / 105 - 1) * 100)

    def test_random_baseline_deterministic(self):
        panel = random_panel()
        a = panel.random_baseline(random_trade(), 10, 40, 0.15, 0.05)
        b = random_panel().random_baseline(random_trade(), 10, 40, 0.15, 0.05)
        self.assertIsNotNone(a[0])
        self.assertEqual(a, b)
        # a different pattern name gives a different (but still fixed) draw
        c = panel.random_baseline({**random_trade(), "pattern": "darvas_box"}, 10, 40, 0.15, 0.05)
        self.assertNotEqual(a, c)

    def test_random_baseline_independent_of_hash_seed(self):
        here = random_panel().random_baseline(random_trade(), 10, 40, 0.15, 0.05)
        code = ("import json; from tests.test_backtest import random_panel, random_trade; "
                "print(json.dumps(random_panel().random_baseline(random_trade(), 10, 40, 0.15, 0.05)))")
        for seed in ("0", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(tuple(json.loads(out.stdout.strip().splitlines()[-1])), here)


class RandomBaselineMechanicsTest(unittest.TestCase):
    """One-ticker panel: every draw is that ticker (the traded stock is elsewhere)."""

    def panel(self, o, h, l, c):
        idx = pd.bdate_range("2024-01-01", periods=len(o))
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
        return bt.MarketPanel(bt.frames_to_long({"X": df})), idx

    def trade(self, idx):
        return {"ticker": "ZZZ", "pattern": "bull_flag", "entry_date": idx[1].strftime("%Y-%m-%d"),
                "entry_price": 200.0, "stop": 190.0, "target": 220.0}

    def test_stop_case_charges_costs_and_slippage(self):
        panel, idx = self.panel([50, 100, 99, 97], [51, 101, 100, 98], [49, 99, 94, 96],
                                [50, 100, 96, 97])
        got = panel.random_baseline(self.trade(idx), 4, 2, 0.15, 0.05)
        want = bt.trade_returns(100, 95, 95, "stop", 0.15, 0.05)
        self.assertAlmostEqual(got[0], want["r_net"])
        self.assertAlmostEqual(got[1], want["ret_net_pct"])

    def test_time_exit_fills_at_the_open(self):
        panel, idx = self.panel([50, 100, 101, 102], [51, 102, 103, 104], [49, 99, 100, 101],
                                [50, 101.5, 102, 103])
        got = panel.random_baseline(self.trade(idx), 3, 2, 0.10, 0.05)
        want = bt.trade_returns(100, 95, 103, "time", 0.10, 0.05)
        self.assertAlmostEqual(got[0], want["r_net"])
        self.assertAlmostEqual(got[1], want["ret_net_pct"])


class RandomPoolTest(unittest.TestCase):
    """Random entries come from liquid stocks with a similar ATR%, never the traded one."""

    def setUp(self):
        n = 80
        idx = pd.bdate_range("2023-01-02", periods=n)

        def flat(price, rng_pct, vol):
            p = np.full(n, float(price))
            return pd.DataFrame({"open": p, "high": p * (1 + rng_pct), "low": p * (1 - rng_pct),
                                 "close": p, "volume": np.full(n, float(vol))}, index=idx)
        frames = {"TRD": flat(100, 0.01, 1e5)}
        for k in range(1, 6):
            frames[f"L{k}"] = flat(100 * (k + 1), 0.01, 1e5)       # liquid, same ATR%
        frames["ILLIQ"] = flat(700, 0.01, 10)                      # below the liquidity floor
        frames["WILD"] = flat(800, 0.02, 1e5)                      # twice the ATR%
        self.price_of = {float(df["open"].iloc[0]): t for t, df in frames.items()}
        self.panel = bt.MarketPanel(bt.frames_to_long(frames), min_dollar_vol=1e6)
        self.idx = idx

    def drawn(self, k, days=range(25, 60), patterns=("a", "b", "c")):
        seen = []
        real = bt.simulate_exit

        def spy(o, h, l, c, fi, fill, *a):
            seen.append(self.price_of[round(fill, 6)])
            return real(o, h, l, c, fi, fill, *a)
        with mock.patch.object(bt, "simulate_exit", spy):
            for d in days:
                for p in patterns:
                    t = {"ticker": "TRD", "pattern": p, "entry_date": self.idx[d].strftime("%Y-%m-%d"),
                         "entry_price": 100.0, "stop": 95.0, "target": 110.0}
                    self.assertIsNotNone(self.panel.random_baseline(t, k, 10, 0.1, 0.05)[0])
        return set(seen)

    def test_illiquid_wild_and_traded_stocks_never_drawn(self):
        got = self.drawn(5)
        self.assertEqual(got, {"L1", "L2", "L3", "L4", "L5"})
        self.assertEqual(self.panel.pool_counts["random_pool_fallback"], 0)

    def test_fallback_when_the_band_is_too_small(self):
        got = self.drawn(6, days=[30], patterns=("a",))
        self.assertEqual(self.panel.pool_counts["random_pool_fallback"], 1)
        self.assertNotIn("ILLIQ", got)
        self.assertNotIn("TRD", got)


class AddBaselinesWiringTest(unittest.TestCase):
    def test_values_written_to_the_trade(self):
        base = BaselineTest()
        base.setUp()
        idx = pd.DatetimeIndex(base.days)
        bench = bt.BenchContext.from_df(pd.DataFrame(
            {"open": [100.0, 101, 102, 103, 104], "high": 106.0, "low": 99.0,
             "close": [100.5, 101.5, 102.5, 103.5, 104.5], "volume": 1e9}, index=idx))
        s = bt.Settings(random_entries_per_trade=0)
        t = {"ticker": "A", "pattern": "bull_flag", "outcome": "time", "entry_date": "2024-01-02",
             "exit_date": "2024-01-04", "entry_price": 11.0, "stop": 10.0, "target": 13.0,
             "ret_pct": 5.0, "ret_net_pct": 4.0, "r_net": 0.4}
        self.assertEqual(bt.add_baselines([t], base.panel, bench, s, "us"), 0)
        idx_price = (103.5 / 101 - 1) * 100
        idx_div = ((103.5 / 101) * 1.015 ** (2 / 365.25) - 1) * 100
        uni = ((13.5 / 11 - 1) + (23.5 / 22 - 1)) / 2 * 100
        self.assertAlmostEqual(t["index_ret_price_pct"], idx_price, places=3)
        self.assertAlmostEqual(t["index_ret_pct"], idx_div, places=3)
        self.assertAlmostEqual(t["universe_ret_pct"], uni, places=3)
        self.assertAlmostEqual(t["excess_vs_index_pct"], 4.0 - idx_div, places=3)
        self.assertAlmostEqual(t["excess_vs_universe_pct"], 4.0 - uni, places=3)
        self.assertAlmostEqual(t["excess_vs_universe_r"], 0.4 - uni / (1 / 11 * 100), places=4)
        self.assertIsNone(t["rand_r_net"])
        # DE: the DAX already includes dividends
        t2 = dict(t)
        bt.add_baselines([t2], base.panel, bench, s, "de")
        self.assertAlmostEqual(t2["index_ret_pct"], idx_price, places=3)

    def test_r_scaled_excess(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        df = pd.DataFrame({"open": [100.0, 100.0, 101.0], "high": 103.0, "low": 99.0,
                           "close": [100.0, 101.0, 102.0]}, index=idx)
        panel = bt.MarketPanel(bt.frames_to_long({"U": df}))
        bench = bt.BenchContext.from_df(df.assign(volume=1e9))
        t = {"ticker": "X", "pattern": "p", "outcome": "time", "entry_date": "2024-01-02",
             "exit_date": "2024-01-03", "entry_price": 100.0, "stop": 95.0, "target": 110.0,
             "ret_pct": 3.0, "ret_net_pct": 3.0, "r_net": 0.6}
        bt.add_baselines([t], panel, bench, bt.Settings(random_entries_per_trade=0), "de")
        self.assertAlmostEqual(t["universe_ret_pct"], 2.0)
        self.assertAlmostEqual(t["excess_vs_universe_r"], 0.6 - 0.4)


class ScannerFilterTest(unittest.TestCase):
    """The scanner-mode filters remove signals before any pattern is traded."""

    def setUp(self):
        self.stock = bt.add_indicators(random_frame(11, n=700, drift=0.0007, vol=0.02))
        bench = random_frame(1, n=700, price=3000, drift=0.0004, vol=0.009)
        self.bench = bt.BenchContext.from_df(bench)
        self.s = bt.Settings()

    def test_downtrend_regime_blocks_every_trade(self):
        self.bench.statuses = ["downtrend"] * len(self.bench.statuses)
        counters = bt.new_counters()
        trades = bt.simulate_scanner(self.stock, "SYN", "us", self.bench, self.s, counters)
        self.assertEqual(trades, [])
        self.assertGreater(counters["removed"]["regime_downtrend"], 0)
        self.assertEqual(counters["no_pattern"], 0)  # detectors never even ran

    def test_liquidity_floor(self):
        thin = self.stock.copy()
        thin["dollar_vol20"] = 1.0
        counters = bt.new_counters()
        trades = bt.simulate_scanner(thin, "SYN", "us", self.bench, self.s, counters)
        self.assertEqual(trades, [])
        self.assertEqual(counters["removed"]["liquidity"], counters["signals_evaluated"])

    def test_regime_filter_off_records_off(self):
        s = bt.Settings(regime_filter=False)
        self.bench.statuses = ["downtrend"] * len(self.bench.statuses)
        counters = bt.new_counters()
        trades = bt.simulate_scanner(self.stock, "SYN", "us", self.bench, s, counters)
        self.assertEqual(counters["removed"]["regime_downtrend"], 0)
        self.assertTrue(all(t["regime"] == "off" for t in trades))


class RegimeTest(unittest.TestCase):
    def test_matches_scan_market_regime(self):
        bench = random_frame(7, n=420, drift=0.0002, vol=0.012)
        statuses = bt.regime_statuses(bench)
        for p in (100, 208, 209, 250, 300, 350, 419):
            self.assertEqual(statuses[p], scan.market_regime(bench.iloc[: p + 1])["status"],
                             f"bar {p}")


def _write_tuning_file(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


class PatternTuningTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cfg = {"pattern_tuning": {"disabled": {"us": ["cup_and_handle"]}}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_file(self):
        disabled, stats, source = scan.load_pattern_tuning("us", self.cfg,
                                                           bt_path=self.dir / "missing.json")
        self.assertEqual(disabled, {"cup_and_handle"})
        self.assertEqual(stats, {})
        self.assertEqual(source, "no backtest available")

    def test_corrupt_file(self):
        p = self.dir / "bt.json"
        p.write_text("{not json")
        disabled, stats, _ = scan.load_pattern_tuning("us", self.cfg, bt_path=p)
        self.assertEqual(disabled, {"cup_and_handle"})
        self.assertEqual(stats, {})

    def test_v1_file_uses_old_rule(self):
        p = _write_tuning_file(self.dir / "bt.json", {
            "generated": "2026-08-18T20:29:00", "available": True,
            "markets": {"us": {"per_pattern": {
                "ascending_triangle": {"trades": 59, "win_rate": 40.0, "avg_r": -0.18},
                "bull_flag": {"trades": 500, "win_rate": 50.0, "avg_r": 0.1},
                "darvas_box": {"trades": 10, "win_rate": 20.0, "avg_r": -0.5},
            }}}})
        disabled, stats, source = scan.load_pattern_tuning("us", self.cfg, bt_path=p)
        self.assertEqual(disabled, {"cup_and_handle", "ascending_triangle"})
        self.assertEqual(source, "backtest 2026-08-18")
        self.assertNotIn("verdict", stats["bull_flag"])
        self.assertIn("avg R over 500 trades", scan.backtest_note("us", stats["bull_flag"]))

    def _v2(self) -> Path:
        common = {"excess_vs_index_pct": 0.5, "excess_vs_universe_pct": 0.4, "win_rate": 52.0,
                  "avg_r": 0.25}
        return _write_tuning_file(self.dir / "bt2.json", {
            "version": 2, "generated": "2026-09-25T12:00:00", "available": True,
            "method": {"period": {"us": {"first_entry": "2017-03-01", "last_exit": "2026-09-01"}}},
            "markets": {"us": {"per_pattern": {
                "ascending_triangle": {**common, "trades": 400, "net_avg_r": 0.03,
                                       "excess_vs_index_pct": -0.4,
                                       "excess_vs_universe_ci95": [-1.0, 0.2],
                                       "verdict": "whatever-was-stored"},
                "bull_flag": {**common, "trades": 300, "net_avg_r": 0.2,
                              "excess_vs_universe_ci95": [-0.1, 0.9]},
                "double_bottom": {**common, "trades": 200, "net_avg_r": 0.3,
                                  "excess_vs_universe_pct": 0.6,
                                  "excess_vs_universe_ci95": [0.3, 0.9]},
                "darvas_box": {**common, "trades": 12, "net_avg_r": -1.0,
                               "excess_vs_universe_ci95": None},
            }}}})

    def test_v2_file_uses_verdicts(self):
        p = self._v2()
        disabled, stats, source = scan.load_pattern_tuning("us", self.cfg, bt_path=p)
        self.assertEqual(disabled, {"cup_and_handle", "ascending_triangle"})
        self.assertEqual(source, "backtest 2026-09-25 (after fees)")
        self.assertEqual({k: v["verdict"] for k, v in stats.items()}, {
            "ascending_triangle": "no_edge", "bull_flag": "unproven",
            "double_bottom": "proven_edge", "darvas_box": "insufficient_data"})
        note = scan.backtest_note("us", stats["ascending_triangle"])
        self.assertIn("(10y, after fees)", note)
        self.assertIn("-0.4% per trade vs the index", note)
        self.assertIn("+0.4% vs the equal-weight market", note)
        self.assertTrue(note.endswith("no edge"))
        note = scan.backtest_note("us", stats["ascending_triangle"], max_hold=40)
        self.assertIn("(10y, after fees, stop/target/40-day exit)", note)

    def test_v2_require_proven_edge_and_current_config(self):
        p = self._v2()
        cfg = {"pattern_tuning": {"require_proven_edge": True, "min_trades": 10}}
        disabled, stats, _ = scan.load_pattern_tuning("us", cfg, bt_path=p)
        # darvas_box now has enough trades (current config) and loses money
        self.assertEqual(disabled, {"ascending_triangle", "bull_flag", "darvas_box"})
        cfg_off = {"pattern_tuning": {"auto_disable": False}}
        disabled, _, _ = scan.load_pattern_tuning("us", cfg_off, bt_path=p)
        self.assertEqual(disabled, set())

    def test_info_splits_manual_and_auto_and_reads_max_hold(self):
        p = self._v2()
        info = scan.load_pattern_tuning_info("us", self.cfg, bt_path=p)
        self.assertEqual(info["manual"], {"cup_and_handle"})
        self.assertEqual(info["auto"], {"ascending_triangle"})
        self.assertEqual(info["disabled"], {"cup_and_handle", "ascending_triangle"})
        self.assertIsNone(info["backtest_max_hold"])                 # the file does not say
        d = json.loads(p.read_text())
        d["method"]["settings"] = {"max_hold": 40}
        p.write_text(json.dumps(d))
        self.assertEqual(scan.load_pattern_tuning_info("us", {}, bt_path=p)["backtest_max_hold"], 40)
        d["method"]["settings_by_market"] = {"us": {"max_hold": 25}}
        p.write_text(json.dumps(d))
        self.assertEqual(scan.load_pattern_tuning_info("us", {}, bt_path=p)["backtest_max_hold"], 25)
        self.assertIsNone(scan.load_pattern_tuning_info("de", {}, bt_path=p)["backtest_max_hold"])

    def test_market_missing_from_the_file(self):
        p = self._v2()
        disabled, stats, source = scan.load_pattern_tuning("de", {}, bt_path=p)
        self.assertTrue(source.startswith("no backtest for DE"), source)
        self.assertEqual(disabled, set())
        self.assertEqual(stats, {})


class BacktestPayloadTest(unittest.TestCase):
    """/api/backtest: verdicts and disabled lists as the scanner applies them now."""

    setUp = PatternTuningTest.setUp
    tearDown = PatternTuningTest.tearDown
    _v2 = PatternTuningTest._v2

    def test_payload_uses_the_current_config(self):
        from atlas2 import server
        p = self._v2()
        d = json.loads(p.read_text())
        d["markets"]["us"]["per_pattern"]["bull_flag"]["verdict"] = "proven_edge"
        d["markets"]["us"]["total"] = {"trades": 900, "net_avg_r": 0.1, "win_rate": 50.0,
                                       "excess_vs_index_pct": 0.3, "excess_vs_universe_pct": 0.2,
                                       "excess_vs_universe_ci95": [-0.1, 0.5], "verdict": "stale"}
        d["markets"]["us"]["live_config"] = {"disabled": ["darvas_box"]}
        p.write_text(json.dumps(d))
        before = p.read_text()
        cfg = {"dashboard_password": "secret-pw", "capital_eur": 1,
               "pattern_tuning": {"require_proven_edge": True, "disabled": {"us": ["cup_and_handle"]}}}
        out = server._backtest_payload(p, cfg)
        us = out["markets"]["us"]
        bf = us["per_pattern"]["bull_flag"]
        self.assertEqual((bf["verdict"], bf["verdict_at_backtest"]), ("unproven", "proven_edge"))
        self.assertEqual(us["per_pattern"]["ascending_triangle"]["verdict"], "no_edge")
        self.assertEqual(us["total"]["verdict_at_backtest"], "stale")
        tuning = scan._deep_merge(scan.PATTERN_TUNING_DEFAULTS, cfg["pattern_tuning"])
        self.assertEqual(us["total"]["verdict"], bt.pattern_verdict(
            d["markets"]["us"]["total"], {**tuning, "n_tests": bt._n_tests(tuning) + 3})[0])
        # manual list + auto-disable (with require_proven_edge), not the backtest-time list
        self.assertEqual(us["currently_disabled"],
                         ["ascending_triangle", "bull_flag", "cup_and_handle"])
        self.assertEqual(out["live_tuning"], {"applied": True, "auto_disable": True,
                                              "require_proven_edge": True, "min_trades": 30,
                                              "min_avg_r": 0.0, "family_alpha": 0.05,
                                              "n_tests": 18})
        self.assertNotIn("secret-pw", json.dumps(out))
        self.assertEqual(p.read_text(), before)                         # read-only
        self.assertEqual(server._backtest_payload(self.dir / "none.json", cfg), {"available": False})

    def test_zero_trade_reason_survives_the_recompute(self):
        from atlas2 import server
        p = self._v2()
        d = json.loads(p.read_text())
        for st in d["markets"]["us"]["per_pattern"].values():      # as a backtest stores them
            st.update(verdict="stale", verdict_reason="stale")
        why = ("Never produced a trade: its best risk:reward was 0.95 (its target is too close "
               "compared with its stop), so the risk:reward minimum (1.8) filters out every signal")
        d["markets"]["us"]["per_pattern"]["cup_and_handle"] = {
            "trades": 0, "net_avg_r": None, "excess_vs_index_pct": None,
            "excess_vs_universe_pct": None, "verdict": "insufficient_data", "verdict_reason": why}
        p.write_text(json.dumps(d))
        for cfg in ({}, {"pattern_tuning": {"min_trades": 5}}):
            ch = server._backtest_payload(p, cfg)["markets"]["us"]["per_pattern"]["cup_and_handle"]
            self.assertEqual((ch["verdict"], ch["verdict_reason"]), ("insufficient_data", why))
            info = scan.load_pattern_tuning_info("us", cfg, bt_path=p)
            self.assertEqual(info["stats"]["cup_and_handle"]["verdict_reason"], why)
        # a pattern with a few trades still gets the current-config reason
        dv = server._backtest_payload(p, {})["markets"]["us"]["per_pattern"]["darvas_box"]
        self.assertIn("Only 12 trades", dv["verdict_reason"])

    def test_note_and_test_count_follow_the_current_config(self):
        from atlas2 import server
        p = self._v2()
        d = json.loads(p.read_text())
        for st in d["markets"]["us"]["per_pattern"].values():      # as a backtest stores them
            st.update(verdict="stale", verdict_reason="stale")
        d["note"] = "US: 4 stale verdicts."
        d["method"]["verdicts"] = {"n_tests": 18, "family_alpha": 0.05}
        # eu / se = 3.015: proven at 18 + 3 = 21 tests' z (3.038)? no; at 18 tests (2.991) yes
        total = {"trades": 900, "net_avg_r": 0.1, "excess_vs_index_pct": 0.3,
                 "excess_vs_universe_pct": 0.3, "excess_vs_universe_se": 0.0995,
                 "excess_vs_universe_ci95": [0.1, 0.5], "verdict": "stale"}
        d["markets"]["us"]["total"] = total
        p.write_text(json.dumps(d))
        tuning = scan._deep_merge(scan.PATTERN_TUNING_DEFAULTS, {})
        self.assertEqual(bt.pattern_verdict(total, tuning)[0], "proven_edge")          # 18
        out = server._backtest_payload(p, {})
        self.assertEqual(out["markets"]["us"]["total"]["verdict"], "unproven")         # 21
        self.assertIn("21 pattern x market tests", out["markets"]["us"]["total"]["verdict_reason"])
        # n_tests 1 and min_trades 20: the headline note and n_tests follow
        out = server._backtest_payload(p, {"pattern_tuning": {"n_tests": 1, "min_trades": 20}})
        per = out["markets"]["us"]["per_pattern"]
        counts: dict = {}
        for st in per.values():
            counts[st["verdict"]] = counts.get(st["verdict"], 0) + 1
        self.assertEqual(out["live_tuning"]["n_tests"], 1)
        self.assertEqual(out["note_at_backtest"], "US: 4 stale verdicts.")
        for v, n in counts.items():
            self.assertIn(f"{n} {bt.VERDICT_LABELS[v]}", out["note"])
        self.assertNotIn("stale", out["note"])
        self.assertIn("once 1 pattern x market tests", per["bull_flag"]["verdict_reason"])

    def test_v1_file_keeps_its_numbers(self):
        from atlas2 import server
        p = _write_tuning_file(self.dir / "v1.json", {
            "generated": "2026-08-18T20:29:00", "available": True,
            "markets": {"us": {"per_pattern": {"bull_flag": {"trades": 500, "avg_r": -0.1,
                                                             "win_rate": 40.0}}}}})
        out = server._backtest_payload(p, {})
        self.assertEqual(out["markets"]["us"]["currently_disabled"], ["bull_flag"])
        self.assertNotIn("verdict", out["markets"]["us"]["per_pattern"]["bull_flag"])
        self.assertTrue(out["live_tuning"]["applied"])


class ScanHelpersTest(unittest.TestCase):
    def test_note_never_prints_negative_zero(self):
        st = {"trades": 100, "win_rate": 45.0, "net_avg_r": -0.004, "excess_vs_index_pct": -0.04,
              "excess_vs_universe_pct": -0.02, "verdict": "no_edge"}
        note = scan.backtest_note("us", st)
        self.assertIn("+0.00R", note)
        self.assertNotIn("-0.0", note)

    def test_score_nudge(self):
        self.assertEqual(scan.backtest_score_nudge({"trades": 100, "net_avg_r": -0.1, "avg_r": 0.3}, 30),
                         -2.0)
        self.assertEqual(scan.backtest_score_nudge({"trades": 100, "avg_r": 0.3}, 30), 6.0)
        self.assertEqual(scan.backtest_score_nudge({"trades": 100, "net_avg_r": 1.0}, 30), 10.0)
        self.assertEqual(scan.backtest_score_nudge({"trades": 100, "net_avg_r": -1.0}, 30), -10.0)
        self.assertEqual(scan.backtest_score_nudge({"trades": 20, "net_avg_r": 0.4}, 30), 0.0)
        self.assertEqual(scan.backtest_score_nudge(None, 30), 0.0)

    def test_nudge_and_note_share_the_min_trades_gate(self):
        st = {"trades": 40, "win_rate": 50.0, "net_avg_r": 0.2, "excess_vs_index_pct": 0.3,
              "excess_vs_universe_pct": 0.1, "verdict": "insufficient_data"}
        self.assertEqual(scan.backtest_nudge("us", st, 50), (0.0, None))
        st = {**st, "trades": 20, "verdict": "unproven"}
        delta, note = scan.backtest_nudge("us", st, 10)
        self.assertNotEqual(delta, 0.0)
        self.assertTrue(note.endswith("edge not proven"))

    def test_tuning_summary(self):
        names = scan.bullish_pattern_names()
        bt_stats = {p: {"trades": 100, "verdict_reason": f"why {p}"} for p in names}
        bt_stats["double_bottom"] = {"trades": 0}
        disabled = {p for p in names if p != "double_bottom"}
        out = scan.tuning_summary(disabled, bt_stats, 30)
        self.assertEqual(out["enabled"], ["double_bottom"])
        self.assertEqual(out["untested"], ["double_bottom"])
        self.assertTrue(out["all_judged_disabled"])
        self.assertEqual(out["reasons"]["bull_flag"], "why bull_flag")
        out = scan.tuning_summary(set(), bt_stats, 30)
        self.assertFalse(out["all_judged_disabled"])
        # one judged pattern still enabled: not everything is switched off
        judged = {p: {"trades": 100} for p in names}
        out = scan.tuning_summary({"bull_flag"}, judged, 30)
        self.assertFalse(out["all_judged_disabled"])
        self.assertEqual(out["auto"], ["bull_flag"])

    def test_manual_disables_are_never_blamed_on_the_backtest(self):
        out = scan.tuning_summary({"bull_flag"}, {}, 30)             # no backtest at all
        self.assertFalse(out["all_judged_disabled"])
        names = scan.bullish_pattern_names()
        bt_stats = {p: {"trades": 100, "verdict_reason": f"why {p}"} for p in names}
        bt_stats["double_bottom"] = {"trades": 0}
        disabled = {p for p in names if p != "double_bottom"}
        out = scan.tuning_summary(disabled, bt_stats, 30, manual=disabled)
        self.assertFalse(out["all_judged_disabled"])                # all manual: not the backtest
        self.assertEqual(out["manual"], sorted(disabled))
        self.assertEqual((out["auto"], out["reasons"]), ([], {}))
        out = scan.tuning_summary(disabled, bt_stats, 30, manual={"bull_flag"})
        self.assertTrue(out["all_judged_disabled"])
        self.assertEqual(out["manual"], ["bull_flag"])
        self.assertNotIn("bull_flag", out["reasons"])
        self.assertNotIn("bull_flag", out["auto"])


class ScanMarketIntegrationTest(unittest.TestCase):
    """scan_market end to end on synthetic prices with every network call patched:
    the backtest's disabled patterns never reach the setups, the nudge / note use
    pattern_tuning.min_trades, and notes quote the backtest's own time exit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        store = PriceStore(self.root / "data" / "cache" / "prices.sqlite3")
        frames = {"^GSPC": random_frame(1, n=400, price=3000, drift=0.0006, vol=0.006),
                  "SYN0": random_frame(2, n=300), "SYN1": random_frame(3, n=310)}
        rows = [(t, d.strftime("%Y-%m-%d"), float(r.open), float(r.high), float(r.low),
                 float(r.close), float(r.volume)) for t, df in frames.items() for d, r in df.iterrows()]
        store.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows)
        store.conn.commit()
        store.close()
        (self.root / "universe").mkdir()
        pd.DataFrame({"ticker": ["SYN0", "SYN1"], "name": ["Zero", "One"],
                      "index": "TEST"}).to_csv(self.root / "universe" / "us.csv", index=False)
        common = {"win_rate": 50.0, "avg_r": 0.3, "excess_vs_index_pct": 0.5,
                  "excess_vs_universe_pct": 0.5, "excess_vs_universe_ci95": [0.2, 0.8]}
        (self.root / "data" / "backtest_latest.json").write_text(json.dumps({
            "version": 2, "generated": "2026-09-25T12:00:00", "available": True,
            "method": {"settings": {"max_hold": 40},
                       "period": {"us": {"first_entry": "2017-03-01", "last_exit": "2026-09-01"}}},
            "markets": {"us": {"per_pattern": {
                "bull_flag": {**common, "trades": 400, "net_avg_r": -0.1,
                              "excess_vs_index_pct": -0.5},                    # no edge
                "double_bottom": {**common, "trades": 40, "net_avg_r": 0.3},    # 30 <= 40 < 50
                "ascending_triangle": {**common, "trades": 120, "net_avg_r": 0.2},
            }}}}))

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def fake_detect(df):
        def h(pattern, conf):
            return PatternHit(pattern, "bullish", 100.0, 95.0, 112.5, conf, "forming",
                              len(df) - 30, len(df) - 1, [(len(df) - 30, 97.0)], "fake")
        if len(df) == 300:                              # SYN0: the disabled pattern is the best
            return [h("bull_flag", 0.9), h("double_bottom", 0.6)]
        return [h("ascending_triangle", 0.7)]           # SYN1

    def scan(self, cfg, msgs=None):
        with mock.patch.object(scan, "ROOT", self.root), \
                mock.patch.object(scan, "refresh_prices"), \
                mock.patch.object(scan, "_fx_to_eur", return_value=1.1), \
                mock.patch.object(scan, "detect_all", side_effect=self.fake_detect), \
                mock.patch.object(scan, "technical_score", return_value=(80.0, [])), \
                mock.patch.object(scan, "fetch_fundamentals", return_value={}), \
                mock.patch.object(scan, "score_fundamentals", return_value=(None, {})), \
                mock.patch.object(scan, "fetch_next_earnings", return_value=None), \
                mock.patch.object(scan, "fetch_dividends", return_value=None), \
                mock.patch.object(scan, "dividend_summary", return_value=None):
            return scan.scan_market("us", cfg, progress=(msgs.append if msgs is not None
                                                         else lambda *_: None))

    def test_scan_market_applies_the_tuning(self):
        cfg = scan._deep_merge(scan.load_config(self.root / "no-config.json"),
                               {"pattern_tuning": {"min_trades": 50,
                                                   "disabled": {"us": ["cup_and_handle"]}},
                                "backtest": {"max_hold": 30}})
        res = self.scan(cfg)
        by = {s_["ticker"]: s_ for s_ in res["setups"]}
        self.assertEqual(set(by), {"SYN0", "SYN1"})
        self.assertNotIn("bull_flag", {s_["pattern"] for s_ in res["setups"]})
        self.assertEqual(by["SYN0"]["pattern"], "double_bottom")      # runner-up, as the backtest
        self.assertEqual(by["SYN0"]["other_patterns"], [])
        # double_bottom: 40 trades < min_trades 50 -> no nudge, no note
        self.assertEqual(by["SYN0"]["pattern_score"], 62.5)            # 0.6 * 100 + (2.5 - 2) * 5
        self.assertFalse(any("backtest" in r for r in by["SYN0"]["reasons"]))
        # ascending_triangle: judged; its note quotes the backtest's 40-day exit, not config 30
        note = [r for r in by["SYN1"]["reasons"] if "backtest" in r]
        self.assertEqual(len(note), 1)
        self.assertIn("stop/target/40-day exit", note[0])
        self.assertEqual(by["SYN1"]["pattern_score"], round(70 + 2.5 + 0.2 * 20, 1))
        for s_ in res["setups"]:
            self.assertIn("max_buy", s_)
            self.assertAlmostEqual(s_["max_buy"], round(100.0 * 1.03, 4))
        self.assertEqual(res["exit_rule"]["max_hold_days"], 30)
        self.assertEqual(res["exit_rule"]["backtest_max_hold_days"], 40)
        self.assertIn("re-run the backtest", res["exit_rule"]["note"])
        pt = res["pattern_tuning"]
        self.assertEqual(pt["disabled"], ["bull_flag", "cup_and_handle"])
        # the manual entry is never blamed on the backtest
        self.assertEqual((pt["auto"], pt["manual"]), (["bull_flag"], ["cup_and_handle"]))
        self.assertEqual(pt["verdicts"]["bull_flag"], "no_edge")
        # untested uses pattern_tuning.min_trades (50): double_bottom's 40 trades are too few
        self.assertIn("double_bottom", pt["untested"])
        self.assertNotIn("ascending_triangle", pt["untested"])
        self.assertNotIn("cup_and_handle", pt["enabled"])
        self.assertFalse(pt["all_judged_disabled"])
        self.assertTrue(pt["has_backtest"])
        self.assertIn("re-run the backtest", pt["note"])
        self.assertTrue((self.root / "data" / "scans" / "us_latest.json").exists())

    def test_every_judged_pattern_switched_off_and_no_backtest(self):
        cfg = scan._deep_merge(scan.load_config(self.root / "no-config.json"),
                               {"pattern_tuning": {"min_trades": 30, "min_avg_r": 0.35}})
        pt = self.scan(cfg)["pattern_tuning"]
        # min_avg_r 0.35: all three judged patterns are no edge
        self.assertEqual(pt["auto"], ["ascending_triangle", "bull_flag", "double_bottom"])
        self.assertTrue(pt["all_judged_disabled"])
        (self.root / "data" / "backtest_latest.json").unlink()
        msgs: list[str] = []
        pt = self.scan(cfg, msgs)["pattern_tuning"]
        self.assertFalse(pt["has_backtest"])
        self.assertEqual((pt["auto"], pt["disabled"]), ([], []))
        self.assertFalse(pt["all_judged_disabled"])
        self.assertTrue(any("no backtest for this market - nothing is tuned" in m for m in msgs))


class ConfigTest(unittest.TestCase):
    def test_partial_user_config_keeps_nested_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps({"backtest": {"costs_pct_per_side": {"us": 0.3}},
                                     "pattern_tuning": {"min_trades": 50}}))
            cfg = scan.load_config(p)
        self.assertEqual(cfg["backtest"]["costs_pct_per_side"], {"us": 0.3, "de": 0.10, "in": 0.20})
        self.assertEqual(cfg["backtest"]["max_hold"], 40)
        self.assertEqual(cfg["pattern_tuning"]["min_trades"], 50)
        self.assertFalse(cfg["pattern_tuning"]["require_proven_edge"])
        self.assertEqual(cfg["pattern_tuning"]["disabled"]["de"], [])
        s = bt.Settings.from_config(cfg)
        self.assertEqual(s.cost_pct("us"), 0.3)
        self.assertEqual(s.cost_pct("in"), 0.2)
        # the module-level defaults were not mutated
        self.assertEqual(scan.BACKTEST_DEFAULTS["costs_pct_per_side"]["us"], 0.15)

    def test_missing_config_gives_defaults(self):
        cfg = scan.load_config(Path(tempfile.gettempdir()) / "definitely-missing-config.json")
        self.assertEqual(cfg["backtest"], scan.BACKTEST_DEFAULTS)
        self.assertEqual(cfg["pattern_tuning"], scan.PATTERN_TUNING_DEFAULTS)

    def test_daily_replay_defaults(self):
        s = bt.Settings.from_config({})
        self.assertEqual((s.step, s.breakout_wait, s.max_hold, s.window), (1, 1, 40, 170))
        self.assertTrue(s.seen_on_fill_only)
        self.assertEqual(s.entry_mode, "buy_stop")
        self.assertEqual(len(s.price_artifacts), 3)

    def test_slippage_float_or_dict(self):
        a = bt.Settings.from_config({"backtest": {"stop_slippage_pct": 0.1, "entry_slippage_pct": 0.2}})
        b = bt.Settings.from_config({"backtest": {
            "stop_slippage_pct": {"us": 0.1, "de": 0.1, "in": 0.1},
            "entry_slippage_pct": {"us": 0.2, "de": 0.2, "in": 0.2}}})
        for m in ("us", "de", "in"):
            self.assertEqual(a.stop_slip(m), b.stop_slip(m))
            self.assertEqual(a.entry_slip(m), b.entry_slip(m))
        c = bt.Settings.from_config({"backtest": {"stop_slippage_pct": {"de": 0.3}}})
        self.assertEqual((c.stop_slip("us"), c.stop_slip("de"), c.stop_slip("in")), (0.05, 0.3, 0.15))

    def test_dividend_yield_defaults_and_partial_config(self):
        s = bt.Settings()
        self.assertEqual(s.div_yield_pct("de"), 0.0)
        self.assertEqual(s.div_yield_pct("us"), 1.5)
        s = bt.Settings.from_config({"backtest": {"index_dividend_yield_pct": {"us": 2.0}}})
        self.assertEqual((s.div_yield_pct("us"), s.div_yield_pct("de"), s.div_yield_pct("in")),
                         (2.0, 0.0, 1.3))

    def test_bad_entry_mode_rejected(self):
        with self.assertRaises(ValueError):
            bt.Settings(entry_mode="market")


def _build_synthetic_market(root: Path, n_tickers: int = 5,
                            markets: tuple[str, ...] = ("us",)) -> tuple[Path, Path, Path]:
    """Temp price DB + universe/<market>.csv + data dir with a few synthetic stocks."""
    db = root / "prices.sqlite3"
    store = PriceStore(db)
    frames = {}
    for mi, m in enumerate(markets):
        frames[BENCHMARKS[m]] = random_frame(1 + mi, n=700, price=3000, drift=0.0004, vol=0.009)
        for k in range(n_tickers):
            frames[f"{m.upper()}{k}" if m != "us" else f"SYN{k}"] = random_frame(
                10 + 20 * mi + k, n=700, drift=0.0007, vol=0.02,
                volume=2e6 if m != "in" else 2e7)
    rows = [(t, d.strftime("%Y-%m-%d"), float(r.open), float(r.high), float(r.low),
             float(r.close), float(r.volume))
            for t, df in frames.items() for d, r in df.iterrows()]
    store.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows)
    store.conn.commit()
    store.close()
    uni = root / "universe"
    uni.mkdir()
    for m in markets:
        names = [f"{m.upper()}{k}" if m != "us" else f"SYN{k}" for k in range(n_tickers)]
        pd.DataFrame({"ticker": names, "name": [f"Synthetic {t}" for t in names],
                      "index": "TEST"}).to_csv(uni / f"{m}.csv", index=False)
    data = root / "data"
    data.mkdir()
    return db, uni, data


def _cfg(**over) -> dict:
    base = scan._deep_merge({"min_risk_reward": 1.8, "regime_filter": True, "max_position_pct": 25.0,
                             "shortlist_size": 45,
                             "pattern_tuning": scan.PATTERN_TUNING_DEFAULTS,
                             "backtest": scan.BACKTEST_DEFAULTS},
                            {"backtest": {"random_entries_per_trade": 3}})
    return scan._deep_merge(base, over)


def _real_data_mtimes() -> dict:
    d = REPO / "data"
    return {p.name: p.stat().st_mtime for p in d.glob("*.json")} if d.exists() else {}


class RunBacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.db, cls.uni, _ = _build_synthetic_market(cls.root)
        cls.cfg = _cfg()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, name: str, workers: int, limit: int | None = None, cfg=None, **kw) -> Path:
        out = self.root / name
        out.mkdir()
        bt.run_backtest(["us"], progress=lambda *_: None, raw=kw.pop("raw", True), limit=limit,
                        workers=workers, portfolio=kw.pop("portfolio", False),
                        cfg=cfg or self.cfg, data_dir=out, db_path=self.db,
                        universe_dir=self.uni, **kw)
        return out

    def test_workers_identical_and_contract(self):
        before = _real_data_mtimes()
        one = self._run("w1", workers=1)
        two = self._run("w2", workers=2)
        for f in ("backtest_trades.json", "backtest_trades_raw.json", "backtest_trades_live.json",
                  "backtest_trades_walkforward.json", "backtest_trades_grade_a.json",
                  "backtest_trades_a_in_caution.json"):
            self.assertEqual((one / f).read_text(), (two / f).read_text(), f)
        a = json.loads((one / "backtest_latest.json").read_text())
        b = json.loads((two / "backtest_latest.json").read_text())
        self.assertEqual(a["markets"], b["markets"])
        # output contract
        self.assertEqual(a["version"], 2)
        self.assertTrue(a["available"])
        for key in ("method", "caveats", "note", "markets", "generated"):
            self.assertIn(key, a)
        us = a["markets"]["us"]
        for key in ("per_pattern", "total", "filters", "raw_total", "live_config"):
            self.assertIn(key, us)
        f = us["filters"]
        for key in ("signals_evaluated", "removed", "detector_exceptions", "censored_at_end",
                    "baseline_missing", "orders_not_filled_today", "price_artifacts"):
            self.assertIn(key, f)
        for key in ("no_bar_on_date", "not_shortlisted", "already_traded"):
            self.assertIn(key, f["removed"])
        # the funnel adds up
        self.assertEqual(f["signals_evaluated"],
                         sum(f["removed"].values()) + f["no_pattern"] + f["orders_not_filled_today"]
                         + f["censored_at_end"] + f["trades"])
        self.assertEqual(set(us["per_pattern"]), set(bt.detector_names()))
        for key in ("family_alpha", "n_tests", "z"):
            self.assertIn(key, a["method"]["verdicts"])
        self.assertIn("data_quality", a["method"])
        self.assertEqual(a["method"]["settings"]["shortlist_size"], 45)
        raw_trades = json.loads((one / "backtest_trades_raw.json").read_text())
        self.assertGreater(len(raw_trades), 0, "synthetic data should produce raw trades")
        trades = json.loads((one / "backtest_trades.json").read_text())
        self.assertGreater(len(trades), 0, "synthetic data should produce scanner trades")
        self.assertEqual(trades, sorted(trades, key=lambda t: (t["market"], t["ticker"],
                                                               t["entry_date"], t["pattern"])))
        for t in trades:
            for key in ("r", "r_net", "conf", "entry_date", "exit_date", "market", "outcome",
                        "entry_level", "grade_proxy", "shortlist_rank"):
                self.assertIn(key, t)
            self.assertNotEqual(t["outcome"], "open_at_end")
            if t["excess_vs_index_pct"] is not None:
                self.assertAlmostEqual(t["excess_vs_index_pct"],
                                       round(t["ret_net_pct"] - t["index_ret_pct"], 3), delta=1.1e-3)
            if t["excess_vs_universe_pct"] is not None:
                self.assertAlmostEqual(t["excess_vs_universe_pct"],
                                       round(t["ret_net_pct"] - t["universe_ret_pct"], 3), delta=1.1e-3)
        self.assertEqual(_real_data_mtimes(), before)

    def test_limit_writes_smoke_files_only(self):
        out = self._run("smoke", workers=1, limit=2)
        names = sorted(p.name for p in out.iterdir())
        self.assertEqual(names, ["backtest_smoke.json", "backtest_trades_a_in_caution_smoke.json",
                                 "backtest_trades_grade_a_smoke.json",
                                 "backtest_trades_live_smoke.json",
                                 "backtest_trades_raw_smoke.json", "backtest_trades_smoke.json",
                                 "backtest_trades_walkforward_smoke.json"])
        payload = json.loads((out / "backtest_smoke.json").read_text())
        self.assertTrue(payload["smoke"])
        self.assertEqual(payload["method"]["universe"]["us"]["run"], 2)

    def test_limit_zero_never_means_a_full_run(self):
        out = self.root / "zero"
        out.mkdir()
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                bt.run_backtest(["us"], progress=lambda *_: None, limit=bad, workers=1,
                                portfolio=False, cfg=self.cfg, data_dir=out, db_path=self.db,
                                universe_dir=self.uni)
        self.assertEqual(list(out.iterdir()), [])

    def test_injected_shortlist_size(self):
        out = self._run("short", workers=1, cfg=_cfg(shortlist_size=1), raw=False)
        a = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(a["method"]["settings"]["shortlist_size"], 1)
        trades = json.loads((out / "backtest_trades.json").read_text())
        self.assertTrue(trades)
        self.assertTrue(all(t["shortlist_rank"] == 1 for t in trades))
        self.assertGreater(a["markets"]["us"]["filters"]["removed"]["not_shortlisted"], 0)

    def test_live_configuration_replay(self):
        cfg = _cfg(pattern_tuning={"disabled": {"us": ["ascending_triangle", "bull_flag"]}})
        out = self._run("live", workers=1, cfg=cfg, raw=False)
        a = json.loads((out / "backtest_latest.json").read_text())
        lc = a["markets"]["us"]["live_config"]
        self.assertTrue({"ascending_triangle", "bull_flag"} <= set(lc["disabled"]))
        self.assertEqual(lc["label"], "in-sample selection")
        live = json.loads((out / "backtest_trades_live.json").read_text())
        self.assertEqual(lc["trades"], len(live))
        self.assertFalse({t["pattern"] for t in live} & set(lc["disabled"]))
        self.assertNotIn("bull_flag", lc["per_pattern"])

    def test_grade_replays(self):
        """The grade profiles' own re-selection: grade-A only with the live
        configuration's switched-off patterns removed before the pick, A-in-caution
        with every pattern; each written with its disabled set."""
        cfg = _cfg(pattern_tuning={"disabled": {"us": ["bull_flag"]}})
        out = self._run("grade", workers=1, cfg=cfg, raw=False)
        a = json.loads((out / "backtest_latest.json").read_text())
        gr = a["markets"]["us"]["grade_replay"]
        lc = a["markets"]["us"]["live_config"]
        self.assertEqual(gr["grade_a"]["disabled"], lc["disabled"])
        self.assertIn("bull_flag", gr["grade_a"]["disabled"])
        self.assertEqual(gr["a_in_caution"]["disabled"], [])
        ga = json.loads((out / "backtest_trades_grade_a.json").read_text())
        gc = json.loads((out / "backtest_trades_a_in_caution.json").read_text())
        headline = json.loads((out / "backtest_trades.json").read_text())
        self.assertEqual(gr["grade_a"]["trades"], len(ga))
        self.assertEqual(gr["a_in_caution"]["trades"], len(gc))
        self.assertTrue(all(t["grade_proxy"] == "A" for t in ga))
        self.assertFalse({t["pattern"] for t in ga} & set(lc["disabled"]))
        self.assertTrue(all(t["grade_proxy"] == "A" or t["regime"] != "caution" for t in gc))
        self.assertGreater(gr["grade_a"]["not_grade_a"], 0)
        for t in ga + gc:
            for k in bt._BASELINE_KEYS:
                self.assertIn(k, t)
        # every headline A trade whose stock was never busy with a B/C order is kept
        self.assertTrue(len(gc) > 0 and len(headline) > 0)
        from atlas2 import portfolio_sim as ps
        exclude, _ = ps.pattern_filter(a, cfg)
        self.assertTrue(ps._grade_stream_ok(a, "grade_a_only", exclude))
        self.assertTrue(ps._grade_stream_ok(a, "as_designed_a_in_caution", exclude))
        # the grade-A replay was made with today's list only
        self.assertFalse(ps._grade_stream_ok(a, "grade_a_only", {"us": []}))
        self.assertTrue(ps._grade_stream_ok(a, "as_designed_a_in_caution", {"us": []}))

    def test_portfolio_step_uses_the_injected_paths_and_config(self):
        before = _real_data_mtimes()
        cfg = _cfg(pattern_tuning={"disabled": {"us": ["ascending_triangle"]}})
        out = self._run("pf", workers=1, cfg=cfg, raw=False, portfolio=True)
        self.assertTrue((out / "portfolio_sim.json").exists())
        a = json.loads((out / "backtest_latest.json").read_text())
        self.assertIn("portfolio", a)
        prof = {p["key"]: p for p in a["portfolio"]["profiles"]}
        self.assertIn("ascending_triangle", prof["as_designed_enabled_only"]["disabled_patterns"]["us"])
        # both hindsight sides come from the backtest's own re-selections
        self.assertEqual(prof["as_designed_enabled_only"]["stream"], "live_config")
        self.assertEqual(prof["as_designed_walk_forward"]["stream"], "walk_forward_replay")
        self.assertEqual(a["portfolio"]["hindsight_basis"], "re-selection")
        self.assertEqual(prof["as_designed_enabled_only"]["hindsight_effect_pp"],
                         a["portfolio"]["hindsight_effect_pp"])
        # the grade profiles trade the backtest's grade replays
        for key in ("grade_a_only", "as_designed_a_in_caution"):
            if "error" not in prof[key]:
                self.assertEqual(prof[key]["stream"], "grade_replay", key)
                self.assertNotIn("approximate", prof[key]["label"], key)
        self.assertEqual(prof["grade_a_only"].get("disabled_patterns", {}).get("us"),
                         prof["as_designed_enabled_only"]["disabled_patterns"]["us"])
        self.assertIn("us", a["portfolio"]["benchmark"])
        self.assertIsNotNone(a["portfolio"]["benchmark"]["us"].get("index_cagr_pct"))
        self.assertEqual(_real_data_mtimes(), before)


    def test_smoke_and_variants_never_touch_headline(self):
        """--limit and the labelled variants write only suffixed files and never run
        the portfolio step, even when it is requested."""
        runs = {"smoke_pf": {"limit": 2}, "grid_pf": {"legacy_grid": True},
                "entry_pf": {"entry_mode": "close_confirm"}}
        for name, kw in runs.items():
            out = self._run(name, workers=1, raw=False, portfolio=True, **kw)
            names = sorted(p.name for p in out.iterdir())
            self.assertNotIn("backtest_latest.json", names, name)
            self.assertNotIn("portfolio_sim.json", names, name)
            self.assertNotIn("backtest_trades.json", names, name)
            tag = {"smoke_pf": "_smoke", "grid_pf": "_legacy_grid",
                   "entry_pf": "_close_confirm"}[name]
            self.assertTrue(names and all(n.endswith(f"{tag}.json") for n in names), (name, names))
            payload = json.loads((out / f"backtest{tag}.json").read_text())
            self.assertNotIn("portfolio", payload, name)

    def test_walk_forward_replay(self):
        manual = ["ascending_triangle", "bull_flag"]
        cfg = _cfg(pattern_tuning={"disabled": {"us": manual}})
        out = self._run("wf", workers=1, cfg=cfg, raw=False)
        a = json.loads((out / "backtest_latest.json").read_text())
        wfb = a["markets"]["us"]["walk_forward"]
        y0, y1 = int(wfb["span"][0]), int(wfb["span"][1])
        self.assertEqual(sorted(wfb["disabled_by_year"]), [str(y) for y in range(y0, y1 + 1)])
        for y, lst in wfb["disabled_by_year"].items():
            self.assertTrue(set(manual) <= set(lst), y)          # the manual list applies every year
        wf = json.loads((out / "backtest_trades_walkforward.json").read_text())
        self.assertEqual(wfb["trades"], len(wf))
        self.assertTrue(wf, "synthetic data should produce walk-forward trades")
        for t in wf:
            self.assertNotIn(t["pattern"], wfb["disabled_by_year"][t["signal_date"][:4]])
            for k in bt._BASELINE_KEYS:
                self.assertIn(k, t)
        # the file's disabled lists are what the portfolio re-judges from the headline stream
        from atlas2 import portfolio_sim as ps
        trades = json.loads((out / "backtest_trades.json").read_text())
        self.assertTrue(ps._walk_forward_stream_ok(a, trades, ps._tuning(cfg)))
        self.assertFalse(ps._walk_forward_stream_ok(a, trades, ps._tuning(_cfg())))

    def test_portfolio_step_replays_the_reference_stream(self):
        from atlas2 import portfolio_sim as ps
        out = self.root / "ref"
        out.mkdir()
        old = [{"market": "us", "ticker": f"OLD{k}", "pattern": "bull_flag", "conf": 0.7,
                "entry_date": f"2019-{1 + k % 12:02d}-{2 + k // 12:02d}",
                "exit_date": f"2019-{1 + k % 12:02d}-{20 + k // 12:02d}", "r": 0.4 - (k % 3) * 0.5,
                "ret_pct": 2.0, "outcome": "time"} for k in range(24)]
        (out / ps.REFERENCE_PRE_COSTS_PATH.name).write_text(json.dumps(old))
        bt.run_backtest(["us"], progress=lambda *_: None, workers=1, portfolio=True, cfg=self.cfg,
                        data_dir=out, db_path=self.db, universe_dir=self.uni)
        a = json.loads((out / "backtest_latest.json").read_text())
        ref = a["portfolio"]["reference_pre_costs"]
        self.assertEqual(ref["source"], ps.REFERENCE_PRE_COSTS_PATH.name)
        self.assertEqual(ref["runs"], ps.TIE_RESHUFFLES + 1)
        self.assertLessEqual(ref["tie_break_cagr_pct"]["p5"], ref["cagr_pct"])
        self.assertLessEqual(ref["cagr_pct"], ref["tie_break_cagr_pct"]["p95"])
        self.assertIn(f"{ref['cagr_pct']}%/yr", " ".join(a["portfolio"]["notes"]))


class CarryOverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.db, cls.uni, _ = _build_synthetic_market(cls.root, n_tickers=4, markets=("us", "de"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_bt(self, out, markets, cfg, **kw):
        return bt.run_backtest(markets, progress=lambda *_: None, workers=1, portfolio=False,
                               cfg=cfg, data_dir=out, db_path=self.db, universe_dir=self.uni, **kw)

    STREAMS = ("backtest_trades_live.json", "backtest_trades_walkforward.json",
               "backtest_trades_grade_a.json", "backtest_trades_a_in_caution.json")

    def test_rerunning_some_markets_keeps_the_others(self):
        from atlas2 import portfolio_sim as ps
        out = self.root / "carry"
        out.mkdir()
        cfg = _cfg(pattern_tuning={"disabled": {"de": ["bull_flag"]}})
        self.run_bt(out, ["us", "de"], cfg)
        first = json.loads((out / "backtest_latest.json").read_text())
        first_trades = json.loads((out / "backtest_trades.json").read_text())
        first_streams = {f: [t for t in json.loads((out / f).read_text()) if t["market"] == "de"]
                         for f in self.STREAMS}
        self.assertTrue(first_streams["backtest_trades_live.json"])
        self.assertTrue(first_streams["backtest_trades_walkforward.json"])
        de_disabled = scan.load_pattern_tuning("de", cfg, bt_path=out / "backtest_latest.json")[0]
        self.assertTrue(any(t["market"] == "de" for t in first_trades))
        self.run_bt(out, ["us"], cfg)
        second = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(list(second["markets"]), ["us", "de"])
        self.assertEqual(second["markets"]["de"], first["markets"]["de"])
        self.assertIn("de", second["method"]["period"])
        self.assertEqual(second["method"]["carried_over"], {"de": first["generated"]})
        trades = json.loads((out / "backtest_trades.json").read_text())
        self.assertEqual([t for t in trades if t["market"] == "de"],
                         [t for t in first_trades if t["market"] == "de"])
        self.assertEqual(scan.load_pattern_tuning("de", cfg, bt_path=out / "backtest_latest.json")[0],
                         de_disabled)
        src = scan.load_pattern_tuning("de", cfg, bt_path=out / "backtest_latest.json")[2]
        self.assertIn("kept from the run of", src)
        # the kept market's replays are carried too, so the portfolio still uses them
        for f in self.STREAMS:
            self.assertEqual([t for t in json.loads((out / f).read_text()) if t["market"] == "de"],
                             first_streams[f], f)
        exclude, _ = ps.pattern_filter(second, cfg)
        self.assertIn("bull_flag", exclude["de"])
        self.assertTrue(ps._live_stream_ok(second, exclude))
        self.assertTrue(ps._walk_forward_stream_ok(second, trades, ps._tuning(cfg)))
        for key in ps.GRADE_STREAMS:
            self.assertTrue(ps._grade_stream_ok(second, key, exclude), key)

    def test_different_costs_need_an_explicit_choice(self):
        """Costs, entry slippage and the index dividend yield are part of what makes
        a kept market comparable, compared with the market's own recorded values."""
        out = self.root / "costs"
        out.mkdir()
        self.run_bt(out, ["us", "de"], _cfg())
        first = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(first["method"]["costs_pct_per_side"]["de"], 0.1)
        for over in ({"costs_pct_per_side": {"de": 0.60}}, {"entry_slippage_pct": {"de": 0.5}},
                     {"index_dividend_yield_pct": {"de": 3.0}}, {"stop_slippage_pct": {"de": 0.9}}):
            with self.assertRaises(ValueError, msg=over):
                self.run_bt(out, ["us"], _cfg(backtest=over))
        other = _cfg(backtest={"costs_pct_per_side": {"de": 0.60}})
        self.run_bt(out, ["us"], other, keep_others=True)
        kept = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(kept["method"]["kept_with_different_settings"], ["de"])
        self.assertEqual(kept["method"]["costs_pct_per_side"]["de"], 0.1)   # its own, not this run's
        self.assertEqual(kept["markets"]["de"], first["markets"]["de"])
        info = scan.load_pattern_tuning_info("de", other, bt_path=out / "backtest_latest.json")
        self.assertTrue(info["kept_with_different_settings"])
        self.assertIn("DIFFERENT settings", info["source"])
        # a cost change for the re-run market only: DE still matches its own costs
        self.run_bt(out, ["us"], _cfg(backtest={"costs_pct_per_side": {"us": 0.3}}))
        back = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(back["method"]["kept_with_different_settings"], [])
        self.assertEqual(back["method"]["costs_pct_per_side"], {"us": 0.3, "de": 0.1})

    def test_different_settings_need_an_explicit_choice(self):
        out = self.root / "choice"
        out.mkdir()
        self.run_bt(out, ["us", "de"], _cfg())
        other = _cfg(backtest={"max_hold": 30})
        with self.assertRaises(ValueError):
            self.run_bt(out, ["us"], other)
        self.run_bt(out, ["us"], other, keep_others=True)
        kept = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(list(kept["markets"]), ["us", "de"])
        self.run_bt(out, ["us"], other, replace_others=True)
        dropped = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(list(dropped["markets"]), ["us"])
        trades = json.loads((out / "backtest_trades.json").read_text())
        self.assertFalse(any(t["market"] == "de" for t in trades))


    def test_old_v1_previous_file_is_never_dropped_silently(self):
        for name, content in (("v1", json.dumps({"generated": "2026-08-18T20:29:00",
                                                 "available": True,
                                                 "markets": {"us": {}, "de": {}}})),
                              ("corrupt", "{not json")):
            out = self.root / f"old_{name}"
            out.mkdir()
            (out / "backtest_latest.json").write_text(content)
            with self.assertRaises(ValueError) as ctx:
                self.run_bt(out, ["us"], _cfg())
            self.assertIn("DE", str(ctx.exception))
            self.assertIn("--replace", str(ctx.exception))
            self.assertEqual((out / "backtest_latest.json").read_text(), content)   # untouched
            self.assertFalse((out / "backtest_trades.json").exists())
            msgs: list[str] = []
            bt.run_backtest(["us"], progress=msgs.append, workers=1, portfolio=False, cfg=_cfg(),
                            data_dir=out, db_path=self.db, universe_dir=self.uni,
                            replace_others=True)
            self.assertTrue(any(m.startswith("WARN dropping the previous results for DE")
                                for m in msgs), (name, msgs[:3]))
            new = json.loads((out / "backtest_latest.json").read_text())
            self.assertEqual(list(new["markets"]), ["us"])
        # re-running every market needs no choice
        out = self.root / "old_full"
        out.mkdir()
        (out / "backtest_latest.json").write_text(json.dumps({"markets": {"us": {}, "de": {}}}))
        self.run_bt(out, ["us", "de"], _cfg())
        self.assertEqual(list(json.loads((out / "backtest_latest.json").read_text())["markets"]),
                         ["us", "de"])

    def test_carried_markets_keep_their_own_date_and_settings(self):
        out = self.root / "dates"
        out.mkdir()
        self.run_bt(out, ["us", "de"], _cfg())
        first = json.loads((out / "backtest_latest.json").read_text())
        other = _cfg(backtest={"max_hold": 30})
        with mock.patch.object(bt, "datetime") as dt:     # a later run
            dt.now.return_value = datetime_cls(2031, 1, 2, 3, 4, 5)
            self.run_bt(out, ["us"], other, keep_others=True)
        second = json.loads((out / "backtest_latest.json").read_text())
        m = second["method"]
        self.assertEqual(m["carried_over"], {"de": first["generated"]})
        self.assertEqual(m["settings_by_market"]["us"]["max_hold"], 30)
        self.assertEqual(m["settings_by_market"]["de"]["max_hold"], 40)
        self.assertEqual(m["settings"]["max_hold"], 30)
        self.assertEqual(m["kept_with_different_settings"], ["de"])
        info = scan.load_pattern_tuning_info("de", other, bt_path=out / "backtest_latest.json")
        self.assertEqual(info["backtest_max_hold"], 40)
        self.assertTrue(info["kept_with_different_settings"])
        self.assertIn(f"kept from the run of {first['generated'][:10]}", info["source"])
        self.assertIn("DIFFERENT settings", info["source"])
        self.assertEqual(scan.load_pattern_tuning_info("us", other,
                                                       bt_path=out / "backtest_latest.json")
                         ["backtest_max_hold"], 30)
        # the same config again: DE is still not comparable, so it needs the choice again
        with self.assertRaises(ValueError):
            self.run_bt(out, ["us"], other)
        self.run_bt(out, ["us"], other, keep_others=True)
        third = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(third["method"]["carried_over"], {"de": first["generated"]})  # not re-dated
        self.assertEqual(third["method"]["kept_with_different_settings"], ["de"])
        self.assertEqual(third["method"]["settings_by_market"]["de"]["max_hold"], 40)
        # back to DE's own settings: comparable again, kept without the flag
        msgs: list[str] = []
        bt.run_backtest(["us"], progress=msgs.append, workers=1, portfolio=False, cfg=_cfg(),
                        data_dir=out, db_path=self.db, universe_dir=self.uni)
        back = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(back["method"]["kept_with_different_settings"], [])
        self.assertEqual(back["method"]["carried_over"], {"de": first["generated"]})
        self.assertTrue(any(x.startswith("NOTE DE had been kept") for x in msgs), msgs[-5:])
        self.assertEqual(back["markets"]["de"], first["markets"]["de"])
        # re-running DE under the other settings clears the flag too
        self.run_bt(out, ["us"], other, keep_others=True)
        self.run_bt(out, ["de"], other)
        fourth = json.loads((out / "backtest_latest.json").read_text())
        self.assertEqual(fourth["method"]["kept_with_different_settings"], [])
        self.assertEqual(fourth["method"]["settings_by_market"]["de"]["max_hold"], 30)

    def test_carry_without_the_live_or_walk_forward_files(self):
        from atlas2 import portfolio_sim as ps
        out = self.root / "nolive"
        out.mkdir()
        cfg = _cfg(pattern_tuning={"disabled": {"de": ["bull_flag"]}})
        self.run_bt(out, ["us", "de"], cfg)
        (out / "backtest_trades_live.json").unlink()
        (out / "backtest_trades_walkforward.json").unlink()
        (out / "backtest_trades_grade_a.json").unlink()
        self.run_bt(out, ["us"], cfg)
        a = json.loads((out / "backtest_latest.json").read_text())
        de = a["markets"]["de"]
        self.assertTrue(de["live_config"]["unavailable"])
        self.assertTrue(de["walk_forward"]["unavailable"])
        self.assertTrue(de["grade_replay"]["unavailable"])
        for f in ("backtest_trades_grade_a.json", "backtest_trades_a_in_caution.json"):
            self.assertFalse(any(t["market"] == "de" for t in json.loads((out / f).read_text())), f)
        for key in ps.GRADE_STREAMS:
            self.assertFalse(ps._grade_stream_ok(a, key, ps.pattern_filter(a, cfg)[0]), key)
        live = json.loads((out / "backtest_trades_live.json").read_text())
        self.assertFalse(any(t["market"] == "de" for t in live))   # no headline trades as "live"
        exclude, _ = ps.pattern_filter(a, cfg)
        self.assertFalse(ps._live_stream_ok(a, exclude))
        trades = json.loads((out / "backtest_trades.json").read_text())
        self.assertFalse(ps._walk_forward_stream_ok(a, trades, ps._tuning(cfg)))


class CliTest(unittest.TestCase):
    def call(self, argv):
        from atlas2 import __main__ as cli
        with mock.patch.object(sys, "argv", ["atlas2"] + argv), \
                mock.patch("atlas2.backtest.run_backtest") as run, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main()
                code = None
            except SystemExit as e:
                code = e.code
        return code, run

    def test_bad_options_never_start_a_run(self):
        for argv in (["backtest", "us", "--limit"], ["backtest", "--limit", "0"],
                     ["backtest", "--limt", "20"], ["backtest", "--limit", "us"],
                     ["backtest", "--workers", "0"], ["backtest", "--limit=-1"],
                     ["backtest", "-h"], ["backtest", "--entry-mode", "market"],
                     ["backtest", "--keep-others", "--replace"]):
            code, run = self.call(argv)
            self.assertEqual(code, 1, argv)
            run.assert_not_called()

    def test_good_options(self):
        code, run = self.call(["backtest", "de", "--limit=5", "--workers", "2", "--raw",
                               "--no-portfolio"])
        self.assertIsNone(code)
        run.assert_called_once_with(["de"], raw=True, limit=5, workers=2, portfolio=False,
                                    legacy_grid=False, entry_mode=None, keep_others=False,
                                    replace_others=False)
        code, run = self.call(["backtest", "--legacy-grid", "--entry-mode", "close_confirm"])
        run.assert_called_once_with(["us", "de", "in"], raw=False, limit=None, workers=None,
                                    portfolio=True, legacy_grid=True, entry_mode="close_confirm",
                                    keep_others=False, replace_others=False)


if __name__ == "__main__":
    unittest.main()
