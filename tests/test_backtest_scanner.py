"""Behavioural tests of the scanner-mode simulation (selection, filters, entries,
de-duplication, shortlist, pending-order occupancy, warm-up) and its no-lookahead
properties. Synthetic data only; never touches data/ or config.json.
"""
from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from atlas2 import backtest as bt
from atlas2 import patterns
from atlas2.indicators import add_indicators, find_pivots, relative_strength_3m
from atlas2.patterns import PatternHit
from tests.test_backtest import random_frame

OHLC = ["open", "high", "low", "close"]
LEGACY = dict(step=5, breakout_wait=15)


def wave_frame(n: int = 600, volume: float = 1e6) -> pd.DataFrame:
    """Sideways sine wave 96..104 (period 40 bars): levels are easy to reason about."""
    t = np.arange(n)
    close = 100 + 4 * np.sin(2 * np.pi * t / 40)
    opn = np.concatenate([[close[0]], close[:-1]])
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    return pd.DataFrame({"open": opn, "high": np.maximum(opn, close) + 0.3,
                         "low": np.minimum(opn, close) - 0.3, "close": close,
                         "volume": np.full(n, volume)}, index=idx)


def flat_frame(n: int = 400, price: float = 100.0, spread: float = 0.005,
               volume: float = 1e6) -> pd.DataFrame:
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    p = np.full(n, price)
    return pd.DataFrame({"open": p, "high": p * (1 + spread), "low": p * (1 - spread),
                         "close": p.copy(), "volume": np.full(n, volume)}, index=idx)


def smooth_bench(n: int = 700, wiggle: float = 0.0) -> pd.DataFrame:
    t = np.arange(n)
    close = 3000 * np.exp(0.0005 * t) * (1 + wiggle * np.sin(t / 7))
    idx = pd.bdate_range("2018-01-01", periods=n, name="date")
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999,
                         "close": close, "volume": 1e9}, index=idx)


def hit(pattern="bull_flag", entry=101.0, stop=90.0, rr=2.0, conf=0.7, status="forming"):
    return PatternHit(pattern, "bullish", entry, stop, entry + rr * (entry - stop), conf,
                      status, 0, 0)


class Script:
    """Fake detector: `fn(i)` gives the hits for the window whose last bar is i."""

    def __init__(self, df: pd.DataFrame, fn=None, by_bar: dict | None = None):
        self.pos = {d: k for k, d in enumerate(df.index)}
        self.fn = fn
        self.by_bar = by_bar or {}
        self.calls: list[tuple[int, int]] = []

    def __call__(self, win):
        i = self.pos[win.index[-1]]
        self.calls.append((i, len(win)))
        out = [f() for f in self.by_bar.get(i, [])]
        if self.fn is not None:
            out += list(self.fn(i))
        return out


def healthy_bench(n: int) -> bt.BenchContext:
    b = bt.BenchContext.from_df(smooth_bench(n))
    b.statuses = ["healthy"] * len(b.statuses)
    return b


def run_scanner(df, bench, s, script, tscore=80.0, **kw):
    counters = bt.new_counters()
    with mock.patch.object(bt, "BULLISH_DETECTORS", [script]), \
            mock.patch.object(bt, "BEARISH_DETECTORS", []), \
            mock.patch.object(bt, "technical_score", return_value=(tscore, [])):
        trades = bt.simulate_scanner(df, "SYN", "us", bench, s, counters, **kw)
    return trades, counters


def run_raw(df, bench, s, script):
    with mock.patch.object(bt, "BULLISH_DETECTORS", [script]):
        return bt.simulate_raw(df, "SYN", "us", bench, s)


class WaveCase(unittest.TestCase):
    def setUp(self):
        self.df = add_indicators(wave_frame())
        self.bench = healthy_bench(len(self.df))
        self.s = bt.Settings()
        self.o, self.h, self.l, self.c = (self.df[k].to_numpy() for k in OHLC)
        self.dates = list(self.df.index.strftime("%Y-%m-%d"))

    def crossing(self, level: float, after: int) -> int:
        """First bar j > after whose high reaches `level` while the previous close is below it."""
        return next(j for j in range(after + 1, len(self.h))
                    if self.h[j] >= level and self.c[j - 1] < level)

    def below(self, level: float, first: int, **kw):
        """Script fn: a forming hit at `level` on every bar >= first that closes below it."""
        return lambda i: [hit(entry=level, **kw)] if i >= first and self.c[i] < level else []


class ScannerScenarioTest(WaveCase):
    def test_forming_setup_redetected_daily_until_the_buy_stop_fills(self):
        j = self.crossing(101.0, 310)
        script = Script(self.df, lambda i: [hit(entry=101.0)] if 310 <= i < j else [])
        trades, cnt = run_scanner(self.df, self.bench, self.s, script)
        self.assertGreater(j, 312)
        self.assertEqual(trades[0]["signal_date"], self.dates[j - 1])
        self.assertEqual(trades[0]["entry_date"], self.dates[j])
        self.assertEqual(trades[0]["entry_price"], 101.0)
        self.assertEqual(trades[0]["entry_kind"], "buy_stop")
        # every earlier day had a one-day order that did not fill
        self.assertEqual(cnt["orders_not_filled_today"], (j - 1) - 310)
        self.assertEqual(cnt["removed"]["no_breakout"], 0)

    def test_legacy_grid_waits_for_the_frozen_buy_stop(self):
        s = bt.Settings(**LEGACY)
        trades, _ = run_scanner(self.df, self.bench, s,
                                Script(self.df, by_bar={310: [lambda: hit(entry=101.0)]}))
        j = self.crossing(101.0, 310)
        self.assertLessEqual(j, 310 + 15)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["signal_date"], self.dates[310])
        self.assertEqual(trades[0]["entry_date"], self.dates[j])
        self.assertEqual(trades[0]["entry_price"], 101.0)

    def test_triggered_fills_next_open(self):
        e = float(self.c[320]) / 1.01
        trades, _ = run_scanner(self.df, self.bench, self.s, Script(self.df, by_bar={
            320: [lambda: hit(entry=e, stop=e * 0.9, status="triggered")]}))
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["entry_date"], self.dates[321])
        self.assertEqual(trades[0]["entry_price"], round(float(self.o[321]), 4))
        self.assertEqual(trades[0]["status_at_signal"], "triggered")
        self.assertEqual(trades[0]["entry_kind"], "next_open")

    def test_tech_score_floor(self):
        always = Script(self.df, lambda i: [hit(entry=101.0)])
        trades, cnt = run_scanner(self.df, self.bench, self.s, always, tscore=39.9)
        self.assertEqual(trades, [])
        self.assertGreater(cnt["signals_evaluated"], 0)
        self.assertEqual(cnt["removed"]["tech_score"], cnt["signals_evaluated"])
        self.assertEqual(always.calls, [])          # detectors never ran
        trades, _ = run_scanner(self.df, self.bench, self.s, Script(self.df, self.below(101.0, 310)),
                                tscore=40.0)
        self.assertTrue(trades)

    def test_risk_reward_floor(self):
        j = self.crossing(101.0, 310)
        trades, cnt = run_scanner(self.df, self.bench, self.s,
                                  Script(self.df, by_bar={j - 1: [lambda: hit(rr=1.7)]}))
        self.assertEqual(trades, [])
        self.assertEqual(cnt["removed"]["risk_reward"], 1)
        trades, _ = run_scanner(self.df, self.bench, self.s,
                                Script(self.df, by_bar={j - 1: [lambda: hit(rr=1.8)]}))
        self.assertEqual(len(trades), 1)

    def test_best_hit_is_max_confidence_then_rr(self):
        j = self.crossing(101.0, 310)
        trades, _ = run_scanner(self.df, self.bench, self.s, Script(self.df, by_bar={j - 1: [
            lambda: hit("bull_flag", conf=0.6, rr=4.0),
            lambda: hit("double_bottom", conf=0.8, rr=2.0),
            lambda: hit("darvas_box", conf=0.8, rr=3.0)]}))
        self.assertEqual([t["pattern"] for t in trades], ["darvas_box"])

    def test_traded_level_blocked_for_window_bars_and_open_position_blocks(self):
        j = self.crossing(101.0, 310)
        other = j + 8                                     # while trade 1 is open
        script = Script(self.df, self.below(101.0, 310),
                        by_bar={other: [lambda: hit("double_bottom", entry=102.0)]})
        trades, cnt = run_scanner(self.df, self.bench, self.s, script)
        sig = [self.dates.index(t["signal_date"]) for t in trades]
        self.assertGreaterEqual(len(sig), 2)
        self.assertEqual(sig[0], j - 1)
        self.assertGreaterEqual(sig[1], sig[0] + self.s.window + 1)
        self.assertNotIn(other, sig)
        self.assertGreater(cnt["removed"]["already_traded"], 0)
        for a, b in zip(trades, trades[1:]):
            self.assertGreater(b["entry_date"], a["exit_date"])

    def test_raw_mode_ignores_triggered(self):
        e = float(self.c[320]) / 1.01
        script = Script(self.df, by_bar={320: [lambda: hit(entry=e, stop=e * 0.9, status="triggered")]})
        self.assertEqual(run_raw(self.df, self.bench, self.s, script), [])
        j = self.crossing(101.0, 310)
        script = Script(self.df, by_bar={j - 1: [lambda: hit(entry=101.0)]})
        self.assertEqual(len(run_raw(self.df, self.bench, self.s, script)), 1)


class DailyReplayTest(WaveCase):
    def test_unfilled_order_redetected_next_day_at_refreshed_levels(self):
        j = self.crossing(101.0, 310)
        script = Script(self.df, by_bar={j - 2: [lambda: hit(entry=104.5)],    # not reached on j-1
                                         j - 1: [lambda: hit(entry=101.0)]})   # refreshed level
        trades, cnt = run_scanner(self.df, self.bench, self.s, script)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["signal_date"], self.dates[j - 1])
        self.assertEqual(trades[0]["entry_date"], self.dates[j])
        self.assertEqual(trades[0]["entry_level"], 101.0)
        self.assertEqual(cnt["orders_not_filled_today"], 1)
        self.assertEqual(cnt["removed"]["no_breakout"], 0)

    def test_unfilled_order_does_not_block_its_level(self):
        j = self.crossing(101.0, 310)
        by_bar = {j - 2: [lambda: hit(entry=101.0)], j - 1: [lambda: hit(entry=101.0)]}
        trades, cnt = run_scanner(self.df, self.bench, self.s, Script(self.df, by_bar=by_bar))
        self.assertEqual([t["signal_date"] for t in trades], [self.dates[j - 1]])
        self.assertEqual(cnt["orders_not_filled_today"], 1)
        self.assertEqual(cnt["removed"]["already_traded"], 0)
        # legacy placement: the attempted order blocks the level the next day
        legacy = dataclasses.replace(self.s, seen_on_fill_only=False)
        trades, cnt = run_scanner(self.df, self.bench, legacy, Script(self.df, by_bar=by_bar))
        self.assertEqual(trades, [])
        self.assertEqual(cnt["removed"]["already_traded"], 1)

    def test_legacy_grid_checks_every_fifth_bar(self):
        s = bt.Settings(**LEGACY)
        script = Script(self.df, lambda i: [])
        run_scanner(self.df, self.bench, s, script)
        bars = sorted({i for i, _ in script.calls})
        self.assertTrue(bars)
        self.assertEqual({(b - s.window) % 5 for b in bars}, {0})


class SelectionTest(WaveCase):
    """The live scanner drops disabled patterns BEFORE it picks the best hit."""

    def two_hits(self, rr_a=2.0):
        j = self.crossing(101.0, 310)
        return j, Script(self.df, by_bar={j - 1: [lambda: hit("A", conf=0.8, rr=rr_a),
                                                  lambda: hit("B", conf=0.6, rr=2.0)]})

    def test_disabled_pattern_is_skipped_before_the_pick(self):
        _, script = self.two_hits()
        base, _ = run_scanner(self.df, self.bench, self.s, script)
        self.assertEqual([t["pattern"] for t in base], ["A"])
        _, script = self.two_hits()
        live, _ = run_scanner(self.df, self.bench, self.s, script, disabled=frozenset({"A"}))
        self.assertEqual([t["pattern"] for t in live], ["B"])
        # deleting A's trades from the unfiltered stream is NOT the live configuration
        self.assertEqual([t for t in base if t["pattern"] != "A"], [])

    def test_best_that_fails_rr_is_recorded_and_runner_up_trades_when_disabled(self):
        _, script = self.two_hits(rr_a=1.5)
        best: set = set()
        base, cnt = run_scanner(self.df, self.bench, self.s, script, best_patterns=best)
        self.assertEqual(base, [])
        self.assertEqual(cnt["removed"]["risk_reward"], 1)
        self.assertIn("A", best)
        _, script = self.two_hits(rr_a=1.5)
        live, _ = run_scanner(self.df, self.bench, self.s, script, disabled=frozenset({"A"}))
        self.assertEqual([t["pattern"] for t in live], ["B"])


class RegimeWarmupTest(unittest.TestCase):
    def test_unknown_regime_is_traded_at_normal_size(self):
        df = add_indicators(random_frame(11, n=700, drift=0.0007, vol=0.02))
        bench = bt.BenchContext.from_df(smooth_bench(700))
        bench.statuses = ["unknown"] * len(bench.statuses)
        c = df["close"].to_numpy()
        script = Script(df, lambda i: [hit(entry=c[i] * 1.01, stop=c[i] * 0.9)])
        trades, cnt = run_scanner(df, bench, bt.Settings(), script)
        self.assertEqual(cnt["removed"]["regime_downtrend"], 0)
        self.assertTrue(trades)
        self.assertTrue(all(t["regime"] == "unknown" for t in trades))

    def test_warmup_days_are_not_signals(self):
        n = 700
        df = add_indicators(random_frame(11, n=n))
        bench = healthy_bench(n)
        s = bt.Settings(**LEGACY)
        c = df["close"].to_numpy()
        script = Script(df, lambda i: [hit(entry=c[i] * 1.01, stop=c[i] * 0.9)])
        trades, cnt = run_scanner(df, bench, s, script)
        warm_end = bench.dates[bt.WARMUP_BARS - 1]
        self.assertEqual(cnt["warmup"], len([k for k in range(120, n - 5, 5) if k < bt.WARMUP_BARS - 1]))
        self.assertTrue(trades)
        self.assertTrue(all(t["signal_date"] >= warm_end for t in trades))
        bars = sorted({i for i, _ in script.calls})
        self.assertEqual(bars, [i for i in range(170, n - 5, 5) if i >= bt.WARMUP_BARS - 1])

    def test_young_listing_scanned_from_its_120th_bar(self):
        n = 700
        bench = healthy_bench(n)
        young = random_frame(4, n=200, start=str(bench.dates[400]))
        self.assertEqual(young.index[0].strftime("%Y-%m-%d"), bench.dates[400])
        df = add_indicators(young)
        c = df["close"].to_numpy()
        script = Script(df, lambda i: [hit(entry=c[i] * 1.01, stop=c[i] * 0.9)])
        run_scanner(df, bench, bt.Settings(**LEGACY), script)
        # the step-5 grid runs through bar 120 (the 121st bar; the old gate needed 230 bars)
        self.assertTrue(script.calls)
        self.assertEqual(script.calls[0], (120, 121))
        # daily: from the 120th bar, the live scanner's minimum
        script = Script(df, lambda i: [hit(entry=c[i] * 1.01, stop=c[i] * 0.9)])
        run_scanner(df, bench, bt.Settings(), script)
        self.assertEqual(script.calls[0], (bt.MIN_HISTORY_BARS - 1, bt.MIN_HISTORY_BARS))


class WindowFloorTest(unittest.TestCase):
    def test_window_below_minimum_rejected(self):
        with self.assertRaises(ValueError):
            bt.Settings(window=143)
        with self.assertRaises(ValueError):
            bt.Settings.from_config({"backtest": {"window": 143}})
        self.assertEqual(bt.Settings(window=144).window, 144)
        self.assertEqual(bt.Settings.from_config({"backtest": {"window": 144}}).window, 144)


def _hit(pattern, conf, entry=9.9, stop=9.0, rr=2.0, status="triggered"):
    return bt.Hit(pattern, conf, rr, entry, stop, entry + rr * (entry - stop), status)


def ticker_data(ticker: str, uorder: int, recs: dict, n: int = 80, price: float = 10.0,
                lows: dict | None = None, highs: dict | None = None) -> bt.TickerData:
    idx = pd.bdate_range("2021-01-04", periods=n)
    o = np.full(n, price)
    h = np.full(n, price * 1.005)
    lo = np.full(n, price * 0.995)
    for k, v in (lows or {}).items():
        lo[k] = v
    for k, v in (highs or {}).items():
        h[k] = v
    r = [None] * n
    for k, v in recs.items():
        r[k] = v
    return bt.TickerData(ticker, uorder, np.asarray(idx.strftime("%Y-%m-%d")), o, h, lo, o.copy(), r)


def calendar_and_bench(n: int = 80):
    idx = pd.bdate_range("2021-01-04", periods=n)
    b = bt.BenchContext.from_df(pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                                              "close": 100.0, "volume": 1e9}, index=idx))
    return list(idx.strftime("%Y-%m-%d")), b


class ShortlistTest(unittest.TestCase):
    def setUp(self):
        self.cal, self.bench = calendar_and_bench()

    def test_top_n_cut_with_ties_in_universe_order(self):
        s = bt.Settings(shortlist_size=2)
        tds = [ticker_data("T0", 0, {10: bt.Setup(10, 50.0, "healthy", (_hit("p", 0.8),), False)}),
               ticker_data("T1", 1, {10: bt.Setup(10, 60.0, "healthy", (_hit("p", 0.8),), False)}),
               ticker_data("T2", 2, {10: bt.Setup(10, 50.0, "healthy", (_hit("p", 0.8),), False)})]
        dec = bt.pick_and_rank(tds, s)
        self.assertEqual([d[10][2] for d in dec], [2, 1, 3])     # T0 beats T2 on universe order
        self.assertEqual({d[10][3] for d in dec}, {3})
        counters = bt.new_counters()
        trades = bt.run_selection(tds, self.cal, self.bench, s, "us", counters=counters)
        self.assertEqual(sorted(t["ticker"] for t in trades), ["T0", "T1"])
        self.assertEqual(counters["removed"]["not_shortlisted"], 1)
        self.assertEqual({t["ticker"]: t["shortlist_rank"] for t in trades}, {"T0": 2, "T1": 1})
        self.assertTrue(all(t["candidates_on_date"] == 3 for t in trades))

    def test_disabled_pattern_falls_back_to_next_hit_and_frees_a_slot(self):
        s = bt.Settings(shortlist_size=2)

        def tds():
            return [ticker_data("T0", 0, {10: bt.Setup(10, 50.0, "healthy",
                                                       (_hit("A", 0.9), _hit("D", 0.5)), False)}),
                    ticker_data("T1", 1, {10: bt.Setup(10, 50.0, "healthy", (_hit("B", 0.8),), False)}),
                    ticker_data("T2", 2, {10: bt.Setup(10, 50.0, "healthy", (_hit("C", 0.7),), False)})]
        base = bt.run_selection(tds(), self.cal, self.bench, s, "us")
        self.assertEqual(sorted((t["ticker"], t["pattern"]) for t in base), [("T0", "A"), ("T1", "B")])
        dec = bt.pick_and_rank(tds(), s, frozenset({"A"}))
        self.assertEqual(dec[0][10][0].pattern, "D")               # fell back, as live does
        live = bt.run_selection(tds(), self.cal, self.bench, s, "us", frozenset({"A"}))
        self.assertEqual(sorted((t["ticker"], t["pattern"]) for t in live), [("T1", "B"), ("T2", "C")])

    def test_disabled_set_by_signal_year(self):
        """Walk-forward: the disabled list in force on the signal day's year applies."""
        s = bt.Settings(shortlist_size=5)
        n, start = 120, "2021-11-15"
        idx = pd.bdate_range(start, periods=n)
        cal = list(idx.strftime("%Y-%m-%d"))
        bench = bt.BenchContext.from_df(pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                                                      "close": 100.0, "volume": 1e9}, index=idx))
        k21, k22 = 10, 40
        self.assertEqual((cal[k21][:4], cal[k22][:4]), ("2021", "2022"))

        def tds():
            def one(name, uorder, k):
                td = ticker_data(name, uorder, {k: bt.Setup(k, 50.0, "healthy",
                                                             (_hit("A", 0.9), _hit("D", 0.5)), False)},
                                 n=n)
                td.dates = np.asarray(cal)
                return td
            return [one("T0", 0, k21), one("T1", 1, k22)]

        def picks(disabled):
            return sorted((t["ticker"], t["pattern"])
                          for t in bt.run_selection(tds(), cal, bench, s, "us", disabled))
        self.assertEqual(picks({"2021": frozenset(), "2022": frozenset({"A"})}),
                         [("T0", "A"), ("T1", "D")])
        self.assertEqual(picks({"2021": frozenset({"A"}), "2022": frozenset()}),
                         [("T0", "D"), ("T1", "A")])
        self.assertEqual(picks(frozenset({"A"})), [("T0", "D"), ("T1", "D")])
        self.assertEqual(picks({}), [("T0", "A"), ("T1", "A")])

    def test_injected_config_shortlist_size(self):
        self.assertEqual(bt.Settings.from_config({"shortlist_size": 7}).shortlist_size, 7)
        self.assertEqual(bt.Settings.from_config({}).shortlist_size, 45)

    def test_funnel_adds_up(self):
        s = bt.Settings(shortlist_size=1)
        recs0 = {5: "liquidity", 6: "tech_score", 7: "no_pattern", 8: "risk_reward",
                 10: bt.Setup(10, 60.0, "healthy", (_hit("p", 0.8),), False),
                 60: bt.Setup(60, 60.0, "healthy", (_hit("p", 0.8, entry=9.8),), False)}
        recs1 = {10: bt.Setup(10, 50.0, "healthy", (_hit("p", 0.8),), False),
                 # triggered at 9.0 but the next open is 10.0: more than 3% above the entry
                 12: bt.Setup(12, 50.0, "healthy", (_hit("q", 0.8, entry=9.0, stop=8.0),), False)}
        counters = bt.new_counters()
        trades = bt.run_selection([ticker_data("T0", 0, recs0), ticker_data("T1", 1, recs1)],
                                  self.cal, self.bench, s, "us", counters=counters)
        total = (sum(counters["removed"].values()) + counters["no_pattern"]
                 + counters["orders_not_filled_today"] + counters["censored_at_end"] + len(trades))
        self.assertEqual(counters["signals_evaluated"], 8)
        self.assertEqual(counters["signals_evaluated"], total)
        self.assertEqual([(t["ticker"], t["signal_date"]) for t in trades], [("T0", self.cal[10])])
        self.assertEqual(counters["censored_at_end"], 1)            # day 60 fills too late
        self.assertEqual(counters["removed"]["not_shortlisted"], 1)
        self.assertEqual(counters["removed"]["gap_too_far"], 1)


class GradeReplayTest(unittest.TestCase):
    """The grade profiles' re-selection: a setup below grade A is never ordered,
    so it does not occupy the stock and a later A-grade setup is traded."""

    def setUp(self):
        self.cal, self.bench = calendar_and_bench()
        self.s = bt.Settings(shortlist_size=5)

    def stock(self, b_regime="healthy"):
        # bar 10: grade B (tscore 50, conf 0.8 -> composite 63.5), holds the stock to bar 51
        # bar 20: grade A (tscore 80, conf 0.9 -> composite 84.5), another level
        b = bt.Setup(10, 50.0, b_regime, (_hit("p", 0.8),), False)
        a = bt.Setup(20, 80.0, "healthy", (_hit("q", 0.9, entry=9.95, stop=9.1),), False)
        self.assertEqual(bt.grade_proxy(50.0, 0.8, 2.0, False)[2], "B")
        self.assertEqual(bt.grade_proxy(80.0, 0.9, 2.0, False)[2], "A")
        return [ticker_data("T0", 0, {10: b, 20: a})]

    def replay(self, tds, mode, disabled=frozenset()):
        with mock.patch.object(bt, "add_baselines", return_value=0) as ab:
            out, counters = bt._replay_grade(tds, self.cal, self.bench, self.s, "us", disabled,
                                             mode, bt.new_counters(), {}, None)
        ab.assert_called_once()
        return out, counters

    def test_b_trade_no_longer_blocks_a_later_a_setup(self):
        head = bt.run_selection(self.stock(), self.cal, self.bench, self.s, "us")
        self.assertEqual([(t["signal_date"], t["grade_proxy"]) for t in head],
                         [(self.cal[10], "B")])             # the B position blocks bar 20
        # deletion (the old grade profile) keeps nothing
        self.assertEqual([t for t in head if t["grade_proxy"] == "A"], [])
        out, counters = self.replay(self.stock(), "A")
        self.assertEqual([(t["signal_date"], t["pattern"], t["grade_proxy"]) for t in out],
                         [(self.cal[20], "q", "A")])
        self.assertEqual(counters["removed"]["not_grade_a"], 1)
        total = (sum(counters["removed"].values()) + counters["no_pattern"]
                 + counters["orders_not_filled_today"] + counters["censored_at_end"] + len(out))
        self.assertEqual(counters["signals_evaluated"], total)

    def test_a_in_caution_only_drops_non_a_setups_signalled_in_caution(self):
        out, _ = self.replay(self.stock("healthy"), "A_in_caution")
        self.assertEqual([t["signal_date"] for t in out], [self.cal[10]])
        out, _ = self.replay(self.stock("caution"), "A_in_caution")
        self.assertEqual([t["signal_date"] for t in out], [self.cal[20]])

    def test_ranks_are_kept_and_disabled_patterns_removed_before_the_pick(self):
        # Y: B grade (conflict penalty) but ranked first; X: A grade, ranked second
        y = bt.Setup(10, 65.0, "healthy", (_hit("p", 1.0),), True)
        x = bt.Setup(10, 80.0, "healthy", (_hit("q", 0.6, rr=6.0),), False)
        self.assertEqual(bt.grade_proxy(65.0, 1.0, 2.0, True)[2], "B")
        self.assertEqual(bt.grade_proxy(80.0, 0.6, 6.0, False)[2], "A")
        tds = [ticker_data("T0", 0, {10: y}), ticker_data("T1", 1, {10: x})]
        dec = bt._grade_decisions(bt.pick_and_rank(tds, self.s), "A")
        self.assertEqual(dec[0][10], "not_grade_a")
        self.assertEqual(dec[1][10][2:], [2, 2])         # live ranks before grading
        self.s = dataclasses.replace(self.s, shortlist_size=1)
        out, counters = self.replay(tds, "A")
        self.assertEqual(out, [])                         # X was not on the top-1 shortlist
        self.assertEqual(counters["removed"]["not_shortlisted"], 1)
        # with Y's pattern switched off before the pick, X is the only candidate
        out, _ = self.replay(tds, "A", frozenset({"p"}))
        self.assertEqual([t["ticker"] for t in out], ["T1"])
        with self.assertRaises(ValueError):
            bt._grade_decisions([], "B")


class CensoringTest(unittest.TestCase):
    def test_late_fill_is_censored_whatever_the_outcome(self):
        cal, bench = calendar_and_bench(60)
        s = bt.Settings()
        # triggered at bar 30, filled at the open of bar 31 and stopped out on that bar
        rec = bt.Setup(30, 60.0, "healthy", (_hit("p", 0.8, entry=9.99, stop=9.96),), False)
        td = ticker_data("T0", 0, {30: rec}, n=60, lows={31: 9.90})
        counters = bt.new_counters()
        trades = bt.run_selection([td], cal, bench, s, "us", counters=counters)
        self.assertEqual(trades, [])
        self.assertEqual(counters["censored_at_end"], 1)
        # with a full max_hold window after the fill the same trade is kept (a stop-out)
        cal, bench = calendar_and_bench(80)
        td = ticker_data("T0", 0, {30: rec}, n=80, lows={31: 9.90})
        trades = bt.run_selection([td], cal, bench, s, "us")
        self.assertEqual([t["outcome"] for t in trades], ["stop"])

    def test_random_draws_need_a_full_holding_window(self):
        n = 60
        idx = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(5)
        c = 40 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
        long_df = pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.02, "low": c * 0.98,
                                "close": c}, index=idx)
        short_df = long_df.iloc[:26] * 1.5              # only 5 bars after the entry day (bar 20)
        panel = bt.MarketPanel(bt.frames_to_long({"LONG": long_df, "SHORT": short_df}))
        entry = idx[20].strftime("%Y-%m-%d")
        base, _ = panel._pools(20, 10)
        self.assertEqual([panel.tickers[k] for k in base], ["LONG"])
        trade = {"ticker": "ZZZ", "entry_date": entry, "pattern": "p", "entry_price": 50.0,
                 "stop": 47.5, "target": 55.0}
        got = panel.random_baseline(trade, 4, 10, 0.1, 0.05)
        o, h, l, cc = (long_df[k].to_numpy() for k in OHLC)
        fill = float(o[20])
        _, px, outcome = bt.simulate_exit(o, h, l, cc, 20, fill, fill * 0.95, fill * 1.10, 10)
        want = bt.trade_returns(fill, fill * 0.95, px, outcome, 0.1, 0.05)
        self.assertAlmostEqual(got[0], want["r_net"])
        self.assertAlmostEqual(got[1], want["ret_net_pct"])


class PendingOrderOccupancyTest(unittest.TestCase):
    """A stock is busy from the signal until its order resolves: whether a later
    bar is examined never depends on prices after that bar."""

    def setUp(self):
        self.s = bt.Settings(regime_filter=False, **LEGACY)
        self.n = 400
        self.bench = bt.BenchContext.from_df(smooth_bench(self.n))
        self.first = hit("p1", entry=105.0, stop=95.0, conf=0.8)
        self.second = hit("p2", entry=101.0, stop=95.0, conf=0.6)

    def frame(self, lows=None, highs=None):
        df = flat_frame(self.n)
        for k, v in (lows or {}).items():
            df.iloc[k, df.columns.get_loc("low")] = v
        for k, v in (highs or {}).items():
            df.iloc[k, df.columns.get_loc("high")] = v
        return add_indicators(df)

    def script(self, df, both_at_300=False):
        if both_at_300:
            return Script(df, by_bar={300: [lambda: self.first, lambda: self.second]})
        return Script(df, by_bar={300: [lambda: self.first], 305: [lambda: self.second]})

    def test_later_bar_not_examined_while_the_order_is_pending(self):
        a = self.frame(lows={308: 94.0})            # first order fails at bar 308
        b = self.frame(highs={308: 106.0})          # first order fills at bar 308
        pd.testing.assert_frame_equal(a.iloc[:308], b.iloc[:308])
        dates = list(a.index.strftime("%Y-%m-%d"))
        ta, ca = run_scanner(a, self.bench, self.s, self.script(a))
        tb, _ = run_scanner(b, self.bench, self.s, self.script(b))
        for trades in (ta, tb):
            self.assertNotIn(dates[305], [t["signal_date"] for t in trades])
        self.assertEqual(ta, [])
        self.assertEqual([t["signal_date"] for t in tb], [dates[300]])
        grid = [k for k in range(120, self.n - 5, 5) if k >= bt.WARMUP_BARS - 1]
        self.assertEqual(ca["signals_evaluated"], len(grid) - 1)   # bar 305 skipped
        self.assertEqual(ca["removed"]["failed_before_trigger"], 1)

    def test_bar_after_the_resolution_is_examined(self):
        c = self.frame(lows={302: 94.0}, highs={307: 102.0})
        dates = list(c.index.strftime("%Y-%m-%d"))
        trades, _ = run_scanner(c, self.bench, self.s, self.script(c))
        self.assertEqual([t["signal_date"] for t in trades], [dates[305]])
        self.assertEqual(trades[0]["entry_date"], dates[307])

    def test_raw_mode_equivalents(self):
        a = self.frame(lows={308: 94.0})
        b = self.frame(highs={308: 106.0})
        dates = list(a.index.strftime("%Y-%m-%d"))
        self.assertEqual(run_raw(a, self.bench, self.s, self.script(a)), [])
        self.assertEqual([t["signal_date"] for t in run_raw(b, self.bench, self.s, self.script(b))],
                         [dates[300]])
        c = self.frame(lows={302: 94.0}, highs={307: 102.0})
        self.assertEqual([t["signal_date"] for t in run_raw(c, self.bench, self.s, self.script(c))],
                         [dates[305]])

    def test_two_hits_on_one_bar_only_the_first_order_is_placed(self):
        d = self.frame(highs={302: 102.0})          # would fill the SECOND hit only
        self.assertEqual(run_raw(d, self.bench, self.s, self.script(d, both_at_300=True)), [])
        trades, _ = run_scanner(d, self.bench, self.s, self.script(d, both_at_300=True))
        self.assertEqual(trades, [])                # scanner: the best hit (p1) never fills


class AlreadyTradedTest(unittest.TestCase):
    """The best hit is chosen before the already-traded check; the runner-up is
    never substituted, and a non-fill does not block the level."""

    def setUp(self):
        self.s = bt.Settings(regime_filter=False, **LEGACY)
        self.n = 400
        self.bench = bt.BenchContext.from_df(smooth_bench(self.n))

    def frame(self, highs):
        df = flat_frame(self.n, price=10.0)
        for k, v in highs.items():
            df.iloc[k, df.columns.get_loc("high")] = v
        return add_indicators(df)

    def test_top_pick_already_traded_skips_the_day(self):
        df = self.frame({302: 10.15, 305: 11.5, 312: 10.2})
        a = hit("A", entry=10.1, stop=9.5, conf=0.8)      # target 11.3
        b = hit("B", entry=10.18, stop=9.5, conf=0.6)
        script = Script(df, lambda i: [a, b] if 300 <= i <= 330 else [])
        trades, cnt = run_scanner(df, self.bench, self.s, script)
        self.assertEqual([(t["pattern"], t["outcome"]) for t in trades], [("A", "target")])
        self.assertGreaterEqual(cnt["removed"]["already_traded"], 1)

    def test_unfilled_setup_rechecked_after_its_order_expires(self):
        df = self.frame({318: 10.2})                      # breakout outside the first 15-bar wait
        a = hit("A", entry=10.1, stop=9.5, conf=0.8)
        script = Script(df, lambda i: [a] if 300 <= i <= 330 else [])
        trades, cnt = run_scanner(df, self.bench, self.s, script)
        dates = list(df.index.strftime("%Y-%m-%d"))
        self.assertEqual([t["signal_date"] for t in trades], [dates[315]])
        self.assertEqual(trades[0]["entry_date"], dates[318])
        self.assertEqual(cnt["removed"]["no_breakout"], 1)

    def test_sig4_level_key(self):
        self.assertNotEqual(bt.sig4(1.14), bt.sig4(1.18))
        self.assertEqual(bt.sig4(12345.6), 12350.0)
        self.assertEqual(bt.sig4(101.04), 101.0)


class NoLookaheadTest(unittest.TestCase):
    """Every decision at signal bar i is unchanged when all bars after i are poisoned."""

    def decisions(self, stock: pd.DataFrame, bench_df: pd.DataFrame):
        df = add_indicators(stock)
        bench = bt.BenchContext.from_df(bench_df)
        pos = {d: k for k, d in enumerate(df.index)}
        dpos = {d.strftime("%Y-%m-%d"): k for k, d in enumerate(df.index)}
        log = []

        def det(win):
            c = float(win["close"].iloc[-1])
            log.append(("win", pos[win.index[-1]], len(win), round(c, 9)))
            return [hit(entry=c * 1.02, stop=c * 0.9)]

        def tscore(row, rs3m):
            log.append(("ts", dpos[row["date"]],
                        tuple(round(float(row[k]), 9) for k in ("close", "sma50", "sma200")),
                        None if rs3m is None else round(rs3m, 9)))
            return 80.0, []

        def forming(o, h, l, i, entry, stop, wait, gap):
            log.append(("entry", i, round(entry, 9)))
            return i + 1, None, "no_breakout"

        def triggered(o, i, entry, stop, gap):
            log.append(("entry", i, round(entry, 9)))
            return i + 1, None, "no_breakout"

        with mock.patch.object(bt, "BULLISH_DETECTORS", [det]), \
                mock.patch.object(bt, "BEARISH_DETECTORS", []), \
                mock.patch.object(bt, "technical_score", tscore), \
                mock.patch.object(bt, "enter_forming", forming), \
                mock.patch.object(bt, "enter_triggered", triggered):
            bt.simulate_scanner(df, "SYN", "us", bench, bt.Settings(), bt.new_counters())
        return log, df

    def test_poisoned_future_changes_nothing_before_the_cut(self):
        stock = random_frame(11, n=700, drift=0.0007, vol=0.02)
        bench = smooth_bench(700, wiggle=0.01)
        k = 600
        cut = stock.index[k]
        s2, b2 = stock.copy(), bench.copy()
        s2.loc[s2.index >= cut, OHLC] *= 0.3
        s2.loc[s2.index >= cut, "volume"] = np.nan
        ramp = np.linspace(0.3, 0.1, int((b2.index >= cut).sum()))
        b2.loc[b2.index >= cut, OHLC] = b2.loc[b2.index >= cut, OHLC].mul(ramp, axis=0)
        full, df = self.decisions(stock, bench)
        pois, _ = self.decisions(s2, b2)
        before = lambda log: [e for e in log if e[1] < k]
        self.assertEqual(before(full), before(pois))
        self.assertIn(k - 1, [e[1] for e in full if e[0] == "entry"])   # not vacuous
        for e in full:
            if e[0] == "win":
                self.assertEqual(e[2], bt.Settings().window + 1)
        dates = df.index
        for e in full:
            if e[0] == "ts":
                i = e[1]
                exp = relative_strength_3m(df.iloc[: i + 1], bench[bench.index <= dates[i]])
                self.assertEqual(e[3], None if exp is None else round(exp, 9))


class CausalityTest(unittest.TestCase):
    """Indicators, pivots and detectors only use past data (read-only use of
    patterns.py and indicators.py)."""

    def test_indicators_pivots_and_detectors_are_causal(self):
        fired: dict[str, int] = {}
        for seed, drift, vol in ((5, 0.0008, 0.022), (3, 0.0003, 0.025), (21, 0.0005, 0.03)):
            raw = random_frame(seed, n=520, drift=drift, vol=vol)
            full = add_indicators(raw)
            fh, fl = find_pivots(raw)
            for k in range(200, 520, 7):
                pd.testing.assert_frame_equal(full.iloc[: k + 1], add_indicators(raw.iloc[: k + 1]))
                ph, pl = find_pivots(raw.iloc[: k + 1])
                self.assertEqual(ph, [p for p in fh if p <= k - 4])
                self.assertEqual(pl, [p for p in fl if p <= k - 4])
                sh = k - 170
                for det in patterns.BULLISH_DETECTORS + patterns.BEARISH_DETECTORS:
                    a = det(full.iloc[sh: k + 1])
                    b = det(full.iloc[: k + 1])
                    ka = [(x.pattern, x.entry, x.stop, x.target, x.confidence, x.status,
                           x.start_idx + sh, x.end_idx + sh) for x in a]
                    kb = [(x.pattern, x.entry, x.stop, x.target, x.confidence, x.status,
                           x.start_idx, x.end_idx) for x in b]
                    self.assertEqual(ka, kb, (det.__name__, seed, k))
                    fired[det.__name__] = fired.get(det.__name__, 0) + len(a)
        for det in patterns.BULLISH_DETECTORS + patterns.BEARISH_DETECTORS:
            self.assertGreater(fired.get(det.__name__, 0), 0, f"{det.__name__} never fired")


if __name__ == "__main__":
    unittest.main()
