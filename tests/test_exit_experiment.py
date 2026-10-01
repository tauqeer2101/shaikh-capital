"""Tests for atlas2.exit_experiment (the pre-registered exit experiment).

Synthetic bars and trades only; files go to temp dirs. Never touches data/ or
config.json.
"""
from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from atlas2 import backtest as bt
from atlas2 import exit_experiment as ee
from atlas2.data import PriceStore
from atlas2.indicators import add_indicators
from tests.test_backtest import random_frame

SETTINGS = bt.Settings()      # default costs: us 0.15 / 0.05 / 0.05, de 0.10 / 0.15 / 0.15


def bars(closes, opens=None, highs=None, lows=None, start="2020-01-01") -> pd.DataFrame:
    """Daily bars from closes; open = previous close, high/low 1% around the body
    unless given."""
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else np.r_[c[0], c[:-1]]
    h = np.asarray(highs, float) if highs is not None else np.maximum(o, c) * 1.01
    l = np.asarray(lows, float) if lows is not None else np.minimum(o, c) * 0.99
    idx = pd.bdate_range(start, periods=len(c), name="date")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1e6}, index=idx)


def ohlc(df: pd.DataFrame):
    return tuple(df[k].to_numpy(float) for k in ("open", "high", "low", "close"))


def flat(n: int, price: float = 100.0) -> tuple[np.ndarray, ...]:
    """Flat bars: open = close = price, high +1, low -1."""
    p = np.full(n, price)
    return p.copy(), p + 1, p - 1, p.copy()


def entry(ticker="AAA", market="us", pattern="bull_flag", signal_date="2020-01-02",
          entry_date="2020-01-03", entry_price=100.0, entry_kind="buy_stop", stop=95.0,
          target=110.0, **carry) -> dict:
    e = {"ticker": ticker, "market": market, "pattern": pattern, "signal_date": signal_date,
         "entry_date": entry_date, "entry_price": entry_price, "entry_kind": entry_kind,
         "stop": stop, "target": target}
    for k in ee.CARRY_KEYS:
        e[k] = carry.get(k)
    e["period"] = ee.period_of(entry_date)
    return e


def stored_trade(df: pd.DataFrame, fi: int, fill: float, stop: float, target: float,
                 market: str = "us", entry_kind: str = "buy_stop", pattern: str = "bull_flag",
                 level: float | None = None, s: bt.Settings = SETTINGS, signal_i=None) -> dict:
    """A backtest-style stored trade: the backtest's own exit rules and costs on df."""
    o, h, l, c = ohlc(df)
    dates = bt._date_strings(df.index)
    exit_i, exit_price, outcome = bt.simulate_exit(o, h, l, c, fi, fill, stop, target, 40)
    res = bt.trade_returns(fill, stop, exit_price, outcome, s.cost_pct(market), s.stop_slip(market),
                           s.entry_slip(market) if entry_kind == "buy_stop" else 0.0)
    si = fi - 1 if signal_i is None else signal_i
    return {"market": market, "ticker": "AAA", "pattern": pattern, "conf": 0.8,
            "entry_kind": entry_kind, "signal_date": str(dates[si]), "entry_date": str(dates[fi]),
            "exit_date": str(dates[exit_i]), "entry_price": round(fill, 4),
            "entry_level": round(level if level is not None else fill, 4), "stop": stop,
            "target": target, "rr": round((target - fill) / (fill - stop), 2), "outcome": outcome,
            "hold_bars": exit_i - fi, "r": round(res["r"], 4), "r_net": round(res["r_net"], 4),
            "ret_pct": round(res["ret_pct"], 3), "ret_net_pct": round(res["ret_net_pct"], 3),
            "composite_proxy": 70.0, "grade_proxy": "B", "regime": "healthy"}


class EntriesTest(unittest.TestCase):
    def test_closed_trades_only_and_period_split(self):
        ok = stored_trade(random_frame(1, n=120), 50, 50.0, 47.0, 56.0)
        trades = [ok, {**ok, "outcome": "open_at_end"}, {**ok, "stop": None},
                  {**ok, "stop": 51.0}, "junk", {**ok, "entry_date": "2023-01-01"}]
        entries, drop = ee.entries_from_trades(trades)
        self.assertEqual(drop, {"open_at_end": 1, "invalid": 3})
        self.assertEqual([e["period"] for e in entries], ["train", "test"])
        self.assertEqual(ee.period_of("2022-12-30"), "train")
        self.assertEqual(ee.period_of("2023-01-01"), "test")
        for k in ee.ENTRY_KEYS + ee.CARRY_KEYS:
            self.assertIn(k, entries[0])

    def test_period_years_share_one_denominator(self):
        es = [entry(entry_date=d) for d in ("2020-01-02", "2021-12-31", "2023-06-30", "2024-06-28")]
        y = ee.period_years(es)
        self.assertAlmostEqual(y["train"], 729 / 365.25)
        self.assertAlmostEqual(y["test"], 364 / 365.25)
        span = (pd.Timestamp("2024-06-28") - pd.Timestamp("2020-01-02")).days
        self.assertAlmostEqual(y["pooled"], span / 365.25)
        self.assertEqual(ee.period_years([entry()])["train"], 0.5)   # at least half a year
        self.assertIsNone(ee.period_years([entry()])["test"])


class CostMathTest(unittest.TestCase):
    def test_stop_target_time_match_the_backtest(self):
        for outcome in ("stop", "target", "time"):
            got = ee.variant_returns(100.0, 95.0, 104.0, outcome, "us", SETTINGS, "buy_stop")
            want = bt.trade_returns(100.0, 95.0, 104.0, outcome, 0.15, 0.05, 0.05)
            self.assertEqual(got, want)
        # next-open entries pay no entry slippage
        got = ee.variant_returns(100.0, 95.0, 110.0, "target", "de", SETTINGS, "next_open")
        self.assertAlmostEqual(got["cost_pct_round_trip"], 0.20)
        self.assertAlmostEqual(got["r_net"], (110 * 0.999 - 100.1) / 5)

    def test_stop_exit_pays_stop_slippage(self):
        got = ee.variant_returns(100.0, 95.0, 95.0, "stop", "de", SETTINGS, "buy_stop")
        self.assertAlmostEqual(got["cost_pct_round_trip"], 0.20 + 0.15 + 0.15)
        exit_eff = 95.0 * (1 - 0.001 - 0.0015)
        self.assertAlmostEqual(got["r_net"], (exit_eff - 100 * 1.0025) / 5)

    def test_trail_exit_is_a_market_order_at_the_open(self):
        got = ee.variant_returns(100.0, 95.0, 104.0, "trail", "de", SETTINGS, "next_open")
        # costs both ways + entry_slippage_pct (0.15 DE) as exit slippage, no entry slippage
        self.assertAlmostEqual(got["cost_pct_round_trip"], 0.20 + 0.15)
        self.assertAlmostEqual(got["r_net"], (104 * (1 - 0.001 - 0.0015) - 100.1) / 5)
        self.assertAlmostEqual(got["r"], 0.8)
        self.assertAlmostEqual(got["ret_pct"], 4.0)
        # the default DE table carries the same stop and entry slippage (0.15 / 0.15), so
        # the two are told apart here: with a 0.40 stop slippage the trail still pays the
        # ENTRY slippage (a market order at the open), the stop exit the stop slippage
        s2 = dataclasses.replace(SETTINGS,
                                 stop_slippage_pct={**SETTINGS.stop_slippage_pct, "de": 0.4})
        trail = ee.variant_returns(100.0, 95.0, 104.0, "trail", "de", s2, "next_open")
        self.assertAlmostEqual(trail["cost_pct_round_trip"], 0.20 + 0.15)
        self.assertAlmostEqual(trail["r_net"], (104 * (1 - 0.001 - 0.0015) - 100.1) / 5)
        stop = ee.variant_returns(100.0, 95.0, 95.0, "stop", "de", s2, "next_open")
        self.assertAlmostEqual(stop["cost_pct_round_trip"], 0.20 + 0.40)
        self.assertAlmostEqual(stop["r_net"], (95 * (1 - 0.001 - 0.004) - 100.1) / 5)


class AtrStopTest(unittest.TestCase):
    def test_atr_multiples_and_fallbacks(self):
        self.assertEqual(ee.atr_stop(100.0, 2.0, 3.0, 95.0), (94.0, "atr"))
        self.assertEqual(ee.atr_stop(100.0, 2.0, 5.0, 95.0), (90.0, "atr"))
        self.assertEqual(ee.atr_stop(100.0, None, 3.0, 95.0), (95.0, "atr_missing"))
        self.assertEqual(ee.atr_stop(100.0, float("nan"), 3.0, 95.0), (95.0, "atr_missing"))
        self.assertEqual(ee.atr_stop(100.0, 0.0, 3.0, 95.0), (95.0, "atr_missing"))
        # an ATR stop at or above the fill (negative ATR multiple never happens; a
        # zero-distance stop can): the pattern stop
        self.assertEqual(ee.atr_stop(100.0, -1.0, 3.0, 95.0), (95.0, "atr_missing"))

    def test_atr_at_the_signal_bar_is_causal(self):
        # the ATR read from the full series at bar k equals the ATR computed on
        # bars up to and including k (ewm adjust=False is a forward recursion)
        df = random_frame(3, n=300)
        full = add_indicators(df)["atr14"].to_numpy(float)
        for k in (20, 150, 299):
            part = add_indicators(df.iloc[: k + 1])["atr14"].to_numpy(float)
            self.assertAlmostEqual(full[k], part[-1], places=12)

    def test_run_ticker_uses_the_signal_bar_atr(self):
        df = random_frame(5, n=200)
        atr = add_indicators(df)["atr14"].to_numpy(float)
        dates = bt._date_strings(df.index)
        o = df["open"].to_numpy(float)
        fi, si = 120, 118          # signal two bars before the fill
        e = entry(signal_date=dates[si], entry_date=dates[fi], entry_price=round(o[fi], 4),
                  entry_kind="next_open", stop=round(o[fi] * 0.9, 4), target=round(o[fi] * 1.3, 4))
        out = ee.run_ticker("AAA", "us", [e], df, (), None, SETTINGS)
        recs = {r["variant"]: r for r in out["records"]}
        fill = float(e["entry_price"])
        self.assertAlmostEqual(recs["S2xE1"]["stop"], round(fill - 3 * atr[si], 4))
        self.assertAlmostEqual(recs["S3xE2"]["stop"], round(fill - 5 * atr[si], 4))
        self.assertAlmostEqual(recs["S2xE1"]["atr14_signal"], round(atr[si], 4))
        self.assertEqual(recs["S1xE1"]["stop"], e["stop"])
        self.assertEqual({recs[v]["stop_kind"] for v in ("S2xE1", "S3xE3")}, {"atr"})


class ExitE1Test(unittest.TestCase):
    """E1 is backtest.simulate_exit itself."""

    def test_reproduces_simulate_exit(self):
        df = random_frame(11, n=400)
        o, h, l, c = ohlc(df)
        for fi in (10, 50, 200):
            fill = float(o[fi])
            for stop, target in ((fill * 0.97, fill * 1.05), (fill * 0.9, fill * 1.4)):
                want = bt.simulate_exit(o, h, l, c, fi, fill, stop, target, 40)
                got = ee.variant_exit(o, h, l, c, fi, fill, stop, target, ee.EXITS["E1"])
                self.assertEqual(got, want)

    def test_open_at_end_is_censored(self):
        o, h, l, c = flat(30)
        self.assertIsNone(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E1"]))
        o, h, l, c = flat(41)
        self.assertEqual(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E1"]),
                         (40, 100.0, "time"))


class ExitE2Test(unittest.TestCase):
    def test_no_target_and_120_bar_time_exit(self):
        o, h, l, c = flat(130)
        h[5] = 150.0              # far above the old target: E2 ignores it
        o[7] = 149.0
        got = ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E2"])
        self.assertEqual(got, (120, 100.0, "time"))
        self.assertEqual(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E1"]),
                         (5, 120.0, "target"))

    def test_initial_stop_still_applies(self):
        o, h, l, c = flat(130)
        l[60] = 89.0
        self.assertEqual(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E2"]),
                         (60, 90.0, "stop"))
        o2, h2, l2, c2 = flat(130)
        o2[61] = 85.0             # gap through the stop: exit at the open
        self.assertEqual(ee.variant_exit(o2, h2, l2, c2, 0, 100.0, 90.0, 120.0, ee.EXITS["E2"]),
                         (61, 85.0, "stop"))
        o3, h3, l3, c3 = flat(130)
        l3[0] = 90.0              # fill bar touches the stop: stop
        self.assertEqual(ee.variant_exit(o3, h3, l3, c3, 0, 100.0, 90.0, 120.0, ee.EXITS["E2"]),
                         (0, 90.0, "stop"))

    def test_censored_when_the_data_ends_first(self):
        o, h, l, c = flat(100)
        self.assertIsNone(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E2"]))


class ExitE3Test(unittest.TestCase):
    def trail(self, o, h, l, c, fill_i=0, stop=90.0, max_hold=250, trail_bars=20):
        return ee.simulate_trailing_exit(o, h, l, c, fill_i, stop, max_hold, trail_bars)

    def test_close_below_trailing_low_exits_at_next_open(self):
        o, h, l, c = flat(60)
        # lows are 99 everywhere; bar 30 closes at 98.5 (below 99) -> exit at bar 31's open
        c[30] = 98.5
        o[31] = 98.0
        self.assertEqual(self.trail(o, h, l, c), (31, 98.0, "trail"))
        got = ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E3"])
        self.assertEqual(got, (31, 98.0, "trail"))

    def test_not_signalled_on_the_fill_bar(self):
        o, h, l, c = flat(60)
        c[0] = 50.0               # the fill bar's close is never a trailing signal
        l[0] = 49.0               # (and its low is above the stop of 40)
        self.assertEqual(self.trail(o, h, l, c, stop=40.0, max_hold=10), (10, 100.0, "time"))

    def test_window_is_since_the_fill_until_20_bars(self):
        o, h, l, c = flat(60)
        l[3] = 95.0               # a low inside the window (bars 0..j-1)
        c[8] = 96.0               # above 95: no signal (window still contains bar 3)
        self.assertEqual(self.trail(o, h, l, c, max_hold=30), (30, 100.0, "time"))
        # bar 3 leaves the 20-bar window on bar 24 (window 4..23): a close at 96 signals
        c[24] = 96.0
        o[25] = 97.0
        self.assertEqual(self.trail(o, h, l, c, max_hold=30), (25, 97.0, "trail"))

    def test_window_is_exactly_20_bars_not_19(self):
        # bar 23 looks back at bars 3..22, which still hold the 95 low: a close of
        # 96 must NOT signal there; bar 24 (window 4..23) is the first that can
        o, h, l, c = flat(60)
        l[3] = 95.0
        c[23] = 96.0
        c[24] = 96.0
        o[24], o[25] = 98.0, 97.0
        self.assertEqual(self.trail(o, h, l, c, max_hold=30), (25, 97.0, "trail"))
        # with the 96 close on bar 23 alone the trade runs to the time exit
        o2, h2, l2, c2 = flat(60)
        l2[3] = 95.0
        c2[23] = 96.0
        self.assertEqual(self.trail(o2, h2, l2, c2, max_hold=30), (30, 100.0, "time"))
        # window since the fill: with fill_i = 2 a close below bar 2's low signals on bar 3
        o2, h2, l2, c2 = flat(60)
        l2[2] = 99.5
        c2[3] = 99.4
        o2[4] = 99.0
        self.assertEqual(self.trail(o2, h2, l2, c2, fill_i=2), (4, 99.0, "trail"))

    def test_stop_before_the_trailing_check_within_a_bar(self):
        o, h, l, c = flat(60)
        # the same bar closes below the trailing low AND touches the stop
        c[30], l[30] = 98.5, 89.0
        self.assertEqual(self.trail(o, h, l, c), (30, 90.0, "stop"))

    def test_gap_through_stop_beats_a_pending_trail_exit(self):
        o, h, l, c = flat(60)
        c[30] = 98.5
        o[31] = 85.0              # opens through the stop: a stop at the open, stop slippage
        self.assertEqual(self.trail(o, h, l, c), (31, 85.0, "stop"))

    def test_pending_exit_beats_an_intraday_stop_on_the_next_bar(self):
        o, h, l, c = flat(60)
        c[30] = 98.5
        o[31], l[31] = 97.0, 80.0     # the open (trail exit) comes before the intraday stop
        self.assertEqual(self.trail(o, h, l, c), (31, 97.0, "trail"))

    def test_hard_time_exit_at_fill_plus_250(self):
        o, h, l, c = flat(260)
        self.assertEqual(self.trail(o, h, l, c), (250, 100.0, "time"))
        self.assertEqual(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E3"]),
                         (250, 100.0, "time"))
        # a trail signal on the last bar exits at that bar's open, before the close
        c[249] = 98.5
        o[250] = 99.0
        self.assertEqual(self.trail(o, h, l, c), (250, 99.0, "trail"))

    def test_censored_when_the_data_ends(self):
        o, h, l, c = flat(100)
        self.assertIsNone(self.trail(o, h, l, c))            # no exit within the data
        c[99] = 98.5                                          # signal on the last bar: no next open
        self.assertIsNone(self.trail(o, h, l, c))
        self.assertIsNone(ee.variant_exit(o, h, l, c, 0, 100.0, 90.0, 120.0, ee.EXITS["E3"]))

    def test_fill_bar_stop(self):
        o, h, l, c = flat(60)
        l[0] = 90.0
        self.assertEqual(self.trail(o, h, l, c), (0, 90.0, "stop"))


class NoOverlapTest(unittest.TestCase):
    def test_entry_skipped_while_the_previous_trade_is_open(self):
        # 200 flat bars: E1 exits at bar fill+40 (time), E2 at fill+120, E3 (fill+250)
        # never reaches its exit
        df = bars(np.full(200, 100.0))
        d = bt._date_strings(df.index)
        first = entry(signal_date=d[9], entry_date=d[10], entry_price=100.0, stop=90.0,
                      target=150.0)
        # 60 bars later: after E1's exit (bar 50), inside E2's and E3's holds
        second = entry(signal_date=d[69], entry_date=d[70], entry_price=100.0, stop=90.0,
                       target=150.0, pattern="darvas_box")
        out = ee.run_ticker("AAA", "us", [second, first], df, (), None, SETTINGS)
        cnt = {v: out["counts"][v]["train"] for v in ee.VARIANTS}
        self.assertEqual((cnt["S1xE1"]["trades"], cnt["S1xE1"]["skipped_overlap"]), (2, 0))
        self.assertEqual((cnt["S1xE2"]["trades"], cnt["S1xE2"]["skipped_overlap"]), (1, 1))
        # E3: the first trade never exits (the data ends before bar 260): censored, still blocks
        self.assertEqual((cnt["S1xE3"]["trades"], cnt["S1xE3"]["skipped_overlap"],
                          cnt["S1xE3"]["censored"]), (0, 1, 1))
        self.assertEqual(cnt["S1xE3"]["entries"], 2)
        recs = [r for r in out["records"] if r["variant"] == "S1xE3"]
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["censored"])
        self.assertIsNone(recs[0]["exit_date"])
        # the entries are processed chronologically whatever the input order
        e1 = [r for r in out["records"] if r["variant"] == "S1xE2"][0]
        self.assertEqual(e1["entry_date"], d[10])
        # nothing here has a full 250-bar window: no common entry, and the last
        # eligible entry per hold follows the fill (E3: none)
        self.assertEqual(out["counts_common"]["S1xE1"]["train"]["entries"], 0)
        self.assertEqual(out["common_entry_dates"], [])
        self.assertEqual(out["last_eligible_entry"], {"E1": d[70], "E2": d[70], "E3": None})
        self.assertFalse(any(r["common"] for r in out["records"]))

    def test_entry_on_the_exit_day_is_still_blocked(self):
        df = bars(np.full(120, 100.0))
        d = bt._date_strings(df.index)
        first = entry(signal_date=d[9], entry_date=d[10], stop=90.0, target=150.0)
        same_day = entry(signal_date=d[49], entry_date=d[50], stop=90.0, target=150.0,
                         pattern="cup_and_handle")          # E1 time exit is on bar 50
        # a SIGNAL on the exit day fills the day after: the backtest never re-scans a
        # stock on its exit day, so this one is skipped too
        signal_on_exit_day = entry(signal_date=d[50], entry_date=d[51], stop=90.0, target=150.0,
                                   pattern="darvas_box")
        next_day = entry(signal_date=d[51], entry_date=d[52], stop=90.0, target=150.0,
                         pattern="ascending_triangle")
        out = ee.run_ticker("AAA", "us", [first, same_day, signal_on_exit_day, next_day], df, (),
                            None, SETTINGS)
        c = out["counts"]["S1xE1"]["train"]
        self.assertEqual((c["trades"], c["skipped_overlap"], c["censored"]), (2, 2, 0))
        traded = [r["entry_date"] for r in out["records"] if r["variant"] == "S1xE1"]
        self.assertEqual(traded, [d[10], d[52]])
        # E2 (120 bars): the first entry's window does not fit in 120 bars, so it is
        # censored at the fill and blocks the other three
        c2 = out["counts"]["S1xE2"]["train"]
        self.assertEqual((c2["trades"], c2["skipped_overlap"], c2["censored"]), (0, 3, 1))

    def test_censoring_is_decided_at_the_fill_not_by_the_outcome(self):
        lows = np.full(400, 99.0)
        lows[305] = 85.0              # bar fill+5 of the late entry breaks the 90 stop
        df = bars(np.full(400, 100.0), lows=lows)
        d = bt._date_strings(df.index)
        early = entry(signal_date=d[4], entry_date=d[5], stop=90.0, target=150.0)
        late = entry(signal_date=d[299], entry_date=d[300], stop=90.0, target=150.0,
                     pattern="darvas_box")             # 100 bars left: room for E1 only
        after = entry(signal_date=d[349], entry_date=d[350], stop=90.0, target=150.0,
                      pattern="cup_and_handle")
        out = ee.run_ticker("AAA", "us", [early, late, after], df, (), None, SETTINGS)
        recs = {(r["variant"], r["entry_date"]): r for r in out["records"]}
        # E1: the late entry stops out on bar 305 and is counted; 'after' follows
        r1 = recs[("S1xE1", d[300])]
        self.assertEqual((r1["outcome"], r1["hold_bars"], r1["exit_date"], r1["censored"]),
                         ("stop", 5, d[305], False))
        c1 = out["counts"]["S1xE1"]["train"]
        self.assertEqual((c1["trades"], c1["skipped_overlap"], c1["censored"]), (3, 0, 0))
        # E2 / E3: the same entry would have stopped out inside the data, but its
        # 120- / 250-bar window does not fit: censored at the fill, no exit recorded,
        # and it still blocks the ticker's later entry
        for v in ("S1xE2", "S1xE3", "S3xE3"):
            r = recs[(v, d[300])]
            self.assertTrue(r["censored"])
            self.assertIsNone(r["exit_date"])
            self.assertIsNone(r["outcome"])
            self.assertIsNone(r["ret_net_pct"])
            self.assertNotIn((v, d[350]), recs)
            cv = out["counts"][v]["train"]
            self.assertEqual((cv["trades"], cv["skipped_overlap"], cv["censored"]), (1, 1, 1))
        # the common flag = room for the longest hold (250 bars): only the early entry
        self.assertTrue(recs[("S1xE1", d[5])]["common"])
        self.assertFalse(recs[("S1xE1", d[300])]["common"])
        self.assertFalse(recs[("S1xE1", d[350])]["common"])
        self.assertEqual(out["common_entry_dates"], [d[5]])
        cc = out["counts_common"]["S1xE1"]["train"]
        self.assertEqual((cc["entries"], cc["trades"], cc["skipped_overlap"]), (1, 1, 0))
        self.assertEqual(out["counts_common"]["S1xE3"]["train"]["entries"], 1)
        self.assertEqual(out["last_eligible_entry"], {"E1": d[350], "E2": d[5], "E3": d[5]})

    def test_trailing_exit_baseline_ends_the_bar_before_the_open_fill(self):
        closes, opens = np.full(300, 100.0), np.full(300, 100.0)
        closes[40] = 98.5             # closes below the 99 lows: exit at bar 41's open
        opens[41] = 98.0
        df = bars(closes, opens=opens, lows=np.full(300, 99.0), highs=np.full(300, 101.0))
        d = bt._date_strings(df.index)
        e = entry(signal_date=d[9], entry_date=d[10], stop=90.0, target=150.0)
        out = ee.run_ticker("AAA", "us", [e], df, (), None, SETTINGS)
        recs = {r["variant"]: r for r in out["records"]}
        trail = recs["S1xE3"]
        self.assertEqual((trail["outcome"], trail["exit_date"]), ("trail", d[41]))
        self.assertEqual(trail["baseline_exit_date"], d[40])
        # stop and time exits keep the exit date (as the backtest's baselines do)
        self.assertEqual(recs["S1xE1"]["outcome"], "time")
        self.assertEqual(recs["S1xE1"]["baseline_exit_date"], recs["S1xE1"]["exit_date"])
        self.assertEqual((recs["S1xE2"]["outcome"], recs["S1xE2"]["exit_date"]), ("time", d[130]))
        self.assertEqual(recs["S1xE2"]["baseline_exit_date"], d[130])
        # a censored record has no baseline date at all
        late = entry(signal_date=d[199], entry_date=d[200], stop=90.0, target=150.0,
                     pattern="darvas_box")
        out2 = ee.run_ticker("AAA", "us", [late], df, (), None, SETTINGS)
        cens = [r for r in out2["records"] if r["variant"] == "S1xE2"][0]
        self.assertTrue(cens["censored"])
        self.assertIsNone(cens["baseline_exit_date"])

    def test_counts_are_per_period_and_missing_bars_are_counted(self):
        df = bars(np.full(400, 100.0), start="2022-07-01")
        d = bt._date_strings(df.index)
        k = int(np.searchsorted(d, "2023-01-01"))
        es = [entry(signal_date=d[5], entry_date=d[6], stop=90.0, target=150.0),
              entry(signal_date=d[k + 4], entry_date=d[k + 5], stop=90.0, target=150.0),
              entry(signal_date="2022-08-05", entry_date="2022-08-06", stop=90.0, target=150.0)]
        self.assertEqual([e["period"] for e in es], ["train", "test", "train"])
        out = ee.run_ticker("AAA", "us", es, df, (), None, SETTINGS)
        c = out["counts"]["S1xE1"]
        self.assertEqual(c["train"]["entries"], 2)
        self.assertEqual(c["test"]["entries"], 1)
        self.assertEqual(c["train"]["bar_missing"], 1)      # 2022-08-06 is a Saturday: no bar
        self.assertEqual((c["train"]["trades"], c["test"]["trades"]), (1, 1))
        self.assertEqual({r["period"] for r in out["records"]}, {"train", "test"})


class ControlReproductionTest(unittest.TestCase):
    def test_s1xe1_reproduces_stored_trades(self):
        df = random_frame(21, n=500)
        o, h, l, c = ohlc(df)
        stored = []
        cases = ((60, "next_open"), (150, "buy_stop"), (260, "buy_stop"), (380, "next_open"))
        for fi, kind in cases:
            fill = float(o[fi])
            stored.append(stored_trade(df, fi, fill, fill * 0.96, fill * 1.06, entry_kind=kind))
        entries, _ = ee.entries_from_trades(stored)
        out = ee.run_ticker("AAA", "us", entries, df, (), None, SETTINGS)
        self.assertEqual(out["control_compared"], 4)
        self.assertEqual(out["control_mismatches"], [])
        self.assertEqual(out["price_scale"], 1.0)
        recs = {r["entry_date"]: r for r in out["records"] if r["variant"] == "S1xE1"}
        for t in stored:
            r = recs[t["entry_date"]]
            self.assertEqual((r["exit_date"], r["outcome"]), (t["exit_date"], t["outcome"]))
            # the stored r_net came from the unrounded fill; the replay only has the
            # 4-decimal entry_price the backtest stores, so the last digit can move
            self.assertAlmostEqual(r["r_net"], t["r_net"], delta=2e-4)
            self.assertAlmostEqual(r["ret_net_pct"], t["ret_net_pct"], delta=2e-3)
            stored_exit = round(t["entry_price"] * (1 + t["ret_pct"] / 100), 4)
            self.assertAlmostEqual(r["exit_price"], stored_exit, delta=r["exit_price"] * 1e-4)
            self.assertEqual(r["hold_bars"], t["hold_bars"])
            self.assertEqual(r["target"], round(t["target"], 4))

    def test_a_mismatch_is_reported(self):
        df = random_frame(21, n=200)
        o = df["open"].to_numpy(float)
        t = stored_trade(df, 60, float(o[60]), o[60] * 0.96, o[60] * 1.06)
        t["exit_date"] = "1999-01-01"
        entries, _ = ee.entries_from_trades([t])
        out = ee.run_ticker("AAA", "us", entries, df, (), None, SETTINGS)
        self.assertEqual(out["control_compared"], 1)
        self.assertEqual(len(out["control_mismatches"]), 1)
        self.assertEqual(out["control_mismatches"][0]["stored"]["exit_date"], "1999-01-01")

    def test_rescaled_cache_is_mapped_back(self):
        # the stored trades were made on the old series; today's cache carries a
        # later dividend adjustment (every bar x 0.99)
        old = random_frame(22, n=400)
        o = old["open"].to_numpy(float)
        stored = [stored_trade(old, fi, float(o[fi]), o[fi] * 0.95, o[fi] * 1.08,
                               entry_kind="next_open") for fi in (60, 180, 300)]
        new = old * 0.99
        new["volume"] = old["volume"]
        entries, _ = ee.entries_from_trades(stored)
        out = ee.run_ticker("AAA", "us", entries, new, (), None, SETTINGS)
        self.assertAlmostEqual(out["price_scale"], 0.99, places=5)
        self.assertGreaterEqual(out["scale_estimates"], 3)
        self.assertEqual(out["control_mismatches"], [])
        rec = [r for r in out["records"] if r["variant"] == "S1xE1"][0]
        self.assertAlmostEqual(rec["entry_price"], round(stored[0]["entry_price"] * 0.99, 4),
                               places=3)
        self.assertEqual(rec["ret_pct"], stored[0]["ret_pct"])
        # without an anchor nothing is rescaled (and counted as such)
        no_anchor = entry(entry_kind="buy_stop", entry_level=100.0, outcome="stop", r=-1.0,
                          ret_pct=-5.0, exit_date="2020-01-10")
        f, n = ee.price_scale_factor([no_anchor], {"2020-01-03": 0, "2020-01-10": 5},
                                     np.full(6, 99.0), np.full(6, 99.0))
        self.assertEqual((f, n), (1.0, 0))
        # rounding noise is not a rescale
        f, n = ee.price_scale_factor([entry(entry_kind="next_open", entry_price=100.0)],
                                     {"2020-01-03": 0}, np.array([100.0004]), np.array([100.0]))
        self.assertEqual((f, n), (1.0, 1))


def rec(variant="S1xE1", entry_date="2020-01-02", ret_net=1.0, excess=0.5, outcome="time",
        censored=False, **kw) -> dict:
    r = {"variant": variant, "entry_date": entry_date, "period": ee.period_of(entry_date),
         "ret_net_pct": ret_net, "excess_vs_universe_pct": excess,
         "excess_vs_index_pct": excess + 0.1, "universe_ret_pct": ret_net - excess,
         "index_ret_pct": ret_net - excess - 0.1, "outcome": outcome, "hold_bars": 10,
         "risk_pct": 5.0, "cost_pct_round_trip": 0.3, "r_net": 0.2, "censored": censored,
         "stop": 95.0, "stop_pattern": 95.0, "common": True}
    if censored:
        for k in ("ret_net_pct", "excess_vs_universe_pct", "excess_vs_index_pct", "outcome",
                  "hold_bars", "r_net", "ret_net_pct", "cost_pct_round_trip"):
            r[k] = None
    r.update(kw)
    return r


class StubPanel:
    """MarketPanel stand-in: records the windows asked for."""

    def __init__(self):
        self.asked = []

    def universe_return(self, entry_date, exit_date):
        self.asked.append((entry_date, exit_date))
        return 1.0


class BaselineExitDateTest(unittest.TestCase):
    def test_trail_exits_use_the_previous_bars_close(self):
        idx = random_frame(41, n=60, start="2020-01-01", price=3000.0, vol=0.01)
        bench = bt.BenchContext.from_df(idx)
        d = bt._date_strings(idx.index)
        trail = rec(entry_date=d[5], outcome="trail", exit_date=d[30], baseline_exit_date=d[29],
                    ret_net=2.0)
        time_ = rec(entry_date=d[5], outcome="time", exit_date=d[30], baseline_exit_date=d[30],
                    ret_net=2.0)
        legacy = rec(entry_date=d[5], outcome="stop", exit_date=d[30], ret_net=2.0)  # no field
        panel = StubPanel()
        missing = ee.add_variant_baselines([trail, time_, legacy], panel, bench, 1.5)
        self.assertEqual(missing, 0)
        self.assertEqual(panel.asked, [(d[5], d[29]), (d[5], d[30]), (d[5], d[30])])
        want_trail = bt.index_return(bench.dates, bench.opens, bench.closes, d[5], d[29], 1.5)
        want_time = bt.index_return(bench.dates, bench.opens, bench.closes, d[5], d[30], 1.5)
        self.assertEqual(trail["index_ret_pct"], round(want_trail, 3))
        self.assertEqual(time_["index_ret_pct"], round(want_time, 3))
        self.assertNotEqual(trail["index_ret_pct"], time_["index_ret_pct"])
        self.assertEqual(trail["excess_vs_index_pct"], round(2.0 - want_trail, 3))
        self.assertEqual(trail["excess_vs_universe_pct"], 1.0)


class PeriodStatsTest(unittest.TestCase):
    def test_numbers_and_censored_exclusion(self):
        recs = [rec(ret_net=2.0, excess=1.0, outcome="target"),
                rec(ret_net=-1.0, excess=-2.0, outcome="stop", entry_date="2020-02-03"),
                rec(ret_net=0.5, excess=0.25, outcome="trail", entry_date="2021-02-03"),
                rec(censored=True, entry_date="2021-03-03")]
        counts = {"entries": 6, "trades": 3, "skipped_overlap": 2, "censored": 1, "bar_missing": 0,
                  "atr_fallback": 0, "atr_missing": 0}
        st = ee.period_stats(recs, counts, 2.0)
        self.assertEqual((st["entries"], st["trades"], st["skipped_overlap"], st["censored"]),
                         (6, 3, 2, 1))
        self.assertEqual(st["trades_per_year"], 1.5)
        self.assertEqual(st["win_rate"], round(200 / 3, 1))
        self.assertEqual(st["stop_share"], 33.3)
        self.assertEqual(st["outcomes"], {"stop": 1, "target": 1, "time": 0, "trail": 1})
        self.assertAlmostEqual(st["net_avg_ret_pct"], 0.5)
        self.assertAlmostEqual(st["excess_vs_universe_pct"], -0.25)
        self.assertAlmostEqual(st["excess_vs_index_pct"], -0.15)
        self.assertIsNone(st["excess_vs_universe_ci95"])       # fewer than 10 quarters
        self.assertEqual(sorted(st["by_year"]), ["2020", "2021"])
        self.assertEqual(st["by_year"]["2020"]["trades"], 2)
        self.assertIn("not comparable across stop variants", st["net_avg_r_note"])

    def test_ci_uses_the_backtests_cluster_stats(self):
        rng = np.random.default_rng(0)
        dates = [f"{2017 + k // 4}-{3 * (k % 4) + 1:02d}-15" for k in range(24)]
        recs = [rec(entry_date=d, excess=float(x)) for d, x in zip(dates, rng.normal(1.0, 0.5, 24))]
        counts = {"entries": 24, "trades": 24, "skipped_overlap": 0, "censored": 0,
                  "bar_missing": 0, "atr_fallback": 0, "atr_missing": 0}
        st = ee.period_stats(recs, counts, 6.0)
        want = bt.cluster_stats([r["excess_vs_universe_pct"] for r in recs], dates)
        self.assertEqual(st["excess_vs_universe_ci95"],
                         [round(want["lo"], 3), round(want["hi"], 3)])
        self.assertEqual(st["excess_vs_universe_df"], want["df"])
        self.assertEqual(st["excess_vs_universe_se_basis"], "month/quarter")
        self.assertIsNotNone(st["excess_vs_universe_se_year_cluster"])
        self.assertIsNotNone(st["excess_vs_universe_se_half_year_cluster"])

    def test_long_holds_take_the_coarser_clustering_and_its_df(self):
        # six years, one trade per month; the excess is constant within each
        # half-year and flips sign between halves, so the half-year clustering has
        # the largest SE (quarters within a half share the sign)
        dates, vals = [], []
        for y in range(2017, 2023):
            for m in range(1, 13):
                dates.append(f"{y}-{m:02d}-15")
                vals.append(1.0 + (2.0 if m <= 6 else -2.0) + 0.01 * ((y * 12 + m) % 7))
        recs = [rec(entry_date=d, excess=v) for d, v in zip(dates, vals)]
        counts = {"entries": 72, "trades": 72, "skipped_overlap": 0, "censored": 0,
                  "bar_missing": 0, "atr_fallback": 0, "atr_missing": 0}
        base = bt.cluster_stats(vals, dates)
        hy = bt.cluster_se(vals, [ee._half_year(d) for d in dates])
        yr = bt.cluster_se(vals, [d[:4] for d in dates])
        self.assertGreater(hy[1], base["se"])
        self.assertLess(yr[1], hy[1])           # the yearly sums cancel: a smaller SE
        # E1 (40 bars): the backtest's month/quarter range, quarters - 1 df
        e1 = ee.period_stats(recs, counts, 6.0, max_hold=40)
        self.assertEqual(e1["excess_vs_universe_se_basis"], "month/quarter")
        self.assertEqual(e1["excess_vs_universe_df"], 23)
        self.assertEqual(e1["excess_vs_universe_ci95"], [round(base["lo"], 3), round(base["hi"], 3)])
        # E2 (120 bars): the half-year SE binds, with 12 half-years -> df 11
        e2 = ee.period_stats(recs, counts, 6.0, max_hold=120)
        self.assertEqual(e2["excess_vs_universe_se_basis"], "half-year")
        self.assertEqual(e2["excess_vs_universe_df"], 11)
        self.assertEqual(e2["excess_vs_universe_se"], round(hy[1], 4))
        crit = bt.t_crit(0.975, 11)
        mean = sum(vals) / len(vals)
        self.assertEqual(e2["excess_vs_universe_ci95"],
                         [round(mean - crit * hy[1], 3), round(mean + crit * hy[1], 3)])
        self.assertGreater(e2["excess_vs_universe_ci95"][1] - e2["excess_vs_universe_ci95"][0],
                           e1["excess_vs_universe_ci95"][1] - e1["excess_vs_universe_ci95"][0])
        # E3 (250 bars) also considers the year, which is smaller here: half-year stays
        e3 = ee.period_stats(recs, counts, 6.0, max_hold=250)
        self.assertEqual(e3["excess_vs_universe_se_basis"], "half-year")
        self.assertEqual(e3["excess_vs_universe_se_year_cluster"], round(yr[1], 4))
        # a year-clustered pattern (constant within the year, flipping between years)
        # binds for E3 with years - 1 df, but not for E2 (year is not considered)
        vals_y = [1.0 + (2.0 if int(d[:4]) % 2 else -2.0) + 0.01 * (k % 7)
                  for k, d in enumerate(dates)]
        recs_y = [rec(entry_date=d, excess=v) for d, v in zip(dates, vals_y)]
        yr_y = bt.cluster_se(vals_y, [d[:4] for d in dates])
        e3y = ee.period_stats(recs_y, counts, 6.0, max_hold=250)
        self.assertEqual(e3y["excess_vs_universe_se_basis"], "year")
        self.assertEqual(e3y["excess_vs_universe_df"], 5)
        self.assertEqual(e3y["excess_vs_universe_se"], round(yr_y[1], 4))
        e2y = ee.period_stats(recs_y, counts, 6.0, max_hold=120)
        self.assertIn(e2y["excess_vs_universe_se_basis"], ("month/quarter", "half-year"))
        # fewer than MIN_CLUSTERS coarser clusters: the coarser clustering does not count
        short = [d for d in dates if d < "2020-01-01"]          # 3 years, 6 half-years
        vals_s = [1.0 + (2.0 if int(d[:4]) % 2 else -2.0) + 0.01 * (k % 7)
                  for k, d in enumerate(short)]
        recs_s = [rec(entry_date=d, excess=v) for d, v in zip(short, vals_s)]
        e3s = ee.period_stats(recs_s, counts, 3.0, max_hold=250)
        self.assertNotEqual(e3s["excess_vs_universe_se_basis"], "year")

    def test_pre_registration_diagnostics(self):
        recs = [rec(stop=96.0, stop_pattern=95.0, risk_pct=4.0),            # ATR stop tighter
                rec(stop=90.0, stop_pattern=95.0, risk_pct=10.0, entry_date="2020-02-03"),
                rec(stop=60.0, stop_pattern=95.0, risk_pct=40.0, entry_date="2020-03-03"),
                rec(outcome="trail", hold_bars=2, entry_date="2020-04-03"),
                rec(outcome="trail", hold_bars=20, entry_date="2020-05-04"),
                rec(outcome="trail", hold_bars=35, entry_date="2020-06-03"),
                rec(censored=True, entry_date="2020-07-03", stop=99.0, stop_pattern=95.0,
                    risk_pct=1.0)]
        counts = {"entries": 7, "trades": 6, "skipped_overlap": 0, "censored": 1, "bar_missing": 0,
                  "atr_fallback": 0, "atr_missing": 0}
        st = ee.period_stats(recs, counts, 1.0, max_hold=250)
        self.assertEqual(st["atr_stop_tighter_than_pattern"], {"count": 1, "share_pct": 16.7})
        self.assertEqual(st["risk_over_30_pct"], 1)
        self.assertEqual(st["trail_exits"], 3)
        self.assertEqual(st["trail_exits_within_20_bars"], {"count": 2, "share_pct": 66.7})
        self.assertEqual(st["trail_exits_on_bar_2"], 1)
        # no trail exits: a count of 0 and no share
        st1 = ee.period_stats(recs[:3], {**counts, "entries": 3, "trades": 3, "censored": 0}, 1.0)
        self.assertEqual(st1["trail_exits_within_20_bars"], {"count": 0, "share_pct": None})


def variant_block(train_ex, train_lo, test_ex=None, test_lo=None, test_cagr=None) -> dict:
    ci_tr = None if train_lo is None else [train_lo, train_ex + (train_ex - train_lo)]
    ci_te = None if test_lo is None else [test_lo, (test_ex or 0) + ((test_ex or 0) - test_lo)]
    return {"train": {"excess_vs_universe_pct": train_ex, "excess_vs_universe_ci95": ci_tr},
            "test": {"excess_vs_universe_pct": test_ex, "excess_vs_universe_ci95": ci_te},
            "pooled": {}, "portfolio": {"test": {"cagr_pct": test_cagr}, "pooled": {}}}


class WinnerRuleTest(unittest.TestCase):
    def test_highest_significant_train_excess_wins(self):
        v = {"S1xE1": variant_block(0.5, 0.1, 0.2, -0.1, 3.0),
             "S2xE2": variant_block(2.0, -0.5, 1.0, 0.2, 5.0),   # highest, but its range includes 0
             "S3xE3": variant_block(1.5, 0.3, 0.8, 0.1, 4.0)}
        d = ee.decide(v)
        self.assertEqual(d["winner"]["variant"], "S3xE3")
        self.assertTrue(d["winner"]["significant_in_train"])
        self.assertIsNone(d["flag"])
        self.assertEqual(d["control"]["variant"], "S1xE1")
        self.assertEqual(d["verdict"], "confirmed")
        # the best TEST variant is named and labelled in-sample
        self.assertEqual(d["best_test_variant"]["variant"], "S2xE2")
        self.assertIn("do not act", d["best_test_variant"]["label"])

    def test_fallback_is_flagged(self):
        v = {"S1xE1": variant_block(0.5, -0.1, 0.2, -0.1, 3.0),
             "S2xE2": variant_block(2.0, -0.5, 1.0, -0.2, 5.0),
             "S3xE3": variant_block(1.5, None, 0.8, None, 4.0)}
        d = ee.decide(v)
        self.assertEqual(d["winner"]["variant"], "S2xE2")
        self.assertEqual(d["flag"], "no variant is significantly positive in training")
        self.assertFalse(d["winner"]["significant_in_train"])
        self.assertEqual(d["verdict"], "improves but unproven")

    def test_verdicts(self):
        ctl = variant_block(0.5, 0.1, 0.2, -0.1, 3.0)
        # TEST range above 0 but the CAGR does not beat the control's
        d = ee.decide({"S1xE1": ctl, "S2xE2": variant_block(1.0, 0.2, 0.9, 0.1, 2.0)})
        self.assertEqual(d["verdict"], "not confirmed")
        self.assertIn("CAGR leg fails", d["verdict_reason"])
        self.assertIn("no seed-paired replay", d["verdict_reason"])
        self.assertIsNone(d["paired_test_cagr"])
        # TEST excess below the control's
        d = ee.decide({"S1xE1": ctl, "S2xE2": variant_block(1.0, 0.2, 0.1, -0.5, 4.0)})
        self.assertEqual(d["verdict"], "not confirmed")
        # beats the control's TEST excess, but the whole TEST range lies below 0: not the
        # 'includes 0' case, so not 'improves but unproven'
        d = ee.decide({"S1xE1": variant_block(0.5, 0.1, -2.0, -3.0, 3.0),
                       "S2xE2": variant_block(1.0, 0.2, -1.0, -1.8, 4.0)})
        self.assertEqual(d["verdict"], "not confirmed")
        self.assertIn("below 0", d["verdict_reason"])
        # no TEST range at all (fewer than 10 quarters) counts as 'includes 0'
        d = ee.decide({"S1xE1": ctl, "S2xE2": variant_block(1.0, 0.2, 0.9, None, 4.0)})
        self.assertEqual(d["verdict"], "improves but unproven")
        self.assertIn("no range", d["verdict_reason"])
        # the control wins training: nothing to confirm
        d = ee.decide({"S1xE1": variant_block(1.0, 0.2, 0.5, 0.1, 3.0),
                       "S2xE2": variant_block(0.4, 0.1, 0.9, 0.2, 4.0)})
        self.assertEqual(d["winner"]["variant"], "S1xE1")
        self.assertEqual(d["verdict"], "not confirmed")
        self.assertIn("control itself", d["verdict_reason"])
        self.assertEqual(d["best_test_variant"]["variant"], "S2xE2")
        # no statistics at all
        d = ee.decide({"S1xE1": variant_block(None, None)})
        self.assertEqual(d["verdict"], "not confirmed")
        self.assertIsNone(d["winner"])

    def test_cagr_leg_is_paired_by_seed(self):
        ctl = variant_block(0.5, 0.1, 0.2, -0.1, 3.0)
        win = variant_block(1.0, 0.2, 0.9, 0.1, 3.5)       # median CAGR beats the control's
        paired = {"seeds": 101, "winner_beats_control_share": 0.62, "required_share": 0.9,
                  "winner_cagr_range": {}, "control_cagr_range": {}}
        d = ee.decide({"S1xE1": ctl, "S2xE2": win}, paired=paired)
        self.assertEqual(d["verdict"], "not confirmed")   # a 3.5 vs 3.0 median is slot luck
        self.assertIn("62%", d["verdict_reason"])
        self.assertIn("needs 90%", d["verdict_reason"])
        self.assertIs(d["paired_test_cagr"], paired)
        d = ee.decide({"S1xE1": ctl, "S2xE2": win},
                      paired={**paired, "winner_beats_control_share": 0.9})
        self.assertEqual(d["verdict"], "confirmed")
        # the paired share rules even when the medians point the other way
        low_median = variant_block(1.0, 0.2, 0.9, 0.1, 2.9)
        d = ee.decide({"S1xE1": ctl, "S2xE2": low_median},
                      paired={**paired, "winner_beats_control_share": 0.95})
        self.assertEqual(d["verdict"], "confirmed")
        self.assertEqual(ee.pick_winner({"S1xE1": ctl, "S2xE2": win}), ("S2xE2", None))


AS_DESIGNED = {"risk_pct": 1.0, "max_pos": 6, "sizing": "dashboard", "max_position_pct": 25.0,
               "cash_limit": True, "rank_by": "composite_proxy", "grade_filter": None}


class PortfolioSettingsTest(unittest.TestCase):
    def test_as_designed_settings(self):
        # no backtest file: the profile's definition with the config's position cap
        self.assertEqual(ee.as_designed_settings(None, {"max_position_pct": 25.0}), AS_DESIGNED)
        self.assertEqual(ee.as_designed_settings({}, {})["max_position_pct"],
                         ee.DEFAULT_MAX_POSITION_PCT)
        # the backtest file's own 'as_designed' profile wins over the config
        profile = {"key": "as_designed", "risk_pct": 1.0, "max_pos": 6, "max_position_pct": 20.0,
                   "sizing": "dashboard", "cash_limit": True, "rank_by": "composite_proxy",
                   "grade_filter": None}
        got = ee.as_designed_settings({"portfolio": {"profiles": [profile]}},
                                      {"max_position_pct": 25.0})
        self.assertEqual(got, {**AS_DESIGNED, "max_position_pct": 20.0})
        # any other profile is ignored, whatever its position in the list
        aggressive = {**profile, "key": "aggressive", "risk_pct": 2.0, "max_pos": 3,
                      "max_position_pct": 40.0}
        got = ee.as_designed_settings({"portfolio": {"profiles": [aggressive]}},
                                      {"max_position_pct": 25.0})
        self.assertEqual(got, AS_DESIGNED)
        got = ee.as_designed_settings({"portfolio": {"profiles": [aggressive, profile]}},
                                      {"max_position_pct": 25.0})
        self.assertEqual(got, {**AS_DESIGNED, "max_position_pct": 20.0})


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def pf_block(window, last_exit, cagr, vs_idx, own=None) -> dict:
    """A portfolio block as portfolio_replay writes it: window, last exit, cash
    tail, CAGR and index shortfall; own = ((start, end), cagr, vs_idx)."""
    blk = {"window": {"start": window[0], "end": window[1]}, "last_exit_date": last_exit,
           "cash_tail_days": days_between(last_exit, window[1]), "cagr_pct": cagr,
           "vs_blended_index_pp": vs_idx}
    if own:
        blk["own_window"] = {"window": {"start": own[0][0], "end": own[0][1]},
                             "cagr_pct": own[1], "vs_blended_index_pp": own[2]}
    return blk


class CashTailTest(unittest.TestCase):
    """The shared window's cash tail: measured per block, given an outcome-
    independent own window per exit rule, and stated with its direction."""

    SHARED = ("2023-01-02", "2026-08-03")             # the full run's TEST window: 1309 days

    def control_block(self):
        return pf_block(self.SHARED, "2025-11-26", 11.2, -6.3,
                        own=(("2023-01-02", "2025-12-01"), 14.0, -5.5))

    def test_last_exit_and_tail(self):
        recs = [rec(exit_date="2024-03-01"),
                rec(exit_date="2024-06-14", entry_date="2024-05-01"),
                rec(censored=True, exit_date=None, entry_date="2024-07-01")]
        self.assertEqual(ee.last_exit_and_tail(recs, ("2024-01-02", "2024-12-31")),
                         {"last_exit_date": "2024-06-14", "cash_tail_days": 200})
        # a window ending before the last exit gives a negative tail; no closed
        # records or no window: None
        early_end = ee.last_exit_and_tail(recs, ("2024-01-02", "2024-06-04"))
        self.assertEqual(early_end["cash_tail_days"], -10)
        self.assertEqual(ee.last_exit_and_tail([recs[2]], ("2024-01-02", "2024-12-31")),
                         {"last_exit_date": None, "cash_tail_days": None})
        self.assertEqual(ee.last_exit_and_tail(recs, None),
                         {"last_exit_date": "2024-06-14", "cash_tail_days": None})

    def test_own_windows_follow_the_benchmark_calendar(self):
        dense = [str(d)[:10] for d in pd.bdate_range("2020-01-01", periods=400)]
        sparse = dense[::2]                 # a sparser calendar reaches further per bar
        ents = [{"entry_date": d, "period": ee.period_of(d)}
                for d in (dense[20], dense[60], dense[100])]
        own = ee.own_windows(ents, {"us": dense, "in": sparse})
        k = sparse.index(dense[100])
        for ek, rule in ee.EXITS.items():
            want_end = sparse[min(k + rule["max_hold"], len(sparse) - 1)]
            self.assertEqual(own[ek]["train"], (dense[20], want_end))
            self.assertEqual(own[ek]["pooled"], (dense[20], want_end))
            self.assertIsNone(own[ek]["test"])                       # no test entries
            # the sparsest calendar's end is never before the dense one's
            self.assertGreaterEqual(want_end, dense[min(100 + rule["max_hold"], 399)])
        self.assertLess(own["E1"]["train"][1], own["E2"]["train"][1])
        # 250 sparse bars run past the data: capped at the last benchmark bar
        self.assertEqual(own["E3"]["train"][1], sparse[-1])
        # a last entry between two benchmark bars counts from the bar on/before it; the
        # end comes from the calendar, never from realized exits
        k0 = sparse.index(max(d for d in sparse if d <= "2020-02-01"))      # a Saturday
        self.assertEqual(ee.own_windows([{"entry_date": "2020-02-01", "period": "train"}],
                                        {"in": sparse})["E1"]["train"],
                         ("2020-02-01", sparse[k0 + 40]))
        self.assertEqual(ee.own_windows([], {"us": dense})["E1"], {p: None for p in ee.PERIODS})
        self.assertIsNone(ee.own_windows(ents, {})["E2"]["train"])

    def test_cash_tail_note_states_direction_and_numbers(self):
        pf = self.control_block()
        self.assertEqual(pf["cash_tail_days"], 250)
        note = ee.cash_tail_note(pf)
        for part in ("last 250 days", "19.1%", "2025-11-26", "not the rule",
                     "understates its CAGR", "overstates its shortfall vs the index",
                     "both reverse when negative", "CAGR 11.2% shared vs 14.0%",
                     "2023-01-02..2025-12-01", "-6.3 vs -5.5 pp"):
            self.assertIn(part, note)
        self.assertNotIn("seed pairs", note)
        with_pair = ee.cash_tail_note(pf, {"own_window": {"winner_beats_control_share": 0.42}})
        self.assertIn("ahead in 42% of the seed pairs", with_pair)
        # a tail of 10% of the window or less is not flagged; nor an unknown one
        small = pf_block(self.SHARED, "2026-04-15", 11.2, -6.3)
        self.assertLessEqual(small["cash_tail_days"], ee.CASH_TAIL_FLAG_SHARE * 1309)
        self.assertIsNone(ee.cash_tail_note(small))
        self.assertIsNone(ee.cash_tail_note({"cagr_pct": 1.0}))
        self.assertIsNone(ee.cash_tail_note(None))
        # the clause for the method notes carries the same numbers
        text = ee.tail_summary_text(pf, "test")
        for part in ("TEST:", "last exit 2025-11-26", "cash tail 250 days = 19.1%",
                     "2023-01-02..2026-08-03", "CAGR 11.2% shared vs 14.0%", "-6.3 vs -5.5 pp"):
            self.assertIn(part, text)
        self.assertEqual(ee.tail_summary_text(None, "train"), "TRAIN: control cash tail n/a")
        self.assertIn("n/a", ee.tail_summary_text(pf_block(self.SHARED, "2025-11-26", None, None),
                                                  "test"))

    def test_decide_appends_the_controls_tail_to_the_reason(self):
        ctl = variant_block(0.5, 0.1, 0.2, -0.1, 11.2)
        ctl["portfolio"]["test"] = self.control_block()
        win = variant_block(1.0, 0.2, 0.9, 0.1, 12.0)
        paired = {"seeds": 101, "winner_beats_control_share": 0.95, "required_share": 0.9,
                  "winner_cagr_range": {}, "control_cagr_range": {},
                  "own_window": {"winner_beats_control_share": 0.6}}
        d = ee.decide({"S1xE1": ctl, "S3xE3": win}, paired=paired)
        self.assertEqual(d["verdict"], "confirmed")            # the rule itself is unchanged
        self.assertIn("CAGR leg holds", d["verdict_reason"])
        self.assertIn("note: the control is in cash for the last 250 days", d["verdict_reason"])
        self.assertIn("ahead in 60% of the seed pairs", d["verdict_reason"])
        # a small tail: no note
        small = {**ctl, "portfolio": {"test": pf_block(self.SHARED, "2026-04-15", 11.2, -6.3),
                                      "pooled": {}}}
        d = ee.decide({"S1xE1": small, "S3xE3": win}, paired=paired)
        self.assertEqual(d["verdict"], "confirmed")
        self.assertNotIn("note:", d["verdict_reason"])
        # the control as the train winner: nothing to read the tail against
        top = {**ctl, "train": {"excess_vs_universe_pct": 5.0,
                                "excess_vs_universe_ci95": [1.0, 9.0]}}
        d = ee.decide({"S1xE1": top, "S3xE3": win}, paired=paired)
        self.assertEqual(d["winner"]["variant"], "S1xE1")
        self.assertNotIn("note:", d["verdict_reason"])


def sim_record(ticker, entry_date, exit_date, r_net, fill=100.0, stop=95.0, market="us",
               pattern="bull_flag", censored=False) -> dict:
    return {"ticker": ticker, "market": market, "pattern": pattern, "entry_date": entry_date,
            "exit_date": exit_date, "outcome": "time", "r_net": r_net, "r": r_net + 0.01,
            "entry_price": fill, "entry_level": fill, "stop": stop, "conf": 0.8,
            "composite_proxy": 70.0, "grade_proxy": "B", "censored": censored}


class PairedCagrTest(unittest.TestCase):
    def test_same_seed_both_sides_and_share(self):
        days = [str(d)[:10] for d in pd.bdate_range("2023-01-02", periods=300)]
        rng = np.random.default_rng(3)
        win, ctl = [], []
        for k in range(0, 280, 4):
            for j in range(3):                          # three candidates per day: slot luck
                t = f"T{j}"
                win.append(sim_record(t, days[k], days[k + 3], float(rng.normal(0.4, 0.5))))
                ctl.append(sim_record(t, days[k], days[k + 3], float(rng.normal(0.1, 0.5))))
        ctl.append(sim_record("X", days[1], days[2], 0.0, censored=True))
        kw = {"risk_pct": 1.0, "max_pos": 1, "sizing": "dashboard", "max_position_pct": 25.0,
              "cash_limit": True, "rank_by": "composite_proxy", "grade_filter": None,
              "regime": None}
        window = (days[0], days[290])
        p = ee.paired_cagr(win, ctl, window, kw, reshuffles=10)
        self.assertEqual(p["seeds"], 11)
        self.assertEqual(p["window"], {"start": days[0], "end": days[290]})
        self.assertEqual(p["required_share"], ee.PAIRED_CAGR_SHARE)
        # reproduce one pair by hand: the same seed on both sides
        from atlas2.portfolio_sim import simulate_portfolio
        w_raw = [simulate_portfolio(win, tie_seed=sd, **{**kw, "window": window})["_cagr_raw"]
                 for sd in range(11)]
        c_raw = [simulate_portfolio([t for t in ctl if not t["censored"]], tie_seed=sd,
                                    **{**kw, "window": window})["_cagr_raw"] for sd in range(11)]
        share = sum(1 for a, b in zip(w_raw, c_raw) if a > b) / 11
        self.assertEqual(p["winner_beats_control_share"], round(share, 3))
        self.assertEqual(p["winner_cagr_range"]["median"], round(float(np.median(w_raw)), 1))
        self.assertEqual(p["control_cagr_range"]["median"], round(float(np.median(c_raw)), 1))
        self.assertIn("p5", p["winner_minus_control_pp"])
        # no closed trades on one side: nothing to pair
        self.assertIsNone(ee.paired_cagr(win, [ctl[-1]], window, kw, 2))
        # the shared window: each side's last exit and the cash tail it leaves
        self.assertEqual(p["winner_last_exit_date"], days[279])
        self.assertEqual(p["control_last_exit_date"], days[279])
        self.assertEqual(p["winner_cash_tail_days"], days_between(days[279], days[290]))
        self.assertNotIn("own_window", p)                      # none asked for

    def test_own_windows_annualise_each_side_over_its_own_span(self):
        from atlas2.portfolio_sim import START_CAPITAL, simulate_portfolio
        days = [str(d)[:10] for d in pd.bdate_range("2023-01-02", periods=300)]
        rng = np.random.default_rng(5)
        win = [sim_record("T", days[k], days[k + 3], float(rng.normal(0.3, 0.5)))
               for k in range(0, 200, 4)]
        ctl = [sim_record("T", days[k], days[k + 3], float(rng.normal(0.5, 0.5)))
               for k in range(0, 120, 4)]
        kw = {"risk_pct": 1.0, "max_pos": 1, "sizing": "dashboard", "max_position_pct": 25.0,
              "cash_limit": True, "rank_by": "composite_proxy", "grade_filter": None,
              "regime": None}
        window = (days[0], days[290])
        ww, cw = (days[0], days[220]), (days[0], days[140])
        p = ee.paired_cagr(win, ctl, window, kw, reshuffles=4, own_windows=(ww, cw))
        self.assertEqual(p["winner_last_exit_date"], days[199])
        self.assertEqual(p["control_last_exit_date"], days[119])
        self.assertEqual(p["control_cash_tail_days"], days_between(days[119], days[290]))
        own = p["own_window"]
        self.assertEqual(own["seeds"], 5)
        self.assertEqual(own["winner_window"], {"start": days[0], "end": days[220]})
        self.assertEqual(own["control_window"], {"start": days[0], "end": days[140]})
        self.assertEqual(own["control_last_exit_date"], days[119])
        self.assertEqual(own["control_cash_tail_days"], days_between(days[119], days[140]))
        self.assertIn("own window", own["note"])
        # by hand: the same seed on both sides, each annualised over ITS window
        w_raw = [simulate_portfolio(win, tie_seed=sd, **{**kw, "window": ww})["_cagr_raw"]
                 for sd in range(5)]
        c_raw = [simulate_portfolio(ctl, tie_seed=sd, **{**kw, "window": cw})["_cagr_raw"]
                 for sd in range(5)]
        self.assertEqual(own["winner_beats_control_share"],
                         round(sum(1 for a, b in zip(w_raw, c_raw) if a > b) / 5, 3))
        self.assertEqual(own["winner_cagr_range"]["median"], round(float(np.median(w_raw)), 1))
        self.assertEqual(own["control_cagr_range"]["median"], round(float(np.median(c_raw)), 1))
        # the window changes only the annualisation: the same equity over fewer years
        # lifts a positive CAGR (and deepens a negative one)
        shared = simulate_portfolio(ctl, tie_seed=0, **{**kw, "window": window})
        short = simulate_portfolio(ctl, tie_seed=0, **{**kw, "window": cw})
        self.assertEqual(shared["final_equity"], short["final_equity"])
        self.assertGreater(short["final_equity"], START_CAPITAL)
        self.assertGreater(short["_cagr_raw"], shared["_cagr_raw"])
        self.assertGreater(own["control_cagr_range"]["median"], p["control_cagr_range"]["median"])
        # without both own windows there is no own_window block
        self.assertNotIn("own_window", ee.paired_cagr(win, ctl, window, kw, 2))
        self.assertNotIn("own_window",
                         ee.paired_cagr(win, ctl, window, kw, 2, own_windows=(ww, None)))


class CliTest(unittest.TestCase):
    def test_parse_opts(self):
        self.assertEqual(ee.parse_opts([]),
                         {"limit": None, "out": None, "trades": None, "workers": None})
        self.assertEqual(ee.parse_opts(["--limit", "300", "--out=/tmp/x.json", "--workers", "2"]),
                         {"limit": 300, "out": "/tmp/x.json", "trades": None, "workers": 2})
        for bad in (["--limit"], ["--limit", "0"], ["--limit", "x"], ["--foo"], ["us"]):
            with self.assertRaises(ValueError):
                ee.parse_opts(bad)

    def test_limit_never_writes_the_real_output(self):
        # the guard must fire with its own message, before any input is read (a later
        # 'trade stream unreadable' ValueError would not do)
        guard = "never write the real output"
        with self.assertRaisesRegex(ValueError, guard):
            ee.run_exit_experiment(limit=5, out_path=ee.DEFAULT_OUT_PATH,
                                   trades_path=Path("/nonexistent/trades.json"))
        # nor the real per-trade file, whether named as --out or reached as its _trades twin
        with self.assertRaisesRegex(ValueError, guard):
            ee.run_exit_experiment(limit=5, out_path=ee.ROOT / "data" / "exit_experiment_trades.json",
                                   trades_path=Path("/nonexistent/trades.json"))
        with self.assertRaisesRegex(ValueError, guard):
            ee.run_exit_experiment(limit=5, out_path=ee.trades_output_path(ee.DEFAULT_OUT_PATH),
                                   trades_path=Path("/nonexistent/trades.json"))
        # a smoke run to another path is not stopped by the guard
        with self.assertRaisesRegex(ValueError, "trade stream unreadable"):
            ee.run_exit_experiment(limit=5, out_path=Path("/nonexistent/out/smoke.json"),
                                   trades_path=Path("/nonexistent/trades.json"), cfg={})
        self.assertEqual(ee.trades_output_path(Path("/x/exit_experiment.json")),
                         Path("/x/exit_experiment_trades.json"))
        self.assertEqual(ee.trades_output_path(Path("/x/exit_experiment_smoke.json")),
                         Path("/x/exit_experiment_smoke_trades.json"))

    def test_main_reports_a_bad_option(self):
        with self.assertRaises(SystemExit) as cm:
            ee.main(["--bogus"])
        self.assertEqual(cm.exception.code, 1)


class EndToEndTest(unittest.TestCase):
    """The driver on a synthetic cache: two tickers, an index, stored trades made
    with the backtest's own rules; workers 1 and 2 give identical files."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data = root / "data"
        cls.data.mkdir()
        (root / "universe").mkdir()
        n = 2300                     # bars to 2025-10: the TEST period has common entries
        frames = {"AAA": random_frame(31, n=n, start="2017-01-02"),
                  "BBB": random_frame(32, n=n, start="2017-01-02", price=30.0),
                  "^GSPC": random_frame(33, n=n, start="2017-01-02", price=3000.0, vol=0.01)}
        cls.frames = frames
        store = PriceStore(cls.data / "cache" / "prices.sqlite3")
        for t, df in frames.items():
            store.upsert(t, df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                               "close": "Close", "volume": "Volume"}))
        store.close()
        (root / "universe" / "us.csv").write_text("ticker,name\nAAA,A\nBBB,B\n")
        trades = []
        for t in ("AAA", "BBB"):
            df = frames[t]
            o = df["open"].to_numpy(float)
            for fi in range(300, n - 60, 45):
                fill = float(o[fi])
                kind = "next_open" if fi % 90 == 30 else "buy_stop"
                st = stored_trade(df, fi, fill, fill * 0.95, fill * 1.09, entry_kind=kind)
                st["ticker"] = t
                trades.append(st)
        cls.trades_path = cls.data / "trades.json"
        cls.trades_path.write_text(json.dumps(trades))
        cls.n_trades = len(trades)
        cls.dates = bt._date_strings(frames["AAA"].index)
        cls.common_cutoff = cls.dates[n - 1 - ee.LONGEST_HOLD]
        cls.cfg = {"max_position_pct": 25.0}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_it(self, out: Path, workers: int, limit=None) -> dict:
        buf = io.StringIO()
        with redirect_stdout(buf):
            payload = ee.run_exit_experiment(self.trades_path, out, limit=limit, workers=workers,
                                             prices_path=self.data / "cache" / "prices.sqlite3",
                                             universe_dir=Path(self.tmp.name) / "universe",
                                             cfg=self.cfg, backtest_path=self.data / "none.json",
                                             reshuffles=2)
        self.assertIn("VERDICT:", buf.getvalue())
        self.assertIn("COMMON entry set", buf.getvalue())
        return payload

    def test_run_and_worker_independence(self):
        out1 = Path(self.tmp.name) / "one.json"
        p1 = self.run_it(out1, 1)
        rp = p1["method"]["control_reproduction"]
        self.assertEqual(rp["compared"], self.n_trades)
        self.assertEqual(rp["mismatch_count"], 0)
        self.assertEqual(rp["match_rate_pct"], 100.0)
        trades = json.loads(ee.trades_output_path(out1).read_text())
        # recomputed as the driver does: the common entry dates (the control never skips
        # on the stored stream, so its common records span them) give the
        # trades_per_year denominator; the shared and own windows come from the method
        common_dates = sorted({r["entry_date"] for r in trades
                               if r["variant"] == "S1xE1" and r["common"]})
        years_c = ee.period_years([{"entry_date": d, "period": ee.period_of(d)}
                                   for d in common_dates])
        windows = p1["method"]["split"]["windows"]
        own_w = p1["method"]["split"]["own_windows"]
        self.assertEqual(p1["method"]["portfolio"]["settings"], AS_DESIGNED)
        self.assertEqual(sorted(own_w), sorted(ee.EXITS))
        dates = list(self.dates)

        def counts_from(st: dict) -> dict:
            return {"entries": st["entries"], "trades": st["trades"],
                    "skipped_overlap": st["skipped_overlap"], "censored": st["censored"],
                    "bar_missing": st["bar_missing"], "atr_fallback": st["atr_stop_fallback"],
                    "atr_missing": st["atr_missing"]}

        def span_years(start: str, end: str) -> float:
            return round(max(0.5, days_between(start, end) / 365.25), 1)

        years_seen = {p: set() for p in ee.PERIODS}
        tails = {p: [] for p in ee.PERIODS}
        for v in ee.VARIANTS:
            blk = p1["variants"][v]
            mine = [r for r in trades if r["variant"] == v]
            for p in ee.PERIODS:
                st = blk[p]
                accounted = (st["trades"] + st["skipped_overlap"] + st["censored"]
                             + st["bar_missing"])
                self.assertEqual(st["entries"], accounted)
                for k in ("win_rate", "stop_share", "avg_hold_bars", "net_avg_ret_pct",
                          "avg_universe_ret_pct", "excess_vs_universe_pct", "excess_vs_index_pct",
                          "net_avg_r", "trades_per_year", "by_year", "excess_vs_universe_se_basis",
                          "atr_stop_tighter_than_pattern", "risk_over_30_pct",
                          "trail_exits_within_20_bars", "trail_exits_on_bar_2"):
                    self.assertIn(k, st)
                pf = blk["portfolio"][p]
                self.assertIn("cagr_pct", pf)
                self.assertIn("vs_blended_index_pp", pf)
                # the replay of a period is fed exactly that period's common closed trades
                in_period = [r for r in mine if r["common"] and not r["censored"]
                             and (p == "pooled" or r["period"] == p)]
                self.assertEqual(st["trades"], len(in_period))
                self.assertEqual(pf["trades_available"], st["trades"])
                self.assertEqual(pf["trades_available"], st["trades"])
                # headline blocks = common set, censored 0 by construction; the variant's
                # own eligible window has at least as many trades
                self.assertEqual(st["censored"], 0)
                fw = blk["full_window"][p]
                self.assertGreaterEqual(fw["trades"], st["trades"])
                self.assertGreaterEqual(fw["entries"], st["entries"])
                # the headline block is period_stats on exactly these records, its 95%
                # range clustered by THIS variant's hold and its trades_per_year divided
                # by the common entry span (not the rounded years_common)
                common_recs = [r for r in mine if r["common"]
                               and (p == "pooled" or r["period"] == p)]
                want = ee.period_stats(common_recs, counts_from(st), years_c[p],
                                       max_hold=ee.EXITS[v[3:]]["max_hold"])
                self.assertEqual(st["excess_vs_universe_ci95"], want["excess_vs_universe_ci95"])
                self.assertEqual(st["excess_vs_universe_se_basis"],
                                 want["excess_vs_universe_se_basis"])
                self.assertEqual(st["excess_vs_universe_df"], want["excess_vs_universe_df"])
                self.assertEqual(st["trades_per_year"], round(st["trades"] / years_c[p], 1))
                self.assertEqual(st, want)
                # the replay is annualised over the shared window as actually used
                w = windows[p]
                self.assertEqual(pf["window"], {"start": w[0], "end": w[1]})
                self.assertEqual(pf["years"], span_years(w[0], w[1]))
                self.assertAlmostEqual(pf["years"], p1["benchmark"][p]["us"]["years"], delta=0.06)
                years_seen[p].add(pf["years"])
                # the variant's last exit on that window and the cash tail after it
                last_exit = max(r["exit_date"] for r in in_period)
                self.assertEqual(pf["last_exit_date"], last_exit)
                self.assertEqual(pf["cash_tail_days"], days_between(last_exit, w[1]))
                self.assertGreaterEqual(pf["cash_tail_days"], 0)
                tails[p].append(pf["cash_tail_days"])
                # the exit rule's own window: first common entry to the last common entry
                # + max_hold benchmark bars (the synthetic index shares the stocks'
                # calendar), capped at the last bar; the same trades, settings and seeds
                # as the shared replay, so only the annualisation differs
                ek = v[3:]
                last_entry = max(d for d in common_dates if p == "pooled" or ee.period_of(d) == p)
                k = dates.index(last_entry)
                own_end = dates[min(k + ee.EXITS[ek]["max_hold"], len(dates) - 1)]
                self.assertEqual(own_w[ek][p], [w[0], own_end])
                ow = pf["own_window"]
                self.assertEqual(ow["window"], {"start": w[0], "end": own_end})
                self.assertEqual(ow["years"], span_years(w[0], own_end))
                self.assertEqual(ow["final_equity"], pf["final_equity"])
                self.assertEqual(ow["trades_taken"], pf["trades_taken"])
                self.assertEqual(ow["last_exit_date"], last_exit)
                self.assertEqual(ow["cash_tail_days"], days_between(last_exit, own_end))
                self.assertNotIn("max_drawdown_pct", ow)
                ob = p1["benchmark"][p]["own_windows"][ek]
                self.assertEqual(ob["window"], ow["window"])
                self.assertEqual(ow["blended_index_cagr_pct"], ob["blended_index_cagr_pct"])
                self.assertAlmostEqual(ob["us"]["years"], ow["years"], delta=0.06)
            self.assertEqual(blk["train"]["entries"] + blk["test"]["entries"],
                             blk["pooled"]["entries"])
            # the test-window replay is not the pooled one
            pt, pp = blk["portfolio"]["test"], blk["portfolio"]["pooled"]
            self.assertLess(pt["trades_available"], pp["trades_available"])
            self.assertNotEqual((pt["trades_taken"], pt["final_equity"]),
                                (pp["trades_taken"], pp["final_equity"]))
            self.assertEqual(pt["window"]["start"], p1["method"]["split"]["windows"]["test"][0])
            self.assertGreaterEqual(pt["window"]["start"], "2023-01-01")
        for p in ee.PERIODS:
            self.assertEqual(len(years_seen[p]), 1)      # one shared window per period
            self.assertEqual(min(tails[p]), 0)           # it ends at some variant's last exit
            self.assertGreater(max(tails[p]), 0)         # and the others sit in cash
        # the coarser clustering is exercised: the 250-bar hold's train range clusters
        # by entry year (years - 1 df), the 40-bar hold's by month/quarter
        e3_train = p1["variants"]["S1xE3"]["train"]
        self.assertEqual(e3_train["excess_vs_universe_se_basis"], "year")
        e3_years = {r["entry_date"][:4] for r in trades if r["variant"] == "S1xE3" and r["common"]
                    and r["period"] == "train" and r["excess_vs_universe_pct"] is not None}
        self.assertEqual(e3_train["excess_vs_universe_df"], len(e3_years) - 1)
        self.assertEqual(p1["variants"]["S1xE1"]["train"]["excess_vs_universe_se_basis"],
                         "month/quarter")
        # the cash tail is stated with its direction and this run's numbers
        note = p1["method"]["split"]["windows_note"]
        self.assertIn("UNDERSTATES its CAGR", note)
        self.assertIn("OVERSTATES its shortfall", note)
        ctl_pf = p1["variants"]["S1xE1"]["portfolio"]
        for p in ee.SPLIT_PERIODS:
            clause = ee.tail_summary_text(ctl_pf[p], p)
            self.assertIn(f"{p.upper()}: control last exit {ctl_pf[p]['last_exit_date']}", clause)
            self.assertIn(f"cash tail {ctl_pf[p]['cash_tail_days']} days", clause)
            self.assertIn(clause, note)
            self.assertIn(clause, p1["method"]["portfolio"]["windows"])
            self.assertTrue(any(clause in c for c in p1["caveats"]))
        self.assertTrue(any("understates the shorter hold's CAGR" in c and "own_window" in c
                            for c in p1["caveats"]))
        self.assertIn("own_window", p1["method"]["portfolio"])
        self.assertIn("cash_tail", p1["method"]["portfolio"])
        # every variant has TEST trades on the common set, and E1's own window has more
        self.assertGreater(min(p1["variants"][v]["test"]["trades"] for v in ee.VARIANTS), 0)
        e1 = p1["variants"]["S1xE1"]
        self.assertGreater(e1["full_window"]["test"]["trades"], e1["test"]["trades"])
        self.assertEqual(e1["full_window"]["train"]["trades"], e1["train"]["trades"])
        # censoring is decided at the fill: the last eligible entry per hold sits
        # max_hold bars before the data end, the common cutoff at the longest hold
        cens = p1["method"]["censoring"]
        n = len(self.dates)
        for ek, rule in ee.EXITS.items():
            last = max(r["entry_date"] for r in trades if r["exit_rule"] == ek
                       and r["entry_date"] <= self.dates[n - 1 - rule["max_hold"]])
            self.assertEqual(cens["last_eligible_entry_date"][ek]["date"], last)
            self.assertEqual(p1["variants"][f"S1x{ek}"]["last_eligible_entry_date"], last)
        self.assertEqual(cens["entries_common"]["pooled"],
                         p1["variants"]["S1xE1"]["pooled"]["entries"])
        for r in trades:
            self.assertEqual(r["common"], r["entry_date"] <= self.common_cutoff)
            if r["censored"]:
                self.assertGreater(r["entry_date"],
                                   self.dates[n - 1 - ee.EXITS[r["exit_rule"]]["max_hold"]])
                self.assertIsNone(r["exit_date"])
        self.assertGreater(cens["censored_at_fill"]["S1xE3"], cens["censored_at_fill"]["S1xE1"])
        self.assertEqual(p1["variants"]["S1xE3"]["censored_at_fill"],
                         cens["censored_at_fill"]["S1xE3"])
        ctl = p1["variants"]["S1xE1"]
        self.assertEqual(ctl["pooled"]["skipped_overlap"], 0)      # stored trades never overlap
        self.assertGreater(p1["variants"]["S1xE3"]["pooled"]["skipped_overlap"], 0)
        self.assertIn(p1["verdict"], ("confirmed", "improves but unproven", "not confirmed"))
        self.assertEqual(p1["control"]["variant"], "S1xE1")
        if p1["winner"]["variant"] != "S1xE1":
            pc = p1["paired_test_cagr"]
            self.assertEqual(pc["seeds"], 3)
            self.assertIn("winner_beats_control_share", pc)
            self.assertEqual(pc["window"], p1["variants"]["S1xE1"]["portfolio"]["test"]["window"])
            # both sides' tails on the shared TEST window, and the same pairing with each
            # side on its exit rule's own window
            ctl_t = p1["variants"]["S1xE1"]["portfolio"]["test"]
            win_t = p1["variants"][p1["winner"]["variant"]]["portfolio"]["test"]
            self.assertEqual(pc["control_last_exit_date"], ctl_t["last_exit_date"])
            self.assertEqual(pc["control_cash_tail_days"], ctl_t["cash_tail_days"])
            self.assertEqual(pc["winner_last_exit_date"], win_t["last_exit_date"])
            self.assertEqual(pc["winner_cash_tail_days"], win_t["cash_tail_days"])
            own = pc["own_window"]
            wk = p1["winner"]["variant"][3:]
            self.assertEqual(own["winner_window"], {"start": own_w[wk]["test"][0],
                                                    "end": own_w[wk]["test"][1]})
            self.assertEqual(own["control_window"], {"start": own_w["E1"]["test"][0],
                                                     "end": own_w["E1"]["test"][1]})
            self.assertEqual(own["control_cash_tail_days"], ctl_t["own_window"]["cash_tail_days"])
            self.assertEqual(own["winner_cash_tail_days"], win_t["own_window"]["cash_tail_days"])
            self.assertEqual(own["seeds"], 3)
            # the control's TEST tail exceeds the flag share here: the verdict reason
            # states it, with the own-window paired share (the verdict rule is unchanged)
            span = days_between(ctl_t["window"]["start"], ctl_t["window"]["end"])
            self.assertGreater(ctl_t["cash_tail_days"], ee.CASH_TAIL_FLAG_SHARE * span)
            self.assertIn("note: the control is in cash for the last "
                          f"{ctl_t['cash_tail_days']} days", p1["verdict_reason"])
            self.assertIn(f"ahead in {own['winner_beats_control_share']:.0%} of the seed pairs",
                          p1["verdict_reason"])
            self.assertEqual(p1["verdict_reason"], p1["verdict_reason"].split("; note:")[0]
                             + "; note: " + ee.cash_tail_note(ctl_t, pc))
        self.assertIn("train", p1["benchmark"])
        self.assertIn("test", p1["benchmark"])
        self.assertEqual(p1["benchmark"]["test"]["window"],
                         p1["variants"]["S1xE1"]["portfolio"]["test"]["window"])
        self.assertTrue(any("multiple testing" in c for c in p1["caveats"]))
        self.assertTrue(any("Quasi out-of-sample" in c for c in p1["caveats"]))
        self.assertTrue(any("decided at the fill" in c for c in p1["caveats"]))
        self.assertEqual(p1["method"]["split"]["rule"], "by ENTRY date: train < 2023-01-01 <= test")
        self.assertIn("quasi out-of-sample", p1["method"]["split"]["label"])
        self.assertIn("SIGNAL bar", p1["method"]["entries"])
        src = p1["method"]["source"]["prices_cache"]
        self.assertEqual(src["last_bar_date"], self.dates[-1])
        self.assertEqual(src["benchmark_last_bar_date"], {"us": self.dates[-1]})
        self.assertIsNotNone(src["file_mtime"])
        self.assertIn("read-only", src["access"])
        # every per-trade record carries the variant name and the two headline returns
        self.assertEqual(sorted({r["variant"] for r in trades}), sorted(ee.VARIANTS))
        closed = [r for r in trades if not r["censored"]]
        for r in closed:
            self.assertIsNotNone(r["ret_net_pct"])
            self.assertIsNotNone(r["r_net"])
            self.assertIsNotNone(r["universe_ret_pct"])
            self.assertAlmostEqual(r["excess_vs_universe_pct"],
                                   r["ret_net_pct"] - r["universe_ret_pct"], delta=2e-3)
        # baselines: the index leg carries the dividend-yield correction and, for a
        # trailing exit, stops at the close before the open fill
        div = ee.dividend_yields(None, self.cfg)["us"]
        self.assertGreater(div, 0)
        bench = bt.BenchContext.from_df(self.frames["^GSPC"])
        trails = [r for r in closed if r["outcome"] == "trail"]
        self.assertGreater(len(trails), 0)
        for r in trails[:5] + [x for x in closed if x["outcome"] == "time"][:5]:
            base = r["baseline_exit_date"]
            self.assertEqual(base < r["exit_date"], r["outcome"] == "trail")
            want = bt.index_return(bench.dates, bench.opens, bench.closes, r["entry_date"], base,
                                   div)
            self.assertEqual(r["index_ret_pct"], round(want, 3))
            no_div = bt.index_return(bench.dates, bench.opens, bench.closes, r["entry_date"], base)
            self.assertNotEqual(r["index_ret_pct"], round(no_div, 3))
        # the regime multipliers reach the replay: the synthetic index has downtrend days
        self.assertGreater(sum(p1["variants"][v]["portfolio"]["pooled"]["skipped_regime"]
                               for v in ee.VARIANTS), 0)
        # identical with two workers (only the timestamp differs)
        out2 = Path(self.tmp.name) / "two.json"
        p2 = self.run_it(out2, 2)
        a, b = json.loads(out1.read_text()), json.loads(out2.read_text())
        a.pop("generated"), b.pop("generated")
        a["method"].pop("workers"), b["method"].pop("workers")
        self.assertEqual(a, b)
        self.assertEqual(ee.trades_output_path(out1).read_text(),
                         ee.trades_output_path(out2).read_text())

    def test_regime_table_is_passed_to_the_replay(self):
        # a regime table that stands aside on every day: no replay takes a trade
        table = {"us": {d: 0.0 for d in self.dates}}
        out = Path(self.tmp.name) / "regime.json"
        with mock.patch.object(ee, "regime_multipliers", return_value=table) as rm:
            p = self.run_it(out, 1)
        self.assertEqual(rm.call_count, 1)
        for v in ee.VARIANTS:
            for per in ee.PERIODS:
                pf = p["variants"][v]["portfolio"][per]
                self.assertEqual(pf["trades_taken"], 0)
                self.assertEqual(pf["skipped_regime"], pf["trades_available"])
                self.assertGreater(pf["skipped_regime"], 0)

    def test_limit_takes_the_first_entries(self):
        out = Path(self.tmp.name) / "smoke.json"
        p = self.run_it(out, 1, limit=7)
        self.assertTrue(p["smoke"])
        self.assertEqual(p["method"]["limit"], 7)
        self.assertEqual(p["method"]["split"]["entries"]["pooled"], 7)
        self.assertEqual(p["method"]["control_reproduction"]["compared"], 7)
        self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
