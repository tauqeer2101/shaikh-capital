"""Tests for atlas2.strategy_lab (the pre-registered strategy lab).

Synthetic series only; files go to temp dirs. Never touches data/, config.json
or the network (the downloader is replaced by a stub wherever it could run).
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from atlas2 import backtest as bt
from atlas2 import strategy_lab as sl
from atlas2.data import PriceStore
from atlas2.scan import BACKTEST_DEFAULTS, MIN_DOLLAR_VOL
from tests.test_backtest import random_frame


def bdays(start: str, n: int | None = None, end: str | None = None) -> np.ndarray:
    idx = pd.bdate_range(start, periods=n) if n is not None else pd.bdate_range(start, end)
    return np.asarray(idx.strftime("%Y-%m-%d"), dtype=str)


def cdays(start: str, n: int) -> np.ndarray:
    return np.asarray(pd.date_range(start, periods=n).strftime("%Y-%m-%d"), dtype=str)


def mini_panel(dates, opens: dict, closes: dict, volume: float = 1e6) -> sl.StockPanel:
    """A StockPanel straight from per-ticker open / close lists (NaN = no bar)."""
    tickers = sorted(opens)
    O = np.column_stack([np.asarray(opens[t], float) for t in tickers])
    C = np.column_stack([np.asarray(closes[t], float) for t in tickers])
    return sl.StockPanel(dates=np.asarray(dates, dtype=str), tickers=tickers, O=O, C=C,
                         V=np.full(O.shape, volume), valid=np.isfinite(O) & np.isfinite(C))


def manual_signals(idx, rows: list[dict], tickers: list[str], lookback: int = 126
                   ) -> sl.SignalData:
    """SignalData with hand-set momentum: rows[j] = {ticker: momentum} (others not eligible)."""
    mom = np.full((len(idx), len(tickers)), np.nan)
    for j, row in enumerate(rows):
        for t, v in row.items():
            mom[j, tickers.index(t)] = v
    return sl.SignalData(idx=np.asarray(idx, int), mom={lookback: mom}, eligible=np.isfinite(mom))


def month_step_equity(dates: np.ndarray, monthly: list[float]) -> np.ndarray:
    """Equity that is flat inside a month and jumps by monthly[k] on the last bar of
    month k (dates[0] is the anchor, itself a month end)."""
    eq = np.ones(len(dates))
    level, k = 1.0, 0
    for i in range(1, len(dates)):
        last_of_month = i + 1 == len(dates) or dates[i + 1][:7] != dates[i][:7]
        if last_of_month:
            level *= 1 + monthly[k]
            k += 1
            eq[i] = level
        else:
            eq[i] = level
    return eq


# --------------------------------------------------------------------------- metrics

class PerfMetricsTest(unittest.TestCase):
    def test_cagr_total_return_and_anchor(self):
        dates = np.array(["2019-06-28", "2019-12-31", "2020-12-31", "2021-12-31"])
        eq = np.array([5.0, 1.0, 1.5, 2.0])
        m = sl.perf_metrics(dates, eq, 1, 3)       # the anchor is index 1, not 0
        years = 731 / 365.25
        self.assertEqual(m["start"], "2019-12-31")
        self.assertEqual(m["end"], "2021-12-31")
        self.assertAlmostEqual(m["total_return_pct"], 100.0, places=6)
        self.assertAlmostEqual(m["cagr_pct"], round((2 ** (1 / years) - 1) * 100, 2), places=6)
        self.assertEqual(m["yearly_pct"], {"2020": 50.0, "2021": 33.33})
        self.assertEqual(m["partial_years"], [])

    def test_max_drawdown_uses_daily_marks(self):
        dates = cdays("2020-01-01", 6)
        eq = np.array([1.0, 1.2, 0.6, 0.9, 1.3, 1.25])
        m = sl.perf_metrics(dates, eq, 0, 5)
        self.assertAlmostEqual(m["max_drawdown_pct"], -50.0, places=6)
        self.assertEqual(m["max_drawdown_peak"], "2020-01-02")
        self.assertEqual(m["max_drawdown_trough"], "2020-01-03")
        # under water from the 2nd (peak) until the 5th sets a new high: 3 days
        self.assertEqual(m["longest_underwater_days"], 3)

    def test_underwater_spell_ends_at_the_recovery_bar(self):
        dates = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
        eq = np.array([1.0, 0.9, 0.8, 0.95, 1.05])
        # below the first bar's high until 01-07 recovers it: 6 calendar days, not 5
        self.assertEqual(sl.perf_metrics(dates, eq, 0, 4)["longest_underwater_days"], 6)
        # a spell still open at the end is measured to the last bar
        self.assertEqual(sl.perf_metrics(dates, eq, 0, 3)["longest_underwater_days"], 5)
        # never below a high: no time under water, also when the curve is flat
        self.assertEqual(sl.perf_metrics(dates, np.array([1.0, 1.1, 1.2, 1.3, 1.4]), 0, 4)[
            "longest_underwater_days"], 0)
        self.assertEqual(sl.perf_metrics(dates, np.ones(5), 0, 4)["longest_underwater_days"], 0)
        # the longest of several spells wins: 01-01 -> 01-03 (2 days), then 01-03 -> open (4)
        eq = np.array([1.0, 0.9, 1.2, 1.1, 1.15])
        self.assertEqual(sl.perf_metrics(dates, eq, 0, 4)["longest_underwater_days"], 4)

    def test_final_year_is_partial_when_the_data_ends_inside_december(self):
        dates = np.array(["2019-12-31", "2020-06-30", "2020-12-15"])
        m = sl.perf_metrics(dates, np.array([1.0, 1.1, 1.2]), 0, 2)
        self.assertEqual(m["partial_years"], ["2020"])
        self.assertTrue(sl.perf_metrics(dates, np.array([1.0, 1.1, 0.9]), 0, 2)[
            "worst_year"]["partial"])
        # the last trading day of December completes the year ...
        full = np.array(["2019-12-31", "2020-06-30", "2020-12-31"])
        self.assertEqual(sl.perf_metrics(full, np.array([1.0, 1.1, 1.2]), 0, 2)["partial_years"],
                         [])
        # ... and so does a later bar in the next year, wherever December stopped
        more = np.array(["2019-12-31", "2020-06-30", "2020-12-15", "2021-01-04"])
        self.assertEqual(sl.perf_metrics(more, np.array([1.0, 1.1, 1.2, 1.3]), 0, 2)[
            "partial_years"], [])
        nov = np.array(["2019-12-31", "2020-06-30", "2020-11-15"])
        self.assertEqual(sl.perf_metrics(nov, np.array([1.0, 1.1, 1.2]), 0, 2)["partial_years"],
                         ["2020"])

    def test_exact_figures_are_kept_next_to_the_rounded_ones(self):
        dates = np.array(["2019-12-31", "2020-12-31"])
        years = 366 / 365.25
        end = 1.399987 ** years                     # a CAGR of 39.9987%
        m = sl.perf_metrics(dates, np.array([1.0, end]), 0, 1)
        self.assertEqual(m["cagr_pct"], 40.0)       # the display figure rounds up ...
        self.assertAlmostEqual(m["cagr_exact_pct"], 39.9987, places=6)   # ... the exact one not
        self.assertLess(sl.exact(m, "cagr_pct"), 40.0)
        dd = sl.perf_metrics(cdays("2020-01-01", 3), np.array([1.0, 0.49996, 0.6]), 0, 2)
        self.assertEqual(dd["max_drawdown_pct"], -50.0)
        self.assertAlmostEqual(dd["max_drawdown_exact_pct"], -50.004, places=9)
        self.assertEqual(sl.exact(dd, "max_drawdown_pct"), dd["max_drawdown_exact_pct"])
        # a period without exact figures falls back to the stored ones
        self.assertEqual(sl.exact({"cagr_pct": 12.0}, "cagr_pct"), 12.0)
        self.assertIsNone(sl.exact(None, "cagr_pct"))

    def test_drawdown_restarts_inside_the_period(self):
        dates = cdays("2020-01-01", 5)
        eq = np.array([2.0, 1.0, 1.1, 1.0, 1.2])
        self.assertAlmostEqual(sl.perf_metrics(dates, eq, 1, 4)["max_drawdown_pct"],
                               round((1.0 / 1.1 - 1) * 100, 2), places=6)
        self.assertAlmostEqual(sl.perf_metrics(dates, eq, 0, 4)["max_drawdown_pct"], -50.0,
                               places=6)

    def test_worst_year_and_partial_years(self):
        dates = np.array(["2019-08-30", "2019-12-31", "2020-12-31", "2021-06-30"])
        eq = np.array([1.0, 1.1, 0.88, 0.968])
        m = sl.perf_metrics(dates, eq, 0, 3)
        self.assertEqual(m["yearly_pct"], {"2019": 10.0, "2020": -20.0, "2021": 10.0})
        self.assertEqual(m["partial_years"], ["2019", "2021"])
        self.assertEqual(m["worst_year"], {"year": "2020", "return_pct": -20.0, "partial": False})

    def test_volatility_is_annualised_daily_std(self):
        dates = cdays("2020-01-01", 5)
        eq = np.array([1.0, 1.01, 1.0, 1.02, 1.01])
        r = eq[1:] / eq[:-1] - 1
        m = sl.perf_metrics(dates, eq, 0, 4)
        self.assertAlmostEqual(m["ann_volatility_pct"],
                               round(float(np.std(r, ddof=1)) * math.sqrt(252) * 100, 2), places=6)

    def test_wiped_out_account(self):
        dates = cdays("2020-01-01", 4)
        m = sl.perf_metrics(dates, np.array([1.0, 0.5, 0.0, 0.0]), 0, 3)
        self.assertEqual(m["cagr_pct"], -100.0)
        self.assertEqual(m["max_drawdown_pct"], -100.0)

    def test_empty_period(self):
        m = sl.perf_metrics(cdays("2020-01-01", 3), np.ones(3), 2, 2)
        self.assertIsNone(m["cagr_pct"])
        self.assertEqual(m["yearly_pct"], {})


class MonthlyStatsTest(unittest.TestCase):
    def setUp(self):
        self.dates = bdays("2019-12-31", end="2020-03-31")
        self.jan = [i for i, d in enumerate(self.dates) if d[:7] == "2020-01"]

    def test_return_low_high_and_fall_inside_the_month(self):
        eq = np.ones(len(self.dates))
        j = self.jan
        eq[j[0]:] = 1.10                      # high of the month
        eq[j[5]:] = 0.88                      # low
        eq[j[-1]:] = 0.99                     # month end, then flat
        ms = sl.monthly_stats(self.dates, eq, 0, len(self.dates) - 1)
        self.assertEqual(ms["months"], ["2020-01", "2020-02", "2020-03"])
        self.assertAlmostEqual(ms["ret"][0], -0.01)
        self.assertAlmostEqual(ms["hi"][0], 1.10)
        self.assertAlmostEqual(ms["lo"][0], 0.88)
        self.assertAlmostEqual(ms["ddin"][0], 1 - 0.88 / 1.10)
        self.assertAlmostEqual(ms["ret"][1], 0.0)
        self.assertEqual(list(ms["me_idx"]), [0, j[-1], len(self.dates) - 22 - 1,
                                              len(self.dates) - 1])

    def test_partial_first_and_last_months_are_left_out(self):
        eq = np.linspace(1.0, 2.0, len(self.dates))
        mid_jan, mid_mar = self.jan[7], len(self.dates) - 5
        ms = sl.monthly_stats(self.dates, eq, mid_jan, mid_mar)
        self.assertEqual(ms["months"], ["2020-02"])
        self.assertEqual(self.dates[ms["me_idx"][0]], "2020-01-31")
        self.assertEqual(self.dates[ms["me_idx"][1]], "2020-02-28")

    def test_month_complete(self):
        self.assertTrue(sl.month_complete(self.dates, len(self.dates) - 1))     # 2020-03-31
        self.assertFalse(sl.month_complete(self.dates[:-1], len(self.dates) - 2))
        self.assertTrue(sl.month_complete(np.array(["2022-12-30"]), 0))          # Friday
        self.assertTrue(sl.month_complete(self.dates, self.jan[-1]))
        self.assertFalse(sl.month_complete(self.dates, self.jan[3]))

    def test_month_complete_when_the_last_weekday_is_a_market_holiday(self):
        # Good Friday 2024-03-29 and Memorial Day 2021-05-31 are the month's last weekday
        self.assertTrue(sl.month_complete(np.array(["2024-03-27", "2024-03-28"]), 1))
        self.assertTrue(sl.month_complete(np.array(["2021-05-27", "2021-05-28"]), 1))
        # a day earlier a trading day is still to come
        self.assertFalse(sl.month_complete(np.array(["2024-03-27"]), 0))
        self.assertFalse(sl.month_complete(np.array(["2021-05-27"]), 0))
        # Good Friday in the middle of a month does not end it (2025-04-18)
        self.assertFalse(sl.month_complete(np.array(["2025-04-17"]), 0))
        # an ordinary Friday before a Monday the 31st is not a month end
        self.assertFalse(sl.month_complete(np.array(["2023-07-28"]), 0))
        # Memorial Day that is not the 31st: Tuesday the 31st still trades (2022-05-30)
        self.assertFalse(sl.month_complete(np.array(["2022-05-27"]), 0))
        for year, day in ((2024, "2024-03-31"), (2021, "2021-04-04"), (2029, "2029-04-01"),
                          (2008, "2008-03-23"), (2038, "2038-04-25")):
            self.assertEqual(sl._easter_sunday(year).isoformat(), day)
        # the next occurrences named in the review: 2027-05-28 and 2029-03-29
        self.assertTrue(sl.month_complete(np.array(["2027-05-28"]), 0))
        self.assertTrue(sl.month_complete(np.array(["2029-03-29"]), 0))

    def test_monthly_stats_keep_a_month_that_ends_on_a_holiday(self):
        dates = bdays("2024-01-31", end="2024-03-28")       # the cache ends before Good Friday
        eq = np.linspace(1.0, 1.2, len(dates))
        ms = sl.monthly_stats(dates, eq, 0, len(dates) - 1)
        self.assertEqual(ms["months"], ["2024-02", "2024-03"])


class RollingWindowsTest(unittest.TestCase):
    def setUp(self):
        self.dates = bdays("2019-12-31", end="2022-12-30")     # anchor + 36 full months

    def test_constant_growth(self):
        eq = month_step_equity(self.dates, [0.03] * 36)
        ref = month_step_equity(self.dates, [0.04] * 36)
        ms = sl.monthly_stats(self.dates, eq, 0, len(self.dates) - 1)
        self.assertEqual(len(ms["ret"]), 36)
        r = sl.rolling_windows(self.dates, eq, ms, ref, months=12)
        want = round((1.03 ** 12 - 1) * 100, 2)
        self.assertEqual(r["n_windows"], 25)
        self.assertEqual((r["min_cagr_pct"], r["median_cagr_pct"], r["max_cagr_pct"]),
                         (want, want, want))
        self.assertEqual(r["share_ge_target_pct"], 100.0)       # 42.6% >= 40%
        self.assertEqual(r["share_ge_1x_nasdaq_pct"], 0.0)
        self.assertEqual(r["worst_drawdown_in_any_window_pct"], 0.0)
        self.assertEqual(r["first_window"], ["2019-12-31", "2020-12-31"])
        self.assertEqual(r["last_window"], ["2021-12-31", "2022-12-30"])

    def test_min_max_share_and_worst_fall(self):
        eq = month_step_equity(self.dates, [0.05] * 12 + [-0.01] * 24)
        ms = sl.monthly_stats(self.dates, eq, 0, len(self.dates) - 1)
        r = sl.rolling_windows(self.dates, eq, ms, eq, months=12)
        self.assertEqual(r["max_cagr_pct"], round((1.05 ** 12 - 1) * 100, 2))
        self.assertEqual(r["min_cagr_pct"], round((0.99 ** 12 - 1) * 100, 2))
        self.assertEqual(r["share_ge_target_pct"], 20.0)        # the first 5 of 25 windows
        self.assertEqual(r["share_ge_1x_nasdaq_pct"], 100.0)    # against itself
        self.assertEqual(r["worst_drawdown_in_any_window_pct"],
                         round((0.99 ** 12 - 1) * 100, 2))
        self.assertEqual(r["max_window"], ["2019-12-31", "2020-12-31"])

    def test_statistics_on_a_history_that_separates_them(self):
        monthly = [0.027] * 12 + [0.031] * 12 + [0.004] * 12
        eq = month_step_equity(self.dates, monthly)
        ref = month_step_equity(self.dates, [0.02] * 36)
        ms = sl.monthly_stats(self.dates, eq, 0, len(self.dates) - 1)
        r = sl.rolling_windows(self.dates, eq, ms, ref, months=12)
        level = np.r_[1.0, np.cumprod(1 + np.asarray(monthly))]
        c = level[12:] / level[:-12] - 1                     # 25 one-year windows
        self.assertEqual(len(c), 25)
        self.assertEqual(r["min_cagr_pct"], round(float(c.min()) * 100, 2))
        self.assertEqual(r["max_cagr_pct"], round(float(c.max()) * 100, 2))
        self.assertEqual(r["median_cagr_pct"], round(float(np.sort(c)[12]) * 100, 2))
        self.assertNotAlmostEqual(float(np.sort(c)[12]), float(c.mean()), places=3)
        self.assertTrue(r["min_cagr_pct"] < r["median_cagr_pct"] < r["max_cagr_pct"])
        # the first window made 37.7%: above 35%, below the 40% target
        self.assertTrue(0.35 < c[0] < 0.40)
        hit = np.nonzero(c >= 0.40)[0]
        self.assertEqual(list(hit), list(range(5, 14)))
        self.assertGreater(int((c >= 0.35).sum()), len(hit))
        self.assertEqual(r["windows_ge_target"], 9)
        self.assertEqual(r["share_ge_target_pct"], 36.0)
        me = ms["me_idx"]
        self.assertEqual(r["windows_ge_target_start_range"],
                         [self.dates[me[5]], self.dates[me[13]]])
        self.assertEqual(r["windows_ge_target_start_years"], ["2020", "2021"])
        self.assertEqual(r["share_ge_1x_nasdaq_pct"],
                         round(100.0 * float((c >= 1.02 ** 12 - 1).mean()), 1))
        # the windows that do not overlap: 2020, 2021, 2022
        dec = r["non_overlapping_decades"]
        self.assertEqual(r["independent_windows"], 3)
        self.assertEqual([d["start"] for d in dec], ["2019-12-31", "2020-12-31", "2021-12-31"])
        self.assertEqual([d["end"] for d in dec], ["2020-12-31", "2021-12-31", "2022-12-30"])
        self.assertEqual([d["cagr_pct"] for d in dec],
                         [round(float(c[k]) * 100, 2) for k in (0, 12, 24)])
        self.assertEqual(dec[0]["nasdaq_1x_cagr_pct"], round((1.02 ** 12 - 1) * 100, 2))

    def test_no_window_at_the_target(self):
        eq = month_step_equity(self.dates, [0.01] * 36)
        ms = sl.monthly_stats(self.dates, eq, 0, len(self.dates) - 1)
        r = sl.rolling_windows(self.dates, eq, ms, months=12)
        self.assertEqual(r["windows_ge_target"], 0)
        self.assertIsNone(r["windows_ge_target_start_range"])
        self.assertEqual(r["windows_ge_target_start_years"], [])
        self.assertNotIn("nasdaq_1x_cagr_pct", r["non_overlapping_decades"][0])

    def test_ten_year_default_needs_enough_history(self):
        eq = month_step_equity(self.dates, [0.01] * 36)
        ms = sl.monthly_stats(self.dates, eq, 0, len(self.dates) - 1)
        self.assertIsNone(sl.rolling_windows(self.dates, eq, ms))


class BootstrapTest(unittest.TestCase):
    def stats(self, seed: int = 5, months: int = 60) -> tuple[dict, np.ndarray, np.ndarray]:
        dates = bdays("2014-12-31", end="2019-12-31")
        rng = np.random.default_rng(seed)
        eq = np.r_[1.0, np.cumprod(1 + 0.001 + 0.02 * rng.standard_normal(len(dates) - 1))]
        ms = sl.monthly_stats(dates, eq, 0, len(dates) - 1)
        self.assertEqual(len(ms["ret"]), months)
        return ms, dates, eq

    def test_same_seed_same_result_other_seed_differs(self):
        ms, _, _ = self.stats()
        a = sl.block_bootstrap(ms, n_paths=300)
        b = sl.block_bootstrap(ms, n_paths=300)
        c = sl.block_bootstrap(ms, n_paths=300, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(a["seed"], 20261001)
        self.assertNotEqual(a["cagr_pct"], c["cagr_pct"])

    def test_spec_defaults(self):
        self.assertEqual((sl.BOOT_PATHS, sl.BOOT_HORIZON, sl.BOOT_BLOCK, sl.BOOT_SEED),
                         (10000, 120, 12, 20261001))
        ms, _, _ = self.stats()
        out = sl.block_bootstrap(ms, n_paths=50)
        self.assertEqual((out["horizon_months"], out["block_months"], out["paths"]), (120, 12, 50))

    def test_constant_months_give_one_outcome(self):
        n = 30
        ms = {"ret": np.full(n, -0.05), "lo": np.full(n, 0.80), "hi": np.full(n, 1.10),
              "ddin": np.full(n, 0.30)}
        out = sl.block_bootstrap(ms, n_paths=40)
        cagr = round((0.95 ** 12 - 1) * 100, 2)
        self.assertEqual(set(out["cagr_pct"].values()), {cagr})
        self.assertEqual(out["p_cagr_ge_40_pct"], 0.0)
        self.assertEqual(out["p_end_below_start_pct"], 100.0)
        # the low of the last month against the high of the first one
        dd = 1 - 0.95 ** 119 * 0.80 / 1.10
        self.assertEqual(out["median_max_drawdown_pct"], round(-dd * 100, 2))
        self.assertEqual(out["p_max_drawdown_worse_than_75_pct"], 100.0)

    def test_probabilities(self):
        n = 24
        ms = {"ret": np.full(n, 0.03), "lo": np.ones(n), "hi": np.full(n, 1.03),
              "ddin": np.zeros(n)}
        out = sl.block_bootstrap(ms, n_paths=20)
        self.assertEqual(out["p_cagr_ge_40_pct"], 100.0)       # 1.03^12 = 42.6%
        self.assertEqual(out["p_cagr_ge_20_pct"], 100.0)
        self.assertEqual(out["p_max_drawdown_worse_than_50_pct"], 0.0)
        self.assertEqual(out["p_end_below_start_pct"], 0.0)

    def test_drawdown_is_on_daily_marks(self):
        """With block = horizon = the whole history every path is a rotation of the
        real months; its drawdown must equal the daily drawdown of that rotation."""
        ms, dates, eq = self.stats(seed=11)
        n = len(ms["ret"])
        paths = 60
        out = sl.block_bootstrap(ms, n_paths=paths, horizon=n, block=n, seed=3)
        starts = np.random.default_rng(3).integers(0, n, size=(paths, 1))[:, 0]
        me = ms["me_idx"]
        monthly_paths = [eq[me[k] + 1:me[k + 1] + 1] / eq[me[k]] for k in range(n)]
        dds, finals = [], []
        for s in starts:
            level, path = 1.0, [1.0]
            for k in range(n):
                seg = monthly_paths[(s + k) % n] * level
                path.extend(seg.tolist())
                level = seg[-1]
            path = np.asarray(path)
            dds.append(float((1 - path / np.maximum.accumulate(path)).max()))
            finals.append(level)
        self.assertEqual(out["median_max_drawdown_pct"], round(-float(np.median(dds)) * 100, 2))
        self.assertEqual(out["p_max_drawdown_worse_than_50_pct"],
                         round(100.0 * float((np.asarray(dds) > 0.5).mean()), 2))
        # a rotation of the whole history always ends where the history ended
        self.assertAlmostEqual(min(finals), max(finals), places=9)
        self.assertEqual(len(set(out["cagr_pct"].values())), 1)

    def test_every_reported_statistic_against_a_per_path_loop(self):
        """An independent path-by-path recomputation with the same block starts, on a
        history wild enough for every threshold to separate paths."""
        n, paths, horizon, block, seed = 60, 400, 30, 7, 11
        rng = np.random.default_rng(123)
        ret = np.clip(rng.normal(0.025, 0.25, n), -0.6, None)
        lo = np.minimum(1.0, 1.0 + ret) - 0.03 * rng.random(n)
        hi = np.maximum(1.0, 1.0 + ret) + 0.03 * rng.random(n)
        ddin = np.maximum(1.0 - lo, 0.0) * (0.5 + 0.5 * rng.random(n))
        ms = {"ret": ret, "lo": lo, "hi": hi, "ddin": ddin}
        out = sl.block_bootstrap(ms, n_paths=paths, horizon=horizon, block=block, seed=seed)

        n_blocks = -(-horizon // block)
        starts = np.random.default_rng(seed).integers(0, n, size=(paths, n_blocks))
        cagr, worst, final = [], [], []
        for row in starts:
            months = [(int(s) + k) % n for s in row for k in range(block)][:horizon]
            level, peak, dd = 1.0, 1.0, 0.0
            for m in months:
                dd = max(dd, 1.0 - level * lo[m] / peak, ddin[m])
                peak = max(peak, level * hi[m])
                level *= 1.0 + ret[m]
            final.append(level)
            worst.append(dd)
            cagr.append(level ** (12.0 / horizon) - 1.0)
        cagr, worst, final = np.asarray(cagr), np.asarray(worst), np.asarray(final)

        def share(mask) -> float:
            return round(100.0 * float(np.mean(mask)), 2)

        q = np.percentile(cagr, [5, 25, 50, 75, 95])
        self.assertEqual(out["cagr_pct"], {k: round(float(v) * 100, 2)
                                           for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q)})
        self.assertEqual(out["p_cagr_ge_40_pct"], share(cagr >= 0.40))
        self.assertEqual(out["p_cagr_ge_20_pct"], share(cagr >= 0.20))
        self.assertEqual(out["p_max_drawdown_worse_than_50_pct"], share(worst > 0.50))
        self.assertEqual(out["p_max_drawdown_worse_than_75_pct"], share(worst > 0.75))
        self.assertEqual(out["p_end_below_start_pct"], share(final < 1.0))
        self.assertEqual(out["median_max_drawdown_pct"], round(-float(np.median(worst)) * 100, 2))
        self.assertEqual((out["months_in_history"], out["block_months"], out["horizon_months"]),
                         (n, block, horizon))
        # the history really separates the thresholds (a shifted one gives another answer)
        self.assertNotEqual(share(cagr >= 0.40), share(cagr >= 0.30))
        self.assertNotEqual(share(cagr >= 0.20), share(cagr >= 0.10))
        self.assertNotEqual(share(worst > 0.50), share(worst > 0.60))
        self.assertNotEqual(share(worst > 0.75), share(worst > 0.90))
        self.assertNotEqual(share(final < 1.0), share(final < 0.5))
        self.assertNotEqual(share(cagr >= 0.40), share(cagr >= 0.20))
        q2 = np.percentile(cagr, [10, 25, 50, 75, 90])
        self.assertNotEqual(round(float(q[0]) * 100, 2), round(float(q2[0]) * 100, 2))
        self.assertNotEqual(round(float(q[4]) * 100, 2), round(float(q2[4]) * 100, 2))
        self.assertTrue(0.0 < out["p_cagr_ge_40_pct"] < out["p_cagr_ge_20_pct"] < 100.0)
        self.assertTrue(0.0 < out["p_max_drawdown_worse_than_75_pct"]
                        < out["p_max_drawdown_worse_than_50_pct"] < 100.0)
        self.assertTrue(0.0 < out["p_end_below_start_pct"] < 100.0)

    def test_longer_blocks_are_a_parameter(self):
        ms, _, _ = self.stats()
        a = sl.block_bootstrap(ms, n_paths=200)
        b = sl.block_bootstrap(ms, n_paths=200, block=36)
        self.assertEqual((a["block_months"], b["block_months"]), (12, 36))
        self.assertNotEqual(a["cagr_pct"], b["cagr_pct"])
        probs = sl._boot_probs(b)
        self.assertEqual(probs["block_months"], 36)
        self.assertEqual(probs["p_cagr_ge_40_pct"], b["p_cagr_ge_40_pct"])
        self.assertEqual(probs["median_cagr_pct"], b["cagr_pct"]["p50"])
        self.assertIsNone(sl._boot_probs(None))

    def test_too_short_history(self):
        self.assertIsNone(sl.block_bootstrap({"ret": np.zeros(5), "lo": np.ones(5),
                                              "hi": np.ones(5), "ddin": np.zeros(5)}))
        self.assertIsNone(sl.block_bootstrap(None))


# --------------------------------------------------------------------------- strategy B

class SyntheticFundTest(unittest.TestCase):
    def test_spec_constants(self):
        self.assertEqual(sl.B_LEVERAGE, (1, 2, 3))
        self.assertEqual(sl.B_FILTERS, ("none", "sma200", "sma200_band"))
        self.assertEqual(sl.B_EXPENSE, {1: 0.0020, 2: 0.0095, 3: 0.0095})
        self.assertEqual((sl.B_DIV_YIELD, sl.B_FINANCING_SPREAD, sl.B_SWITCH_COST, sl.B_BAND),
                         (0.007, 0.005, 0.001, 0.02))
        self.assertEqual((sl.B_TRAIN_END, sl.B_TRAIN_DD_FLOOR_PCT, sl.B_START),
                         ("2009-12-31", -75.0, "1986-10-01"))

    def test_daily_return_with_financing_expense_and_dividends(self):
        close = np.array([100.0, 110.0, 99.0])
        tb = np.array([5.0, 5.0, 2.0])                    # percent per year
        r1, r2 = 0.10, -0.10
        for lev, expense in ((1, 0.0020), (2, 0.0095), (3, 0.0095)):
            got = sl.synthetic_fund_returns(close, tb, lev)
            self.assertTrue(np.isnan(got[0]))
            want1 = lev * (r1 + 0.007 / 252) - (lev - 1) * (0.05 + 0.005) / 252 - expense / 252
            want2 = lev * (r2 + 0.007 / 252) - (lev - 1) * (0.02 + 0.005) / 252 - expense / 252
            self.assertAlmostEqual(got[1], want1, places=14)
            self.assertAlmostEqual(got[2], want2, places=14)

    def test_one_x_pays_no_financing(self):
        close = np.array([100.0, 100.0])
        lo = sl.synthetic_fund_returns(close, np.array([0.0, 0.0]), 1)[1]
        hi = sl.synthetic_fund_returns(close, np.array([9.0, 9.0]), 1)[1]
        self.assertAlmostEqual(lo, hi, places=15)
        self.assertAlmostEqual(lo, (0.007 - 0.002) / 252, places=15)

    def test_cash_return(self):
        np.testing.assert_allclose(sl.cash_returns(np.array([2.52, 0.0, -0.1])),
                                   [0.0001, 0.0, -0.1 / 100 / 252])

    def test_wipe_out_is_capped_and_final(self):
        n = 215
        close = np.r_[np.full(n - 3, 100.0), 60.0, 90.0, 120.0]    # -40% in one day
        dates = bdays("2000-01-03", n)
        self.assertEqual(sl.synthetic_fund_returns(close, np.zeros(n), 3)[n - 3], -1.0)
        res = sl.run_leverage(dates, close, np.zeros(n), 3, "none", d0=205)
        self.assertEqual(res["equity"][n - 3], 0.0)
        self.assertEqual(res["equity"][-1], 0.0)
        self.assertEqual(sl.perf_metrics(dates, res["equity"], 204, n - 1)["cagr_pct"], -100.0)
        two = sl.run_leverage(dates, close, np.zeros(n), 2, "none", d0=205)
        self.assertGreater(two["equity"][-1], 0.0)


class TrendFilterTimingTest(unittest.TestCase):
    """Signal at the close of t, switch at the close of t+1, exposure from t+2."""

    def setUp(self):
        # 205 flat closes (signal off: a close equal to the average is not above it),
        # the first close above the average at t = 205, down through it at t = 209
        self.close = np.r_[np.full(205, 100.0), 110.0, 115.5, 127.05, 127.05, 80.0, 72.0, 72.0,
                           72.0]
        self.n = len(self.close)
        self.dates = bdays("2000-01-03", self.n)
        self.tb = np.full(self.n, 2.52)                   # 0.01% per day in T-bills
        self.t_on, self.t_off = 205, 209

    def test_state(self):
        st = sl.trend_state(self.close, "sma200")
        self.assertTrue((st[:199] == -1).all())           # no 200-day average yet
        self.assertTrue((st[199:205] == 0).all())
        self.assertEqual(list(st[205:]), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertTrue((sl.trend_state(self.close, "none") == 1).all())

    def test_one_day_delay_in_and_out(self):
        res = sl.run_leverage(self.dates, self.close, self.tb, 1, "sma200", d0=201)
        eq, t = res["equity"], self.t_on
        cash = 0.0001
        fund = lambda i: self.close[i] / self.close[i - 1] - 1 + (0.007 - 0.002) / 252
        self.assertTrue(np.isnan(eq[199]))
        self.assertEqual(eq[200], 1.0)                    # the anchor
        # cash through the signal day
        self.assertAlmostEqual(eq[t], (1 + cash) ** 5, places=12)
        # t+1: still T-bills (the +5% of that day is NOT earned); the switch costs 0.10%
        self.assertFalse(res["in_fund"][t + 1])
        self.assertTrue(res["switch"][t + 1])
        self.assertAlmostEqual(eq[t + 1] / eq[t], (1 + cash) * (1 - 0.001), places=12)
        # t+2: the first day in the fund
        self.assertTrue(res["in_fund"][t + 2])
        self.assertAlmostEqual(eq[t + 2] / eq[t + 1], 1 + fund(t + 2), places=12)
        # out: the signal turns off at t_off, the fall of t_off+1 is still taken in the fund
        f = self.t_off
        self.assertTrue(res["in_fund"][f + 1])
        self.assertTrue(res["switch"][f + 1])
        self.assertAlmostEqual(eq[f + 1] / eq[f], (1 + fund(f + 1)) * (1 - 0.001), places=12)
        self.assertFalse(res["in_fund"][f + 2])
        self.assertAlmostEqual(eq[f + 2] / eq[f + 1], 1 + cash, places=12)
        self.assertEqual(int(res["switch"].sum()), 2)

    def test_unfiltered_fund_never_switches_and_pays_no_entry_cost(self):
        res = sl.run_leverage(self.dates, self.close, self.tb, 2, "none", d0=201)
        self.assertEqual(int(res["switch"].sum()), 0)
        self.assertTrue(res["in_fund"][201:].all())
        fund = sl.synthetic_fund_returns(self.close, self.tb, 2)
        self.assertAlmostEqual(res["equity"][201], 1 + fund[201], places=14)
        np.testing.assert_allclose(res["equity"][201:], np.cumprod(1 + fund[201:]), rtol=1e-12)

    def test_period_statistics(self):
        res = sl.run_leverage(self.dates, self.close, self.tb, 1, "sma200", d0=201)
        m = sl.leverage_period(self.dates, res, 200, self.n - 1)
        self.assertEqual(m["switches"], 2)
        days = self.n - 201
        self.assertEqual(m["time_in_market_pct"], round(100.0 * 4 / days, 1))
        self.assertEqual(m["switches_per_year"], round(2 / m["years"], 2))
        # a period that starts on a switch day and ends inside the fund: the anchor bar i0
        # belongs to the previous period, the last bar i1 to this one
        i0, i1 = self.t_on + 1, self.t_off + 1
        self.assertTrue(res["switch"][i0])
        self.assertNotEqual(bool(res["in_fund"][i0]), bool(res["in_fund"][i1]))
        m = sl.leverage_period(self.dates, res, i0, i1)
        self.assertEqual(m["switches"], 1)                     # the exit at i1, not the entry at i0
        self.assertEqual(m["time_in_market_pct"], 100.0)       # days i0+1 .. i1 are all in the fund
        self.assertEqual(m["switches_per_year"], round(1 / m["years"], 2))

    def test_same_close_sensitivity_switches_a_day_earlier(self):
        res = sl.run_leverage(self.dates, self.close, self.tb, 1, "sma200", d0=201, delay=0)
        eq, t, f = res["equity"], self.t_on, self.t_off
        cash = 0.0001
        fund = lambda i: self.close[i] / self.close[i - 1] - 1 + (0.007 - 0.002) / 252
        # the switch is done at the signal's own close; the next day is already in the fund
        self.assertTrue(res["switch"][t])
        self.assertFalse(res["in_fund"][t])
        self.assertAlmostEqual(eq[t] / eq[t - 1], (1 + cash) * (1 - 0.001), places=12)
        self.assertTrue(res["in_fund"][t + 1])
        self.assertAlmostEqual(eq[t + 1] / eq[t], 1 + fund(t + 1), places=12)
        # out at the close of the sell signal: the fall of the next day is NOT taken
        self.assertTrue(res["switch"][f])
        self.assertFalse(res["in_fund"][f + 1])
        self.assertAlmostEqual(eq[f + 1] / eq[f], 1 + cash, places=12)
        self.assertEqual(int(res["switch"].sum()), 2)
        with self.assertRaises(ValueError):
            sl.run_leverage(self.dates, self.close, self.tb, 1, "sma200", d0=201, delay=2)
        # the headline is the delayed reading
        head = sl.run_leverage(self.dates, self.close, self.tb, 1, "sma200", d0=201)
        np.testing.assert_array_equal(
            head["equity"],
            sl.run_leverage(self.dates, self.close, self.tb, 1, "sma200", d0=201, delay=1)[
                "equity"])
        self.assertFalse(head["switch"][t])

    def test_the_day_held_after_a_sell_signal(self):
        res = sl.run_leverage(self.dates, self.close, self.tb, 3, "sma200", d0=201)
        w = sl.held_after_sell_signal(self.dates, self.close, self.tb, res)
        f = self.t_off
        self.assertEqual(w["date"], self.dates[f + 1])       # the signal was given at f
        self.assertEqual(w["days_held_after_a_sell_signal"], 1)
        self.assertEqual(w["index_return_pct"], -10.0)        # 80 -> 72
        self.assertEqual(w["fund_3x_return_pct"],
                         round(float(sl.synthetic_fund_returns(self.close, self.tb, 3)[f + 1])
                               * 100, 2))
        self.assertLess(w["fund_3x_return_pct"], w["fund_2x_return_pct"])
        never = sl.run_leverage(self.dates, self.close, self.tb, 3, "none", d0=201)
        self.assertIsNone(sl.held_after_sell_signal(self.dates, self.close, self.tb, never))

    def test_financing_assumptions_are_parameters(self):
        base = sl.run_leverage(self.dates, self.close, self.tb, 2, "none", d0=201)
        dear = sl.run_leverage(self.dates, self.close, self.tb, 2, "none", d0=201,
                               div_yield=0.002, spread=0.010)
        # 2x: 2 * 0.5% less dividend and 0.5% more financing = 1.5% a year less
        np.testing.assert_allclose(
            np.diff(dear["equity"][200:]) / dear["equity"][200:-1]
            - np.diff(base["equity"][200:]) / base["equity"][200:-1],
            -(2 * 0.005 + 0.005) / 252, atol=1e-12)
        self.assertEqual((sl.B_CONS_SPREAD, sl.B_CONS_DIV_YIELD), (0.010, 0.002))

    def test_haircut_close(self):
        cut = sl.haircut_close(self.close, 4.0)
        self.assertEqual(cut[0], self.close[0])
        np.testing.assert_allclose(cut[1:] / cut[:-1] - 1,
                                   self.close[1:] / self.close[:-1] - 1 - 0.04 / 252, atol=1e-12)
        np.testing.assert_allclose(sl.haircut_close(self.close, 0.0), self.close, rtol=1e-12)
        self.assertEqual((sl.BOOT_SENS_BLOCK, sl.BOOT_SENS_HAIRCUT_PP), (36, 4.0))

    def test_band_needs_two_percent_either_way(self):
        close = np.r_[np.full(300, 100.0), 101.0, 103.0, 99.0, 97.0, 101.0, 103.0]
        plain = sl.trend_state(close, "sma200")
        band = sl.trend_state(close, "sma200_band")
        self.assertEqual(list(plain[300:]), [1, 1, 0, 0, 1, 1])
        self.assertEqual(list(band[300:]), [0, 1, 1, 0, 0, 1])
        self.assertTrue((band[:199] == -1).all())

    def test_band_opening_state_uses_the_plain_rule(self):
        close = np.r_[np.linspace(90.0, 100.0, 199), 101.0]   # the first average: close above it
        self.assertEqual(sl.trend_state(close, "sma200_band")[199], 1)

    def test_first_simulated_day(self):
        dates = bdays("1985-10-01", 400)
        close = np.full(400, 100.0)
        d0 = sl.first_sim_index(dates, close)
        self.assertEqual(dates[d0], "1986-10-01")
        late = bdays("1986-09-01", 400)                    # the average is not ready by 1986-10
        self.assertEqual(sl.first_sim_index(late, close), 201)
        with self.assertRaises(ValueError):
            sl.run_leverage(late, close, np.zeros(400), 1, "sma200", d0=150)


class AfterTaxTest(unittest.TestCase):
    """The German withholding-tax estimate on hand-made curves."""
    T = 0.26375

    def test_rate(self):
        self.assertEqual((sl.DE_TAX_RATE, sl.DE_PARTIAL_EXEMPTION), (0.26375, 0.30))

    def test_two_switches(self):
        # fund on days 1-3 (sold at the close of 3), T-bills on 4-5 (fund bought back at the
        # close of 5), fund on 6-7
        eq = np.array([1.0, 1.1, 1.2, 1.5, 1.5, 1.52, 1.6, 1.368])
        in_fund = np.array([False, True, True, True, False, False, True, True])
        switch = np.array([False, False, False, True, False, True, False, False])
        w = 1.5 - self.T * 0.5                       # 26.375% of the 0.5 gain is withheld
        w2 = w * 1.52 / 1.5                          # interest on what is left ...
        w2 -= self.T * (w2 - w)                      # ... taxed at the switch back
        want = w2 * 1.368 / 1.52                     # the last leg loses: nothing to tax
        got = sl.after_tax_growth(eq, in_fund, switch, 0, 7)
        self.assertAlmostEqual(got, want, places=12)
        self.assertLess(got, 1.368)
        # the partial exemption applies to the fund gain only, not to the interest
        w = 1.5 - self.T * 0.5 * 0.7
        w2 = w * 1.52 / 1.5
        w2 -= self.T * (w2 - w)
        self.assertAlmostEqual(sl.after_tax_growth(eq, in_fund, switch, 0, 7, exemption=0.3),
                               w2 * 1.368 / 1.52, places=12)
        # a period is a fresh account: from day 3 on only the interest is taxed
        self.assertAlmostEqual(sl.after_tax_growth(eq, in_fund, switch, 3, 7),
                               (1.52 / 1.5 - self.T * (1.52 / 1.5 - 1)) * 1.368 / 1.52, places=12)

    def test_loss_carry_forward(self):
        eq = np.array([1.0, 0.8, 0.8, 1.2])
        in_fund = np.array([False, True, False, True])
        switch = np.array([False, True, True, False])
        # sold with a loss of 0.2, bought back, sold at the end with a gain of 0.4:
        # only 0.4 - 0.2 is taxed
        self.assertAlmostEqual(sl.after_tax_growth(eq, in_fund, switch, 0, 3),
                               1.2 - self.T * 0.2, places=12)
        # with the partial exemption both the loss and the gain count 70%
        self.assertAlmostEqual(sl.after_tax_growth(eq, in_fund, switch, 0, 3, exemption=0.3),
                               1.2 - self.T * (0.28 - 0.14), places=12)
        # a loss that is never used is worth nothing
        self.assertAlmostEqual(sl.after_tax_growth(eq, in_fund, switch, 0, 2), 0.8, places=12)

    def test_buy_and_hold_is_taxed_once_at_the_end(self):
        eq = np.array([1.0, 1.5, 2.0])
        held = np.array([False, True, True])
        none = np.zeros(3, bool)
        self.assertAlmostEqual(sl.after_tax_growth(eq, held, none, 0, 2), 2.0 - self.T, places=12)
        self.assertAlmostEqual(sl.after_tax_growth(eq, held, none, 0, 2, exemption=0.3),
                               2.0 - self.T * 0.7, places=12)
        dates = np.array(["2019-12-31", "2020-12-31", "2021-12-31"])
        res = {"equity": eq, "in_fund": held, "switch": none}
        est = sl.after_tax_estimate(dates, res, 0, 2)
        years = 731 / 365.25
        self.assertEqual(est["cagr_pct"], round(((2.0 - self.T) ** (1 / years) - 1) * 100, 2))
        self.assertEqual(est["cagr_with_30pct_partial_exemption_pct"],
                         round(((2.0 - self.T * 0.7) ** (1 / years) - 1) * 100, 2))
        self.assertLess(est["cagr_pct"], est["cagr_with_30pct_partial_exemption_pct"])
        self.assertEqual(sl.leverage_period(dates, res, 0, 2)["after_tax_estimate_de"], est)
        self.assertIsNone(sl.after_tax_estimate(dates, res, 2, 2))

    def test_wiped_out_account(self):
        eq = np.array([1.0, 0.0, 0.0])
        self.assertEqual(sl.after_tax_growth(eq, np.array([False, True, False]),
                                             np.array([False, True, False]), 0, 2), 0.0)


def price_only(adj: np.ndarray) -> np.ndarray:
    """A 'close' that is NOT the dividend-adjusted close: it falls 1e-4 a day behind it
    (about 2.5% a year), so a benchmark read from the wrong column shows."""
    adj = np.asarray(adj, float)
    return adj * np.exp(-1e-4 * np.arange(len(adj)))


def lab_series(end: str = "2015-12-31", tbill=3.0, drag: dict | None = None,
               seed: int = 99, crash: tuple[str, int, float] | None = None
               ) -> dict[str, pd.DataFrame]:
    """Synthetic index / ETF / T-bill frames for leverage_block. The ETFs follow
    the lab's own fund model minus drag[ticker] per day (a number, or an array
    aligned to the ^NDX dates); tbill is a number or an array on those dates.
    crash = (first day, days, daily log return) is added to the index. The ETFs'
    and SPY's close is price-only (below adj_close); ^NDX and ^IRX carry a
    deliberately wrong adj_close (constants), so a read of the wrong column shows."""
    rng = np.random.default_rng(seed)
    nd = bdays("1985-10-01", end=end)
    lr = 0.0005 + 0.012 * rng.standard_normal(len(nd))
    if crash:
        k = int(np.searchsorted(nd, crash[0]))
        lr[k:k + crash[1]] += crash[2]
    close = 100 * np.exp(np.cumsum(lr))
    tb = np.full(len(nd), float(tbill)) if np.isscalar(tbill) else np.asarray(tbill, float)
    drag = drag or {}

    def etf(lev, ticker, start):
        k = int(np.searchsorted(nd, start))
        d = np.broadcast_to(np.asarray(drag.get(ticker, 0.0), float), (len(nd),))
        fund = sl.synthetic_fund_returns(close, tb, lev)
        adj = np.r_[40.0, 40.0 * np.cumprod(1 + fund[k + 1:] - d[k + 1:])]
        return pd.DataFrame({"close": price_only(adj), "adj_close": adj}, index=nd[k:])

    spy = 40.0 * np.exp(np.cumsum(0.0003 + 0.01 * np.random.default_rng(5)
                                  .standard_normal(len(nd))))
    return {"^NDX": pd.DataFrame({"close": close, "adj_close": 1.0}, index=nd),
            "^IRX": pd.DataFrame({"close": tb, "adj_close": 9.9}, index=nd),
            "QQQ": etf(1, "QQQ", "1999-03-10"), "QLD": etf(2, "QLD", "2006-06-21"),
            "TQQQ": etf(3, "TQQQ", "2010-02-11"),
            "SPY": pd.DataFrame({"close": price_only(spy), "adj_close": spy}, index=nd)}


class ValidationTest(unittest.TestCase):
    def setUp(self):
        n = 800
        rng = np.random.default_rng(4)
        self.dates = bdays("2005-01-03", n)
        self.close = 100 * np.exp(np.cumsum(0.0004 + 0.012 * rng.standard_normal(n)))
        self.tb = np.full(n, 3.0)

    def etf(self, lev: int, start: int, drag: float = 0.0) -> pd.DataFrame:
        fund = sl.synthetic_fund_returns(self.close, self.tb, lev)
        adj = np.r_[50.0, 50.0 * np.cumprod(1 + fund[start + 1:] - drag)]
        return pd.DataFrame({"close": price_only(adj), "adj_close": adj},
                            index=self.dates[start:])

    def test_cagrs_compound_over_the_calendar_days_from_the_etfs_first_bar(self):
        # a short live history (about two years), so one day more or less in the year
        # count moves the CAGR by several hundredths of a point
        etf = self.etf(2, 300, drag=0.0001)
        fund = sl.synthetic_fund_returns(self.close, self.tb, 2)
        v = sl.validate_synthetic(self.dates, fund, etf, "QLD", 2)
        adj = etf["adj_close"].to_numpy(float)
        days = (pd.Timestamp(str(self.dates[-1])) - pd.Timestamp(str(self.dates[300]))).days
        real = ((adj[-1] / adj[0]) ** (365.25 / days) - 1) * 100
        syn = (float(np.prod(1 + fund[301:])) ** (365.25 / days) - 1) * 100
        self.assertAlmostEqual(v["real_cagr_pct"], real, delta=0.0051)
        self.assertAlmostEqual(v["synthetic_cagr_pct"], syn, delta=0.0051)
        self.assertEqual(v["years"], round(days / 365.25, 2))
        # the price-only close would have shown a different fund
        close = etf["close"].to_numpy(float)
        self.assertGreater(abs(((close[-1] / close[0]) ** (365.25 / days) - 1) * 100 - real), 1.0)

    def test_a_perfect_fund_matches(self):
        fund = sl.synthetic_fund_returns(self.close, self.tb, 2)
        v = sl.validate_synthetic(self.dates, fund, self.etf(2, 300), "QLD", 2)
        self.assertEqual(v["start"], self.dates[300])
        self.assertEqual(v["days"], 800 - 301)
        self.assertAlmostEqual(v["cagr_diff_pp"], 0.0, places=6)
        self.assertAlmostEqual(v["daily_return_correlation"], 1.0, places=6)
        self.assertTrue(v["within_tolerance"])

    def test_a_deviation_above_one_point_is_flagged(self):
        fund = sl.synthetic_fund_returns(self.close, self.tb, 3)
        v = sl.validate_synthetic(self.dates, fund, self.etf(3, 300, drag=0.0002), "TQQQ", 3)
        self.assertGreater(v["cagr_diff_pp"], 1.0)         # the model is too optimistic
        self.assertFalse(v["within_tolerance"])
        self.assertEqual(v["tolerance_pp"], 1.0)

    def test_a_real_fund_that_beats_the_model_is_outside_the_tolerance_too(self):
        fund = sl.synthetic_fund_returns(self.close, self.tb, 3)
        v = sl.validate_synthetic(self.dates, fund, self.etf(3, 300, drag=-0.0002), "TQQQ", 3)
        self.assertLess(v["cagr_diff_pp"], -1.0)           # the model is too pessimistic
        self.assertFalse(v["within_tolerance"])
        self.assertNotIn("by_tbill_regime", v)             # no T-bill series given

    def test_split_by_period_and_by_t_bill_regime(self):
        n = len(self.dates)
        tb = np.r_[np.full(300, 5.0), np.full(250, 0.5), np.full(200, 2.0), np.full(n - 750, 0.5)]
        fund = sl.synthetic_fund_returns(self.close, tb, 2)
        drag = np.where(tb >= 3.0, 0.0, np.where(tb < 1.0, -0.0001, 0.0001))
        adj = np.r_[50.0, 50.0 * np.cumprod(1 + fund[1:] - drag[1:])]
        etf = pd.DataFrame({"close": adj, "adj_close": adj}, index=self.dates)
        v = sl.validate_synthetic(self.dates, fund, etf, "QLD", 2, tb)
        rows = {r["regime"]: r for r in v["by_tbill_regime"]}
        self.assertEqual(list(rows), ["below 1%", "1% to 3%", "at or above 3%"])
        self.assertEqual([rows[k]["days"] for k in rows], [n - 500, 200, 299])
        self.assertEqual([rows[k]["avg_tbill_pct"] for k in rows], [0.5, 2.0, 5.0])
        lo, mid, hi = (rows[k]["synthetic_minus_real_pp_per_year"] for k in rows)
        self.assertLess(lo, -1.0)                          # the real fund did better there
        self.assertGreater(mid, 1.0)
        self.assertAlmostEqual(hi, 0.0, places=6)
        self.assertEqual([rows[k]["within_tolerance"] for k in rows], [False, False, True])
        self.assertFalse(v["regimes_within_tolerance"])
        self.assertEqual(v["max_abs_regime_diff_pp"], round(max(abs(lo), abs(mid)), 2))
        # an independent annualisation of the low-rate days
        m = tb[1:] < 1.0
        real = adj[1:] / adj[:-1] - 1
        want = (np.prod(1 + fund[1:][m]) ** (252 / m.sum())
                - np.prod(1 + real[m]) ** (252 / m.sum())) * 100
        self.assertEqual(lo, round(float(want), 2))
        # thirds of the live history, in date order
        per = v["by_period"]
        self.assertEqual([r["days"] for r in per], [266, 267, 266])
        self.assertEqual(per[0]["start"], self.dates[1])
        self.assertEqual(per[-1]["end"], self.dates[-1])
        self.assertEqual(per[0]["avg_tbill_pct"], 5.0)
        self.assertAlmostEqual(per[0]["synthetic_minus_real_pp_per_year"], 0.0, places=6)

    def test_a_regime_with_too_few_days_is_not_judged(self):
        tb = np.r_[np.full(40, 5.0), np.full(len(self.dates) - 40, 0.5)]
        fund = sl.synthetic_fund_returns(self.close, tb, 2)
        adj = np.r_[50.0, 50.0 * np.cumprod(1 + fund[1:] - 0.01)]      # an absurd drag
        etf = pd.DataFrame({"close": adj, "adj_close": adj}, index=self.dates)
        v = sl.validate_synthetic(self.dates, fund, etf, "QLD", 2, tb)
        hi = v["by_tbill_regime"][2]
        self.assertEqual(hi["days"], 39)
        self.assertIsNone(hi["synthetic_minus_real_pp_per_year"])
        self.assertIsNone(hi["within_tolerance"])
        self.assertFalse(v["regimes_within_tolerance"])    # the low-rate row is far outside
        self.assertEqual(sl.B_VALIDATION_MIN_DAYS, 63)


class LeverageBlockTest(unittest.TestCase):
    """leverage_block on synthetic series: validation flags, evidence, sensitivities."""

    @staticmethod
    def block(**kw) -> dict:
        return sl.leverage_block(lab_series(**kw), n_paths=60, progress=lambda s: None)["block"]

    def test_a_real_fund_beating_the_model_is_flagged_and_voids_the_evidence(self):
        b = self.block(drag={"TQQQ": -0.0002})
        v = b["validation"]
        self.assertLess(v["TQQQ"]["cagr_diff_pp"], -1.0)
        self.assertFalse(v["TQQQ"]["within_tolerance"])
        self.assertTrue(v["QQQ"]["within_tolerance"] and v["QLD"]["within_tolerance"])
        self.assertTrue(v["flags"][0].startswith("MODEL DEVIATION: the synthetic 3x fund"))
        self.assertIn("TQQQ", v["flags"][0])
        self.assertIn(f"{v['TQQQ']['cagr_diff_pp']:+.2f}", v["flags"][0])
        self.assertFalse(v["usable_as_evidence"])

    def test_a_clean_model_is_usable_and_unflagged(self):
        v = self.block()["validation"]
        self.assertEqual(v["flags"], [])
        self.assertTrue(v["usable_as_evidence"])
        self.assertEqual(v["tbill_average_pct"], {"train": 3.0, "test": 3.0, "full": 3.0})
        for etf in ("QQQ", "QLD", "TQQQ"):
            self.assertTrue(v[etf]["regimes_within_tolerance"])
            self.assertEqual(len(v[etf]["by_period"]), 3)

    def test_offsetting_errors_by_regime_are_flagged_but_the_whole_life_fit_passes(self):
        nd = bdays("1985-10-01", end="2015-12-31")
        tb = np.where(nd < "2011-01-01", 0.5, 4.0)
        b = self.block(tbill=tb, drag={"QLD": np.where(tb >= 3.0, 0.0001, -0.0001)})
        v = b["validation"]
        self.assertTrue(v["QLD"]["within_tolerance"])       # the errors cancel over the whole life
        self.assertFalse(v["QLD"]["regimes_within_tolerance"])
        self.assertEqual(len(v["flags"]), 1)
        self.assertTrue(v["flags"][0].startswith("MODEL DEVIATION BY T-BILL REGIME"))
        self.assertIn("QLD", v["flags"][0])
        self.assertIn("below 1%", v["flags"][0])
        self.assertIn("at or above 3%", v["flags"][0])
        self.assertTrue(v["usable_as_evidence"])
        rows = {r["regime"]: r["synthetic_minus_real_pp_per_year"]
                for r in v["QLD"]["by_tbill_regime"]}
        self.assertLess(rows["below 1%"], -1.0)
        self.assertGreater(rows["at or above 3%"], 1.0)
        self.assertIsNone(rows["1% to 3%"])
        # the computed caveat names the era's average T-bill rate and the overstatement
        caveat = sl._regime_caveat(b)
        self.assertIn("hides offsetting errors", caveat)
        self.assertIn(f"2x: {rows['below 1%']:+.1f} pct-pt/yr while T-bills are below 1%, "
                      f"{rows['at or above 3%']:+.1f} while they are at or above 3%", caveat)
        # the TRAIN era was low-rate here, where the real 2x fund did better than the model
        self.assertIn("T-bills averaged 0.5%", caveat)
        self.assertIn(f"2x results for that era are probably understated by roughly "
                      f"{abs(rows['below 1%']):.1f} pct-pt", caveat)
        # the mirror image: a high-rate TRAIN era with the model too optimistic there
        tb2 = np.where(nd < "2011-01-01", 4.0, 0.5)
        b2 = self.block(tbill=tb2, drag={"QLD": np.where(tb2 >= 3.0, 0.0001, -0.0001),
                                         "TQQQ": np.where(tb2 >= 3.0, 0.0002, -0.0002)})
        caveat2 = sl._regime_caveat(b2)
        self.assertIn("T-bills averaged 4.0%", caveat2)
        self.assertIn("2x / 3x results for that era are probably overstated by roughly",
                      caveat2)
        self.assertIsNone(sl._regime_caveat(self.block()))

    def test_sensitivities_are_stored_next_to_the_headline(self):
        frames = lab_series()
        b = sl.leverage_block(frames, n_paths=60, progress=lambda s: None)["block"]
        dates = np.asarray(frames["^NDX"].index, dtype=str)
        close = frames["^NDX"]["close"].to_numpy(float)
        tb = np.full(len(dates), 3.0)
        a, split = int(np.searchsorted(dates, "1986-09-30")), int(
            np.searchsorted(dates, "2009-12-31"))
        # same-close execution, recomputed directly
        ts = b["timing_sensitivity"]
        self.assertEqual(ts["label"], "sensitivity - not the headline")
        self.assertEqual(set(ts["variants"]), {n for n in b["variants"] if "sma200" in n})
        r = sl.run_leverage(dates, close, tb, 2, "sma200", delay=0)
        m = sl.perf_metrics(dates, r["equity"], a, split)
        self.assertEqual(ts["variants"]["ndx2x_sma200"]["train_cagr_pct"], m["cagr_pct"])
        self.assertEqual(ts["variants"]["ndx2x_sma200"]["train_max_drawdown_pct"],
                         m["max_drawdown_pct"])
        self.assertNotEqual(m["cagr_pct"], b["variants"]["ndx2x_sma200"]["periods"]["train"][
            "cagr_pct"])
        same = {n: {"periods": {"train": {
            "cagr_pct": ts["variants"][n]["train_cagr_pct"],
            "max_drawdown_pct": ts["variants"][n]["train_max_drawdown_pct"]}}}
            if n in ts["variants"] else {"periods": b["variants"][n]["periods"]}
            for n in b["variants"]}
        self.assertEqual(ts["winner_same_close"], sl.pick_winner(same, -75.0)["name"])
        self.assertEqual(ts["winner_changes"], ts["winner_same_close"] != b["winner"]["name"])
        self.assertEqual(ts["filtered_variants"], 6)
        for key in ("delay_higher_train_cagr", "delay_shallower_train_drawdown",
                    "delay_lower_test_cagr"):
            self.assertTrue(0 <= ts[key] <= 6)
        self.assertEqual(ts["delay_higher_train_cagr"], sum(
            b["variants"][n]["periods"]["train"]["cagr_exact_pct"]
            > sl.perf_metrics(dates, sl.run_leverage(
                dates, close, tb, b["variants"][n]["leverage"], b["variants"][n]["filter"],
                delay=0)["equity"], a, split)["cagr_exact_pct"] for n in ts["variants"]))
        self.assertIn("date", ts["worst_day_held_after_sell_signal"])
        # dearer financing, recomputed directly
        cf = b["conservative_financing"]
        r = sl.run_leverage(dates, close, tb, 3, "none", div_yield=0.002, spread=0.010)
        self.assertEqual(cf["variants"]["ndx3x_none"]["full_cagr_pct"],
                         sl.perf_metrics(dates, r["equity"], a, len(dates) - 1)["cagr_pct"])
        for n, v in b["variants"].items():
            self.assertLess(cf["variants"][n]["train_cagr_pct"], v["periods"]["train"]["cagr_pct"])
        self.assertEqual(cf["winner_changes"], cf["winner"] != b["winner"]["name"])
        # bootstrap sensitivities for every variant
        sens = b["bootstrap"]["sensitivity"]
        self.assertEqual(sens["label"], "sensitivity - not the headline")
        for key in ("block_36_months", "drift_haircut"):
            self.assertEqual(set(sens[key]["variants"]), set(b["variants"]))
            for row in sens[key]["variants"].values():
                for k in ("p_cagr_ge_40_pct", "p_cagr_ge_20_pct",
                          "p_max_drawdown_worse_than_50_pct", "p_max_drawdown_worse_than_75_pct",
                          "p_end_below_start_pct"):
                    self.assertIn(k, row)
        self.assertEqual(sens["block_36_months"]["variants"]["ndx1x_none"]["block_months"], 36)
        self.assertEqual(sens["drift_haircut"]["haircut_pp_per_year"], 4.0)
        ms = sl.monthly_stats(dates, sl.run_leverage(dates, close, tb, 1, "none")["equity"], a,
                              len(dates) - 1)
        self.assertEqual(sens["block_36_months"]["variants"]["ndx1x_none"]["p_cagr_ge_20_pct"],
                         sl.block_bootstrap(ms, n_paths=60, block=36)["p_cagr_ge_20_pct"])

    def test_a_lower_index_return_lowers_the_chance_of_the_target(self):
        frames = lab_series()
        b = sl.leverage_block(frames, n_paths=400, progress=lambda s: None)["block"]
        head = b["bootstrap"]["variants"]["ndx3x_none"]
        cut = b["bootstrap"]["sensitivity"]["drift_haircut"]["variants"]["ndx3x_none"]
        self.assertGreater(head["p_cagr_ge_40_pct"], 0.0)
        self.assertLess(cut["p_cagr_ge_40_pct"], head["p_cagr_ge_40_pct"])
        self.assertLess(cut["p_cagr_ge_20_pct"], head["p_cagr_ge_20_pct"])
        self.assertGreater(cut["p_end_below_start_pct"], head["p_end_below_start_pct"])
        self.assertLess(cut["full_history_cagr_pct"],
                        b["variants"]["ndx3x_none"]["periods"]["full"]["cagr_pct"])
        # the 1x fund loses about the haircut itself
        one = b["bootstrap"]["sensitivity"]["drift_haircut"]["variants"]["ndx1x_none"]
        self.assertAlmostEqual(b["variants"]["ndx1x_none"]["periods"]["full"]["cagr_pct"]
                               - one["full_history_cagr_pct"], 4.0, delta=0.8)
        for word in ("by construction", "understate multi-year bear markets",
                     "future may be worse"):
            self.assertIn(word, b["bootstrap"]["warning"])
        caveat = sl._bootstrap_caveat(b)
        self.assertIn(f"from {head['p_cagr_ge_40_pct']:.1f}% to {cut['p_cagr_ge_40_pct']:.1f}%",
                      caveat)
        # the stored warning carries the same sensitivity, built from the two tables
        heads = b["bootstrap"]["variants"]
        cuts = b["bootstrap"]["sensitivity"]["drift_haircut"]["variants"]
        sentence = sl.haircut_fall_text(heads, cuts, 4.0)
        self.assertIn("If the Nasdaq had returned 4 points a year less", sentence)
        self.assertIn(sentence, b["bootstrap"]["warning"])

    def test_the_warning_sentence_on_a_lower_index_return(self):
        def p(x):
            return {"p_cagr_ge_40_pct": x}

        head = {"a": p(20.0), "b": p(10.0), "tiny": p(0.5), "none": None}
        cut = {"a": p(10.0), "b": p(2.0), "tiny": p(0.4), "none": None}
        s = sl.haircut_fall_text(head, cut, 4.0)
        # 20 -> 10 is a fall of 50%, 10 -> 2 of 80%; the variant below 1% is left out
        self.assertEqual(s, " If the Nasdaq had returned 4 points a year less, the chance of "
                            "40%/yr over 10 years falls by 50% to 80% of its value (the variants "
                            "with at least a 1% chance).")
        self.assertIn("falls by about 50% of its value",
                      sl.haircut_fall_text({"a": p(20.0)}, {"a": p(10.0)}, 4.0))
        # no variant with a real chance, a missing table, or a chance that rises: no sentence
        self.assertEqual(sl.haircut_fall_text({"tiny": p(0.5)}, {"tiny": p(0.1)}, 4.0), "")
        self.assertEqual(sl.haircut_fall_text({"a": p(20.0)}, {}, 4.0), "")
        self.assertEqual(sl.haircut_fall_text({"a": p(20.0), "b": p(10.0)},
                                              {"a": p(10.0), "b": p(12.0)}, 4.0), "")

    def test_winner_margins_and_after_tax_figures(self):
        b = self.block()
        w = b["winner"]
        sm = w["selection_margin"]
        tr = {n: b["variants"][n]["periods"]["train"] for n in w["qualified"]}
        runner = max((n for n in tr if n != w["name"]), key=lambda n: tr[n]["cagr_exact_pct"])
        self.assertEqual(sm["runner_up"], runner)
        self.assertEqual(sm["train_cagr_gap_to_runner_up_pp"], round(
            tr[w["name"]]["cagr_exact_pct"] - tr[runner]["cagr_exact_pct"], 2))
        self.assertEqual(sm["train_drawdown_distance_to_limit_pp"],
                         round(tr[w["name"]]["max_drawdown_exact_pct"] + 75.0, 2))
        t = w["test"]
        v = b["variants"][w["name"]]
        self.assertEqual(t["cagr_per_unit_drawdown"],
                         round(t["cagr_pct"] / abs(t["max_drawdown_pct"]), 2))
        base = b["variants"]["ndx1x_none"]["periods"]["test"]
        self.assertEqual(t["nasdaq_1x_cagr_per_unit_drawdown"],
                         round(base["cagr_pct"] / abs(base["max_drawdown_pct"]), 2))
        same = b["variants"][f"ndx{v['leverage']}x_none"]["periods"]["test"]
        self.assertEqual(t["same_leverage_unfiltered"]["name"], f"ndx{v['leverage']}x_none")
        self.assertEqual(t["same_leverage_unfiltered"]["cagr_pct"], same["cagr_pct"])
        self.assertEqual(t["same_leverage_unfiltered"]["max_drawdown_pct"],
                         same["max_drawdown_pct"])
        # buy-and-hold after tax: one tax at the end of the period
        full = b["variants"]["ndx1x_none"]["periods"]["full"]
        g = 1 + full["total_return_pct"] / 100
        est = full["after_tax_estimate_de"]
        self.assertAlmostEqual(est["cagr_pct"],
                               ((g - 0.26375 * (g - 1)) ** (1 / full["years"]) - 1) * 100,
                               delta=0.02)
        self.assertLess(est["cagr_pct"], est["cagr_with_30pct_partial_exemption_pct"])
        self.assertLess(est["cagr_with_30pct_partial_exemption_pct"], full["cagr_pct"])
        self.assertEqual(t["after_tax_estimate_de"], v["periods"]["test"]["after_tax_estimate_de"])
        self.assertEqual(t["nasdaq_1x_after_tax_estimate_de"], base["after_tax_estimate_de"])
        self.assertIn("26.375%", b["after_tax_model_de"]["model"])
        for word in ("allowance", "Vorabpauschale", "church tax"):
            self.assertIn(word, b["after_tax_model_de"]["simplifications"])

    def test_every_variant_is_measured_against_the_1x_fund(self):
        b = self.block()
        one = b["variants"]["ndx1x_none"]["periods"]
        signs = set()
        for name, v in b["variants"].items():
            for p in ("train", "test", "full"):
                got = v["periods"][p]["cagr_minus_1x_nasdaq_pp"]
                if name == "ndx1x_none":
                    self.assertIsNone(got)
                    continue
                want = round(v["periods"][p]["cagr_pct"] - one[p]["cagr_pct"], 2)
                self.assertAlmostEqual(got, want, places=9, msg=(name, p))   # variant minus 1x
                signs.add(want > 0)
        self.assertEqual(signs, {True, False})              # both signs occur in the fixture
        w = b["winner"]
        wv = b["variants"][w["name"]]
        self.assertNotEqual(w["name"], "ndx1x_none")
        for p in ("train", "test", "full"):
            self.assertEqual(w[p]["nasdaq_1x_cagr_pct"], one[p]["cagr_pct"])
            self.assertEqual(w[p]["nasdaq_1x_max_drawdown_pct"], one[p]["max_drawdown_pct"])
            self.assertEqual(w[p]["cagr_minus_1x_nasdaq_pp"],
                             wv["periods"][p]["cagr_minus_1x_nasdaq_pp"])
            self.assertNotEqual(w[p]["nasdaq_1x_cagr_pct"], w[p]["cagr_pct"])
        # the winner's own rolling windows and bootstrap, not the 1x fund's
        self.assertEqual(w["rolling_10y"], wv["rolling_10y"])
        self.assertNotEqual(w["rolling_10y"], b["variants"]["ndx1x_none"]["rolling_10y"])
        self.assertEqual(w["bootstrap_10y"], b["bootstrap"]["variants"][w["name"]])
        self.assertNotEqual(w["bootstrap_10y"], b["bootstrap"]["variants"]["ndx1x_none"])

    def test_benchmarks_and_the_winners_real_qqq_use_the_adjusted_close(self):
        frames = lab_series(drag={"QQQ": 0.00002})          # the real fund a little behind the model
        out = sl.leverage_block(frames, n_paths=60, progress=lambda s: None)
        bench, w = out["benchmarks"]["test"], out["block"]["winner"]
        years = (pd.Timestamp("2015-12-31") - pd.Timestamp("2009-12-31")).days / 365.25
        for etf in ("QQQ", "SPY"):
            adj = frames[etf]["adj_close"]
            want = ((adj["2015-12-31"] / adj["2009-12-31"]) ** (1 / years) - 1) * 100
            self.assertAlmostEqual(bench[etf]["cagr_pct"], want, delta=0.0051, msg=etf)
        self.assertEqual(w["test"]["qqq_real_cagr_pct"], bench["QQQ"]["cagr_pct"])
        self.assertEqual(w["test"]["qqq_real_cagr_exact_pct"], bench["QQQ"]["cagr_exact_pct"])
        self.assertEqual(w["test"]["qqq_real_max_drawdown_pct"], bench["QQQ"]["max_drawdown_pct"])
        self.assertNotEqual(w["test"]["qqq_real_cagr_pct"], w["test"]["nasdaq_1x_cagr_pct"])
        for p in ("train", "full"):                          # QQQ starts in 1999: TEST only
            self.assertNotIn("qqq_real_cagr_pct", w[p])

    def test_the_bootstrap_median_is_not_called_the_historical_average(self):
        # a crash month: paths that hold it are ruinous for the always-on leveraged funds
        b = self.block(crash=("2001-03-01", 21, -0.01))
        boot, variants = b["bootstrap"]["variants"], b["variants"]
        for name, v in variants.items():
            self.assertEqual(boot[name]["full_history_cagr_pct"], v["periods"]["full"]["cagr_pct"])
        three = boot["ndx3x_none"]
        self.assertNotEqual(three["cagr_pct"]["p50"], three["full_history_cagr_pct"])
        far = max(boot, key=lambda n: abs(boot[n]["cagr_pct"]["p50"]
                                          - boot[n]["full_history_cagr_pct"]))
        quote = (f"the median 10-year outcome can still differ from the whole-history CAGR "
                 f"({boot[far]['cagr_pct']['p50']:.1f}% against "
                 f"{boot[far]['full_history_cagr_pct']:.1f}% for {variants[far]['label']})")
        caveat = sl._bootstrap_caveat(b)
        for text in (b["bootstrap"]["warning"], caveat):
            self.assertIn(quote, text)
            self.assertIn("centred on the history by construction", text)
            self.assertNotIn("equals the historical average", text)
        # no comparable figure: the centring is still stated, without a quote
        self.assertEqual(sl.bootstrap_centre_text({"x": None}, {}),
                         sl.bootstrap_centre_text({}, {}))
        self.assertNotIn("median", sl.bootstrap_centre_text({}, {}))


# --------------------------------------------------------------------------- strategy A

class MomentumSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = 330
        cls.frames = {f"S{k:02d}": random_frame(200 + k, n=n, start="2019-01-01", price=60.0)
                      for k in range(6)}
        cls.panel = sl.build_panel(bt.frames_to_long(cls.frames))
        cls.sig_idx = sl.month_end_signal_idx(cls.panel.dates)
        cls.sig = sl.momentum_signals(cls.panel, cls.sig_idx)
        cls.j = len(cls.sig_idx) - 1
        cls.s = int(cls.sig_idx[cls.j])                    # the last month end

    def test_spec_constants(self):
        self.assertEqual(sl.A_LOOKBACKS, (126, 252))
        self.assertEqual((sl.A_SKIP, sl.A_MIN_PRIOR_BARS, sl.A_DV_WINDOW), (21, 273, 20))
        self.assertEqual(sl.A_TOP_N, (10, 20))
        self.assertEqual((sl.A_TRAIN_END, sl.A_TRAIN_DD_FLOOR_PCT), ("2022-12-31", -50.0))
        self.assertEqual(MIN_DOLLAR_VOL["us"], 5e6)

    def test_signal_days_are_month_ends_with_a_next_session(self):
        d = self.panel.dates
        for i in self.sig_idx:
            self.assertNotEqual(d[i][:7], d[i + 1][:7])
        self.assertLess(self.s, len(d) - 1)
        self.assertGreaterEqual(self.s, 273)

    def test_momentum_skips_the_latest_month(self):
        for k, t in enumerate(self.panel.tickers):
            c = self.frames[t]["close"].to_numpy(float)
            for L in (126, 252):
                want = c[self.s - 21] / c[self.s - 21 - L] - 1
                self.assertAlmostEqual(self.sig.mom[L][self.j, k], want, places=12)
        self.assertTrue(self.sig.eligible[self.j].all())

    def test_ranking_highest_first(self):
        top = sl.select_top(self.sig, self.j, 126, 3)
        m = self.sig.mom[126][self.j]
        self.assertEqual(list(top), list(np.argsort(-m)[:3]))
        self.assertEqual(len(sl.select_top(self.sig, self.j, 126, None)), 6)

    def test_ties_break_by_ticker(self):
        sig = manual_signals([5], [{"B": 0.3, "A": 0.3, "C": 0.1}], ["A", "B", "C"])
        self.assertEqual(list(sl.select_top(sig, 0, 126, 2)), [0, 1])

    def test_no_look_ahead(self):
        # the last 21 bars up to the signal day and everything after it may change freely
        frames = {t: f.copy() for t, f in self.frames.items()}
        for f in frames.values():
            f.iloc[self.s - 20:, f.columns.get_indexer(["open", "high", "low", "close"])] *= 3.0
        sig2 = sl.momentum_signals(sl.build_panel(bt.frames_to_long(frames)), self.sig_idx)
        for L in (126, 252):
            np.testing.assert_allclose(sig2.mom[L][self.j], self.sig.mom[L][self.j], rtol=1e-12)
        # bars after the signal day change nothing at all on or before it
        frames = {t: f.copy() for t, f in self.frames.items()}
        for f in frames.values():
            f.iloc[self.s + 1:] *= 0.01
        sig3 = sl.momentum_signals(sl.build_panel(bt.frames_to_long(frames)), self.sig_idx)
        for L in (126, 252):
            np.testing.assert_array_equal(sig3.mom[L][:self.j + 1], self.sig.mom[L][:self.j + 1])
        np.testing.assert_array_equal(sig3.eligible[:self.j + 1], self.sig.eligible[:self.j + 1])

    def test_needs_273_prior_bars(self):
        base = self.frames["S00"]
        frames = {"FULL": base, "OK": base.iloc[self.s - 273:], "SHORT": base.iloc[self.s - 272:]}
        panel = sl.build_panel(bt.frames_to_long(frames))
        sig = sl.momentum_signals(panel, np.array([self.s]))
        col = {t: k for k, t in enumerate(panel.tickers)}
        self.assertTrue(sig.eligible[0, col["FULL"]])
        self.assertTrue(sig.eligible[0, col["OK"]])
        self.assertFalse(sig.eligible[0, col["SHORT"]])
        # exactly 273 prior bars: the 12-month momentum starts at the stock's first bar
        c = base["close"].to_numpy(float)
        self.assertAlmostEqual(sig.mom[252][0, col["OK"]], c[self.s - 21] / c[self.s - 273] - 1,
                               places=12)

    def test_liquidity_floor(self):
        thin = self.frames["S01"].copy()
        thin["volume"] = 5e6 / thin["close"].to_numpy(float).max() * 0.5   # under 5m a day
        edge = self.frames["S02"].copy()
        edge["volume"] = 1e3
        edge.iloc[self.s - 19:self.s + 1, edge.columns.get_loc("volume")] = \
            5.1e6 / edge["close"].iloc[self.s - 19:self.s + 1].to_numpy(float)   # last 20 bars only
        panel = sl.build_panel(bt.frames_to_long(
            {"LIQ": self.frames["S00"], "THIN": thin, "EDGE": edge}))
        sig = sl.momentum_signals(panel, np.array([self.s - 1, self.s]))
        col = {t: k for k, t in enumerate(panel.tickers)}
        self.assertTrue(sig.eligible[1, col["LIQ"]])
        self.assertFalse(sig.eligible[1, col["THIN"]])
        self.assertTrue(sig.eligible[1, col["EDGE"]])        # 20-bar average just above the floor
        self.assertFalse(sig.eligible[0, col["EDGE"]])       # a day earlier one thin bar is inside

    def test_no_bar_on_the_signal_day(self):
        gap = self.frames["S03"].drop(self.frames["S03"].index[self.s])
        panel = sl.build_panel(bt.frames_to_long({"A": self.frames["S00"], "GAP": gap}))
        sig = sl.momentum_signals(panel, np.array([self.s]))
        self.assertFalse(sig.eligible[0, panel.tickers.index("GAP")])
        self.assertTrue(np.isnan(sig.mom[126][0, panel.tickers.index("GAP")]))

    def test_unexplained_jump_makes_a_stock_ineligible(self):
        panel = sl.build_panel(bt.frames_to_long(self.frames),
                               suspect={"S04": [self.panel.dates[self.s - 100]]})
        sig = sl.momentum_signals(panel, np.array([self.s]))
        self.assertFalse(sig.eligible[0, panel.tickers.index("S04")])
        self.assertTrue(sig.eligible[0, panel.tickers.index("S05")])

    def test_stray_dates_are_dropped(self):
        frames = {f"T{k}": random_frame(k, n=40, start="2020-01-01") for k in range(6)}
        stray = frames["T0"].iloc[-1:].copy()
        stray.index = pd.DatetimeIndex(["2020-02-29"], name="date")          # a Saturday
        frames["T0"] = pd.concat([frames["T0"], stray]).sort_index()
        panel = sl.build_panel(bt.frames_to_long(frames))
        self.assertEqual(panel.dropped_dates, 1)
        self.assertNotIn("2020-02-29", list(panel.dates))


class MomentumRunTest(unittest.TestCase):
    C = 0.002          # 0.15% + 0.05%

    def setUp(self):
        self.dates = bdays("2020-01-27", end="2020-04-03")
        d = self.dates
        self.s0 = int(np.nonzero(d == "2020-01-31")[0][0])
        self.s1 = int(np.nonzero(d == "2020-02-28")[0][0])
        self.s2 = int(np.nonzero(d == "2020-03-31")[0][0])
        self.n = len(d)

    def flat(self, price: float = 100.0) -> np.ndarray:
        return np.full(self.n, price)

    def test_cost_model_is_the_repos_us_model(self):
        self.assertAlmostEqual(sl.A_COST_RATE, self.C, places=15)
        self.assertEqual(sl.A_COST_PCT_PER_SIDE, BACKTEST_DEFAULTS["costs_pct_per_side"]["us"])
        self.assertEqual(sl.A_SLIPPAGE_PCT, BACKTEST_DEFAULTS["entry_slippage_pct"]["us"])

    def test_month_end_signal_days(self):
        self.assertEqual(list(sl.month_end_signal_idx(self.dates)), [self.s0, self.s1, self.s2])

    def test_trades_at_the_next_open_not_at_the_signal_close(self):
        o, c = self.flat(), self.flat()
        o[self.s0 + 1] = 110.0            # gap up after the signal
        c[self.s0 + 1:] = 121.0
        o[self.s0 + 2:] = 121.0
        panel = mini_panel(self.dates, {"A": o}, {"A": c})
        sig = manual_signals([self.s0], [{"A": 0.5}], panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1)
        eq = res["equity"]
        self.assertTrue(np.isnan(eq[self.s0 - 1]))
        self.assertEqual(eq[self.s0], 1.0)
        # bought at 110 (not 100, not 121): 1 / (1 + c) invested, marked at 121
        self.assertAlmostEqual(eq[self.s0 + 1], 121 / 110 / (1 + self.C), places=12)
        self.assertEqual(res["log"][0]["signal_date"], "2020-01-31")
        self.assertEqual(res["log"][0]["trade_date"], "2020-02-03")
        self.assertAlmostEqual(res["log"][0]["cost_pct"], 100 * self.C / (1 + self.C), places=10)

    def test_rotation_sells_leavers_and_charges_both_sides(self):
        oa, ca = self.flat(), self.flat()
        oa[self.s1 + 1:] = 120.0
        ca[self.s1 + 1:] = 120.0          # A is sold at the 120 open
        panel = mini_panel(self.dates, {"A": oa, "B": self.flat(50.0)},
                           {"A": ca, "B": self.flat(50.0)})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5, "B": 0.1}, {"A": 0.1, "B": 0.5}],
                             panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1)
        a1 = 1 / (1 + self.C)
        v = 1.2 * a1
        a2 = v * (1 - self.C) / (1 + self.C)
        self.assertAlmostEqual(res["equity"][self.s1], a1, places=12)       # A still at 100
        self.assertAlmostEqual(res["equity"][self.s1 + 1], a2, places=12)
        self.assertAlmostEqual(res["equity"][-1], a2, places=12)
        log = res["log"][1]
        self.assertEqual(log["tickers"], ["B"])
        self.assertAlmostEqual(log["traded_notional_pct"], 100 * (v + a2) / v, places=9)
        self.assertAlmostEqual(log["turnover_one_way_pct"], 100.0, places=9)
        # contributions: A made 20% of the equity it was bought with, B nothing
        self.assertAlmostEqual(res["contrib"]["A"], 0.2, places=12)
        self.assertAlmostEqual(res["contrib"]["B"], 0.0, places=12)

    def test_stayers_are_reweighted_and_only_the_traded_notional_costs(self):
        oa, ca = self.flat(), self.flat()
        oa[self.s1 + 1:] = 200.0
        ca[self.s1 + 1:] = 200.0          # A doubles; both stocks stay in the top 2
        panel = mini_panel(self.dates, {"A": oa, "B": self.flat()}, {"A": ca, "B": self.flat()})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5, "B": 0.4}] * 2, panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=2)
        a1 = 1 / (1 + self.C)
        # holdings a1 and a1/2 -> equal halves: the traded notional is a1/2 in total
        a2 = a1 * (1.5 - 0.5 * self.C)
        self.assertAlmostEqual(res["equity"][-1], a2, places=12)
        self.assertAlmostEqual(res["log"][1]["traded_notional_pct"], 100 * 0.5 / 1.5, places=9)
        # sells (A: a1 - a2/2) are the larger side: a1 * 0.25 * (1 + c)
        self.assertAlmostEqual(res["log"][1]["turnover_one_way_pct"],
                               100 * 0.25 * (1 + self.C) / 1.5, places=9)
        self.assertAlmostEqual(res["log"][1]["cost_pct"], 100 * self.C * 0.5 / 1.5, places=9)

    def test_no_trade_no_cost(self):
        panel = mini_panel(self.dates, {"A": self.flat()}, {"A": self.flat()})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5}] * 2, panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1)
        self.assertAlmostEqual(res["equity"][-1], 1 / (1 + self.C), places=12)
        self.assertAlmostEqual(res["log"][1]["traded_notional_pct"], 0.0, places=9)

    def test_solve_rebalance(self):
        after, traded = sl.solve_rebalance(100.0, np.array([60.0, 40.0]), np.array([0.0, 1.0]),
                                           0.002)
        # sell 60, buy after - 40: after = 100 - 0.002 * (60 + after - 40)
        self.assertAlmostEqual(after, (100 - 0.002 * 20) / 1.002, places=10)
        self.assertAlmostEqual(traded, 60 + after - 40, places=10)
        after, traded = sl.solve_rebalance(100.0, np.array([100.0]), np.array([0.0]), 0.002)
        self.assertAlmostEqual(after, 99.8, places=10)      # all to cash: one side only
        self.assertEqual(sl.solve_rebalance(100.0, np.array([50.0, 50.0]),
                                            np.array([0.5, 0.5]), 0.0)[0], 100.0)

    def test_trend_filter_goes_to_cash_at_the_next_open_and_back(self):
        o, c = self.flat(), self.flat()
        o[self.s1 + 1] = 90.0             # the exit open
        o[self.s1 + 2:] = 80.0
        c[self.s1 + 1:] = 80.0            # falls further once we are out
        panel = mini_panel(self.dates, {"A": o}, {"A": c})
        sig = manual_signals([self.s0, self.s1, self.s2], [{"A": 0.5}] * 3, panel.tickers)
        rf = np.full(self.n, 0.0001)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1, rf_daily=rf,
                              trend_ok=np.array([True, False, True]))
        eq = res["equity"]
        a1 = 1 / (1 + self.C)
        cash0 = a1 * 0.9 * (1 - self.C)                    # sold at the 90 open, one side of costs
        self.assertAlmostEqual(eq[self.s1], a1, places=12)                 # held through the signal close
        self.assertAlmostEqual(eq[self.s1 + 1], cash0, places=12)          # no interest on the exit day
        k = self.s2 - (self.s1 + 1)                        # full days in cash
        self.assertAlmostEqual(eq[self.s2], cash0 * 1.0001 ** k, places=12)
        # back in at the open after the third signal (80), no interest that day
        back = cash0 * 1.0001 ** k / (1 + self.C)
        self.assertAlmostEqual(eq[self.s2 + 1], back, places=12)
        self.assertEqual([r["invested"] for r in res["log"]], [True, False, True])
        m = sl.momentum_period(self.dates, res, self.s0, self.n - 1)
        self.assertEqual(m["months"], 3)
        self.assertEqual(m["months_invested_pct"], 66.7)
        self.assertEqual(m["distinct_stocks_held"], 1)
        # buy 1 / (1 + c) of the equity, sell all of it, buy 1 / (1 + c) again
        self.assertEqual(m["avg_turnover_pct_per_month"],
                         round((100 + 2 * 100 / (1 + self.C)) / 3, 1))

    def test_trend_flag_is_read_at_the_signal_close(self):
        d = bdays("2019-01-01", 260)
        close = np.r_[np.full(230, 100.0), 90.0, 90.0, 120.0, np.full(27, 120.0)]
        sig_days = np.array([d[150], d[229], d[230], d[231], d[232], d[240]])
        above, defined = sl.index_trend_flags(d, close, sig_days)
        # day 150: no 200-day average yet; day 229: equal to it (not above);
        # 230/231: below; 232: the jump counts on its own day, not a day earlier
        self.assertEqual(list(defined), [False, True, True, True, True, True])
        self.assertEqual(list(above), [False, False, False, False, True, True])
        # a signal day without an index bar uses the last bar before it
        above, defined = sl.index_trend_flags(np.delete(d, 233), np.delete(close, 233),
                                              np.array([d[233]]))
        self.assertEqual((bool(above[0]), bool(defined[0])), (True, True))
        # what happens after the signal day never matters
        later = close.copy()
        later[233:] = 1.0
        np.testing.assert_array_equal(sl.index_trend_flags(d, later, sig_days[:5])[0],
                                      [False, False, False, False, True])

    def test_filter_off_ignores_the_trend(self):
        panel = mini_panel(self.dates, {"A": self.flat()}, {"A": self.flat()})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5}] * 2, panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1, trend_ok=None)
        self.assertEqual([r["invested"] for r in res["log"]], [True, True])

    def test_cash_earns_the_t_bill_rate_before_the_first_trade_only_overnight(self):
        panel = mini_panel(self.dates, {"A": self.flat()}, {"A": self.flat()})
        sig = manual_signals([self.s0], [{"A": 0.5}], panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1,
                              rf_daily=np.full(self.n, 0.01))
        # fully invested from the first open: no interest at all
        self.assertAlmostEqual(res["equity"][-1], 1 / (1 + self.C), places=12)

    def test_fewer_eligible_than_n_stays_fully_invested(self):
        panel = mini_panel(self.dates, {"A": self.flat(), "B": self.flat()},
                           {"A": self.flat(), "B": self.flat()})
        sig = manual_signals([self.s0], [{"A": 0.5, "B": 0.2}], panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=10, cost_rate=0.0)
        self.assertEqual(res["log"][0]["n_held"], 2)
        self.assertAlmostEqual(res["equity"][-1], 1.0, places=12)

    def test_a_stock_without_a_bar_is_sold_at_its_last_close(self):
        oa, ca = self.flat(), self.flat()
        ca[self.s1] = 130.0
        oa[self.s1 + 1:] = np.nan
        ca[self.s1 + 1:] = np.nan         # A stops trading after the second signal
        panel = mini_panel(self.dates, {"A": oa, "B": self.flat()}, {"A": ca, "B": self.flat()})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5}, {"B": 0.5}], panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1, cost_rate=0.0)
        self.assertEqual(res["fills_without_bar"], 1)
        self.assertAlmostEqual(res["equity"][-1], 1.3, places=12)

    def test_starts_at_the_chosen_signal_and_stops_at_i_end(self):
        panel = mini_panel(self.dates, {"A": self.flat()}, {"A": self.flat()})
        sig = manual_signals([self.s0, self.s1, self.s2], [{"A": 0.5}] * 3, panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1, first_signal=1, i_end=self.s2)
        self.assertEqual(res["i_start"], self.s1)
        self.assertTrue(np.isnan(res["equity"][self.s1 - 1]))
        self.assertTrue(np.isnan(res["equity"][self.s2 + 1]))
        self.assertEqual(len(res["log"]), 1)

    def test_equal_weight_universe(self):
        d = self.dates[:4]
        panel = mini_panel(d, {"A": [10, 10, 10, 10], "B": [10, 10, np.nan, 10]},
                           {"A": [10, 11, 11, 22], "B": [10, 9, np.nan, 9.9]})
        ew, clipped = sl.equal_weight_returns(panel)
        # day 1: (+10%, -10%); day 2: A only; day 3: A +100% capped at +50%, B +10% over its gap
        np.testing.assert_allclose(ew[1:], [0.0, 0.0, (0.5 + 0.1) / 2], atol=1e-12)
        self.assertEqual(clipped, 1)
        eq = sl.curve_from_returns(ew, 1, 3)
        self.assertTrue(np.isnan(eq[0]))
        np.testing.assert_allclose(eq[1:], [1.0, 1.0, 1.3])

    def test_equal_weight_buy_and_hold_is_not_the_daily_rebalanced_basket(self):
        d = self.dates[:6]
        up = [10.0, 12.0, 14.4, 17.28, 20.736, 24.8832]          # +20% a day
        flat = [10.0] * 6
        panel = mini_panel(d, {"A": up, "B": flat}, {"A": up, "B": flat})
        eq, n = sl.equal_weight_buy_and_hold(panel, 0, 5)
        self.assertEqual(n, 2)
        # equal amounts at the start, never touched: half the money rides the winner
        np.testing.assert_allclose(eq, (np.asarray(up) / 10.0 + 1.0) / 2, rtol=1e-12)
        ew, _ = sl.equal_weight_returns(panel)
        daily = sl.curve_from_returns(ew, 0, 5)
        np.testing.assert_allclose(daily, 1.1 ** np.arange(6), rtol=1e-12)   # +10% every day
        self.assertGreater(eq[-1], daily[-1] * 1.05)
        # a later start, a stock without a bar at the start, and a missing bar on the way
        a = [10.0, 10.0, 20.0, np.nan, 30.0, 30.0]
        b = [np.nan, 5.0, 5.0, 5.0, 5.0, 10.0]
        c = [7.0, np.nan, 7.0, 7.0, 7.0, 7.0]                   # no bar on the anchor day
        panel = mini_panel(d, {"A": a, "B": b, "C": c}, {"A": a, "B": b, "C": c})
        eq, n = sl.equal_weight_buy_and_hold(panel, 1, 5)
        self.assertEqual(n, 2)
        self.assertTrue(np.isnan(eq[0]))
        np.testing.assert_allclose(eq[1:], [1.0, 1.5, 1.5, 2.0, 2.5], rtol=1e-12)
        # the window end is respected and an empty basket gives no curve
        part, _ = sl.equal_weight_buy_and_hold(panel, 1, 3)
        self.assertTrue(np.isnan(part[4]))
        none = mini_panel(d, {"A": [np.nan] * 6}, {"A": [np.nan] * 6})
        self.assertEqual(sl.equal_weight_buy_and_hold(none, 0, 5)[1], 0)

    def test_placebo_pickers(self):
        t = ["A", "B", "C", "D"]
        sig = manual_signals([5, 9], [{"A": 0.4, "B": 0.1, "C": 0.3, "D": -0.2},
                                      {"A": 0.4, "B": 0.1, "C": 0.3}], t)
        self.assertEqual(list(sl.select_top(sig, 0, 126, 2)), [0, 2])
        self.assertEqual(list(sl.select_bottom(sig, 0, 126, 2)), [3, 1])
        self.assertEqual(list(sl.select_bottom(sig, 1, 126, 2)), [1, 2])     # D is not eligible
        draws = {tuple(sl.select_random(sig, 0, 126, 2, seed)) for seed in range(40)}
        self.assertGreater(len(draws), 3)                    # the seeds differ ...
        for seed in range(5):                                # ... each one is repeatable
            a = sl.select_random(sig, 0, 126, 2, seed)
            self.assertEqual(list(a), list(sl.select_random(sig, 0, 126, 2, seed)))
            self.assertEqual(len(set(a)), 2)
            self.assertTrue(set(sl.select_random(sig, 1, 126, 2, seed)) <= {0, 1, 2})
        self.assertEqual(list(sl.select_random(sig, 1, 126, 5, 0)), [0, 1, 2])
        # one stock made ineligible everywhere; the original signals are untouched
        ex = sl.without_ticker(sig, 0)
        self.assertEqual(list(sl.select_top(ex, 0, 126, 2)), [2, 1])
        self.assertTrue(sig.eligible[0, 0])

    def test_a_custom_picker_replaces_the_ranking_only(self):
        panel = mini_panel(self.dates, {"A": self.flat(), "B": self.flat(50.0)},
                           {"A": self.flat(), "B": self.flat(50.0)})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5, "B": 0.1}] * 2, panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1,
                              selector=lambda j: sl.select_bottom(sig, j, 126, 1))
        self.assertEqual([r["tickers"] for r in res["log"]], [["B"], ["B"]])
        self.assertAlmostEqual(res["equity"][-1], 1 / (1 + self.C), places=12)   # same costs
        # the trend filter still overrides the picker
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1, trend_ok=np.array([True, False]),
                              selector=lambda j: sl.select_bottom(sig, j, 126, 1))
        self.assertEqual([r["invested"] for r in res["log"]], [True, False])
        # without its only pick the top-1 rule moves to the other stock
        res = sl.run_momentum(panel, sl.without_ticker(sig, 0), lookback=126, top_n=1)
        self.assertEqual([r["tickers"] for r in res["log"]], [["B"], ["B"]])

    def test_concentration(self):
        oa, ca = self.flat(), self.flat()
        oa[self.s1 + 1:] = 150.0
        ca[self.s1:] = 150.0              # +50% on the last day of February
        panel = mini_panel(self.dates, {"A": oa}, {"A": ca})
        sig = manual_signals([self.s0, self.s1], [{"A": 0.5}] * 2, panel.tickers)
        res = sl.run_momentum(panel, sig, lookback=126, top_n=1, cost_rate=0.0, i_end=self.s2)
        conc = sl.concentration(self.dates, res, self.s0, self.s2)
        self.assertEqual(conc["best_month"], {"month": "2020-02", "return_pct": 50.0})
        self.assertAlmostEqual(conc["cagr_without_best_month_pct"], 0.0, places=6)
        self.assertEqual(conc["top_contributors"][0]["ticker"], "A")
        self.assertEqual(conc["top_contributors"][0]["months_held"], 2)


def trend_flags_by_hand(index_dates: np.ndarray, close: np.ndarray, signal_dates, window: int
                        ) -> np.ndarray:
    """Per signal day: the index's last close on or before it above the mean of its own
    last `window` closes (that close included)."""
    out = np.zeros(len(signal_dates), bool)
    for j, day in enumerate(signal_dates):
        k = int(np.searchsorted(index_dates, day, side="right")) - 1
        out[j] = close[k] > close[k - window + 1:k + 1].mean()
    return out


class MomentumBlockWiringTest(unittest.TestCase):
    """momentum_block's own wiring - the ^GSPC trend flag, the T-bill cash rate, the first
    signal, the series limit and the eligible basket - on a hand-made panel whose index
    series are deterministic."""

    @classmethod
    def setUpClass(cls):
        cls.dates = d = bdays("2021-01-04", end="2023-12-29")
        n = len(d)
        rng = np.random.default_rng(11)
        # 15 stocks from the first day and 10 that list later: the 20th stock has its 273
        # earlier bars only in May 2022, four month ends after the first 15
        cls.starts = [0] * 15 + [20, 35, 50, 65, 80, 95, 110, 125, 140, 155]
        opens, closes = {}, {}
        for k, s0 in enumerate(cls.starts):
            c = 50 * np.exp(np.cumsum(0.0004 + 0.0001 * k + 0.012 * rng.standard_normal(n)))
            o = c * np.exp(0.003 * rng.standard_normal(n))
            c[:s0], o[:s0] = np.nan, np.nan
            opens[f"T{k:02d}"], closes[f"T{k:02d}"] = o, c
        cls.panel = mini_panel(d, opens, closes, volume=1e7)
        cls.sig_idx = sl.month_end_signal_idx(d)
        cls.sig_dates = d[cls.sig_idx]
        # ^GSPC and ^NDX: deterministic waves (own calendar from 2020) that cross their
        # 200-day averages in different months
        gd = bdays("2020-01-02", end="2023-12-29")
        t = np.arange(len(gd))
        gspc = 100 * np.exp(0.06 * np.sin(2 * np.pi * t / 160.0) + 0.0002 * t)
        ndx = 100 * np.exp(0.06 * np.sin(2 * np.pi * (t + 50) / 160.0) + 0.0002 * t)
        # on the 2023-04-28 signal the 199- and the 200-day average disagree: a high bar 199
        # bars earlier lifts only the 200-day one, and the close sits between the two
        k = int(np.searchsorted(gd, "2023-04-28"))
        gspc[k - 199] *= 1.5
        lo = gspc[k - 198:k].mean()
        hi = (gspc[k - 198:k].sum() + gspc[k - 199]) / 199
        gspc[k] = (lo + hi) / 2
        cls.gd, cls.gspc, cls.ndx = gd, gspc, ndx
        # ^IRX: a different yield every day, on a calendar with gaps (carried forward)
        irx_d = gd[np.arange(len(gd)) % 5 != 3]
        irx = 1.0 + 3.0 * np.abs(np.sin(np.arange(len(irx_d)) / 23.0))
        qqq = 30 * np.exp(np.cumsum(0.0005 + 0.01 * np.random.default_rng(3).standard_normal(len(gd))))
        spy = 40 * np.exp(np.cumsum(0.0003 + 0.01 * np.random.default_rng(4).standard_normal(len(gd))))
        cls.frames = {
            "^GSPC": pd.DataFrame({"close": gspc, "adj_close": 1.0}, index=gd),   # wrong column
            "^NDX": pd.DataFrame({"close": ndx, "adj_close": ndx}, index=gd),
            "^IRX": pd.DataFrame({"close": irx, "adj_close": 9.9}, index=irx_d),
            "QQQ": pd.DataFrame({"close": price_only(qqq), "adj_close": qqq}, index=gd),
            "SPY": pd.DataFrame({"close": price_only(spy), "adj_close": spy}, index=gd),
        }
        # the T-bill rate of each panel day, carried forward by hand
        cls.rf = (pd.Series(irx, index=pd.DatetimeIndex(irx_d))
                  .reindex(pd.DatetimeIndex(d), method="ffill").to_numpy(float) / 100.0 / 252)
        calls: list = []
        real = sl.run_momentum

        def record(*a, **kw):
            res = real(*a, **kw)
            calls.append((kw, res))
            return res

        with mock.patch.object(sl, "run_momentum", side_effect=record):
            cls.out = sl.momentum_block(cls.panel, cls.frames, n_placebo=2, progress=lambda s: None)
        cls.calls = calls
        cls.j0 = int(np.nonzero(cls.sig_dates == cls.out["info"]["first_signal"])[0][0])
        cls.i_end = int(np.nonzero(d == cls.out["info"]["last_day"])[0][0])

    def run_of(self, lookback: int, top_n: int, filtered: bool) -> dict:
        """The block's own run of a grid variant (the first matching call)."""
        return next(r for kw, r in self.calls
                    if kw.get("lookback") == lookback and kw.get("top_n") == top_n
                    and (kw.get("trend_ok") is not None) == filtered and kw.get("selector") is None)

    def flags(self, close=None, window: int = 200) -> np.ndarray:
        return trend_flags_by_hand(self.gd, self.gspc if close is None else close,
                                   self.sig_dates, window)

    def test_first_signal_needs_twenty_eligible_stocks(self):
        info = self.out["info"]
        counts = [sum(1 for s0 in self.starts if s - s0 >= 273) for s in self.sig_idx]
        j = next(k for k, c in enumerate(counts) if c >= 20)
        self.assertEqual(info["first_signal"], "2022-05-31")
        self.assertEqual(info["first_signal"], self.sig_dates[j])
        self.assertEqual(info["eligible_at_first_signal"], counts[j])
        self.assertTrue(0 < counts[j - 1] < 20)             # earlier month ends had too few
        self.assertEqual(info["last_day"], "2023-12-29")
        self.assertEqual(info["series_limit"], "2023-12-29")
        self.assertFalse(info["shortened_by_series_cache"])

    def test_cash_months_follow_the_sp500_close_against_its_own_200_day_average(self):
        log = self.run_of(126, 10, True)["log"]
        traded = [r["signal_date"] for r in log]
        self.assertEqual(traded, list(self.sig_dates[self.j0:self.j0 + len(log)]))

        def cash_months(flags) -> set:
            return {s for s, f in zip(self.sig_dates, flags) if s in traded and not f}

        want = cash_months(self.flags())
        self.assertEqual({r["signal_date"] for r in log if not r["invested"]}, want)
        self.assertTrue(0 < len(want) < len(traded))
        self.assertIn("2023-04-28", want)
        # the fixture tells apart every reading a wiring slip would give
        for slip, flags in (("199-day average", self.flags(window=199)),
                            ("^NDX instead of ^GSPC", self.flags(self.ndx)),
                            ("the next signal's flag", np.roll(self.flags(), -1)),
                            ("adj_close (a constant here)", np.zeros(len(self.sig_dates), bool))):
            self.assertNotEqual(cash_months(flags), want, slip)
        # no signal close sits on its average, so the hand-made flags are unambiguous
        for j in range(self.j0, len(self.sig_dates)):
            k = int(np.searchsorted(self.gd, self.sig_dates[j]))
            for w in (199, 200):
                self.assertGreater(abs(self.gspc[k] / self.gspc[k - w + 1:k + 1].mean() - 1), 1e-4)
        # without the filter: invested every month
        self.assertTrue(all(r["invested"] for r in self.run_of(126, 10, False)["log"]))

    def test_cash_earns_each_days_t_bill_yield(self):
        run = self.run_of(126, 10, True)
        eq, log = run["equity"], run["log"]
        checked = {"after cash": 0, "after stocks": 0}
        for k, r in enumerate(log[:-1]):
            if r["invested"]:
                continue
            s, s_next = r["trade_idx"] - 1, log[k + 1]["trade_idx"] - 1
            if k == 0 or not log[k - 1]["invested"]:
                # already in cash at the trade open: interest on every day, the trade day too
                got, want = eq[s_next] / eq[s], np.prod(1 + self.rf[s + 1:s_next + 1])
                checked["after cash"] += 1
            else:
                # sold at the trade open: the day's balance was invested, interest from the next day
                got, want = eq[s_next] / eq[s + 1], np.prod(1 + self.rf[s + 2:s_next + 1])
                checked["after stocks"] += 1
            self.assertAlmostEqual(got, want, delta=1e-12 * want, msg=r["signal_date"])
        self.assertGreaterEqual(min(checked.values()), 2)
        self.assertGreater(np.ptp(self.rf[self.sig_idx[self.j0]:]), 1.0 / 100 / 252)

    def test_the_block_runs_on_these_flags_not_the_next_signals(self):
        sig = sl.momentum_signals(self.panel, self.sig_idx)
        args = dict(lookback=126, top_n=10, rf_daily=self.rf, first_signal=self.j0,
                    i_end=self.i_end)
        mine = sl.run_momentum(self.panel, sig, trend_ok=self.flags(), **args)
        shifted = sl.run_momentum(self.panel, sig, trend_ok=np.roll(self.flags(), -1), **args)
        i0 = int(self.sig_idx[self.j0])
        np.testing.assert_allclose(self.run_of(126, 10, True)["equity"][i0:self.i_end + 1],
                                   mine["equity"][i0:self.i_end + 1], rtol=1e-13)
        pooled = self.out["block"]["variants"]["mom126_top10_filter_on"]["periods"]["pooled"]
        self.assertAlmostEqual(pooled["total_return_pct"],
                               100 * (mine["equity"][self.i_end] - 1), delta=0.0051)
        self.assertGreater(abs(pooled["total_return_pct"]
                               - 100 * (shifted["equity"][self.i_end] - 1)), 0.05)

    def test_the_eligible_basket_is_rebalanced_monthly_without_costs(self):
        p = self.panel
        shares = np.zeros(len(p.tickers))
        for j in range(self.j0, len(self.sig_idx)):
            s = int(self.sig_idx[j])
            if s >= self.i_end:
                break
            d = s + 1
            value = 1.0 if j == self.j0 else float(np.nansum(shares * p.O[d]))
            elig = [k for k, s0 in enumerate(self.starts) if s - s0 >= 273]
            shares[:] = 0.0
            shares[elig] = value / len(elig) / p.O[d, elig]
        growth = float(np.nansum(shares * p.C[self.i_end]))
        got = self.out["benchmarks"]["pooled"]["equal_weight_eligible_monthly"]
        self.assertAlmostEqual(got["total_return_pct"], 100 * (growth - 1), delta=0.0051)

    def test_a_series_cache_that_ends_early_stops_strategy_a_there(self):
        for early, later in (("^IRX", "^GSPC"), ("^GSPC", "^IRX")):
            frames = dict(self.frames)
            frames[early] = frames[early][frames[early].index <= "2023-10-13"]
            frames[later] = frames[later][frames[later].index <= "2023-11-17"]
            lines: list[str] = []
            info = sl.momentum_block(self.panel, frames, n_placebo=0, progress=lines.append)["info"]
            self.assertEqual(info["series_limit"], str(frames[early].index[-1]), early)
            self.assertTrue(info["shortened_by_series_cache"], early)
            self.assertEqual(info["last_day"], "2023-09-29", early)
            self.assertTrue(any(line.startswith("WARN [A]") for line in lines), early)


# --------------------------------------------------------------------------- winner rule

def variant(train_cagr, train_dd, test_cagr=0.0, full_cagr=0.0, full_key="full",
            exact: dict | None = None) -> dict:
    """exact = {(period, key): unrounded figure} adds the *_exact_pct fields."""
    v = {"label": "x", "periods": {
        "train": {"cagr_pct": train_cagr, "max_drawdown_pct": train_dd},
        "test": {"cagr_pct": test_cagr, "max_drawdown_pct": -10.0},
        full_key: {"cagr_pct": full_cagr, "max_drawdown_pct": -10.0}}}
    for (period, key), value in (exact or {}).items():
        v["periods"][period][key[:-4] + "_exact_pct"] = value
    return v


class WinnerRuleTest(unittest.TestCase):
    def test_highest_train_cagr_inside_the_drawdown_limit(self):
        vs = {"a": variant(30.0, -80.0, test_cagr=99.0), "b": variant(20.0, -60.0),
              "c": variant(25.0, -75.0), "d": variant(10.0, -20.0, test_cagr=50.0)}
        pick = sl.pick_winner(vs, -75.0)
        self.assertEqual(pick["name"], "c")                # -75.0 itself is 'no worse than'
        self.assertEqual(pick["qualified"], ["b", "c", "d"])
        self.assertEqual(list(pick["disqualified"]), ["a"])
        self.assertIn("-80.0%", pick["disqualified"]["a"])
        self.assertEqual(sl.pick_winner(vs, -50.0)["name"], "d")

    def test_test_figures_never_pick_the_winner(self):
        vs = {"a": variant(10.0, -10.0, test_cagr=5.0), "b": variant(9.0, -10.0, test_cagr=80.0)}
        self.assertEqual(sl.pick_winner(vs, -50.0)["name"], "a")
        self.assertEqual(sl.best_on(vs, "test"), "b")

    def test_tie_goes_to_the_first_in_grid_order(self):
        vs = {"a": variant(10.0, -10.0), "b": variant(10.0, -5.0)}
        self.assertEqual(sl.pick_winner(vs, -50.0)["name"], "a")

    def test_nobody_qualifies(self):
        vs = {"a": variant(30.0, -80.0), "b": variant(None, None)}
        pick = sl.pick_winner(vs, -75.0)
        self.assertIsNone(pick["name"])
        self.assertEqual(pick["qualified"], [])
        self.assertEqual(set(pick["disqualified"]), {"a", "b"})

    def test_reaches_target_needs_test_and_whole_history(self):
        vs = {"w": variant(50.0, -10.0, test_cagr=45.0, full_cagr=30.0),
              "x": variant(5.0, -10.0, test_cagr=41.0, full_cagr=42.0)}
        r = sl._reaches(vs, "w", "full")
        self.assertTrue(r["winner_test_ge_target"])
        self.assertFalse(r["winner_whole_history_ge_target"])
        self.assertEqual(r["variants_ge_target_test"], ["w", "x"])
        self.assertEqual(r["variants_ge_target_whole_history"], ["x"])
        self.assertEqual(r["variants_ge_target_train"], ["w"])
        both = sl._reaches(vs, "x", "full")
        self.assertTrue(both["winner_test_ge_target"] and both["winner_whole_history_ge_target"])
        none = sl._reaches(vs, None, "full")
        self.assertFalse(none["winner_test_ge_target"] or none["winner_whole_history_ge_target"])

    def test_the_target_boundary_uses_unrounded_figures(self):
        vs = {"at": variant(5.0, -10.0, test_cagr=40.0, full_cagr=40.0),
              "just_under": variant(5.0, -10.0, test_cagr=40.0, full_cagr=40.0,
                                    exact={("test", "cagr_pct"): 39.996,
                                           ("full", "cagr_pct"): 39.9987}),
              "just_over": variant(5.0, -10.0, test_cagr=40.0, full_cagr=40.0,
                                   exact={("test", "cagr_pct"): 40.0004,
                                          ("full", "cagr_pct"): 40.0})}
        r = sl._reaches(vs, "at", "full")
        # exactly 40% counts; 39.996% (displayed as 40.0) does not
        self.assertTrue(r["winner_test_ge_target"] and r["winner_whole_history_ge_target"])
        self.assertEqual(r["variants_ge_target_test"], ["at", "just_over"])
        self.assertEqual(r["variants_ge_target_whole_history"], ["at", "just_over"])
        under = sl._reaches(vs, "just_under", "full")
        self.assertFalse(under["winner_test_ge_target"])
        self.assertFalse(under["winner_whole_history_ge_target"])

    def test_the_drawdown_limit_uses_unrounded_figures(self):
        vs = {"over": variant(30.0, -50.0, exact={("train", "max_drawdown_pct"): -50.004}),
              "at": variant(20.0, -50.0, exact={("train", "max_drawdown_pct"): -50.0}),
              "in": variant(10.0, -50.0, exact={("train", "max_drawdown_pct"): -49.996})}
        pick = sl.pick_winner(vs, -50.0)
        self.assertEqual(pick["qualified"], ["at", "in"])
        self.assertEqual(pick["name"], "at")
        self.assertIn("-50.004%", pick["disqualified"]["over"])
        # the ranking uses the unrounded CAGR as well
        vs = {"a": variant(10.0, -10.0, exact={("train", "cagr_pct"): 9.996}),
              "b": variant(10.0, -10.0, exact={("train", "cagr_pct"): 10.004})}
        self.assertEqual(sl.pick_winner(vs, -50.0)["name"], "b")
        self.assertEqual(sl.best_on(vs, "train"), "b")

    def test_selection_margin(self):
        vs = {"a": variant(30.0, -80.0), "b": variant(20.0, -60.0),
              "c": variant(25.25, -74.0, exact={("train", "cagr_pct"): 25.254,
                                                ("train", "max_drawdown_pct"): -73.996}),
              "d": variant(24.0, -20.0)}
        pick = sl.pick_winner(vs, -75.0)
        m = sl.selection_margin(vs, pick, -75.0)
        self.assertEqual(m["runner_up"], "d")              # the next QUALIFIED variant, not 'a'
        self.assertEqual(m["train_cagr_gap_to_runner_up_pp"], 1.25)
        self.assertEqual(m["train_drawdown_distance_to_limit_pp"], 1.0)
        self.assertEqual(m["train_drawdown_limit_pct"], -75.0)
        only = {"a": variant(10.0, -10.0)}
        m = sl.selection_margin(only, sl.pick_winner(only, -50.0), -50.0)
        self.assertIsNone(m["runner_up"])
        self.assertIsNone(m["train_cagr_gap_to_runner_up_pp"])
        self.assertEqual(m["train_drawdown_distance_to_limit_pp"], 40.0)
        self.assertIsNone(sl.selection_margin(vs, sl.pick_winner(vs, -10.0), -10.0))

    def test_years_text(self):
        self.assertEqual(sl._years_text(["1989", "1990"]), "1989-1990")
        self.assertEqual(sl._years_text(["2011"]), "2011")
        self.assertEqual(sl._years_text(["1989", "1990", "2011", "2016"]),
                         "1989-1990, 2011 or 2016")
        self.assertEqual(sl._years_text(["2016", "1990", "1989", "1991"]), "1989-1991 or 2016")


# --------------------------------------------------------------------------- series cache

def tiny_frames(tickers=sl.SERIES_TICKERS, n: int = 5) -> dict[str, pd.DataFrame]:
    dates = bdays("2020-01-01", n)
    return {t: pd.DataFrame({"close": np.arange(n) + 10.0 + k, "adj_close": np.arange(n) + 9.0 + k},
                            index=dates) for k, t in enumerate(tickers)}


class SeriesCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cache" / "lab_series.sqlite3"
        self.calls: list[list[str]] = []

    def fetch(self, tickers):
        self.calls.append(list(tickers))
        return tiny_frames(tickers)

    def boom(self, tickers):
        raise AssertionError("the network must not be used")

    def quiet(self, *a):
        pass

    def test_offline_fails_clearly_without_a_cache(self):
        with self.assertRaises(FileNotFoundError) as cm:
            sl.ensure_series(self.path, offline=True, fetch=self.boom, progress=self.quiet)
        self.assertIn("--offline", str(cm.exception))
        self.assertIn("missing", str(cm.exception))
        self.assertFalse(self.path.exists())

    def test_downloads_once_then_reuses_the_cache(self):
        frames, info = sl.ensure_series(self.path, fetch=self.fetch, progress=self.quiet)
        self.assertEqual(self.calls, [list(sl.SERIES_TICKERS)])
        self.assertEqual(info["downloaded_this_run"], list(sl.SERIES_TICKERS))
        self.assertTrue(self.path.exists())
        again, info2 = sl.ensure_series(self.path, fetch=self.boom, progress=self.quiet)
        self.assertEqual(info2["downloaded_this_run"], [])
        off, _ = sl.ensure_series(self.path, offline=True, fetch=self.boom, progress=self.quiet)
        for t in sl.SERIES_TICKERS:
            pd.testing.assert_frame_equal(frames[t], again[t])
            pd.testing.assert_frame_equal(frames[t], off[t])
        self.assertEqual(info2["series"]["^NDX"]["first"], "2020-01-01")
        self.assertEqual(info2["series"]["^NDX"]["rows"], 5)

    def test_refresh_downloads_everything_again(self):
        sl.ensure_series(self.path, fetch=self.fetch, progress=self.quiet)
        sl.ensure_series(self.path, refresh=True, fetch=self.fetch, progress=self.quiet)
        self.assertEqual(self.calls, [list(sl.SERIES_TICKERS)] * 2)

    def test_a_missing_series_is_fetched_alone_or_fails_offline(self):
        some = tiny_frames(sl.SERIES_TICKERS[:-1])
        sl.save_series(self.path, some, {t: "2026-01-01T00:00:00" for t in some})
        with self.assertRaises(FileNotFoundError) as cm:
            sl.ensure_series(self.path, offline=True, fetch=self.boom, progress=self.quiet)
        self.assertIn(sl.SERIES_TICKERS[-1], str(cm.exception))
        _, info = sl.ensure_series(self.path, fetch=self.fetch, progress=self.quiet)
        self.assertEqual(self.calls, [[sl.SERIES_TICKERS[-1]]])
        self.assertEqual(info["series"]["^NDX"]["fetched_at"], "2026-01-01T00:00:00")

    def test_offline_and_refresh_exclude_each_other(self):
        with self.assertRaises(ValueError):
            sl.ensure_series(self.path, offline=True, refresh=True, fetch=self.boom)

    def test_an_empty_download_is_an_error(self):
        with self.assertRaises(RuntimeError):
            sl.ensure_series(self.path, fetch=lambda ts: {}, progress=self.quiet)
        self.assertFalse(self.path.exists())

    def test_fetch_drops_the_unfinished_session_and_keeps_adjusted_closes(self):
        idx = pd.DatetimeIndex(["2026-09-29", "2026-09-30", "2026-10-01"], name="Date")
        raw = pd.DataFrame({"Adj Close": [9.0, 9.5, 9.9], "Close": [10.0, 10.5, 10.9],
                            "Open": 1.0, "High": 1.0, "Low": 1.0, "Volume": 0}, index=idx)
        with mock.patch("yfinance.download", return_value=raw) as dl:
            got = sl.fetch_series(["QQQ"], today="2026-10-01")
        self.assertEqual(dl.call_args.args[0], "QQQ")
        self.assertFalse(dl.call_args.kwargs["auto_adjust"])
        self.assertEqual(list(got["QQQ"].index), ["2026-09-29", "2026-09-30"])
        self.assertEqual(list(got["QQQ"]["adj_close"]), [9.0, 9.5])
        self.assertEqual(list(got["QQQ"]["close"]), [10.0, 10.5])
        with mock.patch("yfinance.download", return_value=raw.iloc[:0]):
            with self.assertRaises(RuntimeError):
                sl.fetch_series(["QQQ"], today="2026-10-01")

    def test_asof_values(self):
        got = sl.asof_values(np.array(["2020-01-02", "2020-01-06"]), np.array([1.0, 2.0]),
                             np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-07"]))
        self.assertTrue(np.isnan(got[0]))
        self.assertEqual(list(got[1:]), [1.0, 1.0, 2.0])


# --------------------------------------------------------------------------- CLI

class CliTest(unittest.TestCase):
    def test_parse_opts(self):
        self.assertEqual(sl.parse_opts([]), {"offline": False, "refresh": False, "out": None})
        self.assertEqual(sl.parse_opts(["--offline", "--out", "x.json"]),
                         {"offline": True, "refresh": False, "out": "x.json"})
        self.assertEqual(sl.parse_opts(["--out=y.json", "--refresh"]),
                         {"offline": False, "refresh": True, "out": "y.json"})
        for bad in (["--bogus"], ["--out"], ["--offline", "--refresh"], ["--offline=1"], ["us"]):
            with self.assertRaises(ValueError):
                sl.parse_opts(bad)

    def test_main_reports_a_bad_option(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            sl.main(["--bogus"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("unknown option", err.getvalue())
        self.assertIn("strategy-lab", err.getvalue())

    def test_main_passes_the_options_and_reports_a_missing_cache(self):
        with mock.patch.object(sl, "run_strategy_lab") as run:
            sl.main(["--offline", "--out", "o.json"])
        run.assert_called_once_with(out_path="o.json", offline=True, refresh=False)
        err = io.StringIO()
        with mock.patch.object(sl, "run_strategy_lab",
                               side_effect=FileNotFoundError("--offline: the series cache is missing")), \
                redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            sl.main(["--offline"])
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("series cache is missing", err.getvalue())

    def test_atlas2_cli_dispatch(self):
        from atlas2 import __main__ as cli
        with mock.patch.object(sl, "main") as m, mock.patch("sys.argv",
                                                            ["atlas2", "strategy-lab", "--offline"]):
            cli.main()
        m.assert_called_once_with(["--offline"])
        self.assertIn("strategy-lab", cli.__doc__)

    def test_equity_output_path(self):
        self.assertEqual(sl.equity_output_path(Path("/x/strategy_lab.json")),
                         Path("/x/strategy_lab_equity.json"))


# --------------------------------------------------------------------------- end to end

class EndToEndTest(unittest.TestCase):
    """The whole lab on a synthetic price cache and synthetic index series."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.root = root
        cls.data = root / "data"
        (cls.data / "cache").mkdir(parents=True)
        (root / "universe").mkdir()
        # stocks: 30 tickers, 2016-07-01 .. 2026-09-30
        n = len(pd.bdate_range("2016-07-01", "2026-09-30"))
        cls.prices = cls.data / "cache" / "prices.sqlite3"
        store = PriceStore(cls.prices)
        tickers = [f"S{k:02d}" for k in range(30)]
        cls.stock_closes = {}
        for k, t in enumerate(tickers):
            df = random_frame(700 + k, n=n, start="2016-07-01", drift=0.0002 + 0.00003 * k,
                              vol=0.02)
            cls.stock_closes[t] = df["close"]
            store.upsert(t, df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                               "close": "Close", "volume": "Volume"}))
        store.close()
        cls.universe = root / "universe" / "us.csv"
        cls.universe.write_text("ticker,name,index\n" + "".join(f"{t},{t},S&P 500\n"
                                                                 for t in tickers))
        # index / ETF / T-bill series, 1985 .. 2026-09-30
        rng = np.random.default_rng(99)
        nd = bdays("1985-10-01", end="2026-09-30")
        close = 100 * np.exp(np.cumsum(0.0005 + 0.012 * rng.standard_normal(len(nd))))
        long = bdays("1985-01-02", end="2026-09-30")
        tb = np.full(len(nd), 3.0)

        def walk(dates, seed, price):
            return price * np.exp(np.cumsum(0.0003 + 0.01 * np.random.default_rng(seed)
                                            .standard_normal(len(dates))))

        def etf(lev, start, drag=0.0):
            k = int(np.searchsorted(nd, start))
            fund = sl.synthetic_fund_returns(close, tb, lev)
            adj = np.r_[40.0, 40.0 * np.cumprod(1 + fund[k + 1:] - drag)]
            return pd.DataFrame({"close": price_only(adj), "adj_close": adj}, index=nd[k:])

        # the funds' close is price-only (below the dividend-adjusted close); the indices and
        # the T-bill yield carry a deliberately wrong adj_close: the lab must read
        # adj_close for the funds and close for the rest
        spy_d = bdays("1993-01-29", end="2026-09-30")
        spy = walk(spy_d, 5, 40.0)
        cls.frames = {
            "^NDX": pd.DataFrame({"close": close, "adj_close": 1.0}, index=nd),
            "^IRX": pd.DataFrame({"close": 3.0, "adj_close": 9.9}, index=long),
            "QQQ": etf(1, "1999-03-10"),
            "QLD": etf(2, "2006-06-21", drag=0.0002),      # the real fund lags the model
            "TQQQ": etf(3, "2010-02-11"),
            "SPY": pd.DataFrame({"close": price_only(spy), "adj_close": spy}, index=spy_d),
            "^GSPC": pd.DataFrame({"close": walk(long, 6, 200.0), "adj_close": 1.0}, index=long),
        }
        cls.series = cls.data / "cache" / "lab_series.sqlite3"
        sl.save_series(cls.series, cls.frames, {t: "2026-10-01T00:00:00" for t in cls.frames})
        cls.out = root / "out" / "strategy_lab.json"
        cls.lines: list[str] = []
        cls.before = hashlib.sha256(cls.prices.read_bytes()).hexdigest()
        cls.payload = cls.run_lab(cls.out, cls.lines)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_lab(cls, out: Path, lines: list[str] | None = None, **kw) -> dict:
        def boom(tickers):
            raise AssertionError("the network must not be used")

        args = dict(offline=True, prices_path=cls.prices, series_path=cls.series,
                    universe_path=cls.universe, data_dir=cls.data, cfg={}, fetch=boom,
                    n_paths=200, n_placebo=5,
                    progress=(lines.append if lines is not None else lambda s: None))
        args.update(kw)
        return sl.run_strategy_lab(out, **args)

    def strict_json(self, path: Path) -> dict:
        def bad(x):
            raise AssertionError(f"non-finite number in JSON: {x}")

        return json.loads(path.read_text(), parse_constant=bad)

    def test_files_and_top_level_shape(self):
        d = self.strict_json(self.out)
        for key in ("generated", "target_cagr_pct", "method", "benchmarks", "momentum", "leverage",
                    "answer", "caveats"):
            self.assertIn(key, d)
        self.assertEqual(d["target_cagr_pct"], 40.0)
        self.assertEqual(set(d["momentum"]), {"variants", "winner", "survivorship"})
        self.assertEqual(set(d["leverage"]), {"variants", "winner", "validation", "bootstrap",
                                              "timing_sensitivity", "conservative_financing",
                                              "after_tax_model_de"})
        self.assertIn("sensitivity", d["leverage"]["bootstrap"])
        self.assertEqual(self.strict_json(self.out), json.loads(json.dumps(bt._clean_json(self.payload))))
        self.assertEqual(list(self.out.parent.iterdir()).__len__(), 2)      # no temp files left

    def test_price_cache_is_not_modified(self):
        self.assertEqual(hashlib.sha256(self.prices.read_bytes()).hexdigest(), self.before)

    def test_momentum_grid_split_and_metrics(self):
        mom = self.payload["momentum"]
        self.assertEqual(list(mom["variants"]), [
            "mom126_top10_filter_off", "mom126_top10_filter_on", "mom126_top20_filter_off",
            "mom126_top20_filter_on", "mom252_top10_filter_off", "mom252_top10_filter_on",
            "mom252_top20_filter_off", "mom252_top20_filter_on"])
        win = self.payload["method"]["momentum"]["window"]
        # plain business days from 2016-07-01: bar 273 is 2017-07-19, so the first month end
        # with 273 prior bars is July 2017 (2017-06-30 has only 260)
        self.assertEqual(win["first_signal"], "2017-07-31")
        self.assertEqual(win["train"], ["2017-07-31", "2022-12-30"])
        self.assertEqual(win["test"], ["2022-12-30", "2026-09-30"])
        self.assertEqual(win["eligible_at_first_signal"], 30)
        for name, v in mom["variants"].items():
            for p in ("train", "test", "pooled"):
                m = v["periods"][p]
                for key in ("cagr_pct", "total_return_pct", "max_drawdown_pct", "worst_year",
                            "ann_volatility_pct", "avg_turnover_pct_per_month",
                            "months_invested_pct", "distinct_stocks_held", "cagr_minus_qqq_pp",
                            "cagr_minus_equal_weight_buy_and_hold_pp",
                            "cagr_minus_equal_weight_daily_rebalanced_pp",
                            "cagr_minus_equal_weight_eligible_pp", "yearly_pct"):
                    self.assertIsNotNone(m[key], (name, p, key))
                self.assertLessEqual(m["max_drawdown_pct"], 0.0)
            self.assertEqual(v["periods"]["train"]["start"], "2017-07-31")
            self.assertEqual(v["periods"]["test"]["start"], "2022-12-30")
            self.assertEqual(list(v["periods"]["test"]["yearly_pct"]),
                             ["2023", "2024", "2025", "2026"])
            self.assertEqual(v["periods"]["pooled"]["months"], 110)      # Jul 2017 .. Aug 2026
            self.assertEqual(v["periods"]["train"]["months"], 65)        # signals Jul 2017 .. Nov 2022
            self.assertEqual(v["periods"]["test"]["months"], 45)         # Dec 2022 .. Aug 2026
            if not v["trend_filter"]:
                self.assertEqual(v["periods"]["pooled"]["months_invested_pct"], 100.0)
            self.assertLessEqual(v["periods"]["pooled"]["distinct_stocks_held"], 30)
        off, on = mom["variants"]["mom126_top10_filter_off"], mom["variants"]["mom126_top10_filter_on"]
        self.assertLess(on["periods"]["pooled"]["months_invested_pct"], 100.0)
        self.assertGreaterEqual(off["periods"]["pooled"]["distinct_stocks_held"], 10)
        # train and test chain into pooled
        t, s, p = (off["periods"][k]["total_return_pct"] for k in ("train", "test", "pooled"))
        self.assertAlmostEqual((1 + t / 100) * (1 + s / 100), 1 + p / 100, places=3)

    def test_momentum_benchmarks_share_the_days(self):
        b = self.payload["benchmarks"]["momentum_window"]
        for p, (a, e) in {"train": ("2017-07-31", "2022-12-30"), "test": ("2022-12-30", "2026-09-30"),
                          "pooled": ("2017-07-31", "2026-09-30")}.items():
            for k in ("QQQ", "SPY", "equal_weight_buy_and_hold",
                      "equal_weight_universe_daily_rebalanced", "equal_weight_eligible_monthly"):
                self.assertEqual((b[p][k]["start"], b[p][k]["end"]), (a, e), (p, k))
        fresh = b["test"]["equal_weight_buy_and_hold_from_test_start"]
        self.assertEqual((fresh["start"], fresh["end"]), ("2022-12-30", "2026-09-30"))
        self.assertNotIn("equal_weight_buy_and_hold_from_test_start", b["train"])
        qqq = self.frames["QQQ"]["adj_close"]
        want = (qqq["2026-09-30"] / qqq["2022-12-30"] - 1) * 100
        self.assertAlmostEqual(b["test"]["QQQ"]["total_return_pct"], want, places=2)
        # QQQ and SPY from the dividend-adjusted close (the fixture's close is price-only)
        for p, (a, e) in {"train": ("2017-07-31", "2022-12-30"),
                          "test": ("2022-12-30", "2026-09-30"),
                          "pooled": ("2017-07-31", "2026-09-30")}.items():
            years = (pd.Timestamp(e) - pd.Timestamp(a)).days / 365.25
            for etf in ("QQQ", "SPY"):
                adj, close = self.frames[etf]["adj_close"], self.frames[etf]["close"]
                want = ((adj[e] / adj[a]) ** (1 / years) - 1) * 100
                self.assertAlmostEqual(b[p][etf]["cagr_pct"], want, delta=0.0051, msg=(p, etf))
                self.assertGreater(abs(((close[e] / close[a]) ** (1 / years) - 1) * 100 - want),
                                   0.5)
        v = self.payload["momentum"]["variants"]["mom252_top20_filter_off"]["periods"]["test"]
        self.assertAlmostEqual(v["cagr_minus_qqq_pp"],
                               v["cagr_pct"] - b["test"]["QQQ"]["cagr_pct"], places=2)
        self.assertAlmostEqual(
            v["cagr_minus_equal_weight_buy_and_hold_pp"],
            v["cagr_pct"] - b["test"]["equal_weight_buy_and_hold"]["cagr_pct"], places=2)
        self.assertAlmostEqual(
            v["cagr_minus_equal_weight_daily_rebalanced_pp"],
            v["cagr_pct"] - b["test"]["equal_weight_universe_daily_rebalanced"]["cagr_pct"],
            places=2)
        self.assertNotEqual(v["cagr_minus_equal_weight_buy_and_hold_pp"],
                            v["cagr_minus_equal_weight_daily_rebalanced_pp"])

    def test_winner_rules(self):
        for fam, floor, key in (("momentum", -50.0, "pooled"), ("leverage", -75.0, "full")):
            block = self.payload[fam]
            ok = [n for n, v in block["variants"].items()
                  if v["periods"]["train"]["max_drawdown_pct"] >= floor]
            want = max(ok, key=lambda n: block["variants"][n]["periods"]["train"]["cagr_pct"]) \
                if ok else None
            self.assertEqual(block["winner"]["name"], want)
            self.assertEqual(block["winner"]["qualified"], ok)
            bt_ = block["winner"]["best_test_variant"]
            self.assertEqual(bt_["label"], "in-sample on test — do not act on this")
            self.assertEqual(bt_["name"], max(block["variants"], key=lambda n: block["variants"][n][
                "periods"]["test"]["cagr_pct"]))
            if want:
                self.assertEqual(block["winner"]["test"]["cagr_pct"],
                                 block["variants"][want]["periods"]["test"]["cagr_pct"])
        w = self.payload["momentum"]["winner"]
        if w["name"]:
            bench = self.payload["benchmarks"]["momentum_window"]
            for p in ("train", "test", "pooled"):
                self.assertEqual(w[p]["qqq_cagr_pct"], bench[p]["QQQ"]["cagr_pct"])
                self.assertEqual(w[p]["spy_cagr_pct"], bench[p]["SPY"]["cagr_pct"])
                self.assertEqual(w[p]["qqq_max_drawdown_pct"], bench[p]["QQQ"]["max_drawdown_pct"])
                self.assertNotEqual(w[p]["qqq_cagr_pct"], w[p]["spy_cagr_pct"])
            b = self.payload["benchmarks"]["momentum_window"]["test"]
            self.assertEqual(w["test"]["equal_weight_buy_and_hold_cagr_pct"],
                             b["equal_weight_buy_and_hold"]["cagr_pct"])
            self.assertEqual(w["test"]["equal_weight_daily_rebalanced_cagr_pct"],
                             b["equal_weight_universe_daily_rebalanced"]["cagr_pct"])
            self.assertEqual(w["test"]["equal_weight_buy_and_hold_from_test_start_cagr_pct"],
                             b["equal_weight_buy_and_hold_from_test_start"]["cagr_pct"])

    def test_momentum_winner_diagnostics(self):
        mom = self.payload["momentum"]
        w = mom["winner"]
        self.assertIsNotNone(w["name"])
        v = mom["variants"][w["name"]]
        tr = {n: mom["variants"][n]["periods"]["train"] for n in w["qualified"]}
        sm = w["selection_margin"]
        self.assertEqual(sm["train_drawdown_distance_to_limit_pp"],
                         round(tr[w["name"]]["max_drawdown_exact_pct"] + 50.0, 2))
        others = [n for n in tr if n != w["name"]]
        if others:
            runner = max(others, key=lambda n: tr[n]["cagr_exact_pct"])
            self.assertEqual(sm["runner_up"], runner)
            self.assertEqual(sm["train_cagr_gap_to_runner_up_pp"], round(
                tr[w["name"]]["cagr_exact_pct"] - tr[runner]["cagr_exact_pct"], 2))
        # the winner re-run without its top contributor
        conc = w["concentration"]
        ex = conc["cagr_without_top_contributor_pct"]
        self.assertEqual(ex["ticker"], conc["top_contributors"][0]["ticker"])
        for period in ("train", "test", "pooled"):
            self.assertIsNotNone(ex[period])
        self.assertNotEqual(ex["pooled"], v["periods"]["pooled"]["cagr_pct"])
        self.assertEqual(v["concentration"], conc)
        # how noisy the TEST figure is
        t = v["periods"]["test"]
        self.assertEqual(w["test_noise"]["standard_error_pp"],
                         round(t["ann_volatility_pct"] / math.sqrt(t["years"]), 1))
        self.assertEqual(w["train"]["months"], 65)
        # placebos: same rules, other pickers
        pl = mom["survivorship"]["placebo"]
        self.assertEqual(pl["label"], "diagnostic, not a pre-registered metric")
        self.assertEqual(pl["random_n"]["seeds"], 5)
        for period in ("train", "test", "pooled"):
            r = pl["random_n"][period]
            self.assertTrue(r["p5_cagr_pct"] <= r["median_cagr_pct"] <= r["p95_cagr_pct"]
                            <= r["max_cagr_pct"])
            self.assertEqual(r["winner_cagr_pct"], v["periods"][period]["cagr_pct"])
            self.assertIn(r["winner_percentile"], (0.0, 20.0, 40.0, 60.0, 80.0, 100.0))
            self.assertIsNotNone(pl["bottom_n"][period]["cagr_pct"])
            self.assertLessEqual(pl["bottom_n"][period]["max_drawdown_pct"], 0.0)
        self.assertNotEqual(pl["bottom_n"]["pooled"]["cagr_pct"],
                            v["periods"]["pooled"]["cagr_pct"])
        lq = mom["survivorship"]["liquidity_rule"]
        self.assertEqual(lq["signal_date"], "2026-08-31")
        self.assertEqual((lq["stocks_with_a_bar"], lq["eligible"],
                          lq["excluded_by_liquidity_rule"]), (30, 30, 0))

    def test_survivorship_block(self):
        s = self.payload["momentum"]["survivorship"]
        b = self.payload["benchmarks"]["momentum_window"]["pooled"]
        bh, daily = b["equal_weight_buy_and_hold"], b["equal_weight_universe_daily_rebalanced"]
        # the gap is measured with the buy-and-hold basket, not the daily-rebalanced one
        self.assertAlmostEqual(s["universe_minus_spy_pp"]["pooled"],
                               bh["cagr_pct"] - b["SPY"]["cagr_pct"], places=2)
        self.assertAlmostEqual(s["universe_minus_qqq_pp"]["pooled"],
                               bh["cagr_pct"] - b["QQQ"]["cagr_pct"], places=2)
        self.assertGreater(abs(bh["cagr_pct"] - daily["cagr_pct"]), 0.05)
        self.assertGreater(abs(b["SPY"]["cagr_pct"] - b["QQQ"]["cagr_pct"]), 0.05)
        self.assertNotAlmostEqual(s["universe_minus_spy_pp"]["pooled"],
                                  daily["cagr_pct"] - b["SPY"]["cagr_pct"], places=2)
        # the basket recomputed from the stocks themselves: equal amounts at the first
        # signal's close, never rebalanced
        growth = float(np.mean([c["2026-09-30"] / c["2017-07-31"]
                                for c in self.stock_closes.values()]))
        years = (pd.Timestamp("2026-09-30") - pd.Timestamp("2017-07-31")).days / 365.25
        self.assertAlmostEqual(bh["cagr_pct"], (growth ** (1 / years) - 1) * 100, places=2)
        self.assertEqual(s["equal_weight_buy_and_hold"]["stocks"], 30)
        self.assertEqual(s["equal_weight_buy_and_hold"]["cagr_pct"]["pooled"], bh["cagr_pct"])
        fresh = float(np.mean([c["2026-09-30"] / c["2022-12-30"]
                               for c in self.stock_closes.values()]))
        y2 = (pd.Timestamp("2026-09-30") - pd.Timestamp("2022-12-30")).days / 365.25
        at_test = s["equal_weight_buy_and_hold"]["bought_at_test_start"]
        self.assertAlmostEqual(at_test["cagr_pct"], (fresh ** (1 / y2) - 1) * 100, places=2)
        self.assertEqual(at_test["anchor"], "2022-12-30")
        t = self.payload["benchmarks"]["momentum_window"]["test"]
        self.assertAlmostEqual(at_test["minus_spy_pp"],
                               at_test["cagr_pct"] - t["SPY"]["cagr_pct"], places=2)
        self.assertEqual(s["equal_weight_universe_daily_rebalanced_cagr_pct"]["label"],
                         "not investable (cost-free daily rebalancing)")
        self.assertIn(f"made {bh['cagr_pct']:.1f}%/yr", s["caveat"])
        self.assertIn("not investable", " ".join(self.payload["benchmarks"]["notes"]))
        self.assertIn("SURVIVORSHIP", s["caveat"])
        self.assertIn("UPWARD", s["caveat"])
        self.assertIn(s["caveat"], self.payload["caveats"])

    def test_leverage_grid_split_and_metrics(self):
        lev = self.payload["leverage"]
        self.assertEqual(list(lev["variants"]), [
            "ndx1x_none", "ndx1x_sma200", "ndx1x_sma200_band", "ndx2x_none", "ndx2x_sma200",
            "ndx2x_sma200_band", "ndx3x_none", "ndx3x_sma200", "ndx3x_sma200_band"])
        win = self.payload["method"]["leverage"]["window"]
        self.assertEqual(win["first_day"], "1986-10-01")
        self.assertEqual(win["anchor"], "1986-09-30")
        self.assertEqual(win["train"], ["1986-09-30", "2009-12-31"])
        self.assertEqual(win["test"], ["2009-12-31", "2026-09-30"])
        self.assertEqual(win["full_months"], 480)
        for name, v in lev["variants"].items():
            for p in ("train", "test", "full"):
                m = v["periods"][p]
                for key in ("cagr_pct", "max_drawdown_pct", "worst_year", "longest_underwater_days",
                            "ann_volatility_pct", "switches_per_year", "time_in_market_pct",
                            "yearly_pct"):
                    self.assertIsNotNone(m[key], (name, p, key))
            self.assertEqual(list(v["periods"]["test"]["yearly_pct"])[0], "2010")
            self.assertEqual(v["periods"]["full"]["partial_years"], ["1986", "2026"])
            r = v["rolling_10y"]
            self.assertEqual(r["n_windows"], 480 - 120 + 1)
            self.assertEqual(r["first_window"], ["1986-09-30", "1996-09-30"])
            self.assertLessEqual(r["min_cagr_pct"], r["median_cagr_pct"])
            self.assertLessEqual(r["median_cagr_pct"], r["max_cagr_pct"])
            if v["filter"] == "none":
                self.assertEqual(v["periods"]["full"]["switches_per_year"], 0.0)
                self.assertEqual(v["periods"]["full"]["time_in_market_pct"], 100.0)
            else:
                self.assertGreater(v["periods"]["full"]["switches_per_year"], 0.0)
                self.assertLess(v["periods"]["full"]["time_in_market_pct"], 100.0)
        self.assertIsNone(lev["variants"]["ndx1x_none"]["rolling_10y"]["share_ge_1x_nasdaq_pct"])
        self.assertIsNotNone(lev["variants"]["ndx2x_none"]["rolling_10y"]["share_ge_1x_nasdaq_pct"])
        # the band switches less often than the plain filter
        self.assertLess(lev["variants"]["ndx1x_sma200_band"]["periods"]["full"]["switches"],
                        lev["variants"]["ndx1x_sma200"]["periods"]["full"]["switches"])
        # the same signal for every leverage
        self.assertEqual(lev["variants"]["ndx1x_sma200"]["periods"]["full"]["switches"],
                         lev["variants"]["ndx3x_sma200"]["periods"]["full"]["switches"])

    def test_unfiltered_one_x_matches_the_formula(self):
        f = self.frames["^NDX"]
        nd = np.asarray(f.index, dtype=str)
        c = f["close"].to_numpy(float)
        a, e = int(np.searchsorted(nd, "1986-09-30")), len(nd) - 1
        r = c[a + 1:] / c[a:-1] - 1 + (0.007 - 0.002) / 252
        growth = float(np.prod(1 + r))
        years = (pd.Timestamp(str(nd[e])) - pd.Timestamp(str(nd[a]))).days / 365.25
        got = self.payload["leverage"]["variants"]["ndx1x_none"]["periods"]["full"]
        self.assertAlmostEqual(got["cagr_pct"], (growth ** (1 / years) - 1) * 100, places=2)
        idx = self.payload["benchmarks"]["leverage_window"]["full"]["ndx_price_index"]
        self.assertAlmostEqual(idx["total_return_pct"], (c[e] / c[a] - 1) * 100, places=1)

    def test_leverage_benchmarks_and_validation_read_the_adjusted_close(self):
        b = self.payload["benchmarks"]["leverage_window"]["test"]
        years = (pd.Timestamp("2026-09-30") - pd.Timestamp("2009-12-31")).days / 365.25
        for etf in ("QQQ", "SPY"):
            adj = self.frames[etf]["adj_close"]
            want = ((adj["2026-09-30"] / adj["2009-12-31"]) ** (1 / years) - 1) * 100
            self.assertAlmostEqual(b[etf]["cagr_pct"], want, delta=0.0051, msg=etf)
        val = self.payload["leverage"]["validation"]
        for etf in ("QQQ", "QLD", "TQQQ"):
            adj = self.frames[etf]["adj_close"]
            days = (pd.Timestamp(str(adj.index[-1])) - pd.Timestamp(str(adj.index[0]))).days
            want = ((adj.iloc[-1] / adj.iloc[0]) ** (365.25 / days) - 1) * 100
            self.assertAlmostEqual(val[etf]["real_cagr_pct"], want, delta=0.0051, msg=etf)
        # the winner block and the answer quote the real QQQ over the same TEST days
        w = self.payload["leverage"]["winner"]["test"]
        self.assertEqual(w["qqq_real_cagr_pct"], b["QQQ"]["cagr_pct"])
        self.assertEqual(w["qqq_real_cagr_exact_pct"], b["QQQ"]["cagr_exact_pct"])
        self.assertEqual(w["qqq_real_max_drawdown_pct"], b["QQQ"]["max_drawdown_pct"])
        self.assertIn(f"{w['qqq_real_cagr_exact_pct']:.1f}% for the real QQQ fund",
                      self.payload["answer"]["text"])
        self.assertNotIn("qqq_real_cagr_pct", self.payload["leverage"]["winner"]["train"])

    def test_share_of_windows_ahead_of_the_1x_fund_uses_the_1x_curve(self):
        eq = self.strict_json(sl.equity_output_path(self.out))["leverage"]["series"]
        mine, ref = np.asarray(eq["ndx2x_sma200"]), np.asarray(eq["ndx1x_none"])
        k = np.arange(len(mine) - 120)                      # every 120-month window
        share = 100.0 * float(np.mean(mine[k + 120] / mine[k] >= ref[k + 120] / ref[k]))
        got = self.payload["leverage"]["variants"]["ndx2x_sma200"]["rolling_10y"]
        self.assertEqual(got["n_windows"], len(k))
        self.assertTrue(0.0 < got["share_ge_1x_nasdaq_pct"] < 100.0)
        # the equity file carries 6 significant digits: allow about two windows of rounding
        self.assertAlmostEqual(got["share_ge_1x_nasdaq_pct"], share, delta=0.6)

    def test_validation_flags_the_fund_that_deviates(self):
        val = self.payload["leverage"]["validation"]
        self.assertTrue(val["QQQ"]["within_tolerance"])
        self.assertAlmostEqual(val["QQQ"]["cagr_diff_pp"], 0.0, places=2)
        self.assertEqual(val["QQQ"]["start"], "1999-03-10")
        self.assertTrue(val["TQQQ"]["within_tolerance"])
        self.assertFalse(val["QLD"]["within_tolerance"])
        self.assertGreater(val["QLD"]["cagr_diff_pp"], 1.0)
        # the whole-life deviation first, then the same fund's deviation by T-bill regime
        self.assertEqual(len(val["flags"]), 2)
        self.assertTrue(val["flags"][0].startswith("MODEL DEVIATION: "))
        self.assertTrue(val["flags"][1].startswith("MODEL DEVIATION BY T-BILL REGIME: "))
        self.assertIn("QLD", val["flags"][0])
        self.assertIn("QLD", val["flags"][1])
        self.assertEqual(self.payload["caveats"][:2], val["flags"])         # prominent: first
        self.assertFalse(val["usable_as_evidence"])
        self.assertFalse(self.payload["answer"]["usable_as_evidence"]["leverage"])
        self.assertIsNone(self.payload["answer"]["best_tested"])
        self.assertIn("no family is usable", self.payload["answer"]["best_tested_note"])
        self.assertIn("OUTSIDE", "\n".join(self.lines))
        self.assertFalse(self.payload["method"]["validation"]["QLD"]["within_tolerance"])

    def test_bootstrap_block(self):
        boot = self.payload["leverage"]["bootstrap"]
        self.assertEqual(set(boot["variants"]), set(self.payload["leverage"]["variants"]))
        for name, b in boot["variants"].items():
            self.assertEqual((b["paths"], b["horizon_months"], b["block_months"], b["seed"]),
                             (200, 120, 12, 20261001))
            self.assertEqual(b["months_in_history"], 480)
            q = b["cagr_pct"]
            self.assertTrue(q["p5"] <= q["p25"] <= q["p50"] <= q["p75"] <= q["p95"])
            self.assertGreaterEqual(b["p_cagr_ge_20_pct"], b["p_cagr_ge_40_pct"])
            self.assertGreaterEqual(b["p_max_drawdown_worse_than_50_pct"],
                                    b["p_max_drawdown_worse_than_75_pct"])
            self.assertEqual(b["full_history_cagr_pct"],
                             self.payload["leverage"]["variants"][name]["periods"]["full"]["cagr_pct"])
        self.assertIn("future may be worse", boot["warning"])
        for text in (boot["warning"], " ".join(self.payload["caveats"])):
            self.assertNotIn("equals the historical average", text)
            self.assertIn("the median 10-year outcome can still differ from the whole-history "
                          "CAGR", text)

    def test_leverage_periods_meet_at_the_split_bar(self):
        lev = self.payload["leverage"]
        eq = self.strict_json(sl.equity_output_path(self.out))["leverage"]
        k0, k1 = eq["dates"].index("2009-12-31"), len(eq["dates"]) - 1
        for name, v in lev["variants"].items():
            p = v["periods"]
            # TEST is anchored on the last bar on or before 2009-12-31, where TRAIN ends
            self.assertEqual((p["train"]["start"], p["train"]["end"]),
                             ("1986-09-30", "2009-12-31"))
            self.assertEqual((p["test"]["start"], p["test"]["end"]), ("2009-12-31", "2026-09-30"))
            ser = eq["series"][name]
            self.assertAlmostEqual(1 + p["test"]["total_return_pct"] / 100, ser[k1] / ser[k0],
                                   delta=1e-4 * ser[k1] / ser[k0] + 6e-5)
            for period in ("train", "test", "full"):
                est = p[period]["after_tax_estimate_de"]
                self.assertLessEqual(est["cagr_pct"], est["cagr_with_30pct_partial_exemption_pct"])
                if p[period]["cagr_pct"] > 0:
                    self.assertLess(est["cagr_pct"], p[period]["cagr_pct"])
        b = self.payload["benchmarks"]["leverage_window"]["test"]
        self.assertEqual(b["QQQ"]["start"], "2009-12-31")
        self.assertEqual(b["ndx_price_index"]["start"], "2009-12-31")

    def usable(self) -> tuple[dict, dict]:
        """The momentum and leverage blocks with the fund model passing its check."""
        mom = copy.deepcopy(self.payload["momentum"])
        lev = copy.deepcopy(self.payload["leverage"])
        lev["validation"]["usable_as_evidence"] = True
        return mom, lev

    def test_answer_and_caveats(self):
        a = self.payload["answer"]
        self.assertEqual(set(a["reaches_target"]), {"momentum", "leverage"})
        self.assertEqual(set(a["reaches_target_on_paper"]), {"momentum", "leverage"})
        for fam, key in (("momentum", "pooled"), ("leverage", "full")):
            self.assertIsInstance(a["reaches_target"][fam], bool)
            w = self.payload[fam]["winner"]["name"]
            on_paper = bool(w) and all(
                self.payload[fam]["variants"][w]["periods"][p]["cagr_exact_pct"] >= 40.0
                for p in ("test", key))
            self.assertEqual(a["reaches_target_on_paper"][fam], on_paper)
            self.assertEqual(a["reaches_target"][fam],
                             on_paper and a["usable_as_evidence"][fam])
        # momentum is never evidence; leverage only when the fund model passed its check
        self.assertFalse(a["usable_as_evidence"]["momentum"])
        self.assertFalse(a["reaches_target"]["momentum"])
        self.assertEqual(a["usable_as_evidence"]["leverage"],
                         self.payload["leverage"]["validation"]["usable_as_evidence"])
        self.assertIn("AND", a["reaches_target_basis"])
        self.assertIn("40%", a["text"])
        self.assertIn("not a forecast", a["text"])
        self.assertIn("Medallion", a["text"])
        self.assertEqual(a["context"]["label"], "not a lab result")
        self.assertIn(a["context"]["text"], a["text"])
        self.assertIn("evidence_quality", a)
        text = " ".join(self.payload["caveats"])
        for word in ("SURVIVORSHIP", "wiped out", "1987-10-19", "2020-03-16", "Taxes", "Germany",
                     "USD", "EUR", "Multiple testing", "pre-registered", "not a forecast",
                     "price index", "PRIIPs", "not measured by this lab",
                     "leverage.after_tax_model_de", "nearly all of its gains would be taxed",
                     "not a clean out-of-sample test", "Bootstrap sensitivity",
                     "must not be compared directly", "The momentum winner was chosen on",
                     "still held on the day after a sell signal"):
            self.assertIn(word, text)
        sm = self.payload["leverage"]["winner"]["selection_margin"]
        if sm["train_cagr_gap_to_runner_up_pp"] is not None:
            self.assertIn(f"chosen by {sm['train_cagr_gap_to_runner_up_pp']:.1f} points", text)
        notes = " ".join(self.payload["benchmarks"]["notes"])
        self.assertIn("excludes dividends", notes)
        method = json.dumps(self.payload["method"])
        self.assertIn("NOT independent", method)
        self.assertIn("exceptional eras", method)
        self.assertIn("must not be compared directly", method)
        calls = self.payload["method"]["judgement_calls"]
        self.assertIn("the delayed reading is implemented", calls[0])
        self.assertNotIn("the conservative reading is implemented", " ".join(calls))
        ts = self.payload["leverage"]["timing_sensitivity"]
        self.assertIn(f"a higher TRAIN CAGR in {ts['delay_higher_train_cagr']} of the 6", calls[0])
        self.assertEqual("not uniformly conservative" in calls[0],
                         bool(ts["delay_higher_train_cagr"]
                              or ts["delay_shallower_train_drawdown"]))

    def test_best_tested_comes_from_the_usable_family_only(self):
        mom, lev = self.usable()
        lv = lev["variants"]
        name = max(lv, key=lambda n: lv[n]["periods"]["full"]["cagr_exact_pct"])
        full, test = lv[name]["periods"]["full"], lv[name]["periods"]["test"]
        boot = lev["bootstrap"]["variants"][name]
        # distinct values in every field the answer copies, so a swapped source shows
        full["max_drawdown_pct"], test["max_drawdown_pct"] = -97.31, -12.34
        boot.update({"p_cagr_ge_40_pct": 7.77, "p_cagr_ge_20_pct": 33.33,
                     "p_max_drawdown_worse_than_50_pct": 55.55,
                     "p_max_drawdown_worse_than_75_pct": 22.22, "p_end_below_start_pct": 11.11})
        a = sl.build_answer(mom, lev, self.payload["benchmarks"]["momentum_window"])
        best = a["best_tested"]
        for key in ("name", "cagr_full", "cagr_test", "max_drawdown", "p_10y_ge_40",
                    "p_dd_worse_50", "p_dd_worse_75", "p_end_below_start", "history"):
            self.assertIn(key, best)
        self.assertEqual(best["family"], "leverage")
        self.assertFalse(best["biased_upward"])
        self.assertEqual(best["name"], name)
        self.assertEqual(best["cagr_full"], full["cagr_pct"])
        self.assertEqual(best["cagr_test"], test["cagr_pct"])
        self.assertEqual(best["max_drawdown"], full["max_drawdown_pct"])
        self.assertEqual(best["p_10y_ge_40"], boot["p_cagr_ge_40_pct"])
        self.assertEqual(best["p_dd_worse_50"], boot["p_max_drawdown_worse_than_50_pct"])
        self.assertEqual(best["p_dd_worse_75"], boot["p_max_drawdown_worse_than_75_pct"])
        self.assertEqual(best["p_end_below_start"], boot["p_end_below_start_pct"])
        self.assertEqual(best["history"], ["1986-09-30", "2026-09-30"])
        self.assertEqual(best["history_years"], 40.0)
        self.assertEqual((best["max_drawdown"], best["p_10y_ge_40"]), (-97.31, 7.77))
        self.assertNotEqual(full["cagr_pct"], test["cagr_pct"])
        # the momentum figure is shown apart, flagged, and never ranked against it
        biased = a["best_tested_biased"]
        self.assertEqual(biased, a["best_tested_by_family"]["momentum"])
        self.assertEqual(biased["family"], "momentum")
        self.assertTrue(biased["biased_upward"])
        self.assertIsNone(biased["p_10y_ge_40"])
        mv = mom["variants"]
        self.assertEqual(biased["name"],
                         max(mv, key=lambda n: mv[n]["periods"]["pooled"]["cagr_exact_pct"]))
        for v in mv.values():                       # momentum far ahead: still not 'best_tested'
            v["periods"]["pooled"]["cagr_pct"] = v["periods"]["pooled"]["cagr_exact_pct"] = 500.0
        again = sl.build_answer(mom, lev, self.payload["benchmarks"]["momentum_window"])
        self.assertEqual(again["best_tested"]["name"], name)
        self.assertIsNone(a["best_tested_note"])

    def forced(self, usable: bool = True, **cagr) -> str:
        """The answer text with every CAGR forced to 5% and the named winners to 50%."""
        mom, lev = self.usable()
        lev["validation"]["usable_as_evidence"] = usable
        for block in (mom, lev):
            for v in block["variants"].values():
                for m in v["periods"].values():
                    m["cagr_pct"] = m["cagr_exact_pct"] = 5.0
        for fam, block, key in (("momentum", mom, "pooled"), ("leverage", lev, "full")):
            name = sorted(block["variants"])[0]
            block["winner"]["name"] = name
            if fam in cagr:
                for p in cagr[fam]:
                    block["variants"][name]["periods"][p]["cagr_pct"] = 50.0
                    block["variants"][name]["periods"][p]["cagr_exact_pct"] = 50.0
        ans = sl.build_answer(mom, lev, self.payload["benchmarks"]["momentum_window"])
        self.last = ans
        return ans["text"]

    def verdict_line(self) -> str:
        payload = dict(self.payload)
        payload["answer"] = self.last
        return next(l for l in sl.summary_lines(payload) if l.startswith("ANSWER"))

    def test_answer_text_follows_the_numbers(self):
        self.assertTrue(self.forced().startswith(
            "No. None of the 17 rule sets made 40% a year, in any period tested."))
        self.assertEqual(self.last["reaches_target"], {"momentum": False, "leverage": False})
        self.assertEqual(self.verdict_line(), "ANSWER  reaches 40%/yr: momentum NO, leverage NO")
        # one period only is not 'reaching the target'
        text = self.forced(leverage=["test"])
        self.assertTrue(text.startswith("No. None of the 17 rule sets made 40% a year over its "
                                        "whole tested history; 1 ("))
        self.assertFalse(self.last["reaches_target"]["leverage"])
        self.assertFalse(self.last["reaches_target_on_paper"]["leverage"])
        text = self.forced(leverage=["test", "full"])
        self.assertTrue(text.startswith("On paper yes: the pre-set leveraged"))
        self.assertEqual(self.last["reaches_target"], {"momentum": False, "leverage": True})
        self.assertEqual(self.verdict_line(), "ANSWER  reaches 40%/yr: momentum NO, leverage YES")
        # ... unless the fund model failed its check against the real funds
        text = self.forced(usable=False, leverage=["test", "full"])
        self.assertTrue(text.startswith("Only on paper, and not in a way you can rely on. The "
                                        "pre-set leveraged"))
        self.assertEqual(self.last["reaches_target"], {"momentum": False, "leverage": False})
        self.assertEqual(self.last["reaches_target_on_paper"],
                         {"momentum": False, "leverage": True})
        self.assertIsNone(self.last["best_tested"])
        self.assertEqual(self.verdict_line(),
                         "ANSWER  reaches 40%/yr: momentum NO, leverage NO (on paper yes - the "
                         "fund model missed the real funds, not usable as evidence)")
        # the stock list makes the momentum figure unusable as evidence, whatever it shows
        text = self.forced(momentum=["test", "pooled"])
        self.assertTrue(text.startswith("Only on paper, and not in a way you can rely on."))
        self.assertEqual(self.last["reaches_target"], {"momentum": False, "leverage": False})
        self.assertEqual(self.last["reaches_target_on_paper"],
                         {"momentum": True, "leverage": False})
        self.assertEqual(self.last["usable_as_evidence"], {"momentum": False, "leverage": True})
        self.assertEqual(self.last["best_tested"]["family"], "leverage")
        self.assertEqual(self.last["best_tested_biased"]["family"], "momentum")
        self.assertTrue(self.last["best_tested_biased"]["biased_upward"])
        self.assertEqual(self.verdict_line(),
                         "ANSWER  reaches 40%/yr: momentum NO (on paper yes - "
                         "survivorship-inflated, not usable as evidence), leverage NO")
        payload = dict(self.payload)
        payload["answer"] = self.last
        lines = "\n".join(sl.summary_lines(payload))
        for word in ("best tested (hindsight, usable family only)", "P(10y >= 40%)",
                     "P(DD > 50%)", "P(DD > 75%)", "P(loss)", "inflated, NOT evidence"):
            self.assertIn(word, lines)
        text = self.forced(momentum=["test", "pooled"], leverage=["test", "full"])
        self.assertTrue(text.startswith("On paper both pre-set rules reached 40% a year"))
        self.assertEqual(self.last["reaches_target"], {"momentum": False, "leverage": True})
        text = self.forced(momentum=["pooled"])
        self.assertTrue(text.startswith("Not in a way you can rely on. 1 of the 17 rule sets"))
        # a winner 0.004 points short of the target has not reached it
        mom, lev = self.usable()
        name = lev["winner"]["name"]
        for p in ("test", "full"):
            lev["variants"][name]["periods"][p]["cagr_pct"] = 40.0
            lev["variants"][name]["periods"][p]["cagr_exact_pct"] = 39.996
        ans = sl.build_answer(mom, lev, self.payload["benchmarks"]["momentum_window"])
        self.assertFalse(ans["reaches_target_on_paper"]["leverage"])
        self.assertNotIn(name, ans["reaches_target_detail"]["leverage"][
            "variants_ge_target_whole_history"])
        for p in ("test", "full"):
            lev["variants"][name]["periods"][p]["cagr_exact_pct"] = 40.0
        ans = sl.build_answer(mom, lev, self.payload["benchmarks"]["momentum_window"])
        self.assertTrue(ans["reaches_target"]["leverage"])

    def test_answer_text_quotes_the_computed_figures(self):
        mom, lev = self.usable()
        a = sl.build_answer(mom, lev, self.payload["benchmarks"]["momentum_window"])
        w = lev["winner"]["name"]
        self.assertIsNotNone(w)
        per, base = lev["variants"][w]["periods"], lev["variants"]["ndx1x_none"]["periods"]
        tax_w, tax_b = (x["full"]["after_tax_estimate_de"]["cagr_pct"] for x in (per, base))
        self.assertIn(f"the chosen rule made about {tax_w:.1f}% a year over the whole history "
                      f"against about {tax_b:.1f}% for holding the Nasdaq-100 and paying tax once "
                      "at the end", a["text"])
        tw, tb_ = (x["test"]["after_tax_estimate_de"]["cagr_pct"] for x in (per, base))
        tail = ("the pre-tax edge disappears." if tax_w <= tax_b and tw <= tb_ else
                "an edge remains after tax." if tax_w > tax_b and tw > tb_ else
                "the pre-tax edge largely disappears.")
        self.assertIn(tail, a["text"])
        roll = lev["variants"][w]["rolling_10y"]
        self.assertIn(f"in {roll['share_ge_target_pct']:.1f}% of the overlapping 10-year windows "
                      "of that history (not a probability: the history is about four "
                      "independent decades", a["text"])
        if roll["windows_ge_target"]:
            self.assertIn("every such window began in "
                          + sl._years_text(roll["windows_ge_target_start_years"]), a["text"])
        r1 = lev["variants"]["ndx1x_none"]["rolling_10y"]
        if r1["windows_ge_target"] == 0:
            self.assertIn("The plain Nasdaq-100 never made 40% a year over any 10 years",
                          a["text"])
            self.assertIn(f"made about {r1['max_cagr_pct']:.0f}% a year", a["text"])
        bh = self.payload["benchmarks"]["momentum_window"]
        self.assertIn("that list made "
                      f"{bh['pooled']['equal_weight_buy_and_hold']['cagr_exact_pct']:.1f}% a year",
                      a["text"])
        fresh = bh["test"]["equal_weight_buy_and_hold_from_test_start"]["cagr_exact_pct"]
        self.assertIn(f"and {fresh:.1f}% for buying all the same stocks in equal amounts at the "
                      "start of that period", a["text"])
        self.assertNotIn("Held blindly", a["text"])

    def patched(self) -> tuple[dict, dict, dict, dict, dict]:
        """usable() blocks plus the momentum benchmarks, with distinct values in every field
        the answer text could confuse with a neighbour; in TEST the chosen leveraged rule
        falls further than the 1x fund, over the whole history it falls less."""
        mom, lev = self.usable()
        bench_a = copy.deepcopy(self.payload["benchmarks"]["momentum_window"])
        w = lev["winner"]["name"]
        per, base = lev["variants"][w]["periods"], lev["variants"]["ndx1x_none"]["periods"]
        lev["bootstrap"]["variants"][w].update({
            "p_cagr_ge_40_pct": 7.77, "p_cagr_ge_20_pct": 33.33,
            "p_max_drawdown_worse_than_50_pct": 55.57, "p_max_drawdown_worse_than_75_pct": 22.22})
        lev["variants"][w]["rolling_10y"].update({"min_cagr_pct": -12.34, "median_cagr_pct": 8.76})
        bench_a["test"]["QQQ"].update({"cagr_pct": 11.12, "cagr_exact_pct": 11.1234})
        bench_a["test"]["SPY"].update({"cagr_pct": 9.98, "cagr_exact_pct": 9.9812})
        per["test"].update({"cagr_pct": 20.33, "cagr_exact_pct": 20.334,
                            "max_drawdown_pct": -51.23, "max_drawdown_exact_pct": -51.234})
        base["test"].update({"cagr_pct": 18.75, "cagr_exact_pct": 18.746,     # 18.7, not 18.8
                             "max_drawdown_pct": -35.22, "max_drawdown_exact_pct": -35.224})
        per["full"].update({"max_drawdown_pct": -70.12, "max_drawdown_exact_pct": -70.123})
        base["full"].update({"max_drawdown_pct": -83.41, "max_drawdown_exact_pct": -83.412})
        lev["winner"]["test"].update({"qqq_real_cagr_pct": 19.1, "qqq_real_cagr_exact_pct": 19.0987})
        return mom, lev, bench_a, per, base

    def test_answer_text_quotes_the_figures_it_names(self):
        mom, lev, bench_a, per, base = self.patched()
        text = sl.build_answer(mom, lev, bench_a)["text"]
        # bootstrap, rolling windows and the momentum TEST benchmark: the named field each time
        self.assertIn("and in 7.8% of 200 re-shuffled 10-year histories", text)
        self.assertIn("; in 55.6% of those it lost more than half its value", text)
        self.assertIn("; the worst window made -12.3% a year)", text)
        self.assertIn("against 11.1% for the Nasdaq-100 fund QQQ", text)
        # the TEST sentence carries the TEST drawdowns next to the TEST returns, from the
        # unrounded figures (18.746 -> 18.7) and with the real QQQ beside the 1x model
        i = text.index("years it was not chosen on")
        test_part = text[i:text.index("Over the whole", i)]
        self.assertIn("it made 20.3% a year against 18.7% for the modelled 1x Nasdaq-100 fund and "
                      "19.1% for the real QQQ fund, but its worst fall in those years was -51% "
                      "against -35% for the 1x fund.", test_part)
        self.assertNotIn("18.8% for the modelled", text)
        self.assertNotIn("-70%", test_part)
        span = sl._span(per["train"])
        self.assertIn(f"and its worst fall was -70% (1x fund: -83%) - an advantage earned in "
                      f"{span}, the years the rule was chosen on.", text)
        # the reverse TEST ordering: no contrast, and the whole-history edge is not 'earned in TRAIN'
        per["test"].update({"max_drawdown_pct": -20.0, "max_drawdown_exact_pct": -20.0})
        text = sl.build_answer(mom, lev, bench_a)["text"]
        self.assertIn("for the real QQQ fund, and its worst fall in those years was -20% against "
                      "-35% for the 1x fund.", text)
        self.assertNotIn("an advantage earned in", text)
        # without the real QQQ in the winner block the model stands alone
        del lev["winner"]["test"]["qqq_real_cagr_pct"], lev["winner"]["test"]["qqq_real_cagr_exact_pct"]
        text = sl.build_answer(mom, lev, bench_a)["text"]
        self.assertIn("against 18.7% for the modelled 1x Nasdaq-100 fund, and its worst fall", text)
        self.assertNotIn("real QQQ fund", text)

    def test_a_stale_series_cache_shortens_strategy_a_and_says_so(self):
        cut = {t: f[f.index <= "2026-06-30"] for t, f in self.frames.items()}
        series = self.root / "stale" / "lab_series.sqlite3"
        sl.save_series(series, cut, {t: "2026-07-01T00:00:00" for t in cut})
        lines: list[str] = []
        payload = self.run_lab(self.root / "out5" / "lab.json", lines, series_path=series)
        win = payload["method"]["momentum"]["window"]
        self.assertEqual(win["last_day"], "2026-06-30")
        self.assertTrue(win["shortened_by_series_cache"])
        self.assertTrue(any(l.startswith("WARN [A]") and "--refresh" in l for l in lines))
        self.assertEqual(payload["method"]["leverage"]["window"]["last_bar"], "2026-06-30")
        self.assertFalse(self.payload["method"]["momentum"]["window"]["shortened_by_series_cache"])

    def test_equity_file(self):
        e = self.strict_json(sl.equity_output_path(self.out))
        m, l = e["momentum"], e["leverage"]
        self.assertEqual(m["dates"][0], "2017-07-31")
        self.assertEqual(m["dates"][-1], "2026-09-30")
        self.assertEqual(set(m["series"]), set(self.payload["momentum"]["variants"])
                         | {"QQQ", "SPY", "equal_weight_buy_and_hold",
                            "equal_weight_universe_daily_rebalanced",
                            "equal_weight_eligible_monthly"})
        for name, ser in m["series"].items():
            self.assertEqual(len(ser), len(m["dates"]), name)
            self.assertEqual(ser[0], 1.0, name)
        # each variant's own curve: it ends where its pooled return says, and no two agree
        for name, v in self.payload["momentum"]["variants"].items():
            want = 1 + v["periods"]["pooled"]["total_return_pct"] / 100
            self.assertAlmostEqual(m["series"][name][-1], want, delta=want * 1e-4, msg=name)
        curves = {tuple(m["series"][name]) for name in self.payload["momentum"]["variants"]}
        self.assertEqual(len(curves), len(self.payload["momentum"]["variants"]))
        self.assertEqual(l["dates"][0], "1986-09-30")
        self.assertEqual(len(l["dates"]), 481)
        self.assertEqual(set(l["series"]), set(self.payload["leverage"]["variants"])
                         | {"ndx_price_index"})
        for name, ser in l["series"].items():
            self.assertEqual(len(ser), 481, name)
        full = self.payload["leverage"]["variants"]["ndx2x_sma200"]["periods"]["full"]
        self.assertAlmostEqual(l["series"]["ndx2x_sma200"][-1],
                               1 + full["total_return_pct"] / 100, delta=abs(
                                   1 + full["total_return_pct"] / 100) * 1e-4)

    def test_summary_is_printed(self):
        text = "\n".join(self.lines)
        for word in ("STRATEGY LAB", "A. MOMENTUM ROTATION", "B. LEVERAGED NASDAQ-100",
                     "pre-registered winner", "ANSWER", "CAVEATS", "survivorship",
                     "equal weight, buy and hold", "not investable", "by T-bill regime",
                     "P(DD>75)", "afterTax", "switching at the signal's own close",
                     "dearer financing", "placebo"):
            self.assertIn(word, text)
        self.assertNotIn("True", next(l for l in self.lines if l.startswith("ANSWER")))

    def test_rerun_is_identical(self):
        out2 = self.root / "out2" / "lab.json"
        again = self.run_lab(out2)
        a, b = bt._clean_json(self.payload), bt._clean_json(again)
        for d in (a, b):
            d.pop("generated")
            d.pop("runtime_seconds")
            d.pop("equity_file")
        self.assertEqual(a, b)
        self.assertTrue(sl.equity_output_path(out2).exists())
        self.assertEqual(sl.equity_output_path(out2).name, "lab_equity.json")

    def test_offline_without_the_series_cache_fails_clearly(self):
        out = self.root / "out3" / "lab.json"
        with self.assertRaises(FileNotFoundError) as cm:
            self.run_lab(out, series_path=self.data / "cache" / "nope.sqlite3")
        self.assertIn("--offline", str(cm.exception))
        self.assertFalse(out.exists())

    def test_default_run_downloads_into_the_series_cache_only(self):
        out = self.root / "out4" / "lab.json"
        series = self.root / "fresh" / "lab_series.sqlite3"
        calls = []

        def fetch(tickers):
            calls.append(list(tickers))
            return {t: self.frames[t] for t in tickers}

        payload = self.run_lab(out, offline=False, series_path=series, fetch=fetch)
        self.assertEqual(calls, [list(sl.SERIES_TICKERS)])
        self.assertTrue(series.exists())
        self.assertEqual(payload["leverage"]["winner"]["name"],
                         self.payload["leverage"]["winner"]["name"])
        self.assertEqual(hashlib.sha256(self.prices.read_bytes()).hexdigest(), self.before)

    def test_refuses_to_overwrite_an_input(self):
        with self.assertRaises(ValueError):
            self.run_lab(self.series)
        with self.assertRaises(ValueError):
            self.run_lab(self.out, series_path=self.prices)
        with self.assertRaises(FileNotFoundError):
            self.run_lab(self.out, prices_path=self.data / "cache" / "none.sqlite3")


if __name__ == "__main__":
    unittest.main()
