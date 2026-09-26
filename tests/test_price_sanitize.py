"""Tests for atlas2.backtest.sanitize_bars / sanitize_long (backtest-only price repair).

Synthetic data only; never touches data/ or config.json.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from atlas2 import backtest as bt


def frame(closes, opens=None, volume=1e6, start="2020-01-01", highs=None, lows=None,
          volumes=None) -> pd.DataFrame:
    """Daily bars: open = previous close unless given; high/low 1% around the body."""
    c = np.asarray(closes, float)
    o = np.asarray(opens, float) if opens is not None else np.r_[c[0], c[:-1]]
    h = np.asarray(highs, float) if highs is not None else np.maximum(o, c) * 1.01
    l = np.asarray(lows, float) if lows is not None else np.minimum(o, c) * 0.99
    v = np.asarray(volumes, float) if volumes is not None else np.full(len(c), float(volume))
    idx = pd.bdate_range(start, periods=len(c), name="date")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


def dstr(df: pd.DataFrame, k: int) -> str:
    return df.index[k].strftime("%Y-%m-%d")


class ExactRatioTest(unittest.TestCase):
    def test_quarter_bonus_is_back_adjusted(self):
        # 30 bars around 400, then an unadjusted 1:4 bonus: open exactly prev_close / 4
        closes = [400.0 + (k % 3) for k in range(30)] + [101.0, 102.0, 101.5]
        opens = list(np.r_[closes[0], closes[:-1]])
        opens[30] = closes[29] * 0.25
        vols = [1e6] * 30 + [3e6, 1e6, 1e6]            # the share count changed: volume surge
        df = frame(closes, opens=opens, volumes=vols)
        out, rep = bt.sanitize_bars(df, "BONUS.NS")
        self.assertEqual(rep["counts"], {"adjust_split_ratio": 1})
        self.assertEqual(rep["events"], [["BONUS.NS", dstr(df, 30), "adjust_split_ratio", 0.25]])
        self.assertEqual(len(out), len(df))
        np.testing.assert_allclose(out["close"].to_numpy()[:30], df["close"].to_numpy()[:30] * 0.25)
        np.testing.assert_allclose(out["open"].to_numpy()[:30], df["open"].to_numpy()[:30] * 0.25)
        np.testing.assert_allclose(out["volume"].to_numpy()[:30], df["volume"].to_numpy()[:30] * 4)
        np.testing.assert_allclose(out.iloc[30:].to_numpy(), df.iloc[30:].to_numpy())
        # the input frame is not modified
        self.assertEqual(float(df["close"].iloc[0]), 400.0)

    def test_two_thirds_override_is_honoured(self):
        closes = [300.0] * 20 + [200.5, 201.0, 199.0]
        opens = list(np.r_[closes[0], closes[:-1]])
        opens[20] = 200.0                               # exactly 2/3, no volume surge
        df = frame(closes, opens=opens)
        # the rules deliberately cannot see it (no volume surge) ...
        out, rep = bt.sanitize_bars(df, "TRENT.NS")
        self.assertEqual(rep["counts"], {})
        pd.testing.assert_frame_equal(out, df)
        # ... so a configured override adjusts it
        ov = [{"ticker": "TRENT.NS", "date": dstr(df, 20), "action": "adjust", "ratio": 2 / 3},
              {"ticker": "OTHER.NS", "date": dstr(df, 5), "action": "truncate"}]
        out, rep = bt.sanitize_bars(df, "TRENT.NS", ov)
        self.assertEqual(rep["counts"], {"override_adjust": 1})
        np.testing.assert_allclose(out["close"].to_numpy()[:20], np.full(20, 200.0))
        np.testing.assert_allclose(out["volume"].to_numpy()[:20], np.full(20, 1.5e6))
        np.testing.assert_allclose(out["close"].to_numpy()[20:], df["close"].to_numpy()[20:])

    def test_minus_25pct_gap_near_three_quarters_is_untouched(self):
        closes = [100.0] * 20 + [75.5, 76.0, 75.0]
        opens = list(np.r_[closes[0], closes[:-1]])
        opens[20] = 75.0014                             # 0.750014 x prev close: a real gap
        vols = [1e6] * 20 + [5e6, 1e6, 1e6]
        df = frame(closes, opens=opens, volumes=vols)
        out, rep = bt.sanitize_bars(df, "GAP")
        self.assertEqual(rep["counts"], {})
        self.assertEqual(rep["events"], [])
        pd.testing.assert_frame_equal(out, df)

    def test_exact_half_at_penny_price_is_not_adjusted(self):
        closes = [0.097] * 20 + [0.049, 0.05, 0.051]
        opens = list(np.r_[closes[0], closes[:-1]])
        opens[20] = 0.0485                              # exactly 0.5 x 0.097: a tick artefact
        vols = [1e6] * 20 + [9e6, 1e6, 1e6]
        df = frame(closes, opens=opens, volumes=vols)
        out, rep = bt.sanitize_bars(df, "CHRD")
        self.assertNotIn("adjust_split_ratio", rep["counts"])
        pd.testing.assert_frame_equal(out, df)


class TruncationTest(unittest.TestCase):
    def test_listing_join_is_truncated(self):
        # old shares at 7 cents joined to the new equity at 16.50 (x = 226)
        closes = [0.08, 0.075, 0.073] * 10 + [16.8, 17.0, 16.9, 17.2]
        opens = list(np.r_[closes[0], closes[:-1]])
        opens[30] = 16.5
        df = frame(closes, opens=opens)
        out, rep = bt.sanitize_bars(df, "CHRD")
        self.assertEqual(rep["counts"], {"truncate_join": 1})
        self.assertEqual(len(out), 4)
        self.assertEqual(out.index[0], df.index[30])
        pd.testing.assert_frame_equal(out, df.iloc[30:])

    def test_flat_zero_volume_run_then_jump_is_truncated(self):
        closes = [10.0] * 3 + [12.0] * 6 + [42.0, 43.0, 42.5]
        opens = list(np.r_[closes[0], closes[:-1]])
        vols = [1e6] * 3 + [0.0] * 6 + [2e6, 1e6, 1e6]
        highs = list(np.maximum(opens, closes) * 1.01)
        lows = list(np.minimum(opens, closes) * 0.99)
        for k in range(3, 9):                           # placeholder bars: o = c = h = l
            opens[k] = highs[k] = lows[k] = closes[k] = 12.0
        df = frame(closes, opens=opens, volumes=vols, highs=highs, lows=lows)
        out, rep = bt.sanitize_bars(df, "ACT.DE")
        self.assertEqual(rep["counts"], {"truncate_flat_run": 1})
        self.assertEqual(out.index[0], df.index[9])
        self.assertEqual(len(out), 3)

    def test_stale_head_truncated_only_before_the_refetch_date(self):
        closes = [13.46] * 5 + [6.72, 6.8, 6.75] + [6.8] * 10
        df = frame(closes)
        # without a stale head date the -50% day is left alone (below the join ratio)
        out, rep = bt.sanitize_bars(df, "APH")
        self.assertEqual(rep["counts"], {})
        # the latest download re-fetched from bar 6 on: the older head is dropped
        out, rep = bt.sanitize_bars(df, "APH", stale_before=dstr(df, 6))
        self.assertEqual(rep["counts"], {"truncate_stale_head": 1})
        self.assertEqual(out.index[0], df.index[5])
        # a big move AFTER the refetch date is not a stale-head artefact
        out, rep = bt.sanitize_bars(df, "APH", stale_before=dstr(df, 2))
        self.assertEqual(rep["counts"], {})


class SpikeTest(unittest.TestCase):
    def test_spike_and_revert_drops_one_bar_and_keeps_history(self):
        closes = [100.0 + k * 0.1 for k in range(20)]
        closes[12] = closes[11] * 2.8                   # +180% ...
        closes[13] = closes[11] * 0.98                  # ... then -65% back to where it was
        df = frame(closes)
        out, rep = bt.sanitize_bars(df, "AVB")
        self.assertEqual(rep["counts"], {"drop_spike": 1})
        self.assertEqual(rep["events"], [["AVB", dstr(df, 12), "drop_spike"]])
        self.assertEqual(len(out), len(df) - 1)
        self.assertNotIn(df.index[12], out.index)
        pd.testing.assert_frame_equal(out, df.drop(df.index[12]))

    def test_big_real_moves_are_untouched(self):
        # +177% that holds (no revert), later a -60% crash that holds
        closes = [20.0] * 10 + [55.4, 56.0, 57.0, 58.0] + [23.0, 22.5, 23.5, 24.0]
        df = frame(closes)
        out, rep = bt.sanitize_bars(df, "MRNA")
        self.assertEqual(rep["counts"], {})
        pd.testing.assert_frame_equal(out, df)

    def test_last_bar_is_never_a_spike(self):
        closes = [50.0] * 10 + [150.0]
        df = frame(closes)
        out, rep = bt.sanitize_bars(df, "NEW")
        self.assertNotIn("drop_spike", rep["counts"])
        self.assertEqual(len(out), len(df))

    def test_unresolved_large_move_is_reported(self):
        closes = [10.0] * 10 + [60.0] * 5              # 6x, not a join (< 10x), no revert
        df = frame(closes)
        out, rep = bt.sanitize_bars(df, "ODD")
        self.assertEqual(rep["counts"], {"unresolved": 1})
        self.assertEqual(rep["suspect"], [dstr(df, 10)])
        pd.testing.assert_frame_equal(out, df)


class PanelEquivalenceTest(unittest.TestCase):
    """The baselines on a panel with bad bars (after sanitize_long) equal the clean panel."""

    def setUp(self):
        rng = np.random.default_rng(3)
        n = 120
        self.clean = {}
        for k in range(6):
            c = 50 * np.exp(np.cumsum(0.0005 + 0.015 * rng.standard_normal(n)))
            self.clean[f"T{k}"] = frame(c)
        bad = {t: df.copy() for t, df in self.clean.items()}
        # T1: one-day spike that reverts -> the clean panel lacks that bar
        spike = bad["T1"].copy()
        spike.iloc[60, spike.columns.get_loc("close")] *= 2.8
        spike.iloc[60, spike.columns.get_loc("high")] = spike["close"].iloc[60] * 1.01
        bad["T1"] = spike
        self.clean["T1"] = self.clean["T1"].drop(self.clean["T1"].index[60])
        # T2: garbage head joined to the real listing -> the clean panel starts at the join
        head = bad["T2"].copy()
        head.iloc[:40, :4] = head.iloc[:40, :4] / 300.0
        bad["T2"] = head
        self.clean["T2"] = self.clean["T2"].iloc[40:]
        # T3: unadjusted 1:4 bonus -> the clean panel is back-adjusted
        bonus = bad["T3"].copy()
        k = 80
        bonus.iloc[k:, :4] = bonus.iloc[k:, :4] * 0.25
        bonus.iloc[k, bonus.columns.get_loc("open")] = bonus["close"].iloc[k - 1] * 0.25
        bonus.iloc[k, bonus.columns.get_loc("volume")] = 5e6
        bad["T3"] = bonus
        cl = self.clean["T3"].copy()
        cl.iloc[:, :4] = cl.iloc[:, :4] * 0.25
        cl.iloc[k, cl.columns.get_loc("open")] = bonus["open"].iloc[k]
        cl.iloc[:k, cl.columns.get_loc("volume")] = 4e6
        cl.iloc[k, cl.columns.get_loc("volume")] = 5e6
        cl.iloc[k, cl.columns.get_loc("high")] = bonus["high"].iloc[k]
        cl.iloc[k, cl.columns.get_loc("low")] = bonus["low"].iloc[k]
        self.clean["T3"] = cl
        self.bad = bad

    def test_baselines_match_clean_panel(self):
        long_bad, rep = bt.sanitize_long(bt.frames_to_long(self.bad))
        self.assertEqual(rep["counts"].get("drop_spike"), 1)
        self.assertEqual(rep["counts"].get("truncate_join"), 1)
        self.assertEqual(rep["counts"].get("adjust_split_ratio"), 1)
        fixed = bt.MarketPanel(long_bad)
        clean = bt.MarketPanel(bt.frames_to_long(self.clean))
        np.testing.assert_allclose(fixed.C, clean.C, rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(fixed.O, clean.O, rtol=1e-12, equal_nan=True)
        dates = list(clean.dates)
        for a, b in ((10, 50), (45, 70), (55, 65), (75, 90), (20, 110)):
            got, want = fixed.universe_return(dates[a], dates[b]), clean.universe_return(dates[a], dates[b])
            self.assertAlmostEqual(got, want, places=10)
        trade = {"ticker": "T0", "entry_date": dates[50], "pattern": "bull_flag",
                 "entry_price": 50.0, "stop": 47.5, "target": 56.0}
        self.assertEqual(fixed.random_baseline(trade, 5, 20, 0.15, 0.05),
                         clean.random_baseline(trade, 5, 20, 0.15, 0.05))

    def test_unresolved_prints_are_masked_out_of_the_universe(self):
        frames = {f"T{k}": frame(10.0 * 1.01 ** np.arange(40) * (1 + k)) for k in range(3)}
        odd = frame(np.r_[np.full(20, 10.0), np.full(20, 60.0)])   # 6x jump, no rule explains it
        frames["ODD"] = odd
        long, rep = bt.sanitize_long(bt.frames_to_long(frames))
        self.assertIn("ODD", rep["suspect"])
        masked = bt.MarketPanel(long, suspect=rep["suspect"])
        plain = bt.MarketPanel(bt.frames_to_long({k: v for k, v in frames.items() if k != "ODD"}))
        d = list(masked.dates)
        # a window over the bad print leaves the stock out of the equal-weight basket
        self.assertAlmostEqual(masked.universe_return(d[15], d[25]),
                               plain.universe_return(d[15], d[25]), places=12)
        # a window that does not touch it still counts the stock (flat: 0%)
        self.assertAlmostEqual(masked.universe_return(d[25], d[35]),
                               plain.universe_return(d[25], d[35]) * 3 / 4, places=12)


if __name__ == "__main__":
    unittest.main()
