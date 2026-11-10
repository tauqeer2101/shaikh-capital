"""Tests for atlas2.strategy_lab3: indicators, the swing book and the runner.

Synthetic panels only; nothing reads data/, the network or config.json.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from atlas2 import strategy_lab3 as L3


def wilder_rsi(c: list[float], n: int = 2) -> list[float]:
    up = dn = None
    out = [float("nan")]
    for a, b in zip(c, c[1:]):
        u, d = max(b - a, 0.0), max(a - b, 0.0)
        up = u if up is None else up + (u - up) / n
        dn = d if dn is None else dn + (d - dn) / n
        out.append(100.0 if dn == 0 and up > 0 else 50.0 if dn == 0 else 100 - 100 / (1 + up / dn))
    return out


def book(O, C, entry, *, key=None, exit_sig=None, slots=1, cost=0.0, rf=None, stop=None,
         max_hold=None, i0=0, i1=None):
    O, C = np.asarray(O, float), np.asarray(C, float)
    if O.ndim == 1:
        O, C = O[:, None], C[:, None]
    entry = np.asarray(entry, bool).reshape(C.shape)
    n = len(C)
    return L3.run_book(O, C, entry, np.zeros(C.shape) if key is None else np.asarray(key, float),
                       np.zeros(C.shape, bool) if exit_sig is None
                       else np.asarray(exit_sig, bool).reshape(C.shape),
                       slots=slots, cost=cost, rf=np.zeros(n) if rf is None else rf, i0=i0,
                       i1=n - 1 if i1 is None else i1, stop=stop, max_hold=max_hold)


class IndicatorTest(unittest.TestCase):
    def test_rsi_matches_wilder(self):
        c = [10, 11, 10.5, 10.2, 11.4, 11.4, 12, 11, 10, 10.6]
        got = L3.rsi(np.array(c)[:, None], 2)[:, 0]
        want = wilder_rsi(c, 2)
        self.assertTrue(np.isnan(got[0]))
        np.testing.assert_allclose(got[1:], want[1:], rtol=1e-12)
        self.assertEqual(L3.rsi(np.array([5.0, 6, 7])[:, None])[2, 0], 100.0)   # no falls
        self.assertEqual(L3.rsi(np.array([5.0, 5, 5])[:, None])[2, 0], 50.0)    # no moves

    def test_eligibility_needs_membership_history_and_no_recent_jump(self):
        n = 800
        dates = pd.bdate_range("2000-01-03", periods=n).strftime("%Y-%m-%d").to_numpy()
        C = np.full((n, 2), 10.0)
        C[:100, 1] = np.nan                                  # B lists 100 bars later
        member = np.ones((n, 2), bool)
        member[:300, 0] = False                              # A joins the index at bar 300
        e = L3.eligibility(dates, ["A", "B"], C, member, {"A": [dates[400]]})
        self.assertFalse(e[299, 0])
        self.assertTrue(e[300, 0])
        self.assertFalse(e[400, 0])                          # the jump day ...
        self.assertFalse(e[400 + L3.JUMP_LOOKBACK - 1, 0])   # ... and the 251 bars after
        self.assertTrue(e[400 + L3.JUMP_LOOKBACK, 0])
        self.assertFalse(e[100 + L3.MIN_BARS - 2, 1])        # its own 253 bars first
        self.assertTrue(e[100 + L3.MIN_BARS - 1, 1])

    def test_signals_use_nothing_after_the_day(self):
        rng = np.random.default_rng(5)
        C = 50 * np.cumprod(1 + rng.normal(0.0005, 0.02, (400, 3)), axis=0)
        V = rng.uniform(1e5, 2e5, (400, 3))
        elig = np.ones(C.shape, bool)
        a = L3.build_signals(C, V, elig)
        C2, V2 = C.copy(), V.copy()
        C2[300:] *= 3.0                                      # change the future
        V2[300:] *= 9.0
        b = L3.build_signals(C2, V2, elig)
        for f in ("trend", "above5", "below20", "below50", "rsi2", "r126", "top", "newhigh",
                  "breakout", "rng20", "lo20p", "volx"):
            np.testing.assert_array_equal(getattr(a, f)[:300], getattr(b, f)[:300], err_msg=f)


class BookTest(unittest.TestCase):
    def test_fills_at_the_next_open_and_exits_at_the_open_after_the_exit_signal(self):
        O = [10, 10, 10, 11, 12, 13, 14]
        C = [10, 10, 10.5, 11.5, 12.5, 13.5, 14]
        entry = [0, 1, 0, 0, 0, 0, 0]                        # signal at the close of bar 1
        ex = [0, 0, 0, 1, 0, 0, 0]                           # exit signal at the close of bar 3
        r = book(O, C, entry, exit_sig=ex)
        self.assertEqual(r["trades"], [(2, 4, 0, 12 / 10 - 1)])
        np.testing.assert_allclose(r["equity"], [1, 1, 1.05, 1.15, 1.2, 1.2, 1.2])
        c = 0.002
        r = book(O, C, entry, exit_sig=ex, cost=c)
        self.assertAlmostEqual(r["trades"][0][3], 12 * (1 - c) / (10 * (1 + c)) - 1)
        self.assertAlmostEqual(r["equity"][-1], 12 * (1 - c) / (10 * (1 + c)))

    def test_cash_earns_tbills(self):
        rf = np.array([0, 0.001, 0.001, 0.001])
        r = book([10] * 4, [10] * 4, [0] * 4, rf=rf)
        self.assertAlmostEqual(r["equity"][-1], 1.001 ** 3)

    def test_slots_rank_and_one_position_per_stock(self):
        n = 5
        O = np.full((n, 3), 10.0)
        C = np.full((n, 3), 10.0)
        C[2:, 0] = 12.0
        entry = np.zeros((n, 3), bool)
        entry[0] = True                                      # all three signal at bar 0
        entry[1, 0] = True                                   # A again while held: ignored
        key = np.array([[3.0, 1.0, 2.0]] * n)                # B best, then C, then A
        r = book(O, C, entry, key=key, slots=2)
        self.assertEqual(r["trades"], [])                    # nothing exits
        self.assertAlmostEqual(r["equity"][-1], 1.0)         # B and C never move
        self.assertAlmostEqual(r["invested"][1], 1.0)
        r = book(O, C, entry, key=np.array([[1.0, 2.0, 3.0]] * n), slots=2)
        self.assertAlmostEqual(r["equity"][-1], 0.5 * 1.2 + 0.5)   # A and B, half each

    def test_max_hold_stop_and_delisting(self):
        O = [10] * 8
        C = [10] * 8
        r = book(O, C, [1, 0, 0, 0, 0, 0, 0, 0], max_hold=3)
        self.assertEqual(r["trades"][0][:2], (1, 5))         # held 3 sessions, out at the next open
        C2 = [10, 10, 9.4, 9, 9, 9, 9, 9]
        stop = np.full((8, 1), 9.5)
        r = book(O, C2, [1, 0, 0, 0, 0, 0, 0, 0], stop=stop)
        self.assertEqual(r["trades"][0][:2], (1, 3))         # closed below 9.5 at bar 2
        O3 = np.array([[10.0, 10], [10, 10], [10, 10], [10, 10], [10, 10]])
        C3 = np.array([[10.0, 10], [10, 10], [11, 10], [np.nan, 10], [np.nan, 10]])
        O3[3:, 0] = np.nan
        entry = np.zeros((5, 2), bool)
        entry[0, 0] = True
        r = book(O3, C3, entry)
        self.assertEqual([t[:3] for t in r["trades"]], [(1, 2, 0)])   # its prices end at bar 2
        self.assertAlmostEqual(r["trades"][0][3], 0.1)
        self.assertAlmostEqual(r["equity"][-1], 1.1)

    def test_margin_curve(self):
        eq = np.array([1.0, 1.1, 0.99])
        rf = np.array([0.0, 0.0001, 0.0001])
        m = L3.margin_curve(eq, rf, 0, 2)
        g1 = 1 + 2 * 0.1 - (0.0001 + 0.005 / 252)
        g2 = 1 + 2 * (0.99 / 1.1 - 1) - (0.0001 + 0.005 / 252)
        np.testing.assert_allclose(m, [1, g1, g1 * g2])


class RunnerTest(unittest.TestCase):
    def test_run_lab3_on_a_synthetic_panel(self):
        rng = np.random.default_rng(11)
        n, m = 1300, 12
        dates = pd.bdate_range("2004-01-01", periods=n).strftime("%Y-%m-%d").to_numpy(str)
        C = 40 * np.cumprod(1 + rng.normal(0.0004, 0.018, (n, m)), axis=0)
        O = C * (1 + rng.normal(0, 0.003, (n, m)))
        V = rng.uniform(1e5, 3e5, (n, m))
        member = np.ones((n, m), bool)
        idx = pd.Index(dates)
        g = 1000 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        frames = {"^GSPC": pd.DataFrame({"close": g, "adj_close": g}, index=idx),
                  "^IRX": pd.DataFrame({"close": 2.0, "adj_close": 2.0}, index=idx),
                  "SPY": pd.DataFrame({"close": g, "adj_close": g}, index=idx),
                  "QQQ": pd.DataFrame({"close": g, "adj_close": g}, index=idx)}
        res = L3.run_lab3(dates, [f"S{k}" for k in range(m)], O, C, V, member, frames,
                          start="2005-01-03", train_end="2007-12-31", progress=lambda *a: None)
        self.assertEqual(len(res["variants"]), 24)
        for v in res["variants"].values():
            for p in L3.PERIODS:
                self.assertIsNotNone(v["periods"][p]["cagr_pct"])
        w = res["pick"]["name"]
        self.assertIn(w, res["pick"]["qualified"])
        best = max(L3.exact(res["variants"][q]["periods"]["train"], "cagr_pct")
                   for q in res["pick"]["qualified"])
        self.assertEqual(L3.exact(res["variants"][w]["periods"]["train"], "cagr_pct"), best)
        self.assertEqual(res["winner"]["name"], w)
        self.assertEqual(res["reaches_target"],
                         L3.exact(res["variants"][w]["periods"]["test"], "cagr_pct") >= 40)
        self.assertEqual(res["periods"]["test"][0], res["periods"]["train"][1])
        self.assertIsNotNone(res["benchmarks"]["SPY"]["test"]["cagr_pct"])
        lines = L3.summary_lines({**res, "variants": res["variants"]})
        self.assertTrue(any("reaches 40%" in x for x in lines))

    def test_names(self):
        self.assertEqual(L3.v_name("B", 0.08, "on", 5), "B_base8_filter_on_k5")
        self.assertEqual(L3.v_name("M", 5, "off", 10), "M_rsi5_filter_off_k10")
        self.assertIn("only while the S&P 500", L3.v_label("H", 50, "on", 10))


if __name__ == "__main__":
    unittest.main()
