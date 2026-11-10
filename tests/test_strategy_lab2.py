"""Tests for atlas2.strategy_lab2 and atlas2.lab_data (strategy lab 2).

Synthetic series only; files go to temp dirs. Never touches data/, config.json
or the network.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from atlas2 import lab_data
from atlas2 import strategy_lab as L1
from atlas2 import strategy_lab2 as L2


def bdays(start: str, n: int) -> np.ndarray:
    return np.asarray(pd.bdate_range(start, periods=n).strftime("%Y-%m-%d"), dtype=str)


def random_walk(n: int, seed: int, vol: float = 0.015, drift: float = 0.0004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, n))


class TaxBookTest(unittest.TestCase):
    R = L2.DE_TAX_RATE

    def test_gain_taxed_on_switch_and_paid_from_the_new_position(self):
        b = L2.TaxBook(2)
        b.trade(np.zeros(2), np.array([100.0, 0.0]))
        new = b.trade(np.array([150.0, 0.0]), np.array([0.0, 150.0]))
        self.assertAlmostEqual(b.paid, self.R * 50.0)
        np.testing.assert_allclose(new, [0.0, 150.0 - self.R * 50.0])
        # the new position's basis is what was really bought (after the tax)
        self.assertAlmostEqual(b.basis[1], new[1])

    def test_no_phantom_loss_from_tax_paid(self):
        """QQQ 100 -> 200, switch to IEF, IEF flat, switch back: the only tax is on
        the QQQ gain and nothing is carried forward as a loss."""
        b = L2.TaxBook(2)
        b.trade(np.zeros(2), np.array([100.0, 0.0]))
        ief = b.trade(np.array([200.0, 0.0]), np.array([0.0, 200.0]))
        back = b.trade(ief, np.array([ief[1], 0.0]))
        self.assertAlmostEqual(b.paid, self.R * 100.0)
        self.assertAlmostEqual(b.losses, 0.0)
        self.assertAlmostEqual(back[0], 200.0 - self.R * 100.0)

    def test_loss_carried_forward_and_used(self):
        b = L2.TaxBook(2)
        b.trade(np.zeros(2), np.array([100.0, 0.0]))
        b.trade(np.array([70.0, 0.0]), np.array([0.0, 70.0]))
        self.assertAlmostEqual(b.losses, 30.0)
        self.assertAlmostEqual(b.liquidate(np.array([0.0, 120.0])), self.R * 20.0)
        self.assertAlmostEqual(b.losses, 0.0)

    def test_partial_sale_realises_its_share(self):
        b = L2.TaxBook(2)
        b.basis[1] = 0.0
        b.trade(np.zeros(2), np.array([100.0, 0.0]))
        new = b.trade(np.array([200.0, 0.0]), np.array([50.0, 150.0]))   # sell 3/4 of A
        # paying the tax sells a little more of A, which realises a little more gain
        sold = 1 - new[0] / 200.0
        self.assertAlmostEqual(b.paid, self.R * sold * 100.0)
        self.assertAlmostEqual(new.sum(), 200.0 - b.paid)
        self.assertAlmostEqual(new[0] / new.sum(), 0.25)
        self.assertAlmostEqual(b.basis[0], 100.0 * (1 - sold))

    def test_exemption_shrinks_gains_and_losses(self):
        b = L2.TaxBook(1, np.array([0.3]))
        b.trade(np.zeros(1), np.array([100.0]))
        self.assertAlmostEqual(b.liquidate(np.array([200.0])), self.R * 70.0)
        # -100 on one equity fund, +100 on another: nothing taxed, nothing carried
        b = L2.TaxBook(2, np.array([0.3, 0.3]))
        b.trade(np.zeros(2), np.array([200.0, 0.0]))
        new = b.trade(np.array([100.0, 0.0]), np.array([0.0, 100.0]))
        self.assertAlmostEqual(b.losses, 70.0)
        self.assertAlmostEqual(b.liquidate(new * 2.0), 0.0)
        self.assertAlmostEqual(b.losses, 0.0)

    def test_hold_after_tax(self):
        self.assertAlmostEqual(L2.hold_after_tax_cagr(2.0, 1.0), 100 * (1 - L2.DE_TAX_RATE))
        self.assertAlmostEqual(L2.hold_after_tax_cagr(0.5, 1.0), -50.0)


class VolTargetTest(unittest.TestCase):
    def setUp(self):
        self.dates = bdays("1985-10-01", 2600)
        self.close = random_walk(len(self.dates), 1)
        self.tb = np.full(len(self.dates), 4.0)

    def test_levered_returns_match_lab1_fund_model(self):
        r1 = L1.synthetic_fund_returns(self.close, self.tb, 2)
        r2 = L2.levered_returns(self.close, self.tb, np.full(len(self.close), 2.0))
        np.testing.assert_allclose(r1[1:], r2[1:])
        r1 = L1.synthetic_fund_returns(self.close, self.tb, 1)
        r2 = L2.levered_returns(self.close, self.tb, np.ones(len(self.close)))
        np.testing.assert_allclose(r1[1:], r2[1:])
        cash = L2.levered_returns(self.close, self.tb, np.zeros(len(self.close)))
        np.testing.assert_allclose(cash[1:], L1.cash_returns(self.tb)[1:])

    def test_capped_leverage_equals_lab1_3x(self):
        """A target far above any volatility pins leverage at 3x: no trades, the same
        curve as strategy_lab's always-invested 3x fund."""
        d0 = L1.first_sim_index(self.dates, self.close)
        a = L2.run_vol_target(self.dates, self.close, self.tb, 50.0, 20, "none", d0=d0)
        b = L1.run_leverage(self.dates, self.close, self.tb, 3, "none", d0=d0)
        self.assertFalse(a["trades"].any())
        np.testing.assert_allclose(a["equity"][d0 - 1:], b["equity"][d0 - 1:])

    def test_signal_band_and_filter(self):
        sig = L2.vol_target_signal(self.close, 0.25, 20, "sma200_band")
        ok = np.isfinite(sig)
        self.assertTrue((sig[ok] >= 0).all() and (sig[ok] <= L2.V_LMAX).all())
        st = L1.trend_state(self.close, "sma200_band")
        self.assertTrue((sig[ok & (st == 0)] == 0).all())
        moves = np.abs(np.diff(sig[ok]))
        moves = moves[moves > 0]
        on = sig[ok][1:][np.abs(np.diff(sig[ok])) > 0]
        # every change is either at least a step or a switch to / from T-bills
        prev = sig[ok][:-1][np.abs(np.diff(sig[ok])) > 0]
        self.assertTrue(((moves >= L2.V_STEP - 1e-12) | (on == 0) | (prev == 0)).all())

    def test_signal_uses_only_past_closes(self):
        sig = L2.vol_target_signal(self.close, 0.3, 20, "none")
        changed = self.close.copy()
        changed[1500:] *= 1.5
        sig2 = L2.vol_target_signal(changed, 0.3, 20, "none")
        np.testing.assert_array_equal(sig[:1500], sig2[:1500])

    def test_cost_charged_per_unit_changed(self):
        d0 = L1.first_sim_index(self.dates, self.close)
        a = L2.run_vol_target(self.dates, self.close, self.tb, 0.25, 20, "none", d0=d0)
        b = L2.run_vol_target(self.dates, self.close, self.tb, 0.25, 20, "none", d0=d0,
                              cost_per_unit=0.0)
        self.assertTrue(a["trades"].any())
        self.assertLess(a["equity"][-1], b["equity"][-1])



class VolAccountTest(unittest.TestCase):
    def setUp(self):
        self.dates = bdays("1985-10-01", 2600)
        self.close = random_walk(len(self.dates), 2)
        self.tb = np.full(len(self.dates), 4.0)
        self.fund3 = L1.synthetic_fund_returns(self.close, self.tb, 3)
        self.cash = L1.cash_returns(self.tb)
        self.sig = L2.vol_target_signal(self.close, 0.3, 60, "none")
        self.i0 = 400

    def test_always_3x_is_the_fund_minus_the_opening_trade(self):
        sig = np.full(len(self.dates), 3.0)
        eq = L2.run_vol_account(self.fund3, self.cash, sig, self.i0, len(self.dates) - 1)
        held = np.prod(1 + self.fund3[self.i0 + 1:])
        c = L2.V_ACCOUNT_COST                     # the cash sold and the fund bought both pay
        self.assertAlmostEqual(eq[-1] / (held * (1 - c) / (1 + c)), 1.0, places=12)

    def test_trades_only_when_the_signal_changes(self):
        """Between signal changes nothing is traded: the account equals the drifting
        mix bought at the last change."""
        i1 = len(self.dates) - 1
        eq = L2.run_vol_account(self.fund3, self.cash, self.sig, self.i0, i1)
        changes = [d for d in range(self.i0 + 1, i1 + 1)
                   if self.sig[d - 1] != self.sig[d - 2]]
        self.assertTrue(changes)
        a, b = changes[0], (changes[1] if len(changes) > 1 else i1 + 1)
        lev = self.sig[a - 1]
        fund_part = eq[a] * lev / L2.V_LMAX
        cash_part = eq[a] - fund_part
        for d in range(a + 1, min(b, a + 15)):
            fund_part *= 1 + self.fund3[d]
            cash_part *= 1 + self.cash[d]
            self.assertAlmostEqual(eq[d], fund_part + cash_part, places=10)

    def test_zero_tax_rate_is_untaxed_and_tax_never_helps(self):
        i1 = len(self.dates) - 1
        a = L2.run_vol_account(self.fund3, self.cash, self.sig, self.i0, i1)
        taxed = L2.run_vol_account(self.fund3, self.cash, self.sig, self.i0, i1,
                                   tax_exemption=0.0)
        self.assertLess(taxed[-1], a[-1])
        orig = L2.DE_TAX_RATE
        try:
            L2.TaxBook.__init__.__defaults__ = (None, 0.0)
            b = L2.run_vol_account(self.fund3, self.cash, self.sig, self.i0, i1,
                                   tax_exemption=0.0)
        finally:
            L2.TaxBook.__init__.__defaults__ = (None, orig)
        np.testing.assert_allclose(a[self.i0:], b[self.i0:])

    def test_daily_trades_give_the_daily_leverage_return(self):
        """When the signal changes every day the account is re-weighted every close,
        so (no costs, a fund of exactly 3x the index, no interest) day d earns
        sig[d-2] x the index return - the model's timing."""
        n = len(self.dates)
        r = np.zeros(n)
        r[1:] = self.close[1:] / self.close[:-1] - 1
        sig = np.where(np.arange(n) % 2 == 0, 1.5, 1.8)
        eq = L2.run_vol_account(3 * r, np.zeros(n), sig, self.i0, n - 1, cost=0.0)
        for d in range(self.i0 + 1, self.i0 + 40):
            self.assertAlmostEqual(eq[d] / eq[d - 1], 1 + sig[d - 2] * r[d], places=12)


class RotationTest(unittest.TestCase):
    def test_weights(self):
        ret = {"QQQ": 0.10, "SPY": 0.05, "EFA": 0.08, "IEF": 0.02, "GLD": 0.12}
        self.assertEqual(L2.r_weights("gem_spy", ret, 0.03), {"EFA": 1.0})
        self.assertEqual(L2.r_weights("gem_qqq", ret, 0.03), {"QQQ": 1.0})
        self.assertEqual(L2.r_weights("gem_qqq", ret, 0.20), {"IEF": 1.0})
        self.assertEqual(L2.r_weights("top1", ret, 0.03), {"GLD": 1.0})
        self.assertEqual(L2.r_weights("top1", ret, 0.20), {"CASH": 1.0})
        self.assertEqual(L2.r_weights("top2", ret, 0.11), {"GLD": 0.5, "CASH": 0.5})

    def _setup(self):
        dates = bdays("2004-01-01", 1500)
        px = np.column_stack([random_walk(len(dates), s) for s in range(5)])
        cash = np.full(len(dates), 0.02 / 252)
        me = L1.month_end_signal_idx(dates)
        me_pos = {int(s): j for j, s in enumerate(me)}
        me_pos["idx"] = [int(i) for i in me]
        sig = me[12:]
        return dates, px, cash, sig, me_pos

    def test_trades_follow_previous_close_signal(self):
        dates, px, cash, sig, me_pos = self._setup()
        res = L2.run_rotation(dates, px, cash, sig, me_pos, "top1", 12, int(sig[0]),
                              len(dates) - 1)
        for row in res["log"]:
            self.assertEqual(str(dates[row["trade_idx"] - 1]), row["signal"])
        # the first signal's choice equals the best 12-month return at that close
        s = int(sig[0])
        a0 = me_pos["idx"][me_pos[s] - 12]
        best = L2.R_ASSETS[int(np.argmax(px[s] / px[a0]))]
        self.assertIn(list(res["log"][0]["hold"])[0], (best, "CASH"))

    def test_zero_rate_tax_equals_untaxed(self):
        dates, px, cash, sig, me_pos = self._setup()
        a = L2.run_rotation(dates, px, cash, sig, me_pos, "top2", 6, int(sig[0]), len(dates) - 1)
        orig = L2.DE_TAX_RATE
        try:
            L2.TaxBook.__init__.__defaults__ = (None, 0.0)
            b = L2.run_rotation(dates, px, cash, sig, me_pos, "top2", 6, int(sig[0]),
                                len(dates) - 1, tax_exemption=np.zeros(6))
        finally:
            L2.TaxBook.__init__.__defaults__ = (None, orig)
        np.testing.assert_allclose(a["equity"][int(sig[0]):], b["equity"][int(sig[0]):])

    def test_tax_never_raises_wealth_and_opening_cash_untaxed(self):
        dates, px, cash, sig, me_pos = self._setup()
        a = L2.run_rotation(dates, px, cash, sig, me_pos, "top1", 12, int(sig[0]), len(dates) - 1)
        b = L2.run_rotation(dates, px, cash, sig, me_pos, "top1", 12, int(sig[0]), len(dates) - 1,
                            tax_exemption=np.zeros(6))
        self.assertLessEqual(b["equity"][-1], a["equity"][-1] + 1e-12)
        # the first trade sells only the opening cash: the tax is on that day's interest
        # alone, not on the cash itself
        s = int(sig[0])
        self.assertAlmostEqual(a["equity"][s + 1] - b["equity"][s + 1],
                               L2.DE_TAX_RATE * cash[s + 1], places=6)


class PointInTimeTest(unittest.TestCase):
    def test_member_matrix(self):
        dates = bdays("2000-01-03", 30)
        stints = [{"ticker": "A", "start": str(dates[5]), "end": str(dates[10])},
                  {"ticker": "B", "start": str(dates[20]), "end": None}]
        m = L2.member_matrix(stints, dates, ["A", "B", "C"])
        self.assertEqual(np.nonzero(m[:, 0])[0].tolist(), list(range(5, 11)))
        self.assertEqual(np.nonzero(m[:, 1])[0].tolist(), list(range(20, 30)))
        self.assertFalse(m[:, 2].any())

    def test_usable_stints_drops_possible_ticker_reuse(self):
        stints = [{"ticker": "OLD", "start": "1996-01-02", "end": "2001-05-01"},
                  {"ticker": "OK", "start": "2005-03-01", "end": None},
                  {"ticker": "GONE", "start": "1996-01-02", "end": "2008-09-16"}]
        kept, none, reuse = L2.usable_stints(stints, {"OLD": "2013-10-10", "OK": "1999-01-04"})
        self.assertEqual([s["ticker"] for s in kept], ["OK"])
        self.assertEqual([s["ticker"] for s in none], ["GONE"])
        self.assertEqual([s["ticker"] for s in reuse], ["OLD"])

    def test_coverage_by_year(self):
        dates = bdays("2000-01-03", 260)
        stints = [{"ticker": "A", "start": "2000-01-03", "end": None},
                  {"ticker": "B", "start": "2000-01-03", "end": None}]
        cov = L2.coverage_by_year(stints, stints[:1], dates)
        self.assertEqual(cov["2000"], 50.0)


class LabDataTest(unittest.TestCase):
    def test_yahoo_symbol(self):
        self.assertEqual(lab_data.yahoo_symbol("BRK.B"), "BRK-B")
        self.assertEqual(lab_data.yahoo_symbol("AAPL"), "AAPL")

    def test_ensure_membership_downloads_only_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ref" / "m.csv"
            calls = []

            def fetch(url):
                calls.append(url)
                return b"ticker,start_date,end_date\nAAA,1996-01-02,\n"

            lab_data.ensure_membership(path, fetch=fetch, progress=lambda *a: None)
            lab_data.ensure_membership(path, fetch=fetch, progress=lambda *a: None)
            self.assertEqual(len(calls), 1)
            self.assertEqual(lab_data.load_membership(path),
                             [{"ticker": "AAA", "start": "1996-01-02", "end": None}])
            with self.assertRaises(RuntimeError):
                lab_data.ensure_membership(Path(d) / "x.csv", fetch=lambda u: b"<html>",
                                           progress=lambda *a: None)
            self.assertFalse((Path(d) / "x.csv").exists())

    def test_fetch_logs_every_ticker_offline(self):
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "m.csv"
            mem.write_text("ticker,start_date,end_date\nAAA,1996-01-02,\nZZZ,1996-01-02,2001-01-02\n")
            idx = pd.bdate_range("1996-01-02", periods=3)
            frame = pd.concat({"AAA": pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0,
                                                     "Close": [1.0, 2.0, 3.0], "Volume": 5.0},
                                                    index=idx)}, axis=1)
            out = lab_data.fetch_members(Path(d) / "x.sqlite3", mem, pause_s=0,
                                         progress=lambda *a: None, download=lambda s: frame)
            self.assertEqual(out["with_prices"], 1)
            self.assertEqual(out["without_prices"], 1)
            conn = sqlite3.connect(Path(d) / "x.sqlite3")
            log = dict(conn.execute("SELECT ticker, n_rows FROM fetch_log").fetchall())
            self.assertEqual(log, {"AAA": 3, "ZZZ": 0})
            conn.close()


if __name__ == "__main__":
    unittest.main()
