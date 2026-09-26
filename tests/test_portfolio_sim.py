"""Tests for atlas2.portfolio_sim (synthetic trades, temp price DBs; never touches data/)."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from atlas2 import backtest as bt
from atlas2 import portfolio_sim as ps
from atlas2.data import PriceStore
from tests.test_backtest import make_trade

REPO = Path(__file__).resolve().parent.parent
TUNING = {"auto_disable": True, "min_avg_r": 0.0, "min_trades": 30, "require_proven_edge": False,
          "disabled": {"us": [], "de": [], "in": []}}


def ptrade(**kw) -> dict:
    t = {"market": "us", "ticker": "AAA", "pattern": "p", "conf": 0.8,
         "entry_date": "2020-01-03", "exit_date": "2020-01-10", "r": 1.0, "r_net": 1.0,
         "outcome": "target"}
    t.update(kw)
    return t


def tie_stream() -> list[dict]:
    """Two equally ranked setups on one day with one free slot."""
    return [ptrade(ticker="WIN", r_net=1.0), ptrade(ticker="LOSE", r=-1.0, r_net=-1.0)]


def pick(seed: int) -> str:
    out = ps.simulate_portfolio(tie_stream(), max_pos=1, tie_seed=seed, record_positions=True)
    return out["_positions"][0]["ticker"]


def busy_stream(seed: int = 7, days: int = 60, per_day: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    dates = pd.bdate_range("2019-01-02", periods=days + 20).strftime("%Y-%m-%d")
    for d in range(days):
        for k in range(per_day):
            rn = float(np.round(rng.normal(0.1, 1.0), 3))
            out.append(ptrade(ticker=f"T{d}_{k}", conf=float(rng.choice([0.7, 0.8])),
                              entry_date=dates[d], exit_date=dates[d + int(rng.integers(2, 15))],
                              r=rn, r_net=rn, outcome="time"))
    return out


def write_db(path: Path, frames: dict[str, pd.DataFrame]) -> Path:
    store = PriceStore(path)
    rows = [(t, str(d)[:10], float(r.open), float(r.high), float(r.low), float(r.close),
             float(r.volume)) for t, df in frames.items() for d, r in df.iterrows()]
    store.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows)
    store.conn.commit()
    store.close()
    return path


def closes_frame(dates, closes) -> pd.DataFrame:
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6},
                        index=pd.Index(list(dates), name="date"))


class RegimeTimingTest(unittest.TestCase):
    def test_previous_session_multiplier(self):
        t = ptrade()
        out = ps.simulate_portfolio([t], regime={"us": {"2020-01-02": 1.0, "2020-01-03": 0.0}})
        self.assertEqual((out["trades_taken"], out["skipped_regime"]), (1, 0))
        self.assertEqual(out["final_equity"], 25250)
        out = ps.simulate_portfolio([t], regime={"us": {"2020-01-02": 0.0, "2020-01-03": 1.0}})
        self.assertEqual((out["trades_taken"], out["skipped_regime"]), (0, 1))

    def test_mult_for(self):
        reg = {"us": {"2020-01-02": 0.5, "2020-01-03": 1.0}}
        self.assertEqual(ps._mult_for(reg, "us", "2020-01-06"), 1.0)
        self.assertEqual(ps._mult_for(reg, "us", "2020-01-03"), 0.5)
        self.assertEqual(ps._mult_for(reg, "us", "2020-01-01"), 1.0)
        self.assertEqual(ps._mult_for(reg, "us", "2020-01-02"), 1.0)
        self.assertEqual(ps._mult_for({}, "us", "2020-01-02"), 1.0)

    def test_regime_multipliers_match_the_backtest_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 320
            idx = pd.bdate_range("2019-01-01", periods=n).strftime("%Y-%m-%d")
            c = 3000 * np.exp(np.cumsum(0.0003 + 0.01 * np.random.default_rng(2).standard_normal(n)))
            df = closes_frame(idx, c)
            db = write_db(Path(tmp) / "p.sqlite3", {"^GSPC": df})
            reg = ps.regime_multipliers(db)
            statuses = bt.regime_statuses(df)
            self.assertEqual(list(reg["us"].values()), [ps.REGIME_MULT[s] for s in statuses])
            self.assertEqual(reg["us"][idx[208]], 1.0)          # warm-up: 'unknown' = full size


class TieBreakTest(unittest.TestCase):
    def test_pick_is_order_independent_and_seeded(self):
        a = ps.simulate_portfolio(tie_stream(), max_pos=1, record_positions=True)
        b = ps.simulate_portfolio(list(reversed(tie_stream())), max_pos=1, record_positions=True)
        self.assertEqual(a["_positions"][0]["ticker"], b["_positions"][0]["ticker"])
        picks = [pick(sd) for sd in range(20)]
        self.assertEqual(set(picks), {"WIN", "LOSE"})
        self.assertEqual(picks, [pick(sd) for sd in range(20)])

    def test_pick_independent_of_hash_seed(self):
        here = [pick(sd) for sd in range(10)]
        code = ("import json; from tests.test_portfolio_sim import pick; "
                "print(json.dumps([pick(s) for s in range(10)]))")
        for seed in ("0", "4242"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                                 capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(json.loads(out.stdout.strip().splitlines()[-1]), here)

    def test_legacy_trades_and_open_at_end(self):
        legacy = ptrade(r_net=None)
        legacy.pop("r_net")
        out = ps.simulate_portfolio([legacy])
        self.assertEqual(out["final_equity"], round(25000 + 250 * (1.0 - ps.LEGACY_COST_R)))
        self.assertEqual(out["cost_source"], {"r_net": 0, "legacy_cost_r": 1})
        out = ps.simulate_portfolio([ptrade(outcome="open_at_end"), ptrade(ticker="B")])
        self.assertEqual(out["trades_available"], 1)


class MedianPathTest(unittest.TestCase):
    def setUp(self):
        self.trades = busy_stream()
        self.win = ps.stream_window(self.trades)
        self.kw = {"window": self.win}

    def test_representative_run_is_the_exact_median(self):
        r = ps.run_profile(self.trades, self.kw, blended_idx=5.0, reshuffles=20)
        raws = {sd: ps.simulate_portfolio(self.trades, tie_seed=sd, **self.kw)["_cagr_raw"]
                for sd in range(21)}
        med = sorted(raws.values())[10]
        self.assertEqual(r["representative_tie_seed"], min(sd for sd, v in raws.items() if v == med))
        self.assertAlmostEqual(r["_cagr_raw"], med)
        self.assertEqual(r["seed0_cagr_pct"], round(raws[0], 1))
        tb = r["tie_break_sensitivity"]
        self.assertEqual(tb["runs"], 21)
        for key in ("cagr_pct", "max_drawdown_pct", "vs_blended_index_cagr_pp"):
            q = tb[key]
            self.assertLessEqual(q["p5"], q["median"])
            self.assertLessEqual(q["median"], q["p95"])
        self.assertEqual(tb["cagr_pct"]["median"], r["cagr_pct"])
        self.assertEqual(tb["cagr_pct_median"], r["cagr_pct"])      # back-compat key
        self.assertTrue(0.0 <= tb["beats_blended_index_share"] <= 1.0)
        self.assertGreater(len(set(round(v, 6) for v in raws.values())), 1, "ties should matter")

    def test_yearly_return_is_on_the_previous_year_end_equity(self):
        a = ptrade(ticker="A", entry_date="2020-03-02", exit_date="2020-06-01", r_net=10.0)
        b = ptrade(ticker="B", entry_date="2021-03-01", exit_date="2021-06-01", r_net=1.0)
        r = ps.simulate_portfolio([a, b], sizing="r", cash_limit=False)
        self.assertEqual(r["yearly_pnl"], {"2020": 2500, "2021": 275})
        self.assertEqual(r["yearly_return_pct"], {"2020": 10.0, "2021": 1.0})   # 275 / 27,500

    def test_yearly_returns_compound_to_final_equity(self):
        r = ps.simulate_portfolio(self.trades, **self.kw)
        growth = 1.0
        for v in r["yearly_return_pct"].values():
            growth *= 1 + v / 100
        self.assertAlmostEqual(growth, r["final_equity"] / ps.START_CAPITAL, delta=0.003)
        years = ps._years_between(*self.win)
        self.assertAlmostEqual(((r["final_equity"] / ps.START_CAPITAL) ** (1 / max(0.5, years)) - 1) * 100,
                               r["cagr_pct"], delta=0.06)


class SizingTest(unittest.TestCase):
    def test_dashboard_sizing_from_the_entry_level(self):
        t = ptrade(entry_price=102.0, entry_level=100.0, stop=90.0, r_net=0.5)
        out = ps.simulate_portfolio([t], record_positions=True)
        pos = out["_positions"][0]
        self.assertAlmostEqual(pos["shares"], 25.0)                   # 250 / (100 - 90)
        self.assertAlmostEqual(pos["pnl"], 25.0 * 0.5 * (102.0 - 90.0))
        self.assertEqual(out["sizing_source"], {"entry_level": 1})
        self.assertEqual(out["capped_trades"], 0)

    def test_position_cap(self):
        t = ptrade(entry_price=100.0, entry_level=100.0, stop=99.0, r_net=1.0)
        out = ps.simulate_portfolio([t], record_positions=True)
        self.assertAlmostEqual(out["_positions"][0]["shares"], 25000 * 0.25 / 100)
        self.assertEqual(out["capped_trades"], 1)
        self.assertEqual(out["pct_positions_capped"], 100.0)

    def test_entry_level_rebuilt_from_rr(self):
        t = ptrade(entry_price=101.0, stop=90.0, target=120.0, rr=2.0, r_net=1.0)
        self.assertEqual(ps._entry_level(t), (100.0, "rebuilt_from_rr"))
        out = ps.simulate_portfolio([t])
        self.assertEqual(out["sizing_source"], {"rebuilt_from_rr": 1})

    def test_cash_limit(self):
        a = ptrade(ticker="A", entry_price=100.0, entry_level=100.0, stop=98.75, r_net=0.1)
        b = ptrade(ticker="B", entry_price=100.0, entry_level=100.0, stop=98.75, r_net=0.1)
        out = ps.simulate_portfolio([a, b], max_position_pct=100.0, record_positions=True)
        shares = sorted(p["shares"] for p in out["_positions"])
        self.assertAlmostEqual(shares[1], 200.0)                      # 250 / 1.25
        self.assertAlmostEqual(shares[0], 50.0)                       # the 5,000 that was left
        self.assertEqual(out["cash_limited_trades"], 1)

    def test_capped_counted_only_for_trades_taken(self):
        # three capped trades on one day; the cash runs out before the third
        ts = [ptrade(ticker=k, entry_price=100.0, entry_level=100.0, stop=99.0, r_net=0.1)
              for k in ("A", "B", "C")]
        out = ps.simulate_portfolio(ts, max_position_pct=60.0, record_positions=True)
        self.assertEqual((out["trades_taken"], out["skipped_no_cash"]), (2, 1))
        self.assertEqual(out["cash_limited_trades"], 1)
        self.assertEqual(out["capped_trades"], 2)
        self.assertLessEqual(out["capped_trades"], out["trades_taken"])
        self.assertEqual(out["pct_positions_capped"], 100.0)

    def test_cap_is_measured_at_the_entry_level_not_the_fill(self):
        t = ptrade(entry_price=104.0, entry_level=100.0, stop=99.0, r_net=1.0)
        out = ps.simulate_portfolio([t], record_positions=True)
        self.assertAlmostEqual(out["_positions"][0]["shares"], 25000 * 0.25 / 100.0)
        self.assertEqual(out["capped_trades"], 1)

    def test_r_sizing_is_the_old_replay(self):
        t = ptrade(entry_price=102.0, entry_level=100.0, stop=90.0, r_net=0.5)
        out = ps.simulate_portfolio([t], sizing="r", cash_limit=False)
        self.assertEqual(out["final_equity"], 25125)


class SameDayExitTest(unittest.TestCase):
    """An exit on day d (intraday or at the close) happens after that day's entries
    (at the open), so it frees neither a slot nor cash for them."""

    def test_same_day_exit_does_not_free_the_slot(self):
        a = ptrade(ticker="A", entry_date="2020-01-02", exit_date="2020-01-06")
        b = ptrade(ticker="B", entry_date="2020-01-06", exit_date="2020-01-10")
        out = ps.simulate_portfolio([a, b], max_pos=1, record_positions=True)
        self.assertEqual(out["skipped_no_free_slot"], 1)
        self.assertEqual([p["ticker"] for p in out["_positions"]], ["A"])
        # an exit the day before does free it
        a2 = dict(a, exit_date="2020-01-03")
        out = ps.simulate_portfolio([a2, b], max_pos=1, record_positions=True)
        self.assertEqual(out["skipped_no_free_slot"], 0)
        self.assertEqual([p["ticker"] for p in out["_positions"]], ["A", "B"])

    def test_pooled_markets_and_cash(self):
        # a US position closing on 01-06 does not fund an Indian entry the same date
        us = ptrade(market="us", ticker="U", entry_date="2020-01-02", exit_date="2020-01-06",
                    entry_price=100.0, entry_level=100.0, stop=99.0, r_net=1.0)
        ind = ptrade(market="in", ticker="I", entry_date="2020-01-06", exit_date="2020-01-10",
                     entry_price=100.0, entry_level=100.0, stop=99.0, r_net=1.0)
        out = ps.simulate_portfolio([us, ind], max_position_pct=99.0, record_positions=True)
        self.assertEqual(out["skipped_no_cash"], 1)
        self.assertEqual(out["trades_taken"], 1)
        # sized on equity before the same-day exit's profit is booked
        out = ps.simulate_portfolio([us, dict(ind, entry_price=100.0, stop=90.0, entry_level=100.0)],
                                    max_pos=6, cash_limit=False, max_position_pct=100.0,
                                    record_positions=True)
        second = out["_positions"][1]
        self.assertAlmostEqual(second["shares"], 25000 * 0.01 / 10.0)

    def test_same_day_entry_and_exit_closes_that_day(self):
        t = ptrade(ticker="X", entry_date="2020-01-06", exit_date="2020-01-06", r_net=-1.0)
        later = ptrade(ticker="Y", entry_date="2020-01-07", exit_date="2020-01-09")
        out = ps.simulate_portfolio([t, later], max_pos=1, record_positions=True)
        self.assertEqual([p["ticker"] for p in out["_positions"]], ["X", "Y"])
        self.assertEqual(out["final_equity"], 25000 - 250 + round(24750 * 0.01))


class ExposureTest(unittest.TestCase):
    def test_market_weights_from_capital_days(self):
        v2 = ptrade(market="us", ticker="U", entry_date="2020-01-02", exit_date="2020-01-12",
                    entry_price=102.0, entry_level=100.0, stop=90.0, r_net=0.5)
        v1 = {"market": "de", "ticker": "D", "pattern": "p", "conf": 0.8, "entry_date": "2020-01-02",
              "exit_date": "2020-01-22", "r": 1.0, "ret_pct": 8.0, "outcome": "target"}
        out = ps.simulate_portfolio([v2, v1])
        us, de = 25 * 102 / 25000 * 10, 250 / 0.08 / 25000 * 20
        self.assertAlmostEqual(out["market_exposure_weights"]["us"], round(us / (us + de), 4))
        self.assertAlmostEqual(out["market_exposure_weights"]["de"], round(de / (us + de), 4))


class IndexPortfolioTest(unittest.TestCase):
    def test_monthly_rebalanced_blend_with_a_holiday(self):
        d = ["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"]
        a = pd.Series([100.0, 110, 121, 121], index=d)
        b = pd.Series([100.0, 90, 99], index=[d[0], d[2], d[3]])      # shut on 01-31
        c = pd.Series([100.0, 100, 100, 110], index=d)
        out = ps._index_portfolio({"us": a, "de": b, "in": c}, {"us": 1, "de": 1, "in": 1},
                                  d[0], d[-1])
        jan = (1.1 + 1.0 + 1.0) / 3
        feb = (1.1 + 0.99 + 1.1) / 3
        self.assertAlmostEqual(out["_growth"], jan * feb)
        self.assertNotAlmostEqual(out["_growth"], (1.21 + 0.99 + 1.1) / 3)   # not buy-and-hold
        self.assertEqual(out["yearly_pct"], {"2020": round((jan * feb - 1) * 100, 1)})
        years = 5 / 365.25
        self.assertAlmostEqual(out["cagr_pct"], round(((jan * feb) ** (1 / years) - 1) * 100, 2))

    def test_dividend_yield_accrues_on_a_flat_index(self):
        idx = pd.bdate_range("2018-01-02", "2021-12-31").strftime("%Y-%m-%d")
        flat = pd.Series(100.0, index=idx)
        with_div = ps._index_portfolio({"us": flat}, {"us": 1.0}, idx[0], idx[-1], {"us": 1.5})
        self.assertAlmostEqual(with_div["cagr_pct"], 1.5, places=2)
        self.assertEqual(ps._index_portfolio({"us": flat}, {"us": 1.0}, idx[0], idx[-1])["cagr_pct"],
                         0.0)
        with tempfile.TemporaryDirectory() as tmp:
            db = write_db(Path(tmp) / "p.sqlite3", {"^GSPC": closes_frame(idx, [100.0] * len(idx))})
            for y, want in ((1.5, 1.5), (0.0, 0.0)):
                b = ps.benchmark_block(idx[0], idx[-1], ["us"], db, Path(tmp) / "nouni", {"us": y})
                self.assertAlmostEqual(b["blended_index_cagr_pct"], want, places=2)
                self.assertAlmostEqual(b["us"]["index_cagr_pct"], want, places=2)

    def test_yearly_figures_compound_to_the_cagr(self):
        idx = pd.bdate_range("2019-01-02", "2021-06-30").strftime("%Y-%m-%d")
        t = np.arange(len(idx))
        s1 = pd.Series(100 * np.exp(0.0004 * t + 0.02 * np.sin(t / 9)), index=idx)
        s2 = pd.Series(50 * np.exp(0.0002 * t + 0.03 * np.cos(t / 13)), index=idx)
        out = ps._index_portfolio({"us": s1, "de": s2}, {"us": 0.7, "de": 0.3}, idx[0], idx[-1],
                                  {"us": 1.5, "de": 0.0})
        growth = 1.0
        for v in out["yearly_pct"].values():
            growth *= 1 + v / 100
        self.assertAlmostEqual(growth, out["_growth"], delta=0.002)
        self.assertAlmostEqual(out["cagr_pct"],
                               (out["_growth"] ** (1 / out["years"]) - 1) * 100, delta=0.01)
        self.assertEqual(out["weights"], {"us": 0.7, "de": 0.3})


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_index_dividends_drawdown_and_yearly_segments(self):
        dates = ["2020-01-02", "2020-06-01", "2020-12-31", "2021-06-30"]
        db = write_db(self.dir / "p.sqlite3", {"^GSPC": closes_frame(dates, [100, 110, 99, 121])})
        conn = sqlite3.connect(db)
        try:
            out = ps._market_benchmark(conn, "us", "^GSPC", dates[0], dates[-1],
                                       self.dir / "no_universe", 1.5)
        finally:
            conn.close()
        years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
        cagr_price = (1.21 ** (1 / years) - 1) * 100
        self.assertAlmostEqual(out["index_cagr_price_pct"], round(cagr_price, 2))
        self.assertAlmostEqual(out["index_cagr_pct"],
                               round(((1 + cagr_price / 100) * 1.015 - 1) * 100, 2))
        self.assertAlmostEqual(out["index_yearly_pct"]["2020"],
                               round((0.99 * 1.015 ** (364 / 365.25) - 1) * 100, 1))
        self.assertAlmostEqual(out["index_yearly_pct"]["2021"],
                               round((121 / 99 * 1.015 ** (181 / 365.25) - 1) * 100, 1))
        self.assertAlmostEqual(out["index_max_drawdown_pct"], 10.0)
        self.assertIn("universe_error", out)

    def test_universe_daily_returns_are_capped(self):
        dates = list(pd.bdate_range("2020-01-02", periods=60).strftime("%Y-%m-%d"))
        db = write_db(self.dir / "p.sqlite3", {
            "^GSPC": closes_frame(dates, [100.0] * 60),
            "AAA": closes_frame(dates, [100.0] * 60),
            "BBB": closes_frame(dates, [100.0] * 30 + [170.0] * 30)})    # +70% in one day
        uni = self.dir / "uni"
        uni.mkdir()
        pd.DataFrame({"ticker": ["AAA", "BBB"], "name": "x", "index": "T"}).to_csv(uni / "us.csv",
                                                                                  index=False)
        conn = sqlite3.connect(db)
        try:
            out = ps._market_benchmark(conn, "us", "^GSPC", dates[0], dates[-1], uni, 0.0)
        finally:
            conn.close()
        self.assertEqual(out["clipped_ticker_days"], 1)
        self.assertAlmostEqual(out["universe_total_return_pct"], 25.0, delta=0.01)   # (0 + 50%) / 2

    def test_hurdle_uses_the_profile_weights(self):
        idx = pd.bdate_range("2019-01-02", periods=400).strftime("%Y-%m-%d")
        t = np.arange(400)
        frames = {"^GSPC": closes_frame(idx, 3000 * np.exp(0.0006 * t)),
                  "^GDAXI": closes_frame(idx, 12000 * np.exp(-0.0002 * t))}
        db = write_db(self.dir / "p.sqlite3", frames)
        trades = []
        for k in range(0, 300, 5):
            m = "us" if k % 15 else "de"                               # one third DE
            trades.append(ptrade(market=m, ticker=f"{m}{k}", entry_date=idx[k], exit_date=idx[k + 10],
                                 entry_price=100.0, entry_level=100.0, stop=95.0, r_net=0.2))
        tp = self.dir / "trades.json"
        tp.write_text(json.dumps(trades))
        block = ps.run_portfolio_report(progress=lambda *_: None, trades_path=tp,
                                        backtest_path=self.dir / "missing.json",
                                        out_path=self.dir / "pf.json", prices_path=db,
                                        universe_dir=self.dir / "nouni", config={}, reshuffles=2)
        win = ps.stream_window(trades)
        series = ps.benchmark_block(win[0], win[1], ["us", "de"], db, self.dir / "nouni",
                                    ps.dividend_yields(None, {}))["_series"]
        prof = {p["key"]: p for p in block["profiles"]}["as_designed"]
        w = prof["market_exposure_weights"]
        self.assertGreater(w["us"], w["de"])
        want = ps._index_portfolio(series["index"], w, win[0], win[1], series["div"])["cagr_pct"]
        self.assertEqual(prof["hurdle_index_cagr_pct"], want)
        self.assertNotEqual(prof["hurdle_index_cagr_pct"], block["benchmark"]["blended_index_cagr_pct"])
        self.assertAlmostEqual(prof["vs_blended_index_cagr_pp"],
                               round(prof["cagr_pct"] - block["benchmark"]["blended_index_cagr_pct"], 1),
                               delta=0.11)
        self.assertTrue(json.loads((self.dir / "pf.json").read_text())["profiles"])


    def test_universe_uses_the_backtests_price_repairs(self):
        dates = list(pd.bdate_range("2020-01-02", periods=60).strftime("%Y-%m-%d"))
        split = [100.0] * 30 + [50.0] * 30          # an unadjusted 2:1 split on day 30
        db = write_db(self.dir / "p.sqlite3", {
            "^GSPC": closes_frame(dates, np.linspace(100, 110, 60)),
            "AAA": closes_frame(dates, split), "BBB": closes_frame(dates, [80.0] * 60)})
        uni = self.dir / "uni"
        uni.mkdir()
        pd.DataFrame({"ticker": ["AAA", "BBB"], "name": "x", "index": "T"}).to_csv(uni / "us.csv",
                                                                                  index=False)
        fix = ({"ticker": "AAA", "date": dates[30], "action": "adjust", "ratio": 0.5},)
        conn = sqlite3.connect(db)
        try:
            raw = ps._market_benchmark(conn, "us", "^GSPC", dates[0], dates[-1], uni, 0.0)
            rep = ps._market_benchmark(conn, "us", "^GSPC", dates[0], dates[-1], uni, 0.0,
                                       overrides=fix)
        finally:
            conn.close()
        self.assertAlmostEqual(raw["universe_total_return_pct"], -25.0, delta=0.1)   # split = -50%
        self.assertAlmostEqual(rep["universe_total_return_pct"], 0.0, delta=0.01)    # flat
        # run_portfolio_report feeds the config's repairs to the benchmark
        ts = [ptrade(ticker="AAA", entry_date=dates[2], exit_date=dates[50])]
        tp = self.dir / "t.json"
        tp.write_text(json.dumps(ts))
        block = ps.run_portfolio_report(progress=lambda *_: None, trades_path=tp,
                                        backtest_path=self.dir / "none.json",
                                        out_path=self.dir / "pf.json", prices_path=db,
                                        universe_dir=uni,
                                        config={"backtest": {"price_artifacts": list(fix)}},
                                        reshuffles=0)
        self.assertAlmostEqual(block["benchmark"]["us"]["universe_total_return_pct"], 0.0, delta=0.01)
        self.assertTrue(any("price repairs" in n for n in block["benchmark"]["notes"]))


class MarkToMarketTest(unittest.TestCase):
    def test_open_positions_marked_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = list(pd.bdate_range("2021-01-04", periods=10).strftime("%Y-%m-%d"))
            db = write_db(Path(tmp) / "p.sqlite3", {
                "AAA": closes_frame(d, [100, 95, 80, 70, 75, 90, 100, 100, 100, 100]),
                "BBB": closes_frame(d, [50] * 10)})
            positions = [
                {"ticker": "AAA", "entry_date": d[0], "exit_date": d[9], "entry_price": 100.0,
                 "shares": 10.0, "pnl": 0.0},
                {"ticker": "BBB", "entry_date": d[1], "exit_date": d[6], "entry_price": 50.0,
                 "shares": 20.0, "pnl": 50.0}]
            conn = sqlite3.connect(db)
            try:
                mtm, why = ps.mark_to_market(positions, 25000.0, conn)
            finally:
                conn.close()
        self.assertIsNone(why)
        self.assertAlmostEqual(mtm["mtm_max_drawdown_pct"], 1.2)          # 300 of 25,000
        self.assertGreater(mtm["mtm_max_drawdown_pct"], 0.0)              # realized: 0
        # day 2: AAA 10 x 95 + BBB 20 x 50 held on equity 25,000 - 50 (marked)
        self.assertAlmostEqual(mtm["max_gross_exposure_pct"], round(1950 / 24950 * 100, 1))

    def test_marks_use_the_backtests_repaired_prices(self):
        """Fills come from the backtest's repaired (back-adjusted) series, so the
        marks must too: an unadjusted 2:1 split and a listed override must not
        show a paper gain that vanishes at the exit."""
        with tempfile.TemporaryDirectory() as tmp:
            d = list(pd.bdate_range("2021-01-04", periods=30).strftime("%Y-%m-%d"))
            spl = closes_frame(d, [100.0] * 15 + [50.0] * 15)     # raw, unadjusted split
            spl.iloc[15, spl.columns.get_loc("volume")] = 3e6    # the share count changed
            dem = closes_frame(d, [80.0] * 20 + [30.0] * 10)      # demerger: no exact ratio
            db = write_db(Path(tmp) / "p.sqlite3", {"SPL": spl, "DEM": dem})
            positions = [   # fills on the repaired scale (the backtest's entry_price)
                {"ticker": "SPL", "entry_date": d[2], "exit_date": d[25], "entry_price": 50.0,
                 "shares": 10.0, "pnl": 0.0},
                {"ticker": "DEM", "entry_date": d[3], "exit_date": d[27], "entry_price": 30.0,
                 "shares": 10.0, "pnl": 0.0}]
            override = [{"ticker": "DEM", "date": d[20], "action": "adjust", "ratio": 30 / 80}]
            conn = sqlite3.connect(db)
            try:
                raw, _ = ps.mark_to_market(positions, 25000.0, conn)
                ov, stale = ps.price_repairs({"backtest": {"price_artifacts": override}},
                                             Path(tmp), conn)
                fixed, why = ps.mark_to_market(positions, 25000.0, conn, overrides=ov,
                                               stale_map=stale, cache={})
                no_ov, _ = ps.price_repairs({}, Path(tmp), conn)
                (Path(tmp) / "price_artifacts.json").write_text(json.dumps(override))
                file_ov, _ = ps.price_repairs({}, Path(tmp), conn)
            finally:
                conn.close()
        self.assertIsNone(why)
        self.assertGreater(raw["mtm_max_drawdown_pct"], 1.0)     # fake gains that vanish
        self.assertEqual(fixed["mtm_max_drawdown_pct"], 0.0)
        # config defaults (the known VEDL / TMPV / TRENT actions) + data/price_artifacts.json
        self.assertNotIn("DEM", [o["ticker"] for o in no_ov])
        self.assertIn("VEDL.NS", [o["ticker"] for o in no_ov])
        self.assertEqual([o["ticker"] for o in file_ov],
                         [o["ticker"] for o in no_ov] + ["DEM"])

    def test_report_headline_drawdown_is_marked_to_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            d = list(pd.bdate_range("2021-01-04", periods=12).strftime("%Y-%m-%d"))
            db = write_db(tmp / "p.sqlite3", {
                "^GSPC": closes_frame(d, [3000.0] * 12),
                "AAA": closes_frame(d, [100, 100, 100, 95, 90, 80, 90, 100, 101, 101, 101, 101])})
            t = ptrade(ticker="AAA", entry_date=d[2], exit_date=d[8], entry_price=100.0,
                       entry_level=100.0, stop=95.0, r_net=0.2)
            tp = tmp / "backtest_trades.json"
            tp.write_text(json.dumps([t]))
            block = ps.run_portfolio_report(progress=lambda *_: None, trades_path=tp,
                                            backtest_path=tmp / "none.json", out_path=tmp / "pf.json",
                                            prices_path=db, universe_dir=tmp / "nouni", config={},
                                            reshuffles=0)
        prof = {p["key"]: p for p in block["profiles"]}["as_designed"]
        self.assertEqual(prof["drawdown_basis"], "marked to market daily")
        self.assertEqual(prof["max_drawdown_realized_pct"], 0.0)
        self.assertAlmostEqual(prof["max_drawdown_pct"], 4.0)      # 50 shares x 20 of 25,000
        self.assertGreater(prof["max_drawdown_pct"], prof["max_drawdown_realized_pct"])

    def test_no_share_counts_no_mtm(self):
        mtm, why = ps.mark_to_market([{"ticker": "A", "entry_date": "2021-01-04",
                                       "exit_date": "2021-01-05", "entry_price": None,
                                       "shares": None, "pnl": 1.0}], 25000.0, None)
        self.assertIsNone(mtm)
        self.assertIn("MTM not available", why)


class GradeRankingTest(unittest.TestCase):
    def test_rank_field_and_ranking(self):
        a = ptrade(ticker="A", conf=0.9, composite_proxy=60.0, r_net=-1.0)
        b = ptrade(ticker="B", conf=0.6, composite_proxy=80.0, r_net=1.0)
        self.assertEqual(ps.rank_field([a, b]), "composite_proxy")
        out = ps.simulate_portfolio([a, b], max_pos=1, record_positions=True)
        self.assertEqual(out["_positions"][0]["ticker"], "B")
        b2 = dict(b)
        b2.pop("composite_proxy")
        self.assertEqual(ps.rank_field([a, b2]), "conf")
        out = ps.simulate_portfolio([a, b2], max_pos=1, record_positions=True)
        self.assertEqual(out["_positions"][0]["ticker"], "A")

    def test_a_grades_only_in_caution(self):
        reg = {"us": {"2020-01-02": 0.5, "2020-01-03": 1.0, "2020-01-06": 1.0}}
        caution = ptrade(ticker="C", entry_date="2020-01-03", exit_date="2020-01-08", grade_proxy="B")
        normal = ptrade(ticker="N", entry_date="2020-01-07", exit_date="2020-01-09", grade_proxy="B")
        out = ps.simulate_portfolio([caution, normal], regime=reg, grade_filter="A_in_caution",
                                    record_positions=True)
        self.assertEqual([p["ticker"] for p in out["_positions"]], ["N"])
        self.assertEqual(out["skipped_grade"], 1)
        out = ps.simulate_portfolio([caution, normal], grade_filter="A")
        self.assertEqual(out.get("error"), "no trades pass the filter")


class WalkForwardTest(unittest.TestCase):
    def stream(self):
        ts = []
        for k in range(40):                        # 2018-19: a winner
            y = 2018 + k // 20
            ts.append(make_trade(pattern="p", entry_date=f"{y}-{1 + k % 10:02d}-05",
                                 signal_date=f"{y}-{1 + k % 10:02d}-04",
                                 exit_date=f"{y}-{1 + k % 10:02d}-20", r_net=0.5, r=0.55,
                                 excess_vs_index_pct=1.0, excess_vs_universe_pct=1.0))
        for k in range(200):                       # 2020-21: a loser
            y = 2020 + k // 100
            ts.append(make_trade(pattern="p", entry_date=f"{y}-{1 + k % 11:02d}-05",
                                 signal_date=f"{y}-{1 + k % 11:02d}-04",
                                 exit_date=f"{y}-{1 + k % 11:02d}-20", r_net=-0.5, r=-0.45,
                                 excess_vs_index_pct=-1.0, excess_vs_universe_pct=-1.0))
        for k in range(10):                        # too few trades to judge
            ts.append(make_trade(pattern="q", entry_date=f"2018-{1 + k:02d}-05",
                                 signal_date=f"2018-{1 + k:02d}-04", exit_date=f"2018-{1 + k:02d}-20",
                                 r_net=-1.0, r=-0.9, excess_vs_index_pct=-2.0,
                                 excess_vs_universe_pct=-2.0))
        return ts

    def test_judged_on_earlier_trades_only(self):
        ts = self.stream()
        wf, _ = ps.walk_forward_disabled(ts, TUNING)
        self.assertEqual(wf["us"]["2020"], [])                          # judged on 2018-19 only
        self.assertIn("p", wf["us"]["2021"])
        self.assertNotIn("q", wf["us"]["2019"])                         # under min_trades
        in_sample = bt.disabled_patterns(bt.summarize(ts, TUNING)["per_pattern"], TUNING, "us")
        self.assertIn("p", in_sample)
        json.dumps(wf)
        for m, by_year in wf.items():
            for y, lst in by_year.items():
                self.assertIsInstance(y, str)
                self.assertIsInstance(lst, list)

    def test_later_trades_never_change_earlier_decisions(self):
        ts = self.stream()
        wf, _ = ps.walk_forward_disabled(ts, TUNING)
        for cut in ("2019", "2020"):
            mutated = [dict(t, r_net=-t["r_net"], excess_vs_universe_pct=-t["excess_vs_universe_pct"],
                            excess_vs_index_pct=-t["excess_vs_index_pct"])
                       if t["exit_date"] >= f"{cut}-01-01" else t for t in ts]
            wf2, _ = ps.walk_forward_disabled(mutated, TUNING)
            for y in wf["us"]:
                if y <= cut:
                    self.assertEqual(wf2["us"][y], wf["us"][y], (cut, y))

    def test_old_streams_have_no_walk_forward(self):
        wf, note = ps.walk_forward_disabled([ptrade()], TUNING)
        self.assertIsNone(wf)
        self.assertIn("v2", note)


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, trades, backtest=None, live=None, reshuffles=2):
        tp = self.dir / "backtest_trades.json"
        tp.write_text(json.dumps(trades))
        bp = self.dir / "backtest_latest.json"
        if backtest is not None:
            bp.write_text(json.dumps(backtest))
        if live is not None:
            (self.dir / "backtest_trades_live.json").write_text(json.dumps(live))
        block = ps.run_portfolio_report(progress=lambda *_: None, trades_path=tp, backtest_path=bp,
                                        out_path=self.dir / "pf.json",
                                        prices_path=self.dir / "none.sqlite3",
                                        universe_dir=self.dir / "nouni", config={},
                                        reshuffles=reshuffles)
        return block, {p["key"]: p for p in block["profiles"]}

    def per_pattern(self):
        loser = {"trades": 100, "net_avg_r": -0.1, "excess_vs_index_pct": -0.5,
                 "excess_vs_universe_pct": -0.5}
        return {"version": 2, "generated": "2026-09-25T12:00:00",
                "markets": {"us": {"per_pattern": {"bull_flag": loser},
                                   "live_config": {"disabled": ["bull_flag"]}}}}

    def test_enabled_only_uses_the_live_configuration_stream(self):
        trades = [ptrade(pattern="bull_flag", ticker="A"), ptrade(pattern="cup", ticker="B",
                                                                  entry_date="2020-01-06")]
        live = [ptrade(pattern="cup", ticker="B", entry_date="2020-01-06"),
                ptrade(pattern="cup", ticker="C", entry_date="2020-01-07")]
        block, prof = self.report(trades, self.per_pattern(), live)
        e = prof["as_designed_enabled_only"]
        self.assertEqual(e["stream"], "live_config")
        self.assertEqual(e["disabled_patterns"]["us"], ["bull_flag"])
        self.assertEqual(e["trades_taken"], 2)
        self.assertIn("pick the best pattern among ALL patterns", " ".join(block["notes"]))
        self.assertIn("in-sample", e["label"])

    def test_enabled_only_falls_back_to_deleting_trades(self):
        bt_file = self.per_pattern()
        bt_file["markets"]["us"]["live_config"]["disabled"] = []       # stale replay
        trades = [ptrade(pattern="bull_flag", ticker="A"), ptrade(pattern="cup", ticker="B",
                                                                  entry_date="2020-01-06")]
        _, prof = self.report(trades, bt_file, live=trades)
        e = prof["as_designed_enabled_only"]
        self.assertEqual(e["stream"], "deleted")
        self.assertIn("approximate", e["note"])
        self.assertEqual(e["trades_excluded_by_pattern"], 1)

    def test_grade_profiles_need_grade_proxy(self):
        _, prof = self.report([ptrade(), ptrade(ticker="B", entry_date="2020-01-06")])
        for key in ("grade_a_only", "as_designed_a_in_caution"):
            self.assertIn("grade_proxy", prof[key]["error"])
        self.assertIn("error", prof["as_designed_walk_forward"])

    def test_grade_profiles_use_the_grade_replay(self):
        trades = [ptrade(pattern="bull_flag", ticker="A", grade_proxy="A"),
                  ptrade(pattern="cup", ticker="B", entry_date="2020-01-06", grade_proxy="B")]
        ga = [ptrade(pattern="cup", ticker="C", entry_date="2020-01-07", grade_proxy="A"),
              ptrade(pattern="cup", ticker="D", entry_date="2020-01-08", grade_proxy="A")]
        gc = trades + [ptrade(pattern="cup", ticker="E", entry_date="2020-01-09", grade_proxy="A")]
        (self.dir / "backtest_trades_grade_a.json").write_text(json.dumps(ga))
        (self.dir / "backtest_trades_a_in_caution.json").write_text(json.dumps(gc))
        bt_file = self.per_pattern()             # bull_flag: no edge, switched off today
        bt_file["markets"]["us"]["grade_replay"] = {"grade_a": {"disabled": ["bull_flag"]},
                                                    "a_in_caution": {"disabled": []}}
        block, prof = self.report(trades, bt_file)
        a, c = prof["grade_a_only"], prof["as_designed_a_in_caution"]
        self.assertEqual((a["stream"], c["stream"]), ("grade_replay", "grade_replay"))
        self.assertEqual(a["trades_taken"], 2)
        self.assertEqual(a["disabled_patterns"]["us"], ["bull_flag"])
        self.assertEqual(c["trades_taken"], 3)
        self.assertNotIn("approximate", a["label"] + c["label"])
        self.assertIn("grade replays", " ".join(block["notes"]))
        # made under another disabled list: grade-A only falls back to deletion
        bt_file["markets"]["us"]["grade_replay"]["grade_a"]["disabled"] = []
        block, prof = self.report(trades, bt_file)
        a, c = prof["grade_a_only"], prof["as_designed_a_in_caution"]
        self.assertEqual((a["stream"], c["stream"]), ("deleted", "grade_replay"))
        self.assertTrue(a["label"].endswith(ps.GRADE_APPROX))
        self.assertEqual(a["trades_taken"], 1)       # the headline A trade, hidden pattern and all
        self.assertIn("approximate", " ".join(block["notes"]))
        # no grade replay at all (older backtest): both approximate
        del bt_file["markets"]["us"]["grade_replay"]
        _, prof = self.report(trades, bt_file)
        for key in ("grade_a_only", "as_designed_a_in_caution"):
            self.assertEqual(prof[key]["stream"], "deleted", key)
            self.assertIn("approximate", prof[key]["label"], key)

    # --- walk-forward stream and a like-for-like hindsight effect
    def wf_setup(self):
        ts = WalkForwardTest().stream()
        # p: winner 2018-19, loser 2020-21 -> no edge in-sample; q: too few trades
        loser = {"trades": 240, "net_avg_r": -0.3, "excess_vs_index_pct": -0.6,
                 "excess_vs_universe_pct": -0.6}
        span = ("2018", "2021")
        wf, _ = ps.walk_forward_disabled(ts, TUNING, span=span)
        bt_file = {"version": 2, "generated": "2026-09-25T12:00:00",
                   "markets": {"us": {"per_pattern": {"p": loser},
                                      "live_config": {"disabled": ["p"]},
                                      "walk_forward": {"span": list(span),
                                                       "disabled_by_year": wf["us"]}}}}
        live = [t for t in ts if t["pattern"] != "p"] + [
            make_trade(pattern="r", ticker=f"R{k}", entry_date=f"2019-{k + 1:02d}-07",
                       signal_date=f"2019-{k + 1:02d}-06", exit_date=f"2019-{k + 1:02d}-25",
                       r_net=0.3) for k in range(6)]
        wf_stream = [t for t in ts if not (t["pattern"] == "p" and t["signal_date"] >= "2021")] + [
            make_trade(pattern="r", ticker=f"W{k}", entry_date=f"2021-{k + 1:02d}-07",
                       signal_date=f"2021-{k + 1:02d}-06", exit_date=f"2021-{k + 1:02d}-25",
                       r_net=0.4) for k in range(6)]
        return ts, bt_file, live, wf_stream

    def report_in(self, name, trades, backtest, live=None, wf_stream=None):
        d = self.dir / name
        d.mkdir()
        (d / "backtest_trades.json").write_text(json.dumps(trades))
        (d / "backtest_latest.json").write_text(json.dumps(backtest))
        if live is not None:
            (d / "backtest_trades_live.json").write_text(json.dumps(live))
        if wf_stream is not None:
            (d / "backtest_trades_walkforward.json").write_text(json.dumps(wf_stream))
        block = ps.run_portfolio_report(progress=lambda *_: None, trades_path=d / "backtest_trades.json",
                                        backtest_path=d / "backtest_latest.json",
                                        out_path=d / "pf.json", prices_path=d / "none.sqlite3",
                                        universe_dir=d / "nouni", config={"pattern_tuning": TUNING},
                                        reshuffles=2)
        return block, {p["key"]: p for p in block["profiles"]}

    def test_walk_forward_uses_the_backtests_replay(self):
        ts, bt_file, live, wf_stream = self.wf_setup()
        block, prof = self.report_in("wf_ok", ts, bt_file, live, wf_stream)
        w = prof["as_designed_walk_forward"]
        self.assertEqual(w["stream"], "walk_forward_replay")
        self.assertEqual(w["trades_available"], len(wf_stream))
        self.assertNotIn("approximate", w["label"])
        self.assertEqual(block["hindsight_basis"], "re-selection")
        self.assertIsNone(block["selection_effect_pp"])
        # a replay made under other yearly lists is not used
        stale = json.loads(json.dumps(bt_file))
        stale["markets"]["us"]["walk_forward"]["disabled_by_year"]["2021"] = []
        block, prof = self.report_in("wf_stale", ts, stale, live, wf_stream)
        w = prof["as_designed_walk_forward"]
        self.assertEqual(w["stream"], "deleted")
        self.assertIn("approximate", w["label"])
        self.assertIn("deleted", w["note"])
        self.assertEqual(w["trades_excluded_by_pattern"],
                         sum(1 for t in ts if t["pattern"] == "p" and t["signal_date"] >= "2021"))

    def test_hindsight_effect_is_like_for_like(self):
        ts, bt_file, live, _ = self.wf_setup()
        # in-sample re-selected, walk-forward deleted (no replay file)
        a_block, a = self.report_in("mixed", ts, bt_file, live=live)
        self.assertEqual(a["as_designed_enabled_only"]["stream"], "live_config")
        self.assertEqual(a["as_designed_walk_forward"]["stream"], "deleted")
        self.assertEqual(a_block["hindsight_basis"], "deletion")
        # both deleted (no live file either): the same hindsight effect
        b_block, b = self.report_in("both_deleted", ts, bt_file)
        self.assertEqual(b["as_designed_enabled_only"]["stream"], "deleted")
        self.assertEqual(b_block["hindsight_basis"], "deletion")
        self.assertEqual(a_block["hindsight_effect_pp"], b_block["hindsight_effect_pp"])
        self.assertEqual(a_block["hindsight_effect_pp"],
                         round(b["as_designed_enabled_only"]["cagr_pct"]
                               - b["as_designed_walk_forward"]["cagr_pct"], 1))
        # the selection method's own effect is reported apart
        self.assertAlmostEqual(a_block["selection_effect_pp"],
                               a["as_designed_enabled_only"]["cagr_pct"]
                               - b["as_designed_enabled_only"]["cagr_pct"], delta=0.11)
        self.assertEqual(a["as_designed_enabled_only"]["hindsight_basis"], "deletion")
        self.assertTrue(any("like for like" in n for n in a_block["notes"]))

    def test_walk_forward_profile_shares_the_window(self):
        ts = WalkForwardTest().stream()
        block, prof = self.report(ts)
        wf, ad = prof["as_designed_walk_forward"], prof["as_designed"]
        self.assertNotIn("error", wf)
        self.assertEqual(wf["years"], ad["years"])
        self.assertEqual(wf["disabled_by_year"]["us"]["2020"], [])
        self.assertEqual(block["walk_forward"]["disabled_by_year"], wf["disabled_by_year"])
        self.assertIsNotNone(block["hindsight_effect_pp"])
        json.loads((self.dir / "pf.json").read_text())


class EmbedAndReferenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_never_embeds_into_a_v1_file(self):
        p = self.dir / "backtest_latest.json"
        v1 = json.dumps({"generated": "2026-08-18T20:29:00", "markets": {"us": {}}})
        p.write_text(v1)
        msgs: list[str] = []
        self.assertFalse(ps._embed_in_backtest(p, {"profiles": []}, msgs.append))
        self.assertEqual(p.read_text(), v1)
        self.assertIn("v1", msgs[0])
        p.write_text(json.dumps({"version": 2, "markets": {"us": {}}}))
        self.assertTrue(ps._embed_in_backtest(p, {"profiles": []}, msgs.append))
        self.assertEqual(json.loads(p.read_text())["portfolio"], {"profiles": []})

    def test_reference_is_replayed_like_the_profiles(self):
        old = busy_stream(seed=5, days=40, per_day=8)
        for t in old:
            t.pop("r_net")                               # pre-fees stream: gross r only
        path = self.dir / "old.json"
        path.write_text(json.dumps(old))
        ref = ps._reference_pre_costs(path, {}, lambda *_: None, reshuffles=10)
        win = ps.stream_window(old)
        want = ps.run_profile(old, {"risk_pct": 1.0, "regime": {}, "window": win,
                                    "max_position_pct": ps.DEFAULT_MAX_POSITION_PCT,
                                    "rank_by": ps.rank_field(old)}, None, 10)
        self.assertNotEqual(want["cagr_pct"], want["seed0_cagr_pct"])  # the test needs them apart
        self.assertEqual(ref["cagr_pct"], want["cagr_pct"])            # the median, not seed 0
        self.assertEqual(ref["seed0_cagr_pct"], want["seed0_cagr_pct"])
        self.assertEqual(ref["representative_tie_seed"], want["representative_tie_seed"])
        self.assertEqual(ref["runs"], 11)
        self.assertEqual(ref["tie_break_cagr_pct"], want["tie_break_sensitivity"]["cagr_pct"])
        self.assertIn("closed trades", ref["drawdown_basis"])
        note = ps._reference_note(ref)
        self.assertIn(f"{ref['cagr_pct']}%/yr", note)
        self.assertNotIn("above every reshuffled run", note)
        self.assertIn("not available", ps._reference_note(None))


if __name__ == "__main__":
    unittest.main()
