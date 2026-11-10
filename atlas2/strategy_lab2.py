"""Strategy lab 2 (pre-registered 2026-11-03, before any result was seen).

Follows strategy_lab.py (2026-10-26), whose findings were: no simple rule reaches
40% a year reliably; 2x Nasdaq with a 200-day filter roughly matched holding the
Nasdaq after German tax with a much deeper fall; the 83%/yr momentum figure was
not evidence because the stock list was TODAY's index members. Three follow-ups,
chosen by the owner, every rule below fixed before running:

FAMILY P - momentum on the stocks that were REALLY in the S&P 500 at the time
  Membership: data/reference/sp500_ticker_start_end.csv (fja05680/sp500, MIT),
    stints since 1996-01-02. Prices: data/cache/lab_sp500.sqlite3 (Yahoo,
    adjusted; see lab_data.py), repaired with backtest.sanitize_bars (no
    overrides). A stint whose Yahoo series starts more than 31 days after the
    stint starts is DROPPED: the ticker may now belong to another company.
  Eligible on a signal day: a member that day (stint start <= day <= end), a
    bar that day, at least 273 earlier bars, no unexplained price jump in the
    look-back. No traded-value floor (index membership already implies a
    liquid stock; a nominal dollar floor would shut out the 1990s).
  Rules, grid, costs and engine: exactly strategy_lab's Strategy A (momentum
    126 / 252 bars skipping 21, top 10 / 20, ^GSPC 200-day filter off / on,
    monthly signal, trade at the next open, 0.15% + 0.05% per side).
  TRAIN = first signal .. 2012-12-31, TEST = 2013-01-01 .. last full month.
  Winner = highest TRAIN CAGR among variants with TRAIN max drawdown no worse
    than -50% (strategy_lab's momentum rule).
  The reuse rule cannot catch every recycled ticker (the independent check
    found EP, whose Yahoo series is another company's from one day after El
    Paso's stint began; the price repairs cut it to 2020, so it never became
    eligible); any that slip through add noise, not a known direction.
  Diagnostic: the same grid on a survivorship-biased list (every ticker that
    is a member TODAY, eligible in every year) over the same days and data -
    the difference is the bias the first lab could not remove. Coverage: the
    share of member-days that have prices, by year, and the dropped stints.

FAMILY V - volatility-targeted leverage on the Nasdaq-100
  The strategy_lab fund model with a leverage that changes:
      r(t) = L*(r_ndx + 0.7%/252) - max(L-1,0)*(tbill + 0.5%)/252
             + max(1-L,0)*tbill/252 - expense(L)/252
    expense = 0.95%/yr when L > 1, 0.20% when 0 < L <= 1, 0 in T-bills
    (identical to strategy_lab at L = 0, 1, 2, 3).
  Leverage asked for at the close of t: min(3, target / vol(t)), vol = the
    annualised standard deviation of the last W daily ^NDX log returns;
    0 (T-bills) while the 2%-band 200-day filter is out (filter variants).
    The held leverage moves only when the new figure differs by at least
    0.25, or the filter switches. Signal at the close of t, trade at the
    close of t+1 (as strategy_lab). Cost 0.10% of equity per unit of
    leverage changed.
  Grid: target vol {20%, 30%, 40%} x W {20, 60} x filter {none, sma200_band}
    = 12 variants, from 1986-10 (strategy_lab's start). TRAIN .. 2009-12-31,
    TEST 2010-01-01 ... Winner = highest TRAIN CAGR among variants with TRAIN
    max drawdown no worse than -75% (strategy_lab's leverage rule).

FAMILY R - rotating between markets (dual momentum)
  Funds (dividend-adjusted closes): QQQ, SPY, EFA (stocks outside the US),
    IEF (7-10 year US Treasuries), GLD (gold); cash = T-bills (^IRX).
  Month-end signal, trade at the close of the next session, 0.10% per side
    of the traded value (T-bills count as a fund, so moving into or out of
    cash also pays on both sides). L-month return = adjusted close at this month end
    over the one L month ends earlier; the T-bill return over the same months
    is the 'absolute' hurdle.
    gem_spy / gem_qqq: the better of {SPY or QQQ, EFA}; if it did not beat
      T-bills, IEF instead.
    top1: the best of the five; cash if it did not beat T-bills.
    top2: the best two, half each; a half that did not beat T-bills is cash.
  L in {6, 12} months -> 8 variants. First signal = the 12th month end after
    the first month end in GLD's data (the same start for all). TRAIN .. 2015-12-31,
    TEST 2016-01-01 ... Winner = highest TRAIN CAGR among variants with TRAIN
    max drawdown no worse than -50%.

Every family: TEST and whole-history figures next to QQQ (and SPY), the 40%
target, rolling 10-year windows and a 12-month block bootstrap for the winner
(V and R, where the history is long enough), and a German tax ESTIMATE run
inside a fresh account opened at the start of TEST (TaxBook: 26.375% of each
realised gain withheld at once, losses carried forward, the 30% partial
exemption for equity funds - which shrinks losses too - as a second figure;
no allowance, no Vorabpauschale; T-bill interest taxed when the T-bills are
sold). For V that account is the way an investor would hold the rule: a 3x
fund and T-bills, traded only when the signal changes; the same account with
TQQQ's real prices is the check on the fund model. A family 'reaches the
target' only when its pre-registered winner made 40% on TEST and over its
whole history.

Added after the independent check (2026-11-10), none of it used to pick a
variant: the corrected tax bookkeeping (basis after tax, exemption on losses),
the realistic V account and its real-fund check, and the 60/40 benchmark
paying its opening trade.

Run: python -m atlas2 strategy-lab-2 [--offline] [--out PATH]
Output: data/strategy_lab2.json
"""
from __future__ import annotations

import math
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .backtest import _rnd, load_long_prices, sanitize_long, write_json_atomic
from .lab_data import MEMBERSHIP_PATH, SP500_DB_PATH, load_membership
from .strategy_lab import (
    A_COST_RATE, A_LOOKBACKS, A_TOP_N, A_FILTERS, B_DIV_YIELD, B_FINANCING_SPREAD, B_SMA,
    B_START, B_TRAIN_END, B_TRAIN_DD_FLOOR_PCT, DE_PARTIAL_EXEMPTION, DE_TAX_RATE,
    DEFAULT_SERIES_PATH, SERIES_TICKERS, TARGET_CAGR_PCT, TRADING_DAYS, _days, _last_on_or_before,
    _pct, _r2, a_label, a_name, asof_values, best_on, block_bootstrap, build_panel,
    cash_returns, concentration, ensure_series, exact, first_sim_index, index_trend_flags,
    momentum_period, momentum_signals, month_complete, month_end_signal_idx, monthly_stats,
    perf_metrics, pick_winner, rolling_windows, run_leverage, run_momentum, selection_margin,
    solve_rebalance, synthetic_fund_returns, trend_state,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = ROOT / "data" / "strategy_lab2.json"
LAB2_TICKERS = tuple(SERIES_TICKERS) + ("EFA", "IEF", "GLD")

# family P
P_TRAIN_END = "2012-12-31"
P_DD_FLOOR_PCT = -50.0
P_REUSE_DAYS = 31

# family V
V_TARGETS = (0.20, 0.30, 0.40)
V_WINDOWS = (20, 60)
V_FILTERS = ("none", "sma200_band")
V_LMAX = 3.0
V_STEP = 0.25
V_COST_PER_UNIT = 0.0010
V_EXPENSE_LEVERED = 0.0095
V_EXPENSE_1X = 0.0020
V_ACCOUNT_COST = 0.0010         # the realistic account: per side of the traded value

# family R
R_ASSETS = ("QQQ", "SPY", "EFA", "IEF", "GLD")
R_EQUITY_FUNDS = ("QQQ", "SPY", "EFA")          # partial exemption applies
R_RULES = ("gem_spy", "gem_qqq", "top1", "top2")
R_LOOKBACK_M = (6, 12)
R_COST = 0.0010
R_TRAIN_END = "2015-12-31"
R_DD_FLOOR_PCT = -50.0

PERIODS = ("train", "test", "full")


# --------------------------------------------------------------------------- shared

def years_between(dates: np.ndarray, i0: int, i1: int) -> float:
    d = _days([dates[i0], dates[i1]])
    return float(d[1] - d[0]) / 365.25


def cagr_of(growth: float, years: float) -> float | None:
    if not years > 0 or not np.isfinite(growth):
        return None
    return (growth ** (1 / years) - 1) * 100.0 if growth > 0 else -100.0


def hold_after_tax_cagr(growth: float, years: float, exemption: float = 0.0) -> float | None:
    """Buy and hold, taxed once when sold at the end."""
    if not np.isfinite(growth):
        return None
    gain = growth - 1.0
    net = growth - DE_TAX_RATE * gain * (1 - exemption) if gain > 0 else growth
    return cagr_of(net, years)


class TaxBook:
    """German withholding-tax estimate for one account. Every position has a cost
    basis; a sale realises (sold share) x (value - basis). exemption maps an
    asset to its partial exemption (30% for equity funds), which shrinks gains
    AND losses alike. `rate` of the positive total, after using up carried-
    forward losses, is withheld at once; a negative total is carried forward.
    The tax is paid out of the new positions pro rata (the target weights are
    kept), and the bases are those of the positions actually held after the
    tax - so the tax paid never comes back later as a book loss."""

    def __init__(self, n_assets: int, exemption: np.ndarray | None = None,
                 rate: float = DE_TAX_RATE):
        self.basis = np.zeros(n_assets)
        self.ex = np.zeros(n_assets) if exemption is None else np.asarray(exemption, float)
        self.rate = rate
        self.losses = 0.0
        self.paid = 0.0

    def _due(self, taxable: float) -> tuple[float, float]:
        """(tax, losses carried forward afterwards) for a realised taxable total."""
        if taxable <= 0:
            return 0.0, self.losses - taxable
        used = min(self.losses, taxable)
        return self.rate * (taxable - used), self.losses - used

    def _realised(self, before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, float]:
        held = before > 0
        sold = np.where(held, np.clip(1 - after / np.where(held, before, 1), 0, 1), 0.0)
        return sold, float((sold * (before - self.basis) * (1 - self.ex)).sum())

    def trade(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Move the positions from `before` to `after` (currency, before tax) and pay
        the tax; returns the positions held afterwards (they sum to after - tax)."""
        before = np.asarray(before, float)
        target = np.asarray(after, float)
        total = float(target.sum())
        new, tax = target, 0.0
        for _ in range(100):              # paying the tax sells a little more: iterate
            tax2 = self._due(self._realised(before, new)[1])[0]
            new = target * ((total - tax2) / total) if total > 0 else target
            if abs(tax2 - tax) <= 1e-14 * max(total, 1e-300):
                break
            tax = tax2
        sold, taxable = self._realised(before, new)
        tax, self.losses = self._due(taxable)
        self.paid += tax
        self.basis = self.basis * (1 - sold) + np.maximum(new - before, 0)
        return new

    def liquidate(self, values: np.ndarray) -> float:
        """Tax due when every position is sold at `values`."""
        taxable = float(((np.asarray(values, float) - self.basis) * (1 - self.ex)).sum())
        tax, self.losses = self._due(taxable)
        self.paid += tax
        self.basis[:] = 0
        return tax


def compare_row(m: dict, ref: dict, key: str) -> float | None:
    a, b = m.get("cagr_pct"), ref.get("cagr_pct")
    return _r2(a - b) if a is not None and b is not None else None


def winner_summary(variants: dict, pick: dict, rule: str, dd_floor: float) -> dict:
    name = pick["name"]
    best_test = best_on(variants, "test")
    return {
        "rule": rule, "name": name, "label": variants[name]["label"] if name else None,
        "qualified": pick["qualified"], "disqualified": pick["disqualified"],
        "selection_margin": selection_margin(variants, pick, dd_floor),
        "best_test_variant": {"name": best_test,
                              "label": "in-sample on test - do not act on this",
                              "test_cagr_pct": variants[best_test]["periods"]["test"]["cagr_pct"]
                              if best_test else None},
    }


def reaches_target(variants: dict, name: str | None) -> dict:
    if not name:
        return {"reaches": False, "why": "no pre-registered winner"}
    t = exact(variants[name]["periods"]["test"], "cagr_pct")
    f = exact(variants[name]["periods"]["full"], "cagr_pct")
    ok = t is not None and f is not None and t >= TARGET_CAGR_PCT and f >= TARGET_CAGR_PCT
    return {"reaches": bool(ok), "test_cagr_pct": _r2(t), "full_cagr_pct": _r2(f)}


def month_end_curve(dates: np.ndarray, eq: np.ndarray, i0: int, i1: int) -> list:
    out = []
    keys = [str(d)[:7] for d in dates]
    for i in range(i0, i1 + 1):
        if i == i1 or keys[i + 1] != keys[i]:
            if np.isfinite(eq[i]):
                out.append([str(dates[i]), round(float(eq[i] / eq[i0]), 5)])
    return out


# --------------------------------------------------------------------------- family V

def realized_vol(close: np.ndarray, window: int) -> np.ndarray:
    """Annualised std (ddof=1) of the last `window` daily log returns, known at the
    close of each day (NaN until full)."""
    close = np.asarray(close, float)
    lr = np.full(len(close), np.nan)
    lr[1:] = np.log(close[1:] / close[:-1])
    return pd.Series(lr).rolling(window).std(ddof=1).to_numpy() * math.sqrt(TRADING_DAYS)


def vol_target_signal(close: np.ndarray, target: float, window: int, filt: str,
                      lmax: float = V_LMAX, step: float = V_STEP) -> np.ndarray:
    """The leverage held after acting on each close's signal (NaN = undefined):
    min(lmax, target / vol), 0 while the trend filter is out; a change smaller
    than `step` is ignored unless the filter switched."""
    vol = realized_vol(close, window)
    st = trend_state(close, filt)
    want = np.where(np.isfinite(vol) & (vol > 0), np.minimum(lmax, target / np.where(vol > 0, vol, 1)),
                    np.nan)
    want = np.where(st == 1, want, np.where(st == 0, 0.0, np.nan))
    out = np.full(len(close), np.nan)
    cur = np.nan
    for t in range(len(close)):
        w = want[t]
        if not np.isfinite(w):
            cur = np.nan
            continue
        if not np.isfinite(cur) or (w == 0.0) != (cur == 0.0) or abs(w - cur) >= step:
            cur = w
        out[t] = cur
    return out


def levered_returns(close: np.ndarray, tbill_pct: np.ndarray, lev: np.ndarray, *,
                    div_yield: float = B_DIV_YIELD, spread: float = B_FINANCING_SPREAD
                    ) -> np.ndarray:
    """Daily return at leverage lev[t] (the leverage held over day t)."""
    close = np.asarray(close, float)
    tb = np.asarray(tbill_pct, float) / 100.0
    lev = np.asarray(lev, float)
    r = np.full(len(close), np.nan)
    r[1:] = close[1:] / close[:-1] - 1
    expense = np.where(lev > 1, V_EXPENSE_LEVERED, np.where(lev > 0, V_EXPENSE_1X, 0.0))
    ret = (lev * (r + div_yield / TRADING_DAYS)
           - np.maximum(lev - 1, 0) * (tb + spread) / TRADING_DAYS
           + np.maximum(1 - lev, 0) * tb / TRADING_DAYS - expense / TRADING_DAYS)
    return np.maximum(ret, -1.0)


def run_vol_target(dates: np.ndarray, close: np.ndarray, tbill_pct: np.ndarray, target: float,
                   window: int, filt: str, *, d0: int,
                   cost_per_unit: float = V_COST_PER_UNIT) -> dict:
    """One variant of the pre-registered model (before tax). Day d holds sig[d-2]
    (signal at close d-2, traded at close d-1); a change at the close of day e
    (sig[e-1] != sig[e-2]) costs cost_per_unit x |change| of equity."""
    n = len(dates)
    sig = vol_target_signal(close, target, window, filt)
    if not np.isfinite(sig[d0 - 2:]).all():
        raise ValueError("the volatility signal is undefined inside the simulated period")
    held = sig[d0 - 2:n - 2]
    after = sig[d0 - 1:n - 1]
    ret = levered_returns(close, tbill_pct, np.concatenate([np.full(d0, np.nan), held]))[d0:]
    if not np.isfinite(ret).all():
        raise ValueError("missing index or T-bill data inside the simulated period")
    change = np.abs(after - held)
    eq = np.full(n, np.nan)
    eq[d0 - 1] = 1.0
    eq[d0:] = np.cumprod((1.0 + ret) * (1.0 - cost_per_unit * change))
    lev = np.full(n, np.nan)
    lev[d0:] = held
    trades = np.zeros(n, bool)
    trades[d0:] = change > 0
    return {"equity": eq, "lev": lev, "trades": trades, "d0": d0}


def v_name(target: float, window: int, filt: str) -> str:
    return f"vol{int(round(target * 100))}_w{window}_{filt}"


def v_label(target: float, window: int, filt: str) -> str:
    return (f"Nasdaq-100 leverage aiming at {int(round(target * 100))}% yearly swings "
            f"({window}-day volatility, at most 3x)"
            + (", in T-bills below the 200-day average (2% band)" if filt != "none" else ""))


def v_period(dates: np.ndarray, res: dict, i0: int, i1: int) -> dict:
    m = perf_metrics(dates, res["equity"], i0, i1)
    if m["years"]:
        seg = res["lev"][i0 + 1:i1 + 1]
        m["avg_leverage"] = _r2(float(np.mean(seg)))
        m["time_in_market_pct"] = _rnd(100.0 * float((seg > 0).mean()), 1)
        m["time_above_2x_pct"] = _rnd(100.0 * float((seg > 2).mean()), 1)
        m["trades_per_year"] = _rnd(float(res["trades"][i0 + 1:i1 + 1].sum()) / m["years"], 1)
    return m


def run_vol_account(fund_ret: np.ndarray, cash_ret: np.ndarray, sig: np.ndarray, i_open: int,
                    i_end: int, *, cost: float = V_ACCOUNT_COST,
                    tax_exemption: float | None = None) -> np.ndarray:
    """How an investor would run the rule: a 3x daily-reset Nasdaq-100 fund and
    T-bills. 1.0 in cash at the close of i_open, bought at once into L/3 fund
    and the rest T-bills (L = sig[i_open-1], the signal of the previous close),
    then traded ONLY at the closes where the signal changes (sig[d-1] !=
    sig[d-2]); in between the weights drift. Costs: `cost` per side of the
    traded value (T-bills count as a fund). With tax_exemption set (the fund's
    partial exemption; T-bills get none) the TaxBook estimate is paid at every
    trade and on selling everything at i_end. Returns the equity (1.0 at i_open,
    NaN outside i_open..i_end)."""
    n = len(fund_ret)
    if not (np.isfinite(fund_ret[i_open + 1:i_end + 1]).all()
            and np.isfinite(cash_ret[i_open + 1:i_end + 1]).all()
            and np.isfinite(sig[i_open - 1:i_end]).all()):
        raise ValueError("missing fund, T-bill or signal data inside the account's period")
    book = None
    if tax_exemption is not None:
        book = TaxBook(2, np.array([float(tax_exemption), 0.0]))
        book.basis[1] = 1.0                       # the opening cash is not a gain

    def trade(h: np.ndarray, lev: float) -> np.ndarray:
        w = np.array([lev / V_LMAX, 1.0 - lev / V_LMAX])
        after, _ = solve_rebalance(float(h.sum()), h, w, cost)
        return book.trade(h, w * after) if book is not None else w * after

    eq = np.full(n, np.nan)
    eq[i_open] = 1.0
    h = trade(np.array([0.0, 1.0]), float(sig[i_open - 1]))
    for d in range(i_open + 1, i_end + 1):
        h = h * (1.0 + np.array([fund_ret[d], cash_ret[d]]))
        if sig[d - 1] != sig[d - 2]:
            h = trade(h, float(sig[d - 1]))
        eq[d] = float(h.sum())
    if book is not None:
        eq[i_end] -= book.liquidate(h)
    return eq


def vol_block(frames: dict, *, n_paths: int = 10000, progress=print) -> dict:
    ndx = frames["^NDX"]
    dates = np.asarray(ndx.index, dtype=str)
    close = ndx["close"].to_numpy(float)
    irx = frames["^IRX"]
    tb = asof_values(np.asarray(irx.index, dtype=str), irx["close"].to_numpy(float), dates)
    d0 = first_sim_index(dates, close, B_START, B_SMA)
    n = len(dates)
    i_end = n - 1
    i_split = _last_on_or_before(dates, B_TRAIN_END)
    spans = {"train": (d0 - 1, i_split), "test": (i_split, i_end), "full": (d0 - 1, i_end)}
    qqq = frames["QQQ"]
    q = asof_values(np.asarray(qqq.index, dtype=str), qqq["adj_close"].to_numpy(float), dates)
    ref1x = run_leverage(dates, close, tb, 1, "none", d0=d0)
    ref2x = run_leverage(dates, close, tb, 2, "sma200_band", d0=d0)
    bench = {p: {"nasdaq_1x_model": perf_metrics(dates, ref1x["equity"], *spans[p]),
                 "lab1_winner_2x_band": perf_metrics(dates, ref2x["equity"], *spans[p])}
             for p in PERIODS}
    bench["test"]["QQQ"] = perf_metrics(dates, q, *spans["test"])
    progress(f"[V] ^NDX {dates[d0]} .. {dates[i_end]}; train to {dates[i_split]}")

    variants, runs = {}, {}
    for target in V_TARGETS:
        for window in V_WINDOWS:
            for filt in V_FILTERS:
                name = v_name(target, window, filt)
                res = run_vol_target(dates, close, tb, target, window, filt, d0=d0)
                runs[name] = res
                periods = {p: v_period(dates, res, *spans[p]) for p in PERIODS}
                for p in PERIODS:
                    periods[p]["cagr_minus_nasdaq_1x_pp"] = compare_row(
                        periods[p], bench[p]["nasdaq_1x_model"], "cagr_pct")
                periods["test"]["cagr_minus_qqq_pp"] = compare_row(
                    periods["test"], bench["test"]["QQQ"], "cagr_pct")
                variants[name] = {"target_vol_pct": int(round(target * 100)), "vol_window": window,
                                  "filter": filt, "label": v_label(target, window, filt),
                                  "periods": periods}
    pick = pick_winner(variants, B_TRAIN_DD_FLOOR_PCT)
    winner = winner_summary(
        variants, pick, f"highest TRAIN CAGR (1986-10 to {B_TRAIN_END}) among variants whose "
        f"TRAIN max drawdown is no worse than {B_TRAIN_DD_FLOOR_PCT:.0f}%", B_TRAIN_DD_FLOOR_PCT)
    wname = pick["name"]
    curves = {}
    if wname:
        wv = variants[wname]
        args = (wv["target_vol_pct"] / 100.0, wv["vol_window"], wv["filter"])
        res = runs[wname]
        ms = monthly_stats(dates, res["equity"], d0 - 1, i_end)
        winner["rolling_10y"] = rolling_windows(dates, res["equity"], ms, ref1x["equity"])
        winner["bootstrap"] = block_bootstrap(ms, n_paths=n_paths)
        sig_w = vol_target_signal(close, *args)
        fund3 = synthetic_fund_returns(close, tb, 3)
        cash_r = cash_returns(tb)

        def account(i0: int, ex: float | None, fund: np.ndarray = fund3) -> np.ndarray:
            return run_vol_account(fund, cash_r, sig_w, i0, i_end, tax_exemption=ex)

        i_t = spans["test"][0]
        yrs = years_between(dates, i_t, i_end)
        g_q = q[i_end] / q[i_t]
        acc = {key: account(i_t, ex) for key, ex in (
            ("pre_tax", None), ("after_tax", 0.0), ("after_tax_ex", DE_PARTIAL_EXEMPTION))}
        winner["account_test_de"] = {
            "from": str(dates[i_t]),
            "model_cagr_pct": variants[wname]["periods"]["test"]["cagr_pct"],
            "account_pre_tax_cagr_pct": _r2(cagr_of(acc["pre_tax"][i_end], yrs)),
            "account_pre_tax_max_drawdown_pct": perf_metrics(
                dates, acc["pre_tax"], i_t, i_end)["max_drawdown_pct"],
            "account_after_tax_cagr_pct": _r2(cagr_of(acc["after_tax"][i_end], yrs)),
            "account_after_tax_with_30pct_partial_exemption_pct": _r2(
                cagr_of(acc["after_tax_ex"][i_end], yrs)),
            "qqq_held_after_tax_cagr_pct": _r2(hold_after_tax_cagr(g_q, yrs)),
            "qqq_held_after_tax_with_30pct_partial_exemption_pct": _r2(
                hold_after_tax_cagr(g_q, yrs, DE_PARTIAL_EXEMPTION)),
            "note": "a fresh account at the start of TEST holding a 3x fund (the lab's fund "
                    "model) and T-bills, traded only when the signal changes, 0.10% per side; "
                    "tax estimate: 26.375% of each realised gain withheld at once, losses "
                    "carried forward, everything sold and taxed at the end. Whether a "
                    "leveraged product gets the 30% partial exemption depends on how it is "
                    "built (a swap-based or note-type product may not), so both figures are "
                    "shown; QQQ held is taxed once at the end",
        }
        # real-fund check: TQQQ's own daily prices in place of the modelled 3x fund
        tq = frames.get("TQQQ")
        if tq is not None:
            tqq = asof_values(np.asarray(tq.index, dtype=str), tq["adj_close"].to_numpy(float),
                              dates)
            ok = np.nonzero(np.isfinite(tqq))[0]
            i_r = int(ok[0])
            real_ret = np.full(len(dates), np.nan)
            real_ret[1:] = tqq[1:] / tqq[:-1] - 1
            real = account(i_r, None, real_ret)
            modelled = account(i_r, None)

            def row(eq: np.ndarray) -> dict:
                m = perf_metrics(dates, eq, i_r, i_end)
                return {"cagr_pct": m["cagr_pct"], "max_drawdown_pct": m["max_drawdown_pct"]}

            winner["real_fund_check"] = {
                "from": str(dates[i_r]), "to": str(dates[i_end]),
                "account_with_real_tqqq": row(real),
                "account_with_modelled_3x_fund": row(modelled),
                "headline_model": row(res["equity"]),
                "qqq": row(q),
                "note": "the same signal, trades and costs; only the 3x fund's daily returns "
                        "differ (TQQQ's real, dividend-adjusted prices vs the lab's model)",
            }
        curves[wname] = month_end_curve(dates, res["equity"], d0 - 1, i_end)
        curves["nasdaq_1x_model"] = month_end_curve(dates, ref1x["equity"], d0 - 1, i_end)
    return {"variants": variants, "winner": winner, "benchmarks": bench,
            "reaches_target": reaches_target(variants, wname), "curves": curves,
            "span": {k: [str(dates[a]), str(dates[b])] for k, (a, b) in spans.items()}}


# --------------------------------------------------------------------------- family R

def r_name(rule: str, months: int) -> str:
    return f"{rule}_{months}m"


def r_label(rule: str, months: int) -> str:
    return {
        "gem_spy": "the stronger of US stocks (SPY) and stocks outside the US (EFA); bonds "
                   "(IEF) when neither beat T-bills",
        "gem_qqq": "the stronger of the Nasdaq-100 (QQQ) and stocks outside the US (EFA); "
                   "bonds (IEF) when neither beat T-bills",
        "top1": "the strongest of QQQ / SPY / EFA / IEF / GLD; T-bills when it did not beat "
                "T-bills",
        "top2": "the two strongest of QQQ / SPY / EFA / IEF / GLD, half each; a half goes to "
                "T-bills when its fund did not beat T-bills",
    }[rule] + f" (strength = {months}-month return)"


def r_weights(rule: str, ret: dict, hurdle: float) -> dict:
    """Target weights {asset or 'CASH': w} for one signal."""
    if rule in ("gem_spy", "gem_qqq"):
        us = "SPY" if rule == "gem_spy" else "QQQ"
        best = us if ret[us] >= ret["EFA"] else "EFA"
        return {best: 1.0} if ret[best] > hurdle else {"IEF": 1.0}
    ranked = sorted(R_ASSETS, key=lambda a: (-ret[a], R_ASSETS.index(a)))
    k = 1 if rule == "top1" else 2
    w: dict = {}
    for a in ranked[:k]:
        key = a if ret[a] > hurdle else "CASH"
        w[key] = w.get(key, 0.0) + 1.0 / k
    return w


def run_rotation(dates: np.ndarray, px: np.ndarray, cash_ret: np.ndarray, sig_idx: np.ndarray,
                 me_pos: dict, rule: str, months: int, i_start: int, i_end: int, *,
                 tax_exemption: np.ndarray | None = None, cost: float = R_COST) -> dict:
    """Assets = R_ASSETS columns of px + cash. Start in cash at the close of
    i_start (a signal day); each signal close s is traded at the close of s+1."""
    n, k = px.shape
    R = np.zeros((n, k + 1))
    R[1:, :k] = px[1:] / px[:-1] - 1
    R[:, k] = cash_ret
    sig_set = {int(s) for s in sig_idx}
    h = np.zeros(k + 1)
    h[k] = 1.0
    eq = np.full(n, np.nan)
    eq[i_start] = 1.0
    book = TaxBook(k + 1, tax_exemption) if tax_exemption is not None else None
    if book is not None:
        book.basis[k] = 1.0                 # the opening cash is not a gain
    log = []
    for d in range(i_start + 1, i_end + 1):
        h = h * (1.0 + R[d])
        s = d - 1
        if s in sig_set and s >= i_start:
            a0 = me_pos["idx"][me_pos[s] - months]          # the month end L months earlier
            ret = {a: float(px[s, c] / px[a0, c] - 1) for c, a in enumerate(R_ASSETS)}
            hurdle = float(np.prod(1.0 + cash_ret[a0 + 1:s + 1]) - 1)
            wd = r_weights(rule, ret, hurdle)
            w = np.zeros(k + 1)
            for a, x in wd.items():
                w[k if a == "CASH" else R_ASSETS.index(a)] += x
            value = float(h.sum())
            after, traded = solve_rebalance(value, h, w, cost)
            new = book.trade(h, w * after) if book is not None else w * after
            changed = bool(np.abs(new - h).sum() > 1e-9 * value)
            h = new
            log.append({"signal": str(dates[s]), "trade_idx": d,
                        "hold": {a: round(x, 3) for a, x in wd.items()},
                        "turnover_pct": 100.0 * traded / 2 / value if value > 0 else 0.0,
                        "changed": changed})
        eq[d] = float(h.sum())
    if book is not None:
        eq[i_end] -= book.liquidate(h)
    return {"equity": eq, "log": log}


def r_period(dates: np.ndarray, res: dict, i0: int, i1: int) -> dict:
    m = perf_metrics(dates, res["equity"], i0, i1)
    rows = [r for r in res["log"] if i0 < r["trade_idx"] <= i1]
    if m["years"] and rows:
        m["switch_months_per_year"] = _rnd(
            sum(1 for r in rows if r["turnover_pct"] > 1.0) / m["years"], 1)
        share = {}
        for r in rows:
            for a, x in r["hold"].items():
                share[a] = share.get(a, 0.0) + x
        m["time_share_pct"] = {a: _rnd(100.0 * v / len(rows), 1) for a, v in sorted(share.items())}
    return m


def rotation_block(frames: dict, *, n_paths: int = 10000, progress=print) -> dict:
    spy = frames["SPY"]
    dates_all = np.asarray(spy.index, dtype=str)
    first = max(str(frames[a].index[0]) for a in R_ASSETS)
    last = min(str(frames[a].index[-1]) for a in R_ASSETS + ("^IRX",))
    dates = dates_all[(dates_all >= first) & (dates_all <= last)]
    px = np.column_stack([asof_values(np.asarray(frames[a].index, dtype=str),
                                      frames[a]["adj_close"].to_numpy(float), dates)
                          for a in R_ASSETS])
    irx = frames["^IRX"]
    tb = asof_values(np.asarray(irx.index, dtype=str), irx["close"].to_numpy(float), dates)
    if not (np.isfinite(px).all() and np.isfinite(tb).all()):
        raise ValueError("gaps in the fund or T-bill series")
    cash = cash_returns(tb)
    me = month_end_signal_idx(dates)
    n = len(dates)
    me_all = [int(i) for i in me] + ([n - 1] if month_complete(dates, n - 1) else [])
    me_pos = {int(s): j for j, s in enumerate(me)}
    me_pos["idx"] = me_all
    j0 = max(R_LOOKBACK_M)          # the 12-month look-back starts at the first month end
    i_start = int(me[j0])
    i_end = max(i for i in me_all if i > i_start)
    i_split = _last_on_or_before(dates, R_TRAIN_END)
    spans = {"train": (i_start, i_split), "test": (i_split, i_end), "full": (i_start, i_end)}
    sig = me[j0:]
    progress(f"[R] funds {dates[0]} .. {dates[-1]}; first signal {dates[i_start]}, "
             f"train to {dates[i_split]}")
    bench_eq = {a: px[:, R_ASSETS.index(a)] / px[i_start, R_ASSETS.index(a)]
                for a in ("QQQ", "SPY")}
    # 60/40 SPY/IEF, rebalanced each month with the same costs
    R6 = np.zeros((n, 2))
    R6[1:] = np.column_stack([px[1:, 1] / px[:-1, 1] - 1, px[1:, 3] / px[:-1, 3] - 1])
    h = np.array([0.6, 0.4]) * solve_rebalance(1.0, np.zeros(2), np.array([0.6, 0.4]), R_COST)[0]
    e6 = np.full(n, np.nan)
    e6[i_start] = 1.0
    sig_set = {int(s) for s in sig}
    for d in range(i_start + 1, i_end + 1):
        h = h * (1 + R6[d])
        if d - 1 in sig_set and d - 1 > i_start:
            v = float(h.sum())
            after, _ = solve_rebalance(v, h, np.array([0.6, 0.4]), R_COST)
            h = np.array([0.6, 0.4]) * after
        e6[d] = float(h.sum())
    bench_eq["60_40_spy_ief"] = e6
    bench = {p: {k: perf_metrics(dates, e, *spans[p]) for k, e in bench_eq.items()}
             for p in PERIODS}

    variants, runs = {}, {}
    for rule in R_RULES:
        for months in R_LOOKBACK_M:
            name = r_name(rule, months)
            res = run_rotation(dates, px, cash, sig, me_pos, rule, months, i_start, i_end)
            runs[name] = res
            periods = {p: r_period(dates, res, *spans[p]) for p in PERIODS}
            for p in PERIODS:
                periods[p]["cagr_minus_qqq_pp"] = compare_row(periods[p], bench[p]["QQQ"], "")
                periods[p]["cagr_minus_spy_pp"] = compare_row(periods[p], bench[p]["SPY"], "")
            variants[name] = {"rule": rule, "lookback_months": months,
                              "label": r_label(rule, months), "periods": periods}
    pick = pick_winner(variants, R_DD_FLOOR_PCT)
    winner = winner_summary(
        variants, pick, f"highest TRAIN CAGR (to {R_TRAIN_END}) among variants whose TRAIN max "
        f"drawdown is no worse than {R_DD_FLOOR_PCT:.0f}%", R_DD_FLOOR_PCT)
    wname = pick["name"]
    curves = {}
    if wname:
        wv = variants[wname]
        res = runs[wname]
        ms = monthly_stats(dates, res["equity"], i_start, i_end)
        winner["rolling_10y"] = rolling_windows(dates, res["equity"], ms, bench_eq["QQQ"])
        winner["bootstrap"] = block_bootstrap(ms, n_paths=n_paths)
        i_t = spans["test"][0]
        ex = np.array([DE_PARTIAL_EXEMPTION if a in R_EQUITY_FUNDS else 0.0 for a in R_ASSETS]
                      + [0.0])
        sig_t = sig[sig >= i_t]
        taxed = {}
        for key, e in (("cagr_pct", np.zeros(len(R_ASSETS) + 1)),
                       ("cagr_with_30pct_partial_exemption_pct", ex)):
            r_t = run_rotation(dates, px, cash, sig_t, me_pos, wv["rule"], wv["lookback_months"],
                               int(sig_t[0]), i_end, tax_exemption=e)
            taxed[key] = _r2(cagr_of(r_t["equity"][i_end] / r_t["equity"][int(sig_t[0])],
                                     years_between(dates, int(sig_t[0]), i_end)))
        g_q = px[i_end, 0] / px[int(sig_t[0]), 0]
        yrs = years_between(dates, int(sig_t[0]), i_end)
        taxed.update({
            "from": str(dates[int(sig_t[0])]),
            "qqq_held_cagr_pct": _r2(hold_after_tax_cagr(g_q, yrs)),
            "qqq_held_with_30pct_partial_exemption_pct": _r2(
                hold_after_tax_cagr(g_q, yrs, DE_PARTIAL_EXEMPTION)),
            "note": "estimate: 26.375% of each realised gain withheld at once, losses carried "
                    "forward, 30% exemption (on gains and losses) only for the stock funds; "
                    "gold treated as fully taxable; T-bill interest taxed when the T-bills are "
                    "sold"})
        winner["after_tax_test_de"] = taxed
        curves[wname] = month_end_curve(dates, res["equity"], i_start, i_end)
        curves["QQQ"] = month_end_curve(dates, bench_eq["QQQ"], i_start, i_end)
    return {"variants": variants, "winner": winner, "benchmarks": bench,
            "reaches_target": reaches_target(variants, wname), "curves": curves,
            "span": {k: [str(dates[a]), str(dates[b])] for k, (a, b) in spans.items()}}


# --------------------------------------------------------------------------- family P

def member_matrix(stints: list[dict], dates: np.ndarray, tickers: list[str]) -> np.ndarray:
    """member[d, k]: tickers[k] was in the index on dates[d] (by its stints)."""
    pos = {t: k for k, t in enumerate(tickers)}
    m = np.zeros((len(dates), len(tickers)), bool)
    for s in stints:
        k = pos.get(s["ticker"])
        if k is None:
            continue
        a = int(np.searchsorted(dates, s["start"], side="left"))
        b = len(dates) if s["end"] is None else int(np.searchsorted(dates, s["end"], side="right"))
        m[a:b, k] = True
    return m


def usable_stints(stints: list[dict], first_bar: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """(kept, no_prices, dropped_reuse). A stint is dropped when the ticker's
    Yahoo series starts more than P_REUSE_DAYS after the stint started."""
    kept, none, reuse = [], [], []
    for s in stints:
        f = first_bar.get(s["ticker"])
        if f is None:
            none.append(s)
        elif (date.fromisoformat(f) - date.fromisoformat(s["start"])).days > P_REUSE_DAYS:
            reuse.append(dict(s, yahoo_first_bar=f))
        else:
            kept.append(s)
    return kept, none, reuse


def coverage_by_year(stints: list[dict], kept: list[dict], dates: np.ndarray) -> dict:
    """Share of member-days (business days of the index calendar) with a usable stint."""
    years = np.array([str(d)[:4] for d in dates])
    allm = np.zeros(len(dates))
    okm = np.zeros(len(dates))
    for s in stints:
        a = int(np.searchsorted(dates, s["start"], side="left"))
        b = len(dates) if s["end"] is None else int(np.searchsorted(dates, s["end"], side="right"))
        allm[a:b] += 1
    for s in kept:
        a = int(np.searchsorted(dates, s["start"], side="left"))
        b = len(dates) if s["end"] is None else int(np.searchsorted(dates, s["end"], side="right"))
        okm[a:b] += 1
    out = {}
    for y in sorted(set(years.tolist())):
        sel = years == y
        if allm[sel].sum() > 0:
            out[y] = _rnd(100.0 * okm[sel].sum() / allm[sel].sum(), 1)
    return out


def load_member_panel(db_path: Path, progress=print):
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        first_bar = {t: f for t, f in conn.execute(
            "SELECT ticker, first_date FROM fetch_log WHERE n_rows > 0")}
        n_log = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
        long = load_long_prices(SimpleNamespace(conn=conn), sorted(first_bar))
    finally:
        conn.close()
    long, rep = sanitize_long(long, (), {})
    panel = build_panel(long, rep.get("suspect"))
    return panel, first_bar, n_log, rep


def pit_block(frames: dict, *, db_path: Path = SP500_DB_PATH,
              membership_path: Path = MEMBERSHIP_PATH, progress=print) -> dict:
    stints = load_membership(membership_path)
    panel, first_bar, n_log, rep = load_member_panel(db_path, progress)
    kept, none, reuse = usable_stints(stints, first_bar)
    dates = panel.dates
    n = len(dates)
    member = member_matrix(kept, dates, panel.tickers)
    today_members = {s["ticker"] for s in stints if s["end"] is None}
    signal_idx = month_end_signal_idx(dates)
    sig_all = momentum_signals(panel, signal_idx, min_dollar_vol=0.0)
    sig_pit = SimpleNamespace(idx=sig_all.idx, mom=sig_all.mom,
                              eligible=sig_all.eligible & member[signal_idx])
    biased_cols = np.array([t in today_members for t in panel.tickers])
    sig_bias = SimpleNamespace(idx=sig_all.idx, mom=sig_all.mom,
                               eligible=sig_all.eligible & biased_cols[None, :])

    g = frames["^GSPC"]
    trend_ok, g_def = index_trend_flags(np.asarray(g.index, dtype=str), g["close"].to_numpy(float),
                                        dates[signal_idx])
    irx = frames["^IRX"]
    tb = asof_values(np.asarray(irx.index, dtype=str), irx["close"].to_numpy(float), dates)
    rf = tb / 100.0 / TRADING_DAYS
    strict = sig_pit.eligible & np.isfinite(sig_pit.mom[max(A_LOOKBACKS)])
    usable = (strict.sum(axis=1) >= max(A_TOP_N)) & g_def & np.isfinite(tb[signal_idx]) \
        & (dates[signal_idx] >= "1996-01-02")
    j0 = int(np.nonzero(usable)[0][0])
    i_start = int(signal_idx[j0])
    limit = min(str(frames[t].index[-1]) for t in ("SPY", "^IRX", "^GSPC", "^NDX"))
    ends = [int(i) for i in signal_idx] + ([n - 1] if month_complete(dates, n - 1) else [])
    i_end = max(i for i in ends if str(dates[i]) <= limit and i > i_start)
    i_split = min(max(_last_on_or_before(dates, P_TRAIN_END), i_start), i_end)
    spans = {"train": (i_start, i_split), "test": (i_split, i_end), "full": (i_start, i_end)}
    progress(f"[P] {len(panel.tickers)} tickers with prices, {dates[0]} .. {dates[-1]}; first "
             f"signal {dates[i_start]} ({int(strict[j0].sum())} eligible members); train to "
             f"{dates[i_split]}, test to {dates[i_end]}")

    def ser(t, col):
        f = frames[t]
        return asof_values(np.asarray(f.index, dtype=str), f[col].to_numpy(float), dates)

    spy, qqq = ser("SPY", "adj_close"), ser("QQQ", "adj_close")
    ndx = frames["^NDX"]
    nd = np.asarray(ndx.index, dtype=str)
    ndx_tb = asof_values(np.asarray(irx.index, dtype=str), irx["close"].to_numpy(float), nd)
    ref1x = run_leverage(nd, ndx["close"].to_numpy(float), ndx_tb, 1, "none",
                         d0=first_sim_index(nd, ndx["close"].to_numpy(float), B_START, B_SMA))
    n1 = asof_values(nd, ref1x["equity"], dates)
    ew = run_momentum(panel, sig_pit, lookback=max(A_LOOKBACKS), top_n=None, rf_daily=rf,
                      cost_rate=0.0, first_signal=j0, i_end=i_end)
    bench_eq = {"SPY": spy / spy[i_start], "nasdaq_1x_model": n1 / n1[i_start],
                "equal_weight_members_monthly": ew["equity"]}
    bench = {p: {k: perf_metrics(dates, e, *spans[p]) for k, e in bench_eq.items()}
             for p in PERIODS}
    bench["test"]["QQQ"] = perf_metrics(dates, qqq, *spans["test"])

    def grid(sig) -> tuple[dict, dict]:
        out, runs = {}, {}
        for lookback in A_LOOKBACKS:
            for top_n in A_TOP_N:
                for trend in A_FILTERS:
                    name = a_name(lookback, top_n, trend)
                    res = run_momentum(panel, sig, lookback=lookback, top_n=top_n,
                                       trend_ok=trend_ok if trend else None, rf_daily=rf,
                                       first_signal=j0, i_end=i_end)
                    runs[name] = res
                    periods = {p: momentum_period(dates, res, *spans[p]) for p in PERIODS}
                    for p in PERIODS:
                        for key, bk in (("cagr_minus_spy_pp", "SPY"),
                                        ("cagr_minus_nasdaq_1x_pp", "nasdaq_1x_model"),
                                        ("cagr_minus_equal_weight_members_pp",
                                         "equal_weight_members_monthly")):
                            periods[p][key] = compare_row(periods[p], bench[p][bk], "")
                    periods["test"]["cagr_minus_qqq_pp"] = compare_row(
                        periods["test"], bench["test"]["QQQ"], "")
                    out[name] = {"lookback_bars": lookback, "top_n": top_n, "trend_filter": trend,
                                 "label": a_label(lookback, top_n, trend), "periods": periods,
                                 "fills_without_bar": res["fills_without_bar"]}
        return out, runs

    variants, runs = grid(sig_pit)
    biased, _ = grid(sig_bias)
    bias = {name: {p: _r2((biased[name]["periods"][p]["cagr_pct"] or 0)
                          - (variants[name]["periods"][p]["cagr_pct"] or 0)) for p in PERIODS}
            for name in variants}
    pick = pick_winner(variants, P_DD_FLOOR_PCT)
    winner = winner_summary(
        variants, pick, f"highest TRAIN CAGR (to {P_TRAIN_END}) among variants whose TRAIN max "
        f"drawdown is no worse than {P_DD_FLOOR_PCT:.0f}%", P_DD_FLOOR_PCT)
    wname = pick["name"]
    curves = {}
    if wname:
        winner["concentration"] = concentration(dates, runs[wname], i_start, i_end)
        curves[wname] = month_end_curve(dates, runs[wname]["equity"], i_start, i_end)
    best_train = best_on(variants, "train")
    if best_train and not wname:
        winner["best_train_without_drawdown_limit"] = {
            "name": best_train, "label": "diagnostic - fails the pre-registered drawdown limit",
            "periods": {p: {k: variants[best_train]["periods"][p].get(k)
                            for k in ("cagr_pct", "max_drawdown_pct")} for p in PERIODS}}
    held_missing = sum(v["fills_without_bar"] for v in variants.values())
    return {
        "variants": variants, "winner": winner, "benchmarks": bench,
        "survivorship_biased_list": {
            "note": "diagnostic: the same rules on every ticker that is a member TODAY, eligible "
                    "in every year (what the first lab did); bias_pp = biased CAGR minus the "
                    "point-in-time CAGR",
            "variants": {k: {p: {"cagr_pct": v["periods"][p]["cagr_pct"],
                                 "max_drawdown_pct": v["periods"][p]["max_drawdown_pct"]}
                             for p in PERIODS} for k, v in biased.items()},
            "bias_pp": bias},
        "coverage": {
            "stints": len(stints), "stints_used": len(kept), "stints_without_prices": len(none),
            "stints_dropped_possible_ticker_reuse": len(reuse),
            "tickers_tried": n_log, "tickers_with_prices": len(first_bar),
            "member_days_with_prices_by_year_pct": coverage_by_year(stints, kept, dates),
            "examples_without_prices": sorted({s["ticker"] for s in none})[:40],
            "examples_dropped": [f"{s['ticker']} {s['start']}..{s['end'] or 'now'} (Yahoo from "
                                 f"{s['yahoo_first_bar']})" for s in reuse[:20]],
            "price_repairs": dict(rep.get("counts") or {}),
            "trades_at_last_close_after_data_ended": held_missing,
            "note": "members without Yahoo prices are mostly companies that were bought or went "
                    "bankrupt; they are missing from the lab's index, so some survivorship bias "
                    "remains (bankruptcies flatter the result, takeovers at a premium hurt it)",
        },
        "reaches_target": reaches_target(variants, wname), "curves": curves,
        "span": {k: [str(dates[a]), str(dates[b])] for k, (a, b) in spans.items()},
    }


# --------------------------------------------------------------------------- run

def run_strategy_lab2(out_path: Path | None = None, *, offline: bool = False,
                      series_path: Path | None = None, db_path: Path | None = None,
                      n_paths: int = 10000, progress=print) -> dict:
    started = time.time()
    out_path = Path(out_path) if out_path else DEFAULT_OUT_PATH
    frames, series_info = ensure_series(Path(series_path or DEFAULT_SERIES_PATH),
                                        offline=offline, tickers=LAB2_TICKERS, progress=progress)
    vol = vol_block(frames, n_paths=n_paths, progress=progress)
    rot = rotation_block(frames, n_paths=n_paths, progress=progress)
    pit = pit_block(frames, db_path=Path(db_path or SP500_DB_PATH), progress=progress)
    payload = {
        "version": 1, "generated": datetime.now().isoformat(timespec="seconds"),
        "pre_registered": "2026-11-03", "target_cagr_pct": TARGET_CAGR_PCT,
        "series": series_info["series"],
        "point_in_time_momentum": pit, "vol_target_leverage": vol, "rotation": rot,
        "runtime_seconds": round(time.time() - started, 1),
    }
    write_json_atomic(out_path, payload, indent=1)
    for line in summary_lines(payload):
        progress(line)
    progress(f"wrote {out_path} ({payload['runtime_seconds']}s)")
    return payload


def _fmt(x, suffix="%"):
    return "n/a" if x is None else f"{x:.1f}{suffix}"


def summary_lines(p: dict) -> list[str]:
    out = []
    for key, title in (("point_in_time_momentum", "P momentum on real past members"),
                       ("vol_target_leverage", "V volatility-targeted Nasdaq leverage"),
                       ("rotation", "R rotation between markets")):
        b = p[key]
        w = b["winner"]
        out.append(f"== {title}: winner {w['name']}")
        for name, v in b["variants"].items():
            t, tr, f = v["periods"]["test"], v["periods"]["train"], v["periods"]["full"]
            out.append(f"   {name:28s} train {_fmt(tr['cagr_pct'])} dd {_fmt(tr['max_drawdown_pct'])}"
                       f" | test {_fmt(t['cagr_pct'])} dd {_fmt(t['max_drawdown_pct'])}"
                       f" | full {_fmt(f['cagr_pct'])} dd {_fmt(f['max_drawdown_pct'])}")
        for bk, bv in b["benchmarks"]["test"].items():
            out.append(f"   [test] {bk:26s} {_fmt(bv['cagr_pct'])} dd {_fmt(bv['max_drawdown_pct'])}")
        acct = w.get("account_test_de") or w.get("after_tax_test_de")
        if acct:
            mine = acct.get("account_after_tax_cagr_pct", acct.get("cagr_pct"))
            mine_ex = acct.get("account_after_tax_with_30pct_partial_exemption_pct",
                               acct.get("cagr_with_30pct_partial_exemption_pct"))
            qqq = acct.get("qqq_held_after_tax_cagr_pct", acct.get("qqq_held_cagr_pct"))
            qqq_ex = acct.get("qqq_held_after_tax_with_30pct_partial_exemption_pct",
                              acct.get("qqq_held_with_30pct_partial_exemption_pct"))
            out.append(f"   [test, after German tax est.] winner {_fmt(mine)} "
                       f"({_fmt(mine_ex)} with the 30% exemption) vs QQQ held {_fmt(qqq)} "
                       f"({_fmt(qqq_ex)})")
        out.append(f"   reaches 40%: {b['reaches_target']}")
    return out


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    out, offline, k = None, False, 0
    while k < len(argv):
        if argv[k] == "--offline":
            offline = True
        elif argv[k] == "--out" and k + 1 < len(argv):
            out = Path(argv[k + 1])
            k += 1
        else:
            raise SystemExit(f"unknown option: {argv[k]}")
        k += 1
    run_strategy_lab2(out, offline=offline)


if __name__ == "__main__":
    main()
