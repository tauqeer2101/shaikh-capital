"""Strategy lab (pre-registered): can a simple, mechanical rule reach 40% a year?

Why: the owner's goal is about 40% CAGR in USD over the next 10 years - more
than double the Nasdaq (the owner's reference: the ~39% a year after fees, 66%
before, reported for Jim Simons' Medallion fund). The v2 backtest shows the
chart-pattern system made 1.9%/yr (2017-2026) against 14.9% for the S&P 500 and
20.5% for the Nasdaq-100, and the exit experiment found no exit rule that fixes
it. Two candidate approaches are tested here against that target and against
the Nasdaq, with every rule fixed BEFORE a result was seen. The lab reports what
it finds, also when (as expected) neither reaches 40%.

STRATEGY A - momentum rotation in stocks
  Universe: universe/us.csv (today's S&P 500 + S&P 400 members) from the local
    price cache, repaired like the backtest (backtest.sanitize_bars). A stock is
    eligible on a signal day when it has a bar that day, at least 273 earlier
    bars and a 20-bar average traded value of at least scan.MIN_DOLLAR_VOL['us'].
  Signal: the close of the LAST trading day of each month; trades at the NEXT
    session's open. Momentum = close[t-21] / close[t-21-L] - 1 on the stock's
    own bars (the latest month is skipped), L in {126, 252}.
  Portfolio: the top N eligible stocks, equal weight at every rebalance, N in
    {10, 20}; leavers are sold and stayers re-weighted at that open; costs of
    0.15% per side + 0.05% slippage are charged on the traded notional only.
  Trend filter {off, on}: on = stocks only while ^GSPC closes above its 200-day
    average on the signal day, otherwise cash until the next monthly signal.
    Cash earns the 13-week T-bill yield (^IRX / 100 / 252 per trading day).
  Grid: 2 x 2 x 2 = 8 variants. TRAIN = first signal .. 2022-12-31, TEST =
    2023-01-01 .. last full month. Winner = highest TRAIN CAGR among variants
    whose TRAIN max drawdown is no worse than -50%.
  Survivorship: the universe is TODAY's members, so past winners are in it
    because they won. The equal-weight buy-and-hold basket of the same universe
    (equal amounts at the first signal, never rebalanced) shares that bias and
    is reported next to SPY / QQQ; 'excess vs the equal-weight basket' is the
    bias-reduced measure. The backtest's daily-rebalanced basket is kept and
    labelled not investable.
  Diagnostics (hindsight, not pre-registered): the winner's selection margin,
    its re-run without the top contributor, and placebo pickers (weakest
    momentum, random) with the same rules and costs.

STRATEGY B - leveraged Nasdaq-100 with a trend filter
  Series (Yahoo Finance, cached in data/cache/lab_series.sqlite3): ^NDX daily
    closes from 1985-10, ^IRX, and QQQ / QLD / TQQQ / SPY / ^GSPC.
  Synthetic daily-reset fund, L in {1, 2, 3}:
      r(t) = L * (r_ndx(t) + 0.7%/252) - (L-1) * (tbill(t) + 0.5%)/252 - expense/252
    expense = 0.20%/yr (L=1) or 0.95%/yr (L=2, 3); 0.7%/yr is an assumed dividend
    yield (^NDX is a price index). Validated against the real QQQ / QLD / TQQQ
    over each fund's live history; nothing is tuned to fit.
  Filter in {none, sma200, sma200_band}: the signal is read at the close of day
    t (^NDX close vs its 200-day average; the band variant switches in above
    +2% and out below -2%), the switch is done at the close of t+1, so the new
    holding earns from t+2 on. Out of the fund = T-bills. 0.10% of equity per
    switch.
  Grid: 3 x 3 = 9 variants from 1986-10. TRAIN = .. 2009-12-31, TEST =
    2010-01-01 ... Winner = highest TRAIN CAGR among variants whose TRAIN max
    drawdown is no worse than -75%.
  Also: rolling 10-year windows (monthly steps) and a circular block bootstrap
    (12-month blocks of the variant's own monthly returns, 10,000 paths of 120
    months, numpy default_rng(20261001)).
  Sensitivities, reported next to the headline and never used to choose a
    variant: same-close execution, dearer financing / lower dividends, 36-month
    bootstrap blocks, the index return lowered by 4 points a year, the fund
    model's error by T-bill regime, and an estimate of German withholding tax
    (26.375% of each realised gain, losses carried forward).

The verdict (answer.reaches_target) counts a family only when its pre-registered
winner made the target on TEST and over the whole history AND the family is
usable as evidence: momentum never is (survivorship), leverage is when the fund
model passed its check against the real funds. Threshold comparisons use
unrounded figures.

Stock prices come only from the local cache (opened read-only). The index /
ETF / T-bill series are downloaded once into their own file and reused, so a
run is repeatable offline.

Run: python -m atlas2 strategy-lab [--offline] [--out PATH] [--refresh]
Output: data/strategy_lab.json and data/strategy_lab_equity.json.
"""
from __future__ import annotations

import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .backtest import _rnd, load_long_prices, sanitize_long, write_json_atomic
from .portfolio_sim import DAILY_RET_CLIP, _connect_ro, price_repairs
from .scan import MIN_DOLLAR_VOL, load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = ROOT / "data" / "strategy_lab.json"
DEFAULT_PRICES_PATH = ROOT / "data" / "cache" / "prices.sqlite3"
DEFAULT_SERIES_PATH = ROOT / "data" / "cache" / "lab_series.sqlite3"
DEFAULT_UNIVERSE_PATH = ROOT / "universe" / "us.csv"
DEFAULT_DATA_DIR = ROOT / "data"

TARGET_CAGR_PCT = 40.0
TRADING_DAYS = 252
SERIES_TICKERS = ("^NDX", "^IRX", "QQQ", "QLD", "TQQQ", "SPY", "^GSPC")
SERIES_START = "1985-01-01"

# --- Strategy A (momentum rotation) - fixed before any result was seen
A_LOOKBACKS = (126, 252)
A_SKIP = 21
A_TOP_N = (10, 20)
A_FILTERS = (False, True)
A_MIN_PRIOR_BARS = 273
A_DV_WINDOW = 20
A_COST_PCT_PER_SIDE = 0.15
A_SLIPPAGE_PCT = 0.05
A_COST_RATE = (A_COST_PCT_PER_SIDE + A_SLIPPAGE_PCT) / 100.0
A_TRAIN_END = "2022-12-31"
A_TRAIN_DD_FLOOR_PCT = -50.0
A_SMA = 200
A_MARKET = "us"

# --- Strategy B (leveraged Nasdaq-100) - fixed before any result was seen
B_LEVERAGE = (1, 2, 3)
B_FILTERS = ("none", "sma200", "sma200_band")
B_START = "1986-10-01"
B_TRAIN_END = "2009-12-31"
B_TRAIN_DD_FLOOR_PCT = -75.0
B_DIV_YIELD = 0.007
B_FINANCING_SPREAD = 0.005
B_EXPENSE = {1: 0.0020, 2: 0.0095, 3: 0.0095}
B_SMA = 200
B_BAND = 0.02
B_SWITCH_COST = 0.0010
B_ETF = {1: "QQQ", 2: "QLD", 3: "TQQQ"}
B_VALIDATION_TOL_PP = 1.0
B_VALIDATION_MIN_DAYS = 63          # a sub-sample row needs about a quarter of data
B_TBILL_REGIMES = (("below 1%", -np.inf, 1.0), ("1% to 3%", 1.0, 3.0),
                   ("at or above 3%", 3.0, np.inf))
B_BASE = "ndx1x_none"
# sensitivities (reported next to the headline, never used to choose anything)
B_CONS_SPREAD = 0.010
B_CONS_DIV_YIELD = 0.002
DE_TAX_RATE = 0.26375               # 25% Abgeltungsteuer + 5.5% solidarity surcharge
DE_PARTIAL_EXEMPTION = 0.30         # Teilfreistellung for equity funds

ROLL_MONTHS = 120
BOOT_PATHS = 10000
BOOT_HORIZON = 120
BOOT_BLOCK = 12
BOOT_SEED = 20261001
BOOT_SENS_BLOCK = 36
BOOT_SENS_HAIRCUT_PP = 4.0

IN_SAMPLE_LABEL = "in-sample on test — do not act on this"
A_PLACEBO_SEEDS = 100
# benchmark names of Strategy A
EW_BH = "equal_weight_buy_and_hold"
EW_BH_FRESH = "equal_weight_buy_and_hold_from_test_start"
EW_DAILY = "equal_weight_universe_daily_rebalanced"
EW_ELIGIBLE = "equal_weight_eligible_monthly"
PERIODS_A = ("train", "test", "pooled")
PERIODS_B = ("train", "test", "full")


# --------------------------------------------------------------------------- small helpers

def _r2(x):
    return _rnd(x, 2)


def _pct(x, nd: int = 2):
    """Fraction -> percent, rounded; None / NaN stay None."""
    if x is None:
        return None
    return _rnd(float(x) * 100.0, nd)


def _days(dates) -> np.ndarray:
    """ISO date strings -> day numbers (for calendar-day differences)."""
    return np.asarray(dates, dtype="datetime64[D]").astype(np.int64)


def _last_on_or_before(dates: np.ndarray, d: str) -> int:
    """Index of the last date <= d (-1 when none)."""
    return int(np.searchsorted(dates, d, side="right")) - 1


def asof_values(src_dates: np.ndarray, src_vals: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """The latest source value on or before each date (NaN before the first)."""
    src_vals = np.asarray(src_vals, float)
    if not len(src_dates):
        return np.full(len(dates), np.nan)
    pos = np.searchsorted(src_dates, dates, side="right") - 1
    return np.where(pos >= 0, src_vals[np.maximum(pos, 0)], np.nan)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average over the last `window` values (NaN until it is full)."""
    return pd.Series(np.asarray(x, float)).rolling(window).mean().to_numpy()


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous / Meeus algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    j, k = c // 4, c % 4
    m = (a + 11 * h) // 319
    r = (2 * e + 2 * j - k - h + m + 32) % 7
    month = (h - m + r + 90) // 25
    day = (h - m + r + month + 19) % 32
    return date(year, month, day)


def month_end_market_holiday(d: date) -> bool:
    """The two scheduled NYSE holidays that can be the last weekday of a month:
    Good Friday (in March) and Memorial Day (the last Monday of May)."""
    if d.month == 5 and d.weekday() == 0 and d.day + 7 > 31:
        return True
    return d == _easter_sunday(d.year) - timedelta(days=2)


def month_complete(dates: np.ndarray, i: int) -> bool:
    """True when dates[i] is the last trading day of its month: the next bar is in
    another month, or (at the end of the data) no later weekday of the month is
    a trading day - the remaining weekdays, if any, are Good Friday or Memorial
    Day (the only scheduled NYSE holidays that can end a month)."""
    if i + 1 < len(dates):
        return str(dates[i + 1])[:7] != str(dates[i])[:7]
    d = date.fromisoformat(str(dates[i])[:10])
    nxt = d + timedelta(days=1)
    while (nxt.year, nxt.month) == (d.year, d.month):
        if nxt.weekday() < 5 and not month_end_market_holiday(nxt):
            return False
        nxt += timedelta(days=1)
    return True


def month_end_signal_idx(dates: np.ndarray) -> np.ndarray:
    """Indices of the last trading day of each month that has a NEXT session in
    the data (a signal needs the next open to trade on)."""
    m = np.array([str(d)[:7] for d in dates])
    return np.nonzero(m[1:] != m[:-1])[0]


# --------------------------------------------------------------------------- performance metrics

def longest_underwater(days: np.ndarray, seg: np.ndarray) -> int:
    """Longest spell below an earlier high, in calendar days from the high to the
    bar that RECOVERS it (a new high), or to the last bar when the spell is
    still open at the end. 0 when the curve never falls below a high."""
    seg = np.asarray(seg, float)
    if len(seg) < 2:
        return 0
    highs = np.nonzero(seg >= np.maximum.accumulate(seg))[0]      # bars at a (new) high
    best = 0
    if len(highs) > 1:
        gap = np.diff(highs) > 1                                   # bars under water between
        if gap.any():
            best = int((days[highs[1:]] - days[highs[:-1]])[gap].max())
    if highs[-1] < len(seg) - 1:
        best = max(best, int(days[-1] - days[highs[-1]]))
    return best


def exact(m: dict | None, key: str):
    """The unrounded figure behind m[key] when the period carries one (keys
    'cagr_pct' / 'max_drawdown_pct'); the stored figure otherwise. Used for every
    threshold comparison so that 39.996% is not counted as 40%."""
    if not m:
        return None
    v = m.get(key[:-4] + "_exact_pct") if key.endswith("_pct") else None
    return m.get(key) if v is None else v


def perf_metrics(dates: np.ndarray, equity: np.ndarray, i0: int | None, i1: int | None) -> dict:
    """CAGR, drawdown (daily marks), worst year, volatility and time under water
    of equity[i0..i1]; equity[i0] is the anchor (the close before the first
    day of the period). All figures in percent; drawdowns are negative.
    cagr_exact_pct / max_drawdown_exact_pct are the same figures unrounded (used
    for the target and drawdown-limit comparisons)."""
    out = {"start": None, "end": None, "years": None, "cagr_pct": None, "total_return_pct": None,
           "cagr_exact_pct": None, "max_drawdown_exact_pct": None,
           "max_drawdown_pct": None, "max_drawdown_peak": None, "max_drawdown_trough": None,
           "worst_year": None, "ann_volatility_pct": None, "longest_underwater_days": None,
           "yearly_pct": {}, "partial_years": []}
    if i0 is None or i1 is None or i0 < 0 or i1 <= i0:
        return out
    seg = np.asarray(equity[i0:i1 + 1], float)
    if not np.isfinite(seg).all() or seg[0] <= 0:
        return out
    d = np.asarray(dates[i0:i1 + 1], dtype=str)
    days = _days(d)
    years = (days[-1] - days[0]) / 365.25
    growth = seg[-1] / seg[0]
    out.update({"start": str(d[0]), "end": str(d[-1]), "years": round(float(years), 2),
                "total_return_pct": _pct(growth - 1)})
    if years > 0:
        exact = (growth ** (1 / years) - 1) * 100.0 if growth > 0 else -100.0
        out["cagr_pct"] = _rnd(exact, 2)
        out["cagr_exact_pct"] = float(exact)
    peak = np.maximum.accumulate(seg)
    dd = seg / peak - 1
    kt = int(np.argmin(dd))
    kp = int(np.argmax(seg[:kt + 1]))
    out.update({"max_drawdown_pct": _pct(dd[kt]), "max_drawdown_exact_pct": float(dd[kt]) * 100.0,
                "max_drawdown_peak": str(d[kp]), "max_drawdown_trough": str(d[kt])})
    prev = seg[:-1]
    rets = np.divide(seg[1:], prev, out=np.ones(len(prev)), where=prev > 0) - 1
    if len(rets) > 2:
        out["ann_volatility_pct"] = _pct(float(np.std(rets, ddof=1)) * math.sqrt(TRADING_DAYS))
    out["longest_underwater_days"] = longest_underwater(days, seg)
    yrs = np.array([s[:4] for s in d])
    yearly: dict[str, float] = {}
    for y in sorted(set(yrs[1:].tolist())):
        last = int(np.nonzero(yrs == y)[0][-1])
        before = np.nonzero(yrs < y)[0]
        base = int(before[-1]) if len(before) else 0
        if last > base and seg[base] > 0:
            yearly[y] = seg[last] / seg[base] - 1
    if yearly:
        ys = list(yearly)
        partial = []
        if yrs[0] == ys[0]:
            partial.append(ys[0])
        if i1 + 1 < len(dates):
            ends_in_year = str(dates[i1 + 1])[:4] == ys[-1]
        else:       # the end of the data: complete only on the last trading day of December
            ends_in_year = d[-1][5:7] != "12" or not month_complete(dates, i1)
        if ends_in_year and ys[-1] not in partial:
            partial.append(ys[-1])
        worst = min(ys, key=lambda y: yearly[y])
        out["yearly_pct"] = {y: _pct(v) for y, v in yearly.items()}
        out["partial_years"] = partial
        out["worst_year"] = {"year": worst, "return_pct": _pct(yearly[worst]),
                             "partial": worst in partial}
    return out


def monthly_stats(dates: np.ndarray, equity: np.ndarray, i0: int, i1: int) -> dict | None:
    """Per FULL calendar month inside (i0, i1]: the return and, relative to the
    month's starting equity, the lowest and highest daily close and the largest
    fall from a peak inside the month. A partial first or last month is left
    out. me_idx[0] is the anchor (the last bar before the first full month),
    me_idx[k] the last bar of month k."""
    if i1 <= i0:
        return None
    keys = np.array([str(x)[:7] for x in dates[i0:i1 + 1]])
    ends = [int(k) + i0 for k in np.nonzero(keys[1:] != keys[:-1])[0]]   # last bar of a month
    if month_complete(dates, i1):
        ends.append(i1)
    # ends[0] is i0 itself when i0 is a month end; otherwise the first month is
    # partial and its last bar becomes the anchor
    if len(ends) < 2:
        return None
    eq = np.asarray(equity, float)
    labels, ret, lo, hi, ddin = [], [], [], [], []
    for s, e in zip(ends[:-1], ends[1:]):
        labels.append(str(dates[e])[:7])
        if not eq[s] > 0:
            ret.append(-1.0), lo.append(0.0), hi.append(1.0), ddin.append(1.0)
            continue
        seg = eq[s:e + 1] / eq[s]
        ret.append(float(seg[-1] - 1))
        lo.append(float(seg.min()))
        hi.append(float(seg.max()))
        ddin.append(float((1 - seg / np.maximum.accumulate(seg)).max()))
    return {"months": labels, "me_idx": np.asarray(ends, int), "ret": np.asarray(ret),
            "lo": np.asarray(lo), "hi": np.asarray(hi), "ddin": np.asarray(ddin)}


def rolling_windows(dates: np.ndarray, equity: np.ndarray, ms: dict | None,
                    ref_equity: np.ndarray | None = None, months: int = ROLL_MONTHS,
                    target_pct: float = TARGET_CAGR_PCT) -> dict | None:
    """Every `months`-long window starting at a month end (monthly steps): CAGR
    spread, share of windows at or above the target (with the start dates of
    those windows) and at or above the reference curve's CAGR in the same
    window, worst fall inside any window, and the windows that do not overlap
    (first, first + months, ...) as the independent observations."""
    if ms is None or len(ms["me_idx"]) <= months:
        return None
    me = ms["me_idx"]
    eq = np.asarray(equity, float)
    yrs = months / 12.0
    cagr, ref, dds = [], [], []
    for a, b in zip(me[:-months], me[months:]):
        g = eq[b] / eq[a] if eq[a] > 0 else 0.0
        cagr.append(g ** (1 / yrs) - 1 if g > 0 else -1.0)
        seg = eq[a:b + 1]
        dds.append(float((seg / np.maximum.accumulate(seg) - 1).min()) if eq[a] > 0 else -1.0)
        if ref_equity is not None:
            rg = ref_equity[b] / ref_equity[a]
            ref.append(rg ** (1 / yrs) - 1 if rg > 0 else -1.0)
    c = np.asarray(cagr)
    kmin, kmax = int(np.argmin(c)), int(np.argmax(c))
    kdd = int(np.argmin(dds))

    def span(k: int) -> list[str]:
        return [str(dates[me[k]]), str(dates[me[k + months]])]

    hit = np.nonzero(c >= target_pct / 100.0)[0]
    decades = []
    for k in range(0, len(me) - months, months):          # windows that do not overlap
        row = {"start": str(dates[me[k]]), "end": str(dates[me[k + months]]),
               "cagr_pct": _pct(c[k])}
        if ref_equity is not None:
            row["nasdaq_1x_cagr_pct"] = _pct(ref[k])
        decades.append(row)
    return {
        "window_months": months, "n_windows": len(c),
        "first_window": span(0), "last_window": span(len(c) - 1),
        "min_cagr_pct": _pct(c.min()), "median_cagr_pct": _pct(np.median(c)),
        "max_cagr_pct": _pct(c.max()),
        "min_window": span(kmin), "max_window": span(kmax),
        "share_ge_target_pct": _rnd(100.0 * float((c >= target_pct / 100.0).mean()), 1),
        "windows_ge_target": int(len(hit)),
        "windows_ge_target_start_range": ([str(dates[me[hit[0]]]), str(dates[me[hit[-1]]])]
                                          if len(hit) else None),
        "windows_ge_target_start_years": sorted({str(dates[me[k]])[:4] for k in hit}),
        "share_ge_1x_nasdaq_pct": (_rnd(100.0 * float((c >= np.asarray(ref)).mean()), 1)
                                   if ref_equity is not None else None),
        "worst_drawdown_in_any_window_pct": _pct(dds[kdd]),
        "worst_drawdown_window": span(kdd),
        "independent_windows": len(decades),
        "non_overlapping_decades": decades,
        "note": ("the windows overlap, so a share of windows is NOT a probability; the "
                 "non-overlapping windows are the independent observations"),
    }


def block_bootstrap(ms: dict | None, n_paths: int = BOOT_PATHS, horizon: int = BOOT_HORIZON,
                    block: int = BOOT_BLOCK, seed: int = BOOT_SEED,
                    target_pct: float = TARGET_CAGR_PCT) -> dict | None:
    """Circular moving-block bootstrap of the monthly history: each path joins
    ceil(horizon / block) blocks of `block` consecutive months (uniform start,
    wrapping at the end of the history). The drawdown of a path is on DAILY
    marks: every month carries its own lowest / highest daily close and its
    largest fall inside the month, which compose exactly. A fresh
    default_rng(seed) per call, so every variant sees the same block starts."""
    if ms is None or len(ms["ret"]) < block:
        return None
    ret, lo, hi, ddin = ms["ret"], ms["lo"], ms["hi"], ms["ddin"]
    n = len(ret)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(horizon / block))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_paths, -1)
    idx = idx[:, :horizon]
    cum = np.cumprod(1.0 + ret[idx], axis=1)
    one = np.ones((n_paths, 1))
    start_eq = np.concatenate([one, cum[:, :-1]], axis=1)
    run_peak = np.maximum.accumulate(start_eq * hi[idx], axis=1)
    peak_before = np.concatenate([one, run_peak[:, :-1]], axis=1)
    dd = np.maximum(1.0 - start_eq * lo[idx] / peak_before, ddin[idx]).max(axis=1)
    final = cum[:, -1]
    cagr = np.where(final > 0, np.maximum(final, 1e-300) ** (12.0 / horizon) - 1.0, -1.0)
    q = np.percentile(cagr, [5, 25, 50, 75, 95])
    return {
        "months_in_history": n, "paths": int(n_paths), "horizon_months": int(horizon),
        "block_months": int(block), "seed": int(seed),
        "cagr_pct": {"p5": _pct(q[0]), "p25": _pct(q[1]), "p50": _pct(q[2]),
                     "p75": _pct(q[3]), "p95": _pct(q[4])},
        "p_cagr_ge_40_pct": _rnd(100.0 * float((cagr >= target_pct / 100.0).mean()), 2),
        "p_cagr_ge_20_pct": _rnd(100.0 * float((cagr >= 0.20).mean()), 2),
        "p_max_drawdown_worse_than_50_pct": _rnd(100.0 * float((dd > 0.50).mean()), 2),
        "p_max_drawdown_worse_than_75_pct": _rnd(100.0 * float((dd > 0.75).mean()), 2),
        "p_end_below_start_pct": _rnd(100.0 * float((final < 1.0).mean()), 2),
        "median_max_drawdown_pct": _pct(-float(np.median(dd))),
    }


def pick_winner(variants: dict, dd_floor_pct: float) -> dict:
    """The pre-registered rule: the highest TRAIN CAGR among the variants whose
    TRAIN max drawdown is no worse than dd_floor_pct (ties: the first in grid
    order). Returns {name, qualified, disqualified}."""
    qualified, disq = [], {}
    for name, v in variants.items():
        tr = v["periods"]["train"]
        dd, cagr = exact(tr, "max_drawdown_pct"), exact(tr, "cagr_pct")
        if dd is None or cagr is None:
            disq[name] = "no TRAIN figures"
        elif dd < dd_floor_pct:
            nd = 1 if round(dd, 1) < dd_floor_pct else 3
            disq[name] = f"TRAIN max drawdown {dd:.{nd}f}% is worse than {dd_floor_pct:.0f}%"
        else:
            qualified.append(name)
    name = None
    for q in qualified:
        if name is None or exact(variants[q]["periods"]["train"], "cagr_pct") > \
                exact(variants[name]["periods"]["train"], "cagr_pct"):
            name = q
    return {"name": name, "qualified": qualified, "disqualified": disq}


def selection_margin(variants: dict, pick: dict, dd_floor_pct: float) -> dict | None:
    """How close the winner rule's decision was: the winner's TRAIN CAGR lead over
    the next qualified variant and the distance of its TRAIN drawdown from the
    limit (percentage points; unrounded figures, rounded for display)."""
    name = pick.get("name")
    if not name:
        return None
    tr = variants[name]["periods"]["train"]
    others = [q for q in pick["qualified"] if q != name]
    runner = max(others, key=lambda q: exact(variants[q]["periods"]["train"], "cagr_pct")) \
        if others else None
    return {
        "runner_up": runner,
        "train_cagr_gap_to_runner_up_pp": (
            _r2(exact(tr, "cagr_pct") - exact(variants[runner]["periods"]["train"], "cagr_pct"))
            if runner else None),
        "train_drawdown_distance_to_limit_pp": _r2(exact(tr, "max_drawdown_pct") - dd_floor_pct),
        "train_drawdown_limit_pct": dd_floor_pct,
    }


def best_on(variants: dict, period: str) -> str | None:
    best = None
    for name, v in variants.items():
        c = exact(v["periods"][period], "cagr_pct")
        if c is not None and (best is None
                              or c > exact(variants[best]["periods"][period], "cagr_pct")):
            best = name
    return best


# --------------------------------------------------------------------------- series cache

SERIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL, adj_close REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS meta (
    ticker TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    first_date TEXT, last_date TEXT, n_rows INTEGER
);
"""


def fetch_series(tickers, start: str = SERIES_START, today: str | None = None
                 ) -> dict[str, pd.DataFrame]:
    """Download daily close + dividend-adjusted close from Yahoo Finance (the only
    network access of the lab). Bars dated today (UTC) or later are dropped: the
    session may still be open. Raises RuntimeError when a series comes back empty."""
    import yfinance as yf

    today = today or datetime.now(timezone.utc).date().isoformat()
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            raw = yf.download(t, start=start, interval="1d", auto_adjust=False, progress=False,
                              multi_level_index=False)
        except Exception as exc:
            raise RuntimeError(f"download of {t} failed: {exc.__class__.__name__}: {exc}") from exc
        if raw is None or raw.empty:
            raise RuntimeError(f"download of {t} returned no data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.droplevel(-1, axis=1)
        if "Close" not in raw.columns:
            raise RuntimeError(f"download of {t} has no Close column")
        df = pd.DataFrame({
            "close": raw["Close"].to_numpy(float),
            "adj_close": (raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
                          ).to_numpy(float),
        }, index=pd.DatetimeIndex(raw.index).strftime("%Y-%m-%d"))
        df = df[np.isfinite(df["close"]) & np.isfinite(df["adj_close"])]
        df = df[df.index < today]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if df.empty:
            raise RuntimeError(f"download of {t} returned no usable rows")
        out[t] = df
    return out


def save_series(path: Path, frames: dict[str, pd.DataFrame], fetched_at: dict[str, str]) -> None:
    """Write the whole series cache to a temp file and atomically replace `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(SERIES_SCHEMA)
        for t, df in frames.items():
            rows = [(t, str(d), float(c), float(a))
                    for d, c, a in zip(df.index, df["close"], df["adj_close"])]
            conn.executemany("INSERT OR REPLACE INTO series VALUES (?,?,?,?)", rows)
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?,?,?,?)",
                         (t, fetched_at.get(t, ""), rows[0][1] if rows else None,
                          rows[-1][1] if rows else None, len(rows)))
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, path)


def load_series(path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Read the series cache (read-only). Returns ({ticker: frame}, {ticker: meta})."""
    path = Path(path)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        long = pd.read_sql_query("SELECT ticker, date, close, adj_close FROM series "
                                 "ORDER BY ticker, date", conn)
        meta = {str(t): {"fetched_at": f, "first_date": a, "last_date": b, "rows": n}
                for t, f, a, b, n in conn.execute(
                    "SELECT ticker, fetched_at, first_date, last_date, n_rows FROM meta")}
    finally:
        conn.close()
    frames: dict[str, pd.DataFrame] = {}
    for t, g in long.groupby("ticker", sort=True):
        frames[str(t)] = pd.DataFrame(
            {"close": g["close"].to_numpy(float), "adj_close": g["adj_close"].to_numpy(float)},
            index=pd.Index(g["date"].astype(str).str[:10].to_numpy(dtype=str), name="date"))
    return frames, meta


def ensure_series(path: Path, *, offline: bool = False, refresh: bool = False,
                  tickers=SERIES_TICKERS, fetch=None, progress=print
                  ) -> tuple[dict[str, pd.DataFrame], dict]:
    """The lab's index / ETF / T-bill series: from the cache when it is there,
    downloaded (and cached) when it is not. offline never touches the network
    and fails clearly when the cache or a series is missing; refresh downloads
    everything again."""
    path = Path(path)
    if offline and refresh:
        raise ValueError("use either --offline or --refresh, not both")
    fetch = fetch or fetch_series
    frames: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    if path.exists() and not refresh:
        frames, meta = load_series(path)
    elif offline:
        raise FileNotFoundError(
            f"--offline: the series cache {path} is missing; run once without --offline to "
            f"download {', '.join(tickers)}")
    missing = [t for t in tickers if t not in frames or frames[t].empty]
    downloaded: list[str] = []
    if missing:
        if offline:
            raise FileNotFoundError(
                f"--offline: the series cache {path} has no data for {', '.join(missing)}; run "
                "once without --offline (or with --refresh)")
        progress(f"downloading {', '.join(missing)} from Yahoo Finance into {path.name} ...")
        got = fetch(missing)
        still = [t for t in missing if t not in got or got[t].empty]
        if still:
            raise RuntimeError(f"no data downloaded for {', '.join(still)}")
        now = datetime.now().isoformat(timespec="seconds")
        frames.update({t: got[t] for t in missing})
        fetched = {t: (meta.get(t) or {}).get("fetched_at", "") for t in frames}
        fetched.update({t: now for t in missing})
        save_series(path, frames, fetched)
        frames, meta = load_series(path)
        downloaded = list(missing)
    info = {"path": str(path), "source": "Yahoo Finance (yfinance), daily close and "
            "dividend-adjusted close; ^NDX, ^GSPC and ^IRX are used unadjusted",
            "downloaded_this_run": downloaded,
            "series": {t: {"first": str(frames[t].index[0]), "last": str(frames[t].index[-1]),
                           "rows": int(len(frames[t])),
                           "fetched_at": (meta.get(t) or {}).get("fetched_at")}
                       for t in tickers}}
    return {t: frames[t] for t in tickers}, info


# --------------------------------------------------------------------------- strategy B: leveraged Nasdaq-100

def b_name(lev: int, filt: str) -> str:
    return f"ndx{lev}x_{filt}"


def b_label(lev: int, filt: str) -> str:
    base = "the plain Nasdaq-100 fund (1x)" if lev == 1 else f"a {lev}x Nasdaq-100 fund"
    return base + {"none": ", always invested",
                   "sma200": ", in T-bills while the index is below its 200-day average",
                   "sma200_band": ", in T-bills while the index is below its 200-day average "
                                  "(2% band)"}[filt]


def synthetic_fund_returns(close: np.ndarray, tbill_pct: np.ndarray, leverage: int, *,
                           div_yield: float = B_DIV_YIELD, spread: float = B_FINANCING_SPREAD,
                           expense: float | None = None) -> np.ndarray:
    """Daily returns of a daily-reset leveraged index fund:
        L * (r_index + div_yield/252) - (L-1) * (tbill + spread)/252 - expense/252
    tbill_pct is the annual T-bill yield in percent on the same day. The first
    value is NaN (no previous close); a loss is capped at -100% (the fund is
    wiped out)."""
    close = np.asarray(close, float)
    tb = np.asarray(tbill_pct, float) / 100.0
    expense = B_EXPENSE[leverage] if expense is None else expense
    r = np.full(len(close), np.nan)
    r[1:] = close[1:] / close[:-1] - 1
    fund = (leverage * (r + div_yield / TRADING_DAYS)
            - (leverage - 1) * (tb + spread) / TRADING_DAYS - expense / TRADING_DAYS)
    return np.maximum(fund, -1.0)


def cash_returns(tbill_pct: np.ndarray) -> np.ndarray:
    """Daily return of T-bills: yield / 100 / 252."""
    return np.asarray(tbill_pct, float) / 100.0 / TRADING_DAYS


def trend_state(close: np.ndarray, kind: str, window: int = B_SMA, band: float = B_BAND
                ) -> np.ndarray:
    """The holding the signal asks for at each close: 1 = fund, 0 = T-bills,
    -1 = not defined yet (the average needs `window` closes).
    none: always 1. sma200: close > average. sma200_band: into the fund only
    above average * (1 + band), out only below average * (1 - band), otherwise
    unchanged (the first defined day uses the plain rule)."""
    close = np.asarray(close, float)
    n = len(close)
    if kind == "none":
        return np.ones(n, dtype=np.int8)
    sma = rolling_mean(close, window)
    ok = np.isfinite(sma)
    state = np.full(n, -1, dtype=np.int8)
    if kind == "sma200":
        state[ok] = (close[ok] > sma[ok]).astype(np.int8)
        return state
    if kind != "sma200_band":
        raise ValueError(f"unknown filter: {kind!r}")
    cur = -1
    for t in range(n):
        if not ok[t]:
            continue
        if cur == -1:
            cur = 1 if close[t] > sma[t] else 0
        elif cur == 0 and close[t] > sma[t] * (1 + band):
            cur = 1
        elif cur == 1 and close[t] < sma[t] * (1 - band):
            cur = 0
        state[t] = cur
    return state


def first_sim_index(dates: np.ndarray, close: np.ndarray, start: str = B_START,
                    window: int = B_SMA) -> int:
    """The first day whose return counts: on or after `start`, and late enough for
    the signal two closes earlier to exist (the same day for every variant)."""
    d0 = int(np.searchsorted(dates, start, side="left"))
    return max(d0, window + 1)


def run_leverage(dates: np.ndarray, close: np.ndarray, tbill_pct: np.ndarray, leverage: int,
                 filt: str, *, d0: int | None = None, switch_cost: float = B_SWITCH_COST,
                 delay: int = 1, div_yield: float = B_DIV_YIELD,
                 spread: float = B_FINANCING_SPREAD) -> dict:
    """One variant on the index calendar. Headline timing (delay=1): the signal of
    close t is acted on at the close of t+1, so day d is spent in what
    state[d-2] asked for; a switch at the close of day e (state[e-1] !=
    state[e-2]) costs switch_cost of equity that day. delay=0 is the
    same-close sensitivity (switch at the signal's own close, day d holds
    state[d-1]). No cost for the opening position. Returns full-length arrays:
    equity (NaN before the anchor d0-1, 1.0 at it), in_fund and switch."""
    n = len(dates)
    d0 = first_sim_index(dates, close) if d0 is None else int(d0)
    if delay not in (0, 1):
        raise ValueError("delay must be 0 (same close) or 1 (next close)")
    if d0 < 2 or d0 >= n:
        raise ValueError("not enough index history for the leveraged strategy")
    fund = synthetic_fund_returns(close, tbill_pct, leverage, div_yield=div_yield, spread=spread)
    cash = cash_returns(tbill_pct)
    state = trend_state(close, filt)
    if (state[d0 - 1 - delay:] < 0).any():
        raise ValueError("the trend signal is undefined inside the simulated period")
    held = state[d0 - 1 - delay:n - 1 - delay] == 1      # day d holds state[d-1-delay]
    after = state[d0 - delay:n - delay] == 1             # the holding after the close of day d
    sw = after != held
    day_ret = np.where(held, fund[d0:], cash[d0:])
    if not np.isfinite(day_ret).all():
        raise ValueError("missing index or T-bill data inside the simulated period")
    growth = (1.0 + day_ret) * np.where(sw, 1.0 - switch_cost, 1.0)
    equity = np.full(n, np.nan)
    equity[d0 - 1] = 1.0
    equity[d0:] = np.cumprod(growth)
    in_fund = np.zeros(n, bool)
    in_fund[d0:] = held
    switch = np.zeros(n, bool)
    switch[d0:] = sw
    return {"equity": equity, "in_fund": in_fund, "switch": switch, "d0": d0}


def after_tax_growth(equity: np.ndarray, in_fund: np.ndarray, switch: np.ndarray, i0: int,
                     i1: int, *, rate: float = DE_TAX_RATE, exemption: float = 0.0) -> float:
    """What 1.0 at the close of i0 is worth at the close of i1 after an ESTIMATE of
    German withholding tax on a switching account:
      - at every switch the holding is sold; `rate` of the realised gain since
        it was bought is withheld from the proceeds (fund gains count only
        (1 - exemption) when the partial exemption for equity funds applies;
        T-bill interest is taxed in full, at the switch back into the fund);
      - a realised loss is carried forward and used up by later gains;
      - whatever is held at the end is sold and taxed the same way, so a
        buy-and-hold account is taxed once, at the end.
    Simplifications: no annual allowance, no Vorabpauschale, no church tax; a
    loss carry-forward left at the end is worth nothing."""
    eq = np.asarray(equity, float)
    if i1 <= i0 or not eq[i0] > 0:
        return float("nan")
    wealth, basis, losses = 1.0, 1.0, 0.0
    holding_fund = bool(in_fund[i0 + 1])
    last = i0

    def settle(w: float, fund: bool) -> float:
        nonlocal losses
        gain = (w - basis) * ((1.0 - exemption) if fund else 1.0)
        if gain <= 0:
            losses -= gain
            return w
        used = min(losses, gain)
        losses -= used
        return w - rate * (gain - used)

    for e in np.nonzero(np.asarray(switch[i0 + 1:i1 + 1], bool))[0] + i0 + 1:
        if not eq[last] > 0:
            return 0.0
        wealth = settle(wealth * eq[e] / eq[last], holding_fund)
        basis = wealth
        holding_fund = not holding_fund
        last = int(e)
    if not eq[last] > 0:
        return 0.0
    return float(settle(wealth * eq[i1] / eq[last], holding_fund))


def after_tax_estimate(dates: np.ndarray, res: dict, i0: int, i1: int) -> dict | None:
    """after_tax_growth as CAGRs in percent: plain and with the 30% partial
    exemption (Teilfreistellung) for equity funds."""
    if i1 <= i0:
        return None
    days = _days([dates[i0], dates[i1]])
    years = (days[1] - days[0]) / 365.25
    out = {}
    for key, ex in (("cagr_pct", 0.0),
                    ("cagr_with_30pct_partial_exemption_pct", DE_PARTIAL_EXEMPTION)):
        g = after_tax_growth(res["equity"], res["in_fund"], res["switch"], i0, i1, exemption=ex)
        if not np.isfinite(g) or years <= 0:
            out[key] = None
        else:
            out[key] = _pct(g ** (1 / years) - 1) if g > 0 else -100.0
    return out


def leverage_period(dates: np.ndarray, res: dict, i0: int, i1: int) -> dict:
    m = perf_metrics(dates, res["equity"], i0, i1)
    if m["years"]:
        n_sw = int(res["switch"][i0 + 1:i1 + 1].sum())
        m["switches"] = n_sw
        m["switches_per_year"] = _rnd(n_sw / m["years"], 2)
        m["time_in_market_pct"] = _rnd(100.0 * float(res["in_fund"][i0 + 1:i1 + 1].mean()), 1)
        m["after_tax_estimate_de"] = after_tax_estimate(dates, res, i0, i1)
    return m


def haircut_close(close: np.ndarray, pp_per_year: float) -> np.ndarray:
    """The index with pp_per_year percentage points taken off its return, spread
    evenly over 252 trading days (the first close is unchanged)."""
    close = np.asarray(close, float)
    out = close.copy()
    r = close[1:] / close[:-1] - 1 - pp_per_year / 100.0 / TRADING_DAYS
    out[1:] = close[0] * np.cumprod(1.0 + r)
    return out


def _annualised_diff_pp(real: np.ndarray, syn: np.ndarray) -> float | None:
    """Synthetic minus real, annualised over the given days (252 per year), in pp."""
    n = len(real)
    if n < B_VALIDATION_MIN_DAYS:
        return None
    g_real, g_syn = float(np.prod(1 + real)), float(np.prod(1 + syn))
    if g_real <= 0 or g_syn <= 0:
        return None
    return float((g_syn ** (TRADING_DAYS / n) - g_real ** (TRADING_DAYS / n)) * 100)


def validate_synthetic(dates: np.ndarray, fund_ret: np.ndarray, etf: pd.DataFrame, ticker: str,
                       leverage: int, tbill_pct: np.ndarray | None = None) -> dict:
    """Synthetic fund vs the real ETF (dividend-adjusted) over the ETF's live
    history: CAGR of both, the difference in percentage points and the
    correlation of the daily returns. With tbill_pct (aligned to dates) the
    difference is also split by thirds of the live history and by T-bill
    regime, because one whole-life number can hide offsetting errors. Nothing
    is fitted."""
    e_dates = np.asarray(etf.index, dtype=str)
    adj = etf["adj_close"].to_numpy(float)
    real = pd.Series(adj[1:] / adj[:-1] - 1, index=e_dates[1:])
    syn = pd.Series(np.asarray(fund_ret, float), index=np.asarray(dates, dtype=str))
    cols = {"real": real, "syn": syn}
    if tbill_pct is not None:
        cols["tb"] = pd.Series(np.asarray(tbill_pct, float), index=np.asarray(dates, dtype=str))
    both = pd.concat(cols, axis=1, join="inner").dropna()
    out = {"etf": ticker, "leverage": leverage, "days": int(len(both))}
    if len(both) < 30:
        out["error"] = "fewer than 30 common days"
        return out
    years = (_days([both.index[-1]])[0] - _days([e_dates[0]])[0]) / 365.25
    g_real = float((1 + both["real"]).prod())
    g_syn = float((1 + both["syn"]).prod())
    c_real = g_real ** (1 / years) - 1
    c_syn = g_syn ** (1 / years) - 1
    diff = (c_syn - c_real) * 100
    out.update({
        "start": str(e_dates[0]), "end": str(both.index[-1]), "years": round(float(years), 2),
        "synthetic_cagr_pct": _pct(c_syn), "real_cagr_pct": _pct(c_real),
        "cagr_diff_pp": _r2(diff),
        "daily_return_correlation": _rnd(float(both["real"].corr(both["syn"])), 4),
        "tolerance_pp": B_VALIDATION_TOL_PP,
        "within_tolerance": bool(abs(diff) <= B_VALIDATION_TOL_PP),
    })
    if tbill_pct is None:
        return out
    r, y, tb = both["real"].to_numpy(), both["syn"].to_numpy(), both["tb"].to_numpy()
    idx = np.asarray(both.index, dtype=str)

    def row(mask_or_slice) -> dict:
        d = _annualised_diff_pp(r[mask_or_slice], y[mask_or_slice])
        t = tb[mask_or_slice]
        return {"days": int(len(t)),
                "avg_tbill_pct": _r2(float(t.mean())) if len(t) else None,
                "synthetic_minus_real_pp_per_year": _r2(d) if d is not None else None,
                "within_tolerance": (bool(abs(d) <= B_VALIDATION_TOL_PP) if d is not None
                                     else None)}

    cuts = [round(k * len(idx) / 3) for k in range(4)]
    out["by_period"] = [{"start": str(idx[a]), "end": str(idx[b - 1]), **row(slice(a, b))}
                        for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    out["by_tbill_regime"] = [{"regime": label, **row((tb >= lo) & (tb < hi))}
                              for label, lo, hi in B_TBILL_REGIMES]
    worst = [abs(x["synthetic_minus_real_pp_per_year"]) for x in out["by_tbill_regime"]
             if x["synthetic_minus_real_pp_per_year"] is not None]
    out["max_abs_regime_diff_pp"] = _r2(max(worst)) if worst else None
    out["regimes_within_tolerance"] = all(x["within_tolerance"] is not False
                                          for x in out["by_tbill_regime"])
    return out


def _regime_of(tbill_pct: float) -> str:
    for label, lo, hi in B_TBILL_REGIMES:
        if lo <= tbill_pct < hi:
            return label
    return B_TBILL_REGIMES[-1][0]


def held_after_sell_signal(dates: np.ndarray, close: np.ndarray, tbill_pct: np.ndarray,
                           run: dict) -> dict | None:
    """The worst index day on which the delayed rule still held the fund although
    the sell signal had already been given at the previous close (a switch out
    of the fund is executed at that day's close)."""
    days = np.nonzero(run["switch"] & run["in_fund"])[0]
    if not len(days):
        return None
    r = close[days] / close[days - 1] - 1
    k = int(days[int(np.argmin(r))])
    out = {"date": str(dates[k]), "index_return_pct": _pct(close[k] / close[k - 1] - 1),
           "days_held_after_a_sell_signal": int(len(days))}
    for lev in B_LEVERAGE:
        out[f"fund_{lev}x_return_pct"] = _pct(synthetic_fund_returns(close, tbill_pct, lev)[k])
    return out


def leverage_block(frames: dict[str, pd.DataFrame], *, n_paths: int = BOOT_PATHS,
                   progress=print) -> dict:
    """Strategy B on the cached series. Returns {block, benchmarks, curves, info}."""
    ndx = frames["^NDX"]
    dates = np.asarray(ndx.index, dtype=str)
    close = ndx["close"].to_numpy(float)
    irx = frames["^IRX"]
    tb = asof_values(np.asarray(irx.index, dtype=str), irx["close"].to_numpy(float), dates)
    n = len(dates)
    if n < B_SMA + 3:
        raise ValueError("^NDX history is too short for the 200-day average")
    d0 = first_sim_index(dates, close)
    if d0 >= n - 1:
        raise ValueError("^NDX history ends before the leveraged strategy can start")
    if not np.isfinite(tb[d0:]).all():
        raise ValueError("^IRX has no value for part of the simulated period")
    tb = np.where(np.isfinite(tb), tb, 0.0)
    anchor, i_end = d0 - 1, n - 1
    i_split = _last_on_or_before(dates, B_TRAIN_END)
    spans = {"train": (anchor, i_split), "test": (i_split, i_end), "full": (anchor, i_end)}
    progress(f"[B] ^NDX {dates[0]} .. {dates[-1]} ({n} bars); first 200-day average "
             f"{dates[B_SMA - 1]}; simulated from {dates[d0]} (anchor {dates[anchor]}); "
             f"train to {dates[i_split] if i_split > anchor else '-'}, test from there")
    grid = [(lev, filt) for lev in B_LEVERAGE for filt in B_FILTERS]

    runs: dict[str, dict] = {}
    for lev, filt in grid:
        runs[b_name(lev, filt)] = run_leverage(dates, close, tb, lev, filt, d0=d0)
    base_eq = runs[B_BASE]["equity"]
    base_cagr = {p: perf_metrics(dates, base_eq, *spans[p])["cagr_pct"] for p in PERIODS_B}

    variants: dict[str, dict] = {}
    boot: dict[str, dict | None] = {}
    boot_block36: dict[str, dict | None] = {}
    for lev, filt in grid:
        name = b_name(lev, filt)
        res = runs[name]
        ms = monthly_stats(dates, res["equity"], anchor, i_end)
        v = {"leverage": lev, "filter": filt, "label": b_label(lev, filt),
             "periods": {p: leverage_period(dates, res, *spans[p]) for p in PERIODS_B},
             "rolling_10y": rolling_windows(dates, res["equity"], ms,
                                            None if name == B_BASE else base_eq)}
        for p in PERIODS_B:
            c, b = v["periods"][p]["cagr_pct"], base_cagr[p]
            v["periods"][p]["cagr_minus_1x_nasdaq_pp"] = (
                _r2(c - b) if name != B_BASE and c is not None and b is not None else None)
        variants[name] = v
        bs = block_bootstrap(ms, n_paths=n_paths)
        boot[name] = (None if bs is None else
                      {"full_history_cagr_pct": v["periods"]["full"]["cagr_pct"], **bs})
        boot_block36[name] = _boot_probs(block_bootstrap(ms, n_paths=n_paths,
                                                         block=BOOT_SENS_BLOCK))

    # the pre-registered winner
    pick = pick_winner(variants, B_TRAIN_DD_FLOOR_PCT)
    wname = pick["name"]
    best_test = best_on(variants, "test")
    winner = {
        "rule": (f"highest TRAIN CAGR (to {B_TRAIN_END}) among the variants whose TRAIN max "
                 f"drawdown is no worse than {B_TRAIN_DD_FLOOR_PCT:.0f}%"),
        "name": wname, "label": variants[wname]["label"] if wname else None,
        "qualified": pick["qualified"], "disqualified": pick["disqualified"],
        "selection_margin": selection_margin(variants, pick, B_TRAIN_DD_FLOOR_PCT),
        "benchmark": B_BASE,
        "best_test_variant": {"name": best_test, "label": IN_SAMPLE_LABEL,
                              "test_cagr_pct": variants[best_test]["periods"]["test"]["cagr_pct"]
                              if best_test else None},
    }

    def per_dd(m: dict):
        c, dd = m.get("cagr_pct"), m.get("max_drawdown_pct")
        return _r2(c / abs(dd)) if c is not None and dd else None

    if wname:
        for p in PERIODS_B:
            w, b = variants[wname]["periods"][p], variants[B_BASE]["periods"][p]
            winner[p] = {k: w.get(k) for k in ("start", "end", "years", "cagr_pct",
                                               "max_drawdown_pct", "worst_year",
                                               "longest_underwater_days", "ann_volatility_pct",
                                               "switches_per_year", "time_in_market_pct",
                                               "after_tax_estimate_de")}
            winner[p]["nasdaq_1x_cagr_pct"] = b["cagr_pct"]
            winner[p]["nasdaq_1x_max_drawdown_pct"] = b["max_drawdown_pct"]
            winner[p]["nasdaq_1x_after_tax_estimate_de"] = b.get("after_tax_estimate_de")
            winner[p]["cagr_minus_1x_nasdaq_pp"] = w.get("cagr_minus_1x_nasdaq_pp")
            winner[p]["cagr_per_unit_drawdown"] = per_dd(w)
            winner[p]["nasdaq_1x_cagr_per_unit_drawdown"] = per_dd(b)
            same = variants[b_name(variants[wname]["leverage"], "none")]["periods"][p]
            winner[p]["same_leverage_unfiltered"] = {
                "name": b_name(variants[wname]["leverage"], "none"),
                "cagr_pct": same["cagr_pct"], "max_drawdown_pct": same["max_drawdown_pct"],
                "cagr_per_unit_drawdown": per_dd(same)}
        winner["rolling_10y"] = variants[wname]["rolling_10y"]
        winner["bootstrap_10y"] = boot[wname]
    else:
        winner["note"] = "no variant kept its TRAIN drawdown inside the limit - no winner"

    # synthetic fund vs the real ETFs
    validation: dict[str, dict] = {}
    flags: list[str] = []
    regime_flags: list[str] = []
    for lev, etf in B_ETF.items():
        v = validate_synthetic(dates, synthetic_fund_returns(close, tb, lev), frames[etf], etf,
                               lev, tb)
        validation[etf] = v
        if v.get("error"):
            flags.append(f"{etf}: {v['error']}")
            continue
        if not v["within_tolerance"]:
            flags.append(f"MODEL DEVIATION: the synthetic {lev}x fund differs from the real {etf} "
                         f"by {v['cagr_diff_pp']:+.2f} pct-pt of CAGR over {v['start']} .. "
                         f"{v['end']} (synthetic {v['synthetic_cagr_pct']:.2f}%, real "
                         f"{v['real_cagr_pct']:.2f}%; tolerance {B_VALIDATION_TOL_PP:.1f}). "
                         "The parameters were not tuned to fit.")
        off = [x for x in v["by_tbill_regime"] if x["within_tolerance"] is False]
        if off:
            regime_flags.append(
                f"MODEL DEVIATION BY T-BILL REGIME: the synthetic {lev}x fund differs from the "
                f"real {etf} by "
                + " and ".join(f"{x['synthetic_minus_real_pp_per_year']:+.2f} pct-pt/yr while "
                               f"T-bills are {x['regime']}" for x in off)
                + f" (tolerance {B_VALIDATION_TOL_PP:.1f}; positive = the model is too "
                "optimistic). The parameters were not tuned to fit.")
    validation["flags"] = flags + regime_flags
    validation["note"] = ("synthetic minus real CAGR over each fund's live history, from the "
                          "dividend-adjusted ETF closes; a positive difference means the model "
                          "is too optimistic for that leverage. by_period (thirds of the live "
                          "history) and by_tbill_regime annualise the same difference over the "
                          "days of each row (252 days a year); a regime row needs at least "
                          f"{B_VALIDATION_MIN_DAYS} days")
    validation["tbill_average_pct"] = {
        p: _r2(float(tb[a + 1:b + 1].mean())) if b > a else None for p, (a, b) in spans.items()}
    whole_ok = all(not validation[e].get("error") and validation[e].get("within_tolerance")
                   for e in B_ETF.values())
    validation["usable_as_evidence"] = bool(whole_ok)
    validation["usable_as_evidence_basis"] = (
        "true when the synthetic 1x / 2x / 3x funds each match the real QQQ / QLD / TQQQ within "
        f"{B_VALIDATION_TOL_PP:.1f} pct-pt of CAGR over the fund's whole live history; deviations "
        "by T-bill regime are flagged separately and qualify the era before the funds existed")

    # sensitivity: more expensive financing, lower dividends (nothing is re-tuned)
    cons: dict[str, dict] = {}
    for lev, filt in grid:
        r = run_leverage(dates, close, tb, lev, filt, d0=d0, div_yield=B_CONS_DIV_YIELD,
                         spread=B_CONS_SPREAD)
        cons[b_name(lev, filt)] = {"periods": {p: perf_metrics(dates, r["equity"], *spans[p])
                                               for p in PERIODS_B}}
    cons_pick = pick_winner(cons, B_TRAIN_DD_FLOOR_PCT)
    conservative = {
        "label": "sensitivity - not the headline",
        "assumptions": (f"financing at T-bills + {B_CONS_SPREAD:.1%} (headline "
                        f"{B_FINANCING_SPREAD:.1%}) and a {B_CONS_DIV_YIELD:.1%}/yr dividend yield "
                        f"(headline {B_DIV_YIELD:.1%}); everything else unchanged"),
        "variants": {name: {f"{p}_cagr_pct": c["periods"][p]["cagr_pct"] for p in PERIODS_B}
                     | {"train_max_drawdown_pct": c["periods"]["train"]["max_drawdown_pct"]}
                     for name, c in cons.items()},
        "winner": cons_pick["name"],
        "winner_changes": cons_pick["name"] != wname,
    }

    # sensitivity: switching at the signal's own close instead of one close later
    same_close: dict[str, dict] = {}
    for lev, filt in grid:
        name = b_name(lev, filt)
        if filt == "none":
            same_close[name] = {"periods": variants[name]["periods"]}
            continue
        r = run_leverage(dates, close, tb, lev, filt, d0=d0, delay=0)
        same_close[name] = {"periods": {p: perf_metrics(dates, r["equity"], *spans[p])
                                        for p in PERIODS_B}}
    sc_pick = pick_winner(same_close, B_TRAIN_DD_FLOOR_PCT)
    filtered = [b_name(lev, filt) for lev, filt in grid if filt != "none"]

    def n_where(period: str, key: str, better) -> int:
        return sum(1 for nm in filtered
                   if better(exact(variants[nm]["periods"][period], key),
                             exact(same_close[nm]["periods"][period], key)))

    timing = {
        "label": "sensitivity - not the headline",
        "headline": "signal at the close of t, switch at the close of t+1, exposure from t+2",
        "same_close": "signal and switch at the close of t, exposure from t+1 (needs the "
                      "closing price before the close - an idealised reading)",
        "variants": {name: {f"{p}_{k}": same_close[name]["periods"][p][k]
                            for p in PERIODS_B for k in ("cagr_pct", "max_drawdown_pct")}
                     for name in filtered},
        "winner_same_close": sc_pick["name"],
        "winner_changes": sc_pick["name"] != wname,
        "filtered_variants": len(filtered),
        "delay_higher_train_cagr": n_where("train", "cagr_pct", lambda a, b: a > b),
        "delay_shallower_train_drawdown": n_where("train", "max_drawdown_pct",
                                                  lambda a, b: a > b),
        "delay_lower_test_cagr": n_where("test", "cagr_pct", lambda a, b: a < b),
        "winner_train_same_close": (
            {k: same_close[wname]["periods"]["train"][k] for k in ("cagr_pct", "max_drawdown_pct")}
            if wname else None),
        "worst_day_held_after_sell_signal": held_after_sell_signal(
            dates, close, tb, runs[b_name(max(B_LEVERAGE), "sma200")]),
    }

    # sensitivity of the bootstrap: a lower drift for the index
    cut = haircut_close(close, BOOT_SENS_HAIRCUT_PP)
    boot_haircut: dict[str, dict | None] = {}
    for lev, filt in grid:
        r = run_leverage(dates, cut, tb, lev, filt, d0=d0)
        probs = _boot_probs(block_bootstrap(monthly_stats(dates, r["equity"], anchor, i_end),
                                            n_paths=n_paths))
        if probs is not None:
            probs = {"full_history_cagr_pct":
                     perf_metrics(dates, r["equity"], anchor, i_end)["cagr_pct"], **probs}
        boot_haircut[b_name(lev, filt)] = probs

    # benchmarks on the same calendar
    bench: dict[str, dict] = {p: {} for p in PERIODS_B}
    ndx_eq = close / close[anchor]
    curves_extra = {"ndx_price_index": ndx_eq}
    for p in PERIODS_B:
        bench[p]["ndx_price_index"] = perf_metrics(dates, ndx_eq, *spans[p])
        bench[p]["nasdaq_1x_synthetic"] = dict(variants[B_BASE]["periods"][p])
    for etf in ("QQQ", "SPY"):
        e = frames[etf]
        adj = asof_values(np.asarray(e.index, dtype=str), e["adj_close"].to_numpy(float), dates)
        for p in PERIODS_B:
            a, b = spans[p]
            if b > a >= 0 and np.isfinite(adj[a]):
                bench[p][etf] = perf_metrics(dates, adj, a, b)
            else:
                bench[p][etf] = {"cagr_pct": None,
                                 "note": f"{etf} starts {e.index[0]}, after this period begins"}
    if wname:
        # the real Nasdaq-100 fund over the same TEST days (QQQ starts in 1999, so only
        # TEST has it); the exact figure feeds the answer text without double rounding
        q = bench["test"]["QQQ"]
        winner["test"]["qqq_real_cagr_pct"] = q.get("cagr_pct")
        winner["test"]["qqq_real_cagr_exact_pct"] = q.get("cagr_exact_pct")
        winner["test"]["qqq_real_max_drawdown_pct"] = q.get("max_drawdown_pct")

    me = monthly_stats(dates, base_eq, anchor, i_end)
    me_idx = list(me["me_idx"]) if me is not None else [anchor]
    if me_idx[-1] != i_end:
        me_idx.append(i_end)
    curves = {"frequency": "month-end (plus the latest bar)",
              "dates": [str(dates[i]) for i in me_idx],
              "series": {name: [float(f"{runs[name]['equity'][i]:.6g}") for i in me_idx]
                         for name in runs}}
    for k, eq in curves_extra.items():
        curves["series"][k] = [float(f"{eq[i]:.6g}") for i in me_idx]

    n_months = len(me["ret"]) if me is not None else 0
    info = {"first_bar": str(dates[0]), "last_bar": str(dates[-1]), "bars": n,
            "first_sma200": str(dates[B_SMA - 1]), "anchor": str(dates[anchor]),
            "first_day": str(dates[d0]),
            "train": [str(dates[anchor]), str(dates[i_split])] if i_split > anchor else None,
            "test": [str(dates[i_split]), str(dates[i_end])] if i_end > i_split >= anchor else None,
            "full_months": n_months}
    block = {"variants": variants, "winner": winner, "validation": validation,
             "bootstrap": {
                 "method": (f"circular moving-block bootstrap: {n_paths:,} "
                            f"paths of {BOOT_HORIZON} months, each joined from blocks of "
                            f"{BOOT_BLOCK} consecutive months of the variant's own monthly "
                            f"history (uniform block start, wrapping at the end), "
                            f"numpy default_rng(seed={BOOT_SEED}) restarted per variant so all "
                            "variants see the same block starts; drawdowns on daily marks"),
                 "warning": ("this only re-orders the past (which holds two exceptional eras for "
                             "the Nasdaq); the future may be worse than every path here. "
                             + bootstrap_centre_text(boot, variants)
                             + haircut_fall_text(boot, boot_haircut, BOOT_SENS_HAIRCUT_PP)
                             + f" The history is {n_months // 12} years - about "
                             f"{n_months // BOOT_BLOCK} distinct {BOOT_BLOCK}-month blocks "
                             f"dominated by two bull eras; {BOOT_BLOCK}-month blocks understate "
                             "multi-year bear markets. See 'sensitivity' for longer blocks and "
                             "for a lower index return."),
                 "variants": boot,
                 "sensitivity": {
                     "label": "sensitivity - not the headline",
                     f"block_{BOOT_SENS_BLOCK}_months": {
                         "note": (f"the same bootstrap with {BOOT_SENS_BLOCK}-month blocks, which "
                                  "keep multi-year bear markets together"),
                         "variants": boot_block36},
                     "drift_haircut": {
                         "haircut_pp_per_year": BOOT_SENS_HAIRCUT_PP,
                         "note": (f"{BOOT_SENS_HAIRCUT_PP:.0f} percentage points a year are taken "
                                  "off the ^NDX daily return (spread over 252 days), the trend "
                                  "filter and the funds are re-run on that index, then the same "
                                  f"{BOOT_BLOCK}-month bootstrap"),
                         "variants": boot_haircut}}},
             "timing_sensitivity": timing,
             "conservative_financing": conservative,
             "after_tax_model_de": {
                 "label": "estimate - the headline figures are before tax",
                 "model": (f"{DE_TAX_RATE:.3%} (25% + 5.5% solidarity surcharge) of the realised "
                           "gain is withheld at every switch; realised losses are carried forward "
                           "and used up by later gains; T-bill interest is taxed at the switch "
                           "back into the fund; whatever is held at the end of the period is "
                           "sold and taxed, so buy-and-hold is taxed once at the end; each period "
                           "starts as a fresh account"),
                 "partial_exemption": (f"cagr_with_30pct_partial_exemption_pct: only "
                                       f"{1 - DE_PARTIAL_EXEMPTION:.0%} of a fund gain or loss "
                                       "counts (Teilfreistellung for equity funds; whether a "
                                       "leveraged product qualifies depends on the product)"),
                 "simplifications": ("no EUR 1,000 annual allowance, no Vorabpauschale, no church "
                                     "tax; USD figures (the tax is really due on EUR gains); a "
                                     "loss carry-forward left at the end is worth nothing"),
                 "where": "leverage.variants.<name>.periods.<period>.after_tax_estimate_de"}}
    return {"block": block, "benchmarks": bench, "curves": curves, "info": info}


def _boot_probs(b: dict | None) -> dict | None:
    """The probabilities of a bootstrap result (for the sensitivity tables)."""
    if b is None:
        return None
    return {k: b[k] for k in ("p_cagr_ge_40_pct", "p_cagr_ge_20_pct",
                              "p_max_drawdown_worse_than_50_pct",
                              "p_max_drawdown_worse_than_75_pct", "p_end_below_start_pct")} \
        | {"median_cagr_pct": b["cagr_pct"]["p50"], "block_months": b["block_months"]}


def bootstrap_centre_text(boot: dict, variants: dict) -> str:
    """What the bootstrap keeps from the history, from the numbers: every month is
    equally likely at every point of a path, so the paths keep the history's
    average monthly LOG return on average - which does not make the median
    10-year CAGR the whole-history CAGR. Quotes the variant where the two differ
    most (ties: the first in grid order)."""
    gaps = [(abs(b["cagr_pct"]["p50"] - b["full_history_cagr_pct"]), name)
            for name, b in boot.items()
            if b and b["cagr_pct"].get("p50") is not None
            and b.get("full_history_cagr_pct") is not None]
    s = ("The paths are centred on the history by construction (every month of the history is "
         "equally likely at every point of a path, which keeps the history's average monthly log "
         "return on average), so on average they are no more pessimistic than the past")
    if gaps:
        name = max(gaps, key=lambda g: g[0])[1]
        b = boot[name]
        s += ("; the median 10-year outcome can still differ from the whole-history CAGR "
              f"({b['cagr_pct']['p50']:.1f}% against {b['full_history_cagr_pct']:.1f}% for "
              f"{variants[name]['label']})")
    return s + "."


def haircut_fall_text(head: dict, cut: dict, pp_per_year: float, min_chance_pct: float = 1.0
                      ) -> str:
    """One sentence for the bootstrap warning: by how much of its value the chance
    of the target shrinks when the index returns pp_per_year less, over the
    variants whose headline chance is at least min_chance_pct. Empty when no
    variant has such a chance or when the chance does not fall for all of them
    (the sensitivity table then speaks for itself)."""
    rel = []
    for name, h in head.items():
        c = cut.get(name)
        if h and c and h["p_cagr_ge_40_pct"] >= min_chance_pct:
            rel.append(100.0 * (1.0 - c["p_cagr_ge_40_pct"] / h["p_cagr_ge_40_pct"]))
    if not rel or min(rel) < 0:
        return ""
    lo, hi = min(rel), max(rel)
    span = f"about {hi:.0f}%" if round(lo) == round(hi) else f"{lo:.0f}% to {hi:.0f}%"
    return (f" If the Nasdaq had returned {pp_per_year:.0f} points a year less, the chance of "
            f"{TARGET_CAGR_PCT:.0f}%/yr over 10 years falls by {span} of its value (the variants "
            f"with at least a {min_chance_pct:.0f}% chance).")


# --------------------------------------------------------------------------- strategy A: momentum rotation

def a_name(lookback: int, top_n: int, trend: bool) -> str:
    return f"mom{lookback}_top{top_n}_filter_{'on' if trend else 'off'}"


def a_label(lookback: int, top_n: int, trend: bool) -> str:
    months = {126: "6", 252: "12"}.get(lookback, f"{lookback}-bar")
    return (f"the top {top_n} stocks by {months}-month momentum"
            + (", in cash while the S&P 500 is below its 200-day average" if trend else ""))


@dataclass
class StockPanel:
    """Date x ticker open / close / volume of the universe (NaN = no bar)."""
    dates: np.ndarray
    tickers: list[str]
    O: np.ndarray
    C: np.ndarray
    V: np.ndarray
    valid: np.ndarray
    suspect: dict = field(default_factory=dict)
    dropped_dates: int = 0


def build_panel(long: pd.DataFrame, suspect: dict | None = None) -> StockPanel:
    """Long (ticker, date, open, close, volume) rows -> StockPanel. Dates on which
    fewer than 20% of the busiest day's tickers have a bar are dropped (stray
    bars must not become a 'month end')."""
    long = long.dropna(subset=["open", "close"])
    long = long[(long["open"] > 0) & (long["close"] > 0)].copy()
    long["date"] = long["date"].astype(str).str[:10]
    long["ticker"] = long["ticker"].astype(str)
    long = long.drop_duplicates(["date", "ticker"], keep="last")
    piv = {f: long.pivot(index="date", columns="ticker", values=f)
           for f in ("open", "close", "volume")}
    O = piv["open"].to_numpy(float)
    C = piv["close"].to_numpy(float)
    V = piv["volume"].to_numpy(float)
    dates = np.asarray(piv["open"].index, dtype=str)
    valid = np.isfinite(O) & np.isfinite(C)
    counts = valid.sum(axis=1)
    keep = counts >= 0.2 * counts.max() if len(counts) else np.zeros(0, bool)
    dropped = int((~keep).sum())
    if dropped:
        O, C, V, valid, dates = O[keep], C[keep], V[keep], valid[keep], dates[keep]
    return StockPanel(dates=dates, tickers=[str(t) for t in piv["open"].columns], O=O, C=C, V=V,
                      valid=valid, suspect=dict(suspect or {}), dropped_dates=dropped)


@dataclass
class SignalData:
    """Per signal day (row) and ticker (column): momentum per look-back (NaN = not
    available) and eligibility (bar that day, enough history, liquid, no
    unexplained price jump in the look-back)."""
    idx: np.ndarray
    mom: dict
    eligible: np.ndarray


def momentum_signals(panel: StockPanel, signal_idx: np.ndarray, lookbacks=A_LOOKBACKS,
                     skip: int = A_SKIP, min_prior: int = A_MIN_PRIOR_BARS,
                     dv_window: int = A_DV_WINDOW,
                     min_dollar_vol: float = MIN_DOLLAR_VOL[A_MARKET]) -> SignalData:
    """Momentum and eligibility on the signal days, from each stock's OWN bars up
    to and including the signal day only:
      momentum(L) = close[i - skip] / close[i - skip - L] - 1   (i = the signal bar)
      eligible    = a bar on the signal day, at least min_prior earlier bars,
                    mean(close * volume) of the last dv_window bars >= min_dollar_vol."""
    signal_idx = np.asarray(signal_idx, int)
    n_s, n_t = len(signal_idx), len(panel.tickers)
    mom = {L: np.full((n_s, n_t), np.nan) for L in lookbacks}
    elig = np.zeros((n_s, n_t), bool)
    date_pos = {d: k for k, d in enumerate(panel.dates)}
    for k, ticker in enumerate(panel.tickers):
        rows = np.nonzero(panel.valid[:, k])[0]
        if len(rows) <= min_prior or not n_s:
            continue
        pos = np.searchsorted(rows, signal_idx)
        has = (pos < len(rows)) & (rows[np.minimum(pos, len(rows) - 1)] == signal_idx)
        js = np.nonzero(has & (pos >= min_prior))[0]
        if not len(js):
            continue
        i = pos[js]
        c = panel.C[rows, k]
        cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(c * panel.V[rows, k]))])
        ok = (i >= dv_window - 1) & (
            (cs[i + 1] - cs[np.maximum(i + 1 - dv_window, 0)]) / dv_window >= min_dollar_vol)
        sus = [date_pos[d] for d in panel.suspect.get(ticker, ()) if d in date_pos]
        if sus:
            p = np.searchsorted(rows, np.asarray(sus))
            ok &= ~((i[:, None] >= p[None, :]) & (i[:, None] <= p[None, :] + min_prior)).any(axis=1)
        elig[js, k] = ok
        for L in lookbacks:
            need = i >= skip + L
            m = np.full(len(i), np.nan)
            m[need] = c[i[need] - skip] / c[i[need] - skip - L] - 1
            mom[L][js, k] = m
    return SignalData(idx=signal_idx, mom=mom, eligible=elig)


def select_top(sig: SignalData, j: int, lookback: int, top_n: int | None) -> np.ndarray:
    """Columns of the top_n eligible stocks by momentum on signal row j (highest
    first; ties by ticker order). top_n None = every eligible stock."""
    m = sig.mom[lookback][j]
    cols = np.nonzero(sig.eligible[j] & np.isfinite(m))[0]
    order = cols[np.argsort(-m[cols], kind="stable")]
    return order if top_n is None else order[:top_n]


def select_bottom(sig: SignalData, j: int, lookback: int, top_n: int) -> np.ndarray:
    """Placebo: the top_n eligible stocks with the LOWEST momentum on signal row j."""
    m = sig.mom[lookback][j]
    cols = np.nonzero(sig.eligible[j] & np.isfinite(m))[0]
    return cols[np.argsort(m[cols], kind="stable")][:top_n]


def select_random(sig: SignalData, j: int, lookback: int, top_n: int, seed: int) -> np.ndarray:
    """Placebo: top_n eligible stocks drawn at random on signal row j (the draw
    depends only on (seed, j), so it is the same whatever was held before)."""
    cols = np.nonzero(sig.eligible[j] & np.isfinite(sig.mom[lookback][j]))[0]
    if len(cols) <= top_n:
        return cols
    return np.sort(np.random.default_rng([int(seed), int(j)]).choice(cols, size=top_n,
                                                                     replace=False))


def without_ticker(sig: SignalData, col: int) -> SignalData:
    """The same signals with one stock never eligible."""
    elig = sig.eligible.copy()
    elig[:, col] = False
    return SignalData(idx=sig.idx, mom=sig.mom, eligible=elig)


def solve_rebalance(value: float, holdings: np.ndarray, weights: np.ndarray, cost_rate: float
                    ) -> tuple[float, float]:
    """Equity after costs when moving `holdings` (currency per stock) to
    `weights` of the post-cost equity, costs = cost_rate * traded notional.
    Returns (equity_after, traded_notional); solved by fixed-point iteration."""
    v = float(value)
    traded = float(np.abs(weights * v - holdings).sum())
    for _ in range(200):
        v2 = value - cost_rate * traded
        traded = float(np.abs(weights * v2 - holdings).sum())
        if abs(v2 - v) <= 1e-14 * max(value, 1e-300):
            v = v2
            break
        v = v2
    return v, traded


def run_momentum(panel: StockPanel, sig: SignalData, *, lookback: int, top_n: int | None,
                 trend_ok: np.ndarray | None = None, rf_daily: np.ndarray | None = None,
                 cost_rate: float = A_COST_RATE, first_signal: int = 0,
                 i_end: int | None = None, selector=None) -> dict:
    """One account, starting with 1.0 in cash at the close of signal day
    sig.idx[first_signal]. For a signal at the close of day s the trades are
    done at the OPEN of day s+1: stocks leaving the list are sold, the list is
    re-weighted to equal weight of the post-cost equity, and cost_rate is
    charged on the traded notional only. trend_ok[j] False = all cash until the
    next signal. Cash earns rf_daily on the balance held over the whole day
    (the smaller of the previous close's and the post-trade balance). Marked at
    every close (a stock without a bar keeps its last close). Returns
    {equity (NaN before the start), i_start, i_end, log, fills_without_bar,
    contrib}; contrib[ticker] = sum over its holding months of the position's
    profit as a fraction of the equity at that rebalance (not compounded).
    selector(j) -> columns replaces the momentum ranking (placebo runs only)."""
    n = len(panel.dates)
    n_t = len(panel.tickers)
    i_start = int(sig.idx[first_signal])
    i_end = n - 1 if i_end is None else int(i_end)
    sig_row = {int(s): j for j, s in enumerate(sig.idx) if j >= first_signal}
    rf = np.zeros(n) if rf_daily is None else np.asarray(rf_daily, float)
    shares = np.zeros(n_t)
    last_close = np.zeros(n_t)
    seen = np.isfinite(panel.C[:i_start + 1])
    for k in np.nonzero(seen.any(axis=0))[0]:
        last_close[k] = panel.C[np.nonzero(seen[:, k])[0][-1], k]
    cash = 1.0
    held = np.zeros(0, int)
    equity = np.full(n, np.nan)
    equity[i_start] = 1.0
    log: list[dict] = []
    no_bar = 0
    alloc = np.zeros(n_t)          # value given to each stock at the last rebalance
    alloc_equity = 1.0
    contrib: dict[str, float] = {}

    def book(cols_: np.ndarray, values: np.ndarray) -> None:
        for k_, v_ in zip(cols_, values):
            t_ = panel.tickers[int(k_)]
            contrib[t_] = contrib.get(t_, 0.0) + float(v_ - alloc[k_]) / alloc_equity

    for d in range(i_start + 1, i_end + 1):
        cash_before = cash
        j = sig_row.get(d - 1)
        if j is not None:
            if trend_ok is not None and not bool(trend_ok[j]):
                target = np.zeros(0, int)
            elif selector is not None:
                target = np.asarray(selector(j), int)
            else:
                target = select_top(sig, j, lookback, top_n)
            cols = np.union1d(held, target).astype(int)
            px = panel.O[d, cols]
            miss = ~np.isfinite(px)
            if miss.any():
                px = np.where(miss, last_close[cols], px)
                no_bar += int((miss & ((shares[cols] > 0) | np.isin(cols, target))).sum())
            h = shares[cols] * px
            value = cash + float(h.sum())
            was = np.isin(cols, held)
            book(cols[was], h[was])
            w = np.where(np.isin(cols, target), 1.0 / len(target), 0.0) if len(target) \
                else np.zeros(len(cols))
            after, traded = solve_rebalance(value, h, w, cost_rate)
            tgt = w * after
            buys = float(np.maximum(tgt - h, 0).sum())
            sells = float(np.maximum(h - tgt, 0).sum())
            shares[:] = 0.0
            if len(cols):
                shares[cols] = np.divide(tgt, px, out=np.zeros(len(cols)), where=px > 0)
            cash = 0.0 if len(target) else after
            held = np.asarray(target, int)
            alloc[:] = 0.0
            if len(cols):
                alloc[cols] = tgt
            alloc_equity = after if after > 0 else 1.0
            log.append({
                "signal_date": str(panel.dates[d - 1]), "trade_date": str(panel.dates[d]),
                "trade_idx": d, "invested": bool(len(target)), "n_held": int(len(target)),
                "traded_notional_pct": 100.0 * traded / value if value > 0 else 0.0,
                "turnover_one_way_pct": 100.0 * max(buys, sells) / value if value > 0 else 0.0,
                "cost_pct": 100.0 * (value - after) / value if value > 0 else 0.0,
                "tickers": [panel.tickers[int(k)] for k in target],
            })
        cash += min(cash_before, cash) * rf[d]
        cd = panel.C[d]
        last_close = np.where(np.isfinite(cd), cd, last_close)
        equity[d] = cash + float((shares[held] * last_close[held]).sum())
    book(held, shares[held] * last_close[held])
    return {"equity": equity, "i_start": i_start, "i_end": i_end, "log": log,
            "fills_without_bar": no_bar, "contrib": contrib}


def equal_weight_buy_and_hold(panel: StockPanel, i_start: int, i_end: int
                              ) -> tuple[np.ndarray, int]:
    """Equal amounts in every stock with a bar at the close of i_start, then never
    touched again (no rebalancing, no costs); a stock without a bar keeps its
    last close. An investable basket, unlike the daily-rebalanced one. Returns
    (equity, NaN outside i_start..i_end; number of stocks bought)."""
    eq = np.full(len(panel.dates), np.nan)
    cols = np.nonzero(panel.valid[i_start])[0]
    if not len(cols) or i_end < i_start:
        return eq, 0
    c = pd.DataFrame(panel.C[i_start:i_end + 1][:, cols]).ffill().to_numpy()
    eq[i_start:i_end + 1] = (c / c[0]).mean(axis=1)
    return eq, int(len(cols))


def equal_weight_returns(panel: StockPanel, clip: float = DAILY_RET_CLIP
                         ) -> tuple[np.ndarray, int]:
    """Daily return of the DAILY-REBALANCED equal-weight basket of every stock
    with a bar (the backtest's convention: each stock's own close-to-close
    return, single-stock moves capped at +/-clip, unexplained jumps masked, no
    costs). Not investable - rebalancing every day for free earns a premium a
    real basket does not. Returns (returns per date, capped stock-days)."""
    R = np.full(panel.C.shape, np.nan)
    date_pos = {d: k for k, d in enumerate(panel.dates)}
    for k, ticker in enumerate(panel.tickers):
        rows = np.nonzero(panel.valid[:, k])[0]
        if len(rows) < 2:
            continue
        c = panel.C[rows, k]
        R[rows[1:], k] = c[1:] / c[:-1] - 1
        for d in panel.suspect.get(ticker, ()):
            if d in date_pos:
                R[date_pos[d], k] = np.nan
    clipped = int((np.abs(R) > clip).sum())
    R = np.clip(R, -clip, clip)
    cnt = np.isfinite(R).sum(axis=1)
    ew = np.where(cnt > 0, np.nansum(R, axis=1) / np.maximum(cnt, 1), 0.0)
    return ew, clipped


def curve_from_returns(ret: np.ndarray, i_start: int, i_end: int) -> np.ndarray:
    eq = np.full(len(ret), np.nan)
    eq[i_start] = 1.0
    eq[i_start + 1:i_end + 1] = np.cumprod(1.0 + ret[i_start + 1:i_end + 1])
    return eq


def momentum_period(dates: np.ndarray, res: dict, i0: int, i1: int) -> dict:
    """perf_metrics plus the trading statistics of the rebalances inside (i0, i1]."""
    m = perf_metrics(dates, res["equity"], i0, i1)
    rows = [r for r in res["log"] if i0 < r["trade_idx"] <= i1]
    m["months"] = len(rows)
    if rows:
        m["avg_turnover_pct_per_month"] = _rnd(float(np.mean(
            [r["turnover_one_way_pct"] for r in rows])), 1)
        m["avg_traded_notional_pct_per_month"] = _rnd(float(np.mean(
            [r["traded_notional_pct"] for r in rows])), 1)
        m["avg_cost_pct_per_month"] = _rnd(float(np.mean([r["cost_pct"] for r in rows])), 3)
        m["months_invested_pct"] = _rnd(100.0 * float(np.mean([r["invested"] for r in rows])), 1)
        m["distinct_stocks_held"] = len({t for r in rows for t in r["tickers"]})
    else:
        m.update({"avg_turnover_pct_per_month": None, "avg_traded_notional_pct_per_month": None,
                  "avg_cost_pct_per_month": None, "months_invested_pct": None,
                  "distinct_stocks_held": 0})
    return m


def concentration(dates: np.ndarray, res: dict, i0: int, i1: int, top: int = 5) -> dict | None:
    """How much of a variant's whole-run result hangs on one month or a few
    stocks. A diagnostic (hindsight), not one of the pre-registered metrics."""
    ms = monthly_stats(dates, res["equity"], i0, i1)
    eq = res["equity"]
    if ms is None or not len(ms["ret"]) or not eq[i0] > 0:
        return None
    k = int(np.argmax(ms["ret"]))
    days = _days([dates[i0], dates[i1]])
    years = (days[1] - days[0]) / 365.25
    growth = eq[i1] / eq[i0]
    without = growth / (1.0 + float(ms["ret"][k]))
    held_months: dict[str, int] = {}
    for r in res["log"]:
        for t in r["tickers"]:
            held_months[t] = held_months.get(t, 0) + 1
    ranked = sorted(res.get("contrib", {}).items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return {
        "note": "diagnostic, not a pre-registered metric: the best calendar month, the CAGR "
                "with that one month's return set to zero, and the stocks that added most "
                "(sum of each holding month's profit in % of the equity at that rebalance)",
        "best_month": {"month": ms["months"][k], "return_pct": _pct(ms["ret"][k])},
        "cagr_without_best_month_pct": (_pct(without ** (1 / years) - 1)
                                        if years > 0 and without > 0 else None),
        "top_contributors": [{"ticker": t, "months_held": held_months.get(t, 0),
                              "contribution_pp": _rnd(100.0 * v, 1)} for t, v in ranked],
    }


def index_trend_flags(index_dates: np.ndarray, index_close: np.ndarray,
                      signal_dates: np.ndarray, window: int = A_SMA
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(above, defined) per signal day: is the index's close of that day (its last
    bar on or before it) above its own `window`-day average of the same day?
    defined is False while the average does not exist yet (above is then False)."""
    close = np.asarray(index_close, float)
    sma = rolling_mean(close, window)
    pos = np.searchsorted(index_dates, signal_dates, side="right") - 1
    k = np.maximum(pos, 0)
    defined = (pos >= 0) & np.isfinite(sma[k])
    return np.where(defined, close[k] > sma[k], False), defined


def load_universe(path: Path) -> list[str]:
    return [str(t) for t in pd.read_csv(path)["ticker"].dropna().tolist()]


def load_stock_panel(prices_path: Path, tickers: list[str], cfg: dict | None,
                     data_dir: Path | None, progress=print) -> tuple[StockPanel, dict]:
    """The universe's bars from the local price cache (opened READ-ONLY), with the
    backtest's price repairs (sanitize_bars with the same overrides / stale map)."""
    prices_path = Path(prices_path)
    conn = _connect_ro(prices_path)
    if conn is None:
        raise FileNotFoundError(f"price cache not found: {prices_path}")
    try:
        overrides, stale = price_repairs(cfg, data_dir, conn)
        long = load_long_prices(SimpleNamespace(conn=conn), tickers)
    finally:
        conn.close()
    if long.empty:
        raise ValueError("no universe prices in the price cache")
    long, rep = sanitize_long(long, overrides, stale)
    panel = build_panel(long, rep.get("suspect"))
    info = {"path": str(prices_path),
            "file_modified": datetime.fromtimestamp(prices_path.stat().st_mtime
                                                    ).isoformat(timespec="seconds"),
            "opened": "read-only", "universe_tickers": len(tickers),
            "tickers_with_bars": len(panel.tickers),
            "first_bar": str(panel.dates[0]), "last_bar": str(panel.dates[-1]),
            "trading_days": int(len(panel.dates)), "stray_dates_dropped": panel.dropped_dates,
            "price_repairs": dict(rep.get("counts") or {}),
            "price_repair_events": list(rep.get("events") or [])[:200],
            "unresolved_jumps": {t: list(ds) for t, ds in (rep.get("suspect") or {}).items()}}
    return panel, info


def momentum_block(panel: StockPanel, frames: dict[str, pd.DataFrame], *,
                   n_placebo: int = A_PLACEBO_SEEDS, progress=print) -> dict:
    """Strategy A on the stock panel. Returns {block, benchmarks, curves, info}."""
    dates = panel.dates
    n = len(dates)
    signal_idx = month_end_signal_idx(dates)
    sig = momentum_signals(panel, signal_idx)
    strict = sig.eligible & np.isfinite(sig.mom[max(A_LOOKBACKS)])
    n_elig = strict.sum(axis=1)

    def series_on(ticker: str, col: str) -> np.ndarray:
        f = frames[ticker]
        return asof_values(np.asarray(f.index, dtype=str), f[col].to_numpy(float), dates)

    # ^GSPC trend flag at each signal close (on the index's own bars)
    g = frames["^GSPC"]
    trend_ok, g_def = index_trend_flags(np.asarray(g.index, dtype=str),
                                        g["close"].to_numpy(float), dates[signal_idx])

    qqq, spy = series_on("QQQ", "adj_close"), series_on("SPY", "adj_close")
    irx = series_on("^IRX", "close")
    usable = (n_elig >= max(A_TOP_N)) & g_def & np.isfinite(qqq[signal_idx]) \
        & np.isfinite(spy[signal_idx]) & np.isfinite(irx[signal_idx])
    if not usable.any():
        raise ValueError(f"no month end with at least {max(A_TOP_N)} eligible stocks and index data")
    j0 = int(np.nonzero(usable)[0][0])
    i_start = int(signal_idx[j0])

    # the last full month that every series covers
    limit = min(str(frames[t].index[-1]) for t in ("QQQ", "SPY", "^IRX", "^GSPC"))
    ends = [int(i) for i in signal_idx] + ([n - 1] if month_complete(dates, n - 1) else [])
    ends = [i for i in ends if str(dates[i]) <= limit and i > i_start]
    if not ends:
        raise ValueError("no full month after the first signal (is the series cache stale? "
                         "run with --refresh)")
    i_end = max(ends)
    all_ends = [int(i) for i in signal_idx] + ([n - 1] if month_complete(dates, n - 1) else [])
    truncated = i_end < max(all_ends)
    if truncated:
        progress(f"WARN [A] the series cache ends {limit}, before the price cache's last full "
                 f"month ({dates[max(all_ends)]}): Strategy A stops at {dates[i_end]} - run with "
                 "--refresh to extend it")
    i_split = _last_on_or_before(dates, A_TRAIN_END)
    i_split = min(max(i_split, i_start), i_end)
    spans = {"train": (i_start, i_split), "test": (i_split, i_end), "pooled": (i_start, i_end)}
    rf = irx / 100.0 / TRADING_DAYS
    progress(f"[A] {len(panel.tickers)} stocks, {dates[0]} .. {dates[-1]}; first signal "
             f"{dates[i_start]} ({int(n_elig[j0])} eligible), last full month ends "
             f"{dates[i_end]}; train to {dates[i_split]}, test from there")

    # benchmarks over the same days
    ew_ret, clipped = equal_weight_returns(panel)
    bh_eq, bh_n = equal_weight_buy_and_hold(panel, i_start, i_end)
    bench_eq = {
        "QQQ": qqq / qqq[i_start], "SPY": spy / spy[i_start],
        EW_BH: bh_eq,
        EW_DAILY: curve_from_returns(ew_ret, i_start, i_end),
    }
    elig_run = run_momentum(panel, sig, lookback=max(A_LOOKBACKS), top_n=None, rf_daily=rf,
                            cost_rate=0.0, first_signal=j0, i_end=i_end)
    bench_eq[EW_ELIGIBLE] = elig_run["equity"]
    bench = {p: {k: perf_metrics(dates, eq, *spans[p]) for k, eq in bench_eq.items()}
             for p in PERIODS_A}
    # the same basket bought fresh at the start of TEST (the continuous one has drifted)
    bh_test_eq, bh_test_n = equal_weight_buy_and_hold(panel, i_split, i_end)
    bench["test"][EW_BH_FRESH] = perf_metrics(dates, bh_test_eq, i_split, i_end)

    def period_metrics(res: dict) -> dict:
        out = {}
        for p in PERIODS_A:
            m = momentum_period(dates, res, *spans[p])
            c = m["cagr_pct"]
            for key, bk in (("cagr_minus_qqq_pp", "QQQ"), ("cagr_minus_spy_pp", "SPY"),
                            ("cagr_minus_equal_weight_buy_and_hold_pp", EW_BH),
                            ("cagr_minus_equal_weight_daily_rebalanced_pp", EW_DAILY),
                            ("cagr_minus_equal_weight_eligible_pp", EW_ELIGIBLE)):
                b = bench[p][bk]["cagr_pct"]
                m[key] = _r2(c - b) if c is not None and b is not None else None
            out[p] = m
        return out

    variants: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    for lookback in A_LOOKBACKS:
        for top_n in A_TOP_N:
            for trend in A_FILTERS:
                name = a_name(lookback, top_n, trend)
                res = run_momentum(panel, sig, lookback=lookback, top_n=top_n,
                                   trend_ok=trend_ok if trend else None, rf_daily=rf,
                                   first_signal=j0, i_end=i_end)
                runs[name] = res
                variants[name] = {
                    "lookback_bars": lookback, "top_n": top_n, "trend_filter": trend,
                    "label": a_label(lookback, top_n, trend), "periods": period_metrics(res),
                    "fills_without_bar": res["fills_without_bar"],
                    "concentration": concentration(dates, res, i_start, i_end)}

    pick = pick_winner(variants, A_TRAIN_DD_FLOOR_PCT)
    wname = pick["name"]
    best_test = best_on(variants, "test")
    winner = {
        "rule": (f"highest TRAIN CAGR (first signal to {A_TRAIN_END}) among the variants whose "
                 f"TRAIN max drawdown is no worse than {A_TRAIN_DD_FLOOR_PCT:.0f}%"),
        "name": wname, "label": variants[wname]["label"] if wname else None,
        "qualified": pick["qualified"], "disqualified": pick["disqualified"],
        "selection_margin": selection_margin(variants, pick, A_TRAIN_DD_FLOOR_PCT),
        "best_test_variant": {"name": best_test, "label": IN_SAMPLE_LABEL,
                              "test_cagr_pct": variants[best_test]["periods"]["test"]["cagr_pct"]
                              if best_test else None},
    }
    placebo = None
    if wname:
        wv = variants[wname]
        w_args = dict(lookback=wv["lookback_bars"], top_n=wv["top_n"],
                      trend_ok=trend_ok if wv["trend_filter"] else None, rf_daily=rf,
                      first_signal=j0, i_end=i_end)
        for p in PERIODS_A:
            w = wv["periods"][p]
            winner[p] = {k: w.get(k) for k in (
                "start", "end", "years", "months", "cagr_pct", "total_return_pct",
                "max_drawdown_pct", "worst_year", "ann_volatility_pct",
                "avg_turnover_pct_per_month", "months_invested_pct", "distinct_stocks_held",
                "cagr_minus_qqq_pp", "cagr_minus_equal_weight_buy_and_hold_pp",
                "cagr_minus_equal_weight_daily_rebalanced_pp",
                "cagr_minus_equal_weight_eligible_pp")}
            winner[p]["qqq_cagr_pct"] = bench[p]["QQQ"]["cagr_pct"]
            winner[p]["spy_cagr_pct"] = bench[p]["SPY"]["cagr_pct"]
            winner[p]["equal_weight_buy_and_hold_cagr_pct"] = bench[p][EW_BH]["cagr_pct"]
            winner[p]["equal_weight_daily_rebalanced_cagr_pct"] = bench[p][EW_DAILY]["cagr_pct"]
            winner[p]["qqq_max_drawdown_pct"] = bench[p]["QQQ"]["max_drawdown_pct"]
        winner["test"]["equal_weight_buy_and_hold_from_test_start_cagr_pct"] = \
            bench["test"][EW_BH_FRESH]["cagr_pct"]
        # how noisy the TEST figure is: one standard error of a CAGR ~ volatility / sqrt(years)
        t = wv["periods"]["test"]
        if t.get("ann_volatility_pct") is not None and t.get("years"):
            winner["test_noise"] = {
                "note": "rough: annual volatility / sqrt(years) = one standard error of the "
                        "TEST-period yearly return; differences of this size between variants "
                        "cannot be told from chance",
                "ann_volatility_pct": t["ann_volatility_pct"], "years": t["years"],
                "standard_error_pp": _rnd(t["ann_volatility_pct"] / math.sqrt(t["years"]), 1)}
        # the winner re-run without its largest contributor (diagnostic)
        conc = wv["concentration"]
        if conc and conc.get("top_contributors"):
            top_t = conc["top_contributors"][0]["ticker"]
            ex = run_momentum(panel, without_ticker(sig, panel.tickers.index(top_t)), **w_args)
            conc["cagr_without_top_contributor_pct"] = {
                "ticker": top_t,
                **{p: perf_metrics(dates, ex["equity"], *spans[p])["cagr_pct"] for p in PERIODS_A}}
        winner["concentration"] = conc
        # placebos with the winner's own rules, engine and costs (diagnostic)
        lb, tn = wv["lookback_bars"], wv["top_n"]
        bottom = run_momentum(panel, sig, selector=lambda j: select_bottom(sig, j, lb, tn),
                              **w_args)
        rand = {p: [] for p in PERIODS_A}
        for seed in range(n_placebo):
            r = run_momentum(panel, sig, selector=lambda j, s_=seed: select_random(
                sig, j, lb, tn, s_), **w_args)
            for p in PERIODS_A:
                rand[p].append(perf_metrics(dates, r["equity"], *spans[p])["cagr_exact_pct"])

        def rand_row(p: str) -> dict | None:
            x = np.asarray([v for v in rand[p] if v is not None], float)
            if not len(x):
                return None
            wc = exact(wv["periods"][p], "cagr_pct")
            q = np.percentile(x, [5, 50, 95])
            return {"p5_cagr_pct": _r2(q[0]), "median_cagr_pct": _r2(q[1]),
                    "p95_cagr_pct": _r2(q[2]), "max_cagr_pct": _r2(x.max()),
                    "winner_cagr_pct": wv["periods"][p]["cagr_pct"],
                    "winner_percentile": (_rnd(100.0 * float((x < wc).mean()), 1)
                                          if wc is not None else None)}

        placebo = {
            "label": "diagnostic, not a pre-registered metric",
            "note": (f"the winner's rules ({wv['label']}), engine and costs with a different "
                     f"stock picker: bottom_n holds the {tn} eligible stocks with the LOWEST "
                     f"momentum, random_n holds {tn} eligible stocks drawn at random each month "
                     f"({n_placebo} fixed seeds). They share the stock list and so its "
                     "survivorship bias"),
            "bottom_n": {p: {k: perf_metrics(dates, bottom["equity"], *spans[p])[k]
                             for k in ("cagr_pct", "max_drawdown_pct")} for p in PERIODS_A},
            "random_n": {"seeds": int(n_placebo), **{p: rand_row(p) for p in PERIODS_A}},
        }
    else:
        winner["note"] = "no variant kept its TRAIN drawdown inside the limit - no winner"

    # survivorship: how much the universe itself beats the index
    def gap(p: str, a: str, b: str):
        x, y = bench[p][a]["cagr_pct"], bench[p][b]["cagr_pct"]
        return _r2(x - y) if x is not None and y is not None else None

    # how much the liquidity rule excludes (eligibility with and without it)
    no_liq = momentum_signals(panel, signal_idx, min_dollar_vol=0.0).eligible
    last_j = int(np.nonzero(signal_idx < i_end)[0][-1]) if (signal_idx < i_end).any() else j0
    liquidity = {
        "signal_date": str(dates[signal_idx[last_j]]),
        "stocks_with_a_bar": int(panel.valid[signal_idx[last_j]].sum()),
        "eligible_without_liquidity_rule": int(no_liq[last_j].sum()),
        "eligible": int(sig.eligible[last_j].sum()),
        "excluded_by_liquidity_rule": int((no_liq[last_j] & ~sig.eligible[last_j]).sum()),
        "max_excluded_at_any_signal": int((no_liq[j0:] & ~sig.eligible[j0:]).sum(axis=1).max()),
    }

    surv = {
        "universe": (f"{len(panel.tickers)} of TODAY's S&P 500 + S&P 400 members "
                     "(universe/us.csv); stocks that left the indices or were delisted are missing, "
                     "and stocks that grew into them are present from before they qualified"),
        "equal_weight_buy_and_hold": {
            "definition": ("equal amounts in every stock with a bar at the first signal's close, "
                           "never rebalanced, no costs, a stock without a bar keeps its last "
                           "close - an investable basket; train / test / pooled are periods of "
                           "that one account"),
            "stocks": bh_n,
            "cagr_pct": {p: bench[p][EW_BH]["cagr_pct"] for p in PERIODS_A},
            "bought_at_test_start": {
                "anchor": str(dates[i_split]), "stocks": bh_test_n,
                "cagr_pct": bench["test"][EW_BH_FRESH]["cagr_pct"],
                "minus_spy_pp": gap("test", EW_BH_FRESH, "SPY"),
                "minus_qqq_pp": gap("test", EW_BH_FRESH, "QQQ")}},
        "equal_weight_universe_daily_rebalanced_cagr_pct": {
            "label": "not investable (cost-free daily rebalancing)",
            **{p: bench[p][EW_DAILY]["cagr_pct"] for p in PERIODS_A}},
        "equal_weight_eligible_monthly_cagr_pct": {
            p: bench[p][EW_ELIGIBLE]["cagr_pct"] for p in PERIODS_A},
        "spy_cagr_pct": {p: bench[p]["SPY"]["cagr_pct"] for p in PERIODS_A},
        "qqq_cagr_pct": {p: bench[p]["QQQ"]["cagr_pct"] for p in PERIODS_A},
        "universe_minus_spy_pp": {p: gap(p, EW_BH, "SPY") for p in PERIODS_A},
        "universe_minus_qqq_pp": {p: gap(p, EW_BH, "QQQ") for p in PERIODS_A},
        "universe_gap_basis": "equal_weight_buy_and_hold (the investable basket)",
        "bias_reduced_measure": ("cagr_minus_equal_weight_buy_and_hold_pp (and the stricter "
                                 "cagr_minus_equal_weight_daily_rebalanced_pp / "
                                 "cagr_minus_equal_weight_eligible_pp): the baskets share the "
                                 "survivorship bias, so the excess over them is closer to skill - "
                                 "but momentum picks exactly the stocks whose later index "
                                 "membership the list already knows, so some bias remains"),
        "clipped_stock_days": clipped,
        "liquidity_rule": liquidity,
        "placebo": placebo,
    }
    gp = surv["universe_minus_spy_pp"]["pooled"]
    surv["caveat"] = (
        "SURVIVORSHIP BIAS: the absolute CAGR of every momentum variant is biased UPWARD. The "
        "stock list is today's index members, so yesterday's winners are in it because they won. "
        + (f"Bought in equal amounts at the start and simply held, the same list made "
           f"{bench['pooled'][EW_BH]['cagr_pct']:.1f}%/yr against "
           f"{bench['pooled']['SPY']['cagr_pct']:.1f}%/yr for SPY ({gp:+.1f} pct-pt/yr) and "
           f"{bench['pooled']['QQQ']['cagr_pct']:.1f}%/yr for QQQ over "
           f"{bench['pooled']['SPY']['start']} .. {bench['pooled']['SPY']['end']}. "
           if gp is not None else "")
        + "That gap is NOT the size of the bias for a momentum rule: it is the bias of holding "
        "the whole list, mixed with the equal-weight / mid-cap tilt. A rule that buys the "
        "10-20 strongest stocks concentrates on exactly the names that later earned their index "
        "place by rising, so its bias is far larger and cannot be measured with this data (no "
        "historical index membership, no delisted stocks). Even the excess over the "
        "equal-weight basket is therefore only bias-REDUCED, not bias-free. Do not read the "
        "absolute CAGR of Strategy A as achievable.")

    idx = np.arange(i_start, i_end + 1)
    curves = {"frequency": "daily", "dates": [str(d) for d in dates[idx]],
              "series": {name: [float(f"{x:.6g}") for x in runs[name]["equity"][idx]]
                         for name in runs}}
    for k, eq in bench_eq.items():
        curves["series"][k] = [float(f"{x:.6g}") for x in eq[idx]]

    info = {"first_signal": str(dates[i_start]), "last_day": str(dates[i_end]),
            "train": [str(dates[i_start]), str(dates[i_split])],
            "test": [str(dates[i_split]), str(dates[i_end])],
            "signals": int(((signal_idx >= i_start) & (signal_idx < i_end)).sum()),
            "eligible_at_first_signal": int(n_elig[j0]),
            "eligible_median": int(np.median(n_elig[j0:])),
            "buy_and_hold_stocks": bh_n, "turnover_range_pct_per_month": [
                min(v["periods"]["pooled"]["avg_turnover_pct_per_month"] for v in variants.values()),
                max(v["periods"]["pooled"]["avg_turnover_pct_per_month"] for v in variants.values())],
            "series_limit": limit, "shortened_by_series_cache": bool(truncated),
            "fills_without_bar": {k: r["fills_without_bar"] for k, r in runs.items()}}
    block = {"variants": variants, "winner": winner, "survivorship": surv}
    return {"block": block, "benchmarks": bench, "curves": curves, "info": info}


# --------------------------------------------------------------------------- answer, method, caveats

MEDALLION_CONTEXT = (
    "For context (not a lab result): the roughly 39% a year after fees (66% before fees) "
    "reported for Jim Simons' Medallion fund - the owner's reference for the 40% goal - came "
    "from a capacity-limited, short-horizon, largely market-neutral trading operation; a "
    "long-only rule on daily prices does not replicate it.")


def _fmt(x, nd: int = 1, signed: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def _yr(d) -> str:
    return str(d)[:4] if d else "?"


def _span(m: dict) -> str:
    """'2010-2026' from a period's yearly table (the anchor day is not a year of it)."""
    ys = sorted(m.get("yearly_pct") or {})
    return f"{ys[0]}-{ys[-1]}" if ys else "?"


def _years_text(years: list[str]) -> str:
    """['1989', '1990', '2011'] -> '1989-1990 or 2011' (runs of consecutive years)."""
    ys = sorted({int(y) for y in years})
    runs: list[list[int]] = []
    for y in ys:
        if runs and y == runs[-1][-1] + 1:
            runs[-1].append(y)
        else:
            runs.append([y])
    txt = [str(r[0]) if len(r) == 1 else f"{r[0]}-{r[-1]}" for r in runs]
    return txt[0] if len(txt) == 1 else ", ".join(txt[:-1]) + " or " + txt[-1]


def _count_word(n: int) -> str:
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    return words[n] if 0 <= n < len(words) else str(n)


def _cagr_txt(m: dict | None) -> str:
    """A period's CAGR for the answer text, from the unrounded figure (18.746 -> '18.7')."""
    return _fmt(exact(m, "cagr_pct"))


def _dd_txt(m: dict | None) -> str:
    """A period's max drawdown for the answer text, whole percent, from the unrounded figure."""
    return _fmt(exact(m, "max_drawdown_pct"), 0)


def _leverage_vs_1x_text(w: dict, b1: dict, win_test: dict) -> str:
    """The chosen leveraged rule against the modelled 1x fund: the TEST return and the
    TEST drawdown in one sentence (with the real QQQ when the winner block carries it),
    then the whole history. The conjunctions follow the unrounded numbers."""
    def better(key: str, p: str) -> bool | None:
        a, b = exact(w[p], key), exact(b1[p], key)
        return None if a is None or b is None else a > b

    te, fu = w["test"], w["full"]
    ret_up, dd_up = better("cagr_pct", "test"), better("max_drawdown_pct", "test")
    qqq = exact(win_test, "qqq_real_cagr_pct")
    s = (f"In {_span(te)}, years it was not chosen on, it made {_cagr_txt(te)}% a year against "
         f"{_cagr_txt(b1['test'])}% for the modelled 1x Nasdaq-100 fund"
         + (f" and {_fmt(qqq)}% for the real QQQ fund" if qqq is not None else "")
         + (", but" if None not in (ret_up, dd_up) and ret_up != dd_up else ", and")
         + f" its worst fall in those years was {_dd_txt(te)}% against {_dd_txt(b1['test'])}% "
         f"for the 1x fund. Over the whole {_span(fu)} history it made {_cagr_txt(fu)}% a year "
         f"against {_cagr_txt(b1['full'])}%, and its worst fall was {_dd_txt(fu)}% (1x fund: "
         f"{_dd_txt(b1['full'])}%)")
    if better("max_drawdown_pct", "full") and dd_up is False:
        s += f" - an advantage earned in {_span(w['train'])}, the years the rule was chosen on"
    return s + "."


def _reaches(variants: dict, winner: str | None, full_key: str) -> dict:
    """Which variants made the target, on the UNROUNDED CAGR (39.996% is not 40%)."""
    t = TARGET_CAGR_PCT

    def ge(name, period):
        c = exact(variants[name]["periods"][period], "cagr_pct")
        return c is not None and c >= t

    return {
        "winner": winner,
        "winner_test_ge_target": bool(winner and ge(winner, "test")),
        "winner_whole_history_ge_target": bool(winner and ge(winner, full_key)),
        "variants_ge_target_test": [n for n in variants if ge(n, "test")],
        "variants_ge_target_train": [n for n in variants if ge(n, "train")],
        "variants_ge_target_whole_history": [n for n in variants if ge(n, full_key)],
    }


def build_answer(mom: dict, lev: dict, bench_a: dict) -> dict:
    """The plain-English verdict, built from the numbers only."""
    mv, lv = mom["variants"], lev["variants"]
    mw, lw = mom["winner"]["name"], lev["winner"]["name"]
    det = {"momentum": _reaches(mv, mw, "pooled"), "leverage": _reaches(lv, lw, "full")}
    on_paper = {k: bool(d["winner_test_ge_target"] and d["winner_whole_history_ge_target"])
                for k, d in det.items()}
    # momentum: the stock list is today's index members - never evidence, whatever it shows;
    # leverage: only when the fund model passed its check against the real funds
    usable = {"momentum": False,
              "leverage": bool((lev.get("validation") or {}).get("usable_as_evidence", False))}
    reaches = {k: bool(on_paper[k] and usable[k]) for k in on_paper}
    boot = lev["bootstrap"]["variants"]
    sens = (lev["bootstrap"].get("sensitivity") or {})
    haircut = (sens.get("drift_haircut") or {})
    hc_pp = haircut.get("haircut_pp_per_year")
    hc_var = haircut.get("variants") or {}

    def lev_entry(name: str) -> dict:
        v = lv[name]
        full = v["periods"]["full"]
        bv = boot.get(name) or {}
        return {"name": name, "family": "leverage", "label": v["label"],
                "cagr_full": full["cagr_pct"],
                "cagr_test": v["periods"]["test"]["cagr_pct"],
                "max_drawdown": full["max_drawdown_pct"],
                "p_10y_ge_40": bv.get("p_cagr_ge_40_pct"),
                "p_dd_worse_50": bv.get("p_max_drawdown_worse_than_50_pct"),
                "p_dd_worse_75": bv.get("p_max_drawdown_worse_than_75_pct"),
                "p_end_below_start": bv.get("p_end_below_start_pct"),
                "history": [full.get("start"), full.get("end")],
                "history_years": full.get("years"),
                "biased_upward": False}

    def mom_entry(name: str) -> dict:
        v = mv[name]
        pooled = v["periods"]["pooled"]
        return {"name": name, "family": "momentum", "label": v["label"],
                "cagr_full": pooled["cagr_pct"],
                "cagr_test": v["periods"]["test"]["cagr_pct"],
                "max_drawdown": pooled["max_drawdown_pct"],
                "p_10y_ge_40": None, "p_dd_worse_50": None, "p_dd_worse_75": None,
                "p_end_below_start": None,
                "history": [pooled.get("start"), pooled.get("end")],
                "history_years": pooled.get("years"),
                "biased_upward": True,
                "note": "survivorship-inflated and shorter than 10 years (no bootstrap) - not "
                        "evidence, and not comparable with the leverage family's history"}

    bl, bm = best_on(lv, "full"), best_on(mv, "pooled")
    by_family = {"leverage": lev_entry(bl) if bl else None,
                 "momentum": mom_entry(bm) if bm else None}
    # the families cover different histories (about 9 and 40 years), so they are never
    # ranked against each other: best_tested comes from the family usable as evidence
    best = None
    if usable["leverage"] and by_family["leverage"] is not None:
        best = dict(by_family["leverage"])
        best["selection"] = (
            f"the highest whole-history CAGR of the {len(lv)} leveraged variants - picked with "
            "hindsight, so an upper bound on what was tested, not a forecast. The momentum "
            "family is left out: its figures are survivorship-inflated and cover a much shorter "
            "history (see best_tested_biased)")
    best_note = None if best is not None else (
        "no family is usable as evidence (momentum: survivorship-inflated stock list; leverage: "
        "the fund model missed the real funds by more than the tolerance) - see "
        "best_tested_by_family for the raw hindsight figures")

    n_all = len(mv) + len(lv)
    whole = det["momentum"]["variants_ge_target_whole_history"] \
        + det["leverage"]["variants_ge_target_whole_history"]
    one_period = sorted(set(det["momentum"]["variants_ge_target_test"]
                            + det["momentum"]["variants_ge_target_train"]
                            + det["leverage"]["variants_ge_target_test"]
                            + det["leverage"]["variants_ge_target_train"]) - set(whole))
    t = f"{TARGET_CAGR_PCT:.0f}%"
    parts: list[str] = []
    if on_paper["momentum"] and on_paper["leverage"]:
        parts.append(f"On paper both pre-set rules reached {t} a year - but neither figure can be "
                     "taken at face value: the stock-momentum figure is inflated by its stock list "
                     "(today's index members) and the leveraged one rests on a model and on two "
                     "exceptional eras for the Nasdaq.")
    elif on_paper["momentum"]:
        parts.append(f"Only on paper, and not in a way you can rely on. The pre-set stock-momentum "
                     f"rule shows more than {t} a year, but it picks from today's index members - a "
                     "list drawn up after the fact, which already holds the stocks that went on "
                     "to win - so that figure is not "
                     f"evidence that {t} a year is achievable. The leveraged Nasdaq-100 rule, which "
                     f"does not have that flaw, did not reach {t} a year.")
    elif on_paper["leverage"] and usable["leverage"]:
        parts.append(f"On paper yes: the pre-set leveraged Nasdaq-100 rule made {t} a year or more "
                     "both in the years it was not chosen on and over its whole history. That "
                     "rests on a model of leveraged funds and on two exceptional eras for the "
                     "Nasdaq - read the caveats before relying on it.")
    elif on_paper["leverage"]:
        parts.append(f"Only on paper, and not in a way you can rely on. The pre-set leveraged "
                     f"Nasdaq-100 rule shows {t} a year or more, but the model of leveraged funds "
                     "behind it missed the real funds by more than the tolerance (see "
                     f"leverage.validation), so that figure is not evidence that {t} a year is "
                     "achievable.")
    elif whole:
        parts.append(f"Not in a way you can rely on. {len(whole)} of the {n_all} rule sets "
                     f"({', '.join(whole)}) show {t} a year or more over their whole tested "
                     "history, but the rule chosen in advance did not hold that level, so this is "
                     "hindsight.")
    elif one_period:
        parts.append(f"No. None of the {n_all} rule sets made {t} a year over its whole tested "
                     f"history; {len(one_period)} ({', '.join(one_period)}) did so in one part of "
                     "it only.")
    else:
        parts.append(f"No. None of the {n_all} rule sets made {t} a year, in any period tested.")

    b1 = lv[B_BASE]["periods"]
    if lw:
        w = lv[lw]["periods"]
        roll, bw = lv[lw]["rolling_10y"] or {}, boot.get(lw) or {}
        s = (f"Leveraged Nasdaq-100: the rule fixed in advance on {_span(w['train'])} was "
             f"{lv[lw]['label']}. "
             + _leverage_vs_1x_text(w, b1, lev["winner"].get("test") or {}))
        tax = {p: ((w[p].get("after_tax_estimate_de") or {}).get("cagr_pct"),
                   (b1[p].get("after_tax_estimate_de") or {}).get("cagr_pct"))
               for p in ("full", "test")}
        if all(x is not None for pair in tax.values() for x in pair):
            behind = [tax[p][0] <= tax[p][1] for p in ("full", "test")]
            s += (" After German withholding tax on every profitable switch (an estimate) the "
                  f"chosen rule made about {_fmt(tax['full'][0])}% a year over the whole history "
                  f"against about {_fmt(tax['full'][1])}% for holding the Nasdaq-100 and paying "
                  f"tax once at the end ({_span(w['test'])}: {_fmt(tax['test'][0])}% against "
                  f"{_fmt(tax['test'][1])}%) - "
                  + ("the pre-tax edge disappears." if all(behind) else
                     "an edge remains after tax." if not any(behind) else
                     "the pre-tax edge largely disappears."))
        if roll:
            yrs = roll.get("windows_ge_target_start_years") or []
            s += (f" It made {t} a year or more in {_fmt(roll.get('share_ge_target_pct'))}% of the "
                  "overlapping 10-year windows of that history (not a probability: the history "
                  f"is about {_count_word(roll.get('independent_windows') or 0)} independent "
                  "decades"
                  + (f", and every such window began in {_years_text(yrs)}" if yrs else "")
                  + f"; the worst window made {_fmt(roll.get('min_cagr_pct'))}% a year)")
            if bw:
                s += (f", and in {_fmt(bw.get('p_cagr_ge_40_pct'))}% of {bw.get('paths', 0):,} "
                      "re-shuffled 10-year histories")
                h = (hc_var.get(lw) or {}).get("p_cagr_ge_40_pct")
                if h is not None and hc_pp is not None:
                    s += (f" ({_fmt(h)}% if the Nasdaq had returned {_fmt(hc_pp, 0)} points a "
                          "year less)")
                s += (f"; in {_fmt(bw.get('p_max_drawdown_worse_than_50_pct'))}% of those it "
                      "lost more than half its value on the way.")
            else:
                s += "."
        parts.append(s)
    else:
        parts.append(f"Leveraged Nasdaq-100: no variant kept its {_span(b1['train'])} drawdown "
                     f"inside the {B_TRAIN_DD_FLOOR_PCT:.0f}% limit, so no rule could be chosen in "
                     "advance.")
    r1 = lv[B_BASE].get("rolling_10y") or {}
    if r1:
        mx = r1.get("max_window") or [None, None]
        if not r1.get("windows_ge_target"):
            parts.append(f"The plain Nasdaq-100 never made {t} a year over any 10 years in this "
                         f"history; its best 10 years ({_yr(mx[0])}-{_yr(mx[1])}) made about "
                         f"{_fmt(r1.get('max_cagr_pct'), 0)}% a year.")
        else:
            parts.append(f"The plain Nasdaq-100 made {t} a year or more in "
                         f"{_fmt(r1.get('share_ge_target_pct'))}% of the overlapping 10-year "
                         f"windows (those starting in "
                         f"{_years_text(r1.get('windows_ge_target_start_years') or [])}).")
    single = [n_ for n_ in det["leverage"]["variants_ge_target_test"]
              if n_ != lw and n_ not in det["leverage"]["variants_ge_target_whole_history"]]
    if single:
        n_ = max(single, key=lambda x: lv[x]["periods"]["test"]["cagr_pct"])
        pv = lv[n_]["periods"]
        worst = abs(exact(pv["full"], "max_drawdown_pct"))
        only = len(single) == 1 and not det["leverage"]["variants_ge_target_whole_history"] \
            and not det["leverage"]["winner_test_ge_target"]
        lead = (f"The only leveraged variant that made {t} a year did so in a single era:"
                if only else
                f"{_count_word(len(single)).capitalize()} more leveraged "
                f"variant{'s' if len(single) > 1 else ''} made {t} a year in a single era only; "
                "the highest:")
        parts.append(f"{lead} {lv[n_]['label']}, {_cagr_txt(pv['test'])}% a year in "
                     f"{_span(pv['test'])}, "
                     f"at the price of a fall of "
                     f"{_fmt(abs(exact(pv['test'], 'max_drawdown_pct')), 0)}% "
                     f"on the way. The same rule made {_cagr_txt(pv['train'])}% a year in "
                     f"{_span(pv['train'])} and lost "
                     f"{_fmt(worst, 2 if worst >= 99.5 else 0)}% of its value at its worst "
                     f"({_yr(pv['full'].get('max_drawdown_peak'))}-"
                     f"{_yr(pv['full'].get('max_drawdown_trough'))}).")
    if bl and bl != lw:
        e = by_family["leverage"]
        s = (f"With hindsight the highest return of the {len(lv)} leveraged variants was "
             f"{e['label']}: {_cagr_txt(lv[bl]['periods']['full'])}% a year over the whole "
             f"history, with a worst fall of {_dd_txt(lv[bl]['periods']['full'])}%")
        if e.get("p_10y_ge_40") is not None:
            s += (f"; re-shuffled 10-year histories gave it {t} a year in "
                  f"{_fmt(e['p_10y_ge_40'])}% of cases, a fall of more than three quarters in "
                  f"{_fmt(e['p_dd_worse_75'])}% and a loss over the 10 years in "
                  f"{_fmt(e['p_end_below_start'])}%")
        parts.append(s + ".")

    pa = bench_a["pooled"]
    if mw:
        w = mv[mw]["periods"]
        fresh = bench_a["test"].get(EW_BH_FRESH)
        s = (f"Momentum in stocks: the rule fixed in advance on {_span(w['train'])} was "
             f"{mv[mw]['label']}. In {_span(w['test'])} it made {_cagr_txt(w['test'])}% a "
             f"year against {_cagr_txt(bench_a['test']['QQQ'])}% for the Nasdaq-100 fund "
             f"QQQ and {_cagr_txt(fresh)}% for buying all the same stocks in equal amounts at the "
             f"start of that period and holding them; its worst fall since "
             f"{_span(w['pooled'])[:4]} was {_dd_txt(w['pooled'])}%.")
        conc = mv[mw].get("concentration") or {}
        if conc.get("cagr_without_best_month_pct") is not None:
            top = [c["ticker"] for c in conc.get("top_contributors") or []][:3]
            s += (f" Much of it hangs on little: without its single best month "
                  f"({conc['best_month']['month']}, {_fmt(conc['best_month']['return_pct'], 0, True)}"
                  f"%) its whole-history return falls from {_cagr_txt(w['pooled'])}% to "
                  f"{_fmt(conc['cagr_without_best_month_pct'])}% a year"
                  + (f"; the largest contributors were {', '.join(top)}." if top else "."))
        ex = conc.get("cagr_without_top_contributor_pct") or {}
        if ex.get("train") is not None:
            s += (f" Without one stock ({ex['ticker']}) its {_span(w['train'])} return falls "
                  f"from {_cagr_txt(w['train'])}% to {_fmt(ex['train'])}% a year.")
        parts.append(s)
    else:
        parts.append("Momentum in stocks: no variant kept its training drawdown inside the "
                     f"{A_TRAIN_DD_FLOOR_PCT:.0f}% limit, so no rule could be chosen in advance.")
    ew_c, spy_c = exact(pa[EW_BH], "cagr_pct"), exact(pa["SPY"], "cagr_pct")
    gp = ew_c - spy_c if ew_c is not None and spy_c is not None else None
    parts.append(
        "The stock figures are inflated and cannot be taken at face value: the list is today's "
        "index members, so the rule chooses among stocks already known to have done well enough "
        "to be in the index today"
        + (f". Bought in equal amounts at the start and simply held, that list made "
           f"{_fmt(ew_c)}% a "
           f"year, {_fmt(abs(gp))} points {'more' if gp >= 0 else 'less'} than the S&P 500 fund "
           f"SPY ({_fmt(spy_c)}%); for a rule that buys only the strongest 10-20 "
           "names the distortion is much larger and cannot be measured with this data."
           if gp is not None else "."))
    parts.append(
        "All figures are in USD, before tax unless stated, and describe the past - which for the "
        "Nasdaq contains two exceptional eras (1990-2000 and 2010-2026); they are not a forecast.")
    parts.append(MEDALLION_CONTEXT)
    return {
        "target_cagr_pct": TARGET_CAGR_PCT,
        "reaches_target": reaches,
        "reaches_target_on_paper": on_paper,
        "reaches_target_basis": (
            "reaches_target[family] = reaches_target_on_paper[family] AND "
            "usable_as_evidence[family]. On paper = the variant chosen by the pre-registered "
            f"TRAIN rule made at least {TARGET_CAGR_PCT:.0f}%/yr (unrounded) BOTH on the TEST "
            "period and over its whole tested history. Momentum is never usable as evidence "
            "(the stock list is today's index members); leverage is usable when the fund model "
            "passed its check against the real funds (leverage.validation.usable_as_evidence). "
            "See reaches_target_detail for every variant that made the target in a single period"),
        "reaches_target_detail": det,
        "usable_as_evidence": usable,
        "evidence_quality": {
            "momentum": "NOT usable as evidence that the target is achievable: the universe is "
                        "today's index members (survivorship / look-ahead in the stock list), "
                        "which inflates a concentrated momentum rule by an amount this data "
                        "cannot measure; about nine years of history, no long bear market",
            "leverage": "a model of daily-reset leveraged funds before the real funds existed "
                        "(checked against QQQ / QLD / TQQQ, see leverage.validation: "
                        + ("within the tolerance over each fund's whole life"
                           if usable["leverage"] else
                           "OUTSIDE the tolerance for at least one fund - not usable")
                        + ", with offsetting errors by T-bill regime where flagged); about "
                        "forty years of history holding two exceptional eras for the Nasdaq",
        },
        "best_tested": best,
        "best_tested_note": best_note,
        "best_tested_biased": by_family["momentum"],
        "best_tested_by_family": by_family,
        "context": {"label": "not a lab result", "text": MEDALLION_CONTEXT},
        "text": " ".join(parts),
    }


def _regime_caveat(lev: dict) -> str | None:
    """The offsetting errors of the fund model by T-bill regime, from the numbers."""
    val = lev["validation"]
    rows = {etf: {r["regime"]: r["synthetic_minus_real_pp_per_year"]
                  for r in (val.get(etf) or {}).get("by_tbill_regime") or []}
            for etf in B_ETF.values()}
    lo_label, hi_label = B_TBILL_REGIMES[0][0], B_TBILL_REGIMES[-1][0]
    out = []
    pairs = []
    for lev_, etf in B_ETF.items():
        lo, hi = rows[etf].get(lo_label), rows[etf].get(hi_label)
        if lev_ > 1 and lo is not None and hi is not None and lo * hi < 0 \
                and max(abs(lo), abs(hi)) > B_VALIDATION_TOL_PP:
            pairs.append(f"{lev_}x: {lo:+.1f} pct-pt/yr while T-bills are {lo_label}, "
                         f"{hi:+.1f} while they are {hi_label}")
    if pairs:
        out.append("The model's overall fit hides offsetting errors by interest-rate regime ("
                   + "; ".join(pairs) + "; positive = the model is too optimistic).")
    avg = (val.get("tbill_average_pct") or {}).get("train")
    tr = lev["variants"][B_BASE]["periods"]["train"]
    if avg is not None:
        regime = _regime_of(avg)
        err = [(lev_, rows[etf].get(regime)) for lev_, etf in B_ETF.items() if lev_ > 1]
        err = [(lev_, d) for lev_, d in err if d is not None and abs(d) > 0.05]
        if err and all(d > 0 for _, d in err) or err and all(d < 0 for _, d in err):
            out.append(f"T-bills averaged {avg:.1f}% in {_span(tr)}, mostly before the real "
                       "leveraged funds existed, so synthetic "
                       + " / ".join(f"{lev_}x" for lev_, _ in err)
                       + " results for that era are probably "
                       + ("overstated" if err[0][1] > 0 else "understated") + " by roughly "
                       + " / ".join(f"{abs(d):.1f}" for _, d in err)
                       + " pct-pt per year (the model's error on the real funds' days with "
                       f"T-bills {regime}).")
    cons = lev.get("conservative_financing") or {}
    if out and cons:
        wn = lev["winner"]["name"]
        c = (cons.get("variants") or {}).get(wn) if wn else None
        out.append(f"With more expensive financing ({cons['assumptions']}) "
                   + (f"the chosen rule's {_span(tr)} return falls from "
                      f"{_fmt(lev['variants'][wn]['periods']['train']['cagr_pct'])}% to "
                      f"{_fmt(c['train_cagr_pct'])}% a year and " if c else "")
                   + ("the winner rule would pick a different variant "
                      f"({cons.get('winner')})." if cons.get("winner_changes")
                      else "the winner rule still picks the same variant."))
    return " ".join(out) if out else None


def _timing_caveat(lev: dict) -> str | None:
    ts, w = lev.get("timing_sensitivity") or {}, lev["winner"]
    out = []
    wn = w.get("name")
    if wn and ts:
        tr = lev["variants"][wn]["periods"]["train"]
        wt = ts.get("winner_train_same_close") or {}
        other = ts.get("winner_same_close")
        other_txt = (f"the rule would pick {lev['variants'][other]['label']} instead"
                     if other else "no variant would qualify at all")
        if ts.get("winner_changes"):
            dd = wt.get("max_drawdown_pct")
            if dd is not None and dd < B_TRAIN_DD_FLOOR_PCT:
                out.append(f"Timing: the chosen rule ({lev['variants'][wn]['label']}) passes the "
                           f"{B_TRAIN_DD_FLOOR_PCT:.0f}% drawdown limit only with the one-day "
                           f"delay; switching at the signal's own close its {_span(tr)} drawdown "
                           f"is {dd:.0f}% and {other_txt}.")
            else:
                out.append("Timing: the choice of rule depends on the one-day delay; switching "
                           f"at the signal's own close {other_txt}.")
        else:
            out.append("Timing: switching at the signal's own close instead of one close later "
                       "would pick the same rule.")
    wd = ts.get("worst_day_held_after_sell_signal")
    if wd:
        top = max(B_LEVERAGE)
        out.append("With the delay the fund is still held on the day after a sell signal: the "
                   f"{top}x fund lost about {abs(wd[f'fund_{top}x_return_pct']):.0f}% on "
                   f"{wd['date']} (the index {wd['index_return_pct']:+.1f}%).")
    return " ".join(out) if out else None


def _bootstrap_caveat(lev: dict) -> str | None:
    boot = lev["bootstrap"]
    sens = boot.get("sensitivity") or {}
    hc = (sens.get("drift_haircut") or {})
    hv = hc.get("variants") or {}
    b36 = (sens.get(f"block_{BOOT_SENS_BLOCK}_months") or {}).get("variants") or {}
    head = {k: v for k, v in boot["variants"].items() if v}
    if not head or not hv:
        return None
    full = lev["variants"][B_BASE]["periods"]["full"]
    months = next(iter(head.values()))["months_in_history"]
    top = max(head, key=lambda k: head[k]["p_cagr_ge_40_pct"])
    names = [top] + ([lev["winner"]["name"]] if lev["winner"]["name"] not in (None, top) else [])
    falls = [f"from {head[k]['p_cagr_ge_40_pct']:.1f}% to {hv[k]['p_cagr_ge_40_pct']:.1f}% for "
             f"{lev['variants'][k]['label']}" for k in names if hv.get(k)]
    centre = bootstrap_centre_text(boot["variants"], lev["variants"])
    s = ("Bootstrap sensitivity: " + centre[0].lower() + centre[1:]
         + f" If the Nasdaq had returned {hc['haircut_pp_per_year']:.0f} points a year "
         f"less than in {_span(full)}, the chance of {TARGET_CAGR_PCT:.0f}%/yr over 10 years "
         "falls " + " and ".join(falls) + f". The history is {months // 12} years - about "
         f"{months // BOOT_BLOCK} distinct {BOOT_BLOCK}-month blocks dominated by two bull eras; "
         f"{BOOT_BLOCK}-month blocks understate multi-year bear markets")
    a, b = head.get(B_BASE), b36.get(B_BASE)
    if a and b:
        s += (f" (with {BOOT_SENS_BLOCK}-month blocks the chance that the plain Nasdaq-100 fund "
              f"falls by more than 75% is {b['p_max_drawdown_worse_than_75_pct']:.1f}% instead of "
              f"{a['p_max_drawdown_worse_than_75_pct']:.1f}%)")
    return s + "."


def _momentum_caveat(mom: dict) -> str | None:
    w = mom["winner"]
    if not w.get("name"):
        return None
    per = mom["variants"][w["name"]]["periods"]
    tr, te = per["train"], per["test"]
    s = (f"The momentum winner was chosen on {tr['months']} months and tested on "
         f"{te['months']} months")
    noise = w.get("test_noise")
    if noise:
        s += (f"; with about {noise['ann_volatility_pct']:.0f}% annual volatility, differences "
              f"below roughly {noise['standard_error_pp']:.0f} points a year between variants "
              "over the test period are noise")
    s += "."
    sm = w.get("selection_margin") or {}
    ex = (w.get("concentration") or {}).get("cagr_without_top_contributor_pct") or {}
    bits = []
    if sm.get("train_drawdown_distance_to_limit_pp") is not None:
        bits.append(f"its {_span(tr)} drawdown was within "
                    f"{sm['train_drawdown_distance_to_limit_pp']:.1f} points of the "
                    f"{sm['train_drawdown_limit_pct']:.0f}% limit")
    if ex.get("train") is not None:
        bits.append(f"without one stock ({ex['ticker']}) its {_span(tr)} return falls from "
                    f"{tr['cagr_pct']:.0f}% to {ex['train']:.0f}% a year")
    if bits:
        txt = ", and ".join(bits)
        s += " " + txt[0].upper() + txt[1:] + "."
    pl = mom["survivorship"].get("placebo") or {}
    rnd, bot = (pl.get("random_n") or {}), (pl.get("bottom_n") or {})
    if (rnd.get("pooled") or {}).get("median_cagr_pct") is not None:
        r = rnd["pooled"]
        s += (f" Placebos with the same rules: {rnd['seeds']} random picks from the same list "
              f"made a median of {r['median_cagr_pct']:.1f}% a year over the whole period (95th "
              f"percentile {r['p95_cagr_pct']:.1f}%)")
        if (bot.get("test") or {}).get("cagr_pct") is not None:
            s += (f", and the WEAKEST stocks by momentum made {bot['test']['cagr_pct']:.1f}% a "
                  f"year in {_span(te)} ({bot['train']['cagr_pct']:.1f}% in {_span(tr)})")
        s += "."
    lq = mom["survivorship"].get("liquidity_rule") or {}
    if lq.get("eligible_without_liquidity_rule"):
        ex_n, base = lq["excluded_by_liquidity_rule"], lq["eligible_without_liquidity_rule"]
        most = lq.get("max_excluded_at_any_signal")
        s += (f" The liquidity rule excludes {'almost nothing' if ex_n <= 0.01 * base else 'little'}"
              f" in this universe ({ex_n} of {base} stocks at the last signal, at most {most} at "
              "any signal)."
              if ex_n <= 0.05 * base else
              f" The liquidity rule excludes {ex_n} of {base} stocks at the last signal.")
    return s


def build_caveats(lev: dict, mom: dict, a_info: dict) -> list[str]:
    out = list(lev["validation"].get("flags") or [])
    lw = lev["winner"]["name"]
    tax = None
    if lw:
        w, b = lev["variants"][lw]["periods"]["full"], lev["variants"][B_BASE]["periods"]["full"]
        wt = (w.get("after_tax_estimate_de") or {})
        bt_ = (b.get("after_tax_estimate_de") or {})
        if wt.get("cagr_pct") is not None and bt_.get("cagr_pct") is not None:
            tax = (f" An estimate (leverage.after_tax_model_de: {DE_TAX_RATE:.3%} of every "
                   "realised gain, losses carried forward, no allowance, no Vorabpauschale, no "
                   f"church tax) puts the chosen leveraged rule at about {wt['cagr_pct']:.1f}% a "
                   f"year over {_span(w)} ({wt['cagr_with_30pct_partial_exemption_pct']:.1f}% "
                   "with the 30% partial exemption for equity funds) against about "
                   f"{bt_['cagr_pct']:.1f}% ({bt_['cagr_with_30pct_partial_exemption_pct']:.1f}%) "
                   "for holding the Nasdaq-100 and paying tax once at the end.")
    turn = a_info.get("turnover_range_pct_per_month") or [None, None]
    sm = (lev["winner"].get("selection_margin") or {}).get("train_cagr_gap_to_runner_up_pp")
    out += [
        mom["survivorship"]["caveat"],
        "Synthetic leverage before the real funds existed (QLD 2006, TQQQ 2010) is a MODEL: "
        "daily reset, financing at T-bills + 0.5%, fixed expense ratios and an assumed 0.7%/yr "
        "dividend yield. Real leveraged funds can deviate in crashes and can be closed. A daily-"
        "reset 3x fund is wiped out by a one-day fall of 33%; ^NDX fell 15% on 1987-10-19 and "
        "about 12% on 2020-03-16.",
        _regime_caveat(lev),
        _timing_caveat(lev),
        "Taxes are ignored in the headline figures. In Germany every profitable switch or sale "
        "realises taxable gains, which lowers compounding - most for the rules that trade most."
        + (tax or "")
        + (f" Strategy A replaces roughly {turn[0]:.0f}-{turn[1]:.0f}% of the book every month, "
           "so nearly all of its gains would be taxed within the year."
           if turn[0] is not None else ""),
        "Currency: all results are in USD. A EUR investor also carries the USD/EUR exchange "
        "rate, in both directions; the EUR/USD effect is not measured by this lab.",
        "Instruments: US-listed QLD / TQQQ are generally not sold to EU retail investors (no "
        "PRIIPs KID); UCITS / ETP substitutes have different fees, trading hours and, for ETPs, "
        "issuer risk - not modelled.",
        "Multiple testing: the grids are small (8 momentum and 9 leverage variants) but the best "
        "of several rules always looks better than it is; only the pre-registered winner's TEST "
        "figures are free of that, and each family has just one TEST period.",
        "Parameters, grids, splits and selection rules were fixed before this run "
        "(pre-registered) and nothing was tuned afterwards, including the leverage model. The "
        "rule families themselves are well known and were chosen with knowledge of the "
        "historical period, so the test period is not a clean out-of-sample test."
        + (f" The leverage winner was chosen by {sm:.1f} points of return over the runner-up."
           if sm is not None else ""),
        _momentum_caveat(mom),
        "Past returns of the Nasdaq-100 are not a forecast: 1990-2000 and 2010-2026 were "
        "exceptional eras and both sit inside this history. The rolling 10-year windows overlap "
        "(they are not independent observations - the history holds about four independent "
        "decades) and the bootstrap only re-orders the past; the future can be worse than "
        "anything in either.",
        _bootstrap_caveat(lev),
        "^NDX is a price index (no dividends, roughly 0.5-1%/yr); QQQ and SPY figures use "
        "dividend-adjusted closes. ^IRX is a discount yield, used as the cash return and the "
        "financing base without conversion.",
        f"Strategy A covers {a_info['first_signal']} .. {a_info['last_day']} only (the local "
        "price cache holds about ten years), a period without a long bear market; its TEST "
        f"period is {a_info['test'][0]} .. {a_info['test'][1]}. Strategy B covers about forty "
        "years: the two families' figures are for different periods and must not be compared "
        "directly.",
    ]
    return [c for c in out if c]


def build_method(series_info: dict, price_info: dict, a_info: dict, b_info: dict,
                 lev: dict, n_paths: int) -> dict:
    val = {k: {kk: lev["validation"][k].get(kk)
               for kk in ("start", "end", "synthetic_cagr_pct", "real_cagr_pct", "cagr_diff_pp",
                          "daily_return_correlation", "within_tolerance",
                          "max_abs_regime_diff_pp", "regimes_within_tolerance")}
           for k in B_ETF.values() if isinstance(lev["validation"].get(k), dict)}
    ts = lev.get("timing_sensitivity") or {}
    n_f = ts.get("filtered_variants") or 0
    not_conservative = bool(ts.get("delay_higher_train_cagr") or
                            ts.get("delay_shallower_train_drawdown"))
    timing_call = (
        "B timing: 'hold the fund on day t+1' and 'the switch executes at the close of t+1' "
        "cannot both hold; the delayed reading is implemented (signal at the close of t, switch "
        "at the close of t+1, exposure from t+2)."
        + (f" Against switching at the signal's own close it gave a higher TRAIN CAGR in "
           f"{ts.get('delay_higher_train_cagr')} of the {n_f} filtered variants, a shallower "
           f"TRAIN drawdown in {ts.get('delay_shallower_train_drawdown')} and a lower TEST CAGR "
           f"in {ts.get('delay_lower_test_cagr')}"
           + (" - it is not uniformly conservative" if not_conservative else
              " - here it was the conservative reading")
           + "; see leverage.timing_sensitivity." if n_f else ""))
    return {
        "pre_registered": "parameters, grids, splits and selection rules were fixed before the "
                          "run and nothing was tuned afterwards; the rule families themselves "
                          "are well known and were chosen with knowledge of the historical "
                          "period, so the test periods are not clean out-of-sample tests",
        "currency": "USD",
        "taxes": "none in the headline figures; an estimate of German withholding tax for "
                 "Strategy B is stored per variant and period (leverage.after_tax_model_de)",
        "comparability": "Strategy A (about nine years, a survivorship-biased stock list) and "
                         "Strategy B (about forty years, an index) cover different periods and "
                         "must not be compared directly; threshold comparisons (target, drawdown "
                         "limits) use unrounded figures",
        "momentum": {
            "universe": "universe/us.csv - today's S&P 500 + S&P 400 members, local price cache, "
                        "bars repaired with backtest.sanitize_bars (same overrides and stale map "
                        "as the backtest); prices are dividend- and split-adjusted",
            "eligibility": f"a bar on the signal day, at least {A_MIN_PRIOR_BARS} earlier bars, "
                           f"{A_DV_WINDOW}-bar average traded value >= "
                           f"{MIN_DOLLAR_VOL[A_MARKET]:,.0f} USD (scan.MIN_DOLLAR_VOL['us'])",
            "signal": f"close of the last trading day of each month; momentum = close[t-{A_SKIP}] "
                      f"/ close[t-{A_SKIP}-L] - 1 on the stock's own bars, L in "
                      f"{list(A_LOOKBACKS)}; trades at the next session's open",
            "portfolio": f"top N in {list(A_TOP_N)} by momentum, equal weight at every rebalance; "
                         "leavers sold and stayers re-weighted at that open; fully invested, no "
                         "leverage",
            "trend_filter": f"off / on; on = stocks only while ^GSPC closes above its {A_SMA}-day "
                            "average on the signal day, else cash until the next monthly signal; "
                            "cash earns ^IRX / 100 / 252 per trading day",
            "costs": f"{A_COST_PCT_PER_SIDE}% per side + {A_SLIPPAGE_PCT}% slippage on the traded "
                     "notional only (the repo's US cost model); the equal-weight baskets and the "
                     "ETFs pay nothing",
            "split": {"train": a_info["train"], "test": a_info["test"],
                      "rule": f"TRAIN = first signal .. {A_TRAIN_END}, TEST = from then to the "
                              "last full month; one continuous account, measured per period"},
            "winner_rule": f"highest TRAIN CAGR among variants whose TRAIN max drawdown is no "
                           f"worse than {A_TRAIN_DD_FLOOR_PCT:.0f}%",
            "benchmarks": "QQQ and SPY (dividend-adjusted closes); equal_weight_buy_and_hold = "
                          "equal amounts in every stock with a bar at the first signal's close, "
                          "never rebalanced, no costs (the 'equal-weight universe' of the "
                          "survivorship gap; one continuous account, plus the same basket bought "
                          "fresh at the start of TEST); "
                          "equal_weight_universe_daily_rebalanced = every stock with a bar, "
                          "rebalanced daily from close-to-close returns, single-stock days "
                          f"capped at +/-{DAILY_RET_CLIP:.0%}, no costs - the backtest's "
                          "convention, NOT investable (cost-free daily rebalancing); "
                          "equal_weight_eligible_monthly = all ELIGIBLE stocks rebalanced on the "
                          "strategy's own monthly schedule (no costs)",
            "diagnostics": "selection_margin, concentration (best month, top contributors, the "
                           "winner re-run without its top contributor), test_noise and the "
                           "placebos (survivorship.placebo) are hindsight diagnostics, not "
                           "pre-registered metrics",
            "window": a_info,
        },
        "leverage": {
            "series": "^NDX daily closes (price index, no dividends), ^IRX 13-week T-bill yield",
            "synthetic_fund": f"daily return = L * (r_ndx + {B_DIV_YIELD:.1%}/252) - (L-1) * "
                              f"(tbill + {B_FINANCING_SPREAD:.1%})/252 - expense/252; expense "
                              f"{B_EXPENSE[1]:.2%}/yr (1x), {B_EXPENSE[2]:.2%}/yr (2x, 3x); "
                              "a daily loss is capped at -100%",
            "filters": f"none / sma200 (hold the fund when ^NDX closes above its {B_SMA}-day "
                       f"average) / sma200_band (in above +{B_BAND:.0%}, out below -{B_BAND:.0%}); "
                       "the signal of close t is executed at the close of t+1, so the new holding "
                       "earns from t+2; out of the fund = T-bills at ^IRX / 100 / 252 per day; "
                       f"{B_SWITCH_COST:.2%} of equity per switch",
            "split": {"train": b_info["train"], "test": b_info["test"],
                      "rule": f"TRAIN = start .. {B_TRAIN_END}, TEST = from then to the latest "
                              "bar; one continuous account, measured per period"},
            "winner_rule": f"highest TRAIN CAGR among variants whose TRAIN max drawdown is no "
                           f"worse than {B_TRAIN_DD_FLOOR_PCT:.0f}%",
            "rolling_10y": f"{ROLL_MONTHS}-month windows starting at every month end; CAGR, share "
                           f">= {TARGET_CAGR_PCT:.0f}%, share >= the 1x unfiltered synthetic fund "
                           "in the same window, worst fall inside any window (daily marks). "
                           "Overlapping windows are NOT independent (a share of windows is not a "
                           "probability; non_overlapping_decades lists the independent ones); "
                           "1990-2000 and 2010-2026 were exceptional eras",
            "bootstrap": f"circular moving-block: {n_paths:,} paths x {BOOT_HORIZON} months, "
                         f"{BOOT_BLOCK}-month blocks of the variant's own monthly returns, "
                         f"default_rng(seed={BOOT_SEED}); it re-orders the past, the future may "
                         f"be worse. Sensitivities: {BOOT_SENS_BLOCK}-month blocks, and the index "
                         f"return lowered by {BOOT_SENS_HAIRCUT_PP:.0f} points a year",
            "sensitivities": "timing_sensitivity (same-close execution), conservative_financing "
                             "(dearer financing, lower dividends), bootstrap.sensitivity and "
                             "after_tax_estimate_de are computed next to the headline; none of "
                             "them is used to choose a variant",
            "window": b_info,
        },
        "validation": val,
        "data": {"stock_prices": price_info, "series": series_info},
        "judgement_calls": [
            timing_call,
            f"B start: fixed at {B_START} as written; the first 200-day average of the "
            f"downloaded ^NDX exists on {b_info['first_sma200']}, earlier than that.",
            "B dividends: the assumed 0.7%/yr is added to the index return BEFORE leverage "
            "(L * (r + 0.7%/252)), as a total-return swap would pay it.",
            "B financing: tbill(t) is the ^IRX value of the same day (carried forward on days "
            "without a quote), divided by 252 trading days as written.",
            "B switch cost: no cost for the opening position; every later switch costs 0.10% of "
            "equity on the day it is done. The band filter's opening state uses the plain rule.",
            "B bootstrap: a circular moving-block bootstrap (blocks wrap from the last month to "
            "the first); the same block starts for every variant; drawdown probabilities are on "
            "daily marks (each month carries its intra-month low, high and fall).",
            "B monthly statistics (rolling windows, bootstrap) use full calendar months only. At "
            "the end of the data a month counts as complete when no later weekday of it is a "
            "trading day (Good Friday and Memorial Day are the only scheduled NYSE holidays that "
            "can end a month); a calendar year likewise only on the last trading day of December.",
            "B validation: usable_as_evidence for the leverage family is derived from the "
            "whole-life check against QQQ / QLD / TQQQ; the by-period and by-T-bill-regime rows "
            "are reported and flagged but nothing is tuned to them.",
            "B after-tax estimate: each period is taxed as a fresh account that is sold at its "
            "end; the tax on a switch is taken from the proceeds on the same day.",
            f"A first signal: the first month end with at least {max(A_TOP_N)} eligible stocks "
            "(the same start for all 8 variants).",
            "A momentum, liquidity and the 273-bar rule are counted on the stock's OWN bars; the "
            "same eligibility is used for both look-backs.",
            "A costs: the post-cost equity is solved so the portfolio is exactly fully invested "
            "after paying costs on the traded notional (no leverage, no cash drag).",
            "A cash interest accrues on the balance held over the whole day (the smaller of the "
            "previous close's and the post-trade balance).",
            "A trend filter uses the downloaded ^GSPC series (full history for the 200-day "
            "average); a stock without a bar at the trade open is traded at its last close "
            "(counted in fills_without_bar).",
            "A equal-weight universe: the survivorship gap and 'CAGR minus the equal-weight "
            "universe' use the buy-and-hold basket (equal amounts at the first signal, never "
            "rebalanced). The backtest's daily-rebalanced basket is kept as "
            "equal_weight_universe_daily_rebalanced and labelled not investable; a "
            "monthly-rebalanced equal weight of the eligible stocks is reported next to them.",
            "A turnover: avg_turnover_pct_per_month is one-way (the larger of buys and sells as "
            "% of equity); avg_traded_notional_pct_per_month is buys + sells, the base of the "
            "costs.",
            "Worst calendar year includes partial first / last years (flagged 'partial').",
            "Time under water is counted from a high to the bar that recovers it (or to the "
            "last bar while still below it).",
            "answer.reaches_target is true only when the pre-registered winner made the target "
            "both on TEST and over its whole history (reaches_target_on_paper) AND the family "
            "is usable as evidence; best_tested is the hindsight best of the usable family only "
            "- families with different history lengths are never ranked against each other.",
            "Downloaded series: bars dated on or after the UTC day of the download are dropped "
            "(the session may be unfinished).",
        ],
    }


# --------------------------------------------------------------------------- run

def equity_output_path(out_path: Path) -> Path:
    out_path = Path(out_path)
    return out_path.with_name(out_path.stem + "_equity" + (out_path.suffix or ".json"))


def run_strategy_lab(out_path: Path | None = None, *, offline: bool = False, refresh: bool = False,
                     prices_path: Path | None = None, series_path: Path | None = None,
                     universe_path: Path | None = None, data_dir: Path | None = None,
                     cfg: dict | None = None, fetch=None, n_paths: int = BOOT_PATHS,
                     n_placebo: int = A_PLACEBO_SEEDS, progress=print) -> dict:
    """Run both strategies and write out_path (+ its _equity file). Returns the
    payload. prices_path / series_path / universe_path / data_dir / cfg / fetch
    are injectable for tests."""
    started = time.time()
    out_path = Path(out_path) if out_path else DEFAULT_OUT_PATH
    eq_path = equity_output_path(out_path)
    prices_path = Path(prices_path or DEFAULT_PRICES_PATH)
    series_path = Path(series_path or DEFAULT_SERIES_PATH)
    universe_path = Path(universe_path or DEFAULT_UNIVERSE_PATH)
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    for p in (prices_path, series_path, universe_path):
        if p.exists() and p.resolve() in (out_path.resolve(), eq_path.resolve()):
            raise ValueError(f"refusing to overwrite an input file: {p}")
    if series_path.resolve() == prices_path.resolve():
        raise ValueError("the series cache must be a separate file from the price cache")
    if not prices_path.exists():
        raise FileNotFoundError(f"price cache not found: {prices_path}")
    if not universe_path.exists():
        raise FileNotFoundError(f"universe file not found: {universe_path}")

    frames, series_info = ensure_series(series_path, offline=offline, refresh=refresh,
                                        fetch=fetch, progress=progress)
    progress("series: " + ", ".join(f"{t} {v['first']}..{v['last']}"
                                    for t, v in series_info["series"].items()))

    lev = leverage_block(frames, n_paths=n_paths, progress=progress)

    cfg = cfg if cfg is not None else load_config()
    tickers = load_universe(universe_path)
    progress(f"[A] loading and repairing {len(tickers)} stocks from {prices_path.name} "
             "(read-only) ...")
    panel, price_info = load_stock_panel(prices_path, tickers, cfg, data_dir, progress)
    mom = momentum_block(panel, frames, n_placebo=n_placebo, progress=progress)

    benchmarks = {
        "momentum_window": mom["benchmarks"],
        "leverage_window": lev["benchmarks"],
        "notes": [
            "QQQ = Nasdaq-100 fund and SPY = S&P 500 fund, dividends reinvested (Yahoo "
            "dividend-adjusted closes), no trading costs.",
            "ndx_price_index = ^NDX itself: a PRICE index that excludes dividends (about "
            "0.5-1%/yr); nasdaq_1x_synthetic = the lab's 1x fund model (index + 0.7%/yr "
            "dividends - 0.20%/yr expense), the 'simply hold the Nasdaq' line for the long "
            "history.",
            "equal_weight_buy_and_hold = today's index members bought in equal amounts at the "
            "first signal and never rebalanced (shares the survivorship bias of Strategy A); "
            "equal_weight_buy_and_hold_from_test_start (TEST only) = the same basket bought "
            "fresh at the start of TEST.",
            "equal_weight_universe_daily_rebalanced - not investable (cost-free daily "
            "rebalancing); equal_weight_eligible_monthly = the eligible stocks, re-weighted "
            "monthly without costs.",
            "Benchmarks are measured over exactly the same days as the strategies' periods.",
        ],
    }
    answer = build_answer(mom["block"], lev["block"], mom["benchmarks"])
    payload = {
        "version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "target_cagr_pct": TARGET_CAGR_PCT,
        "method": build_method(series_info, price_info, mom["info"], lev["info"], lev["block"],
                               n_paths),
        "benchmarks": benchmarks,
        "momentum": mom["block"],
        "leverage": lev["block"],
        "answer": answer,
        "caveats": build_caveats(lev["block"], mom["block"], mom["info"]),
        "equity_file": eq_path.name,
        "runtime_seconds": round(time.time() - started, 1),
    }
    equity = {"generated": payload["generated"],
              "note": "equity curves, start = 1.0; momentum daily, leverage at month ends",
              "momentum": mom["curves"], "leverage": lev["curves"]}
    write_json_atomic(out_path, payload, indent=1)
    write_json_atomic(eq_path, equity)
    for line in summary_lines(payload):
        progress(line)
    progress(f"wrote {out_path} and {eq_path.name} ({payload['runtime_seconds']}s)")
    return payload


def summary_lines(payload: dict) -> list[str]:
    """A readable summary of the payload (plain text table)."""
    L: list[str] = []
    f = _fmt
    L.append("")
    L.append(f"STRATEGY LAB - target {payload['target_cagr_pct']:.0f}%/yr (USD, before tax)")
    mom, lev = payload["momentum"], payload["leverage"]
    ba = payload["benchmarks"]["momentum_window"]
    info = payload["method"]["momentum"]["window"]
    L.append("")
    L.append(f"A. MOMENTUM ROTATION  train {info['train'][0]}..{info['train'][1]}, "
             f"test ..{info['test'][1]}   (CAGR %/yr; maxDD %; vsEW = CAGR minus the "
             "equal-weight buy-and-hold basket)")
    L.append(f"  {'variant':<31}{'train':>8}{'trainDD':>9}{'test':>8}{'testDD':>8}"
             f"{'pooled':>8}{'poolDD':>8}{'vsQQQ(test)':>12}{'vsEW(test)':>11}{'turn%/m':>8}")
    for name, v in mom["variants"].items():
        p = v["periods"]
        mark = " *" if name == mom["winner"]["name"] else ""
        L.append(f"  {name + mark:<31}{f(p['train']['cagr_pct']):>8}"
                 f"{f(p['train']['max_drawdown_pct']):>9}{f(p['test']['cagr_pct']):>8}"
                 f"{f(p['test']['max_drawdown_pct']):>8}{f(p['pooled']['cagr_pct']):>8}"
                 f"{f(p['pooled']['max_drawdown_pct']):>8}"
                 f"{f(p['test']['cagr_minus_qqq_pp'], 1, True):>12}"
                 f"{f(p['test']['cagr_minus_equal_weight_buy_and_hold_pp'], 1, True):>11}"
                 f"{f(p['pooled']['avg_turnover_pct_per_month'], 0):>8}")
    for k, label in (("QQQ", "QQQ"), ("SPY", "SPY"), (EW_BH, "equal weight, buy and hold"),
                     (EW_DAILY, "equal weight, daily rebal. (n/i)"),
                     (EW_ELIGIBLE, "equal weight, eligible monthly")):
        L.append(f"  {label:<31}{f(ba['train'][k]['cagr_pct']):>8}"
                 f"{f(ba['train'][k]['max_drawdown_pct']):>9}{f(ba['test'][k]['cagr_pct']):>8}"
                 f"{f(ba['test'][k]['max_drawdown_pct']):>8}{f(ba['pooled'][k]['cagr_pct']):>8}"
                 f"{f(ba['pooled'][k]['max_drawdown_pct']):>8}")
    fresh = ba["test"].get(EW_BH_FRESH) or {}
    L.append(f"  {'equal weight, bought at test start':<48}{f(fresh.get('cagr_pct')):>8}"
             f"{f(fresh.get('max_drawdown_pct')):>8}      (n/i = not investable)")
    w = mom["winner"]
    L.append(f"  * pre-registered winner: {w['name']}   best on TEST: "
             f"{w['best_test_variant']['name']} ({w['best_test_variant']['label']})")
    sm = w.get("selection_margin") or {}
    ex = (w.get("concentration") or {}).get("cagr_without_top_contributor_pct") or {}
    noise = w.get("test_noise") or {}
    if sm:
        L.append(f"    margin: TRAIN drawdown {f(sm.get('train_drawdown_distance_to_limit_pp'))} "
                 f"pp inside the limit, TRAIN CAGR {f(sm.get('train_cagr_gap_to_runner_up_pp'))} "
                 f"pp ahead of {sm.get('runner_up')}"
                 + (f"; without {ex['ticker']}: train {f(ex.get('train'))}, test "
                    f"{f(ex.get('test'))}, pooled {f(ex.get('pooled'))} %/yr" if ex else "")
                 + (f"; TEST noise about +/-{f(noise.get('standard_error_pp'), 0)} pp/yr"
                    if noise else ""))
    pl = mom["survivorship"].get("placebo") or {}
    if pl:
        rn, bn = pl["random_n"], pl["bottom_n"]
        L.append("    placebo (same rules, other picks; pooled %/yr): random median "
                 f"{f((rn.get('pooled') or {}).get('median_cagr_pct'))}, 95th pct "
                 f"{f((rn.get('pooled') or {}).get('p95_cagr_pct'))}; weakest-momentum picks "
                 f"train {f(bn['train']['cagr_pct'])}, test {f(bn['test']['cagr_pct'])}, pooled "
                 f"{f(bn['pooled']['cagr_pct'])}")
    L.append("  ! survivorship: today's index members - absolute CAGRs are biased upward "
             f"(buy-and-hold universe minus SPY: "
             f"{f(mom['survivorship']['universe_minus_spy_pp']['pooled'], 1, True)}"
             " pct-pt/yr pooled)")

    binfo = payload["method"]["leverage"]["window"]
    boot = lev["bootstrap"]["variants"]
    L.append("")
    L.append(f"B. LEVERAGED NASDAQ-100  {binfo['first_day']}..{binfo['last_bar']}, train to "
             f"{B_TRAIN_END}   (CAGR %/yr; maxDD %; 10y = rolling 10-year windows)")
    L.append(f"  {'variant':<20}{'train':>7}{'trainDD':>9}{'test':>7}{'testDD':>8}{'full':>7}"
             f"{'fullDD':>8}{'worstYr':>9}{'sw/yr':>7}{'inMkt%':>8}{'10y min':>9}{'10y med':>9}"
             f"{'10y>=40%':>10}{'P(>=40)':>9}{'P(DD>50)':>10}{'P(DD>75)':>10}{'P(loss)':>9}"
             f"{'afterTax':>10}")
    for name, v in lev["variants"].items():
        p = v["periods"]
        r, b = v["rolling_10y"] or {}, boot.get(name) or {}
        mark = " *" if name == lev["winner"]["name"] else ""
        wy = p["full"]["worst_year"] or {}
        L.append(f"  {name + mark:<20}{f(p['train']['cagr_pct']):>7}"
                 f"{f(p['train']['max_drawdown_pct']):>9}{f(p['test']['cagr_pct']):>7}"
                 f"{f(p['test']['max_drawdown_pct']):>8}{f(p['full']['cagr_pct']):>7}"
                 f"{f(p['full']['max_drawdown_pct']):>8}{f(wy.get('return_pct')):>9}"
                 f"{f(p['full'].get('switches_per_year')):>7}"
                 f"{f(p['full'].get('time_in_market_pct')):>8}{f(r.get('min_cagr_pct')):>9}"
                 f"{f(r.get('median_cagr_pct')):>9}{f(r.get('share_ge_target_pct')):>10}"
                 f"{f(b.get('p_cagr_ge_40_pct')):>9}"
                 f"{f(b.get('p_max_drawdown_worse_than_50_pct')):>10}"
                 f"{f(b.get('p_max_drawdown_worse_than_75_pct')):>10}"
                 f"{f(b.get('p_end_below_start_pct')):>9}"
                 f"{f((p['full'].get('after_tax_estimate_de') or {}).get('cagr_pct')):>10}")
    bb = payload["benchmarks"]["leverage_window"]
    for k in ("ndx_price_index", "QQQ", "SPY"):
        L.append(f"  {k:<20}{f(bb['train'][k].get('cagr_pct')):>7}"
                 f"{f(bb['train'][k].get('max_drawdown_pct')):>9}"
                 f"{f(bb['test'][k].get('cagr_pct')):>7}"
                 f"{f(bb['test'][k].get('max_drawdown_pct')):>8}"
                 f"{f(bb['full'][k].get('cagr_pct')):>7}"
                 f"{f(bb['full'][k].get('max_drawdown_pct')):>8}")
    L.append("  (10y>=40% = share of overlapping windows, not a probability; P(..) = block "
             "bootstrap of the past; afterTax = whole-history CAGR after an estimate of German "
             "withholding tax)")
    w = lev["winner"]
    L.append(f"  * pre-registered winner: {w['name']}   best on TEST: "
             f"{w['best_test_variant']['name']} ({w['best_test_variant']['label']})")
    sm = w.get("selection_margin") or {}
    if sm:
        L.append(f"    margin: TRAIN CAGR {f(sm.get('train_cagr_gap_to_runner_up_pp'), 2)} pp "
                 f"ahead of {sm.get('runner_up')}, TRAIN drawdown "
                 f"{f(sm.get('train_drawdown_distance_to_limit_pp'))} pp inside the limit")
    ts = lev.get("timing_sensitivity") or {}
    if ts:
        L.append(f"    sensitivity, switching at the signal's own close: winner would be "
                 f"{ts.get('winner_same_close')}"
                 + (f" (the chosen rule's TRAIN drawdown: "
                    f"{f((ts.get('winner_train_same_close') or {}).get('max_drawdown_pct'))}%)"
                    if ts.get("winner_train_same_close") else ""))
    cf = lev.get("conservative_financing") or {}
    if cf:
        L.append(f"    sensitivity, dearer financing: winner would be {cf.get('winner')}"
                 + (" (unchanged)" if not cf.get("winner_changes") else " (CHANGED)"))
    hc = ((lev["bootstrap"].get("sensitivity") or {}).get("drift_haircut") or {})
    if hc.get("variants"):
        L.append(f"    sensitivity, index return {f(hc['haircut_pp_per_year'], 0)} pp/yr lower: "
                 "P(10y >= 40%) " + ", ".join(
                     f"{n_} {f((hc['variants'].get(n_) or {}).get('p_cagr_ge_40_pct'))}%"
                     for n_ in lev["variants"] if (boot.get(n_) or {}).get("p_cagr_ge_40_pct")))
    L.append("  synthetic fund vs real ETF (CAGR, synthetic minus real):")
    for etf in B_ETF.values():
        v = lev["validation"].get(etf) or {}
        if v.get("error"):
            L.append(f"    {etf}: {v['error']}")
            continue
        L.append(f"    {etf:<5}{v['start']}..{v['end']}  synthetic {f(v['synthetic_cagr_pct'], 2)}%"
                 f"  real {f(v['real_cagr_pct'], 2)}%  diff {f(v['cagr_diff_pp'], 2, True)} pp"
                 f"  corr {v['daily_return_correlation']}"
                 + ("" if v["within_tolerance"] else "   <-- OUTSIDE the 1.0 pp tolerance"))
        regimes = [r for r in v.get("by_tbill_regime") or []
                   if r["synthetic_minus_real_pp_per_year"] is not None]
        if regimes:
            L.append("         by T-bill regime: " + "; ".join(
                f"{r['regime']} {f(r['synthetic_minus_real_pp_per_year'], 2, True)} pp/yr"
                + ("" if r["within_tolerance"] else " <-- OUTSIDE")
                for r in regimes))
    a = payload["answer"]

    def verdict(fam: str, why: str) -> str:
        if a["reaches_target"][fam]:
            return "YES"
        if a["reaches_target_on_paper"][fam]:
            return f"NO (on paper yes - {why}, not usable as evidence)"
        return "NO"

    L.append("")
    L.append(f"ANSWER  reaches {payload['target_cagr_pct']:.0f}%/yr: momentum "
             f"{verdict('momentum', 'survivorship-inflated')}, leverage "
             f"{verdict('leverage', 'the fund model missed the real funds')}")
    bt = a["best_tested"]
    if bt:
        L.append(f"  best tested (hindsight, usable family only): {bt['name']}  "
                 f"{_yr(bt['history'][0])}-{_yr(bt['history'][1])} {f(bt['cagr_full'])}%/yr, "
                 f"test {f(bt['cagr_test'])}%/yr, max drawdown {f(bt['max_drawdown'])}%; "
                 f"P(10y >= 40%) {f(bt['p_10y_ge_40'])}%, P(DD > 50%) {f(bt['p_dd_worse_50'])}%, "
                 f"P(DD > 75%) {f(bt['p_dd_worse_75'])}%, P(loss) {f(bt['p_end_below_start'])}%")
    else:
        L.append(f"  best tested: none - {a.get('best_tested_note')}")
    bb_ = a.get("best_tested_biased")
    if bb_:
        L.append(f"  inflated, NOT evidence: {bb_['name']}  "
                 f"{_yr(bb_['history'][0])}-{_yr(bb_['history'][1])} {f(bb_['cagr_full'])}%/yr, "
                 f"test {f(bb_['cagr_test'])}%/yr, max drawdown {f(bb_['max_drawdown'])}% "
                 "(today's index members; no 10-year history)")
    L.append("  " + a["text"])
    L.append("")
    L.append("CAVEATS")
    for c in payload["caveats"]:
        L.append("  - " + c)
    return L


# --------------------------------------------------------------------------- CLI

USAGE = "usage: python -m atlas2 strategy-lab [--offline] [--out PATH] [--refresh]"


def parse_opts(opts: list[str]) -> dict:
    """Strict option parsing; raises ValueError on anything unexpected."""
    out: dict = {"offline": False, "refresh": False, "out": None}
    k = 0
    while k < len(opts):
        name, eq, val = opts[k].partition("=")
        if name in ("--offline", "--refresh"):
            if eq:
                raise ValueError(f"{name} takes no value")
            out[name[2:]] = True
        elif name == "--out":
            if not eq:
                if k + 1 >= len(opts):
                    raise ValueError("--out needs a path")
                k += 1
                val = opts[k]
            if not val:
                raise ValueError("--out needs a path")
            out["out"] = val
        else:
            raise ValueError(f"unknown option: {opts[k]}")
        k += 1
    if out["offline"] and out["refresh"]:
        raise ValueError("use either --offline or --refresh, not both")
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    try:
        o = parse_opts(argv)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)
    try:
        run_strategy_lab(out_path=o["out"], offline=o["offline"], refresh=o["refresh"])
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
