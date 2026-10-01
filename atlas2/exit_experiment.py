"""Exit experiment (pre-registered): keep the backtest's ENTRIES fixed, vary only
the exits, judge on a train/test split decided in advance.

Why: the v2 backtest (data/backtest_latest.json) shows no pattern beating the
market after fees, and a post-hoc slice suggested the exits are the problem
(54% of trades stop out at -1.12R on average, tight stops lose, trades that
survive the 40-bar time exit win 75%, fee drag 0.44% per trade). That slice was
found by looking at 8 cuts on the pooled 2017-2026 sample, which includes the
TEST years, so the split here is only quasi out-of-sample (the exits were not
tuned on TEST, but the hypothesis was formed with TEST in view); the design
was fixed before any result of this experiment was seen, and the only true
out-of-sample check is the live journal from the pre-registration date on.

Design
  Entries: every closed trade of data/backtest_trades.json (scanner mode,
    shortlist applied): ticker, market, pattern, signal_date, entry_date (fill
    date), entry_price (fill), entry_kind, stop (pattern stop), target. Entries
    are NOT re-selected. Under each variant a ticker's entries are processed
    chronologically and an entry is SKIPPED while that ticker's previous trade
    under the same variant is still open (counted per variant and period). The
    test is on the SIGNAL bar, as backtest.trade_ticker's is: a signal raised
    on or before the previous trade's exit bar is not traded (the live scanner
    does not re-scan a stock until the bar after its exit).
  Stops (initial stop, checked from the fill bar on with the backtest's
    fill-bar rule: low <= stop on the fill bar = stopped at the stop):
    S1 the pattern stop as stored (control);
    S2 fill - 3 x ATR14, ATR14 of the SIGNAL bar (indicators.add_indicators on
       the same repaired series the backtest used);
    S3 fill - 5 x ATR14.
    An ATR stop at or above the fill falls back to the pattern stop (counted).
  Exits:
    E1 current: target (a gap above it exits at the open) + time exit at the
       close of bar fill+40 - backtest.simulate_exit itself;
    E2 long hold: no target; time exit at the close of bar fill+120; the
       initial stop still applies (simulate_exit with an unreachable target);
    E3 trailing: no target; after the fill bar, a bar that CLOSES below the
       lowest low since the fill, capped at the previous 20 bars, exits at the
       NEXT bar's open (so the window is short early in the trade: a one-bar
       low break on day 2 already exits); the initial stop applies every bar
       (a gap through the stop at the open, then the stop, before the trailing
       check); hard time exit at the close of bar fill+250.
    9 variants: S{1,2,3} x E{1,2,3}. S1xE1 is checked against the stored
    trades (exit day, outcome, exit price) and the match rate is reported.
    No risk cap applies to the ATR stops (S3 can imply a risk above 30% of the
    fill); S2 is fill - 3xATR as pre-registered even where that is TIGHTER
    than the pattern stop (counted: atr_stop_tighter_than_pattern).
  Censoring: the variant's full hold window (fill + max_hold bars) must fit in
    the stock's data, decided at the FILL as backtest.trade_ticker does, so it
    does not depend on the outcome (a trade that would have stopped out within
    the data is censored all the same). Censored entries are EXCLUDED from that
    variant's statistics and portfolio replay (counted); the last close is
    never used. A censored entry still blocks later entries of its ticker under
    that variant, as an open position would. Because the cutoff differs per
    hold (40 / 120 / 250 bars before the data end), every number decide()
    reads - winner and control statistics, verdict, portfolio replays and
    benchmarks - is computed on the COMMON entry set: entries whose fill has a
    full 250-bar window ('common' flag on each record; the last eligible entry
    date is reported). Each variant's numbers on its own full eligible window
    are reported separately under 'full_window'.
  Costs: backtest.trade_returns with the costs the backtest recorded
    (method.costs_pct_per_side etc.): costs on entry and exit, stop slippage
    on stop exits (a gap through the stop included), entry slippage on
    buy-stop entries. A trailing exit is a market order at the next open, so
    it pays the market's entry_slippage_pct as exit slippage. Target exits,
    including a gap above the target (a limit fill), and time exits pay no
    slippage, exactly as the backtest.
  Baselines per trade over the trade's own dates, no costs: equal-weight
    universe (backtest.MarketPanel) and the index with the backtest's
    dividend-yield correction (backtest.index_return), both to the CLOSE of the
    baseline exit date. Stop, target and time exits use the exit date (as the
    backtest does, so the control's baselines match the stored ones); an E3
    trailing exit fills at the OPEN, so its baseline runs to the close of the
    previous bar (baseline_exit_date), the last bar fully held.
  Split by ENTRY date: train < 2023-01-01 <= test. Every statistic is given
    for train, test and pooled.
  Primary metric: excess_vs_universe_pct (per-trade % after fees minus the
    equal-weight market over the same days). r_net is per unit of the VARIANT
    stop, so it is NOT comparable across stop variants.
  95% ranges: backtest.cluster_stats (SE = max of the CR1 SEs clustered by
    entry month and quarter, t with quarters - 1 df) for E1; E2 holds outlast
    a quarter, so its SE also considers the half-year clustering, and E3's the
    half-year and the year clustering, each taking the df of the clustering
    that binds (se_basis). A coarser clustering counts only with at least 4
    clusters.
  Portfolio replay per variant (train, test, pooled windows): the backtest's
    'SYSTEM AS DESIGNED' profile via portfolio_sim.run_profile (1% risk,
    regime multipliers, position cap, cash limit, median of the slot
    reshuffles), sized on the VARIANT stop; drawdown marked to market
    (portfolio_sim.mark_to_market); vs the blended index of the same window
    (portfolio_sim.benchmark_block). The shared window of a period ends at the
    last exit of ANY variant's common trade, and the common set stops entries
    at the 250-bar cutoff, so a shorter hold sits in cash for a tail that the
    truncation created, not the rule: with a positive return the shared window
    understates its CAGR (the same final equity annualised over more years)
    and, when the tail market is up, overstates its shortfall vs the index;
    both reverse when negative. The shared-window figures stay the
    pre-registered headline; next to them each exit rule gets an
    outcome-independent OWN window (own_windows: first common entry to the
    last common entry + max_hold benchmark bars) with its own benchmark and
    replay (portfolio.<period>.own_window, paired_test_cagr.own_window), and
    every block reports last_exit_date and cash_tail_days. decide() appends
    the control's tail to the verdict reason when it exceeds 10% of the TEST
    window; the verdict rule itself is unchanged.
  Winner rule (fixed before results): the variant with the highest TRAIN
    excess_vs_universe_pct whose TRAIN 95% range lies above 0; if none, the
    highest TRAIN excess, flagged 'no variant is significantly positive in
    training'. Verdict from its TEST figures next to the control's (S1xE1):
    'confirmed' when the winner's TEST range lies above 0 AND its TEST
    portfolio CAGR beats the control's - paired by tie-break seed: the winner
    must beat the control in at least 90% of the slot orderings (a margin
    against slot-ordering noise; the medians are kept for display);
    'improves but unproven' when its TEST excess beats the control's but the
    range includes 0; else 'not confirmed'. The best TEST variant is also
    named, labelled in-sample.

Price scale: the price cache is re-downloaded (auto-adjusted) by every scan,
so a dividend paid after the backtest ran rescales that stock's whole history.
The stored levels of such a ticker are mapped onto today's bars with a factor
estimated from its own trades (next-open fills, gap fills, time-exit closes,
gap exits); returns and R are scale-free. Tickers without such an anchor are
assumed unscaled and counted; a wrong assumption shows up in the S1xE1 check.

Run: python -m atlas2 exit-experiment [--limit N] [--out PATH] [--trades PATH]
                                      [--workers N]
--limit N uses the first N entries and writes to --out (default
data/exit_experiment_smoke.json), never to the real output nor its _trades
file; it exits with status 2 when S1xE1 does not reproduce the stored trades.
Output: data/exit_experiment.json and data/exit_experiment_trades.json.
Results are identical with any --workers (per-ticker jobs are pure). The price
cache is opened read-only and never refreshed during the run; its last bar
date and file time are recorded under method.source.
"""
from __future__ import annotations

import dataclasses
import json
import math
import sqlite3
import statistics
import sys
import time
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .backtest import (MARKET_ORDER, BenchContext, MarketPanel, Settings, _date_strings,
                       _default_workers, _mean, _pct, _rnd, cluster_se, cluster_stats,
                       index_return, load_long_prices, sanitize_bars, sanitize_long,
                       simulate_exit, t_crit, trade_returns, write_json_atomic)
from .data import BENCHMARKS
from .indicators import add_indicators
from .portfolio_sim import (DEFAULT_MAX_POSITION_PCT, START_CAPITAL, TIE_RESHUFFLES, _connect_ro,
                            _load_bars, _quantiles, benchmark_block, dividend_yields,
                            mark_to_market, price_repairs, regime_multipliers, run_profile,
                            simulate_portfolio)
from .scan import load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRADES_PATH = ROOT / "data" / "backtest_trades.json"
DEFAULT_OUT_PATH = ROOT / "data" / "exit_experiment.json"
DEFAULT_SMOKE_OUT_PATH = ROOT / "data" / "exit_experiment_smoke.json"
DEFAULT_PRICES_PATH = ROOT / "data" / "cache" / "prices.sqlite3"
DEFAULT_UNIVERSE_DIR = ROOT / "universe"

SPLIT_DATE = "2023-01-01"          # entries before it = train, from it = test
SPLIT_PERIODS = ("train", "test")
PERIODS = ("train", "test", "pooled")
CONTROL = "S1xE1"
PRIMARY_METRIC = "excess_vs_universe_pct"
SCALE_NOISE = 2e-5                 # |factor - 1| below this = 4-decimal rounding, not a rescale
PRICE_TOL = 1e-4                   # control check: relative tolerance on the exit price

# stop rules: (kind, ATR multiple)
STOPS = {"S1": ("pattern", None), "S2": ("atr", 3.0), "S3": ("atr", 5.0)}
# exit rules: target kept?, time exit bar (after the fill), trailing look-back
EXITS = {"E1": {"target": True, "max_hold": 40, "trail": None},
         "E2": {"target": False, "max_hold": 120, "trail": None},
         "E3": {"target": False, "max_hold": 250, "trail": 20}}
VARIANTS = [f"{sk}x{ek}" for sk in STOPS for ek in EXITS]
# the common entry set: entries whose fill leaves room for the LONGEST hold
LONGEST_HOLD = max(r["max_hold"] for r in EXITS.values())
# 95% ranges: holds longer than a quarter / half a year add the coarser clusterings
QUARTER_BARS = 63
HALF_YEAR_BARS = 126
MIN_CLUSTERS = 4                   # a coarser clustering counts only with this many clusters (df >= 3)
PAIRED_CAGR_SHARE = 0.9            # 'confirmed': winner CAGR above the control's in this share of seed pairs
RISK_FLAG_PCT = 30.0               # diagnostics: ATR stops implying more than this risk
CASH_TAIL_FLAG_SHARE = 0.10        # decide() notes the control's cash tail above this share of the window
# the own-window replay reports these keys of portfolio_replay (no marks: CAGR, years, index)
OWN_WINDOW_KEYS = ("window", "cagr_pct", "years", "blended_index_cagr_pct", "vs_blended_index_pp",
                   "cagr_range", "trades_taken", "final_equity", "last_exit_date",
                   "cash_tail_days")

STOP_TEXT = {
    "S1": "pattern stop as stored in the backtest trade (control)",
    "S2": "fill minus 3 x ATR14 of the signal bar (pattern stop if that is at or above the fill); "
          "as pre-registered even where it is TIGHTER than the pattern stop, so S2 is not a "
          "clean 'wider stop' test - see atr_stop_tighter_than_pattern; no risk cap",
    "S3": "fill minus 5 x ATR14 of the signal bar (pattern stop if that is at or above the fill); "
          "no risk cap - see risk_over_30_pct",
}
EXIT_TEXT = {
    "E1": "current rules: target (a gap above it exits at the open) or time exit at the close of "
          "bar fill+40; stop and target stay live through that bar (backtest.simulate_exit)",
    "E2": "long hold: no target; time exit at the close of bar fill+120; the initial stop still "
          "applies (gap through it at the open, else at the stop)",
    "E3": "trailing: no target; after the fill bar, a close below the lowest low SINCE THE FILL, "
          "capped at the previous 20 bars, exits at the NEXT bar's open (the window is short "
          "early in the trade: a one-bar low break on day 2 already exits - see "
          "trail_exits_within_20_bars / trail_exits_on_bar_2); the initial stop applies every "
          "bar (gap-through at the open, then the stop, before the trailing check); hard time "
          "exit at the close of bar fill+250",
}
ENTRY_KEYS = ("ticker", "market", "pattern", "signal_date", "entry_date", "entry_price",
              "entry_kind", "stop", "target")
# stored fields carried along: for the price-scale anchors and the control check
# (exit_date, outcome, ret_pct, r, entry_level, target), for the portfolio ranking
# (conf, composite_proxy, grade_proxy) and for reading (regime, tscore, rr, rank)
CARRY_KEYS = ("entry_level", "exit_date", "outcome", "ret_pct", "r", "conf", "composite_proxy",
              "grade_proxy", "regime", "tscore", "rr", "shortlist_rank")
COUNT_KEYS = ("entries", "trades", "skipped_overlap", "censored", "bar_missing",
              "atr_fallback", "atr_missing")


# --------------------------------------------------------------------------- entries

def period_of(entry_date: str) -> str:
    return "train" if str(entry_date)[:10] < SPLIT_DATE else "test"


def entries_from_trades(trades: list) -> tuple[list[dict], dict]:
    """The fixed entry set: closed trades with every required field, in file
    order. Returns (entries, {"open_at_end": n, "invalid": n})."""
    counts = {"open_at_end": 0, "invalid": 0}
    out: list[dict] = []
    for t in trades:
        if not isinstance(t, dict):
            counts["invalid"] += 1
            continue
        if t.get("outcome") == "open_at_end":
            counts["open_at_end"] += 1
            continue
        if any(t.get(k) is None for k in ENTRY_KEYS):
            counts["invalid"] += 1
            continue
        try:
            e = {k: t[k] for k in ENTRY_KEYS}
            for k in ("entry_price", "stop", "target"):
                e[k] = float(e[k])
            if not (e["entry_price"] > 0 and e["stop"] > 0 and e["target"] > 0
                    and e["entry_price"] > e["stop"]):
                counts["invalid"] += 1
                continue
        except (TypeError, ValueError):
            counts["invalid"] += 1
            continue
        for k in ("ticker", "market", "pattern", "signal_date", "entry_date", "entry_kind"):
            e[k] = str(e[k])
        for k in CARRY_KEYS:
            e[k] = t.get(k)
        e["period"] = period_of(e["entry_date"])
        out.append(e)
    return out, counts


def period_years(entries: list[dict]) -> dict[str, float | None]:
    """Span of the entry dates per period (the same denominator for every
    variant's trades_per_year), at least half a year."""
    out: dict[str, float | None] = {}
    for p in PERIODS:
        ds = [e["entry_date"] for e in entries if p == "pooled" or e["period"] == p]
        if not ds:
            out[p] = None
            continue
        days = (date.fromisoformat(max(ds)[:10]) - date.fromisoformat(min(ds)[:10])).days
        out[p] = max(0.5, days / 365.25)
    return out


# --------------------------------------------------------------------------- price scale

def price_scale_factor(entries: list[dict], pos: dict, o: np.ndarray, c: np.ndarray
                       ) -> tuple[float, int]:
    """Factor that maps a ticker's stored trade levels onto today's (repaired) bars.

    Anchors, each giving bar price / stored price: next-open fills and buy-stop
    fills that gapped over the level (fill = that day's open); time exits (exit
    = that day's close); gap-through-stop and gap-above-target exits (exit =
    that day's open). The stored exit is fill x (1 + ret_pct/100). Median of the
    anchors; within SCALE_NOISE of 1 it is treated as exactly 1 (rounding).
    Returns (factor, number of anchors); (1.0, 0) when nothing can be said."""
    est: list[float] = []
    for e in entries:
        fill = float(e["entry_price"])
        if not fill > 0:
            continue
        fi = pos.get(str(e["entry_date"])[:10])
        if fi is not None and o[fi] > 0:
            level = e.get("entry_level")
            gapped = level is not None and round(float(level), 4) != round(fill, 4)
            if e.get("entry_kind") == "next_open" or gapped:
                est.append(float(o[fi]) / fill)
        xi = pos.get(str(e.get("exit_date") or "")[:10])
        ret = e.get("ret_pct")
        if xi is None or ret is None:
            continue
        implied = fill * (1 + float(ret) / 100)
        if not implied > 0:
            continue
        outcome = e.get("outcome")
        r = e.get("r")
        if outcome == "time" and c[xi] > 0:
            est.append(float(c[xi]) / implied)
        elif outcome == "stop" and r is not None and abs(float(r) + 1) > 1e-3 and o[xi] > 0:
            est.append(float(o[xi]) / implied)          # gap through the stop: exit at the open
        elif (outcome == "target" and e.get("target") and o[xi] > 0
              and abs(implied / float(e["target"]) - 1) > 1e-4):
            est.append(float(o[xi]) / implied)          # gap above the target: exit at the open
    if not est:
        return 1.0, 0
    med = float(statistics.median(est))
    if not math.isfinite(med) or med <= 0 or abs(med - 1) < SCALE_NOISE:
        return 1.0, len(est)
    return med, len(est)


# --------------------------------------------------------------------------- exit mechanics

def atr_stop(fill: float, atr: float | None, mult: float, pattern_stop: float
             ) -> tuple[float, str]:
    """(stop, kind): fill - mult x ATR ('atr'); the pattern stop when the ATR
    is missing ('atr_missing') or the ATR stop is at or above the fill
    ('pattern_fallback')."""
    if atr is None or not math.isfinite(atr) or atr <= 0:
        return float(pattern_stop), "atr_missing"
    stop = fill - mult * atr
    if stop >= fill:
        return float(pattern_stop), "pattern_fallback"
    return float(stop), "atr"


def simulate_trailing_exit(o, h, l, c, fill_i: int, stop: float, max_hold: int,
                           trail_bars: int) -> tuple[int, float, str] | None:
    """E3: walk forward from the fill bar. Returns (exit_i, exit_price, outcome)
    with outcome stop / trail / time, or None when the data ends first.

    Fill bar: a touched stop is a stop (no trailing check on the fill bar).
    Later bars, in this order: a gap through the stop exits at the open
    ('stop'); a trailing exit signalled by the previous close exits at the
    open ('trail'); a touched stop exits at the stop; bar fill_i + max_hold
    exits at its close ('time'); otherwise a close below the lowest low of bars
    max(fill_i, j - trail_bars) .. j-1 signals an exit at the next open.
    """
    n = len(c)
    last = fill_i + max_hold
    pending = False
    for j in range(fill_i, min(last, n - 1) + 1):
        if j > fill_i:
            if o[j] <= stop:
                return j, float(o[j]), "stop"
            if pending:
                return j, float(o[j]), "trail"
            if l[j] <= stop:
                return j, float(stop), "stop"
        elif l[j] <= stop:
            return j, float(stop), "stop"
        if j == last:
            return j, float(c[j]), "time"
        if j > fill_i and c[j] < float(np.min(l[max(fill_i, j - trail_bars): j])):
            pending = True
    return None


def variant_exit(o, h, l, c, fill_i: int, fill: float, stop: float, target: float,
                 rule: dict) -> tuple[int, float, str] | None:
    """One exit rule from the fill bar. E1 and E2 are backtest.simulate_exit
    (E2 with an unreachable target); E3 is simulate_trailing_exit. None = the
    data ends before the exit (censored)."""
    if rule["trail"] is not None:
        return simulate_trailing_exit(o, h, l, c, fill_i, stop, rule["max_hold"], rule["trail"])
    tgt = float(target) if rule["target"] else math.inf
    exit_i, exit_price, outcome = simulate_exit(o, h, l, c, fill_i, fill, stop, tgt,
                                                rule["max_hold"])
    if outcome == "open_at_end":
        return None
    return exit_i, exit_price, outcome


def variant_returns(fill: float, stop: float, exit_price: float, outcome: str, market: str,
                    s: Settings, entry_kind: str) -> dict:
    """backtest.trade_returns with the backtest's cost model: costs both ways,
    stop slippage on stop exits, entry slippage on buy-stop entries. A trailing
    exit is a market order at the next open, so it pays the market's entry
    slippage as exit slippage (trade_returns charges exit slippage only for the
    'stop' outcome, so the trail is passed as one, with that slippage)."""
    entry_slip = s.entry_slip(market) if entry_kind == "buy_stop" else 0.0
    if outcome == "trail":
        return trade_returns(fill, stop, exit_price, "stop", s.cost_pct(market),
                             s.entry_slip(market), entry_slip)
    return trade_returns(fill, stop, exit_price, outcome, s.cost_pct(market),
                         s.stop_slip(market), entry_slip)


# --------------------------------------------------------------------------- one ticker

def _new_counts() -> dict:
    return {k: 0 for k in COUNT_KEYS}


def _add_counts(total: dict, part: dict) -> None:
    for k in COUNT_KEYS:
        total[k] += part.get(k, 0)


def _control_mismatch(entry: dict, rec: dict, scale: float) -> dict | None:
    """None when a S1xE1 trade reproduces the stored backtest trade: same exit
    day and outcome, and the same exit price on the stored price scale (the
    stored exit is fill x (1 + ret_pct/100); ret_pct is rounded to 3 decimals,
    so PRICE_TOL relative). Otherwise the details."""
    stored_ret, stored_r = entry.get("ret_pct"), entry.get("r")
    fill_stored = float(entry["entry_price"])
    stored_exit = fill_stored * (1 + float(stored_ret) / 100) if stored_ret is not None else None
    ours_exit = float(rec["exit_price"]) / scale
    same_date = rec["exit_date"] == entry.get("exit_date")
    same_outcome = rec["outcome"] == entry.get("outcome")
    price_ok = stored_exit is not None and stored_exit > 0 and \
        abs(ours_exit / stored_exit - 1) <= PRICE_TOL
    if same_date and same_outcome and price_ok:
        return None
    return {"ticker": entry["ticker"], "market": entry["market"], "entry_date": entry["entry_date"],
            "pattern": entry["pattern"], "price_scale": round(scale, 6),
            "stored": {"exit_date": entry.get("exit_date"), "outcome": entry.get("outcome"),
                       "exit_price": _rnd(stored_exit, 4), "ret_pct": stored_ret, "r": stored_r},
            "got": {"exit_date": rec["exit_date"], "outcome": rec["outcome"],
                    "exit_price": _rnd(ours_exit, 4), "ret_pct": rec["ret_pct"], "r": rec["r"]}}


def _ticker_result(ticker: str, market: str) -> dict:
    return {"ticker": ticker, "market": market, "records": [],
            "counts": {v: {p: _new_counts() for p in SPLIT_PERIODS} for v in VARIANTS},
            # the same counts restricted to the COMMON entry set (full LONGEST_HOLD window)
            "counts_common": {v: {p: _new_counts() for p in SPLIT_PERIODS} for v in VARIANTS},
            "common_entry_dates": [],
            # last entry date whose fill leaves room for each exit rule's full hold
            "last_eligible_entry": {ek: None for ek in EXITS},
            "control_compared": 0, "control_mismatches": [],
            "price_scale": 1.0, "scale_estimates": 0, "atr_bar_shifted": 0,
            "price_events": [], "error": None}


def _bump(cnts: list[dict], key: str) -> None:
    for c_ in cnts:
        c_[key] += 1


def run_ticker(ticker: str, market: str, entries: list[dict], df: pd.DataFrame | None,
               overrides, stale_before: str | None, s: Settings) -> dict:
    """Every variant for one stock's entries on its repaired bars (pure).

    Overlap: an entry whose SIGNAL bar is on or before the previous trade's exit
    bar is skipped (backtest.trade_ticker re-scans a stock from the bar after
    its exit). Censoring: decided at the fill - the full hold window must fit in
    the data (fi + max_hold <= n - 1), as trade_ticker does; a censored entry
    still blocks the ticker's later entries. Each record carries 'common' =
    the fill leaves room for the LONGEST hold (the comparison set)."""
    out = _ticker_result(ticker, market)
    counts, counts_c = out["counts"], out["counts_common"]
    entries = sorted(entries, key=lambda e: (e["entry_date"], e["signal_date"], e["pattern"]))
    for v in VARIANTS:
        for e in entries:
            counts[v][e["period"]]["entries"] += 1
    n = 0 if df is None else len(df)
    if n >= 2:
        df, rep = sanitize_bars(df, ticker, overrides, stale_before)
        out["price_events"] = [list(x) for x in (rep.get("events") or [])]
        n = len(df)
    if n < 2:
        for v in VARIANTS:
            for e in entries:
                counts[v][e["period"]]["bar_missing"] += 1
        return out
    dates = _date_strings(df.index)
    pos = {str(d): k for k, d in enumerate(dates)}
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = add_indicators(df)["atr14"].to_numpy(float)
    scale, n_est = price_scale_factor(entries, pos, o, c)
    out["price_scale"], out["scale_estimates"] = scale, n_est

    prepared = []
    for e in entries:
        fi = pos.get(str(e["entry_date"])[:10])
        if fi is None:
            for v in VARIANTS:
                counts[v][e["period"]]["bar_missing"] += 1
            continue
        sd = str(e["signal_date"])[:10]
        k = pos.get(sd)
        if k is None:      # signal bar not in the repaired series: last bar on/before it
            k = int(np.searchsorted(dates, sd, side="right")) - 1
            out["atr_bar_shifted"] += 1
        atr_sig = float(atr[k]) if k >= 0 and math.isfinite(atr[k]) else None
        common = fi + LONGEST_HOLD <= n - 1
        if common:
            out["common_entry_dates"].append(e["entry_date"])
            for v in VARIANTS:
                counts_c[v][e["period"]]["entries"] += 1
        for ek, rule in EXITS.items():
            if fi + rule["max_hold"] <= n - 1:
                last = out["last_eligible_entry"][ek]
                if last is None or e["entry_date"] > last:
                    out["last_eligible_entry"][ek] = e["entry_date"]
        prepared.append((e, fi, k, atr_sig, common))

    for variant in VARIANTS:
        sk, ek = variant.split("x")
        stop_kind, mult = STOPS[sk]
        rule = EXITS[ek]
        busy_until = None                   # bar index through which the previous trade is open
        for e, fi, k, atr_sig, common in prepared:
            cnts = [counts[variant][e["period"]]]
            if common:
                cnts.append(counts_c[variant][e["period"]])
            if busy_until is not None and k <= busy_until:
                _bump(cnts, "skipped_overlap")
                continue
            fill = float(e["entry_price"]) * scale
            pattern_stop = float(e["stop"]) * scale
            target = float(e["target"]) * scale
            level = float(e["entry_level"]) * scale if e.get("entry_level") is not None else None
            if stop_kind == "pattern":
                stop, kind = pattern_stop, "pattern"
            else:
                stop, kind = atr_stop(fill, atr_sig, mult, pattern_stop)
                if kind == "pattern_fallback":
                    _bump(cnts, "atr_fallback")
                elif kind == "atr_missing":
                    _bump(cnts, "atr_missing")
            rec = {
                "variant": variant, "stop_rule": sk, "exit_rule": ek,
                "market": market, "ticker": ticker, "pattern": e["pattern"],
                "signal_date": e["signal_date"], "entry_date": e["entry_date"],
                "entry_kind": e["entry_kind"], "period": e["period"], "common": common,
                "entry_price": round(fill, 4), "entry_level": _rnd(level, 4),
                "price_scale": round(scale, 6),
                "stop": round(stop, 4), "stop_kind": kind, "stop_pattern": round(pattern_stop, 4),
                "atr14_signal": _rnd(atr_sig, 4),
                "risk_pct": round((fill - stop) / fill * 100, 3),
                "target": round(target, 4) if rule["target"] else None,
                "conf": e.get("conf"), "composite_proxy": e.get("composite_proxy"),
                "grade_proxy": e.get("grade_proxy"), "regime": e.get("regime"),
                "exit_date": None, "baseline_exit_date": None, "exit_price": None,
                "outcome": None, "hold_bars": None,
                "r": None, "r_net": None, "ret_pct": None, "ret_net_pct": None,
                "cost_pct_round_trip": None, "censored": False,
                "index_ret_pct": None, "universe_ret_pct": None,
                "excess_vs_index_pct": None, "excess_vs_universe_pct": None,
            }
            # censoring is decided at the fill (the full hold window must fit in the
            # data), never by the outcome; the None branch below is only a safety net
            res = None
            if fi + rule["max_hold"] <= n - 1:
                res = variant_exit(o, h, l, c, fi, fill, stop, target, rule)
            if res is None:
                rec["censored"] = True
                _bump(cnts, "censored")
                busy_until = n                  # open until the data ends: later entries skip
                out["records"].append(rec)
                continue
            exit_i, exit_price, outcome = res
            r = variant_returns(fill, stop, exit_price, outcome, market, s, e["entry_kind"])
            # a trailing exit fills at the OPEN of exit_i: the last bar fully held is
            # exit_i - 1, so the baseline runs to that bar's close
            base_i = exit_i - 1 if outcome == "trail" else exit_i
            rec.update({
                "exit_date": str(dates[exit_i]), "baseline_exit_date": str(dates[base_i]),
                "exit_price": round(float(exit_price), 4),
                "outcome": outcome, "hold_bars": int(exit_i - fi),
                "r": round(r["r"], 4), "r_net": round(r["r_net"], 4),
                "ret_pct": round(r["ret_pct"], 3), "ret_net_pct": round(r["ret_net_pct"], 3),
                "cost_pct_round_trip": round(r["cost_pct_round_trip"], 3),
            })
            _bump(cnts, "trades")
            busy_until = exit_i
            out["records"].append(rec)
            if variant == CONTROL:
                out["control_compared"] += 1
                mm = _control_mismatch(e, rec, scale)
                if mm is not None:
                    out["control_mismatches"].append(mm)
    return out


_WORKER_CONNS: dict[str, sqlite3.Connection] = {}


def _worker_conn(db_path: str) -> sqlite3.Connection:
    if db_path not in _WORKER_CONNS:
        conn = _connect_ro(Path(db_path))
        if conn is None:
            raise FileNotFoundError(f"price cache not found: {db_path}")
        _WORKER_CONNS[db_path] = conn
    return _WORKER_CONNS[db_path]


def _ticker_job(job: tuple) -> dict:
    """Top-level (picklable) worker: one stock, read-only price cache."""
    ticker, market, db_path, entries, overrides, stale_before, s = job
    try:
        df = _load_bars(_worker_conn(db_path), ticker)
        return run_ticker(ticker, market, entries, df, overrides, stale_before, s)
    except Exception as exc:  # one bad ticker must not kill the run
        out = _ticker_result(ticker, market)
        out["error"] = f"{exc.__class__.__name__}: {exc}"
        for v in VARIANTS:
            for e in entries:
                out["counts"][v][e["period"]]["entries"] += 1
                out["counts"][v][e["period"]]["bar_missing"] += 1
        return out


# --------------------------------------------------------------------------- baselines + stats

def add_variant_baselines(records: list[dict], panel: MarketPanel, bench: BenchContext,
                          div_yield_pct: float) -> int:
    """Index (with the dividend correction) and equal-weight universe return over
    each closed trade's own dates, no costs. Returns how many lack a baseline."""
    missing = 0
    for t in records:
        if t["censored"]:
            continue
        # a trailing exit fills at the open, so its baseline stops at the previous
        # bar's close (baseline_exit_date); other exits use the exit date, as the backtest
        exit_date = t.get("baseline_exit_date") or t["exit_date"]
        idx = index_return(bench.dates, bench.opens, bench.closes, t["entry_date"], exit_date,
                           div_yield_pct)
        uni = panel.universe_return(t["entry_date"], exit_date)
        t["index_ret_pct"] = _rnd(idx, 3)
        t["universe_ret_pct"] = _rnd(uni, 3)
        t["excess_vs_index_pct"] = None if idx is None else _rnd(t["ret_net_pct"] - idx, 3)
        t["excess_vs_universe_pct"] = None if uni is None else _rnd(t["ret_net_pct"] - uni, 3)
        if idx is None or uni is None:
            missing += 1
    return missing


def _half_year(d: str) -> str:
    return f"{d[:4]}H{1 if int(d[5:7]) <= 6 else 2}"


def _ci_block(values: list[float], dates: list[str], nd: int, max_hold: int = 40) -> dict:
    """95% range of the mean. Base: backtest.cluster_stats (SE = max of the CR1
    SEs clustered by entry month and quarter, t with quarters - 1 df, none below
    10 quarters). A hold longer than a quarter (E2) also considers the SE
    clustered by entry half-year, one longer than half a year (E3) the year as
    well: neighbouring quarters then share market weather. The largest SE
    binds, with the df of its own clustering (se_basis); a coarser clustering
    counts only with MIN_CLUSTERS clusters or more (t_crit is a Cornish-Fisher
    expansion, within about 1% from df 3)."""
    st = cluster_stats(values, dates)
    out = {"ci95": None, "se": None, "df": None, "se_basis": None,
           "se_month_quarter": None if st is None else _rnd(st["se"], 4),
           "se_half_year_cluster": None, "se_year_cluster": None}
    if not values:
        return out
    labels_h = [_half_year(d) for d in dates]
    labels_y = [d[:4] for d in dates]
    hy, yr = cluster_se(values, labels_h), cluster_se(values, labels_y)
    out["se_half_year_cluster"] = None if hy is None else _rnd(hy[1], 4)
    out["se_year_cluster"] = None if yr is None else _rnd(yr[1], 4)
    if st is None:
        return out
    se, df, basis = st["se"], st["df"], "month/quarter"
    coarser = []
    if max_hold > QUARTER_BARS:
        coarser.append(("half-year", hy, len(set(labels_h))))
    if max_hold > HALF_YEAR_BARS:
        coarser.append(("year", yr, len(set(labels_y))))
    for name, r, g in coarser:
        if r is not None and g >= MIN_CLUSTERS and r[1] > se:
            se, df, basis = r[1], g - 1, name
    crit = t_crit(0.975, df)
    out.update({"ci95": [_rnd(st["mean"] - crit * se, nd), _rnd(st["mean"] + crit * se, nd)],
                "se": _rnd(se, 4), "df": df, "se_basis": basis})
    return out


def period_stats(records: list[dict], counts: dict, years: float | None,
                 max_hold: int = 40) -> dict:
    """Statistics of one variant over one period (records = that variant's
    records whose entry falls in the period, censored ones included). max_hold
    picks the clusterings of the 95% range (_ci_block)."""
    taken = [t for t in records if not t["censored"]]
    n = len(taken)
    with_uni = [t for t in taken if t.get("excess_vs_universe_pct") is not None]
    with_idx = [t for t in taken if t.get("excess_vs_index_pct") is not None]
    ex_u = [t["excess_vs_universe_pct"] for t in with_uni]
    ex_i = [t["excess_vs_index_pct"] for t in with_idx]
    ci_u = _ci_block(ex_u, [t["entry_date"] for t in with_uni], 3, max_hold)
    ci_i = _ci_block(ex_i, [t["entry_date"] for t in with_idx], 3, max_hold)
    # pre-registration diagnostics: ATR stops tighter than the pattern stop, ATR stops
    # with extreme risk, E3 trail exits before a full 20-bar window exists
    tighter = sum(1 for t in taken if t.get("stop") is not None
                  and t.get("stop_pattern") is not None and t["stop"] > t["stop_pattern"])
    risky = sum(1 for t in taken if (t.get("risk_pct") or 0) > RISK_FLAG_PCT)
    trails = [t for t in taken if t["outcome"] == "trail"]
    trail_bars = EXITS["E3"]["trail"]
    early = sum(1 for t in trails if t["hold_bars"] is not None and t["hold_bars"] <= trail_bars)
    on_bar_2 = sum(1 for t in trails if t["hold_bars"] == 2)
    by_year_recs: dict[str, list[dict]] = defaultdict(list)
    for t in taken:
        by_year_recs[t["entry_date"][:4]].append(t)
    by_year = {}
    for y, g in sorted(by_year_recs.items()):
        gu = [t["excess_vs_universe_pct"] for t in g if t.get("excess_vs_universe_pct") is not None]
        by_year[y] = {"trades": len(g),
                      "win_rate": _pct(sum(1 for t in g if t["ret_net_pct"] > 0), len(g)),
                      "net_avg_ret_pct": _rnd(_mean([t["ret_net_pct"] for t in g]), 3),
                      "excess_vs_universe_pct": _rnd(_mean(gu), 3)}
    outcomes = {k: sum(1 for t in taken if t["outcome"] == k)
                for k in ("stop", "target", "time", "trail")}
    return {
        "entries": counts["entries"],
        "trades": n,
        "skipped_overlap": counts["skipped_overlap"],
        "censored": counts["censored"],
        "bar_missing": counts["bar_missing"],
        "atr_stop_fallback": counts["atr_fallback"],
        "atr_missing": counts["atr_missing"],
        "trades_per_year": _rnd(n / years, 1) if years else None,
        "win_rate": _pct(sum(1 for t in taken if t["ret_net_pct"] > 0), n),
        "stop_share": _pct(outcomes["stop"], n),
        "outcomes": outcomes,
        "avg_hold_bars": _rnd(_mean([t["hold_bars"] for t in taken]), 1),
        "avg_risk_pct": _rnd(_mean([t["risk_pct"] for t in taken]), 3),
        "net_avg_ret_pct": _rnd(_mean([t["ret_net_pct"] for t in taken]), 3),
        "avg_cost_pct_round_trip": _rnd(_mean([t["cost_pct_round_trip"] for t in taken]), 3),
        "avg_universe_ret_pct": _rnd(_mean([t["universe_ret_pct"] for t in with_uni]), 3),
        "avg_index_ret_pct": _rnd(_mean([t["index_ret_pct"] for t in with_idx]), 3),
        "excess_vs_universe_pct": _rnd(_mean(ex_u), 3),
        "excess_vs_universe_ci95": ci_u["ci95"],
        "excess_vs_universe_se": ci_u["se"],
        "excess_vs_universe_df": ci_u["df"],
        "excess_vs_universe_se_basis": ci_u["se_basis"],
        "excess_vs_universe_se_month_quarter": ci_u["se_month_quarter"],
        "excess_vs_universe_se_half_year_cluster": ci_u["se_half_year_cluster"],
        "excess_vs_universe_se_year_cluster": ci_u["se_year_cluster"],
        "excess_vs_universe_n": len(ex_u),
        "excess_vs_index_pct": _rnd(_mean(ex_i), 3),
        "excess_vs_index_ci95": ci_i["ci95"],
        "excess_vs_index_se_basis": ci_i["se_basis"],
        "excess_vs_index_se_half_year_cluster": ci_i["se_half_year_cluster"],
        "excess_vs_index_se_year_cluster": ci_i["se_year_cluster"],
        "excess_vs_index_n": len(ex_i),
        "net_avg_r": _rnd(_mean([t["r_net"] for t in taken]), 4),
        "net_avg_r_note": "R per unit of the VARIANT stop: not comparable across stop variants",
        "atr_stop_tighter_than_pattern": {"count": tighter, "share_pct": _pct(tighter, n)},
        "risk_over_30_pct": risky,
        "trail_exits": len(trails),
        "trail_exits_within_20_bars": {"count": early, "share_pct": _pct(early, len(trails))},
        "trail_exits_on_bar_2": on_bar_2,
        "by_year": by_year,
    }


# --------------------------------------------------------------------------- portfolio

def as_designed_settings(backtest: dict | None, cfg: dict | None) -> dict:
    """simulate_portfolio kwargs of the backtest's 'SYSTEM AS DESIGNED' profile
    (from the backtest file's portfolio block when present, else the profile's
    definition in portfolio_sim.PROFILES with the config's position cap)."""
    kw = {"risk_pct": 1.0, "max_pos": 6, "sizing": "dashboard",
          "max_position_pct": float((cfg or {}).get("max_position_pct", DEFAULT_MAX_POSITION_PCT)),
          "cash_limit": True, "rank_by": "composite_proxy", "grade_filter": None}
    profiles = ((backtest or {}).get("portfolio") or {}).get("profiles") or []
    for p in profiles:
        if isinstance(p, dict) and p.get("key") == "as_designed":
            for k in ("risk_pct", "max_pos", "sizing", "max_position_pct", "cash_limit",
                      "rank_by", "grade_filter"):
                if k in p and p[k] is not None:
                    kw[k] = p[k]
            break
    return kw


def _days_between(a: str, b: str) -> int:
    return (date.fromisoformat(str(b)[:10]) - date.fromisoformat(str(a)[:10])).days


def last_exit_and_tail(records: list[dict], window: tuple[str, str] | None) -> dict:
    """last_exit_date = the last exit of the closed records; cash_tail_days =
    the window's end minus it, i.e. the calendar days the variant sits in cash
    at the end of the window (negative when its last exit lies beyond the
    window's end). None without closed records or a window."""
    xs = [t["exit_date"] for t in records if not t["censored"] and t.get("exit_date")]
    last = max(xs) if xs else None
    tail = _days_between(last, window[1]) if last is not None and window else None
    return {"last_exit_date": last, "cash_tail_days": tail}


def portfolio_replay(records: list[dict], window: tuple[str, str], sim_kw: dict,
                     blended_idx: float | None, reshuffles: int,
                     conn: sqlite3.Connection | None, overrides=(), stale_map: dict | None = None,
                     cache: dict | None = None, *, mark: bool = True) -> dict:
    """portfolio_sim.run_profile (median of the slot orderings) on one variant's
    closed trades, sized on the VARIANT stop; drawdown marked to market when the
    helper allows it (mark=True), else the realized figure (stated in
    drawdown_basis). Reports the variant's last exit and the cash tail the
    window leaves after it (last_exit_and_tail)."""
    usable = [t for t in records if not t["censored"]]
    if not usable:
        return {"error": "no closed trades", "trades_taken": 0,
                "window": {"start": window[0], "end": window[1]},
                **last_exit_and_tail(usable, window)}
    r = run_profile(usable, {**sim_kw, "window": window}, blended_idx, reshuffles)
    if "error" in r:
        return {"error": r["error"], "trades_taken": 0,
                "window": {"start": window[0], "end": window[1]},
                **last_exit_and_tail(usable, window)}
    raw = r.pop("_cagr_raw", None)
    positions = r.pop("_positions", None)
    if mark:
        mtm, why = mark_to_market(positions or [], START_CAPITAL, conn, overrides=overrides,
                                  stale_map=stale_map, cache=cache)
    else:
        mtm, why = None, "not marked"
    tb = r.get("tie_break_sensitivity") or {}
    return {
        "window": {"start": window[0], "end": window[1]},
        **last_exit_and_tail(usable, window),
        "cagr_pct": r.get("cagr_pct"),
        "max_drawdown_pct": mtm["mtm_max_drawdown_pct"] if mtm else r.get("max_drawdown_pct"),
        "drawdown_basis": "marked to market daily" if mtm else f"closed trades only ({why})",
        "max_drawdown_realized_pct": r.get("max_drawdown_pct"),
        "ann_volatility_pct": mtm["ann_volatility_pct"] if mtm else None,
        "avg_gross_exposure_pct": (mtm["avg_gross_exposure_pct"] if mtm
                                   else r.get("avg_gross_exposure_pct")),
        "trades_taken": r.get("trades_taken"),
        "trades_available": r.get("trades_available"),
        "skipped_no_free_slot": r.get("skipped_no_free_slot"),
        "skipped_regime": r.get("skipped_regime"),
        "skipped_no_cash": r.get("skipped_no_cash"),
        "capped_trades": r.get("capped_trades"),
        "win_rate": r.get("win_rate"),
        "final_equity": r.get("final_equity"),
        "years": r.get("years"),
        "yearly_return_pct": r.get("yearly_return_pct"),
        "representative_tie_seed": r.get("representative_tie_seed"),
        "cagr_range": tb.get("cagr_pct"),
        "blended_index_cagr_pct": blended_idx,
        "vs_blended_index_pp": (_rnd(raw - blended_idx, 1)
                                if raw is not None and blended_idx is not None else None),
        "_cagr_raw": raw,
    }


OWN_WINDOW_NOTE = ("outcome-independent window of this EXIT RULE: the period's first common "
                   "entry to its last common entry + max_hold benchmark bars (the latest day a "
                   "trade entered on that date can still be held under the rule); same "
                   "settings and trades as the shared-window replay, annualised over this "
                   "window and measured against its own benchmark; no marks (CAGR only); not "
                   "the pre-registered headline")


def own_window_replay(records: list[dict], window: tuple[str, str] | None, sim_kw: dict,
                      blended_idx: float | None, reshuffles: int) -> dict:
    """The same replay as portfolio_replay, annualised over the exit rule's own
    window (own_windows) against that window's blended index; CAGR, years and
    the index figures only (OWN_WINDOW_KEYS), no marks."""
    if window is None:
        return {"error": "no window", "note": OWN_WINDOW_NOTE}
    pr = portfolio_replay(records, window, sim_kw, blended_idx, reshuffles, None, mark=False)
    out = {k: pr.get(k) for k in OWN_WINDOW_KEYS}
    if "error" in pr:
        out["error"] = pr["error"]
    out["note"] = OWN_WINDOW_NOTE
    return out


# --------------------------------------------------------------------------- decision

WINNER_RULE = ("winner = the variant with the highest TRAIN excess_vs_universe_pct whose TRAIN "
               "95% range lies above 0; if none qualifies, the highest TRAIN excess, flagged. "
               "Verdict on TEST, winner next to the control (S1xE1): 'confirmed' if the winner's "
               "TEST range lies above 0 AND its TEST portfolio CAGR beats the control's - paired "
               "by tie-break seed: the winner must beat the control in at least "
               f"{PAIRED_CAGR_SHARE:.0%} of the slot orderings (a margin against slot-ordering "
               "noise; the medians are kept for display); 'improves but unproven' if its TEST "
               "excess beats the control's but the range includes 0; else 'not confirmed'. "
               "Fixed before the results were seen; all statistics on the common entry set "
               "(fills with a full 250-bar window).")


def _paired_seeds(w_use: list[dict], c_use: list[dict], w_window: tuple[str, str],
                  c_window: tuple[str, str], sim_kw: dict, reshuffles: int
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """Raw CAGRs of the winner and the control for seeds 0..reshuffles, the same
    seed on both sides of a pair; each side annualised over its window."""
    w_raw, c_raw = [], []
    for sd in range(reshuffles + 1):
        rw = simulate_portfolio(w_use, tie_seed=sd, **{**sim_kw, "window": w_window})
        rc = simulate_portfolio(c_use, tie_seed=sd, **{**sim_kw, "window": c_window})
        if "error" in rw or "error" in rc:
            return None
        w_raw.append(rw["_cagr_raw"])
        c_raw.append(rc["_cagr_raw"])
    return np.array(w_raw), np.array(c_raw)


def _paired_block(w: np.ndarray, c: np.ndarray, w_use: list[dict], c_use: list[dict],
                  w_window: tuple[str, str], c_window: tuple[str, str], reshuffles: int) -> dict:
    wt, ct = last_exit_and_tail(w_use, w_window), last_exit_and_tail(c_use, c_window)
    return {"seeds": reshuffles + 1,
            "winner_beats_control_share": round(float((w > c).mean()), 3),
            "required_share": PAIRED_CAGR_SHARE,
            "winner_cagr_range": _quantiles(w),
            "control_cagr_range": _quantiles(c),
            "winner_minus_control_pp": _quantiles(w - c),
            "winner_last_exit_date": wt["last_exit_date"],
            "winner_cash_tail_days": wt["cash_tail_days"],
            "control_last_exit_date": ct["last_exit_date"],
            "control_cash_tail_days": ct["cash_tail_days"]}


def paired_cagr(winner_records: list[dict], control_records: list[dict],
                window: tuple[str, str], sim_kw: dict, reshuffles: int,
                own_windows: tuple[tuple[str, str] | None, tuple[str, str] | None] | None = None
                ) -> dict | None:
    """The CAGR leg of the verdict, paired by tie-break seed: simulate_portfolio
    for the winner and the control with the SAME seed over the same window and
    settings, seeds 0..reshuffles. Reports the share of seed pairs in which the
    winner's CAGR beats the control's and both quantile ranges side by side,
    plus each side's last exit and cash tail on that window.
    own_windows = (winner's own window, control's own window): when given, the
    same pairing with each side annualised over its own outcome-independent
    window is reported under 'own_window' (context, not the pre-registered leg).
    None when either side has no closed trades or a replay fails."""
    w_use = [t for t in winner_records if not t["censored"]]
    c_use = [t for t in control_records if not t["censored"]]
    if not w_use or not c_use:
        return None
    pair = _paired_seeds(w_use, c_use, window, window, sim_kw, reshuffles)
    if pair is None:
        return None
    w, c = pair
    out = {"window": {"start": window[0], "end": window[1]},
           **_paired_block(w, c, w_use, c_use, window, window, reshuffles),
           "note": "same tie-break seed for both sides in every pair (slot-ordering noise "
                   "cancels); the headline CAGRs are the medians of the same seeds; both sides "
                   "annualised over the shared window (cash_tail_days = days each side is out "
                   "of the market at its end)"}
    ww, cw = own_windows or (None, None)
    if ww is not None and cw is not None:
        pair2 = _paired_seeds(w_use, c_use, ww, cw, sim_kw, reshuffles)
        if pair2 is not None:
            w2, c2 = pair2
            out["own_window"] = {
                "winner_window": {"start": ww[0], "end": ww[1]},
                "control_window": {"start": cw[0], "end": cw[1]},
                **_paired_block(w2, c2, w_use, c_use, ww, cw, reshuffles),
                "note": "each side annualised over ITS exit rule's own window (first common "
                        "entry to last common entry + max_hold benchmark bars, outcome-"
                        "independent); context for the shared-window share above, not the "
                        "pre-registered leg",
            }
    return out


def cash_tail_note(control_pf: dict | None, paired: dict | None = None,
                   label: str = "TEST") -> str | None:
    """The control's cash tail on the shared window when it exceeds
    CASH_TAIL_FLAG_SHARE of that window: the fact, its direction, the
    own-window figures and (when paired) the own-window paired share. None
    when the tail is small or unknown."""
    pf = control_pf or {}
    tail, w = pf.get("cash_tail_days"), pf.get("window") or {}
    if tail is None or not w.get("start") or not w.get("end"):
        return None
    span = _days_between(w["start"], w["end"])
    if span <= 0 or tail <= CASH_TAIL_FLAG_SHARE * span:
        return None
    own = pf.get("own_window") or {}
    ow = own.get("window") or {}
    text = (f"the control is in cash for the last {tail} days of the shared {label} window "
            f"({100 * tail / span:.1f}% of {w['start']}..{w['end']}; its last exit "
            f"{pf.get('last_exit_date')}), a tail the common-set cutoff created, not the rule; "
            "with a positive return that understates its CAGR and, when the tail market is up, "
            "overstates its shortfall vs the index (both reverse when negative): CAGR "
            f"{_f(pf.get('cagr_pct'), 1)}% shared vs {_f(own.get('cagr_pct'), 1)}% on its own "
            f"window ({ow.get('start')}..{ow.get('end')}), vs the blended index "
            f"{_f(pf.get('vs_blended_index_pp'), 1, True)} vs "
            f"{_f(own.get('vs_blended_index_pp'), 1, True)} pp")
    share = ((paired or {}).get("own_window") or {}).get("winner_beats_control_share")
    if share is not None:
        text += (f"; with each side on its own window the winner is ahead in {share:.0%} of "
                 "the seed pairs (paired_test_cagr.own_window)")
    return text


def tail_summary_text(control_pf: dict | None, label: str) -> str:
    """One clause with this run's numbers for the method notes and the caveat:
    the control's last exit and cash tail on a period's shared window (days and
    share of the window) and its CAGR / index shortfall on the shared window
    next to the same figures on its own window. 'n/a' parts when a replay or
    window is missing."""
    pf = control_pf or {}
    w, tail = pf.get("window") or {}, pf.get("cash_tail_days")
    name = label.upper()
    if tail is None or not w.get("start") or not w.get("end"):
        return f"{name}: control cash tail n/a"
    span = _days_between(w["start"], w["end"])
    share = f"{100 * tail / span:.1f}%" if span > 0 else "n/a"
    own = pf.get("own_window") or {}
    ow = own.get("window") or {}
    return (f"{name}: control last exit {pf.get('last_exit_date')}, cash tail {tail} days = "
            f"{share} of the shared window {w['start']}..{w['end']}; CAGR "
            f"{_f(pf.get('cagr_pct'), 1)}% shared vs {_f(own.get('cagr_pct'), 1)}% on its own "
            f"window ({ow.get('start')}..{ow.get('end')}), vs the blended index "
            f"{_f(pf.get('vs_blended_index_pp'), 1, True)} vs "
            f"{_f(own.get('vs_blended_index_pp'), 1, True)} pp")


def _ci_lo(stats: dict | None) -> float | None:
    ci = (stats or {}).get("excess_vs_universe_ci95")
    return None if not ci or ci[0] is None else float(ci[0])


def _ci_hi(stats: dict | None) -> float | None:
    ci = (stats or {}).get("excess_vs_universe_ci95")
    return None if not ci or len(ci) < 2 or ci[1] is None else float(ci[1])


def _cagr(variant: dict | None, period: str) -> float | None:
    return (((variant or {}).get("portfolio") or {}).get(period) or {}).get("cagr_pct")


def pick_winner(variants: dict) -> tuple[str | None, str | None]:
    """(winner, flag): the variant with the highest TRAIN excess_vs_universe_pct
    whose TRAIN range lies above 0; else the highest TRAIN excess, flagged."""
    rows = [(name, (v.get("train") or {}).get(PRIMARY_METRIC), _ci_lo(v.get("train")))
            for name, v in variants.items()]
    scored = [r for r in rows if r[1] is not None]
    significant = [r for r in scored if r[2] is not None and r[2] > 0]
    if significant:
        return max(significant, key=lambda r: r[1])[0], None
    if scored:
        return max(scored, key=lambda r: r[1])[0], "no variant is significantly positive in training"
    return None, "no train statistics"


def decide(variants: dict, control: str = CONTROL, paired: dict | None = None) -> dict:
    """The pre-registered winner rule and verdict (WINNER_RULE). paired = the
    seed-paired TEST CAGR comparison of winner vs control (paired_cagr); the
    CAGR leg then needs the winner ahead in PAIRED_CAGR_SHARE of the seed pairs.
    Without it the leg falls back to the two median CAGRs (stated in the reason)."""
    name, flag = pick_winner(variants)
    if name is None:
        return {"rule": WINNER_RULE, "winner": None, "control": None, "verdict": "not confirmed",
                "verdict_reason": "no train statistics", "flag": "no train statistics",
                "best_test_variant": None, "paired_test_cagr": None}
    w, ctl = variants[name], variants.get(control) or {}
    w_test, c_test = w.get("test") or {}, ctl.get("test") or {}
    w_cagr, c_cagr = _cagr(w, "test"), _cagr(ctl, "test")
    w_ex, c_ex = w_test.get(PRIMARY_METRIC), c_test.get(PRIMARY_METRIC)
    lo, hi = _ci_lo(w_test), _ci_hi(w_test)
    test_sig = lo is not None and lo > 0
    # 'includes 0': the range straddles 0, or there is no range (fewer than 10 quarters)
    includes_zero = lo is None or hi is None or (lo <= 0 <= hi)
    share = (paired or {}).get("winner_beats_control_share")
    if share is not None:
        beats_cagr = share >= PAIRED_CAGR_SHARE
        cagr_text = (f"TEST portfolio CAGR {w_cagr}% vs the control's {c_cagr}% (medians); the "
                     f"winner is ahead in {share:.0%} of {paired['seeds']} seed-paired replays "
                     f"(needs {PAIRED_CAGR_SHARE:.0%})")
    else:
        beats_cagr = w_cagr is not None and c_cagr is not None and w_cagr > c_cagr
        cagr_text = (f"TEST portfolio CAGR {w_cagr}% vs the control's {c_cagr}% (medians; no "
                     "seed-paired replay)")
    beats_excess = w_ex is not None and c_ex is not None and w_ex > c_ex
    if test_sig and beats_cagr and name != control:
        verdict = "confirmed"
        reason = (f"TEST range of {PRIMARY_METRIC} above 0 (lower bound {lo:+.3f}) and the "
                  f"CAGR leg holds: {cagr_text}")
    elif beats_excess and includes_zero and name != control:
        verdict = "improves but unproven"
        reason = (f"TEST {PRIMARY_METRIC} {w_ex:+.3f} beats the control's {c_ex:+.3f} but the "
                  "range includes 0" + (f" ({lo:+.3f} .. {hi:+.3f})" if lo is not None else
                                        " (no range: fewer than 10 entry quarters)"))
    else:
        verdict = "not confirmed"
        if name == control:
            reason = "the train winner is the control itself: no exit change is supported"
        elif test_sig and not beats_cagr:
            reason = f"TEST range above 0 but the CAGR leg fails: {cagr_text}"
        elif beats_excess and hi is not None and hi < 0:
            reason = (f"TEST {PRIMARY_METRIC} {w_ex:+.3f} beats the control's {c_ex:+.3f}, but "
                      f"the whole TEST range lies below 0 ({lo:+.3f} .. {hi:+.3f}): the winner "
                      "still loses to the market, only less than the control")
        elif w_ex is not None and c_ex is not None:
            reason = f"TEST {PRIMARY_METRIC} {w_ex:+.3f} does not beat the control's {c_ex:+.3f}"
        else:
            reason = "TEST statistics missing"
    # the control's cash tail on the shared window (an artefact of the common-set
    # cutoff) is stated next to the CAGR leg when it exceeds CASH_TAIL_FLAG_SHARE
    # of the window; the verdict rule itself is unchanged (pre-registered)
    tail_note = cash_tail_note((ctl.get("portfolio") or {}).get("test"), paired)
    if tail_note and name != control:
        reason += f"; note: {tail_note}"
    best_test = None
    tests = [(n_, (v.get("test") or {}).get(PRIMARY_METRIC)) for n_, v in variants.items()
             if (v.get("test") or {}).get(PRIMARY_METRIC) is not None]
    if tests:
        bt_name = max(tests, key=lambda r: r[1])[0]
        if bt_name != name:
            best_test = {"variant": bt_name, "test": variants[bt_name].get("test"),
                         "portfolio_test": (variants[bt_name].get("portfolio") or {}).get("test"),
                         "label": "in-sample on test - do not act on this"}
    return {
        "rule": WINNER_RULE,
        "winner": {"variant": name, "significant_in_train": flag is None, "flag": flag,
                   "train": w.get("train"), "test": w_test,
                   "portfolio_test": (w.get("portfolio") or {}).get("test"),
                   "portfolio_pooled": (w.get("portfolio") or {}).get("pooled")},
        "control": {"variant": control, "train": ctl.get("train"), "test": c_test,
                    "portfolio_test": (ctl.get("portfolio") or {}).get("test"),
                    "portfolio_pooled": (ctl.get("portfolio") or {}).get("pooled")},
        "verdict": verdict,
        "verdict_reason": reason,
        "flag": flag,
        "best_test_variant": best_test,
        "paired_test_cagr": paired,
    }


# --------------------------------------------------------------------------- driver

def trades_output_path(out_path: Path) -> Path:
    out_path = Path(out_path)
    stem = out_path.name[:-5] if out_path.name.endswith(".json") else out_path.name
    return out_path.with_name(stem + "_trades.json")


def settings_for(cfg: dict, backtest: dict | None) -> Settings:
    """Backtest Settings from the config, with the four cost / yield tables
    taken from the backtest file's method block when present (the costs the
    stored trades were charged)."""
    s = Settings.from_config(cfg)
    method = (backtest or {}).get("method") or {}
    repl = {}
    for k in ("costs_pct_per_side", "stop_slippage_pct", "entry_slippage_pct",
              "index_dividend_yield_pct"):
        v = method.get(k)
        if isinstance(v, dict) and v:
            repl[k] = {**getattr(s, k), **{m: float(x) for m, x in v.items() if x is not None}}
    return dataclasses.replace(s, **repl) if repl else s


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _iter(jobs: list[tuple], pool: ProcessPoolExecutor | None):
    if pool is None:
        return map(_ticker_job, jobs)
    return pool.map(_ticker_job, jobs, chunksize=8)


def _data_quality_diff(ours: dict[str, list], backtest: dict | None,
                       tickers: dict[str, set]) -> dict:
    """Our repair events vs the backtest's method.data_quality, restricted to
    the tickers processed (per market)."""
    bq = ((backtest or {}).get("method") or {}).get("data_quality") or {}
    out = {"events": ours, "matches_backtest": True, "not_in_backtest": {}, "not_reproduced": {}}
    if not bq:
        out["matches_backtest"] = None
        return out
    for m, evs in ours.items():
        mine = {tuple(map(str, e)) for e in evs}
        theirs = {tuple(map(str, e)) for e in (bq.get(m) or [])
                  if e and str(e[0]) in tickers.get(m, set())}
        a, b = sorted(mine - theirs), sorted(theirs - mine)
        if a:
            out["not_in_backtest"][m] = [list(x) for x in a]
        if b:
            out["not_reproduced"][m] = [list(x) for x in b]
        if a or b:
            out["matches_backtest"] = False
    return out


def cache_provenance(conn: sqlite3.Connection, prices_path: Path, markets: list[str]) -> dict:
    """What the results depend on: the cache's last bar date (all tickers and
    each market's benchmark) and the file's modification time. The connection
    is read-only and the cache is never refreshed during a run."""
    out: dict = {"last_bar_date": None, "benchmark_last_bar_date": {}, "file_mtime": None,
                 "access": "opened read-only (sqlite mode=ro); never refreshed during the run"}
    try:
        out["last_bar_date"] = conn.execute("SELECT max(date) FROM bars").fetchone()[0]
        for m in markets:
            if m in BENCHMARKS:
                row = conn.execute("SELECT max(date) FROM bars WHERE ticker = ?",
                                   (BENCHMARKS[m],)).fetchone()
                out["benchmark_last_bar_date"][m] = row[0] if row else None
    except sqlite3.Error as exc:
        out["error"] = f"{exc.__class__.__name__}: {exc}"
    try:
        out["file_mtime"] = datetime.fromtimestamp(Path(prices_path).stat().st_mtime) \
            .isoformat(timespec="seconds")
    except OSError:
        pass
    return out


def _windows(entries: list[dict], records: list[dict]) -> dict[str, tuple[str, str] | None]:
    """Per period: first entry of the period to the last exit of any variant's
    non-censored trade entered in it. Shared by every variant, so the portfolio
    CAGRs and the benchmark of a period are measured over the same days (a
    variant that is out of the market earlier simply holds cash)."""
    out: dict[str, tuple[str, str] | None] = {}
    for p in PERIODS:
        es = [e["entry_date"] for e in entries if p == "pooled" or e["period"] == p]
        xs = [t["exit_date"] for t in records
              if not t["censored"] and (p == "pooled" or t["period"] == p)]
        out[p] = (min(es), max(xs)) if es and xs else None
    return out


def own_windows(entries: list[dict], bench_dates: dict[str, list[str]]
                ) -> dict[str, dict[str, tuple[str, str] | None]]:
    """Per exit rule and period, an OUTCOME-INDEPENDENT window: start = the
    period's first common entry (as the shared window); end = the period's last
    common entry + the rule's max_hold benchmark bars, the latest day a trade
    entered on that date can still be held under the rule. Taken over the
    markets' benchmark calendars (the sparsest calendar reaches furthest, so
    every market's hold fits - the common cutoff itself is defined that way)
    and capped at the last benchmark bar. Computed from the benchmark bars,
    never from realized exits, so a rule that exits early is not charged with
    the cash tail the common-set truncation leaves on the shared window."""
    out: dict[str, dict[str, tuple[str, str] | None]] = {ek: {} for ek in EXITS}
    cals = [sorted(str(d)[:10] for d in ds) for ds in bench_dates.values() if len(ds)]
    for p in PERIODS:
        es = [str(e["entry_date"])[:10] for e in entries if p == "pooled" or e["period"] == p]
        if not es or not cals:
            for ek in EXITS:
                out[ek][p] = None
            continue
        first, last = min(es), max(es)
        for ek, rule in EXITS.items():
            ends = []
            for dates in cals:
                k = max(bisect_right(dates, last) - 1, 0)   # last benchmark bar on/before it
                ends.append(dates[min(k + rule["max_hold"], len(dates) - 1)])
            out[ek][p] = (first, max(max(ends), last))
    return out


def run_exit_experiment(trades_path: Path | None = None, out_path: Path | None = None, *,
                        limit: int | None = None, workers: int | None = None, progress=print,
                        backtest_path: Path | None = None, prices_path: Path | None = None,
                        universe_dir: Path | None = None, cfg: dict | None = None,
                        reshuffles: int = TIE_RESHUFFLES) -> dict:
    """Run the experiment and write out_path (+ its _trades file). Returns the payload.
    limit: first N entries only (smoke) - refuses to write the real output file.
    backtest_path / prices_path / universe_dir / cfg are injectable for tests."""
    started = time.time()
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be >= 1")
    if workers is not None and int(workers) <= 0:
        raise ValueError("workers must be >= 1")
    trades_path = Path(trades_path or DEFAULT_TRADES_PATH)
    data_dir = trades_path.parent
    smoke = limit is not None
    default_out = DEFAULT_SMOKE_OUT_PATH if smoke else DEFAULT_OUT_PATH
    out_path = Path(out_path) if out_path else default_out
    trades_out = trades_output_path(out_path)
    real = {DEFAULT_OUT_PATH.resolve(), trades_output_path(DEFAULT_OUT_PATH).resolve()}
    if smoke and (out_path.resolve() in real or trades_out.resolve() in real):
        raise ValueError(f"--limit runs never write the real output ({DEFAULT_OUT_PATH.name} / "
                         f"{trades_output_path(DEFAULT_OUT_PATH).name}); pass --out")
    backtest_path = Path(backtest_path) if backtest_path else data_dir / "backtest_latest.json"
    for p in (trades_path, backtest_path):
        if p.exists() and p.resolve() in (out_path.resolve(), trades_out.resolve()):
            raise ValueError(f"refusing to overwrite an input file: {p}")
    prices_path = Path(prices_path or DEFAULT_PRICES_PATH)
    universe_dir = Path(universe_dir or DEFAULT_UNIVERSE_DIR)
    cfg = cfg if cfg is not None else load_config()
    backtest = _read_json(backtest_path) if backtest_path.exists() else None
    s = settings_for(cfg, backtest)
    workers = int(workers) if workers else _default_workers()

    trades = _read_json(trades_path)
    if not isinstance(trades, list):
        raise ValueError(f"trade stream unreadable: {trades_path}")
    entries, drop = entries_from_trades(trades)
    if smoke:
        entries = entries[: int(limit)]
    if not entries:
        raise ValueError("no usable entries")
    markets = [m for m in MARKET_ORDER if any(e["market"] == m for e in entries)]
    markets += sorted({e["market"] for e in entries if e["market"] not in markets})
    years = period_years(entries)
    n_entries = {p: sum(1 for e in entries if p == "pooled" or e["period"] == p) for p in PERIODS}
    progress(f"entries: {len(entries)} closed trades from {trades_path.name} "
             f"({drop['open_at_end']} open_at_end and {drop['invalid']} invalid dropped"
             f"{f'; first {limit} only - SMOKE RUN' if smoke else ''}); train "
             f"{n_entries['train']} / test {n_entries['test']} (split at {SPLIT_DATE} by entry "
             "date)")

    conn = _connect_ro(prices_path)
    if conn is None:
        raise FileNotFoundError(f"price cache not found: {prices_path}")
    try:
        overrides, stale = price_repairs(cfg, data_dir, conn)
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in entries:
            groups[(e["market"], e["ticker"])].append(e)
        jobs = [(t, m, str(prices_path), es, overrides, stale.get(t), s)
                for (m, t), es in sorted(groups.items())]
        progress(f"simulating {len(VARIANTS)} variants on {len(jobs)} tickers "
                 f"({workers} worker{'s' if workers > 1 else ''})...")
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        records: list[dict] = []
        counts = {v: {p: _new_counts() for p in PERIODS} for v in VARIANTS}
        counts_c = {v: {p: _new_counts() for p in PERIODS} for v in VARIANTS}
        common_dates: list[str] = []
        last_eligible: dict[str, str | None] = {ek: None for ek in EXITS}
        compared = 0
        mismatches: list[dict] = []
        events: dict[str, list] = {m: [] for m in markets}
        tickers_by_market: dict[str, set] = defaultdict(set)
        scale_info: dict = {"tickers": 0, "rescaled": [], "without_anchor": 0,
                            "atr_bar_shifted": 0, "errors": {}}
        try:
            for k, res in enumerate(_iter(jobs, pool), 1):
                if res.get("error"):
                    scale_info["errors"][res["ticker"]] = res["error"]
                    progress(f"WARN {res['ticker']}: {res['error']}")
                scale_info["tickers"] += 1
                if res["price_scale"] != 1.0:
                    scale_info["rescaled"].append([res["ticker"], round(res["price_scale"], 6),
                                                   res["scale_estimates"]])
                if res["scale_estimates"] == 0 and not res.get("error"):
                    scale_info["without_anchor"] += 1
                scale_info["atr_bar_shifted"] += res["atr_bar_shifted"]
                events[res["market"]].extend(res["price_events"])
                tickers_by_market[res["market"]].add(res["ticker"])
                records.extend(res["records"])
                compared += res["control_compared"]
                mismatches.extend(res["control_mismatches"])
                for v in VARIANTS:
                    for p in SPLIT_PERIODS:
                        _add_counts(counts[v][p], res["counts"][v][p])
                        _add_counts(counts[v]["pooled"], res["counts"][v][p])
                        _add_counts(counts_c[v][p], res["counts_common"][v][p])
                        _add_counts(counts_c[v]["pooled"], res["counts_common"][v][p])
                common_dates.extend(res["common_entry_dates"])
                for ek, d in res["last_eligible_entry"].items():
                    if d is not None and (last_eligible[ek] is None or d > last_eligible[ek]):
                        last_eligible[ek] = d
                if k % 200 == 0:
                    progress(f"  {k} tickers done")
        finally:
            if pool is not None:
                pool.shutdown()
        records.sort(key=lambda t: (t["variant"], t["market"], t["ticker"], t["entry_date"],
                                    t["pattern"]))
        mismatches.sort(key=lambda m: (m["market"], m["ticker"], m["entry_date"], m["pattern"]))
        scale_info["rescaled"].sort()
        # the common entry set: fills with a full LONGEST_HOLD window (per ticker), the
        # set every variant is compared on
        common_entries = [{"entry_date": d, "period": period_of(d)} for d in sorted(common_dates)]
        years_c = period_years(common_entries)
        n_common = {p: sum(1 for e in common_entries if p == "pooled" or e["period"] == p)
                    for p in PERIODS}
        common_records = [t for t in records if t["common"]]
        n_cens = {ek: sum(1 for t in records if t["censored"] and t["exit_rule"] == ek)
                  for ek in EXITS}
        progress(f"  {len(records)} variant trades (censored at the fill: "
                 + ", ".join(f"{ek} {n_cens[ek]}" for ek in EXITS) + "); "
                 f"{len(scale_info['rescaled'])} tickers rescaled onto today's price cache, "
                 f"{scale_info['without_anchor']} without a scale anchor, "
                 f"{counts[CONTROL]['pooled']['bar_missing']} entries without a bar")
        progress(f"  common entry set (full {LONGEST_HOLD}-bar window): {n_common['pooled']} of "
                 f"{len(entries)} entries, train {n_common['train']} / test {n_common['test']}; "
                 f"last eligible entry per hold: "
                 + ", ".join(f"{ek} ({EXITS[ek]['max_hold']} bars) {last_eligible[ek]}"
                             for ek in EXITS))

        # control reproduction
        repro = {"compared": compared, "matched": compared - len(mismatches),
                 "match_rate_pct": _pct(compared - len(mismatches), compared),
                 "mismatch_count": len(mismatches), "mismatches": mismatches[:50],
                 "criteria": "exit_date and outcome equal; exit price on the stored scale within "
                             f"{PRICE_TOL:g} relative of fill x (1 + stored ret_pct/100)"}
        progress(f"control check (S1xE1 vs stored trades): {repro['matched']}/{compared} matched "
                 f"({repro['match_rate_pct']}%)")
        for m_ in mismatches[:10]:
            progress(f"  MISMATCH {m_['ticker']} {m_['entry_date']} {m_['pattern']}: "
                     f"stored {m_['stored']} got {m_['got']}")

        # baselines
        div = dividend_yields(backtest, cfg)
        store = SimpleNamespace(conn=conn)
        base_missing = 0
        bench_dates: dict[str, list[str]] = {}     # each market's benchmark calendar
        for m in markets:
            progress(f"[{m}] baselines (index + equal-weight universe over each trade's days)...")
            bench = BenchContext.from_df(_load_bars(conn, BENCHMARKS[m]))
            bench_dates[m] = [str(d) for d in bench.dates]
            uni_path = universe_dir / f"{m}.csv"
            universe = (pd.read_csv(uni_path)["ticker"].astype(str).tolist() if uni_path.exists()
                        else sorted(tickers_by_market[m]))
            long, prep = sanitize_long(load_long_prices(store, universe), overrides, stale)
            panel = MarketPanel(long, suspect=prep.get("suspect"))
            del long
            base_missing += add_variant_baselines([t for t in records if t["market"] == m], panel,
                                                  bench, float(div.get(m, 0.0)))
            del panel

        # statistics per variant per period: the headline blocks (train/test/pooled,
        # what decide() reads) on the COMMON entry set; each variant's numbers on its
        # own full eligible window under 'full_window'
        by_vp: dict[tuple[str, str], list[dict]] = defaultdict(list)
        by_vp_c: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for t in records:
            by_vp[(t["variant"], t["period"])].append(t)
            by_vp[(t["variant"], "pooled")].append(t)
            if t["common"]:
                by_vp_c[(t["variant"], t["period"])].append(t)
                by_vp_c[(t["variant"], "pooled")].append(t)
        full_note = ("this variant's numbers on every entry whose fill leaves room for ITS OWN "
                     "hold (fi + max_hold <= last bar); the cutoff differs per hold, so these "
                     "blocks are NOT comparable across E1/E2/E3 - the headline train/test/pooled "
                     "blocks use the common entry set")
        variants: dict[str, dict] = {}
        for v in VARIANTS:
            ek = v[3:]
            mh = EXITS[ek]["max_hold"]
            variants[v] = {"stop_rule": STOP_TEXT[v[:2]], "exit_rule": EXIT_TEXT[ek],
                           "max_hold": mh, "last_eligible_entry_date": last_eligible[ek],
                           "censored_at_fill": counts[v]["pooled"]["censored"],
                           "stats_basis": f"common entry set: fills with a full {LONGEST_HOLD}-bar "
                                          f"window (entries up to {last_eligible['E3']})"}
            for p in PERIODS:
                variants[v][p] = period_stats(by_vp_c[(v, p)], counts_c[v][p], years_c[p], mh)
            variants[v]["full_window"] = {"note": full_note}
            for p in PERIODS:
                variants[v]["full_window"][p] = period_stats(by_vp[(v, p)], counts[v][p],
                                                             years[p], mh)

        # benchmarks + portfolio replays over the shared windows of the common set (the
        # headline), and per exit rule over its own outcome-independent window (own_windows)
        windows = _windows(common_entries, common_records)
        own = own_windows(common_entries, bench_dates)
        sim_kw = as_designed_settings(backtest, cfg)
        reg = regime_multipliers(prices_path)
        sim_kw["regime"] = reg
        price_cache: dict = {}      # repaired closes, shared by the benchmark and the marks
        bench_by_window: dict[tuple[str, str], dict] = {}   # one benchmark_block per window

        def bench_for(w: tuple[str, str]) -> dict:
            if w not in bench_by_window:
                b = benchmark_block(w[0], w[1], markets, prices_path, universe_dir, div,
                                    overrides=overrides, stale_map=stale, cache=price_cache)
                b.pop("_series", None)
                bench_by_window[w] = b
            return bench_by_window[w]

        benchmark: dict[str, dict] = {}
        for p in PERIODS:
            if windows[p] is None:
                benchmark[p] = {"error": "no window"}
                continue
            progress(f"benchmark {p}: {windows[p][0]} .. {windows[p][1]} (own windows: "
                     + ", ".join(f"{ek} ..{own[ek][p][1]}" for ek in EXITS if own[ek][p])
                     + ")...")
            benchmark[p] = dict(bench_for(windows[p]))
            benchmark[p]["own_windows"] = {
                ek: (dict(bench_for(own[ek][p])) if own[ek][p] is not None
                     else {"error": "no window"}) for ek in EXITS}
            benchmark[p]["own_windows_note"] = ("the same buy-and-hold benchmark over each "
                                                "exit rule's own window (method.split."
                                                "own_windows)")
        for v in VARIANTS:
            ek = v[3:]
            variants[v]["portfolio"] = {}
            for p in PERIODS:
                if windows[p] is None:
                    variants[v]["portfolio"][p] = {"error": "no window"}
                    continue
                pr = portfolio_replay(by_vp_c[(v, p)], windows[p], sim_kw,
                                      benchmark[p].get("blended_index_cagr_pct"), reshuffles,
                                      conn, overrides, stale, price_cache)
                pr.pop("_cagr_raw", None)
                ob = benchmark[p]["own_windows"].get(ek) or {}
                pr["own_window"] = own_window_replay(by_vp_c[(v, p)], own[ek][p], sim_kw,
                                                     ob.get("blended_index_cagr_pct"), reshuffles)
                variants[v]["portfolio"][p] = pr
            pt = variants[v]["portfolio"]
            ow_t = pt["test"].get("own_window") or {}
            progress(f"  {v}: portfolio CAGR train {pt['train'].get('cagr_pct')}% | test "
                     f"{pt['test'].get('cagr_pct')}% | pooled {pt['pooled'].get('cagr_pct')}% "
                     f"(test on its own window {_f(ow_t.get('cagr_pct'), 1)}%; cash tail "
                     f"{pt['test'].get('cash_tail_days')} d shared / "
                     f"{ow_t.get('cash_tail_days')} d own)")

        # the verdict's CAGR leg, paired by tie-break seed (winner vs control, TEST window);
        # the same pairing with each side on its own window is reported next to it
        winner_name, _ = pick_winner(variants)
        paired = None
        if winner_name is not None and winner_name != CONTROL and windows["test"] is not None:
            progress(f"paired TEST replay {winner_name} vs {CONTROL} ({reshuffles + 1} seeds)...")
            paired = paired_cagr(by_vp_c[(winner_name, "test")], by_vp_c[(CONTROL, "test")],
                                 windows["test"], sim_kw, reshuffles,
                                 own_windows=(own[winner_name[3:]]["test"],
                                              own[CONTROL[3:]]["test"]))
        provenance = cache_provenance(conn, prices_path, markets)
    finally:
        conn.close()

    decision = decide(variants, paired=paired)
    dq = _data_quality_diff(events, backtest, tickers_by_market)
    s2 = variants["S2xE1"]["full_window"]["pooled"]
    s2_tighter = _pct(s2["atr_stop_tighter_than_pattern"]["count"], s2["trades"])
    # the control's cash tail on the shared windows, with this run's numbers
    tail_summary = "; ".join(
        tail_summary_text((variants[CONTROL].get("portfolio") or {}).get(p), p)
        for p in SPLIT_PERIODS)
    method = {
        "design": "pre-registered: entries fixed, exits varied, train/test split by entry date, "
                  "winner rule fixed before results (see atlas2/exit_experiment.py)",
        "entries": f"every closed trade of {trades_path.name} (scanner mode, shortlist applied): "
                   "ticker, market, pattern, signal_date, entry_date, entry_price (fill), "
                   "entry_kind, pattern stop, target; not re-selected. Per variant a ticker's "
                   "entries run chronologically and an entry is skipped while the ticker's "
                   "previous trade under that variant is still open; the test is on the SIGNAL "
                   "bar, as backtest.trade_ticker's: a signal raised on or before the previous "
                   "trade's exit bar is not traded (the scanner does not re-scan a stock until "
                   "the bar after its exit), so a fill on the day after the exit is skipped too",
        "stops": STOP_TEXT,
        "stops_note": "no risk cap applies to the ATR stops (S3 can imply a risk above "
                      f"{RISK_FLAG_PCT:g}% of the fill: risk_over_30_pct); S2 is fill - 3xATR14 as "
                      "pre-registered even where that is TIGHTER than the pattern stop "
                      f"({'n/a' if s2_tighter is None else s2_tighter}% of S2xE1 trades), so S2 "
                      "does not cleanly test a 'wider stop' - see atr_stop_tighter_than_pattern "
                      "per variant",
        "exits_note": "E3 = lowest low since the fill, 20-bar cap: the window is short early in "
                      "the trade (trail_exits_within_20_bars / trail_exits_on_bar_2 per variant)",
        "exits": EXIT_TEXT,
        "fill_bar": "stop (and target, E1) checked on the fill bar: low <= stop counts as a stop "
                    "at the stop (backtest.simulate_exit)",
        "variants": VARIANTS,
        "control": CONTROL,
        "costs": {
            "costs_pct_per_side": s.costs_pct_per_side,
            "stop_slippage_pct": s.stop_slippage_pct,
            "entry_slippage_pct": s.entry_slippage_pct,
            "text": "backtest.trade_returns: costs_pct_per_side on entry and on exit; "
                    "stop_slippage_pct extra on stop exits (a gap through the stop included); "
                    "entry_slippage_pct extra on buy-stop entries; an E3 trailing exit is a "
                    "market order at the next open and pays the market's entry_slippage_pct as "
                    "exit slippage; time exits and target exits (a gap above the target is a "
                    "limit fill) pay no slippage, as in the backtest",
            "source": "backtest_latest.json method" if backtest else "config / defaults",
        },
        "split": {"rule": f"by ENTRY date: train < {SPLIT_DATE} <= test",
                  "label": "quasi out-of-sample: the exit hypothesis (54% stops, the >12%-stop "
                           "slice, 75% winners after 40 bars) was formed on the pooled 2017-2026 "
                           "backtest sample that includes TEST; the exits were not tuned on TEST, "
                           "but the only true out-of-sample check is the live journal from the "
                           "pre-registration date on",
                  "entries": n_entries,
                  "entries_common": n_common,
                  "entries_note": "entries = all closed trades; entries_common = the common "
                                  f"entry set (fill with a full {LONGEST_HOLD}-bar window), on "
                                  "which the headline statistics, verdict, portfolio replays and "
                                  "benchmarks are computed",
                  "years": {p: _rnd(years[p], 2) for p in PERIODS},
                  "years_common": {p: _rnd(years_c[p], 2) for p in PERIODS},
                  "years_note": "span of the period's entry dates (trades_per_year denominator; "
                                "years for full_window, years_common for the headline blocks)",
                  "windows": {p: list(w) if w else None for p, w in windows.items()},
                  "windows_note": "first common entry of the period to the last exit of any "
                                  "variant's common trade entered in it; shared by every variant "
                                  "(portfolio CAGR and the benchmark use it). The common set "
                                  "stops entries at the 250-bar cutoff while the longest hold "
                                  "stays invested, so a shorter hold sits in cash for a tail the "
                                  "truncation created, not the rule: with a positive return the "
                                  "shared window UNDERSTATES its CAGR (the same final equity "
                                  "annualised over more years) and, when the tail market is up, "
                                  "OVERSTATES its shortfall vs the index; both reverse when "
                                  "negative, and the handicap grows with the hold difference. "
                                  f"This run: {tail_summary}. Every portfolio block reports "
                                  "last_exit_date and cash_tail_days; the outcome-independent "
                                  "figures per exit rule are under own_windows / own_window",
                  "own_windows": {ek: {p: (list(w) if w else None) for p, w in own[ek].items()}
                                  for ek in EXITS},
                  "own_windows_note": "per exit rule: first common entry of the period to the "
                                      "last common entry + max_hold benchmark bars (the latest "
                                      "day a trade entered then can still be held under the "
                                      "rule; the sparsest market calendar, capped at the last "
                                      "bar), computed from the benchmark bars, never from "
                                      "realized exits. Replay and benchmark over it: "
                                      "variants.<name>.portfolio.<period>.own_window and "
                                      "benchmark.<period>.own_windows; the paired comparison: "
                                      "paired_test_cagr.own_window. Context next to the "
                                      "pre-registered shared-window headline, not a "
                                      "replacement"},
        "atr_source": "indicators.add_indicators atr14 (Wilder EWM, alpha 1/14, adjust=False) on "
                      "the repaired series, read at the signal bar; the recursion is causal, so "
                      "the value equals the one computed on bars up to and including the signal "
                      f"bar ({scale_info['atr_bar_shifted']} signal bars missing from the "
                      "repaired series: last bar before them used)",
        "data_repairs": {
            "rule": "backtest.sanitize_bars with the config's price_artifacts (+ "
                    "data/price_artifacts.json) and the cache's stale-head map "
                    "(portfolio_sim.price_repairs), as the backtest; events compared with the "
                    "backtest's method.data_quality for the tickers processed",
            "overrides": [dict(x) for x in overrides],
            **dq,
        },
        "price_scale": {
            "note": "the cache is re-downloaded auto-adjusted by every scan; a dividend after "
                    "the backtest run rescales a ticker's whole history, so its stored levels "
                    "are mapped onto today's bars with a per-ticker factor (median over "
                    "next-open fills, gap fills, time-exit closes and gap exits); returns and R "
                    "are scale-free; tickers without an anchor are assumed unscaled",
            "tickers": scale_info["tickers"],
            "rescaled": scale_info["rescaled"],
            "without_anchor": scale_info["without_anchor"],
            "errors": scale_info["errors"],
        },
        "baselines": "per trade over its own dates, no costs: index (S&P 500 / DAX / NIFTY 50) "
                     "open on the first bar on/after entry to close on the last bar on/before "
                     "the baseline exit date, price indices raised by the backtest's dividend "
                     f"yield ({div}); universe = equal-weight basket of the market's current "
                     "members over the same days (backtest.MarketPanel; survivorship bias). "
                     "Baseline exit date: the exit date for stop, target and time exits, gap "
                     "fills included (as the backtest, so the control's baselines match the "
                     "stored ones); an E3 trailing exit fills at the OPEN, so its baseline runs "
                     "to the close of the previous bar (baseline_exit_date = the last bar fully "
                     f"held), not one more day of market drift. {base_missing} trades lack a "
                     "baseline",
        "censoring": {
            "rule": "the variant's full hold window must fit in the stock's data, decided at "
                    "the FILL (fill bar + max_hold <= last bar), as backtest.trade_ticker does, "
                    "so censoring does not depend on the outcome; a censored entry is excluded "
                    "from the statistics and the portfolio replay (the last close is never "
                    "used) and still blocks later entries of its ticker under that variant",
            "censored_at_fill": {v: counts[v]["pooled"]["censored"] for v in VARIANTS},
            "last_eligible_entry_date": {ek: {"max_hold": EXITS[ek]["max_hold"],
                                              "date": last_eligible[ek]} for ek in EXITS},
            "common_set": f"every number decide() reads (headline train/test/pooled blocks, "
                          f"verdict, portfolio replays, benchmarks) is computed on entries whose "
                          f"fill has a full {LONGEST_HOLD}-bar window (record flag 'common'; "
                          f"last eligible entry {last_eligible['E3']}); otherwise the variants "
                          f"would be compared on different calendar windows and a long hold "
                          f"credited for skipping the last months. Each variant's numbers on "
                          f"its own eligible window are under variants.<name>.full_window",
            "entries_common": n_common,
        },
        "control_reproduction": repro,
        "primary_metric": PRIMARY_METRIC,
        "ci": "95% range: mean +/- t(df) x SE. E1 (40-bar holds): SE = max(SE clustered by "
              "entry month, by entry quarter), df = entry quarters - 1 (backtest.cluster_stats; "
              "none below 10 quarters). E2 (120 bars, longer than a quarter) also considers the "
              "SE clustered by entry half-year, E3 (250 bars) the half-year and the year; the "
              "largest SE binds with the df of its own clustering (se_basis; e.g. 12 train "
              "half-years -> df 11, about 6 test half-years -> df 5), so the ranges of long "
              "holds are wider and use a t with fewer df. A coarser clustering counts only "
              f"with {MIN_CLUSTERS}+ clusters (backtest.t_crit is a Cornish-Fisher expansion, "
              "within about 1% from df 3). The month/quarter, half-year and year SEs are all "
              "reported",
        "r_net_note": "r_net is per unit of the VARIANT stop and is NOT comparable across stop "
                      "variants",
        "portfolio": {"profile": "SYSTEM AS DESIGNED (1% risk + regime filter)",
                      "settings": {k: v for k, v in sim_kw.items() if k != "regime"},
                      "regime": ("multipliers from the index state at the previous close "
                                 "(portfolio_sim.regime_multipliers)" if reg else "none"),
                      "sizing": "shares = risk / (entry level - VARIANT stop), capped by "
                                "max_position_pct and the free cash (portfolio_sim)",
                      "headline": f"median CAGR of {reshuffles + 1} slot orderings "
                                  "(portfolio_sim.run_profile)",
                      "verdict_cagr_leg": "winner vs control paired by tie-break seed "
                                          "(simulate_portfolio with the same seed, window and "
                                          "settings on both sides); 'confirmed' needs the winner "
                                          f"ahead in >= {PAIRED_CAGR_SHARE:.0%} of the seed pairs "
                                          "- see paired_test_cagr",
                      "records": "the common entry set (see censoring.common_set)",
                      "drawdown": "marked to market daily (portfolio_sim.mark_to_market) where "
                                  "the helper allows it, else realized - see drawdown_basis",
                      "windows": "shared per period (method.split.windows): every variant and "
                                 "the benchmark on the same days, so a variant that is out of "
                                 "the market earlier holds cash to the window's end. That tail "
                                 "is an artefact of the common-set cutoff: with a positive "
                                 "return the shared window understates the shorter hold's CAGR "
                                 "and, when the tail market is up, overstates its shortfall vs "
                                 "the index (both reverse when negative). This run: "
                                 f"{tail_summary}",
                      "own_window": OWN_WINDOW_NOTE + "; under portfolio.<period>.own_window "
                                    "next to the shared figures and under "
                                    "paired_test_cagr.own_window",
                      "cash_tail": "last_exit_date / cash_tail_days per block: the variant's "
                                   "last common exit and the window's end minus it; decide() "
                                   "states the control's TEST tail next to the CAGR leg when it "
                                   f"exceeds {CASH_TAIL_FLAG_SHARE:.0%} of the window (the "
                                   "verdict rule is unchanged)"},
        "limit": limit,
        "workers": workers,
        "source": {"trades": str(trades_path), "backtest": str(backtest_path) if backtest else None,
                   "backtest_generated": (backtest or {}).get("generated"),
                   "prices": str(prices_path),
                   "prices_cache": provenance,
                   "note": "the bars may end after the backtest's own run (a later scan "
                           "refreshes the cache); one-bar differences in the control check and "
                           "rescaled tickers (method.price_scale) trace back to that cache state"},
    }
    caveats = [
        "The entries come from a scanner whose patterns, filters and thresholds were tuned on "
        "this same history (in-sample); only the exits are varied here.",
        "Universe = today's index members (survivorship bias); the equal-weight baseline shares "
        "it, and it matters more for long holds (E2 120 bars, E3 up to 250 bars).",
        "Baselines carry no costs (they stand for a cheap index ETF held over the same days).",
        "ATR stops are sized from the signal bar's volatility, not the fill bar's.",
        "E3 acts on the next open after a close below the trailing low; the initial stop is "
        "checked first on every bar.",
        "9 variants x 2 periods is still multiple testing; only the pre-registered winner rule "
        "and its TEST verdict count, and even that winner was picked among 9 candidates.",
        "The train winner rule was fixed before the results were seen.",
        "The 95% ranges of E1 cluster by entry month and quarter, as the backtest does for "
        "40-bar holds; E2 and E3 trades last longer than a quarter, so their ranges also take "
        "the SE clustered by entry half-year (E3: and by year) with the df of the clustering "
        "that binds - fewer clusters, a wider t: about 11 df in train and 5 in test instead of "
        "the quarter count (see excess_vs_universe_se_basis / _df per block).",
        "Quasi out-of-sample only: the exit hypothesis (54% stops, the >12%-stop slice, 75% "
        "winners after 40 bars) was formed on the pooled 2017-2026 backtest sample, which "
        "includes the TEST years; the exits were not tuned on TEST, but the split cannot make "
        "them fresh - the only true out-of-sample check is the live journal from the "
        "pre-registration date on. The test period (2023-2025) is also shorter than training "
        "and mostly a bull market, and the entries were tuned on the whole history.",
        "Censoring is decided at the fill (the full hold window must fit in the data, as "
        "backtest.trade_ticker does), so it never depends on the outcome; the cutoff differs "
        f"per hold ({', '.join(f'{ek} {last_eligible[ek]}' for ek in EXITS)}), so every number "
        f"the verdict reads is computed on the common entry set (fills with a full "
        f"{LONGEST_HOLD}-bar window) and the last months of the data are not compared at all; "
        "a censored entry still blocks its ticker's later entries.",
        "Skipped entries differ per variant (a longer hold blocks more later entries), so the "
        "variants trade different entry sets; entries, trades and trades_per_year show it.",
        "Portfolio windows are shared per period (first common entry to the last exit of any "
        "variant's common trade); trades entered before the split can close after it, and a "
        "variant that is out of the market earlier holds cash for the rest of the window. That "
        "tail is an artefact of the common-set cutoff (entries stop 250 bars before the data "
        "end while the longest hold stays invested): with a positive return the shared window "
        "understates the shorter hold's CAGR and, when the tail market is up, overstates its "
        "shortfall vs the index (both reverse when negative), and the handicap grows with the "
        f"hold difference - towards 'confirmed' for a long-hold winner. This run: {tail_summary}. "
        "The outcome-independent own-window figures (own_window per portfolio block, "
        "paired_test_cagr.own_window) stand next to the pre-registered shared-window headline; "
        "the verdict rule is unchanged.",
        "Trailing and time exits are modelled at the bar's open / close; the live monitor's "
        "delayed alerts are not.",
        "The verdict's CAGR leg is paired by tie-break seed (winner ahead in at least "
        f"{PAIRED_CAGR_SHARE:.0%} of the slot orderings); the headline CAGRs are medians whose "
        "5-95% ranges from slot luck alone span a few points (cagr_range).",
        "No risk cap applies to the ATR stops, and S2 (3xATR) is tighter than the pattern stop "
        "for a sizeable share of entries (atr_stop_tighter_than_pattern), so S2 is not a clean "
        "'wider stop' test; E3's trailing window is short early in the trade "
        "(trail_exits_within_20_bars).",
    ]
    if scale_info["rescaled"]:
        caveats.append(f"{len(scale_info['rescaled'])} tickers were rescaled onto today's price "
                       "cache (dividend adjustments after the backtest run); returns are "
                       "scale-free - see method.price_scale and control_reproduction.")
    if repro["mismatch_count"]:
        caveats.append(f"S1xE1 did not reproduce {repro['mismatch_count']} of "
                       f"{repro['compared']} stored trades - see method.control_reproduction.")
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "smoke": smoke,
        "method": method,
        "variants": variants,
        "winner": decision.get("winner"),
        "control": decision.get("control"),
        "verdict": decision.get("verdict"),
        "verdict_reason": decision.get("verdict_reason"),
        "winner_rule": decision.get("rule"),
        "best_test_variant": decision.get("best_test_variant"),
        "paired_test_cagr": decision.get("paired_test_cagr"),
        "benchmark": benchmark,
        "caveats": caveats,
    }
    write_json_atomic(out_path, payload, indent=1)
    write_json_atomic(trades_out, records)
    progress(f"\n{len(records)} variant trades -> {trades_out}\nresults -> {out_path} "
             f"({time.time() - started:.0f}s)\n")
    for line in summary_lines(payload):
        progress(line)
    return payload


# --------------------------------------------------------------------------- report

def _f(x, nd: int = 2, signed: bool = False, unit: str = "") -> str:
    if x is None:
        return "n/a"
    return (f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}") + unit


def summary_lines(payload: dict) -> list[str]:
    """Plain-English summary table of the run."""
    method = payload.get("method") or {}
    cens = method.get("censoring") if isinstance(method.get("censoring"), dict) else {}
    last = {k: v.get("date") for k, v in (cens.get("last_eligible_entry_date") or {}).items()}
    lines = ["EXIT EXPERIMENT - 'vs market' = per-trade % after fees minus the equal-weight "
             "market over the same days (primary metric), 95% range clustered by entry "
             "month/quarter (E2: also half-year, E3: also half-year and year); r_net is per "
             "VARIANT stop (not comparable across S1/S2/S3); portfolio = SYSTEM AS DESIGNED "
             "sizing on the variant stop, CAGR vs the blended index of the same window",
             f"All rows below are on the COMMON entry set: fills with a full {LONGEST_HOLD}-bar "
             f"window (entries up to {last.get('E3')}); censoring is decided at the fill "
             f"(last eligible entry per hold: {last}); each variant's numbers on its own "
             "eligible window are under variants.<name>.full_window ('cens' = censored at the "
             "fill within the common set, 0 by construction)"]
    head = (f"{'variant':<7} {'period':<6} {'trades':>6} {'skip':>5} {'cens':>5} {'win%':>5} "
            f"{'stop%':>5} {'hold':>5} {'net%':>7} {'vs market':>10} {'95% range':>18} "
            f"{'vs index':>9} {'r_net':>7} {'CAGR%':>6} {'maxDD%':>6} {'vs idx pp':>9}")
    lines.append(head)
    for v, blk in payload["variants"].items():
        for p in PERIODS:
            st = blk.get(p) or {}
            pf = (blk.get("portfolio") or {}).get(p) or {}
            ci = st.get("excess_vs_universe_ci95")
            ci_s = f"[{_f(ci[0], 2, True)} .. {_f(ci[1], 2, True)}]" if ci else "no range"
            lines.append(
                f"{v:<7} {p:<6} {st.get('trades', 0):>6} {st.get('skipped_overlap', 0):>5} "
                f"{st.get('censored', 0):>5} {_f(st.get('win_rate'), 1):>5} "
                f"{_f(st.get('stop_share'), 1):>5} {_f(st.get('avg_hold_bars'), 1):>5} "
                f"{_f(st.get('net_avg_ret_pct'), 2, True):>7} "
                f"{_f(st.get('excess_vs_universe_pct'), 2, True):>10} {ci_s:>18} "
                f"{_f(st.get('excess_vs_index_pct'), 2, True):>9} "
                f"{_f(st.get('net_avg_r'), 3, True):>7} "
                f"{_f(pf.get('cagr_pct'), 1):>6} {_f(pf.get('max_drawdown_pct'), 1):>6} "
                f"{_f(pf.get('vs_blended_index_pp'), 1, True):>9}")
    w = payload.get("winner") or {}
    c = payload.get("control") or {}
    if w:
        wt, ct = w.get("test") or {}, c.get("test") or {}
        lines.append("")
        lines.append(f"TRAIN winner: {w.get('variant')}"
                     + (f" - {w['flag']}" if w.get("flag") else " (train range above 0)"))
        for label, name, st, pf in (("winner ", w.get("variant"), wt, w.get("portfolio_test")),
                                    ("control", c.get("variant"), ct, c.get("portfolio_test"))):
            lines.append(f"  TEST {label} {name}: vs market {_f(st.get(PRIMARY_METRIC), 2, True)}% "
                         f"{st.get('excess_vs_universe_ci95')} | {st.get('trades')} trades | win "
                         f"{_f(st.get('win_rate'), 1)}% | CAGR {_f((pf or {}).get('cagr_pct'), 1)}%"
                         f" | maxDD {_f((pf or {}).get('max_drawdown_pct'), 1)}%")
        pc = payload.get("paired_test_cagr")
        if pc:
            lines.append(f"  paired TEST CAGR ({pc.get('seeds')} seeds, same tie-break seed both "
                         f"sides): winner ahead in {_f(100 * pc['winner_beats_control_share'], 0)}% "
                         f"of the pairs (needs {_f(100 * pc['required_share'], 0)}%); winner "
                         f"p5/median/p95 {pc['winner_cagr_range'].get('p5')} / "
                         f"{pc['winner_cagr_range'].get('median')} / "
                         f"{pc['winner_cagr_range'].get('p95')}%, control "
                         f"{pc['control_cagr_range'].get('p5')} / "
                         f"{pc['control_cagr_range'].get('median')} / "
                         f"{pc['control_cagr_range'].get('p95')}%")
        lines.append(f"VERDICT: {payload.get('verdict')} - {payload.get('verdict_reason')}")
        best = payload.get("best_test_variant")
        if best:
            lines.append(f"  best on TEST: {best['variant']} "
                         f"({_f((best.get('test') or {}).get(PRIMARY_METRIC), 2, True)}% vs "
                         f"market) - {best['label']}")
    rp = (payload.get("method") or {}).get("control_reproduction") or {}
    lines.append(f"control reproduction: {rp.get('matched')}/{rp.get('compared')} stored trades "
                 f"matched ({rp.get('match_rate_pct')}%)")
    return lines


# --------------------------------------------------------------------------- CLI

USAGE = ("usage: python -m atlas2 exit-experiment [--limit N] [--out PATH] [--trades PATH] "
         "[--workers N]")


def parse_opts(opts: list[str]) -> dict:
    """Strict option parsing; raises ValueError on anything unexpected."""
    out: dict = {"limit": None, "out": None, "trades": None, "workers": None}
    k = 0
    while k < len(opts):
        name, eq, val = opts[k].partition("=")
        if name not in ("--limit", "--out", "--trades", "--workers"):
            raise ValueError(f"unknown option: {opts[k]}")
        if not eq:
            if k + 1 >= len(opts):
                raise ValueError(f"{name} needs a value")
            k += 1
            val = opts[k]
        key = name[2:]
        if key in ("limit", "workers"):
            try:
                n = int(val)
            except ValueError:
                raise ValueError(f"{name} needs a whole number, got {val!r}") from None
            if n <= 0:
                raise ValueError(f"{name} must be >= 1, got {n}")
            out[key] = n
        else:
            out[key] = val
        k += 1
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
        payload = run_exit_experiment(trades_path=o["trades"], out_path=o["out"], limit=o["limit"],
                                      workers=o["workers"])
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if payload.get("smoke"):
        rp = payload["method"]["control_reproduction"]
        if rp["mismatch_count"]:
            print(f"SMOKE ASSERTION FAILED: S1xE1 reproduced only {rp['matched']}/{rp['compared']} "
                  "stored trades - investigate before the full run", file=sys.stderr)
            sys.exit(2)
