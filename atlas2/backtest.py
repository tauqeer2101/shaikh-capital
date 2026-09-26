"""Walk-forward backtest of the live scanner's pattern strategy (format v2).

Every trading day of the cached ~10-year history (`step` days apart, 1 by
default), for every stock, ask: what would the live scanner have shown today,
using only the data known today? The same liquidity floor, technical score,
market-regime filter, pattern choice, risk:reward floor AND top-N shortlist cut
as scan.py are applied; the chosen setup is then traded the way a person would:
a buy-stop above the entry level for 'forming' setups (one day by default,
re-detected with fresh levels the next day), a buy at the next open for
'triggered' ones, then stop / target / time exit.

The simulation runs in three phases per market:
  1. per stock (in worker processes): sanitise the cached prices, then on
     every signal day apply the filters and store every bullish pattern hit;
  2. per signal day (main process): pick each stock's best hit exactly like
     scan.py and rank the day's candidates for the shortlist;
  3. per stock (main process): walk the days in order with the stock's own
     state (open position / pending order, levels already traded) and trade.
Phase 2+3 are cheap, so the "live configuration" replay (today's disabled
patterns removed before the best hit is picked, as the live scanner does), the
walk-forward replay and the grade replays (setups below grade A never ordered,
so they never block a later A-grade setup) need no re-detection.

Every trade is charged realistic costs and compared with three baselines over
the same dates: holding the market index (plus an estimated dividend yield for
price indices), holding an equal-weight basket of the whole universe, and
buying random liquid, similarly volatile stocks on the same day with the same
stop/target distances and exits. Each pattern gets a plain verdict
(proven_edge / unproven / no_edge / insufficient_data) that scan.py uses to
switch patterns off.

Outputs (data/): backtest_latest.json (summary for the dashboard and the
scanner), backtest_trades.json (every trade, for the portfolio simulation),
backtest_trades_live.json (the replay with today's disabled patterns),
backtest_trades_walkforward.json, backtest_trades_grade_a.json and
backtest_trades_a_in_caution.json (the portfolio's walk-forward and grade
profiles), backtest_trades_raw.json with --raw. A --limit smoke run writes *_smoke.json
files instead and never touches the real results.

Run: python -m atlas2 backtest [us de in] [--raw] [--limit N] [--workers N] [--no-portfolio]
                               [--legacy-grid] [--entry-mode buy_stop|close_confirm]
                               [--keep-others | --replace]
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import zlib
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .data import BENCHMARKS, HISTORY_DAYS, PriceStore
from .indicators import add_indicators
from .patterns import BEARISH_DETECTORS, BULLISH_DETECTORS, PatternHit
from .scan import (BACKTEST_DEFAULTS, MIN_DOLLAR_VOL, PATTERN_TUNING_DEFAULTS,
                   _deep_merge, load_config, technical_score)

ROOT = Path(__file__).resolve().parent.parent
VERSION = 2
MARKET_ORDER = ("us", "de", "in")
TECH_SCORE_MIN = 40      # same floor as scan.scan_market
REGIME_MIN_BARS = 210    # scan.market_regime reports 'unknown' below this
# The longest look-back of any input: high_52w (252 bars), which also covers
# sma200 and REGIME_MIN_BARS. Signal days before the benchmark's 252nd bar are
# warm-up only.
WARMUP_BARS = 252
MIN_HISTORY_BARS = 120   # the live scanner skips stocks with fewer bars (scan.py)
# cup_and_handle's 140-bar base needs len(win) = window+1 >= 145; IHS needs
# window >= 133; the cache shows 0/11,280 mismatches against full-history
# detection at 144 and 226 at 143.
MIN_WINDOW = 144
TRIGGER_BAND = 1.03      # mirrors patterns._status: a close > 1.03 * entry is not shown live
ENTRY_MODES = ("buy_stop", "close_confirm")
REMOVAL_REASONS = ("no_bar_on_date", "liquidity", "tech_score", "regime_downtrend",
                   "risk_reward", "not_shortlisted", "already_traded",
                   "failed_before_trigger", "no_breakout", "ran_away", "gap_too_far",
                   "gap_below_stop", "regime_downtrend_at_fill", "not_grade_a")
GRADE_MODES = ("A", "A_in_caution")
VERDICT_LABELS = {
    "proven_edge": "proven edge",
    "unproven": "edge not proven",
    "no_edge": "no edge",
    "insufficient_data": "too few trades to judge",
}
INDEX_NAMES = {"us": "S&P 500", "de": "DAX", "in": "NIFTY 50"}
MARKET_CCY = {"us": "USD", "de": "EUR", "in": "INR"}

# grade proxy (scan.py's no-fundamentals composite, without the backtest nudge)
PS_CAP = 100
RR_BONUS_FROM = 2.0
RR_BONUS_PER = 5
W_TECH = 0.55
W_PAT = 0.45
CONFLICT_PENALTY = 10
CONFLICT_MIN_CONF = 0.55
GRADE_A = 72
GRADE_B = 58

# price-data sanitising (backtest only; the live scanner is unchanged)
SPLIT_RATIOS = (1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 2 / 3, 3 / 4, 2 / 5, 3 / 5,
                2.0, 3.0, 4.0, 5.0, 10.0)   # 0.8 / 0.9 / 5:6 left out: circuit limits and cent ticks
RATIO_TOL = 1e-6          # an unadjusted corporate action sets open = prev_close * ratio exactly
PENNY_GUARD = 1.0         # exact ratios of sub-1.0 prices are tick artefacts
VOLUME_SURGE = 1.5        # the share count changed: volume >= 1.5x the prior 20-day median
JOIN_RATIO = 10.0         # a 10x jump in a day = two different securities joined
FLAT_RUN_MIN = 5          # zero-volume flat bars before a jump = placeholder history
FLAT_RUN_JUMP = 0.2
STALE_JUMP = 0.25         # jump in the head that an older download adjusted differently
SPIKE_MOVE = 0.6          # one-bar spike ...
SPIKE_REVERT = 0.10       # ... that fully reverts the next day
SUSPECT_RATIO = 5.0       # unresolved moves this large are masked out of the baselines


# --------------------------------------------------------------------------- settings

def _per_market(v, default: dict) -> dict:
    """A float applies to every market; a dict is merged over the defaults."""
    if isinstance(v, dict):
        return {m: float(x) for m, x in {**default, **v}.items()}
    return {m: float(v) for m in MARKET_ORDER}


@dataclass(frozen=True)
class Settings:
    """Backtest knobs (config.json -> "backtest", defaults in scan.BACKTEST_DEFAULTS)."""
    step: int = 1
    window: int = 170
    breakout_wait: int = 1
    max_hold: int = 40
    max_gap_above_entry_pct: float = 3.0
    costs_pct_per_side: dict = field(default_factory=lambda: dict(
        BACKTEST_DEFAULTS["costs_pct_per_side"]))
    stop_slippage_pct: dict = field(default_factory=lambda: dict(
        BACKTEST_DEFAULTS["stop_slippage_pct"]))
    entry_slippage_pct: dict = field(default_factory=lambda: dict(
        BACKTEST_DEFAULTS["entry_slippage_pct"]))
    random_entries_per_trade: int = 10
    min_risk_reward: float = 1.8
    regime_filter: bool = True
    seen_on_fill_only: bool = True
    shortlist_size: int = 45
    entry_mode: str = "buy_stop"
    index_dividend_yield_pct: dict = field(default_factory=lambda: dict(
        BACKTEST_DEFAULTS["index_dividend_yield_pct"]))
    price_artifacts: tuple = ()

    def __post_init__(self):
        if int(self.window) < MIN_WINDOW:
            raise ValueError(f"backtest window must be at least {MIN_WINDOW} bars "
                             f"(cup_and_handle / inverse head-and-shoulders need it), got {self.window}")
        if int(self.step) < 1 or int(self.breakout_wait) < 1 or int(self.max_hold) < 1:
            raise ValueError("backtest step, breakout_wait and max_hold must be >= 1")
        if self.entry_mode not in ENTRY_MODES:
            raise ValueError(f"entry_mode must be one of {ENTRY_MODES}, got {self.entry_mode!r}")
        if int(self.shortlist_size) < 1:
            raise ValueError("shortlist_size must be >= 1")

    @classmethod
    def from_config(cls, cfg: dict) -> "Settings":
        bt = _deep_merge(BACKTEST_DEFAULTS, cfg.get("backtest") or {})
        costs = _per_market(bt.get("costs_pct_per_side") or {}, BACKTEST_DEFAULTS["costs_pct_per_side"])
        return cls(
            step=int(bt["step"]),
            window=int(bt["window"]),
            breakout_wait=int(bt["breakout_wait"]),
            max_hold=int(bt["max_hold"]),
            max_gap_above_entry_pct=float(bt["max_gap_above_entry_pct"]),
            costs_pct_per_side=costs,
            stop_slippage_pct=_per_market(bt["stop_slippage_pct"],
                                          BACKTEST_DEFAULTS["stop_slippage_pct"]),
            entry_slippage_pct=_per_market(bt["entry_slippage_pct"],
                                           BACKTEST_DEFAULTS["entry_slippage_pct"]),
            random_entries_per_trade=int(bt["random_entries_per_trade"]),
            min_risk_reward=float(cfg.get("min_risk_reward", 1.8)),
            regime_filter=bool(cfg.get("regime_filter", True)),
            seen_on_fill_only=bool(bt.get("seen_on_fill_only", True)),
            shortlist_size=int(cfg.get("shortlist_size", 45)),
            entry_mode=str(bt.get("entry_mode", "buy_stop")),
            index_dividend_yield_pct=_per_market(bt["index_dividend_yield_pct"],
                                                 BACKTEST_DEFAULTS["index_dividend_yield_pct"]),
            price_artifacts=tuple(dict(x) for x in (bt.get("price_artifacts") or [])
                                  if isinstance(x, dict)),
        )

    def cost_pct(self, market: str) -> float:
        """Percent of trade value charged on entry AND on exit."""
        return float(self.costs_pct_per_side.get(market, 0.15))

    def stop_slip(self, market: str) -> float:
        """Extra slippage (%) on stop-loss exits (market orders)."""
        return float(self.stop_slippage_pct.get(market, 0.05))

    def entry_slip(self, market: str) -> float:
        """Extra slippage (%) on buy-stop entries (a triggered stop is a market order)."""
        return float(self.entry_slippage_pct.get(market, 0.05))

    def div_yield_pct(self, market: str) -> float:
        """Estimated dividend yield (%/yr) added to a price index baseline."""
        return float(self.index_dividend_yield_pct.get(market, 0.0))


def first_signal_bar(s: Settings) -> int:
    """First bar index that can be a signal bar: the 120th bar (live's minimum),
    aligned to the `step` grid through `window` (so the post-warm-up grid does
    not move when step > 1)."""
    base = MIN_HISTORY_BARS - 1
    return base + ((s.window - base) % s.step)


# --------------------------------------------------------------------------- small helpers

def _rnd(x, nd: int):
    """Round, mapping None / NaN / inf to None (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, nd)


def _sg(x: float, nd: int) -> str:
    """Signed number that never prints a negative zero."""
    v = round(float(x), nd) + 0.0
    return f"{v:+.{nd}f}"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _pct(part: int, whole: int) -> float | None:
    return round(100.0 * part / whole, 1) if whole else None


def sig4(x: float) -> float:
    """Round to 4 significant figures (the 'same level' key for a pattern)."""
    return float(f"{float(x):.4g}")


def _date_strings(index: pd.Index) -> np.ndarray:
    if isinstance(index, pd.DatetimeIndex):
        return np.asarray(index.strftime("%Y-%m-%d"), dtype=str)
    try:
        return np.asarray(pd.DatetimeIndex(index).strftime("%Y-%m-%d"), dtype=str)
    except Exception:
        return np.asarray([str(x)[:10] for x in index], dtype=str)


def _clean_json(obj):
    """Recursively replace NaN / inf floats with None so the file is valid JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_clean_json(v) for v in obj)
    if isinstance(obj, np.generic):
        return _clean_json(obj.item())
    return obj


def write_json_atomic(path: Path, obj, indent: int | None = None) -> None:
    """Write JSON to a temp file next to `path`, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(_clean_json(obj), indent=indent, allow_nan=False))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- price sanitising

def sanitize_bars(df: pd.DataFrame, ticker: str, overrides=(), stale_before: str | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """Repair known defects of the free price data, for the backtest only.

    Pure function: returns (clean copy, report) where report = {"counts":
    {action: n}, "events": [[ticker, date, action(, detail)]], "suspect": [dates]}.
    All tests use the raw ratios x = open/prev_close and y = close/prev_close:
      a) overrides ({ticker, date, action: adjust|drop_bar|truncate, ratio});
         'adjust' multiplies earlier OHLC by ratio and earlier volume by 1/ratio;
      b) exact split / bonus ratio (open = prev_close * ratio to 1e-6, prev
         close >= 1.0, volume surge) -> back-adjust like a);
      c) truncate at the latest bar that is a listing join (a 10x move), the
         first move of more than 20% after >= 5 flat zero-volume bars, or a
         >= 25% move in the stale head (on/before the first bar that the latest
         download re-fetched, `stale_before`; universe stocks only);
      d) drop a one-bar spike (>= 60%) that reverts within 10% the next day.
    Bigger moves (5x) that no rule explains are reported as 'unresolved' and
    masked out of the baselines.
    """
    rep: dict = {"counts": defaultdict(int), "events": [], "suspect": []}
    n = len(df) if df is not None else 0
    if n < 2:
        return df, rep
    dates = _date_strings(df.index)
    o = df["open"].to_numpy(float).copy()
    h = df["high"].to_numpy(float).copy()
    l = df["low"].to_numpy(float).copy()
    c = df["close"].to_numpy(float).copy()
    v = df["volume"].to_numpy(float).copy() if "volume" in df.columns else None
    pc = np.r_[np.nan, c[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = o / pc
        y = c / pc
    handled = np.zeros(n, bool)
    drop = np.zeros(n, bool)
    adj = np.ones(n)
    trunc = 0

    def event(k: int, action: str, detail=None) -> None:
        rep["counts"][action] += 1
        rep["events"].append([ticker, str(dates[k]), action] + ([detail] if detail is not None else []))

    # a) overrides
    for ov in overrides or ():
        if str(ov.get("ticker")) != ticker:
            continue
        d = str(ov.get("date", ""))[:10]
        k = int(np.searchsorted(dates, d))
        if k == 0 or k >= n or dates[k] != d:
            continue
        act = str(ov.get("action", "adjust"))
        if act == "adjust":
            r = float(ov.get("ratio") or 0)
            if not r > 0:
                continue
            adj[:k] *= r
            handled[k] = True
            event(k, "override_adjust", round(r, 6))
        elif act == "drop_bar":
            drop[k] = handled[k] = True
            event(k, "override_drop_bar")
        elif act == "truncate":
            trunc = max(trunc, k)
            handled[k] = True
            event(k, "override_truncate")

    # b) exact split / bonus ratio with a volume surge
    if v is not None:
        cand = np.nonzero((pc >= PENNY_GUARD) & ~handled & np.isfinite(x) & (np.abs(x - 1) > 0.05))[0]
        for k in cand:
            q = min(SPLIT_RATIOS, key=lambda r: abs(x[k] / r - 1))
            if abs(x[k] / q - 1) > RATIO_TOL:
                continue
            prior = v[max(0, k - 20):k]
            prior = prior[prior > 0]
            if len(prior) == 0 or not v[k] >= VOLUME_SURGE * float(np.median(prior)):
                continue
            adj[:k] *= q
            handled[k] = True
            event(k, "adjust_split_ratio", round(q, 6))

    # c) truncation: join, flat zero-volume run, stale head
    free = ~handled & np.isfinite(x) & np.isfinite(y)
    cands: list[tuple[int, str]] = []
    join = free & ((x <= 1 / JOIN_RATIO) | (x >= JOIN_RATIO) | (y <= 1 / JOIN_RATIO) | (y >= JOIN_RATIO))
    cands += [(int(k), "truncate_join") for k in np.nonzero(join)[0]]
    if v is not None:
        flat = (v == 0) & (o == c) & (h == l)
        cs = np.r_[0, np.cumsum(flat)]
        big = np.maximum(np.abs(x - 1), np.abs(y - 1)) > FLAT_RUN_JUMP
        for k in range(FLAT_RUN_MIN, n):
            if free[k] and not flat[k] and big[k] and cs[k] - cs[k - FLAT_RUN_MIN] == FLAT_RUN_MIN:
                cands.append((k, "truncate_flat_run"))
    if stale_before:
        ksb = int(np.searchsorted(dates, str(stale_before)[:10]))   # first bar re-fetched
        stale = free & (np.maximum(np.abs(x - 1), np.abs(y - 1)) >= STALE_JUMP)
        cands += [(int(k), "truncate_stale_head") for k in np.nonzero(stale[: ksb + 1])[0]]
    if cands:
        k = max(k for k, _ in cands)
        if k > trunc:
            trunc = k
            event(k, sorted(w for kk, w in cands if kk == k)[-1])

    o, h, l, c = o * adj, h * adj, l * adj, c * adj
    if v is not None:
        v = v / adj
    keep = ~drop
    keep[:trunc] = False
    idx = np.nonzero(keep)[0]
    o, h, l, c = o[idx], h[idx], l[idx], c[idx]
    if v is not None:
        v = v[idx]
    d2 = dates[idx]

    # d) one-bar spike that reverts
    m = len(idx)
    spike = np.zeros(m, bool)
    for k in range(1, m - 1):
        if (abs(c[k] / c[k - 1] - 1) >= SPIKE_MOVE
                and abs(c[k + 1] / c[k - 1] - 1) <= SPIKE_REVERT):
            spike[k] = True
            rep["counts"]["drop_spike"] += 1
            rep["events"].append([ticker, str(d2[k]), "drop_spike"])
    keep2 = ~spike
    out = df.iloc[idx[keep2]].copy()
    out["open"], out["high"], out["low"], out["close"] = o[keep2], h[keep2], l[keep2], c[keep2]
    if v is not None:
        out["volume"] = v[keep2]

    # e) moves no rule explains: reported, masked out of the baselines
    cc, oo = c[keep2], o[keep2]
    d3 = d2[keep2]
    if len(cc) > 1:
        pcc = cc[:-1]                       # bar k's previous close, for k = 1 .. m-1
        o1, c1 = oo[1:], cc[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            worst = np.max(np.nan_to_num(np.vstack([o1 / pcc, pcc / o1, c1 / pcc, pcc / c1]),
                                         nan=0.0, posinf=np.inf), axis=0)
        for k in np.nonzero(worst >= SUSPECT_RATIO)[0] + 1:
            rep["suspect"].append(str(d3[k]))
            rep["counts"]["unresolved"] += 1
            rep["events"].append([ticker, str(d3[k]), "unresolved"])
    rep["counts"] = dict(rep["counts"])
    return out, rep


def merge_price_reports(total: dict, part: dict) -> None:
    for k, n in (part.get("counts") or {}).items():
        total["counts"][k] = total["counts"].get(k, 0) + n
    total["events"].extend(part.get("events") or [])


def new_price_report() -> dict:
    return {"counts": {}, "events": []}


def stale_before_map(store: PriceStore, tickers: list[str] | None = None) -> dict[str, str]:
    """Per ticker: the first date the latest download re-fetched (last_fetch - HISTORY_DAYS).
    Bars before it come from older downloads and may carry older adjustments."""
    out: dict[str, str] = {}
    try:
        rows = store.conn.execute("SELECT ticker, last_fetch FROM meta").fetchall()
    except Exception:
        return out
    want = set(tickers) if tickers is not None else None
    for t, lf in rows:
        if want is not None and t not in want:
            continue
        try:
            out[str(t)] = (date.fromisoformat(str(lf)[:10]) - timedelta(days=HISTORY_DAYS)).isoformat()
        except Exception:
            continue
    return out


def load_price_artifacts(data_dir: Path) -> list[dict]:
    """Optional extra overrides from data/price_artifacts.json (a list of dicts)."""
    p = Path(data_dir) / "price_artifacts.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text())
    except Exception:
        return []
    return [dict(x) for x in d if isinstance(x, dict)] if isinstance(d, list) else []


# --------------------------------------------------------------------------- trade mechanics

def trade_returns(fill: float, stop: float, exit_price: float, outcome: str,
                  cost_pct: float, slip_pct: float, entry_slip_pct: float = 0.0) -> dict:
    """Gross and net (after costs) result of one trade.

    cost_pct is charged on entry and on exit; slip_pct is extra slippage on
    stop-loss exits and entry_slip_pct on buy-stop entries (both are market
    orders once triggered). R is measured against the gross risk from the fill.
    """
    c = cost_pct / 100.0
    s = slip_pct / 100.0 if outcome == "stop" else 0.0
    e = entry_slip_pct / 100.0
    entry_eff = fill * (1 + c + e)
    exit_eff = exit_price * (1 - c - s)
    risk = fill - stop
    return {
        "r": (exit_price - fill) / risk,
        "r_net": (exit_eff - entry_eff) / risk,
        "ret_pct": (exit_price / fill - 1) * 100,
        "ret_net_pct": (exit_eff / entry_eff - 1) * 100,
        "cost_pct_round_trip": (2 * c + e + s) * 100,
    }


def enter_forming(o, h, l, i: int, entry: float, stop: float, wait: int,
                  max_gap_pct: float) -> tuple[int, float | None, str | None]:
    """Buy-stop at `entry`, live for `wait` bars after signal bar i.

    Returns (fill_i, fill, None) or (resolved_i, None, reason), where resolved_i
    is the bar on which the order stopped being live. A bar that opens at or
    above the entry fills at its open. A bar that opens at or below the stop
    ends the setup. A bar that opens between them and reaches the entry fills
    at the entry, even if it also touches the stop, because a resting buy-stop
    fills either way; simulate_exit then books the same-day stop (conservative).
    """
    n = len(o)
    for j in range(i + 1, min(i + 1 + wait, n)):
        if o[j] >= entry:
            fill = float(o[j])                  # gapped over the entry: filled at the open
        elif o[j] <= stop:
            return j, None, "failed_before_trigger"   # opened at/through the stop: the stop came first
        elif h[j] >= entry:
            fill = float(entry)                 # opened between stop and entry: the buy-stop fills
        elif l[j] <= stop:
            return j, None, "failed_before_trigger"
        else:
            continue
        if fill > entry * (1 + max_gap_pct / 100):
            return j, None, "gap_too_far"
        return j, fill, None
    return min(i + wait, n - 1), None, "no_breakout"


def enter_triggered(o, i: int, entry: float, stop: float,
                    max_gap_pct: float) -> tuple[int, float | None, str | None]:
    """The breakout already happened by the close of bar i: buy at the next open.
    Returns (fill_i, fill, None) or (resolved_i, None, reason)."""
    j = i + 1
    if j >= len(o):
        return len(o) - 1, None, "no_breakout"
    fill = float(o[j])
    if fill > entry * (1 + max_gap_pct / 100):
        return j, None, "gap_too_far"
    if fill <= stop:
        return j, None, "gap_below_stop"
    return j, fill, None


def enter_close_confirm(o, l, c, i: int, entry: float, stop: float, wait: int,
                        max_gap_pct: float) -> tuple[int, float | None, str | None]:
    """'Check back after the close': buy at the next open once a close is at or
    above the entry (but not more than 3% above it, where the live scanner would
    no longer show it). Returns (fill_i, fill, None) or (resolved_i, None, reason)."""
    n = len(o)
    for j in range(i + 1, min(i + 1 + wait, n)):
        if l[j] <= stop:
            return j, None, "failed_before_trigger"
        if c[j] >= entry:
            if c[j] > entry * TRIGGER_BAND:
                return j, None, "ran_away"
            return enter_triggered(o, j, entry, stop, max_gap_pct)
    return min(i + wait, n - 1), None, "no_breakout"


def simulate_exit(o, h, l, c, fill_i: int, fill: float, stop: float, target: float,
                  max_hold: int) -> tuple[int, float, str]:
    """Walk forward from the fill bar. Returns (exit_i, exit_price, outcome).

    Fill bar: a touched stop counts as a stop (intrabar order unknown, so the
    conservative reading), else a touched target counts. Later bars: a gap
    through the stop exits at the open, then the stop, then a gap above the
    target exits at the open, then the target. Stop and target stay live
    through bar fill_i + max_hold (checked in the same order); if neither is
    hit, a time exit at that bar's close. If the data ends first, 'open_at_end'.
    """
    n = len(c)
    for j in range(fill_i, min(fill_i + max_hold + 1, n)):
        if j == fill_i:
            if l[j] <= stop:
                return j, float(stop), "stop"
            if h[j] >= target:
                # max(): a fill that gapped above a very close target sells at the fill
                return j, float(max(target, fill)), "target"
            continue
        if o[j] <= stop:
            return j, float(o[j]), "stop"
        if l[j] <= stop:
            return j, float(stop), "stop"
        if o[j] >= target:
            return j, float(o[j]), "target"
        if h[j] >= target:
            return j, float(target), "target"
    if fill_i + max_hold <= n - 1:
        j = fill_i + max_hold
        return j, float(c[j]), "time"
    return n - 1, float(c[n - 1]), "open_at_end"


def grade_proxy(tscore: float, conf: float, rr: float | None,
                conflicting_bearish: bool) -> tuple[float, float, str]:
    """scan.py's grade without fundamentals and without the backtest nudge.
    Returns (pattern_score_raw, composite_proxy, grade) from unrounded inputs."""
    psr = min(PS_CAP, conf * 100 + max(0.0, (rr or 0) - RR_BONUS_FROM) * RR_BONUS_PER)
    composite = W_TECH * tscore + W_PAT * psr - (CONFLICT_PENALTY if conflicting_bearish else 0)
    grade = "A" if composite >= GRADE_A else "B" if composite >= GRADE_B else "C"
    return psr, composite, grade


# --------------------------------------------------------------------------- benchmark context

@dataclass
class BenchContext:
    """One market's benchmark index: raw bars plus the regime on every bar."""
    df: pd.DataFrame
    dates: np.ndarray       # 'YYYY-MM-DD' strings
    opens: np.ndarray
    closes: np.ndarray
    statuses: list[str]     # healthy / caution / downtrend / unknown, per bar
    warmup_end: str = "9999-12-31"   # first signal day (the 252nd index bar)

    @classmethod
    def from_df(cls, bench: pd.DataFrame) -> "BenchContext":
        bench = bench.sort_index()
        dates = _date_strings(bench.index)
        return cls(df=bench, dates=dates,
                   opens=bench["open"].to_numpy(float),
                   closes=bench["close"].to_numpy(float),
                   statuses=regime_statuses(bench),
                   warmup_end=str(dates[WARMUP_BARS - 1]) if len(dates) >= WARMUP_BARS
                   else "9999-12-31")

    def positions(self, dates: np.ndarray) -> np.ndarray:
        """Index of the last benchmark bar on or before each date (-1 if none)."""
        return np.searchsorted(self.dates, dates, side="right") - 1

    def position(self, d: str) -> int:
        return int(np.searchsorted(self.dates, d, side="right")) - 1


def regime_statuses(bench: pd.DataFrame) -> list[str]:
    """scan.market_regime's status for every bar, as if run on bench[:p+1].

    The indicators are causal (rolling / ewm), so computing them once on the
    full history gives the same values as recomputing on each slice.
    """
    if len(bench) == 0:
        return []
    b = add_indicators(bench)
    close = b["close"].to_numpy(float)
    s50 = b["sma50"].to_numpy(float)
    s200 = b["sma200"].to_numpy(float)
    off = b["pct_off_high"].to_numpy(float)
    out = []
    for p in range(len(b)):
        if p + 1 < REGIME_MIN_BARS:
            out.append("unknown")
            continue
        above200 = close[p] > s200[p]
        golden = s50[p] > s200[p]
        if above200 and golden and off[p] > -8:
            out.append("healthy")
        elif above200:
            out.append("caution")
        else:
            out.append("downtrend")
    return out


def signal_calendar(bench: BenchContext, s: Settings) -> list[str]:
    """Signal days shared by every stock of a market: benchmark days on the
    `step` grid (the last 5 benchmark days are never signal days)."""
    return [str(bench.dates[k]) for k in range(first_signal_bar(s), len(bench.dates) - 5, s.step)]


# --------------------------------------------------------------------------- phase 1: detection

Hit = namedtuple("Hit", "pattern confidence risk_reward entry stop target status")
Setup = namedtuple("Setup", "i tscore regime hits conflict")


def new_counters() -> dict:
    return {"signals_evaluated": 0, "removed": {r: 0 for r in REMOVAL_REASONS},
            "no_pattern": 0, "detector_exceptions": 0, "censored_at_end": 0,
            "orders_not_filled_today": 0, "warmup": 0}


def _merge_counters(total: dict, part: dict) -> None:
    for k, v in part.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                total[k][kk] = total[k].get(kk, 0) + vv
        else:
            total[k] = total.get(k, 0) + v


def _hit_key(h) -> tuple:
    return (h.pattern, sig4(h.entry))


def _expire(seen: dict, oldest_bar: int) -> dict:
    """Forget pattern levels first seen before `oldest_bar` (they may trade again)."""
    return {k: b for k, b in seen.items() if b >= oldest_bar}


def _detect_all_bullish(win: pd.DataFrame, counters: dict) -> list[PatternHit]:
    hits: list[PatternHit] = []
    for det in BULLISH_DETECTORS:
        try:
            hits.extend(det(win))
        except Exception:
            counters["detector_exceptions"] += 1
    return hits


def _bearish_conflict(win: pd.DataFrame, counters: dict) -> bool:
    """scan.py's warning rule: any bearish hit with confidence >= 0.55 (any status)."""
    conflict = False
    for det in BEARISH_DETECTORS:
        try:
            if any(h.confidence >= CONFLICT_MIN_CONF for h in det(win)):
                conflict = True
        except Exception:
            counters["detector_exceptions"] += 1
    return conflict


def _as_hit(h) -> Hit:
    return Hit(h.pattern, float(h.confidence), h.risk_reward, float(h.entry), float(h.stop),
               float(h.target), h.status)


def _rs3m(c: np.ndarray, i: int, bclose: np.ndarray, p: int) -> float | None:
    """indicators.relative_strength_3m(df.iloc[:i+1], bench.iloc[:p+1]), from arrays."""
    if i + 1 < 64 or p + 1 < 64:
        return None
    try:
        s = float(c[i] / c[i - 63] - 1) * 100
        b = float(bclose[p] / bclose[p - 63] - 1) * 100
        return s - b
    except Exception:
        return None


def scan_ticker(df: pd.DataFrame, market: str, bench: BenchContext, s: Settings,
                cal: list[str], counters: dict) -> list:
    """Phase 1 for one stock (df has indicators): one entry per calendar day.

    None: not a signal day for this stock (warm-up, before its 120th bar or in
    its last 5 bars); a reason string: removed by a filter that does not depend
    on the pattern; a Setup: the filters passed, with every forming/triggered
    bullish hit on the chart (in detector order). Uses only data up to the day.
    """
    n = len(df)
    dates = _date_strings(df.index)
    if n < MIN_HISTORY_BARS + 6:
        return [None] * len(cal)
    pos = {d: k for k, d in enumerate(dates)}
    first_date, last_date = dates[MIN_HISTORY_BARS - 1], dates[n - 6]
    c = df["close"].to_numpy(float)
    dvol = df["dollar_vol20"].to_numpy(float)
    cols = {k: df[k].to_numpy(float) for k in ("sma50", "sma200", "rsi14", "pct_off_high")}
    min_dv = MIN_DOLLAR_VOL[market]
    recs: list = []
    for d in cal:
        if d < first_date or d > last_date:
            recs.append(None)
            continue
        i = pos.get(d)
        if d < bench.warmup_end:
            if i is not None:
                counters["warmup"] += 1
            recs.append(None)
            continue
        if i is None:
            recs.append("no_bar_on_date")
            continue
        # a) liquidity, as of the signal day
        if not dvol[i] >= min_dv:  # also rejects NaN
            recs.append("liquidity")
            continue
        # b) technical score with relative strength measured on data up to day d
        p = bench.position(d)
        rs3m = _rs3m(c, i, bench.closes, p) if p >= 0 else None
        row = {"close": c[i], "sma50": cols["sma50"][i], "sma200": cols["sma200"][i],
               "rsi14": cols["rsi14"][i], "pct_off_high": cols["pct_off_high"][i], "date": d}
        tscore, _ = technical_score(row, rs3m)
        if tscore < TECH_SCORE_MIN:
            recs.append("tech_score")
            continue
        # c) market regime on the signal day
        regime = "off" if not s.regime_filter else (bench.statuses[p] if p >= 0 else "unknown")
        if regime == "downtrend":
            recs.append("regime_downtrend")
            continue
        # d) every forming / triggered bullish pattern (the pick happens in phase 2)
        win = df.iloc[max(0, i - s.window): i + 1]
        hits = [_as_hit(x) for x in _detect_all_bullish(win, counters)
                if x.status in ("forming", "triggered")]
        conflict = _bearish_conflict(win, counters) if hits else False
        recs.append(Setup(i, float(tscore), regime, tuple(hits), conflict))
    return recs


# --------------------------------------------------------------------------- phase 2: pick + rank

@dataclass
class TickerData:
    ticker: str
    uorder: int              # row in universe/<market>.csv (live tie order)
    dates: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    recs: list


def select_best(setup: Setup, disabled, min_rr: float, info: dict | None = None,
                best_patterns: set | None = None):
    """scan.py's pick: drop disabled patterns, best = max(confidence, rr); the
    risk:reward floor applies to that best hit only (never falls back)."""
    hits = [x for x in setup.hits if x.pattern not in disabled]
    if info is not None:
        for x in hits:
            pi = info[x.pattern]
            pi["detected"] += 1
            if x.risk_reward is not None:
                pi["max_rr"] = max(pi["max_rr"], x.risk_reward) if pi["max_rr"] is not None \
                    else x.risk_reward
    if not hits:
        return "no_pattern"
    best = max(hits, key=lambda x: (x.confidence, x.risk_reward or 0))
    if best_patterns is not None:
        best_patterns.add(best.pattern)
    if info is not None:
        info[best.pattern]["best"] += 1
    if best.risk_reward is None or best.risk_reward < min_rr:
        if info is not None:
            info[best.pattern]["rr_rejected"] += 1
        return "risk_reward"
    return best


def _new_pattern_info() -> dict:
    return defaultdict(lambda: {"detected": 0, "best": 0, "rr_rejected": 0, "max_rr": None})


def pick_and_rank(tds: list[TickerData], s: Settings, disabled=frozenset(),
                  info: dict | None = None, best_patterns: set | None = None) -> list[list]:
    """Phase 2: per stock and day the decision (None / reason / [best, setup,
    rank, n_candidates]); candidates are ranked per day like scan.py's
    shortlist: key = confidence * 50 + technical score, ties in universe order.
    disabled: a set of patterns, or {signal year 'YYYY': set} (walk-forward: the
    list in force on the signal day; a year missing from the map disables nothing)."""
    by_year = disabled if isinstance(disabled, dict) else None
    decisions: list[list] = []
    by_day: dict[int, list] = defaultdict(list)
    for tk, td in enumerate(tds):
        dec: list = []
        for ci, rec in enumerate(td.recs):
            if rec is None or isinstance(rec, str):
                dec.append(rec)
                continue
            dis = (by_year.get(str(td.dates[rec.i])[:4], frozenset()) if by_year is not None
                   else disabled)
            b = select_best(rec, dis, s.min_risk_reward, info, best_patterns)
            if isinstance(b, str):
                dec.append(b)
                continue
            by_day[ci].append((-(b.confidence * 50 + rec.tscore), td.uorder, tk))
            dec.append([b, rec, None, None])
        decisions.append(dec)
    for ci, lst in by_day.items():
        lst.sort()
        for rank, (_, _, tk) in enumerate(lst, 1):
            decisions[tk][ci][2] = rank
            decisions[tk][ci][3] = len(lst)
    return decisions


# --------------------------------------------------------------------------- phase 3: trade

def _record(market: str, ticker: str, dates: np.ndarray, hit, regime: str,
            i: int, fill_i: int, fill: float, exit_i: int, exit_price: float,
            outcome: str, s: Settings, extra: dict | None = None) -> dict:
    kind = "buy_stop" if hit.status == "forming" and s.entry_mode == "buy_stop" else "next_open"
    res = trade_returns(fill, hit.stop, exit_price, outcome, s.cost_pct(market),
                        s.stop_slip(market), s.entry_slip(market) if kind == "buy_stop" else 0.0)
    plan_risk = hit.entry - hit.stop
    t = {
        "market": market,
        "ticker": ticker,
        "pattern": hit.pattern,
        "conf": hit.confidence,
        "status_at_signal": hit.status,
        "entry_kind": kind,
        "regime": regime,
        "signal_date": str(dates[i]),
        "entry_date": str(dates[fill_i]),
        "exit_date": str(dates[exit_i]),
        "entry_price": round(fill, 4),
        "entry_level": hit.entry,
        "stop": hit.stop,
        "target": hit.target,
        "rr": hit.risk_reward,
        "risk_pct": round((fill - hit.stop) / fill * 100, 3),
        "risk_pct_plan": round(plan_risk / hit.entry * 100, 3) if hit.entry > 0 else None,
        "outcome": outcome,
        "hold_bars": int(exit_i - fill_i),
        "r": round(res["r"], 4),
        "r_net": round(res["r_net"], 4),
        # result per PLANNED risk (entry level to stop), as the dashboard sizes positions
        "r_plan_net": round(res["r_net"] * (fill - hit.stop) / plan_risk, 4) if plan_risk > 0 else None,
        "ret_pct": round(res["ret_pct"], 3),
        "ret_net_pct": round(res["ret_net_pct"], 3),
        "cost_pct_round_trip": round(res["cost_pct_round_trip"], 3),
        "index_ret_pct": None,
        "index_ret_price_pct": None,
        "universe_ret_pct": None,
        "excess_vs_index_pct": None,
        "excess_vs_universe_pct": None,
        "excess_vs_index_r": None,
        "excess_vs_universe_r": None,
        "rand_r_net": None,
        "rand_ret_net_pct": None,
    }
    if extra:
        t.update(extra)
    return t


def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return tuple(df[k].to_numpy(float) for k in ("open", "high", "low", "close"))


def _enter(hit, o, h, l, c, i: int, s: Settings):
    """Entry for one setup under the configured entry mode (module-level names
    are looked up at call time so tests can patch them)."""
    if hit.status == "triggered":
        return enter_triggered(o, i, hit.entry, hit.stop, s.max_gap_above_entry_pct)
    if s.entry_mode == "close_confirm":
        return enter_close_confirm(o, l, c, i, hit.entry, hit.stop, s.breakout_wait,
                                   s.max_gap_above_entry_pct)
    return enter_forming(o, h, l, i, hit.entry, hit.stop, s.breakout_wait,
                         s.max_gap_above_entry_pct)


def trade_ticker(market: str, td: TickerData, decisions: list, cal: list[str],
                 bench: BenchContext, s: Settings, counters: dict) -> list[dict]:
    """Phase 3 for one stock: walk the signal days with the stock's own state.

    A stock is not re-scanned while its order is pending (until the order fills,
    fails or expires) or its position is open; whether a later day is examined
    therefore depends only on data known by that day. A filled trade blocks its
    pattern level for `window` bars; if the day's best pick is such a level the
    stock is skipped that day (the runner-up is never substituted).
    """
    o, h, l, c = td.o, td.h, td.l, td.c
    dates = td.dates
    n = len(c)
    removed = counters["removed"]
    trades: list[dict] = []
    seen: dict[tuple, int] = {}
    busy_date = ""          # skip days before this date (pending order / open position)

    def busy_from(b: int) -> str:
        return str(dates[b]) if b < n else "9999-12-31"

    for ci, dec in enumerate(decisions):
        if dec is None:
            continue
        if cal[ci] < busy_date:
            continue
        counters["signals_evaluated"] += 1
        if isinstance(dec, str):
            if dec == "no_pattern":
                counters["no_pattern"] += 1
            else:
                removed[dec] += 1
            continue
        best, rec, rank, ncand = dec
        i = rec.i
        seen = _expire(seen, i - s.window)
        if rank is not None and rank > s.shortlist_size:
            removed["not_shortlisted"] += 1
            continue
        key = _hit_key(best)
        if key in seen:              # live shows this level again; never fall back to the runner-up
            removed["already_traded"] += 1
            continue
        if not s.seen_on_fill_only:  # legacy: an attempted order blocks the level
            seen[key] = i
        res_i, fill, why = _enter(best, o, h, l, c, i, s)
        if why is None and fill - best.stop <= 0:
            why = "gap_below_stop"
        if why is not None:
            if why == "no_breakout" and s.breakout_wait == 1:
                counters["orders_not_filled_today"] += 1   # one-day order: re-detected tomorrow
            else:
                removed[why] += 1
            # the order is dead by the close of bar res_i: that evening's scan may
            # show the stock again (with a one-day order: re-detected the next day)
            busy_date = busy_from(res_i)
            continue
        fill_i = res_i
        # the full max_hold window must fit in this stock's data; decided at the
        # fill, so it does not depend on the outcome (end-of-data censoring)
        if fill_i + s.max_hold > n - 1:
            counters["censored_at_end"] += 1
            break            # any later signal would fill even later
        if s.seen_on_fill_only:
            seen[key] = i
        if s.regime_filter:
            k = int(np.searchsorted(bench.dates, dates[fill_i], side="left")) - 1
            if k >= 0 and bench.statuses[k] == "downtrend":
                removed["regime_downtrend_at_fill"] += 1   # the index turned down before the fill
                busy_date = busy_from(fill_i)
                continue
        exit_i, exit_price, outcome = simulate_exit(o, h, l, c, fill_i, fill, best.stop,
                                                    best.target, s.max_hold)
        psr, comp, grade = grade_proxy(rec.tscore, best.confidence, best.risk_reward, rec.conflict)
        trades.append(_record(market, td.ticker, dates, best, rec.regime, i, fill_i, fill,
                              exit_i, exit_price, outcome, s, {
                                  "shortlist_rank": rank,
                                  "candidates_on_date": ncand,
                                  "tscore": round(rec.tscore, 1),
                                  "pattern_score_raw": round(psr, 1),
                                  "conflicting_bearish": bool(rec.conflict),
                                  "composite_proxy": round(comp, 1),
                                  "grade_proxy": grade,
                              }))
        busy_date = busy_from(exit_i + 1)
    return trades


def _trade_sort_key(t: dict) -> tuple:
    return (t["market"], t["ticker"], t["entry_date"], t["pattern"])


def run_selection(tds: list[TickerData], cal: list[str], bench: BenchContext, s: Settings,
                  market: str, disabled=frozenset(), counters: dict | None = None,
                  info: dict | None = None, best_patterns: set | None = None) -> list[dict]:
    """Phases 2 + 3 for a market: pick, rank (shortlist) and trade every stock."""
    counters = counters if counters is not None else new_counters()
    decisions = pick_and_rank(tds, s, disabled, info, best_patterns)
    trades: list[dict] = []
    for td, dec in zip(tds, decisions):
        trades.extend(trade_ticker(market, td, dec, cal, bench, s, counters))
    trades.sort(key=_trade_sort_key)
    return trades


def _ticker_data(df: pd.DataFrame, ticker: str, uorder: int, recs: list) -> TickerData:
    o, h, l, c = _arrays(df)
    return TickerData(ticker, uorder, _date_strings(df.index), o, h, l, c, recs)


def simulate_scanner(df: pd.DataFrame, ticker: str, market: str, bench: BenchContext,
                     s: Settings, counters: dict, *, disabled=frozenset(),
                     best_patterns: set | None = None, cal: list[str] | None = None) -> list[dict]:
    """Trades the live scanner would have shown for ONE stock (df has
    indicators). The shortlist is trivially passed (rank 1 of 1); `disabled`
    patterns are dropped before the best hit is picked, like scan.py."""
    cal = cal if cal is not None else signal_calendar(bench, s)
    recs = scan_ticker(df, market, bench, s, cal, counters)
    td = _ticker_data(df, ticker, 0, recs)
    return run_selection([td], cal, bench, s, market, frozenset(disabled), counters,
                         best_patterns=best_patterns)


def simulate_raw(df: pd.DataFrame, ticker: str, market: str, bench: BenchContext,
                 s: Settings) -> list[dict]:
    """The OLD selection (first 'forming' hit in detector order, no scanner filters)
    with the new entry / exit rules and costs - for attribution only. The first
    eligible hit places the order and is resolved whatever the outcome (no
    second hit on the same bar), so later bars never depend on future prices."""
    n = len(df)
    first = first_signal_bar(s)
    if n < first + 6:
        return []
    o, h, l, c = _arrays(df)
    dates = _date_strings(df.index)
    bpos = bench.positions(dates)
    trades: list[dict] = []
    seen: dict[tuple, int] = {}
    busy_until = 0
    for i in range(first, n - 5, s.step):
        if i < busy_until:
            continue
        if dates[i] < bench.warmup_end:
            continue
        seen = _expire(seen, i - s.window)
        win = df.iloc[max(0, i - s.window): i + 1]
        placed = False
        for det in BULLISH_DETECTORS:
            try:
                hits = det(win)
            except Exception:
                continue
            for hit in hits:
                if hit.status != "forming":
                    continue
                key = _hit_key(hit)
                if key in seen:
                    continue
                if not s.seen_on_fill_only:
                    seen[key] = i
                res_i, fill, why = _enter(hit, o, h, l, c, i, s)
                placed = True
                if why is None and fill - hit.stop > 0:
                    fill_i = res_i
                    if fill_i + s.max_hold > n - 1:
                        busy_until = n
                        break
                    if s.seen_on_fill_only:
                        seen[key] = i
                    exit_i, exit_price, outcome = simulate_exit(o, h, l, c, fill_i, fill,
                                                                hit.stop, hit.target, s.max_hold)
                    p = int(bpos[i])
                    regime = bench.statuses[p] if p >= 0 else "unknown"
                    trades.append(_record(market, ticker, dates, _as_hit(hit), regime, i, fill_i,
                                          fill, exit_i, exit_price, outcome, s))
                    busy_until = exit_i + 1
                else:
                    busy_until = res_i      # order dead by the close of bar res_i
                break
            if placed:
                break
    return trades


# Per-process caches for worker processes (each opens its own PriceStore).
_WORKER_STORES: dict[str, PriceStore] = {}
_WORKER_BENCH: dict[tuple[str, str], BenchContext] = {}


def _worker_store(db_path: str) -> PriceStore:
    if db_path not in _WORKER_STORES:
        _WORKER_STORES[db_path] = PriceStore(Path(db_path))
    return _WORKER_STORES[db_path]


def _worker_bench(db_path: str, market: str) -> BenchContext:
    key = (db_path, market)
    if key not in _WORKER_BENCH:
        _WORKER_BENCH[key] = BenchContext.from_df(_worker_store(db_path).load(BENCHMARKS[market]))
    return _WORKER_BENCH[key]


def _reset_worker_caches() -> None:
    for st in _WORKER_STORES.values():
        try:
            st.close()
        except Exception:
            pass
    _WORKER_STORES.clear()
    _WORKER_BENCH.clear()


def simulate_job(job: tuple) -> dict:
    """Top-level (picklable) worker: phase 1 (+ raw mode) for one stock.

    job = (ticker, market, db_path, settings, raw, calendar, stale_before, uorder)."""
    ticker, market, db_path, s, raw, cal, stale_before, uorder = job
    out = {"ticker": ticker, "market": market, "uorder": uorder, "simulated": False,
           "td": None, "raw_trades": [], "counters": new_counters(), "error": None,
           "price_report": new_price_report()}
    try:
        df = _worker_store(db_path).load(ticker)
        df, rep = sanitize_bars(df, ticker, s.price_artifacts, stale_before)
        out["price_report"] = {"counts": rep["counts"], "events": rep["events"]}
        if df is None or len(df) < MIN_HISTORY_BARS + 6:
            return out
        df = add_indicators(df)
        bench = _worker_bench(db_path, market)
        recs = scan_ticker(df, market, bench, s, cal, out["counters"])
        out["td"] = _ticker_data(df, ticker, uorder, recs)
        if raw:
            out["raw_trades"] = simulate_raw(df, ticker, market, bench, s)
        out["simulated"] = True
    except Exception as exc:  # one bad ticker must not kill a long run
        out["error"] = f"{exc.__class__.__name__}: {exc}"
        out["td"], out["raw_trades"], out["counters"] = None, [], new_counters()
    return out


# --------------------------------------------------------------------------- baselines

def index_return(dates: np.ndarray, opens: np.ndarray, closes: np.ndarray,
                 entry_date: str, exit_date: str, div_yield_pct: float = 0.0) -> float | None:
    """Index open on the first bar on/after entry -> close on the last bar on/before
    exit, %, plus an estimated dividend yield accrued over those calendar days
    (for price indices; 0 leaves the value bit-identical)."""
    po = int(np.searchsorted(dates, entry_date, side="left"))
    pc = int(np.searchsorted(dates, exit_date, side="right")) - 1
    if po >= len(dates) or pc < 0 or po > pc:
        return None
    o, c = float(opens[po]), float(closes[pc])
    if not (o > 0 and c > 0):
        return None
    if not div_yield_pct:
        return (c / o - 1) * 100
    days = (date.fromisoformat(str(dates[pc])[:10]) - date.fromisoformat(str(dates[po])[:10])).days
    return ((c / o) * (1 + div_yield_pct / 100) ** (days / 365.25) - 1) * 100


def load_long_prices(store: PriceStore, tickers: list[str]) -> pd.DataFrame:
    """ticker, date, open, high, low, close, volume rows for many tickers (chunked)."""
    parts = []
    for k in range(0, len(tickers), 400):
        chunk = tickers[k: k + 400]
        q = ("SELECT ticker, date, open, high, low, close, volume FROM bars WHERE ticker IN ("
             + ",".join("?" * len(chunk)) + ") ORDER BY ticker, date")
        parts.append(pd.read_sql_query(q, store.conn, params=chunk))
    if not parts:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])
    return pd.concat(parts, ignore_index=True)


def frames_to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{ticker: OHLC(V) DataFrame with a date index} -> long format (used by tests)."""
    rows = []
    for t, df in frames.items():
        cols = [k for k in ("open", "high", "low", "close", "volume") if k in df]
        d = df[cols].copy()
        d["date"] = _date_strings(df.index)
        d["ticker"] = t
        rows.append(d.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def sanitize_long(long: pd.DataFrame, overrides=(), stale_map: dict | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """sanitize_bars per ticker on a long frame (before the MarketPanel pivot), so
    the baselines see the same series as the simulation. Returns (long, report)
    with report["suspect"] = {ticker: [unresolved dates]}."""
    report = {"counts": {}, "events": [], "suspect": {}}
    if long is None or long.empty:
        return long, report
    stale_map = stale_map or {}
    cols = [k for k in ("open", "high", "low", "close", "volume") if k in long.columns]
    parts = []
    long = long.copy()
    long["date"] = long["date"].astype(str).str[:10]
    for t, g in long.groupby("ticker", sort=True):
        g = g.sort_values("date")
        df = g.set_index("date")[cols]
        clean, rep = sanitize_bars(df, str(t), overrides, stale_map.get(str(t)))
        merge_price_reports(report, rep)
        if rep["suspect"]:
            report["suspect"][str(t)] = list(rep["suspect"])
        part = clean.reset_index().rename(columns={"index": "date"})
        part["ticker"] = t
        parts.append(part)
    return pd.concat(parts, ignore_index=True), report


class MarketPanel:
    """Date x ticker price panel of one market's universe, for the baselines.

    Dates are the union of all tickers' trading days; a ticker's missing days
    (holidays, data gaps, not yet listed) are simply absent for that ticker.
    With volume and min_dollar_vol, random entries are drawn only from stocks
    that passed the liquidity floor the day before, with a similar 14-day ATR%.
    `suspect` ({ticker: [dates]}) masks unresolved bad prints out of both
    baselines.
    """

    def __init__(self, long: pd.DataFrame, min_dollar_vol: float | None = None,
                 suspect: dict | None = None):
        long = long.dropna(subset=["open", "close"])
        long = long[(long["open"] > 0) & (long["close"] > 0)].copy()
        long["date"] = long["date"].astype(str).str[:10]
        long["ticker"] = long["ticker"].astype(str)
        has_vol = "volume" in long.columns
        fields = ("open", "high", "low", "close") + (("volume",) if has_vol else ())
        piv = {f: long.pivot(index="date", columns="ticker", values=f) for f in fields}
        self.dates = np.asarray(piv["open"].index, dtype=str)
        self.tickers = [str(t) for t in piv["open"].columns]
        self.col = {t: k for k, t in enumerate(self.tickers)}
        self.O = piv["open"].to_numpy(float)
        self.H = piv["high"].to_numpy(float)
        self.L = piv["low"].to_numpy(float)
        self.C = piv["close"].to_numpy(float)
        V = piv["volume"].to_numpy(float) if has_vol else None
        n = self.O.shape[0]
        self.valid = ~np.isnan(self.O) & ~np.isnan(self.C)
        self.n_valid = self.valid.sum(axis=0)
        pos = np.arange(n, dtype=np.int32)[:, None]
        nxt = np.where(self.valid, pos, n).astype(np.int32)
        # first valid bar at or after each date (n = none), per ticker
        self.next_idx = np.ascontiguousarray(np.minimum.accumulate(nxt[::-1], axis=0)[::-1])
        prv = np.where(self.valid, pos, -1).astype(np.int32)
        # last valid bar at or before each date (-1 = none), per ticker
        self.prev_idx = np.maximum.accumulate(prv, axis=0)
        self.comp_idx = np.cumsum(self.valid, axis=0, dtype=np.int32) - 1
        self.last_valid = self.prev_idx[-1] if n else np.array([], dtype=np.int32)
        self.compact = [tuple(a[self.valid[:, k], k] for a in (self.O, self.H, self.L, self.C))
                        for k in range(len(self.tickers))]
        # liquidity (20-day average traded value) and 14-day ATR%, per ticker's own bars
        self.min_dollar_vol = min_dollar_vol if has_vol else None
        self.DV = np.full(self.O.shape, np.nan)
        self.ATRP = np.full(self.O.shape, np.nan)
        for k in range(len(self.tickers)):
            rows = np.nonzero(self.valid[:, k])[0]
            if not len(rows):
                continue
            o, h, l, c = self.compact[k]
            pc = np.r_[np.nan, c[:-1]]
            tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
            atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
            self.ATRP[rows, k] = atr / c * 100
            if V is not None:
                self.DV[rows, k] = pd.Series(c * V[rows, k]).rolling(20).mean().to_numpy()
        # unresolved bad prints: cumulative count per ticker on the date grid
        self.suspect_cs = None
        self.suspect_compact: dict[int, np.ndarray] = {}
        if suspect:
            flags = np.zeros(self.O.shape, np.int32)
            date_pos = {d: k for k, d in enumerate(self.dates)}
            for t, ds in suspect.items():
                k = self.col.get(str(t))
                if k is None:
                    continue
                rows = [date_pos[d] for d in ds if d in date_pos]
                if rows:
                    flags[rows, k] = 1
                    self.suspect_compact[k] = np.sort(self.comp_idx[rows, k])
            if flags.any():
                self.suspect_cs = np.cumsum(flags, axis=0)
        self.pool_counts = {"random_pool_fallback": 0, "random_pool_unmatched": 0}
        self._uni_cache: dict[tuple[str, str], float | None] = {}
        self._cand_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def universe_return(self, entry_date: str, exit_date: str) -> float | None:
        """Equal-weight mean return (%) of all tickers listed across the window:
        each bought at its open on its first bar on/after entry_date and sold at
        its close on its last bar on/before exit_date. No costs."""
        key = (entry_date, exit_date)
        if key not in self._uni_cache:
            self._uni_cache[key] = self._universe_return(entry_date, exit_date)
        return self._uni_cache[key]

    def _universe_return(self, entry_date: str, exit_date: str) -> float | None:
        n = len(self.dates)
        pe = int(np.searchsorted(self.dates, entry_date, side="left"))
        px = int(np.searchsorted(self.dates, exit_date, side="right")) - 1
        if pe >= n or px < 0 or pe > px:
            return None
        # only tickers that already traded at entry and still traded at exit
        # (a listing or a data gap in mid-window must not count as a full hold)
        at_entry = int(np.searchsorted(self.dates, entry_date, side="right")) - 1
        at_exit = int(np.searchsorted(self.dates, exit_date, side="left"))
        if at_entry < 0 or at_exit >= n:
            return None
        o_idx = self.next_idx[pe]
        c_idx = self.prev_idx[px]
        ok = ((o_idx < n) & (c_idx >= 0) & (o_idx <= c_idx)
              & (self.prev_idx[at_entry] >= 0) & (self.next_idx[at_exit] < n))
        cols = np.nonzero(ok)[0]
        if self.suspect_cs is not None and len(cols):
            hi = self.suspect_cs[c_idx[cols], cols]
            lo_i = o_idx[cols] - 1
            lo = np.where(lo_i >= 0, self.suspect_cs[np.maximum(lo_i, 0), cols], 0)
            cols = cols[(hi - lo) == 0]
        if len(cols) == 0:
            return None
        rets = self.C[c_idx[cols], cols] / self.O[o_idx[cols], cols] - 1
        return float(rets.mean() * 100)

    def _pools(self, p: int, max_hold: int) -> tuple[np.ndarray, np.ndarray]:
        """(base, liquid) candidate columns for entry day p (sorted): a bar on day
        p and at least max_hold bars after it; liquid = also passed the liquidity
        floor on the day before."""
        key = (p, max_hold)
        if key not in self._cand_cache:
            base = self.valid[p] & ((self.n_valid - 1 - self.comp_idx[p]) >= max_hold)
            if self.min_dollar_vol is None:
                liquid = base
            else:
                prev = self.prev_idx[p - 1] if p >= 1 else np.full(len(self.tickers), -1)
                dv = np.where(prev >= 0, self.DV[np.maximum(prev, 0), np.arange(len(self.tickers))],
                              np.nan)
                liquid = base & (np.nan_to_num(dv, nan=-1.0) >= self.min_dollar_vol)
            self._cand_cache[key] = (np.nonzero(base)[0], np.nonzero(liquid)[0])
        return self._cand_cache[key]

    def _clean_of_suspects(self, cols: np.ndarray, p: int, max_hold: int) -> np.ndarray:
        if not self.suspect_compact or not len(cols):
            return cols
        keep = []
        for col in cols:
            flags = self.suspect_compact.get(int(col))
            if flags is not None:
                fi = int(self.comp_idx[p, col])
                if ((flags >= fi) & (flags <= fi + max_hold)).any():
                    continue
            keep.append(col)
        return np.asarray(keep, dtype=int)

    def random_baseline(self, trade: dict, k: int, max_hold: int, cost_pct: float,
                        slip_pct: float) -> tuple[float | None, float | None]:
        """Mean (r_net, ret_net_pct) of k random same-day entries with the trade's
        stop / target distances, exit rules and costs. Seeded by crc32 of
        ticker|entry_date|pattern, so it is identical in every process."""
        entry_date = trade["entry_date"]
        p = int(np.searchsorted(self.dates, entry_date, side="left"))
        if k <= 0 or p >= len(self.dates) or self.dates[p] != entry_date:
            return None, None
        base, liquid = self._pools(p, max_hold)
        tcol = self.col.get(str(trade.get("ticker")))
        if tcol is not None:
            base = base[base != tcol]
            liquid = liquid[liquid != tcol]
        base = self._clean_of_suspects(base, p, max_hold)
        liquid = self._clean_of_suspects(liquid, p, max_hold)
        pool = None
        if tcol is not None and self.min_dollar_vol is not None and len(liquid):
            prev_t = int(self.prev_idx[p - 1, tcol]) if p >= 1 else -1
            my_atr = self.ATRP[prev_t, tcol] if prev_t >= 0 else np.nan
            if np.isfinite(my_atr) and my_atr > 0:
                prev = self.prev_idx[p - 1, liquid] if p >= 1 else np.full(len(liquid), -1)
                atr = np.where(prev >= 0, self.ATRP[np.maximum(prev, 0), liquid], np.nan)
                for lo, hi in ((0.8, 1.25), (0.67, 1.5)):
                    band = liquid[(atr >= lo * my_atr) & (atr <= hi * my_atr)]
                    if len(band) >= k:
                        pool = band
                        break
                if pool is None:
                    self.pool_counts["random_pool_fallback"] += 1
        if pool is None:
            pool = liquid
        if len(pool) == 0:
            pool = base
            if self.min_dollar_vol is not None:
                self.pool_counts["random_pool_unmatched"] += 1
        if len(pool) == 0:
            return None, None
        fill0, stop0, target0 = trade["entry_price"], trade["stop"], trade["target"]
        risk_frac = (fill0 - stop0) / fill0
        target_frac = (target0 - fill0) / fill0
        if risk_frac <= 0:
            return None, None
        seed = zlib.crc32(f"{trade['ticker']}|{entry_date}|{trade['pattern']}".encode())
        rng = np.random.default_rng(seed)
        picks = rng.choice(pool, size=k, replace=len(pool) < k)
        r_nets, rets = [], []
        for col in picks:
            o, h, l, c = self.compact[int(col)]
            fi = int(self.comp_idx[p, col])
            fill = float(o[fi])
            stop = fill * (1 - risk_frac)
            target = fill * (1 + target_frac)
            _, exit_price, outcome = simulate_exit(o, h, l, c, fi, fill, stop, target, max_hold)
            if outcome == "open_at_end":
                continue   # unreachable: candidates have max_hold bars after the entry
            res = trade_returns(fill, stop, exit_price, outcome, cost_pct, slip_pct)
            r_nets.append(res["r_net"])
            rets.append(res["ret_net_pct"])
        if not r_nets:
            return None, None
        return sum(r_nets) / len(r_nets), sum(rets) / len(rets)


def add_baselines(trades: list[dict], panel: MarketPanel, bench: BenchContext,
                  s: Settings, market: str) -> int:
    """Fill each trade's index / universe / random-entry baselines in place.
    Returns how many closed trades are missing the index or universe baseline."""
    missing = 0
    cost = s.cost_pct(market)
    yld = s.div_yield_pct(market)
    for t in trades:
        if t["outcome"] == "open_at_end":
            continue
        idx = index_return(bench.dates, bench.opens, bench.closes,
                           t["entry_date"], t["exit_date"], yld)
        idx0 = index_return(bench.dates, bench.opens, bench.closes,
                            t["entry_date"], t["exit_date"], 0.0)
        uni = panel.universe_return(t["entry_date"], t["exit_date"])
        rr, rret = panel.random_baseline(t, s.random_entries_per_trade, s.max_hold, cost,
                                         s.stop_slip(market))
        t["index_ret_pct"] = _rnd(idx, 3)
        t["index_ret_price_pct"] = _rnd(idx0, 3)
        t["universe_ret_pct"] = _rnd(uni, 3)
        t["excess_vs_index_pct"] = None if idx is None else _rnd(t["ret_net_pct"] - idx, 3)
        t["excess_vs_universe_pct"] = None if uni is None else _rnd(t["ret_net_pct"] - uni, 3)
        # per unit of risk: holding the market with the same position size
        rp = (t["entry_price"] - t["stop"]) / t["entry_price"] * 100
        ok = rp > 0
        t["excess_vs_index_r"] = None if (idx is None or not ok) else _rnd(t["r_net"] - idx / rp, 4)
        t["excess_vs_universe_r"] = None if (uni is None or not ok) else _rnd(t["r_net"] - uni / rp, 4)
        t["rand_r_net"] = _rnd(rr, 4)
        t["rand_ret_net_pct"] = _rnd(rret, 3)
        if idx is None or uni is None:
            missing += 1
    return missing


_BASELINE_KEYS = ("index_ret_pct", "index_ret_price_pct", "universe_ret_pct",
                  "excess_vs_index_pct", "excess_vs_universe_pct", "excess_vs_index_r",
                  "excess_vs_universe_r", "rand_r_net", "rand_ret_net_pct")


def _trade_identity(t: dict) -> tuple:
    return (t["ticker"], t["pattern"], t["signal_date"], t["entry_date"], t["exit_date"],
            t["entry_price"], t["stop"], t["target"])


# --------------------------------------------------------------------------- statistics

def _cr1_se(values: list[float], labels: list[str], mean: float) -> tuple[float | None, int]:
    """CR1 cluster-robust SE of the mean: sqrt(G/(G-1) * sum_g (sum_{i in g}(x_i - mean))^2) / N."""
    sums: dict[str, float] = {}
    for x, g in zip(values, labels):
        sums[g] = sums.get(g, 0.0) + (x - mean)
    g_count = len(sums)
    if g_count < 2:
        return None, g_count
    return math.sqrt(g_count / (g_count - 1) * sum(v * v for v in sums.values())) / len(values), g_count


def cluster_se(values: list[float], clusters: list[str]) -> tuple[float, float] | None:
    """(mean, SE) with the SE clustered by `clusters`; None with fewer than 2 clusters."""
    n = len(values)
    if n == 0 or n != len(clusters):
        return None
    mean = sum(values) / n
    se, _ = _cr1_se(values, clusters, mean)
    return None if se is None else (mean, se)


def cluster_ci95(values: list[float], clusters: list[str]) -> tuple[float, float] | None:
    """mean +/- 1.96 * SE, with SE clustered by `clusters` (one clustering only;
    kept for reference - the stats use cluster_ci)."""
    r = cluster_se(values, clusters)
    if r is None:
        return None
    mean, se = r
    return mean - 1.96 * se, mean + 1.96 * se


def t_crit(p: float, nu: float) -> float:
    """Student-t quantile by the Cornish-Fisher expansion around the normal
    quantile. Valid for nu >= 8, error at most 4e-4 at p = 0.975 (about 7e-3 at
    the far tail with nu = 9)."""
    z = NormalDist().inv_cdf(p)
    return (z + (z ** 3 + z) / (4 * nu) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * nu ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * nu ** 3))


def _quarter(d: str) -> str:
    return f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}"


def cluster_stats(values: list[float], entry_dates: list[str], level: float = 0.95,
                  min_quarters: int = 10) -> dict | None:
    """Mean, SE, degrees of freedom and range for one group of trades.

    SE = the larger of the CR1 standard errors clustered by entry month and by
    entry quarter (a trade lasts up to 40 bars, so neighbouring months share
    market weather); range = mean +/- t(df = quarters - 1) * SE. None when the
    trades span fewer than `min_quarters` quarters."""
    n = len(values)
    if n == 0 or n != len(entry_dates):
        return None
    mean = sum(values) / n
    se_m, _ = _cr1_se(values, [d[:7] for d in entry_dates], mean)
    se_q, g_q = _cr1_se(values, [_quarter(d) for d in entry_dates], mean)
    if se_m is None or se_q is None or g_q < min_quarters:
        return None
    se = max(se_m, se_q)
    crit = t_crit((1 + level) / 2, g_q - 1)
    return {"mean": mean, "se": se, "df": g_q - 1, "lo": mean - crit * se, "hi": mean + crit * se}


def cluster_ci(values: list[float], entry_dates: list[str], level: float = 0.95,
               min_quarters: int = 10) -> tuple[float, float] | None:
    st = cluster_stats(values, entry_dates, level, min_quarters)
    return None if st is None else (st["lo"], st["hi"])


def _ci(values: list[float], entry_dates: list[str], nd: int) -> list[float] | None:
    ci = cluster_ci(values, entry_dates)
    return None if ci is None else [_rnd(ci[0], nd), _rnd(ci[1], nd)]


def _brief_stats(ts: list[dict]) -> dict:
    n = len(ts)
    rn = [t["r_net"] for t in ts]
    wi = [t for t in ts if t.get("excess_vs_index_pct") is not None]
    wu = [t for t in ts if t.get("excess_vs_universe_pct") is not None]
    return {
        "trades": n,
        "win_rate": _pct(sum(1 for x in rn if x > 0), n),
        "net_avg_r": _rnd(_mean(rn), 4),
        "net_avg_ret_pct": _rnd(_mean([t["ret_net_pct"] for t in ts]), 3),
        "excess_vs_index_pct": _rnd(_mean([t["excess_vs_index_pct"] for t in wi]), 3),
        "excess_vs_universe_pct": _rnd(_mean([t["excess_vs_universe_pct"] for t in wu]), 3),
        "net_avg_r_ci95": _ci(rn, [t["entry_date"] for t in ts], 4),
        "excess_vs_universe_ci95": _ci([t["excess_vs_universe_pct"] for t in wu],
                                       [t["entry_date"] for t in wu], 3),
    }


def pattern_stats(ts: list[dict], tuning: dict) -> dict:
    """All A5 statistics for one group of closed trades, plus its verdict."""
    n = len(ts)
    dates = [t["entry_date"] for t in ts]
    r = [t["r"] for t in ts]
    rn = [t["r_net"] for t in ts]
    with_idx = [t for t in ts if t.get("excess_vs_index_pct") is not None]
    with_uni = [t for t in ts if t.get("excess_vs_universe_pct") is not None]
    ex_i = [t["excess_vs_index_pct"] for t in with_idx]
    ex_u = [t["excess_vs_universe_pct"] for t in with_uni]
    rand = [t["rand_r_net"] for t in ts if t.get("rand_r_net") is not None]
    pairs = [(t["r_net"] - t["rand_r_net"], t["entry_date"]) for t in ts
             if t.get("rand_r_net") is not None]
    xr_i = [t for t in ts if t.get("excess_vs_index_r") is not None]
    xr_u = [t for t in ts if t.get("excess_vs_universe_r") is not None]
    idx_price = [t["index_ret_price_pct"] for t in with_idx if t.get("index_ret_price_pct") is not None]
    net_avg_r = _mean(rn)
    by_year_trades: dict[str, list[dict]] = defaultdict(list)
    for t in ts:
        by_year_trades[t["entry_date"][:4]].append(t)
    by_year = {}
    for y, g in sorted(by_year_trades.items()):
        gu = [t["excess_vs_universe_pct"] for t in g if t.get("excess_vs_universe_pct") is not None]
        by_year[y] = {"trades": len(g), "net_avg_r": _rnd(_mean([t["r_net"] for t in g]), 4),
                      "excess_vs_universe_pct": _rnd(_mean(gu), 3)}
    counted = [y for y, v in by_year.items() if v["trades"] >= 10]
    positive = sum(1 for y in counted if (by_year[y]["excess_vs_universe_pct"] or 0) > 0)
    su = cluster_stats(ex_u, [t["entry_date"] for t in with_uni])
    ci_u = None if su is None else [_rnd(su["lo"], 3), _rnd(su["hi"], 3)]
    stats = {
        "trades": n,
        "win_rate": _pct(sum(1 for x in rn if x > 0), n),
        "gross_win_rate": _pct(sum(1 for x in r if x > 0), n),
        "avg_r": _rnd(_mean(r), 4),
        "net_avg_r": _rnd(net_avg_r, 4),
        "avg_ret_pct": _rnd(_mean([t["ret_pct"] for t in ts]), 3),
        "net_avg_ret_pct": _rnd(_mean([t["ret_net_pct"] for t in ts]), 3),
        "avg_hold_bars": _rnd(_mean([t["hold_bars"] for t in ts]), 1),
        "outcomes": {o: sum(1 for t in ts if t["outcome"] == o) for o in ("target", "stop", "time")},
        "avg_index_ret_pct": _rnd(_mean([t["index_ret_pct"] for t in with_idx]), 3),
        "avg_index_ret_price_pct": _rnd(_mean(idx_price), 3),
        "avg_universe_ret_pct": _rnd(_mean([t["universe_ret_pct"] for t in with_uni]), 3),
        "excess_vs_index_pct": _rnd(_mean(ex_i), 3),
        "excess_vs_universe_pct": _rnd(_mean(ex_u), 3),
        "gross_excess_vs_universe_pct": _rnd(_mean([t["ret_pct"] - t["universe_ret_pct"]
                                                    for t in with_uni]), 3),
        "beat_index_rate": _pct(sum(1 for x in ex_i if x > 0), len(ex_i)),
        "random_entry_net_avg_r": _rnd(_mean(rand), 4),
        # paired: each trade minus its own random draws, so the value sits inside its range
        "edge_vs_random_r": _rnd(_mean([d for d, _ in pairs]), 4),
        "edge_vs_random_r_ci95": _ci([d for d, _ in pairs], [g for _, g in pairs], 4),
        "excess_vs_index_r": _rnd(_mean([t["excess_vs_index_r"] for t in xr_i]), 4),
        "excess_vs_universe_r": _rnd(_mean([t["excess_vs_universe_r"] for t in xr_u]), 4),
        "excess_vs_index_r_ci95": _ci([t["excess_vs_index_r"] for t in xr_i],
                                      [t["entry_date"] for t in xr_i], 4),
        "excess_vs_universe_r_ci95": _ci([t["excess_vs_universe_r"] for t in xr_u],
                                         [t["entry_date"] for t in xr_u], 4),
        "net_avg_r_ci95": _ci(rn, dates, 4),
        "excess_vs_index_ci95": _ci(ex_i, [t["entry_date"] for t in with_idx], 3),
        "excess_vs_universe_ci95": ci_u,
        "excess_vs_universe_se": None if su is None else _rnd(su["se"], 4),
        "excess_vs_universe_df": None if su is None else su["df"],
        "clearly_negative": bool(ci_u) and ci_u[1] is not None and ci_u[1] < 0,
        "by_year": by_year,
        "positive_years": positive,
        "years_counted": len(counted),
    }
    if any(t.get("grade_proxy") for t in ts):
        stats["by_grade"] = {g: _brief_stats([t for t in ts if t.get("grade_proxy") == g])
                             for g in ("A", "B", "C")}
    stats["verdict"], stats["verdict_reason"] = pattern_verdict(stats, tuning)
    return stats


def detector_names() -> list[str]:
    return [d.__name__.removeprefix("detect_") for d in BULLISH_DETECTORS]


def _zero_trade_reason(name: str, info: dict | None, min_rr: float) -> str:
    pi = (info or {}).get(name) if info is not None else None
    if not pi or not pi.get("detected"):
        return ("Never produced a trade: the pattern was not found on any chart that passed "
                "the scanner's filters")
    mx = pi.get("max_rr")
    if mx is not None and mx < min_rr:
        return (f"Never produced a trade: its best risk:reward was {mx:.2f} (its target is "
                f"too close compared with its stop), so the risk:reward minimum ({min_rr:g}) "
                "filters out every signal")
    return ("Never produced a trade: every signal was filtered out (risk:reward, shortlist, "
            "entry rules) or lost the pick to a stronger pattern on the same chart")


def summarize(trades: list[dict], tuning: dict, info: dict | None = None,
              min_rr: float = 1.8) -> dict:
    """Per-pattern and total stats over closed trades (open_at_end excluded).
    Every bullish detector gets a row (and a verdict), even with 0 trades."""
    closed = [t for t in trades if t["outcome"] != "open_at_end"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for name in detector_names():
        groups.setdefault(name, [])
    for t in closed:
        groups[t["pattern"]].append(t)
    per_pattern = {}
    for pat, ts in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        st = pattern_stats(ts, tuning)
        if not ts:
            st["verdict_reason"] = _zero_trade_reason(pat, info, min_rr)
        per_pattern[pat] = st
    total_tuning = {**tuning, "n_tests": _n_tests(tuning) + 3}
    return {"per_pattern": per_pattern, "total": pattern_stats(closed, total_tuning)}


def _short_stats(ts: list[dict]) -> dict:
    closed = [t for t in ts if t["outcome"] != "open_at_end"]
    n = len(closed)
    return {
        "trades": n,
        "avg_r": _rnd(_mean([t["r"] for t in closed]), 4),
        "net_avg_r": _rnd(_mean([t["r_net"] for t in closed]), 4),
        "win_rate": _pct(sum(1 for t in closed if t["r_net"] > 0), n),
        "net_avg_ret_pct": _rnd(_mean([t["ret_net_pct"] for t in closed]), 3),
    }


def raw_summary(raw_trades: list[dict]) -> tuple[dict, dict]:
    """(raw_total, raw_per_pattern) for the attribution block."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in raw_trades:
        groups[t["pattern"]].append(t)
    per = {pat: _short_stats(ts) for pat, ts in sorted(groups.items())}
    return _short_stats(raw_trades), per


# --------------------------------------------------------------------------- verdict

def _n_tests(tuning: dict) -> int:
    """Pattern x market cells the verdicts are corrected for (default 6 x 3 = 18)."""
    return int((tuning or {}).get("n_tests") or len(BULLISH_DETECTORS) * 3)


def _verdict_z(tuning: dict, df: float | None = None) -> float:
    """Critical value for 'proven': two-sided family_alpha split over n_tests
    (Bonferroni); a t value when the degrees of freedom are known."""
    alpha = float((tuning or {}).get("family_alpha", 0.05))
    p = 1 - alpha / (2 * _n_tests(tuning))
    if df is not None and df >= 1:
        return t_crit(p, df)
    return NormalDist().inv_cdf(p)


def _has_range(ci) -> bool:
    return isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None \
        and ci[1] is not None


def pattern_verdict(stats: dict, tuning: dict) -> tuple[str, str]:
    """Plain verdict for one pattern's v2 stats under a pattern_tuning config.

    insufficient_data: fewer than min_trades trades.
    no_edge:  net avg R <= min_avg_r, or it did not beat holding the index, or
              did not beat an equal-weight basket of the market (after fees).
    unproven: positive, but the range of the edge over the market, widened for
              the number of pattern x market tests, includes 0.
    proven_edge: otherwise.
    Used by the backtest (stored) and by scan.load_pattern_tuning (current config).
    """
    tuning = tuning or {}
    min_trades = int(tuning.get("min_trades", 30))
    min_avg_r = float(tuning.get("min_avg_r", 0.0))
    n = int(stats.get("trades") or 0)
    if n < min_trades:
        return "insufficient_data", (f"Only {n} trades in the backtest - at least "
                                     f"{min_trades} are needed to judge")
    failed = []
    nr = stats.get("net_avg_r")
    if nr is None:
        nr = stats.get("avg_r") or 0.0
    if nr <= min_avg_r:
        if min_avg_r != 0:
            failed.append(f"after fees it averaged {_sg(nr, 2)}R per trade, not above the "
                          f"required {_sg(min_avg_r, 2)}R")
        elif nr < -0.005:
            failed.append(f"after fees it lost money: {_sg(nr, 2)}R per trade on average")
        else:
            failed.append("after fees it made no money on average (0.00R per trade)")
    ei = stats.get("excess_vs_index_pct")
    if ei is not None and ei <= 0:
        if ei < -0.005:
            failed.append(f"after fees it made {abs(ei):.2f}% per trade less than just "
                          "holding the market index over the same days")
        else:
            failed.append("after fees it did no better than just holding the market index "
                          "over the same days")
    eu = stats.get("excess_vs_universe_pct")
    if eu is not None and eu <= 0:
        if eu < -0.005:
            failed.append(f"after fees it made {abs(eu):.2f}% per trade less than an "
                          "equal-weight basket of the market's stocks over the same days")
        else:
            failed.append("after fees it did no better than an equal-weight basket of the "
                          "market's stocks over the same days")
    ci = stats.get("excess_vs_universe_ci95")
    has_ci = _has_range(ci)
    if failed:
        reason = "; ".join(failed)
        if eu is not None and eu <= 0:
            # the market comparison failed: how sure, and was it the fees?
            if has_ci:
                rng = f"95% range vs the market {ci[0]:+.2f}% to {ci[1]:+.2f}% per trade"
                reason += (f" ({rng} - the whole range is below zero)" if ci[1] < 0
                           else f" ({rng} - not clearly worse than the market, just not "
                           "better on average)")
            g = stats.get("gross_excess_vs_universe_pct")
            if g is not None:
                reason += ("; it lost to the market even before fees" if g <= 0
                           else f"; fees ate it (break-even cost about {g / 2:.2f}% per side)")
        else:
            ci_i, ci_r = stats.get("excess_vs_index_ci95"), stats.get("net_avg_r_ci95")
            if ei is not None and ei <= 0 and _has_range(ci_i):
                reason += f" (95% range vs the index {ci_i[0]:+.2f}% to {ci_i[1]:+.2f}% per trade)"
            elif nr <= min_avg_r and _has_range(ci_r):
                reason += f" (95% range {ci_r[0]:+.2f}R to {ci_r[1]:+.2f}R per trade)"
            if eu is not None:
                reason += (f"; it did beat the equal-weight market by +{eu:.2f}% per trade "
                           "after fees")
        return "no_edge", reason[0].upper() + reason[1:]
    se = stats.get("excess_vs_universe_se")
    df = stats.get("excess_vs_universe_df")
    if se is None and has_ci:
        se, df = (ci[1] - ci[0]) / (2 * 1.96), None   # older v2 files: 1.96-based range
    if eu is None or se is None:
        if eu is None:
            return "unproven", ("Positive on average, but there is no market comparison to "
                                "tell it apart from luck")
        return "unproven", (f"Positive on average (+{eu:.2f}% per trade vs the market after "
                            "fees) but there are too few quarters of trades to measure the "
                            "uncertainty")
    z = _verdict_z(tuning, df)
    lo, hi = eu - z * se, eu + z * se
    k = _n_tests(tuning)
    single = f"; single-test 95% range {ci[0]:+.2f}% to {ci[1]:+.2f}%" if has_ci else ""
    if lo <= 0:
        return "unproven", (f"Positive on average (+{eu:.2f}% per trade vs the market after "
                            f"fees) but not distinguishable from luck once {k} pattern x market "
                            f"tests are counted (adjusted range {lo:+.2f}% to {hi:+.2f}%{single})")
    return "proven_edge", (f"Beat the market by +{eu:.2f}% per trade after fees, statistically "
                           f"clear in this backtest after adjusting for {k} pattern x market "
                           f"tests (adjusted range {lo:+.2f}% to {hi:+.2f}%{single}; still "
                           f"in-sample), {_sg(nr, 2)}R per trade")


def disabled_patterns(per_pattern: dict, tuning: dict, market: str) -> set[str]:
    """Patterns the live scanner switches off in `market`: the manual
    pattern_tuning.disabled[market] list, plus (with auto_disable) verdict
    'no_edge' - or 'unproven' with require_proven_edge - recomputed under
    `tuning`; stats without net_avg_r (v1 files) use the old gross rule."""
    tuning = _deep_merge(PATTERN_TUNING_DEFAULTS, tuning or {})
    manual = tuning.get("disabled")
    manual = manual if isinstance(manual, dict) else {}
    out = {str(p) for p in (manual.get(market) or [])}
    if not tuning.get("auto_disable", True):
        return out
    for pat, st in (per_pattern or {}).items():
        if not isinstance(st, dict):
            continue
        if "net_avg_r" in st:
            v = pattern_verdict(st, tuning)[0]
            if v == "no_edge" or (tuning.get("require_proven_edge") and v == "unproven"):
                out.add(str(pat))
        elif (st.get("trades", 0) >= tuning.get("min_trades", 30)
              and st.get("avg_r", 0) < tuning.get("min_avg_r", 0.0)):
            out.add(str(pat))
    return out


def disable_rule_text(tuning: dict) -> str:
    rule = "'no edge' or 'unproven'" if tuning.get("require_proven_edge") else "'no edge'"
    if not tuning.get("auto_disable", True):
        return "manual pattern_tuning.disabled lists only (auto_disable is off)"
    return f"verdict {rule} under the pattern_tuning config, plus the manual disabled lists"


# --------------------------------------------------------------------------- report text

def _settings_dict(s: Settings) -> dict:
    """The simulation settings recorded in method.settings."""
    return {"step": s.step, "window": s.window, "breakout_wait": s.breakout_wait,
            "max_hold": s.max_hold,
            "max_gap_above_entry_pct": s.max_gap_above_entry_pct,
            "min_risk_reward": s.min_risk_reward,
            "tech_score_min": TECH_SCORE_MIN, "regime_filter": s.regime_filter,
            "shortlist_size": s.shortlist_size,
            "seen_on_fill_only": s.seen_on_fill_only, "entry_mode": s.entry_mode}


def _method(s: Settings, markets_run: list[str], periods: dict, universe: dict,
            raw: bool, limit: int | None, tuning: dict) -> dict:
    daily = s.step == 1 and s.breakout_wait == 1
    filters = [
        "liquidity: 20-day average traded value on the signal day at least "
        + ", ".join(f"{MIN_DOLLAR_VOL[m]:,.0f} {MARKET_CCY[m]} ({m.upper()})" for m in markets_run)
        + " - the live scanner's floor",
        f"technical score at least {TECH_SCORE_MIN} (the live scanner's formula; relative "
        "strength vs the index uses only data up to the signal day)",
    ]
    if s.regime_filter:
        filters.append("market regime: no new trades while the index is below its 200-day "
                       "average ('downtrend'), judged on the signal day and re-checked on the "
                       "fill day (previous close)")
    filters += [
        "pattern choice: the highest-confidence bullish pattern on the chart (ties broken by "
        "risk:reward), exactly like the scanner - chosen BEFORE the already-traded check (a "
        "deliberate change from spec A2 d) so the backtest matches live)",
        f"risk:reward at least {s.min_risk_reward:g} for that best pattern (no fall-back to a "
        "weaker one)",
        f"shortlist: each signal day the candidates of the whole universe are ranked by pattern "
        f"confidence x 50 + technical score and only the top {s.shortlist_size} are kept, as "
        "the live scanner does (ties in universe order)",
    ]
    if daily:
        filters.append(
            "entry: signals are checked every trading day; a 'forming' setup gets a one-day "
            "buy-stop at that day's levels and, if not filled, is re-detected the next day with "
            "refreshed levels and filters; 'triggered' setups are bought at the next open "
            f"(skipped if that open is more than {s.max_gap_above_entry_pct:g}% above the entry "
            "or below the stop)")
    else:
        filters.append(
            f"entry: signals are checked every {s.step} trading days; 'forming' setups need a "
            f"breakout above the entry level within {s.breakout_wait} bars (skipped if the stop "
            f"breaks first or the fill is more than {s.max_gap_above_entry_pct:g}% above the "
            "entry); 'triggered' setups are bought at the next open (skipped if that open is too "
            "far above the entry or below the stop)")
    if s.entry_mode == "close_confirm":
        filters.append("entry mode close_confirm: a forming setup is bought at the next open "
                       "after a close at or above the entry (skipped if that close ran more than "
                       "3% above it)")
    filters += [
        "a stock is not re-scanned while its order is pending (up to breakout_wait bars) or its "
        "position is open",
        f"a filled trade blocks that pattern level for {s.window} bars; if the scanner's top "
        "pick is such a level, the stock is skipped that day (the runner-up is never "
        "substituted); setups that did not fill are re-checked at the next signal day, as the "
        "live scanner keeps showing them"
        + ("" if s.seen_on_fill_only else " (legacy: any attempted order blocks the level)"),
        f"trades whose {s.max_hold}-bar holding window would run past the stock's last bar are "
        "left out whatever their outcome (counted as censored_at_end)",
        "removal counts are per signal day checked, so one setup can be counted more than once",
    ]
    z = _verdict_z(tuning)
    return {
        "mode": "scanner",
        "settings": _settings_dict(s),
        "costs_pct_per_side": {m: s.cost_pct(m) for m in markets_run},
        "stop_slippage_pct": {m: s.stop_slip(m) for m in markets_run},
        "entry_slippage_pct": {m: s.entry_slip(m) for m in markets_run},
        "random_entries_per_trade": s.random_entries_per_trade,
        "index_dividend_yield_pct": {m: s.div_yield_pct(m) for m in markets_run},
        "exits": (f"stop-loss, target, or a time exit at the close {s.max_hold} bars after the "
                  "fill (stop and target stay live through that last bar); a stop touched on "
                  "the fill day counts as hit; a gap through the stop exits at the open"),
        "costs": ("charged on entry and on exit as a percent of the trade value (commission + "
                  "spread, plus the FX fee for US trades from a EUR account, STT and charges "
                  "for India); buy-stop entries and stop-loss exits pay extra slippage (they "
                  "are market orders once triggered)"),
        "filters": filters,
        "baselines": {
            "index": ("buy the market index (" + ", ".join(
                f"{INDEX_NAMES[m]} {BENCHMARKS[m]}" for m in markets_run)
                + ") at the open of the first index bar on/after the entry day and sell at the "
                "close of the last index bar on/before the exit day; price indices are raised "
                "by an estimated dividend yield accrued daily (" + ", ".join(
                    f"{m.upper()} {s.div_yield_pct(m):g}%/yr" for m in markets_run)
                + "; the DAX already includes dividends); no costs charged (stands for a cheap "
                "index ETF). Changing index_dividend_yield_pct needs a backtest re-run"),
            "universe": ("equal-weight basket of every stock in the market's universe that "
                         "traded across the same days (open on entry day to close on exit "
                         "day); no costs charged"),
            "random": (f"{s.random_entries_per_trade} random stocks from the same market that "
                       "passed the liquidity floor on the day before entry, with 14-day ATR% "
                       "within 0.8-1.25x the traded stock's (widened to 0.67-1.5x, then any "
                       f"liquid stock, if fewer than {s.random_entries_per_trade} qualify); the "
                       "traded stock itself is excluded; bought at the entry-day open with the "
                       "trade's percentage stop and target, the same exits and the same costs; "
                       "only stocks with a full holding window of data are drawn"),
        },
        "stats": ("win rate and average R are after costs unless labelled gross; 'vs index' and "
                  "'vs market' are the average per-trade return after costs minus the "
                  "baseline's return over the same days; the '(R)' versions are per unit of "
                  "risk, above holding the equal-weight market with the same position size. "
                  "The (R) metrics match the uncapped 'Notional R account' portfolio profile; "
                  "the headline portfolio profiles size like the dashboard (entry-level "
                  "sizing, max_position_pct cap, cash limit), so stops under about 4% are "
                  "capped there; tight-stop trades dominate the (R) figure, so its range is "
                  "wider. 95% ranges are mean +/- a t critical "
                  "value (degrees of freedom = number of entry quarters - 1) times the larger of "
                  "two standard errors, one clustered by entry month and one by entry quarter. "
                  f"A trade lasts up to {s.max_hold} bars (about 1-2 months), so trades entered "
                  "in neighbouring months share market weather. No range (so at best "
                  "'unproven') when trades span fewer than 10 quarters. 'vs random entry' is "
                  "paired: each trade minus the average of its own random draws."),
        "verdicts": {
            "insufficient_data": "fewer trades than pattern_tuning.min_trades",
            "no_edge": ("average R after fees at or below pattern_tuning.min_avg_r, or did not "
                        "beat holding the index, or did not beat the equal-weight market"),
            "unproven": ("beat both on average, but the range vs the market - widened for the "
                         "number of pattern x market tests - includes zero"),
            "proven_edge": ("beat the market with the whole adjusted range above zero: "
                            "statistically clear in this backtest after adjusting for "
                            f"{_n_tests(tuning)} pattern x market tests (still in-sample)"),
            "family_alpha": float(tuning.get("family_alpha", 0.05)),
            "n_tests": _n_tests(tuning),
            "z": round(z, 3),
            "z_note": ("normal critical value; with the degrees of freedom of each pattern's "
                       "range the matching t value is used; the per-market totals use "
                       "n_tests + 3"),
        },
        "period": periods,
        "universe": universe,
        "raw_mode_included": raw,
        "limit": limit,
    }


def _caveats(s: Settings, tuning: dict) -> list[str]:
    out = [
        "Universe = today's index members (survivorship bias): stocks that were dropped or "
        "went bust during the decade are missing, which flatters the results. The "
        "equal-weight market baseline shares this bias; the index baseline does not.",
        "Fundamentals, earnings dates and news are not backtested (there is no point-in-time "
        "data for them), so the fundamental score and earnings warnings are not part of "
        "these results.",
        "The S&P 500 (^GSPC) and NIFTY 50 (^NSEI) are price indices; their baseline is raised by "
        f"an estimated dividend yield accrued daily (US {s.div_yield_pct('us'):g}%, IN "
        f"{s.div_yield_pct('in'):g}%, DE {s.div_yield_pct('de'):g}% because the DAX (^GDAXI) "
        "already includes dividends). Real dividends are lumpy and the yield drifted from about "
        "2% to 1.2% over 2016-26, so per-trade figures are approximate (about +/-0.05pp) while "
        "averages are reliable.",
        "Daily bars cannot show the order of moves within a day: when the stop and the target "
        "are both touched on the same day, the stop is assumed to have come first; when the "
        "entry and the stop are both touched on a day that opened between them, the trade is "
        "counted as filled and stopped out that day.",
    ]
    if s.step == 1 and s.breakout_wait == 1:
        out.append("Not modelled: the alert monitor's A-grade / top-10 cut and the partial "
                   "Indian bar at the 07:30 scan.")
    else:
        out.append(f"Signals are checked every {s.step} days with a {s.breakout_wait}-day "
                   "buy-stop frozen at the signal day's levels; shifting the grid start by 1-4 "
                   "days moved market averages by up to ~0.04R and flipped some pattern "
                   "verdicts. The live scanner runs daily and refreshes the levels every day.")
    out += [
        "Currency moves are not modelled: returns are in each market's local currency "
        "(the US cost allowance includes the EUR/USD conversion fee, not FX gains or losses).",
        "Baselines are charged no costs (they stand for a cheap index ETF held over the "
        "same days).",
        f"The top-{s.shortlist_size} shortlist cut is simulated; only the grades (which need "
        "fundamentals) and your own choice of setups remain unmodelled. The live scanner ranks "
        f"daily against that day's universe while this replay ranks every {s.step} "
        f"trading day{'s' if s.step > 1 else ''}, and a stock that is already in a trade still "
        "takes a shortlist slot, as it does live.",
        "Headline per-pattern stats pick the best pattern among ALL patterns. The live scanner "
        "skips disabled patterns first, so an enabled pattern gets more (and different) trades "
        "live than its headline row shows - see 'live configuration' (in-sample selection, "
        "informational only).",
        f"Trades whose {s.max_hold}-day holding window would run past a stock's last bar are "
        "left out whatever their outcome, so the last entries are about "
        f"{s.max_hold} trading days before the data ends (the last exits are near the end).",
        f"The results assume the {s.max_hold}-day time exit is followed: sell at the close of "
        f"the {s.max_hold}th trading day after the fill if neither the stop nor the target was "
        "hit (each setup on the dashboard shows the limit and the price monitor reminds you on "
        "that day). Holding with stop/target only changes the numbers materially.",
        "R is measured from the actual fill, but the dashboard sizes positions from the entry "
        "level; for triggered setups (bought at the next open) the real risk averages about "
        "1.27x the planned risk (95th percentile about 1.7x). Each trade also carries "
        "r_plan_net, the result per planned risk.",
        f"Fills more than {s.max_gap_above_entry_pct:g}% above the entry level are skipped. The "
        "dashboard and the monitor's entry alert show this buy-up-to price, but a plain "
        "overnight buy-stop does not enforce it. The entry level is assumed unchanged while an order waits, and late market buys "
        "after the monitor's delayed hourly alerts are not modelled.",
        "At small capital a few very high-priced stocks get 0 suggested shares on the dashboard "
        "(e.g. NVR, AZO, MRF.NS at EUR 10k), but the backtest still trades them.",
        "Grades use the no-fundamentals formula with no backtest nudge (grade_proxy). On 2,567 "
        "live shortlisted setups (Aug-Sep 2026) the A/not-A call agreed with the live grade "
        "about 93% of the time, but only about 50-67% for live composites between 65 and 75; "
        "the proxy leans toward A because it has no fundamentals term, so the grade-A figures "
        "are approximate. The monitor's top-10 and your own choice are not modelled.",
        "The pattern set, the signal grid and window, the technical-score floor 40, the "
        "risk:reward minimum 1.8, the costs and the verdict thresholds were all chosen after "
        "looking at earlier results on the same history (backtest_pre_darvas.json, "
        "backtest_2y.json, the pre_costs backups, and darvas_box added later). Even the "
        f"adjusted ranges ({_n_tests(tuning)} pattern x market tests) are therefore optimistic, "
        "entries after 2024-07 are not a holdout, and only paper or forward trades after the "
        "2026-09-25 freeze count as clean out-of-sample evidence.",
        "Verdicts use the whole decade. That is fine for choosing today's patterns and is not "
        "lookahead for the live scanner, but it makes in-sample replays (the 'enabled patterns "
        "only' profile) unreliable. Per-pattern results were unstable between 2017-21 and "
        "2022-26 (half-to-half correlation about 0.03 on the old stream, about 0.00 on the v2 "
        "smoke run) - read by_year and positive_years before trusting a verdict.",
        "The price history starts in July 2016; the first year is used only to warm up the "
        "200-day average, the 52-week high and the market regime (which needs 210 index bars), "
        "so signals start around July 2017 and no signal is judged with an 'unknown' regime. "
        "Stocks that listed later are scanned from their 120th trading day, as the live "
        "scanner does.",
        "The free price data contains unadjusted splits/bonuses, demergers, bad listing joins, "
        "one-day spikes and a stale-adjusted 2016 head for some stocks. They are repaired for "
        "the backtest only (listed in method.data_quality): exact split ratios with a volume "
        "surge and a few known corporate actions are back-adjusted, the history before a bad "
        "join, a flat zero-volume run or a stale head is dropped, and one-day spikes that "
        "fully revert are removed. The live scanner still sees the raw data.",
        "Flat zero-volume filler bars (exchange holidays with no index bar, e.g. IN 2026-01-15, "
        "05-01, 05-28, 06-26, 09-14 and DE 2017-06-05, 10-03, 10-31, 2018-05-21, an IN data gap "
        "on 2025-03-18, and pre-listing placeholder bars) are treated as ordinary days; "
        "2026-09-22 is missing for most US stocks and all three benchmarks; the newest IN bar "
        "is intraday (live India scans also see a partial NSE bar). Measured effect under "
        "0.001R per trade.",
    ]
    return out


def _note(results: dict, periods: dict) -> str:
    parts = []
    for m, block in results.items():
        counts: dict[str, int] = defaultdict(int)
        for st in ((block or {}).get("per_pattern") or {}).values():
            if isinstance(st, dict):
                counts[st.get("verdict")] += 1
        found = ", ".join(f"{counts[v]} {VERDICT_LABELS[v]}" for v in VERDICT_LABELS if counts[v])
        parts.append(f"{m.upper()}: {found or 'no trades'}")
    spans = [p for p in periods.values() if p and p.get("first_entry")]
    span = ""
    if spans:
        span = (f" {min(p['first_entry'] for p in spans)[:4]}-"
                f"{max(p['last_exit'] for p in spans)[:4]}")
    return (f"Replay of what the live scanner would have shown{span}, after fees and "
            "slippage, compared with holding the market over the same days. "
            + "; ".join(parts) + ".")


# --------------------------------------------------------------------------- driver

def _iter_results(jobs: list[tuple], pool: ProcessPoolExecutor | None):
    if pool is None:
        return map(simulate_job, jobs)
    return pool.map(simulate_job, jobs, chunksize=4)


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 3) - 2)


def _market_settings(s: Settings) -> dict:
    """Per-market record of what makes two runs' results comparable."""
    return {**_settings_dict(s), "random_entries_per_trade": s.random_entries_per_trade}


# per-market cost settings recorded in method (carried markets keep their own values)
_COST_KEYS = ("costs_pct_per_side", "stop_slippage_pct", "entry_slippage_pct",
              "index_dividend_yield_pct")


def _market_fingerprint(method: dict, market: str) -> dict:
    """What a results file says `market` was computed with: its own entry in
    method.settings_by_market (else, older files, the file-wide settings) and
    its own costs, slippage and dividend yield. A key missing from an older file
    compares as None, so such a market always needs an explicit choice."""
    m = method or {}
    sbm = (m.get("settings_by_market") or {}).get(market)
    if isinstance(sbm, dict):
        settings = dict(sbm)
    else:
        settings = {**(m.get("settings") or {}),
                    "random_entries_per_trade": m.get("random_entries_per_trade")}
    fp: dict = {"settings": settings}
    for k in _COST_KEYS:
        v = m.get(k)
        fp[k] = v.get(market) if isinstance(v, dict) else v
    return fp


def _current_fingerprint(s: Settings, market: str) -> dict:
    return {"settings": _market_settings(s),
            "costs_pct_per_side": s.cost_pct(market),
            "stop_slippage_pct": s.stop_slip(market),
            "entry_slippage_pct": s.entry_slip(market),
            "index_dividend_yield_pct": s.div_yield_pct(market)}


def _differing_markets(prev_method: dict, s: Settings, candidates: list[str]) -> list[str]:
    """Carry candidates whose previous results were computed with other settings
    than this run: compared with each market's OWN recorded settings, so a market
    kept earlier with --keep-others is caught on every later partial run."""
    return [m for m in candidates
            if _market_fingerprint(prev_method, m) != _current_fingerprint(s, m)]


def _reuse_baselines(out: list[dict], known: dict, panel: "MarketPanel", bench: BenchContext,
                    s: Settings, market: str) -> None:
    """Copy the baselines of identical, already measured trades; measure the rest."""
    fresh = []
    for t in out:
        old = known.get(_trade_identity(t))
        if old is not None:
            for k in _BASELINE_KEYS:
                t[k] = old[k]
        else:
            fresh.append(t)
    add_baselines(fresh, panel, bench, s, market)


def _replay_with_disabled(tds: list[TickerData], cal: list[str], bench: BenchContext,
                          s: Settings, market: str, disabled, p1: dict, headline: list[dict],
                          panel: "MarketPanel") -> tuple[list[dict], dict]:
    """Re-run phases 2 + 3 with `disabled` (a set, or {year: set}) removed before
    the pick; trades identical to headline ones reuse their baselines."""
    counters = new_counters()
    _merge_counters(counters, p1)
    out = run_selection(tds, cal, bench, s, market, disabled, counters)
    _reuse_baselines(out, {_trade_identity(t): t for t in headline}, panel, bench, s, market)
    return out, counters


def _grade_decisions(decisions: list[list], mode: str) -> list[list]:
    """The grade profiles' view of phase 2's decisions: a candidate the profile
    would not trade becomes the untraded decision 'not_grade_a', so its order
    never occupies the stock or blocks its level and a later A-grade setup on
    the stock is still taken. Ranks and candidate counts stay as they were: the
    live scanner ranks every candidate before anyone looks at the grade.
    mode 'A': every non-A candidate; 'A_in_caution': non-A candidates signalled
    while the index regime was 'caution' only."""
    if mode not in GRADE_MODES:
        raise ValueError(f"unknown grade mode {mode!r}")
    out: list[list] = []
    for dec in decisions:
        row: list = []
        for d in dec:
            if isinstance(d, list):
                best, rec = d[0], d[1]
                g = grade_proxy(rec.tscore, best.confidence, best.risk_reward, rec.conflict)[2]
                if g != "A" and (mode == "A" or rec.regime == "caution"):
                    row.append("not_grade_a")
                    continue
            row.append(d)
        out.append(row)
    return out


def _replay_grade(tds: list[TickerData], cal: list[str], bench: BenchContext, s: Settings,
                  market: str, disabled, mode: str, p1: dict, known: dict,
                  panel: "MarketPanel") -> tuple[list[dict], dict]:
    """Phases 2 + 3 for a grade profile: the pick (with `disabled` removed before
    it), then only the setups the profile trades are ordered (_grade_decisions).
    Trades identical to already measured ones (`known`, by _trade_identity)
    reuse their baselines."""
    counters = new_counters()
    _merge_counters(counters, p1)
    decisions = _grade_decisions(pick_and_rank(tds, s, disabled), mode)
    out: list[dict] = []
    for td, dec in zip(tds, decisions):
        out.extend(trade_ticker(market, td, dec, cal, bench, s, counters))
    out.sort(key=_trade_sort_key)
    _reuse_baselines(out, known, panel, bench, s, market)
    return out, counters


def run_backtest(markets: list[str], progress=print, *, raw: bool = False,
                 limit: int | None = None, workers: int | None = None, portfolio: bool = True,
                 cfg: dict | None = None, data_dir: Path | None = None,
                 db_path: Path | None = None, universe_dir: Path | None = None,
                 legacy_grid: bool = False, entry_mode: str | None = None,
                 keep_others: bool = False, replace_others: bool = False) -> dict:
    """Run the v2 backtest and write its files. Returns the summary payload.

    limit: simulate only the first N tickers per market (smoke run) - writes
    *_smoke.json files and skips the portfolio step.
    legacy_grid / entry_mode: labelled variants (written to tagged files, never
    over the headline results).
    keep_others / replace_others: what to do with markets in the previous
    results that this run does not re-run when they cannot simply be kept
    (other settings, or an old v1 / unreadable previous file).
    data_dir / db_path / universe_dir / cfg are injectable for tests.
    """
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be >= 1")
    if workers is not None and int(workers) <= 0:
        raise ValueError("workers must be >= 1")
    started = time.time()
    cfg = cfg if cfg is not None else load_config()
    s = Settings.from_config(cfg)
    tags: list[str] = []
    if legacy_grid:
        s = dataclasses.replace(s, step=5, breakout_wait=15, seen_on_fill_only=False)
        tags.append("legacy_grid")
    if entry_mode is not None and entry_mode != s.entry_mode:
        s = dataclasses.replace(s, entry_mode=entry_mode)
        tags.append(entry_mode)
    tuning = _deep_merge(PATTERN_TUNING_DEFAULTS, cfg.get("pattern_tuning") or {})
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    db_path = Path(db_path) if db_path else ROOT / "data" / "cache" / "prices.sqlite3"
    universe_dir = Path(universe_dir) if universe_dir else ROOT / "universe"
    extra_artifacts = load_price_artifacts(data_dir)
    if extra_artifacts:
        s = dataclasses.replace(s, price_artifacts=tuple(s.price_artifacts) + tuple(extra_artifacts))
    workers = int(workers) if workers else _default_workers()
    smoke = limit is not None
    variant = ("_" + "_".join(tags)) if tags else ""
    suffix = variant + ("_smoke" if smoke else "")
    headline = not smoke and not variant
    summary_path = data_dir / (f"backtest{suffix}.json" if not headline else "backtest_latest.json")
    trades_path = data_dir / f"backtest_trades{suffix}.json"
    live_path = data_dir / f"backtest_trades_live{suffix}.json"
    wf_path = data_dir / f"backtest_trades_walkforward{suffix}.json"
    grade_a_path = data_dir / f"backtest_trades_grade_a{suffix}.json"
    a_caution_path = data_dir / f"backtest_trades_a_in_caution{suffix}.json"
    raw_path = data_dir / f"backtest_trades_raw{suffix}.json"
    markets = [m for m in MARKET_ORDER if m in markets] + [m for m in markets if m not in MARKET_ORDER]

    # carry-over: decide BEFORE the long run whether other markets' results are kept
    prev = None
    carry_candidates: list[str] = []
    kept_different: list[str] = []
    if headline and summary_path.exists():
        unreadable = False
        try:
            prev = json.loads(summary_path.read_text())
        except Exception:
            prev, unreadable = None, True
        if not (isinstance(prev, dict) and int(prev.get("version") or 1) >= 2
                and not prev.get("smoke")):
            # an old v1 (or unreadable / smoke) file cannot be carried: never drop
            # its other markets silently
            old = prev.get("markets") if isinstance(prev, dict) else None
            if isinstance(old, dict):
                dropped = [m for m in old if m not in markets]
            else:
                dropped = [m for m in MARKET_ORDER if m not in markets]
            what = ("unreadable" if unreadable or not isinstance(prev, dict)
                    else "from a smoke run" if prev.get("smoke") else "in the old v1 format")
            prev = None
            if dropped:
                names = ", ".join(m.upper() for m in dropped)
                if not replace_others:
                    raise ValueError(
                        f"the previous results for {names} are {what} and cannot be kept; "
                        "re-run all markets, or pass --replace to drop them")
                progress(f"WARN dropping the previous results for {names} ({what}; --replace)")
        if prev is not None:
            carry_candidates = [m for m in (prev.get("markets") or {}) if m not in markets]
            prev_kept = set(((prev.get("method") or {}).get("kept_with_different_settings")) or [])
            if carry_candidates and not replace_others:
                differ = _differing_markets(prev.get("method") or {}, s, carry_candidates)
                settled = [m for m in carry_candidates if m in prev_kept and m not in differ]
                if settled:
                    progress("NOTE " + ", ".join(m.upper() for m in settled) + " had been kept "
                             "with different settings (--keep-others); its own settings match "
                             "this run, so it is kept and no longer flagged")
                if differ:
                    if not keep_others:
                        raise ValueError(
                            "the previous results for " + ", ".join(m.upper() for m in differ)
                            + " were computed with different backtest settings (or kept "
                            "earlier with --keep-others); re-run all markets, or pass "
                            "--keep-others (keep them anyway, labelled) or --replace (drop them)")
                    progress("WARN keeping " + ", ".join(m.upper() for m in differ)
                             + " from a backtest with DIFFERENT settings (--keep-others)")
                    kept_different = differ
            if replace_others:
                carry_candidates = []

    _reset_worker_caches()
    store = PriceStore(db_path)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    results: dict[str, dict] = {}
    periods: dict[str, dict] = {}
    universe_info: dict[str, dict] = {}
    data_quality: dict[str, list] = {}
    every_trade: list[dict] = []
    every_live: list[dict] = []
    every_wf: list[dict] = []
    every_ga: list[dict] = []
    every_gc: list[dict] = []
    every_raw: list[dict] = []
    try:
        for market in markets:
            uni_path = universe_dir / f"{market}.csv"
            if not uni_path.exists():
                progress(f"[{market}] universe/{market}.csv missing - skipped")
                continue
            universe = pd.read_csv(uni_path)["ticker"].astype(str).tolist()
            run_list = universe[:limit] if limit else universe
            bench = BenchContext.from_df(store.load(BENCHMARKS[market]))
            cal = signal_calendar(bench, s)
            stale = stale_before_map(store, universe)
            progress(f"[{market}] phase 1: scanning {len(run_list)} tickers on {len(cal)} signal "
                     f"days ({workers} worker{'s' if workers > 1 else ''})...")
            jobs = [(t, market, str(db_path), s, raw, cal, stale.get(t), k)
                    for k, t in enumerate(run_list)]
            p1 = new_counters()
            tds: list[TickerData] = []
            raw_trades: list[dict] = []
            done = errors = 0
            for res in _iter_results(jobs, pool):
                if res["error"]:
                    errors += 1
                    progress(f"[{market}] WARN {res['ticker']}: {res['error']}")
                if not res["simulated"]:
                    continue
                done += 1
                tds.append(res["td"])
                raw_trades.extend(res["raw_trades"])
                _merge_counters(p1, res["counters"])
                if done % 100 == 0:
                    progress(f"[{market}] {done} tickers scanned")
            raw_trades.sort(key=_trade_sort_key)

            # phases 2 + 3: headline (every pattern measured)
            counters = new_counters()
            _merge_counters(counters, p1)
            info = _new_pattern_info()
            trades = run_selection(tds, cal, bench, s, market, frozenset(), counters, info)

            progress(f"[{market}] {len(trades)} trades from {done} tickers; computing baselines "
                     "(index, equal-weight market, random entries)...")
            long, prep = sanitize_long(load_long_prices(store, universe), s.price_artifacts, stale)
            panel = MarketPanel(long, min_dollar_vol=MIN_DOLLAR_VOL[market],
                                suspect=prep.get("suspect"))
            del long
            missing = add_baselines(trades, panel, bench, s, market)

            block = summarize(trades, tuning, info, s.min_risk_reward)
            block["filters"] = {
                "signals_evaluated": counters["signals_evaluated"],
                "removed": counters["removed"],
                "no_pattern": counters["no_pattern"],
                "detector_exceptions": counters["detector_exceptions"],
                "censored_at_end": counters["censored_at_end"],
                "orders_not_filled_today": counters["orders_not_filled_today"],
                "warmup": counters["warmup"],
                "baseline_missing": missing,
                "trades": len(trades),
                "tickers_simulated": done,
                "tickers_failed": errors,
                "price_artifacts": prep["counts"],
                "random_pool_fallback": panel.pool_counts["random_pool_fallback"],
                "random_pool_unmatched": panel.pool_counts["random_pool_unmatched"],
            }
            block["pattern_detection"] = {p: dict(v) for p, v in sorted(info.items())}
            data_quality[market] = prep["events"]

            # live configuration: today's disabled patterns dropped BEFORE the pick
            dis = disabled_patterns(block["per_pattern"], tuning, market)
            if dis:
                progress(f"[{market}] live-configuration replay without "
                         f"{', '.join(sorted(dis))}...")
                live, live_counters = _replay_with_disabled(tds, cal, bench, s, market,
                                                            frozenset(dis), p1, trades, panel)
                lsum = summarize(live, tuning, None, s.min_risk_reward)
                block["live_config"] = {
                    "disabled": sorted(dis),
                    "rule": disable_rule_text(tuning),
                    "label": "in-sample selection",
                    "note": ("Replay with the patterns switched off at backtest time removed "
                             "before the best hit is picked, as the live scanner does. Verdicts "
                             "here are informational only: they are not fed back into the "
                             "tuning."),
                    "per_pattern": {p: st for p, st in lsum["per_pattern"].items() if p not in dis},
                    "total": lsum["total"],
                    "trades": len(live),
                    "censored_at_end": live_counters["censored_at_end"],
                }
            else:
                live = [dict(t) for t in trades]
                block["live_config"] = {"disabled": [], "rule": disable_rule_text(tuning),
                                        "label": "in-sample selection",
                                        "note": "No pattern is switched off: identical to the headline.",
                                        "trades": len(live), "same_as_headline": True}

            # grade profiles: the same pick, but only the setups the profile trades are
            # ordered, so a B/C trade never blocks a later A-grade setup on the stock.
            # Grade-A only is what you trade: the live configuration's switched-off
            # patterns are removed before the pick; A-in-caution is 'as designed'
            # (every pattern).
            progress(f"[{market}] grade replays (grade-A only; A-grades only in caution)...")
            known = {_trade_identity(t): t for t in trades}
            for t in live:
                known.setdefault(_trade_identity(t), t)
            ga, ga_counters = _replay_grade(tds, cal, bench, s, market, frozenset(dis), "A",
                                            p1, known, panel)
            gc, gc_counters = _replay_grade(tds, cal, bench, s, market, frozenset(),
                                            "A_in_caution", p1, known, panel)
            block["grade_replay"] = {
                "label": "grade replay",
                "note": ("Replays for the portfolio's grade profiles: after the pick and the "
                         "shortlist, setups that are not grade A (for A-in-caution: not grade "
                         "A while the index is in caution) are never ordered, so they do not "
                         "block a later A-grade setup on the same stock. Grade-A only removes "
                         "the patterns switched off at backtest time before the pick; "
                         "A-in-caution keeps every pattern."),
                "grade_a": {"disabled": sorted(dis), "trades": len(ga),
                            "not_grade_a": ga_counters["removed"]["not_grade_a"],
                            "censored_at_end": ga_counters["censored_at_end"]},
                "a_in_caution": {"disabled": [], "trades": len(gc),
                                 "not_grade_a": gc_counters["removed"]["not_grade_a"],
                                 "censored_at_end": gc_counters["censored_at_end"]},
            }

            # walk-forward: each signal year's disabled list judged only on trades
            # that closed before that year, removed before the pick (like live)
            from .portfolio_sim import walk_forward_disabled

            span = [cal[0][:4], cal[-1][:4]] if cal else None
            wf_by_year: dict = {}
            if span is not None:
                wf_all, _ = walk_forward_disabled(trades, tuning, span=tuple(span))
                wf_by_year = (wf_all or {}).get(market) or {}
            if any(wf_by_year.values()):
                progress(f"[{market}] walk-forward replay (each year's switched-off patterns "
                         "judged on earlier trades only)...")
                wf_trades, wf_counters = _replay_with_disabled(
                    tds, cal, bench, s, market,
                    {y: frozenset(v) for y, v in wf_by_year.items()}, p1, trades, panel)
                censored_wf = wf_counters["censored_at_end"]
            else:
                wf_trades, censored_wf = [dict(t) for t in trades], counters["censored_at_end"]
            wsum = summarize(wf_trades, tuning, None, s.min_risk_reward)
            block["walk_forward"] = {
                "span": span,
                "disabled_by_year": wf_by_year,
                "rule": disable_rule_text(tuning),
                "label": "walk-forward selection",
                "note": ("Replay with each signal year's switched-off patterns - judged only on "
                         "trades that closed before that year started - removed before the best "
                         "hit is picked, as the live scanner would have done. Feeds the "
                         "portfolio's walk-forward profile."),
                "total": wsum["total"],
                "trades": len(wf_trades),
                "censored_at_end": censored_wf,
                "same_as_headline": not any(wf_by_year.values()),
            }
            del panel

            if raw:
                block["raw_total"], block["raw_per_pattern"] = raw_summary(raw_trades)
            results[market] = block
            closed = [t for t in trades if t["outcome"] != "open_at_end"]
            periods[market] = {
                "first_entry": min((t["entry_date"] for t in closed), default=None),
                "last_exit": max((t["exit_date"] for t in closed), default=None),
            }
            universe_info[market] = {"tickers": len(universe), "run": len(run_list),
                                     "simulated": done,
                                     "too_short_or_missing": len(run_list) - done}
            every_trade.extend(trades)
            every_live.extend(live)
            every_wf.extend(wf_trades)
            every_ga.extend(ga)
            every_gc.extend(gc)
            every_raw.extend(raw_trades)
            _progress_market(progress, market, block, raw)
            del tds
    finally:
        if pool is not None:
            pool.shutdown()
        store.close()
        _reset_worker_caches()

    carried: dict[str, str] = {}
    prev_method = (prev or {}).get("method") or {}
    prev_carried = prev_method.get("carried_over") or {}
    if prev is not None and carry_candidates:
        missing_m = [m for m in carry_candidates if m not in results]
        prev_trades = None
        prev_streams: dict[str, list | None] = {}
        if missing_m:
            try:
                prev_trades = json.loads(trades_path.read_text())
            except Exception:
                progress("WARN previous backtest_trades.json unreadable - markets not re-run are dropped")
            for name, path in (("live", live_path), ("wf", wf_path), ("ga", grade_a_path),
                               ("gc", a_caution_path)):
                try:
                    v = json.loads(path.read_text())
                except Exception:
                    v = None
                prev_streams[name] = v if isinstance(v, list) else None
        prev_live, prev_wf = prev_streams.get("live"), prev_streams.get("wf")
        prev_ga, prev_gc = prev_streams.get("ga"), prev_streams.get("gc")
        if missing_m and isinstance(prev_trades, list):
            for m in missing_m:
                block = dict(prev["markets"][m])
                results[m] = block
                periods[m] = (prev_method.get("period") or {}).get(m)
                universe_info[m] = (prev_method.get("universe") or {}).get(m)
                data_quality[m] = ((prev_method.get("data_quality") or {}).get(m)) or []
                # the date of the run that actually computed this market
                carried[m] = prev_carried.get(m) or prev.get("generated", "")
                old_trades = [t for t in prev_trades if t.get("market") == m]
                every_trade.extend(old_trades)
                lc = block.get("live_config") or {}
                if isinstance(prev_live, list) and not lc.get("unavailable"):
                    every_live.extend(t for t in prev_live if t.get("market") == m)
                elif lc.get("same_as_headline") or lc.get("disabled") == []:
                    every_live.extend(dict(t) for t in old_trades)   # nothing was switched off
                else:
                    block["live_config"] = {
                        "unavailable": True, "rule": lc.get("rule"),
                        "label": "in-sample selection",
                        "note": ("The live-configuration trades of the run this market was kept "
                                 "from could not be read, so the portfolio's enabled-only profile "
                                 "falls back to deleting the switched-off patterns' trades "
                                 "(approximate) until this market is re-run.")}
                    progress(f"WARN {m.upper()}: previous live-configuration trades unreadable - "
                             "its enabled-only portfolio figures fall back to the approximation")
                wfb = block.get("walk_forward")
                if isinstance(wfb, dict) and not wfb.get("unavailable") \
                        and isinstance(prev_wf, list):
                    every_wf.extend(t for t in prev_wf if t.get("market") == m)
                else:
                    block["walk_forward"] = {
                        "unavailable": True, "label": "walk-forward selection",
                        "note": ("No walk-forward replay for this market in the run it was kept "
                                 "from, so the portfolio's walk-forward profile deletes the "
                                 "switched-off patterns' trades (approximate) until this market "
                                 "is re-run.")}
                gr = block.get("grade_replay")
                if isinstance(gr, dict) and not gr.get("unavailable") \
                        and prev_ga is not None and prev_gc is not None:
                    every_ga.extend(t for t in prev_ga if t.get("market") == m)
                    every_gc.extend(t for t in prev_gc if t.get("market") == m)
                else:
                    block["grade_replay"] = {
                        "unavailable": True, "label": "grade replay",
                        "note": ("No grade replay for this market in the run it was kept from, "
                                 "so the portfolio's grade profiles delete the other grades' "
                                 "trades from the headline stream (approximate) until this "
                                 "market is re-run.")}
            if raw:
                try:
                    old_raw = json.loads(raw_path.read_text())
                    every_raw.extend(t for t in old_raw if t.get("market") in carried)
                except Exception:
                    pass
            progress("NOTE kept previous results for " + ", ".join(m.upper() for m in carried)
                     + " (not re-run; from backtest "
                     + ", ".join(sorted(set(str(v)[:10] for v in carried.values()))) + ")")
    order = [m for m in MARKET_ORDER if m in results] + [m for m in results if m not in MARKET_ORDER]
    results = {m: results[m] for m in order}
    every_trade.sort(key=_trade_sort_key)
    every_live.sort(key=_trade_sort_key)
    every_wf.sort(key=_trade_sort_key)
    every_ga.sort(key=_trade_sort_key)
    every_gc.sort(key=_trade_sort_key)
    every_raw.sort(key=_trade_sort_key)

    method = _method(s, list(results), periods, universe_info, raw, limit, tuning)
    for m in carried:   # carried markets keep the settings they were computed with
        for k in _COST_KEYS:
            pv = prev_method.get(k)
            if isinstance(pv, dict) and m in pv and isinstance(method.get(k), dict):
                method[k][m] = pv[m]
    method["settings_by_market"] = {
        m: (_market_fingerprint(prev_method, m)["settings"] if m in carried else _market_settings(s))
        for m in results}
    # flagged while a kept market's own settings differ from this run's
    method["kept_with_different_settings"] = [m for m in results
                                              if m in carried and m in kept_different]
    method["carried_over"] = carried
    method["data_quality"] = data_quality
    method["variant"] = tags or None
    payload = {
        "version": VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "available": True,
        "smoke": smoke,
        "method": method,
        "caveats": _caveats(s, tuning),
        "note": _note(results, periods),
        "markets": results,
    }
    write_json_atomic(summary_path, payload, indent=1)
    write_json_atomic(trades_path, every_trade)
    write_json_atomic(live_path, every_live)
    write_json_atomic(wf_path, every_wf)
    write_json_atomic(grade_a_path, every_ga)
    write_json_atomic(a_caution_path, every_gc)
    progress(f"summary -> {summary_path}; {len(every_trade)} trades -> {trades_path}; "
             f"live-configuration trades ({len(every_live)}) -> {live_path}; walk-forward "
             f"trades ({len(every_wf)}) -> {wf_path}; grade replays ({len(every_ga)} grade-A, "
             f"{len(every_gc)} A-in-caution) -> {grade_a_path.name}, {a_caution_path.name}")
    if raw:
        write_json_atomic(raw_path, every_raw)
        progress(f"raw-mode trades ({len(every_raw)}) -> {raw_path}")
    progress(f"backtest finished in {time.time() - started:.0f}s")

    if portfolio and headline:
        try:
            from .portfolio_sim import REFERENCE_PRE_COSTS_PATH, run_portfolio_report

            run_portfolio_report(progress=progress, trades_path=trades_path,
                                 backtest_path=summary_path,
                                 out_path=data_dir / "portfolio_sim.json",
                                 prices_path=db_path, universe_dir=universe_dir,
                                 config=cfg, live_trades_path=live_path,
                                 walk_forward_trades_path=wf_path,
                                 grade_a_trades_path=grade_a_path,
                                 a_in_caution_trades_path=a_caution_path,
                                 reference_path=data_dir / REFERENCE_PRE_COSTS_PATH.name)
        except Exception as exc:
            progress(f"WARN portfolio step failed ({exc.__class__.__name__}: {exc}); "
                     "the backtest results were saved - run: python -m atlas2 portfolio")
    return payload


def _fmt_num(x, unit: str) -> str:
    return "n/a" if x is None else f"{_sg(x, 2)}{unit}"


def _progress_market(progress, market: str, block: dict, raw: bool) -> None:
    tot = block["total"]
    f = block["filters"]
    removed = ", ".join(f"{k} {v}" for k, v in f["removed"].items() if v) or "none"
    progress(f"[{market}] {tot['trades']} closed trades | wins after fees {tot['win_rate']}% | "
             f"net {_fmt_num(tot['net_avg_r'], 'R')} per trade | vs index "
             f"{_fmt_num(tot['excess_vs_index_pct'], '%')} | vs market "
             f"{_fmt_num(tot['excess_vs_universe_pct'], '%')} | removed: {removed} | "
             f"one-day orders not filled {f['orders_not_filled_today']} | censored at end "
             f"{f['censored_at_end']}")
    if f.get("price_artifacts"):
        progress(f"[{market}] price data repaired: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(f["price_artifacts"].items())))
    for pat, st in block["per_pattern"].items():
        progress(f"[{market}]   {pat:<24} {st['trades']:>5} trades  "
                 f"net {_fmt_num(st['net_avg_r'], 'R'):>7}  "
                 f"vs market {_fmt_num(st['excess_vs_universe_pct'], '%'):>7}  -> {st['verdict']}")
    lc = block.get("live_config") or {}
    if lc.get("disabled"):
        lt = lc.get("total") or {}
        progress(f"[{market}] live configuration (without {', '.join(lc['disabled'])}): "
                 f"{lt.get('trades')} trades, net {_fmt_num(lt.get('net_avg_r'), 'R')}, "
                 f"vs market {_fmt_num(lt.get('excess_vs_universe_pct'), '%')} (in-sample)")
    wf = block.get("walk_forward") or {}
    if wf.get("disabled_by_year") and not wf.get("same_as_headline"):
        wt = wf.get("total") or {}
        progress(f"[{market}] walk-forward selection (each year's switched-off patterns judged on "
                 f"earlier trades only): {wt.get('trades')} trades, net "
                 f"{_fmt_num(wt.get('net_avg_r'), 'R')}, vs market "
                 f"{_fmt_num(wt.get('excess_vs_universe_pct'), '%')}")
    if raw and block.get("raw_total"):
        rt = block["raw_total"]
        progress(f"[{market}] raw mode (old selection, no filters): {rt['trades']} trades, "
                 f"net {_fmt_num(rt['net_avg_r'], 'R')}, wins {rt['win_rate']}%")
