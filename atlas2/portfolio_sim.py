"""Portfolio-level simulation over the backtest trade stream, plus a
buy-and-hold benchmark over the same dates.

Replays the decade as one account: signals arrive chronologically, at most
`max_pos` positions are open at once, each trade risks `risk_pct` of CURRENT
(realized) equity (so results compound) and is sized the way the dashboard
sizes it - shares = risk / (entry level - stop), capped at max_position_pct of
equity per position and by the free cash - and when more signals arrive than
slots exist the best-ranked setup wins (the grade composite when the stream
has it, else pattern confidence), approximating a selective human trader.
Which of several equally ranked setups gets the last free slot is pure luck,
so every profile is replayed over 101 slot orderings and the headline is the
median path.

Costs: every trade's result is taken AFTER fees and slippage - the backtest's
per-trade `r_net` when the stream has it (v2 streams), else `r - cost_r` for
old streams that carry only the gross R multiple.

Benchmark: for the same window (first entry .. last exit of the stream) the
market index (price index plus an estimated dividend yield for ^GSPC / ^NSEI)
and an equal-weight, daily-rebalanced basket of the market's universe (on the
backtest's repaired prices), blended
across markets as monthly-rebalanced portfolios, so each profile can be read as
"system vs simply holding the market".

Output: the portfolio block is written to data/portfolio_sim.json and, when the
backtest results file is v2, embedded under backtest_latest.json["portfolio"]
for the dashboard.
Run: python -m atlas2 portfolio
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import zlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRADES_PATH = ROOT / "data" / "backtest_trades.json"
DEFAULT_BACKTEST_PATH = ROOT / "data" / "backtest_latest.json"
DEFAULT_OUT_PATH = ROOT / "data" / "portfolio_sim.json"
DEFAULT_PRICES_PATH = ROOT / "data" / "cache" / "prices.sqlite3"
DEFAULT_UNIVERSE_DIR = ROOT / "universe"
REFERENCE_PRE_COSTS_PATH = ROOT / "data" / "backtest_trades_pre_costs_20260925.json"

START_CAPITAL = 25000.0
LEGACY_COST_R = 0.05     # friction per trade (R units) for old streams without r_net
DAILY_RET_CLIP = 0.50    # cap on a single stock's daily return in the universe basket
TIE_RESHUFFLES = 100     # seeds 0..100 = 101 slot orderings (odd: the median is a real path)
DEFAULT_MAX_POSITION_PCT = 25.0
MIN_FREE_CASH_FRAC = 0.02   # below this much free cash a new trade is skipped
MARKET_ORDER = ("us", "de", "in")
WF_REQUIRED_KEYS = ("r", "r_net", "ret_pct", "ret_net_pct", "hold_bars", "index_ret_pct",
                    "universe_ret_pct", "excess_vs_index_pct", "excess_vs_universe_pct",
                    "rand_r_net", "entry_date", "exit_date", "pattern", "market", "outcome")

# (key, label, simulate_portfolio kwargs, regime filter, pattern mode, grade filter)
# pattern mode: None (all patterns) / 'walk_forward' / 'in_sample'
PROFILES = [
    ("no_regime", None, {"risk_pct": 1.0}, False, None, None),   # label set at run time
    ("as_designed", "SYSTEM AS DESIGNED (1% risk + regime filter)",
     {"risk_pct": 1.0}, True, None, None),
    ("as_designed_walk_forward",
     "System as designed, auto-disable re-judged each year on earlier trades only (walk-forward)",
     {"risk_pct": 1.0}, True, "walk_forward", None),
    ("as_designed_enabled_only",
     "System as designed, enabled patterns only - in-sample (patterns chosen with hindsight "
     "on the same data)",
     {"risk_pct": 1.0}, True, "in_sample", None),
    ("grade_a_only", "Grade-A proxy only (what you and the monitor actually trade)",
     {"risk_pct": 1.0}, True, None, "A"),
    ("as_designed_a_in_caution", "As designed + A-grades only in caution",
     {"risk_pct": 1.0}, True, None, "A_in_caution"),
    ("balanced", "Balanced (1.5% risk + regime filter)", {"risk_pct": 1.5}, True, None, None),
    ("aggressive", "Aggressive (2% risk + regime filter; position cap binds on most trades)",
     {"risk_pct": 2.0}, True, None, None),
    ("very_aggressive",
     "Very aggressive (3% risk + regime filter; position cap binds on most trades)",
     {"risk_pct": 3.0}, True, None, None),
    ("notional_r", "Notional R account (uncapped, 1% risk + regime) - the old replay, for comparison",
     {"risk_pct": 1.0, "sizing": "r", "cash_limit": False}, True, None, None),
]


# --------------------------------------------------------------------------- helpers

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _round(x, nd: int = 1):
    return round(float(x), nd) if _finite(x) else None


def _json_safe(obj):
    """Recursively turn NaN/inf into None and numpy scalars into Python numbers
    (the dashboard parses these files with JSON.parse, which rejects NaN)."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    kind = getattr(getattr(obj, "dtype", None), "kind", "")
    if kind == "b":
        return bool(obj)
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    if not math.isfinite(f):
        return None
    return int(obj) if kind in ("i", "u") else f


def _atomic_write_json(path: Path, obj, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, indent=indent))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _version(payload) -> int:
    try:
        return int(payload.get("version") or 1)
    except (TypeError, ValueError, AttributeError):
        return 1


def _connect_ro(prices_path: Path | None) -> sqlite3.Connection | None:
    """Read-only connection to the price cache (never creates or writes it)."""
    p = Path(prices_path or DEFAULT_PRICES_PATH)
    if not p.exists():
        return None
    return sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)


def _load_bars(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Same query and shape as data.PriceStore.load."""
    return pd.read_sql_query(
        "SELECT date, open, high, low, close, volume FROM bars "
        "WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,), index_col="date", parse_dates=["date"],
    )


def _load_closes(conn: sqlite3.Connection, tickers: list[str], lo: str, hi: str) -> pd.DataFrame:
    """Long frame (ticker, date 'YYYY-MM-DD', close) for lo <= date <= hi."""
    frames = []
    for i in range(0, len(tickers), 400):
        chunk = tickers[i:i + 400]
        q = ("SELECT ticker, date, close FROM bars WHERE ticker IN (%s) "
             "AND date >= ? AND date <= ? AND close > 0" % ",".join("?" * len(chunk)))
        frames.append(pd.read_sql_query(q, conn, params=[*chunk, lo, hi]))
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].astype(str).str[:10]
    return df


def _years_between(a: str, b: str) -> float:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days / 365.25


# --------------------------------------------------------------------------- regime

REGIME_MULT = {"healthy": 1.0, "caution": 0.5, "downtrend": 0.0, "unknown": 1.0}


def regime_multipliers(prices_path: Path | None = None) -> dict:
    """Per market, per date: 1.0 healthy / 0.5 caution / 0.0 downtrend / 1.0
    unknown (fewer than 210 index bars) - the same rules and warm-up the live
    scanner (scan.market_regime) and the backtest (backtest.regime_statuses) use.
    Empty dict when the price cache is missing (then no regime filter applies)."""
    from .backtest import regime_statuses
    from .data import BENCHMARKS

    conn = _connect_ro(prices_path)
    if conn is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    try:
        for m, bench in BENCHMARKS.items():
            df = _load_bars(conn, bench)
            if len(df) < 210:
                continue  # scan.market_regime reports 'unknown' -> multiplier 1.0
            statuses = regime_statuses(df)
            out[m] = {str(idx)[:10]: REGIME_MULT.get(st, 1.0)
                      for idx, st in zip(df.index, statuses)}
    finally:
        conn.close()
    return out


def _mult_for(regime: dict, market: str, day: str, days_sorted: dict | None = None) -> float:
    """Regime multiplier for a trade entered on `day`: the index state at the
    close of the last session BEFORE `day` - what the evening scan showed
    before the entry (using `day`'s own close would peek at the future)."""
    table = regime.get(market) if regime else None
    if not table:
        return 1.0
    days = (days_sorted or {}).get(market) or sorted(table)
    k = bisect_left(days, day) - 1
    return table[days[k]] if k >= 0 else 1.0


# --------------------------------------------------------------------------- trades

def trade_r_net(t: dict, cost_r: float = LEGACY_COST_R) -> float | None:
    """Per-trade result after fees/slippage in R: the backtest's r_net when
    present (v2 streams), else gross r minus a flat cost_r (old streams)."""
    rn = t.get("r_net")
    if _finite(rn):
        return float(rn)
    r = t.get("r")
    if _finite(r):
        return float(r) - cost_r
    return None


def usable_trades(trades: list[dict]) -> list[dict]:
    """Closed trades with dates; 'open_at_end' (data ended mid-trade) is skipped
    (defensive: v2 streams made after the censoring fix contain none)."""
    return [t for t in trades
            if isinstance(t, dict) and t.get("outcome") != "open_at_end"
            and t.get("entry_date") and t.get("exit_date")]


def stream_window(trades: list[dict]) -> tuple[str, str] | None:
    ts = usable_trades(trades)
    if not ts:
        return None
    return min(t["entry_date"] for t in ts), max(t["exit_date"] for t in ts)


def signal_year(t: dict) -> str:
    return str(t.get("signal_date") or t.get("entry_date") or "")[:4]


def _tiebreak(t: dict, seed: int = 0) -> int:
    # stable across processes and runs (never Python hash(), which is salted)
    key = f"{t.get('ticker', '')}|{t.get('entry_date', '')}|{t.get('pattern', '')}"
    if seed:
        key += f"|{seed}"
    return zlib.crc32(key.encode())


def rank_field(trades: list[dict]) -> str:
    """'composite_proxy' when every usable trade has it (grade-aware streams),
    else 'conf' - chosen once per stream so the two scales are never mixed."""
    ts = usable_trades(trades)
    if ts and all(_finite(t.get("composite_proxy")) for t in ts):
        return "composite_proxy"
    return "conf"


def _entry_level(t: dict) -> tuple[float | None, str]:
    """Planned entry level for dashboard sizing, and where it came from."""
    stop = t.get("stop")
    e = t.get("entry_level")
    if _finite(e) and _finite(stop) and e > stop:
        return float(e), "entry_level"
    rr, target = t.get("rr"), t.get("target")
    if _finite(rr) and _finite(target) and _finite(stop) and rr > 0:
        e = (target + rr * stop) / (1 + rr)       # rr = (target - E) / (E - stop)
        if e > stop:
            return float(e), "rebuilt_from_rr"
    f = t.get("entry_price")
    if _finite(f) and _finite(stop) and f > stop:
        return float(f), "fill"
    return None, "legacy"


def _stop_frac(t: dict) -> float | None:
    """Stop distance as a fraction of the fill (for exposure estimates)."""
    f, stop = t.get("entry_price"), t.get("stop")
    if _finite(f) and _finite(stop) and f > stop > 0:
        return (f - stop) / f
    rp = t.get("risk_pct")
    if _finite(rp) and rp > 0:
        return rp / 100
    r, ret = t.get("r"), t.get("ret_pct")
    if _finite(r) and _finite(ret) and abs(r) > 0.05:
        x = ret / (100 * r)
        if x > 0:
            return x
    return None


def simulate_portfolio(trades: list[dict], start_capital: float = START_CAPITAL,
                       risk_pct: float = 1.0, max_pos: int = 6,
                       min_conf: float = 0.0, cost_r: float = LEGACY_COST_R,
                       regime: dict | None = None,
                       exclude: dict | None = None,
                       window: tuple[str, str] | None = None,
                       tie_seed: int = 0, *,
                       max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
                       cash_limit: bool = True, sizing: str = "dashboard",
                       grade_filter: str | None = None, rank_by: str | None = None,
                       record_positions: bool = False) -> dict:
    """Replay the trade stream as one account.

    cost_r: friction per trade in R, used ONLY for trades without 'r_net'.
    regime: optional {market: {date: multiplier}} - 0 skips, 0.5 halves risk.
    exclude: optional {market: iterable of pattern names} to leave out.
    window: (start, end) dates for the CAGR period; default = the pool's span.
    tie_seed: reshuffles which equally ranked setup gets a free slot (0 = default).
    sizing: 'dashboard' = shares from the entry level, capped at max_position_pct
      of equity and (cash_limit) by the free cash; 'r' = the old notional replay
      (P&L = risk x r_net, no cap). Trades without fill/stop use the 'r' rule.
    grade_filter: None, 'A' (only grade_proxy A) or 'A_in_caution' (only A while
      the regime halves the size).
    """
    exclude = {m: set(p) for m, p in (exclude or {}).items()}
    rank_by = rank_by or rank_field(trades)
    pool = []
    excluded = 0
    skipped_grade = 0
    for t in usable_trades(trades):
        conf = t.get("conf")
        if (conf if _finite(conf) else 1.0) < min_conf:
            continue
        if t.get("pattern") in exclude.get(t.get("market", ""), ()):
            excluded += 1
            continue
        if grade_filter == "A" and t.get("grade_proxy") != "A":
            skipped_grade += 1
            continue
        rn = trade_r_net(t, cost_r)
        if rn is None:
            continue
        pool.append({"t": t, "r_net": rn, "has_r_net": _finite(t.get("r_net"))})
    if not pool:
        return {"error": "no trades pass the filter", "risk_pct": risk_pct,
                "trades_excluded_by_pattern": excluded}

    entries_by_day: dict[str, list[dict]] = defaultdict(list)
    for p in pool:
        entries_by_day[p["t"]["entry_date"]].append(p)
    for lst in entries_by_day.values():  # best first, reproducible ties
        lst.sort(key=lambda p: (-(p["t"].get(rank_by) if _finite(p["t"].get(rank_by)) else 0.0),
                                _tiebreak(p["t"], tie_seed)))
    days_sorted = {m: sorted(tbl) for m, tbl in (regime or {}).items()}

    state = {"equity": float(start_capital), "peak": float(start_capital),
             "max_dd": 0.0, "wins": 0}
    yearly_pnl: dict[str, float] = {}
    net_source = {"r_net": 0, "legacy_cost_r": 0}
    sizing_source: dict[str, int] = defaultdict(int)
    exposure_days: dict[str, float] = defaultdict(float)
    gross_samples: list[float] = []
    positions: list[dict] = []
    capped = cash_limited = skipped_no_cash = 0

    def close(pos: dict) -> None:
        pnl = pos["pnl"]
        state["equity"] += pnl
        if pos["r_net"] > 0:
            state["wins"] += 1
        year = pos["t"]["exit_date"][:4]
        yearly_pnl[year] = yearly_pnl.get(year, 0.0) + pnl
        state["peak"] = max(state["peak"], state["equity"])
        if state["peak"] > 0:
            state["max_dd"] = max(state["max_dd"],
                                  (state["peak"] - state["equity"]) / state["peak"])

    open_pos: list[dict] = []
    taken = skipped_regime = skipped_full = 0

    def close_through(day: str, same_day: bool) -> list[dict]:
        keep = []
        for pos in open_pos:
            ex = pos["t"]["exit_date"]
            if ex < day or (same_day and ex == day):
                close(pos)
            else:
                keep.append(pos)
        return keep

    # unified timeline. Entries fill at the open (next-day open or an intraday
    # buy-stop) while exits happen intraday (stop / target) or at the close (time
    # exit), so a position exiting on day d still holds its slot, cash and P&L
    # when day d's entries are sized: earlier exits close first, then the day's
    # entries, then the positions that exit that same day. Markets are pooled on
    # one calendar, so this also stops a US close from freeing a slot for an
    # Indian open on the same date.
    timeline = sorted(set(entries_by_day) | {p["t"]["exit_date"] for p in pool})
    for day in timeline:
        open_pos = close_through(day, same_day=False)
        for p in entries_by_day.get(day, ()):
            t = p["t"]
            equity = state["equity"]
            if len(open_pos) >= max_pos or equity <= 0:
                skipped_full += 1
                continue
            mult = _mult_for(regime, t.get("market", ""), day, days_sorted) if regime else 1.0
            if mult <= 0:
                skipped_regime += 1  # market in downtrend: the system stands aside
                continue
            if grade_filter == "A_in_caution" and 0 < mult < 1 and t.get("grade_proxy") != "A":
                skipped_grade += 1   # caution: A-grades only
                continue
            risk_d = equity * risk_pct / 100.0 * mult
            fill, stop = t.get("entry_price"), t.get("stop")
            level, src = _entry_level(t)
            notional = shares = None
            if (sizing == "dashboard" and p["has_r_net"] and level is not None
                    and _finite(fill) and _finite(stop) and fill > stop):
                shares = risk_d / (level - stop)
                cap = equity * max_position_pct / 100.0
                was_capped = shares * level > cap
                if was_capped:
                    shares = cap / level
                if cash_limit:
                    free = equity - sum(x["notional"] or 0.0 for x in open_pos)
                    if shares * fill > free:
                        if free < MIN_FREE_CASH_FRAC * equity:
                            skipped_no_cash += 1   # no cash left: the slot stays free
                            continue
                        shares = free / fill
                        cash_limited += 1
                notional = shares * fill
                pnl = shares * p["r_net"] * (fill - stop)
                capped += was_capped     # counted only for trades actually taken
                sizing_source[src] += 1
            else:
                pnl = risk_d * p["r_net"]
                sf = _stop_frac(t)
                notional = risk_d / sf if sf else None
                sizing_source["r" if sizing == "r" and p["has_r_net"] else "legacy"] += 1
            pos = {"t": t, "r_net": p["r_net"], "pnl": pnl, "notional": notional,
                   "shares": shares}
            open_pos.append(pos)
            if notional is not None and equity > 0:
                days = max(1, (date.fromisoformat(t["exit_date"][:10])
                               - date.fromisoformat(day[:10])).days)
                exposure_days[t.get("market", "")] += notional / equity * days
            gross_samples.append(sum(x["notional"] or 0.0 for x in open_pos) / equity * 100)
            if record_positions:
                positions.append({"ticker": t.get("ticker"), "market": t.get("market"),
                                  "entry_date": day, "exit_date": t["exit_date"],
                                  "entry_price": fill if _finite(fill) else None,
                                  "shares": shares, "pnl": pnl, "notional": notional})
            net_source["r_net" if p["has_r_net"] else "legacy_cost_r"] += 1
            taken += 1
        open_pos = close_through(day, same_day=True)
    for pos in open_pos:  # anything left closes at its recorded exit
        close(pos)

    first, last = window or (min(p["t"]["entry_date"] for p in pool),
                             max(p["t"]["exit_date"] for p in pool))
    window_days = max(1, (date.fromisoformat(last[:10]) - date.fromisoformat(first[:10])).days)
    years = max(0.5, window_days / 365.25)
    equity = state["equity"]
    cagr = (equity / start_capital) ** (1 / years) - 1 if equity > 0 else -1.0

    yearly_ret: dict[str, float | None] = {}
    eq = float(start_capital)
    for y in sorted(yearly_pnl):
        yearly_ret[y] = round(yearly_pnl[y] / eq * 100, 1) if eq > 0 else None
        eq += yearly_pnl[y]

    tot_exp = sum(exposure_days.values())
    weights = ({m: round(v / tot_exp, 4) for m, v in sorted(exposure_days.items())} if tot_exp > 0
               else {})
    if not weights and taken:   # no size information at all: share of trades taken
        cnt: dict[str, int] = defaultdict(int)
        for p in pool:
            cnt[p["t"].get("market", "")] += 1
        weights = {m: round(v / len(pool), 4) for m, v in sorted(cnt.items())}
    out = {
        "risk_pct": risk_pct,
        "min_conf": min_conf,
        "max_pos": max_pos,
        "sizing": sizing,
        "max_position_pct": max_position_pct if sizing == "dashboard" else None,
        "cash_limit": bool(cash_limit) if sizing == "dashboard" else False,
        "rank_by": rank_by,
        "grade_filter": grade_filter,
        "trades_taken": taken,
        "trades_available": len(pool),
        "trades_excluded_by_pattern": excluded,
        "skipped_regime": skipped_regime,
        "skipped_no_free_slot": skipped_full,
        "skipped_grade": skipped_grade,
        "skipped_no_cash": skipped_no_cash,
        "capped_trades": capped,
        "pct_positions_capped": round(100 * capped / taken, 1) if taken else 0.0,
        "cash_limited_trades": cash_limited,
        "sizing_source": dict(sizing_source),
        "win_rate": round(100 * state["wins"] / taken, 1) if taken else 0,
        "final_equity": round(equity),
        "cagr_pct": round(cagr * 100, 1),
        "max_drawdown_pct": round(state["max_dd"] * 100, 1),
        "years": round(years, 1),
        "yearly_pnl": {y: round(v) for y, v in sorted(yearly_pnl.items())},
        "yearly_return_pct": yearly_ret,
        "cost_source": net_source,
        "market_exposure_weights": weights,
        "avg_gross_exposure_pct": round(tot_exp / window_days * 100, 1) if tot_exp else None,
        "max_gross_exposure_pct": round(max(gross_samples), 1) if gross_samples else None,
        "_cagr_raw": cagr * 100,
    }
    if record_positions:
        out["_positions"] = positions
    return out


# --------------------------------------------------------------------------- mark to market

def price_repairs(config: dict | None, data_dir: Path | None,
                  conn: sqlite3.Connection | None) -> tuple[tuple, dict]:
    """The backtest's price repairs for this cache: (overrides, stale_before map).

    overrides = config backtest.price_artifacts + data_dir/price_artifacts.json,
    stale map from the cache's meta table - exactly what run_backtest feeds
    sanitize_bars, so marked prices share the fills' price scale."""
    from types import SimpleNamespace

    from .backtest import Settings, load_price_artifacts, stale_before_map
    try:
        overrides = tuple(Settings.from_config(config or {}).price_artifacts)
    except Exception:
        overrides = ()
    if data_dir is not None:
        overrides += tuple(load_price_artifacts(Path(data_dir)))
    stale = stale_before_map(SimpleNamespace(conn=conn)) if conn is not None else {}
    return overrides, stale


def _sanitized_closes(conn: sqlite3.Connection, tickers: list[str], lo: str, hi: str,
                      overrides=(), stale_map: dict | None = None,
                      cache: dict | None = None) -> pd.DataFrame:
    """Like _load_closes, but each ticker's WHOLE history is first repaired with
    backtest.sanitize_bars (the same overrides / stale map the backtest used), so
    the closes are on the scale of the backtest's fill prices: back-adjusted
    splits and demergers, dropped one-day spikes and truncated bad joins.
    `cache` ({ticker: Series of closes by 'YYYY-MM-DD'}) is shared across calls."""
    from .backtest import sanitize_bars

    stale_map = stale_map or {}
    cache = cache if cache is not None else {}
    frames = []
    for t in tickers:
        if t not in cache:
            df = _load_bars(conn, t)
            if df.empty:
                cache[t] = pd.Series(dtype=float)
            else:
                clean, _ = sanitize_bars(df, t, overrides, stale_map.get(t))
                idx = pd.DatetimeIndex(clean.index).strftime("%Y-%m-%d")
                cache[t] = pd.Series(clean["close"].to_numpy(float), index=idx)
        ser = cache[t]
        if not len(ser):
            continue
        ser = ser[(ser.index >= lo) & (ser.index <= hi) & (ser > 0)]
        if len(ser):
            frames.append(pd.DataFrame({"ticker": t, "date": ser.index, "close": ser.to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close"])
    return pd.concat(frames, ignore_index=True)


def mark_to_market(positions: list[dict], start_capital: float,
                   conn: sqlite3.Connection | None, *, overrides=(),
                   stale_map: dict | None = None,
                   cache: dict | None = None) -> tuple[dict | None, str | None]:
    """Daily equity with open positions valued at each cached close.

    The closes are repaired like the backtest's own series (_sanitized_closes;
    pass the backtest's overrides and stale map from price_repairs), because the
    fills come from those repaired series - raw closes of a back-adjusted stock
    would show gains that vanish at the exit.

    Returns ({mtm_max_drawdown_pct, ann_volatility_pct, avg_gross_exposure_pct,
    max_gross_exposure_pct}, None) or (None, reason). Needs a fill price and a
    share count for every position (no reference price is invented)."""
    if not positions:
        return None, "no positions"
    if any(not _finite(p.get("entry_price")) or not _finite(p.get("shares")) or p["shares"] <= 0
           for p in positions):
        return None, "old stream or 'r' sizing: no fill/share count, MTM not available"
    if conn is None:
        return None, "price cache not found"
    tickers = sorted({str(p["ticker"]) for p in positions})
    first = min(p["entry_date"] for p in positions)
    last = max(p["exit_date"] for p in positions)
    long = _sanitized_closes(conn, tickers, first, last, overrides, stale_map, cache)
    panel = (long.drop_duplicates(["date", "ticker"])
             .pivot(index="date", columns="ticker", values="close").sort_index()
             if not long.empty else pd.DataFrame())
    grid = sorted(set(panel.index) | set(pd.bdate_range(first, last).strftime("%Y-%m-%d")))
    panel = panel.reindex(grid).ffill()
    d = np.asarray(grid)
    equity = np.full(len(d), float(start_capital))
    gross = np.zeros(len(d))
    for p in positions:
        closed = d >= p["exit_date"]
        equity += np.where(closed, p["pnl"], 0.0)
        live = (d >= p["entry_date"]) & (d < p["exit_date"])
        if not live.any():
            continue
        col = panel[str(p["ticker"])].to_numpy(float) if str(p["ticker"]) in panel else None
        px = np.where(np.isfinite(col), col, p["entry_price"]) if col is not None \
            else np.full(len(d), p["entry_price"])
        equity += np.where(live, p["shares"] * (px - p["entry_price"]), 0.0)
        gross += np.where(live, p["shares"] * px, 0.0)
    peak = np.maximum.accumulate(equity)
    dd = np.where(peak > 0, (peak - equity) / peak, 0.0)
    rets = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])
    expo = np.where(equity > 0, gross / equity * 100, np.nan)
    return {
        "mtm_max_drawdown_pct": _round(dd.max() * 100),
        "ann_volatility_pct": _round(float(np.std(rets, ddof=1)) * math.sqrt(252) * 100)
        if len(rets) > 2 else None,
        "avg_gross_exposure_pct": _round(np.nanmean(expo)),
        "max_gross_exposure_pct": _round(np.nanmax(expo)),
    }, None


# --------------------------------------------------------------------------- pattern filters

def _tuning(cfg: dict | None) -> dict:
    from .scan import PATTERN_TUNING_DEFAULTS, _deep_merge
    if cfg is None:
        try:
            from .scan import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
    user = (cfg or {}).get("pattern_tuning")
    tuning = _deep_merge(PATTERN_TUNING_DEFAULTS, user if isinstance(user, dict) else {})
    if not isinstance(tuning.get("disabled"), dict):
        tuning["disabled"] = {}
    return tuning


def pattern_filter(backtest: dict | None, cfg: dict | None = None
                   ) -> tuple[dict[str, list[str]] | None, str]:
    """Patterns the live scanner would disable, per market, from the v2 backtest
    verdicts under the CURRENT config's pattern_tuning (backtest.disabled_patterns,
    the same rule scan.load_pattern_tuning uses). Returns (None, reason) when no
    filter can be applied (missing / v1 backtest file)."""
    if not isinstance(backtest, dict) or not backtest.get("markets"):
        return None, "no backtest results file - no pattern filter applied"
    if _version(backtest) < 2:
        return None, ("old v1 backtest file (no fees, no verdicts) - no pattern filter "
                      "applied, so this profile equals SYSTEM AS DESIGNED")
    try:
        from .backtest import disable_rule_text, disabled_patterns
    except Exception as exc:
        return None, (f"backtest.disabled_patterns unavailable ({exc.__class__.__name__}) - "
                      "no pattern filter applied")
    tuning = _tuning(cfg)
    manual = tuning["disabled"]
    markets = list(backtest.get("markets") or {})
    markets += [m for m in manual if m not in markets]
    out: dict[str, list[str]] = {}
    for m in markets:
        per_pattern = ((backtest["markets"].get(m) or {}).get("per_pattern") or {})
        out[m] = sorted(disabled_patterns(per_pattern, tuning, m))
    note = (f"patterns the scanner currently disables ({disable_rule_text(tuning)}) are left "
            "out. The same 10 years of data pick the patterns and score them (in-sample), so "
            "this profile can come out higher or lower than the walk-forward one - it is not "
            "evidence of what the live scanner will deliver.")
    return out, note


def walk_forward_disabled(trades: list[dict], tuning: dict,
                          span: tuple[str | int, str | int] | None = None
                          ) -> tuple[dict[str, dict[str, list[str]]] | None, str]:
    """{market: {year: sorted disabled patterns}}: for each signal year Y the
    auto-disable decision is re-made from the trades that EXITED before Y-01-01
    only (plus the manual lists, a present-day choice). None for streams without
    the per-trade baseline fields.
    span: (first year, last year) to cover for every market, e.g. the signal
    calendar of the backtest's walk-forward replay; default = each market's
    first to last signal year."""
    from .backtest import pattern_stats, pattern_verdict

    closed = usable_trades(trades)
    if not closed or not all(all(k in t for k in WF_REQUIRED_KEYS) for t in closed):
        return None, "walk-forward needs a v2 trade stream with the per-trade baselines"
    manual = tuning.get("disabled") if isinstance(tuning.get("disabled"), dict) else {}
    auto = bool(tuning.get("auto_disable", True))
    need_proven = bool(tuning.get("require_proven_edge"))
    out: dict[str, dict[str, list[str]]] = {}
    markets = sorted({t["market"] for t in closed})
    for m in markets:
        mt = [t for t in closed if t["market"] == m]
        years = sorted({signal_year(t) for t in mt if signal_year(t)})
        if span is not None:
            years = [str(span[0])[:4], str(span[1])[:4]]
        if not years:
            continue
        out[m] = {}
        for y in range(int(years[0]), int(years[-1]) + 1):
            prior = [t for t in mt if t["exit_date"] < f"{y}-01-01"]
            groups: dict[str, list[dict]] = defaultdict(list)
            for t in prior:
                groups[t["pattern"]].append(t)
            dis = {str(p) for p in (manual.get(m) or [])}
            if auto:
                for pat, g in groups.items():
                    v = pattern_verdict(pattern_stats(g, tuning), tuning)[0]
                    if v == "no_edge" or (need_proven and v == "unproven"):
                        dis.add(pat)
            out[m][str(y)] = sorted(dis)
    return out, ("each year's disabled list is judged only on trades that closed before that "
                 "year started; patterns with too few earlier trades stay enabled")


# --------------------------------------------------------------------------- benchmark

def _index_portfolio(closes: dict[str, pd.Series], weights: dict[str, float], start: str,
                     end: str, div_yield_pct: dict | None = None) -> dict:
    """CAGR and yearly returns of a monthly-rebalanced portfolio of index series.

    Union of dates up to `end`, forward-filled (a market's holiday = 0% that day);
    anchored at the last date on/before `start` where every series has a value;
    weights renormalised over the markets present; each month's growth is
    sum_m w_m * prod(1 + r_m) over the month, rebalanced at each month end. With
    div_yield_pct, each step also accrues (1 + y/100)^(days/365.25)."""
    use = {m: s for m, s in closes.items() if s is not None and len(s) and weights.get(m, 0) > 0}
    if not use:
        return {"cagr_pct": None, "yearly_pct": {}, "weights": {}}
    panel = pd.concat({m: s[s.index <= end] for m, s in use.items()}, axis=1).sort_index().ffill()
    full = panel.notna().all(axis=1)
    before = panel.index[(panel.index <= start) & full.to_numpy()]
    if len(before):
        anchor = before[-1]
    else:
        after = panel.index[full.to_numpy()]
        if not len(after):
            return {"cagr_pct": None, "yearly_pct": {}, "weights": {}}
        anchor = after[0]
    panel = panel[panel.index >= anchor]
    if len(panel) < 2:
        return {"cagr_pct": None, "yearly_pct": {}, "weights": {}}
    growth_f = (panel / panel.shift(1)).fillna(1.0)
    if div_yield_pct:
        days = pd.Series([0] + [(date.fromisoformat(b) - date.fromisoformat(a)).days
                                for a, b in zip(panel.index[:-1], panel.index[1:])],
                         index=panel.index)
        for m in growth_f.columns:
            y = float((div_yield_pct or {}).get(m) or 0.0)
            if y:
                growth_f[m] = growth_f[m] * (1 + y / 100) ** (days / 365.25)
    tot = sum(weights[m] for m in use)
    w = {m: weights[m] / tot for m in use}
    growth = 1.0
    yearly: dict[str, float] = {}
    for month, g in growth_f.groupby(growth_f.index.str[:7]):
        mg = sum(w[m] * float(g[m].prod()) for m in w)
        growth *= mg
        y = month[:4]
        yearly[y] = yearly.get(y, 1.0) * mg
    years = _years_between(str(anchor), str(panel.index[-1]))
    cagr = (growth ** (1 / years) - 1) * 100 if years > 0 and growth > 0 else None
    return {"cagr_pct": _round(cagr, 2), "yearly_pct": {y: _round((v - 1) * 100) for y, v in yearly.items()},
            "start": str(anchor), "end": str(panel.index[-1]), "years": round(years, 2),
            "weights": {m: round(v, 4) for m, v in w.items()}, "_growth": growth}


def _market_benchmark(conn: sqlite3.Connection | None, market: str, bench_ticker: str,
                      start: str, end: str, universe_dir: Path,
                      div_yield_pct: float = 0.0, *, overrides=(),
                      stale_map: dict | None = None, cache: dict | None = None) -> dict:
    """Index and equal-weight universe buy-and-hold over [start, end]. The
    universe closes get the backtest's price repairs (_sanitized_closes with
    `overrides` / `stale_map` from price_repairs), so the basket and the trades
    share one price scale; the index series is used as cached."""
    out = {"index_ticker": bench_ticker, "index_cagr_pct": None,
           "universe_equal_weight_cagr_pct": None,
           "index_yearly_pct": {}, "universe_yearly_pct": {},
           "index_dividend_yield_pct": div_yield_pct}
    if conn is None:
        out["error"] = "price cache not found"
        return out
    b = _load_closes(conn, [bench_ticker], "0000-00-00", end)
    b = b.drop_duplicates("date").sort_values("date")
    dates = b["date"].tolist()
    closes = b["close"].astype(float).tolist()
    k0 = bisect_right(dates, start) - 1        # last index bar on/before the window start
    k0 = max(k0, 0)
    k1 = bisect_right(dates, end) - 1          # last index bar on/before the window end
    if not dates or k1 <= k0:
        out["error"] = "not enough index history in the window"
        return out
    a_date, e_date = dates[k0], dates[k1]
    years = (date.fromisoformat(e_date) - date.fromisoformat(a_date)).days / 365.25
    out.update({"start": a_date, "end": e_date, "years": round(years, 2)})
    y = float(div_yield_pct or 0.0)

    def cagr(total_growth: float) -> float | None:
        if years <= 0:
            return None
        return ((total_growth ** (1 / years)) - 1) * 100 if total_growth > 0 else -100.0

    idx_growth = closes[k1] / closes[k0]
    cagr_price = cagr(idx_growth)
    out["index_total_return_price_pct"] = _round((idx_growth - 1) * 100)
    out["index_cagr_price_pct"] = _round(cagr_price, 2)
    tot = idx_growth * (1 + y / 100) ** years
    out["index_total_return_pct"] = _round((tot - 1) * 100)
    out["index_cagr_pct"] = (_round(((1 + cagr_price / 100) * (1 + y / 100) - 1) * 100, 2)
                             if cagr_price is not None else None)
    for yr in range(int(a_date[:4]), int(e_date[:4]) + 1):
        base = max(bisect_right(dates, f"{yr - 1}-12-31") - 1, k0)
        last = min(bisect_right(dates, f"{yr}-12-31") - 1, k1)
        if last > base:
            seg_days = (date.fromisoformat(dates[last]) - date.fromisoformat(dates[base])).days
            g = closes[last] / closes[base] * (1 + y / 100) ** (seg_days / 365.25)
            out["index_yearly_pct"][str(yr)] = _round((g - 1) * 100)
    seg = np.asarray(closes[k0:k1 + 1], float)
    out["index_max_drawdown_pct"] = _round(float(np.max(1 - seg / np.maximum.accumulate(seg))) * 100)
    dr = seg[1:] / seg[:-1] - 1
    out["index_ann_volatility_pct"] = _round(float(np.std(dr, ddof=1)) * math.sqrt(252) * 100) \
        if len(dr) > 2 else None
    out["_closes"] = pd.Series(closes[: k1 + 1], index=dates[: k1 + 1], dtype=float)

    uni_path = Path(universe_dir) / f"{market}.csv"
    if not uni_path.exists():
        out["universe_error"] = f"universe/{market}.csv missing"
        return out
    tickers = [str(t) for t in pd.read_csv(uni_path)["ticker"].dropna().tolist()]
    out["universe_size"] = len(tickers)
    long = _sanitized_closes(conn, tickers, a_date, e_date, overrides, stale_map, cache)
    if long.empty:
        out["universe_error"] = "no universe prices in the window"
        return out
    panel = (long.drop_duplicates(["date", "ticker"])
             .pivot(index="date", columns="ticker", values="close").sort_index())
    # each stock's return on its own trading days (a missing bar spans to the next one)
    rets = pd.concat([panel[c].dropna().pct_change().rename(c) for c in panel.columns],
                     axis=1).sort_index()
    rets = rets[rets.index > a_date]
    clipped = int((rets.abs() > DAILY_RET_CLIP).sum().sum())
    rets = rets.clip(lower=-DAILY_RET_CLIP, upper=DAILY_RET_CLIP)
    ew = rets.mean(axis=1, skipna=True).dropna()
    out["universe_tickers_with_data"] = int(rets.notna().any().sum())
    out["clipped_ticker_days"] = clipped
    if ew.empty:
        out["universe_error"] = "no universe returns in the window"
        return out
    growth = float((1 + ew).prod())
    out["universe_total_return_pct"] = _round((growth - 1) * 100)
    out["universe_equal_weight_cagr_pct"] = _round(cagr(growth), 2)
    yearly = (1 + ew).groupby(ew.index.str[:4]).prod() - 1
    out["universe_yearly_pct"] = {str(yy): _round(v * 100) for yy, v in yearly.items()}
    wealth = pd.concat([pd.Series([1.0], index=[a_date]), (1 + ew).cumprod()])
    out["universe_max_drawdown_pct"] = _round(float((1 - wealth / wealth.cummax()).max()) * 100)
    out["_wealth"] = wealth
    return out


def dividend_yields(backtest: dict | None, cfg: dict | None) -> dict:
    """index_dividend_yield_pct from the backtest file (so both use the same),
    else the config, else the defaults."""
    from .scan import BACKTEST_DEFAULTS
    y = dict(BACKTEST_DEFAULTS["index_dividend_yield_pct"])
    c = ((cfg or {}).get("backtest") or {}).get("index_dividend_yield_pct")
    if isinstance(c, dict):
        y.update({k: float(v) for k, v in c.items()})
    b = ((backtest or {}).get("method") or {}).get("index_dividend_yield_pct") \
        if isinstance(backtest, dict) else None
    if isinstance(b, dict):
        y.update({k: float(v) for k, v in b.items() if _finite(v)})
    return y


def benchmark_block(start: str, end: str, markets: list[str] | None = None,
                    prices_path: Path | None = None,
                    universe_dir: Path | None = None,
                    div_yield_pct: dict | None = None, *, overrides=(),
                    stale_map: dict | None = None, cache: dict | None = None) -> dict:
    """Buy-and-hold over [start, end] per market, from the local price cache only.
    overrides / stale_map: the backtest's price repairs (price_repairs), applied
    to the universe closes. Series needed for per-profile hurdles are kept under
    the private key '_series'."""
    from .data import BENCHMARKS

    markets = [m for m in (markets or MARKET_ORDER) if m in BENCHMARKS]
    div = div_yield_pct or {}
    conn = _connect_ro(prices_path)
    block: dict = {"window": {"start": start, "end": end}}
    per: dict[str, dict] = {}
    try:
        for m in markets:
            try:
                per[m] = _market_benchmark(conn, m, BENCHMARKS[m], start, end,
                                           Path(universe_dir or DEFAULT_UNIVERSE_DIR),
                                           float(div.get(m, 0.0)), overrides=overrides,
                                           stale_map=stale_map, cache=cache)
            except Exception as exc:  # a broken market must not sink the report
                per[m] = {"index_ticker": BENCHMARKS[m], "index_cagr_pct": None,
                          "universe_equal_weight_cagr_pct": None, "index_yearly_pct": {},
                          "universe_yearly_pct": {},
                          "error": f"{exc.__class__.__name__}: {exc}"}
    finally:
        if conn is not None:
            conn.close()
    series = {"index": {m: v.pop("_closes") for m, v in per.items() if "_closes" in v},
              "universe": {m: v.pop("_wealth") for m, v in per.items() if "_wealth" in v},
              "div": {m: float(div.get(m, 0.0)) for m in per}}
    block.update(per)
    idx = {m: v["index_cagr_pct"] for m, v in per.items() if v.get("index_cagr_pct") is not None}
    uni = {m: v["universe_equal_weight_cagr_pct"] for m, v in per.items()
           if v.get("universe_equal_weight_cagr_pct") is not None}
    block["markets_included"] = list(per)
    eq_idx = {m: 1.0 for m in idx if m in series["index"]}
    ip = _index_portfolio(series["index"], eq_idx, start, end, series["div"]) if eq_idx else {}
    eq_uni = {m: 1.0 for m in uni if m in series["universe"]}
    up = _index_portfolio(series["universe"], eq_uni, start, end) if eq_uni else {}
    block["blended_index_cagr_pct"] = ip.get("cagr_pct")
    block["blended_index_yearly_pct"] = ip.get("yearly_pct", {})
    block["blended_index_cagr_simple_avg_pct"] = (_round(sum(idx.values()) / len(idx), 2)
                                                  if idx else None)
    block["blended_universe_cagr_pct"] = up.get("cagr_pct")
    block["blended_universe_yearly_pct"] = up.get("yearly_pct", {})
    block["blended_universe_cagr_simple_avg_pct"] = (_round(sum(uni.values()) / len(uni), 2)
                                                     if uni else None)
    block["blended_markets"] = list(eq_idx)
    clipped = sum(v.get("clipped_ticker_days", 0) for v in per.values())
    notes = [
        "System returns are % of one account sized like the dashboard, in local-currency "
        "prices with no FX; benchmarks are in each market's local currency.",
        "^GSPC and ^NSEI are price indices; their figures include an estimated dividend "
        "yield accrued daily (" + ", ".join(f"{m.upper()} {series['div'].get(m, 0):g}%/yr"
                                            for m in per)
        + "); ^GDAXI is a performance index (dividends included). The price-only figures are "
        "kept as index_cagr_price_pct.",
        "The equal-weight universe is today's index constituents held over the whole window "
        "(survivorship bias flatters it, as it flatters the backtest trades); rebalanced "
        "daily from each stock's close-to-close returns.",
        "Buy-and-hold benchmarks pay no fees (think of a cheap ETF); the system's results "
        "are after fees and slippage.",
        "The equal-weight universe uses the backtest's price repairs (known unadjusted "
        "splits and demergers, one-day spikes, bad joins), so it is measured on the same "
        "prices the trades were filled on; the index series are used as cached.",
        f"Single-stock daily returns are capped at +/-{DAILY_RET_CLIP:.0%} as a last guard "
        f"against bad bars the repairs miss ({clipped} stock-days affected).",
        f"Window {start} .. {end} (first entry to last exit of the trade stream); the first "
        "and last calendar years are partial.",
        "Blended = a portfolio holding the markets' indices (1/3 each), rebalanced monthly, "
        "with the yearly figures from the same monthly chain; the simple average of the CAGRs "
        "is kept as blended_index_cagr_simple_avg_pct for context. Each profile's "
        "'exposure-weighted index' hurdle uses the same method with that profile's own "
        "market weights (its capital-days per market). It is an opportunity-cost hurdle for "
        "a fully invested holding, while the system is only partly invested "
        "(avg_gross_exposure_pct), so it is not risk-matched; FX is not modelled.",
    ]
    missing = [m for m in per if m not in idx]
    if missing:
        notes.append(f"No index figure for: {', '.join(missing)} - left out of the blend.")
    block["notes"] = notes
    block["_series"] = series
    return block


# --------------------------------------------------------------------------- report

def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _embed_in_backtest(backtest_path: Path, block: dict, progress) -> bool:
    if not Path(backtest_path).exists():
        progress(f"WARN {backtest_path.name} missing - portfolio block not embedded")
        return False
    bt = _read_json(backtest_path)
    if not isinstance(bt, dict):
        progress(f"WARN {backtest_path.name} unreadable - portfolio block not embedded")
        return False
    if _version(bt) < 2:
        progress(f"WARN {backtest_path.name} is the old v1 format - portfolio block not "
                 "embedded (re-run: python -m atlas2 backtest)")
        return False
    bt["portfolio"] = block
    _atomic_write_json(backtest_path, bt, indent=1)
    return True


def _fmt_pct(v, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def _fmt_pp(v) -> str:
    return "n/a" if v is None else f"{v:+.1f} pp/yr"


def _quantiles(a) -> dict:
    a = np.asarray(a, float)
    return {"p5": _round(np.percentile(a, 5)), "median": _round(np.percentile(a, 50)),
            "p95": _round(np.percentile(a, 95)), "min": _round(a.min()), "max": _round(a.max())}


def run_profile(trades: list[dict], sim_kw: dict, blended_idx: float | None = None,
                reshuffles: int = TIE_RESHUFFLES) -> dict:
    """simulate_portfolio over seeds 0..reshuffles; the headline is the run whose
    CAGR is the exact median (lowest seed on ties), re-run with its positions."""
    runs = [(sd, simulate_portfolio(trades, tie_seed=sd, **sim_kw)) for sd in range(reshuffles + 1)]
    ok = [(sd, x) for sd, x in runs if "error" not in x]
    if not ok:
        r = dict(runs[0][1])
        r.pop("_cagr_raw", None)
        return r
    raws = sorted(x["_cagr_raw"] for _, x in ok)
    med = raws[len(raws) // 2]                                   # exact median (odd count)
    rep_seed = min(sd for sd, x in ok if x["_cagr_raw"] == med)  # lowest seed on ties
    r = simulate_portfolio(trades, tie_seed=rep_seed, record_positions=True, **sim_kw)
    r["representative_tie_seed"] = rep_seed
    r["seed0_cagr_pct"] = runs[0][1].get("cagr_pct")
    c = np.array([x["_cagr_raw"] for _, x in ok])
    dd = np.array([x["max_drawdown_pct"] for _, x in ok])
    tb = {"runs": len(ok), "seeds": [0, reshuffles], "cagr_pct": _quantiles(c),
          "max_drawdown_pct": _quantiles(dd), "max_drawdown_basis": "closed trades (realized)",
          # back-compat keys read by older dashboards (btTieRange)
          "cagr_pct_min": _round(c.min()), "cagr_pct_median": _round(np.median(c)),
          "cagr_pct_max": _round(c.max()), "max_drawdown_pct_worst": _round(dd.max())}
    if blended_idx is not None:
        tb["vs_blended_index_cagr_pp"] = _quantiles(c - blended_idx)
        tb["beats_blended_index_share"] = round(float((c > blended_idx).mean()), 3)
    r["tie_break_sensitivity"] = tb
    return r


def _reference_pre_costs(path: Path, reg: dict, progress, reshuffles: int = TIE_RESHUFFLES,
                         max_position_pct: float = DEFAULT_MAX_POSITION_PCT) -> dict | None:
    """The as-designed replay of the old (pre-fees) stream on THIS simulator,
    the same way as the profiles (run_profile: every slot ordering, the exact
    median path), so before/after figures come from the same code. Read-only.
    The old stream has no fill prices, so it is sized with the notional R rule
    and its drawdown is on closed trades."""
    old = _read_json(path)
    if not isinstance(old, list) or not old:
        return None
    win = stream_window(old)
    if win is None:
        return None
    r = run_profile(old, {"risk_pct": 1.0, "regime": reg, "window": win,
                          "max_position_pct": max_position_pct, "rank_by": rank_field(old)},
                    None, reshuffles)
    if "error" in r:
        return None
    tb = r.get("tie_break_sensitivity") or {}
    q = tb.get("cagr_pct") or {}
    progress(f"  reference (old pre-fees stream, this simulator): CAGR median {r['cagr_pct']}% "
             f"(p5 {q.get('p5')}% .. p95 {q.get('p95')}%, max {q.get('max')}% over "
             f"{tb.get('runs')} slot orderings)")
    return {"source": Path(path).name,
            "cagr_pct": r["cagr_pct"],
            "cagr_basis": f"median of {tb.get('runs')} slot orderings (seeds 0..{reshuffles})",
            "tie_break_cagr_pct": q,
            "runs": tb.get("runs"),
            "representative_tie_seed": r.get("representative_tie_seed"),
            "seed0_cagr_pct": r.get("seed0_cagr_pct"),
            "max_drawdown_pct": r.get("max_drawdown_pct"),
            "drawdown_basis": "closed trades (the old stream has no fill data)",
            "trades_taken": r.get("trades_taken"),
            # back-compat keys
            "tie_range_cagr_pct": [q.get("min"), q.get("max")],
            "tie_median_cagr_pct": q.get("median"),
            "window": list(win)}


def _reference_note(reference: dict | None) -> str:
    head = ("The portfolio replay was corrected on 2026-09-25: it reads the market regime at "
            "the previous session's close (the old replay used the entry day's own close, a "
            "one-day look-ahead) and, when more equally ranked setups than free slots trigger "
            "on one day, picks among them in a fixed pseudo-random order instead of the order "
            "they sat in the trade file. Since 2026-09-26 a position that exits on a day keeps "
            "its slot and cash until that day's entries are placed. The old replay showed 7.7%/yr "
            "'as designed' (drawdown 16.8%) on the old, pre-fees trade stream, from a single "
            "slot ordering.")
    q = (reference or {}).get("tie_break_cagr_pct") or {}
    if reference and _finite(reference.get("cagr_pct")) and _finite(q.get("p5")):
        body = (f" Replayed like the profiles above, that old stream gives {reference['cagr_pct']}%/yr "
                f"({reference.get('cagr_basis') or 'median slot ordering'}; p5-p95 {q.get('p5')}-"
                f"{q.get('p95')}%, highest {q.get('max')}%; closed-trade drawdown "
                f"{reference.get('max_drawdown_pct')}%), see reference_pre_costs.")
    else:
        body = (" reference_pre_costs (written when a full backtest runs the portfolio step) "
                "replays that old stream on the current simulator; it is not available in this "
                "report.")
    return head + body + " Figures from before the fees update are not comparable one-to-one."


def _stream_path(trades_path: Path, kind: str) -> Path:
    """backtest_trades<suffix>.json -> backtest_trades_<kind><suffix>.json"""
    return trades_path.with_name(trades_path.name.replace("backtest_trades", f"backtest_trades_{kind}", 1))


def _live_stream_ok(backtest: dict, exclude: dict) -> bool:
    """The backtest's live-configuration stream matches today's disabled lists."""
    for m, md in (backtest.get("markets") or {}).items():
        lc = (md or {}).get("live_config") or {}
        if lc.get("unavailable") or sorted(lc.get("disabled", ["?"])) != exclude.get(m, []):
            return False
    return True


# grade profile key -> (grade_replay entry, stream file kind, uses today's disabled lists)
GRADE_STREAMS = {"grade_a_only": ("grade_a", "grade_a", True),
                 "as_designed_a_in_caution": ("a_in_caution", "a_in_caution", False)}
GRADE_APPROX = " - approximate: B/C trades still block A setups; includes patterns the scanner hides"


def _grade_stream_ok(backtest: dict, key: str, exclude: dict | None) -> bool:
    """The backtest's grade replay for profile `key` exists for every market and
    was made with the pattern set the profile needs: today's disabled lists
    (grade-A only, like _live_stream_ok) or every pattern (A-in-caution)."""
    entry, _, uses_live = GRADE_STREAMS[key]
    markets = (backtest or {}).get("markets") or {}
    if not markets or (uses_live and exclude is None):
        return False
    for m, md in markets.items():
        gr = (md or {}).get("grade_replay")
        if not isinstance(gr, dict) or gr.get("unavailable"):
            return False
        g = gr.get(entry)
        if not isinstance(g, dict):
            return False
        want = exclude.get(m, []) if uses_live else []
        if sorted(g.get("disabled", ["?"])) != want:
            return False
    return True


def _walk_forward_stream_ok(backtest: dict, trades: list[dict], tuning: dict) -> bool:
    """The backtest's walk-forward replay used the yearly disabled lists that the
    current config gives (re-judged here from the headline stream)."""
    markets = backtest.get("markets") or {}
    if not markets:
        return False
    for m, md in markets.items():
        wfb = (md or {}).get("walk_forward")
        if not isinstance(wfb, dict) or wfb.get("unavailable") or not wfb.get("span"):
            return False
        cur, _ = walk_forward_disabled([t for t in trades if t.get("market") == m], tuning,
                                       span=tuple(wfb["span"]))
        if ((cur or {}).get(m) or {}) != (wfb.get("disabled_by_year") or {}):
            return False
    return True


def run_portfolio_report(progress=print, trades_path: Path | None = None,
                         backtest_path: Path | None = None, out_path: Path | None = None,
                         *, prices_path: Path | None = None,
                         universe_dir: Path | None = None,
                         config: dict | None = None,
                         live_trades_path: Path | None = None,
                         walk_forward_trades_path: Path | None = None,
                         grade_a_trades_path: Path | None = None,
                         a_in_caution_trades_path: Path | None = None,
                         reference_path: Path | None = None,
                         reshuffles: int = TIE_RESHUFFLES) -> dict:
    """Simulate every profile on the trade stream, add the buy-and-hold benchmark,
    write out_path and embed the block into the (v2) backtest file. Returns the block.
    prices_path / universe_dir / config are injectable for tests (config=None reads
    the user's config via scan.load_config for pattern_tuning and max_position_pct).
    reference_path: the old pre-fees stream to replay for reference_pre_costs
    (default: the backup, for a stand-alone run on the default files)."""
    default_run = trades_path is None
    trades_path = Path(trades_path or DEFAULT_TRADES_PATH)
    backtest_path = Path(backtest_path or DEFAULT_BACKTEST_PATH)
    out_path = Path(out_path or DEFAULT_OUT_PATH)
    live_trades_path = Path(live_trades_path) if live_trades_path else _stream_path(trades_path, "live")
    walk_forward_trades_path = (Path(walk_forward_trades_path) if walk_forward_trades_path
                                else _stream_path(trades_path, "walkforward"))
    grade_paths = {
        "grade_a_only": (Path(grade_a_trades_path) if grade_a_trades_path
                         else _stream_path(trades_path, "grade_a")),
        "as_designed_a_in_caution": (Path(a_in_caution_trades_path) if a_in_caution_trades_path
                                     else _stream_path(trades_path, "a_in_caution")),
    }
    if reference_path is None and default_run:
        reference_path = REFERENCE_PRE_COSTS_PATH
    if config is None:
        try:
            from .scan import load_config
            config = load_config()
        except Exception:
            config = {}
    max_position_pct = float((config or {}).get("max_position_pct", DEFAULT_MAX_POSITION_PCT))
    if not trades_path.exists():
        progress("No trade stream found - run: python -m atlas2 backtest first")
        return {"error": "no trade stream found"}
    trades = _read_json(trades_path)
    if not isinstance(trades, list):
        progress(f"Trade stream {trades_path.name} is unreadable - re-run the backtest")
        return {"error": "trade stream unreadable"}
    usable = usable_trades(trades)
    window = stream_window(trades)
    if window is None:
        progress("Trade stream has no closed trades - nothing to simulate")
        return {"error": "no closed trades"}
    has_net = sum(1 for t in usable if _finite(t.get("r_net")))
    open_at_end = sum(1 for t in trades if isinstance(t, dict) and t.get("outcome") == "open_at_end")
    if has_net == len(usable):
        cost_note = "Every trade is counted after fees and slippage (r_net from the backtest)."
    elif has_net == 0:
        cost_note = (f"Old trade stream without per-trade costs: fees and slippage approximated "
                     f"as {LEGACY_COST_R}R per trade.")
    else:
        cost_note = (f"{has_net} trades use the backtest's r_net; {len(usable) - has_net} old "
                     f"trades use r - {LEGACY_COST_R}R.")
    progress(f"trade stream: {len(trades)} trades ({len(usable)} closed, "
             f"{open_at_end} open at end of data skipped), {window[0]} .. {window[1]}")
    progress(f"  {cost_note}\n")

    reg = regime_multipliers(prices_path)
    backtest = _read_json(backtest_path) if backtest_path.exists() else None
    tuning = _tuning(config)
    exclude, filter_note = pattern_filter(backtest, config)
    markets = [m for m in MARKET_ORDER if any(t.get("market") == m for t in usable)]
    markets += sorted({t.get("market") for t in usable
                       if t.get("market") and t.get("market") not in markets})
    reg_conn = _connect_ro(prices_path)
    # the backtest's price repairs: used for the marks AND the equal-weight universe
    mtm_overrides, mtm_stale = price_repairs(config, trades_path.parent, reg_conn)
    mtm_cache: dict = {}
    bench = benchmark_block(window[0], window[1], markets or None, prices_path, universe_dir,
                            dividend_yields(backtest, config), overrides=mtm_overrides,
                            stale_map=mtm_stale, cache=mtm_cache)
    series = bench.pop("_series", {"index": {}, "universe": {}, "div": {}})
    blended_idx = bench.get("blended_index_cagr_pct")
    blended_uni = bench.get("blended_universe_cagr_pct")
    rank_by = rank_field(trades)

    # enabled-only: the backtest's live-configuration stream when its disabled
    # lists match today's filter, else the approximate deletion
    live_trades, live_note = None, None
    if exclude is not None and isinstance(backtest, dict):
        lt = _read_json(live_trades_path) if live_trades_path.exists() else None
        if _live_stream_ok(backtest, exclude) and isinstance(lt, list) and lt:
            live_trades = lt
            live_note = ("uses the backtest's live-configuration replay (disabled patterns "
                         "removed before the best setup was picked, runner-up setups included)")
        else:
            live_note = ("approximate (switched-off patterns' trades deleted, runner-up setups "
                         "not simulated) - re-run the backtest")
    wf, wf_note = walk_forward_disabled(trades, tuning)
    # walk-forward: the backtest's own re-selection with each year's disabled list
    # (removed before the pick, like live) when it matches the current config, else deletion
    wf_trades = None
    if wf is not None and isinstance(backtest, dict) and _version(backtest) >= 2 \
            and _walk_forward_stream_ok(backtest, trades, tuning):
        wt = _read_json(walk_forward_trades_path) if walk_forward_trades_path.exists() else None
        if isinstance(wt, list):
            wf_trades = wt
    if wf is None:
        wf_stream_note = None
    elif wf_trades is not None:
        wf_stream_note = ("uses the backtest's walk-forward replay: each year's switched-off "
                          "patterns are removed before the best setup is picked, as the live "
                          "scanner does, so runner-up setups are traded")
    else:
        wf_stream_note = ("approximate: switched-off patterns' trades are deleted and runner-up "
                          "setups are not simulated (no matching walk-forward replay in the "
                          "backtest files - re-run the backtest)")
    has_grade = bool(usable) and all(t.get("grade_proxy") for t in usable)
    # grade profiles: the backtest's grade replays (non-A setups never ordered, so
    # they never block a later A setup) when they match the current config, else
    # the headline stream with the other grades' trades deleted (approximate)
    grade_streams: dict[str, list[dict] | None] = {}
    for gkey, gpath in grade_paths.items():
        gs = None
        if isinstance(backtest, dict) and _version(backtest) >= 2 \
                and _grade_stream_ok(backtest, gkey, exclude) and gpath.exists():
            gs = _read_json(gpath)
        grade_streams[gkey] = gs if isinstance(gs, list) else None
    regime_off = any(t.get("regime") == "off" for t in usable)
    no_regime_label = ("No regime filter (1% risk) - every signal, downtrend markets included"
                       if regime_off else
                       "Full size in every market state (1% risk) - no half-sizing in caution; "
                       "the backtest already left out signals raised, and orders filled, while "
                       "the index was in a downtrend")

    def hurdle(r: dict, kind: str) -> float | None:
        w = r.get("market_exposure_weights") or {}
        src = series.get(kind) or {}
        w = {m: v for m, v in w.items() if m in src and v > 0}
        if not w:
            return None
        return _index_portfolio(src, w, window[0], window[1],
                                series.get("div") if kind == "index" else None).get("cagr_pct")

    def wf_deleted_stream() -> list[dict]:
        return [t for t in usable
                if t.get("pattern") not in (wf.get(t.get("market")) or {}).get(signal_year(t), [])]

    profiles = []
    raws: dict[str, float] = {}
    bases: dict[str, dict] = {}
    try:
        for key, label, kw, use_regime, mode, grade in PROFILES:
            label = label or no_regime_label
            base = {"regime": reg if use_regime else None, "window": window,
                    "max_position_pct": max_position_pct, "rank_by": rank_by,
                    "grade_filter": grade, **kw}
            bases[key] = dict(base)
            stream = trades
            extra: dict = {}
            if grade and not has_grade:
                r = {"error": "the trade stream has no grade_proxy (re-run the backtest)",
                     "risk_pct": kw.get("risk_pct")}
            elif mode == "walk_forward" and wf is None:
                r = {"error": wf_note, "risk_pct": kw.get("risk_pct")}
            else:
                if mode == "walk_forward":
                    if wf_trades is not None:
                        stream = wf_trades
                        extra = {"stream": "walk_forward_replay"}
                    else:
                        stream = wf_deleted_stream()
                        extra = {"stream": "deleted",
                                 "trades_excluded_by_pattern": len(usable) - len(stream)}
                        label += " - approximate: switched-off patterns' trades deleted"
                    extra.update({"disabled_by_year": wf,
                                  "note": f"{wf_note}; {wf_stream_note}."})
                elif mode == "in_sample":
                    if live_trades is not None:
                        stream = live_trades
                    else:
                        base["exclude"] = exclude
                elif key in GRADE_STREAMS:
                    gs = grade_streams.get(key)
                    if gs is not None:
                        # grade_filter stays on as a second pass: a no-op for grade A;
                        # for A-in-caution it also applies the entry-day regime
                        stream = gs
                        extra = {"stream": "grade_replay"}
                        if GRADE_STREAMS[key][2]:
                            extra["disabled_patterns"] = exclude or {}
                            extra["note"] = (
                                "Uses the backtest's grade replay: setups below grade A are "
                                "never ordered, so they do not block a later A setup, and the "
                                "patterns the scanner switches off today are removed before the "
                                "pick (chosen with hindsight on the same data).")
                        else:
                            extra["note"] = (
                                "Uses the backtest's grade replay: in caution, setups below "
                                "grade A are never ordered, so they do not block a later A "
                                "setup; every pattern is traded, as designed.")
                    else:
                        extra = {"stream": "deleted",
                                 "note": ("Approximate: the headline stream with the other "
                                          "grades' trades deleted - their orders still block "
                                          "later A-grade setups, and patterns the scanner hides "
                                          "are traded (no matching grade replay in the "
                                          "backtest files - re-run the backtest).")}
                        label += GRADE_APPROX
                r = run_profile(stream, base, blended_idx, reshuffles)
            raw = r.pop("_cagr_raw", None)
            positions = r.pop("_positions", None)
            r.update(extra)
            r["key"] = key
            r["label"] = label
            r["regime_filter"] = use_regime
            if raw is not None:
                raws[key] = raw
                mtm, why = mark_to_market(positions or [], START_CAPITAL, reg_conn,
                                          overrides=mtm_overrides, stale_map=mtm_stale,
                                          cache=mtm_cache)
                r["max_drawdown_realized_pct"] = r.get("max_drawdown_pct")
                if mtm:
                    r["max_drawdown_pct"] = mtm["mtm_max_drawdown_pct"]
                    r["drawdown_basis"] = "marked to market daily"
                    r["ann_volatility_pct"] = mtm["ann_volatility_pct"]
                    r["avg_gross_exposure_pct"] = mtm["avg_gross_exposure_pct"]
                    r["max_gross_exposure_pct"] = mtm["max_gross_exposure_pct"]
                else:
                    r["drawdown_basis"] = f"closed trades only ({why})"
                hi, hu = hurdle(r, "index"), hurdle(r, "universe")
                r["hurdle_index_cagr_pct"] = hi
                r["vs_hurdle_index_cagr_pp"] = _round(raw - hi) if hi is not None else None
                r["hurdle_universe_cagr_pct"] = hu
                r["vs_hurdle_universe_cagr_pp"] = _round(raw - hu) if hu is not None else None
            r["vs_blended_index_cagr_pp"] = (_round(raw - blended_idx)
                                             if raw is not None and blended_idx is not None else None)
            r["vs_blended_universe_cagr_pp"] = (_round(raw - blended_uni)
                                                if raw is not None and blended_uni is not None else None)
            if mode == "in_sample":
                r["disabled_patterns"] = exclude or {}
                r["note"] = filter_note + (f" Stream: {live_note}." if live_note else "")
                r["stream"] = "live_config" if live_trades is not None else "deleted"
            if key == "as_designed":
                r["note"] = ("All patterns, including the ones the live scanner currently hides "
                             "(headline per-pattern selection).")
            profiles.append(r)
            if "error" in r:
                progress(f"{label}\n  {r['error']}\n")
                continue
            tb = r.get("tie_break_sensitivity") or {}
            tc = tb.get("cagr_pct") or {}
            td = tb.get("max_drawdown_pct") or {}
            progress(f"{label}\n  CAGR median {r['cagr_pct']}% (p5 {tc.get('p5')}% .. p95 "
                     f"{tc.get('p95')}%, seed-0 path {r.get('seed0_cagr_pct')}%) | max drawdown "
                     f"{r['max_drawdown_pct']}% ({r.get('drawdown_basis')}; closed trades p5 "
                     f"{td.get('p5')}% .. p95 {td.get('p95')}%) | final €{r['final_equity']:,} from "
                     f"€{START_CAPITAL:,.0f} in {r['years']}y | {r['trades_taken']} trades | win "
                     f"{r['win_rate']}%")
            if tb.get("beats_blended_index_share") is not None:
                progress(f"  beats the blended index in {tb['beats_blended_index_share']:.0%} of "
                         f"{tb['runs']} slot orderings")
            if mode == "in_sample":
                dis = ", ".join(f"{m.upper()}: {', '.join(p) or '-'}" for m, p in (exclude or {}).items())
                progress(f"  {r['note']}" + (f"\n  disabled -> {dis}" if exclude else ""))
            if mode == "walk_forward" or (key in GRADE_STREAMS and r.get("note")):
                progress(f"  {r['note']}")
            progress("  yearly P&L: " + ", ".join(f"{y}: {v:+,}" for y, v in r["yearly_pnl"].items()))
            progress("  yearly return: " + ", ".join(f"{y}: {_fmt_pct(v, True)}"
                                                     for y, v in r["yearly_return_pct"].items()))
            if r["vs_blended_index_cagr_pp"] is not None:
                progress(f"  system vs buy-and-hold: {_fmt_pp(r['vs_blended_index_cagr_pp'])} vs the "
                         f"blended index ({_fmt_pct(blended_idx)}/yr), "
                         f"{_fmt_pp(r['vs_blended_universe_cagr_pp'])} vs "
                         f"equal-weight markets ({_fmt_pct(blended_uni)}/yr); "
                         f"{_fmt_pp(r.get('vs_hurdle_index_cagr_pp'))} vs its exposure-weighted "
                         f"index ({_fmt_pct(r.get('hurdle_index_cagr_pct'))}/yr)\n")
            else:
                progress("  system vs buy-and-hold: benchmark unavailable\n")

        # hindsight effect, like for like: both sides re-selected (patterns removed
        # before the pick) or both sides with the switched-off patterns' trades deleted
        hindsight = selection_effect = hindsight_basis = None
        is_key, wf_key = "as_designed_enabled_only", "as_designed_walk_forward"
        if is_key in raws and wf_key in raws:
            is_sel, wf_sel = live_trades is not None, wf_trades is not None
            is_raw, wf_raw = raws[is_key], raws[wf_key]
            if is_sel == wf_sel:
                hindsight_basis = "re-selection" if is_sel else "deletion"
            elif is_sel:      # in-sample re-selected, walk-forward deleted: delete in-sample too
                x = run_profile(trades, {**bases[is_key], "exclude": exclude}, None, reshuffles)
                if "_cagr_raw" in x:
                    selection_effect = round(is_raw - x["_cagr_raw"], 1)
                    is_raw, hindsight_basis = x["_cagr_raw"], "deletion"
            else:             # walk-forward re-selected, in-sample deleted: delete walk-forward too
                x = run_profile(wf_deleted_stream(), bases[wf_key], None, reshuffles)
                if "_cagr_raw" in x:
                    wf_raw, hindsight_basis = x["_cagr_raw"], "deletion"
            if hindsight_basis is not None:
                hindsight = round(is_raw - wf_raw, 1)
                for r in profiles:
                    if r.get("key") == is_key:
                        r["hindsight_effect_pp"] = hindsight
                        r["hindsight_basis"] = hindsight_basis
                        if selection_effect is not None:
                            r["selection_effect_pp"] = selection_effect
                progress(f"hindsight effect (in-sample minus walk-forward, both by "
                         f"{hindsight_basis}): {hindsight:+.1f} pp/yr"
                         + (f"; re-selection instead of deletion moves the in-sample figure by "
                            f"{selection_effect:+.1f} pp/yr" if selection_effect is not None else ""))
    finally:
        if reg_conn is not None:
            reg_conn.close()

    progress(f"Buy-and-hold over the same window ({window[0]} .. {window[1]}, local currency, no fees):")
    for m in bench.get("markets_included", []):
        b = bench[m]
        if b.get("error"):
            progress(f"  {m.upper()} {b['index_ticker']}: {b['error']}")
            continue
        progress(f"  {m.upper()} {b['index_ticker']}: index {_fmt_pct(b['index_cagr_pct'])}/yr "
                 f"(price only {_fmt_pct(b.get('index_cagr_price_pct'))}/yr, max drawdown "
                 f"{_fmt_pct(b.get('index_max_drawdown_pct'))}) | equal-weight universe "
                 f"{_fmt_pct(b['universe_equal_weight_cagr_pct'])}/yr")
    progress(f"  blended (monthly rebalanced): index {_fmt_pct(blended_idx)}/yr | "
             f"equal-weight universe {_fmt_pct(blended_uni)}/yr")

    reference = None
    if reference_path is not None and Path(reference_path).exists() \
            and Path(reference_path).resolve() != trades_path.resolve():
        try:
            reference = _reference_pre_costs(Path(reference_path), reg, progress, reshuffles,
                                             max_position_pct)
        except Exception as exc:
            progress(f"  (reference replay skipped: {exc.__class__.__name__})")

    hs_text = ("Hindsight effect = in-sample CAGR minus walk-forward CAGR (either sign), "
               "compared like for like: "
               + ("both streams re-selected (switched-off patterns removed before the pick)."
                  if hindsight_basis == "re-selection" else
                  "both with the switched-off patterns' trades deleted"
                  + (f" (re-selecting instead of deleting moves the in-sample figure by "
                     f"{selection_effect:+.1f} pp/yr)." if selection_effect is not None else ".")
                  if hindsight_basis == "deletion" else "not available."))
    block = _json_safe({
        "generated": datetime.now().isoformat(timespec="seconds"),
        "profiles": profiles,
        "benchmark": bench,
        "source_trades": len(trades),
        "closed_trades": len(usable),
        "open_at_end_skipped": open_at_end,
        "window": {"start": window[0], "end": window[1]},
        "start_capital": START_CAPITAL,
        "rank_by": rank_by,
        "hindsight_effect_pp": hindsight,
        "hindsight_basis": hindsight_basis,
        "selection_effect_pp": selection_effect,
        "walk_forward": {"disabled_by_year": wf, "note": wf_note,
                         "stream": None if wf is None else
                         ("walk_forward_replay" if wf_trades is not None else "deleted")},
        "reference_pre_costs": reference,
        "notes": [
            cost_note,
            "Each trade risks a % of current closed-trade equity and is sized like the "
            "dashboard: shares = risk / (entry level - stop), at most "
            f"{max_position_pct:g}% of equity per position and never more than the free cash "
            "(trades that do not fit are shrunk, or skipped when under 2% of equity is free); "
            "at most 6 positions at once. The 2% / 3% profiles hit the position cap on most "
            "trades. 'Notional R account' keeps the old uncapped replay for comparison.",
            "Entries fill at the open (next-day open or an intraday buy-stop) while exits "
            "happen intraday or at the close, so a position that exits on a day keeps its slot "
            "and its cash until that day's entries are placed (all markets share one calendar).",
            f"When more setups trigger on one day than there are free slots, the best "
            f"{'grade composite (grade_proxy score)' if rank_by == 'composite_proxy' else 'pattern confidence'} "
            f"wins. Headline figures are the median of {reshuffles + 1} slot orderings (seeds "
            f"0..{reshuffles}) that differ only in which equally ranked setup gets a free slot; "
            "the p5-p95 range covers slot ordering only and is a lower bound on uncertainty (it "
            "ignores sampling error in the trades themselves). The live scanner ranks by "
            "composite score with fundamentals, so this ordering is itself an approximation. "
            "Each profile picks its own median path (representative_tie_seed).",
            "Regime filter: the index state at the previous session's close (what the evening "
            "scan showed before the entry day) - caution halves the risk, downtrend skips the "
            "trade. Scanner-mode (v2) trade streams already leave out signals raised while "
            "the index was in a downtrend, and orders that filled after it turned down (the "
            "backtest re-checks the fill day with this same previous-close rule), so for them "
            "the downtrend skip here does not bind and the 'full size' profile differs from "
            "as designed only by not halving the risk in caution; with older streams it also "
            "keeps the downtrend entries. The regime needs 210 index bars; before that it is "
            "'unknown' and traded at normal size, as the live scanner does (v2 streams start "
            "after a 252-bar warm-up, so none of their signals is 'unknown').",
            "Max drawdown is marked to market daily from cached closes (open positions valued at "
            "each close, with the backtest's price repairs applied so marks and fills share one "
            "price scale); max_drawdown_realized_pct is the closed-trade figure. Yearly returns "
            "are booked by exit year.",
            f"Enabled-patterns profile: {filter_note}" + (f" Stream: {live_note}." if live_note else ""),
            "Walk-forward profile: " + wf_note + (f"; {wf_stream_note}" if wf_stream_note else "")
            + ". It is still a backtest: the settings and the pattern set were chosen on this "
            "same history. " + hs_text,
            "Grade profiles (grades are a no-fundamentals proxy): "
            + ("they trade the backtest's grade replays, where setups below grade A are never "
               "ordered and so never block a later A-grade setup on the same stock; grade-A only "
               "also leaves out the patterns the scanner switches off today (in-sample choice), "
               "A-grades only in caution keeps every pattern."
               if all(v is not None for v in grade_streams.values()) else
               "at least one has no matching grade replay in the backtest files and deletes the "
               "other grades' trades from the headline stream instead (approximate: those "
               "trades still block later A-grade setups, and hidden patterns are traded) - "
               "re-run the backtest."),
            "SYSTEM AS DESIGNED trades every pattern, including those the live scanner currently "
            "hides; headline per-pattern stats pick the best pattern among ALL patterns, so an "
            "enabled pattern gets more (and different) trades live than its headline row shows.",
            "Per-pattern results were unstable between 2017-21 and 2022-26 (half-to-half "
            "correlation about 0.03 on the old stream, about 0.00 on the v2 smoke run), so read "
            "the year-by-year figures before trusting any profile.",
            _reference_note(reference),
        ],
    })
    _atomic_write_json(out_path, block, indent=1)
    embedded = _embed_in_backtest(backtest_path, block, progress)
    progress(f"\nportfolio report -> {out_path}" + (f" (+ embedded in {backtest_path.name})"
                                                   if embedded else ""))
    return block
