"""Strategy lab 3 (pre-registered 2026-11-10, before any result was seen).

The owner asked for swing-trade rules that make 40% a year, leaving tax and fees
out. Labs 1 and 2 found no rule that does after costs; the scanner's chart
patterns made 3.4% a year BEFORE costs (backtest 2026-09-26, all markets,
2017-2026) while the S&P 500 made about 15%. This lab tests three families of
swing-trade entry rules, every rule below fixed before running:

UNIVERSE - the stocks that were REALLY in the S&P 500 at the time
  Lab 2's family P data and rules: data/cache/lab_sp500.sqlite3 (Yahoo, split-
  and dividend-adjusted), repaired with backtest.sanitize_bars, membership
  stints from data/reference/sp500_ticker_start_end.csv; a stint whose Yahoo
  series starts more than 31 days after it began is dropped (recycled ticker).
  Eligible on day t: a member on t, a bar on t, at least 253 of its own bars up
  to t, no unresolved price jump in its last 252 bars.

ENGINE (every family)
  Signals at the close of day t; orders fill at the OPEN of t+1, sells first,
  then buys in rank order. K slots: a buy gets min(equity at the close of t / K,
  cash); one position per stock. Idle cash earns T-bills (^IRX). A stock whose
  prices end while it is held is sold at its last close. Positions are marked at
  every close (the last close when a bar is missing). Exits are decided at a
  close and filled at the next open (no intraday stops).
  COSTS: none - the owner's request; this run picks the winner and is the one
  compared with 40%. Next to it the same run with 0.20% per side (the scanner's
  US allowance: 0.15% fee + 0.05% slippage), never used to pick anything.

FAMILY M - buy a short dip in an uptrend (mean reversion)
  entry: close above its 200-day average and RSI(2) below X (Wilder smoothing,
    alpha 1/2, seeded with the first change); rank: lowest RSI(2).
  exit: close above its 5-day average, or at the 10th session after the fill.
  X in {5, 10}.
FAMILY H - 52-week-high momentum
  entry: close above every close of the previous 251 sessions and a 126-session
    return in the top 10% of that day's eligible stocks; rank: highest
    126-session return.
  exit: close below its M-day average. M in {20, 50}.
FAMILY B - breakout from a tight base
  entry: close above every close of the previous 50 sessions, the previous 20
    closes within R ((highest - lowest) / the previous close <= R), volume above
    1.5x its previous 50-session average, close above its 200-day average;
    rank: highest 126-session return.
  exit: close below its 20-day average, or below the base's lowest close (the
    lowest of the 20 closes before the signal, fixed at the signal).
  R in {8%, 12%}.
Each family x index filter {off, on = entries only while the ^GSPC close is above
its 200-day average on the signal day} x K {5, 10} = 8 variants, 24 in all.

TRAIN = 1998-01-02 .. 2012-12-31, TEST = 2013-01-01 .. the last bar: one
continuous run per variant, both periods measured on the same curve. Winner (per
family and overall) = the highest TRAIN CAGR, no costs, among the variants whose
TRAIN max drawdown is no worse than -50%. The lab 'reaches the target' only when
the overall winner made 40% a year on TEST, no costs.
Diagnostics, none used to pick: every variant's TEST figures (the best of 24 on
TEST is hindsight, labelled so), the 0.20%-per-side run, SPY and QQQ, the
winner's yearly returns, trade statistics, rolling 10-year windows, and the
winner with 2x margin (daily returns doubled, the borrowed half paying T-bills
+ 0.5% - an approximation of a margin account, not a simulation of one).
Known bias: before 2012 only 38-70% of member-days have prices (lab 2's
coverage, repeated in the output). The missing names are mostly companies that
failed or were bought, so TRAIN figures are flattered; TEST much less so.

Run: python -m atlas2 strategy-lab-3 [--offline] [--out PATH]
Output: data/strategy_lab3.json
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .backtest import _rnd, write_json_atomic
from .lab_data import MEMBERSHIP_PATH, SP500_DB_PATH, load_membership
from .strategy_lab import (
    DEFAULT_SERIES_PATH, TARGET_CAGR_PCT, TRADING_DAYS, asof_values, cash_returns,
    ensure_series, exact, index_trend_flags, monthly_stats, perf_metrics, pick_winner,
    rolling_windows,
)
from .strategy_lab2 import coverage_by_year, load_member_panel, member_matrix, usable_stints

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = ROOT / "data" / "strategy_lab3.json"
LAB3_TICKERS = ("^GSPC", "^IRX", "SPY", "QQQ")

START = "1998-01-02"
TRAIN_END = "2012-12-31"
DD_FLOOR_PCT = -50.0
MIN_BARS = 253
JUMP_LOOKBACK = 252
SLOTS = (5, 10)
FILTERS = ("off", "on")
COST_AFTER = 0.0020             # per side, diagnostic only
MARGIN = 2.0
MARGIN_SPREAD = 0.005

M_RSI = (5, 10)
M_MAX_HOLD = 10
H_EXIT_SMA = (20, 50)
H_TOP = 0.10
B_RANGE = (0.08, 0.12)
B_BREAKOUT = 50
B_BASE = 20
B_VOL_X = 1.5

FAMILIES = {
    "M": ("dip in an uptrend", M_RSI),
    "H": ("52-week-high momentum", H_EXIT_SMA),
    "B": ("tight-base breakout", B_RANGE),
}
PERIODS = ("train", "test", "full")


# --------------------------------------------------------------------------- indicators

def rsi(close: np.ndarray, n: int = 2) -> np.ndarray:
    """Wilder RSI per column (alpha 1/n, seeded with the first change). NaN until a
    column's second bar; 100 when there were no falls, 50 when nothing moved."""
    d = pd.DataFrame(close).diff()
    a = 1.0 / n
    up = d.clip(lower=0).ewm(alpha=a, adjust=False).mean().to_numpy()
    dn = (-d).clip(lower=0).ewm(alpha=a, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r = np.where(dn > 0, r, np.where(up > 0, 100.0, 50.0))
    return np.where(np.isfinite(up) & np.isfinite(dn), r, np.nan)


def _roll(df: pd.DataFrame, n: int, how: str) -> np.ndarray:
    return getattr(df.rolling(n, min_periods=n), how)().to_numpy()


def eligibility(dates: np.ndarray, tickers: list[str], C: np.ndarray, member: np.ndarray,
                suspect: dict | None) -> np.ndarray:
    valid = np.isfinite(C)
    enough = np.cumsum(valid, axis=0) >= MIN_BARS
    jump = np.zeros(C.shape, bool)
    pos = {t: k for k, t in enumerate(tickers)}
    for t, ds in (suspect or {}).items():
        k = pos.get(str(t))
        if k is None:
            continue
        for d in ds:
            i = int(np.searchsorted(dates, str(d)[:10]))
            if i < len(dates) and dates[i] == str(d)[:10]:
                jump[i, k] = True
    cj = np.cumsum(jump, axis=0)
    recent = cj - np.vstack([np.zeros((JUMP_LOOKBACK, C.shape[1]), cj.dtype), cj[:-JUMP_LOOKBACK]]
                            )[:len(cj)] > 0
    return member & valid & enough & ~recent


def build_signals(C: np.ndarray, V: np.ndarray, eligible: np.ndarray) -> SimpleNamespace:
    """Every family's building blocks, from each stock's bars up to the day itself."""
    c = pd.DataFrame(C)
    prev = c.shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r126 = C / c.shift(126).to_numpy() - 1.0
        rng20 = (_roll(prev, B_BASE, "max") - _roll(prev, B_BASE, "min")) / prev.to_numpy()
        vavg = _roll(pd.DataFrame(V).shift(1), B_BREAKOUT, "mean")
        pct = pd.DataFrame(np.where(eligible, r126, np.nan)).rank(axis=1, pct=True).to_numpy()
        s = SimpleNamespace(
            trend=C > _roll(c, 200, "mean"),
            above5=C > _roll(c, 5, "mean"),
            below20=C < _roll(c, 20, "mean"),
            below50=C < _roll(c, 50, "mean"),
            rsi2=rsi(C, 2),
            r126=r126,
            top=pct > 1.0 - H_TOP,
            newhigh=C > _roll(prev, 251, "max"),
            breakout=C > _roll(prev, B_BREAKOUT, "max"),
            rng20=rng20,
            lo20p=_roll(prev, B_BASE, "min"),
            volx=V > B_VOL_X * vavg,
        )
    return s


def family_inputs(fam: str, param, S: SimpleNamespace):
    """(entry, rank key - lower is better, exit signal, stop fixed at the signal, max hold)."""
    if fam == "M":
        with np.errstate(invalid="ignore"):
            entry = S.trend & (S.rsi2 < param)
        return entry, S.rsi2, S.above5, None, M_MAX_HOLD
    if fam == "H":
        return S.newhigh & S.top, -S.r126, S.below20 if param == 20 else S.below50, None, None
    if fam == "B":
        with np.errstate(invalid="ignore"):
            entry = S.breakout & (S.rng20 <= param) & S.volx & S.trend
        return entry, -S.r126, S.below20, S.lo20p, None
    raise ValueError(fam)


def v_name(fam: str, param, filt: str, slots: int) -> str:
    p = {"M": f"rsi{param}", "H": f"sma{param}", "B": f"base{int(round(param * 100))}"}[fam]
    return f"{fam}_{p}_filter_{filt}_k{slots}"


def v_label(fam: str, param, filt: str, slots: int) -> str:
    what = {"M": f"RSI(2) below {param}, out above the 5-day average or after {M_MAX_HOLD} days",
            "H": f"new 52-week high + top-10% momentum, out below the {param}-day average",
            "B": f"breakout from a base within {param:.0%}, out below the 20-day average "
                 f"or the base"}[fam]
    return (f"{FAMILIES[fam][0]}: {what}; {slots} slots"
            + (", only while the S&P 500 is above its 200-day average" if filt == "on" else ""))


# --------------------------------------------------------------------------- engine

def last_valid_index(C: np.ndarray) -> np.ndarray:
    valid = np.isfinite(C)
    n = len(C)
    rev = np.argmax(valid[::-1], axis=0)
    return np.where(valid.any(axis=0), n - 1 - rev, -1)


def run_book(O: np.ndarray, C: np.ndarray, entry: np.ndarray, key: np.ndarray,
             exit_sig: np.ndarray, *, slots: int, cost: float, rf: np.ndarray, i0: int, i1: int,
             stop: np.ndarray | None = None, max_hold: int | None = None,
             last_valid: np.ndarray | None = None) -> dict:
    """The swing book of the module docstring. equity[i0] = 1; NaN outside [i0, i1].
    trades: (fill index, exit index, column, return after costs)."""
    n, m = C.shape
    if last_valid is None:
        last_valid = last_valid_index(C)
    eq = np.full(n, np.nan)
    inv = np.full(n, np.nan)
    last = np.full(m, np.nan)
    cash = 1.0
    pos: dict[int, list] = {}           # column -> [shares, fill index, fill price, stop]
    buys: list[tuple] = []
    sells: set[int] = set()
    trades: list[tuple] = []

    def close_out(k: int, px: float, i: int) -> None:
        nonlocal cash
        sh, fi, epx, _ = pos.pop(k)
        cash += sh * px * (1 - cost)
        trades.append((fi, i, k, px * (1 - cost) / (epx * (1 + cost)) - 1))

    for t in range(i0, i1 + 1):
        if t > i0:
            cash *= 1 + rf[t]
            pending = set()
            for k in sorted(sells):
                if k not in pos:
                    continue
                px = O[t, k]
                if np.isfinite(px) and px > 0:
                    close_out(k, px, t)
                else:
                    pending.add(k)              # no bar today: try again at the next open
            sells = pending
            for k, alloc, stp in buys:
                px = O[t, k]
                a = min(alloc, cash)
                if k in pos or not (np.isfinite(px) and px > 0) or a <= 0:
                    continue
                pos[k] = [a / (px * (1 + cost)), t, px, stp]
                cash -= a
            buys = []
        row = C[t]
        ok = np.isfinite(row)
        last[ok] = row[ok]
        for k in [k for k in pos if last_valid[k] == t and t < n - 1]:
            close_out(k, row[k], t)             # its prices end here
            sells.discard(k)
        val = sum(p[0] * last[k] for k, p in pos.items())
        eq[t] = cash + val
        inv[t] = val / eq[t] if eq[t] > 0 else 0.0
        if t == i1:
            break
        for k, p in pos.items():
            if k in sells or not ok[k]:
                continue
            stp = p[3]
            if exit_sig[t, k] or (stp is not None and np.isfinite(stp) and row[k] < stp) \
                    or (max_hold is not None and t - p[1] >= max_hold):
                sells.add(k)
        free = slots - (len(pos) - len(sells))
        if free > 0:
            cand = np.flatnonzero(entry[t])
            if pos:
                cand = cand[~np.isin(cand, list(pos))]
            if len(cand) > free:
                kk = key[t, cand]
                kk = np.where(np.isfinite(kk), kk, np.inf)
                cand = cand[np.argsort(kk, kind="stable")[:free]]
            alloc = eq[t] / slots
            buys = [(int(k), alloc, None if stop is None else float(stop[t, k])) for k in cand]
    return {"equity": eq, "invested": inv, "trades": trades, "open_at_end": len(pos)}


def margin_curve(eq: np.ndarray, rf: np.ndarray, i0: int, i1: int, lev: float = MARGIN,
                 spread: float = MARGIN_SPREAD) -> np.ndarray:
    """Daily returns times lev, the borrowed (lev - 1) paying T-bills + spread."""
    out = np.full(len(eq), np.nan)
    r = eq[i0 + 1:i1 + 1] / eq[i0:i1] - 1
    g = np.maximum(0.0, 1 + lev * r - (lev - 1) * (rf[i0 + 1:i1 + 1] + spread / TRADING_DAYS))
    out[i0] = 1.0
    out[i0 + 1:i1 + 1] = np.cumprod(g)
    return out


# --------------------------------------------------------------------------- reporting

def trade_stats(trades: list[tuple], dates: np.ndarray, lo: int, hi: int) -> dict:
    sel = [t for t in trades if lo <= t[0] <= hi]
    if not sel:
        return {"trades": 0}
    r = np.array([t[3] for t in sel])
    held = np.array([t[1] - t[0] for t in sel])
    days = (pd.Timestamp(str(dates[hi])) - pd.Timestamp(str(dates[lo]))).days
    years = max(days / 365.25, 1e-9)
    return {"trades": len(sel), "per_year": _rnd(len(sel) / years, 1),
            "win_rate_pct": _rnd(float((r > 0).mean()) * 100, 1),
            "avg_return_pct": _rnd(float(r.mean()) * 100, 2),
            "median_hold_sessions": int(np.median(held))}


def _slim(m: dict) -> dict:
    keep = ("start", "end", "years", "cagr_pct", "cagr_exact_pct", "max_drawdown_pct",
            "max_drawdown_exact_pct", "worst_year", "ann_volatility_pct")
    return {k: m.get(k) for k in keep}


def _spy_keys(x):
    """rolling_windows names its reference curve after lab 1's Nasdaq; here it is SPY."""
    if isinstance(x, dict):
        return {k.replace("nasdaq_1x", "spy").replace("1x_nasdaq", "spy"): _spy_keys(v)
                for k, v in x.items()}
    if isinstance(x, list):
        return [_spy_keys(v) for v in x]
    return x


def periods_of(dates: np.ndarray, eq: np.ndarray, idx: dict, slim: bool = True) -> dict:
    out = {}
    for p in PERIODS:
        m = perf_metrics(dates, eq, *idx[p])
        out[p] = _slim(m) if slim else m
    return out


def run_lab3(dates: np.ndarray, tickers: list[str], O: np.ndarray, C: np.ndarray, V: np.ndarray,
             member: np.ndarray, frames: dict, *, suspect: dict | None = None,
             start: str = START, train_end: str = TRAIN_END, progress=print) -> dict:
    """The lab on a ready panel (tests call it on synthetic data)."""
    n = len(dates)
    i_start = int(np.searchsorted(dates, start))
    i_tr = int(np.searchsorted(dates, train_end, side="right")) - 1
    if not (0 <= i_start < i_tr < n - 1):
        raise ValueError("the panel does not cover the TRAIN and TEST periods")
    idx = {"train": (i_start, i_tr), "test": (i_tr, n - 1), "full": (i_start, n - 1)}
    elig = eligibility(dates, tickers, C, member, suspect)
    S = build_signals(C, V, elig)
    g = frames["^GSPC"]
    up, defined = index_trend_flags(np.asarray(g.index, dtype=str), g["close"].to_numpy(float),
                                    dates)
    up = up & defined
    irx = frames["^IRX"]
    rf = cash_returns(np.nan_to_num(asof_values(np.asarray(irx.index, dtype=str),
                                                irx["close"].to_numpy(float), dates)))
    lv = last_valid_index(C)

    variants, curves = {}, {}
    for fam, (_, params) in FAMILIES.items():
        for param in params:
            entry, key, ex, stop, mh = family_inputs(fam, param, S)
            for filt in FILTERS:
                e = entry & elig & (up[:, None] if filt == "on" else True)
                for k in SLOTS:
                    name = v_name(fam, param, filt, k)
                    runs = {c: run_book(O, C, e, key, ex, slots=k, cost=c, rf=rf, i0=i_start,
                                        i1=n - 1, stop=stop, max_hold=mh, last_valid=lv)
                            for c in (0.0, COST_AFTER)}
                    free, paid = runs[0.0], runs[COST_AFTER]
                    variants[name] = {
                        "family": fam, "label": v_label(fam, param, filt, k),
                        "periods": periods_of(dates, free["equity"], idx),
                        "after_costs": {p: _slim(perf_metrics(dates, paid["equity"], *idx[p]))
                                        for p in ("test", "full")},
                        "trades": {p: trade_stats(free["trades"], dates, *idx[p])
                                   for p in ("train", "test")},
                        "avg_invested_pct": _rnd(float(np.nanmean(free["invested"][i_start:]))
                                                 * 100, 1),
                    }
                    curves[name] = free["equity"]
                    t = variants[name]["periods"]["test"]["cagr_pct"]
                    progress(f"  {name:28s} test {t}%/yr (no costs)")

    picks = {fam: pick_winner({k: v for k, v in variants.items() if v["family"] == fam},
                              DD_FLOOR_PCT) for fam in FAMILIES}
    pick = pick_winner(variants, DD_FLOOR_PCT)
    best_test = max(variants, key=lambda k: exact(variants[k]["periods"]["test"], "cagr_pct")
                    if exact(variants[k]["periods"]["test"], "cagr_pct") is not None else -1e9)

    bench = {}
    for t in ("SPY", "QQQ"):
        f = frames[t]
        px = asof_values(np.asarray(f.index, dtype=str), f["adj_close"].to_numpy(float), dates)
        b = {}
        for p in PERIODS:
            i0, i1 = idx[p]
            b[p] = _slim(perf_metrics(dates, px / px[i0], i0, i1)) if np.isfinite(px[i0]) \
                else None
        bench[t] = b

    winner = None
    if pick["name"]:
        w = pick["name"]
        eq = curves[w]
        full = perf_metrics(dates, eq, *idx["full"])
        lev = margin_curve(eq, rf, i_start, n - 1)
        spy = frames["SPY"]
        spy_px = asof_values(np.asarray(spy.index, dtype=str), spy["adj_close"].to_numpy(float),
                             dates)
        ms = monthly_stats(dates, eq, i_start, n - 1)
        winner = {
            "name": w, "label": variants[w]["label"],
            "yearly_pct": full.get("yearly_pct"),
            "with_2x_margin": {p: _slim(perf_metrics(dates, lev, *idx[p])) for p in PERIODS},
            "rolling_10y": _spy_keys(rolling_windows(dates, eq, ms,
                                                     ref_equity=spy_px / spy_px[i_start])),
        }
    test_cagr = exact(variants[pick["name"]]["periods"]["test"], "cagr_pct") if pick["name"] \
        else None
    return {
        "variants": variants, "pick": pick, "family_picks": picks, "winner": winner,
        "best_on_test_hindsight": {"name": best_test,
                                   "test": variants[best_test]["periods"]["test"],
                                   "note": "the best of 24 chosen by looking at TEST - "
                                           "hindsight, not evidence"},
        "benchmarks": bench,
        "reaches_target": bool(test_cagr is not None and test_cagr >= TARGET_CAGR_PCT),
        "any_variant_40_on_test": sorted(
            k for k, v in variants.items()
            if (exact(v["periods"]["test"], "cagr_pct") or -1e9) >= TARGET_CAGR_PCT),
        "periods": {p: [str(dates[a]), str(dates[b])] for p, (a, b) in idx.items()},
    }


def run_strategy_lab3(out_path: Path | None = None, *, offline: bool = False,
                      series_path: Path | None = None, db_path: Path | None = None,
                      progress=print) -> dict:
    started = time.time()
    out_path = Path(out_path) if out_path else DEFAULT_OUT_PATH
    frames, series_info = ensure_series(Path(series_path or DEFAULT_SERIES_PATH),
                                        offline=offline, tickers=LAB3_TICKERS, progress=progress)
    progress("loading the S&P 500 member prices ...")
    stints = load_membership(MEMBERSHIP_PATH)
    panel, first_bar, n_log, rep = load_member_panel(Path(db_path or SP500_DB_PATH), progress)
    kept, none, reuse = usable_stints(stints, first_bar)
    member = member_matrix(kept, panel.dates, panel.tickers)
    res = run_lab3(panel.dates, panel.tickers, panel.O, panel.C, panel.V, member, frames,
                   suspect=panel.suspect, progress=progress)
    payload = {
        "version": 1, "generated": datetime.now().isoformat(timespec="seconds"),
        "pre_registered": "2026-11-10", "target_cagr_pct": TARGET_CAGR_PCT,
        "costs": {"winner_and_target": "none (the owner's request)",
                  "after_costs_per_side_pct": COST_AFTER * 100},
        "series": series_info["series"],
        "universe": {"stints": len(stints), "usable_stints": len(kept),
                     "stints_without_prices": len(none), "stints_dropped_reuse": len(reuse),
                     "tickers_with_prices": len(panel.tickers),
                     "coverage_by_year": coverage_by_year(stints, kept, panel.dates)},
        **res,
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
    out = [f"== lab 3, no costs; TRAIN {p['periods']['train'][0]}..{p['periods']['train'][1]}, "
           f"TEST {p['periods']['test'][0]}..{p['periods']['test'][1]}"]
    for name, v in p["variants"].items():
        tr, t, f = (v["periods"][k] for k in PERIODS)
        out.append(f"   {name:28s} train {_fmt(tr['cagr_pct'])} dd {_fmt(tr['max_drawdown_pct'])}"
                   f" | test {_fmt(t['cagr_pct'])} dd {_fmt(t['max_drawdown_pct'])}"
                   f" | full {_fmt(f['cagr_pct'])} | test after 0.2%/side "
                   f"{_fmt(v['after_costs']['test']['cagr_pct'])}")
    for b, bv in p["benchmarks"].items():
        t = bv.get("test") or {}
        out.append(f"   [test] {b:8s} {_fmt(t.get('cagr_pct'))} dd {_fmt(t.get('max_drawdown_pct'))}")
    w = p.get("winner")
    if w:
        m2 = w["with_2x_margin"]["test"]
        out.append(f"   winner {w['name']} ({w['label']}); with 2x margin on TEST "
                   f"{_fmt(m2['cagr_pct'])} dd {_fmt(m2['max_drawdown_pct'])}")
    h = p["best_on_test_hindsight"]
    out.append(f"   best on TEST (hindsight): {h['name']} {_fmt(h['test']['cagr_pct'])}")
    out.append(f"   reaches 40%: {p['reaches_target']}; variants at 40%+ on TEST: "
               f"{', '.join(p['any_variant_40_on_test']) or 'none'}")
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
    run_strategy_lab3(out, offline=offline)


if __name__ == "__main__":
    main()
