"""Nasdaq signal: paper-track strategy lab 2's leverage rule on live prices.

The rule is lab 2's pre-registered winner (family V, vol30_w60_none): leverage
on the Nasdaq-100 = min(3, 30% / the volatility of the last 60 daily moves),
read after every US close, changed only when it moves by 0.25 or more, traded
at the NEXT close. It is held the way the lab's realistic account holds it: a
3x Nasdaq-100 fund for L/3 of the account, T-bills for the rest.

Every run downloads ^NDX / ^IRX / QQQ / TQQQ into their own cache
(data/cache/signal_series.sqlite3 - the lab caches are left as they are),
replays the signal over the whole ^NDX history (the 0.25 band makes today's
value depend on the path, so it is never computed from a short window), and
replays a PAPER account from the close tracking started - fixed on the first
run in data/leverage_paper.json - with TQQQ's real prices, 0.10% per side.
Nothing is bought or sold and no money is involved. When the latest close
changed the signal, one macOS notification says so.

Run: python -m atlas2 leverage-signal [--offline] [--no-notify]
     (`python -m atlas2 scan` runs it too, so it updates every morning at 07:30)
Output: data/leverage_signal.json, shown in the dashboard's Nasdaq signal tab.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .backtest import _rnd, write_json_atomic
from .strategy_lab import asof_values, cash_returns, ensure_series, perf_metrics
from .strategy_lab2 import V_LMAX, V_STEP, realized_vol, run_vol_account, vol_target_signal

ROOT = Path(__file__).resolve().parent.parent
SERIES_PATH = ROOT / "data" / "cache" / "signal_series.sqlite3"
STATE_PATH = ROOT / "data" / "leverage_paper.json"
OUT_PATH = ROOT / "data" / "leverage_signal.json"
LAB2_PATH = ROOT / "data" / "strategy_lab2.json"

TICKERS = ("^NDX", "^IRX", "QQQ", "TQQQ")
TARGET_VOL = 0.30            # lab 2's pre-registered winner: vol30_w60_none
WINDOW = 60
FILTER = "none"
HISTORY_SESSIONS = 260       # about a year of the signal for the chart
NOTIFY_TITLE = "Shaikh Capital — Nasdaq signal"


def _r(x, nd: int = 4):
    return _rnd(x, nd)


def load_series(path: Path, offline: bool, fetch=None, progress=print) -> tuple[dict, dict]:
    """Fresh series (downloaded again) unless offline; when the download fails the
    cached copy is used and the result says so."""
    warning = None
    if not offline:
        try:
            frames, info = ensure_series(path, refresh=True, tickers=TICKERS, fetch=fetch,
                                         progress=progress)
            return frames, {"stale": False, "warning": None, "info": info}
        except Exception as exc:
            warning = (f"download failed ({exc.__class__.__name__}: {exc}); showing the last "
                       "saved prices")
            progress(f"WARN {warning}")
    frames, info = ensure_series(path, offline=True, tickers=TICKERS, progress=progress)
    return frames, {"stale": warning is not None, "warning": warning, "info": info}


def load_state(path: Path, last_close: str) -> tuple[dict, str | None]:
    """The paper account's settings; created on the first run with the latest close
    as the start (never moved afterwards). An unreadable file is kept as .bad and
    a new one is started."""
    path = Path(path)
    note = None
    if path.exists():
        try:
            state = json.loads(path.read_text())
            if isinstance(state, dict) and isinstance(state.get("start"), str):
                return state, None
            raise ValueError("no start date")
        except (ValueError, OSError) as exc:
            path.replace(path.with_name(path.name + ".bad"))
            note = f"the paper account file was unreadable ({exc}); a new one was started"
    state = {"start": last_close, "created": datetime.now().isoformat(timespec="seconds"),
             "last_notified": None}
    write_json_atomic(path, state, indent=1)
    return state, note


def lab_evidence(path: Path) -> dict | None:
    """The backtest figures behind the rule, from data/strategy_lab2.json."""
    try:
        lab = json.loads(Path(path).read_text())
        v = lab["vol_target_leverage"]
        w = v["winner"]
        p = v["variants"][w["name"]]["periods"]
    except Exception:
        return None
    pick = lambda m, *ks: {k: m.get(k) for k in ks} if isinstance(m, dict) else None

    def span(m: dict | None) -> str | None:
        """'2010–2026' for a period anchored at the 2009-12-31 close."""
        if not isinstance(m, dict) or not m.get("start") or not m.get("end"):
            return None
        y0 = int(m["start"][:4]) + (1 if m["start"][5:] >= "12-24" else 0)
        return f"{y0}–{m['end'][:4]}"

    boot = w.get("bootstrap") or {}
    test = pick(p.get("test"), "start", "end", "cagr_pct", "max_drawdown_pct", "worst_year",
                "avg_leverage")
    full = pick(p.get("full"), "start", "end", "cagr_pct", "max_drawdown_pct", "worst_year")
    for m in (test, full):
        if m is not None:
            m["span"] = span(m)
    roll = w.get("rolling_10y") or {}
    return {
        "variant": w["name"], "generated": lab.get("generated"),
        "test": test, "full": full,
        "qqq_test": pick(v["benchmarks"]["test"].get("QQQ"), "cagr_pct", "max_drawdown_pct"),
        "nasdaq_full": pick(v["benchmarks"]["full"].get("nasdaq_1x_model"), "cagr_pct",
                            "max_drawdown_pct"),
        "account_test_de": w.get("account_test_de"),
        "real_fund_check": w.get("real_fund_check"),
        "bootstrap": pick(boot, "p_cagr_ge_40_pct", "p_max_drawdown_worse_than_50_pct",
                          "cagr_pct"),
        "rolling_10y": pick(roll, "share_ge_target_pct", "share_ge_1x_nasdaq_pct",
                            "min_cagr_pct", "median_cagr_pct"),
    }


def build_signal(frames: dict, state: dict) -> dict:
    """Everything the tab shows, from the series and the paper account's start."""
    ndx = frames["^NDX"]
    dates = np.asarray(ndx.index, dtype=str)
    close = ndx["close"].to_numpy(float)
    n = len(dates)

    def asof(t: str, col: str) -> np.ndarray:
        f = frames[t]
        return asof_values(np.asarray(f.index, dtype=str), f[col].to_numpy(float), dates)

    tb = asof("^IRX", "close")
    sig = vol_target_signal(close, TARGET_VOL, WINDOW, FILTER)
    vol = realized_vol(close, WINDOW)
    t = n - 1
    if not (np.isfinite(sig[t]) and np.isfinite(sig[t - 1])):
        raise ValueError("not enough ^NDX history for the signal")
    held, new = float(sig[t - 1]), float(sig[t])
    changes = np.nonzero(sig[1:] != sig[:-1])[0] + 1       # signal closes where it moved
    last_change = int(changes[-1]) if len(changes) else None

    # the paper account, opened at the close of the start day
    i_open = int(np.searchsorted(dates, state["start"], side="right")) - 1
    if i_open < 1:
        raise ValueError(f"the paper start {state['start']} is before the price history")
    tq = asof("TQQQ", "adj_close")
    q = asof("QQQ", "adj_close")
    fund_ret = np.full(n, np.nan)
    fund_ret[1:] = tq[1:] / tq[:-1] - 1
    eq, h = run_vol_account(fund_ret, cash_returns(tb), sig, i_open, t, return_holdings=True)
    qq = q / q[i_open]
    trades = [{"date": str(dates[i_open]), "from": 0.0, "to": _r(sig[i_open - 1], 2),
               "opening": True}]
    trades += [{"date": str(dates[d]), "from": _r(sig[d - 2], 2), "to": _r(sig[d - 1], 2)}
               for d in range(i_open + 1, t + 1) if sig[d - 1] != sig[d - 2]]
    pm = perf_metrics(dates, eq, i_open, t) if t > i_open else {}
    qm = perf_metrics(dates, qq, i_open, t) if t > i_open else {}
    value = float(h.sum())
    lo = max(0, n - HISTORY_SESSIONS)
    return {
        "as_of": str(dates[t]),
        "ndx_close": _r(close[t], 2),
        "vol_60d_pct": _r(vol[t] * 100, 2),
        "leverage_formula_now": _r(min(V_LMAX, TARGET_VOL / vol[t]), 3),
        "signal": {
            "held": _r(held, 3), "new": _r(new, 3), "trade_due": bool(new != held),
            "fund_share_pct": _r(new / V_LMAX * 100, 1),
            "cash_share_pct": _r((1 - new / V_LMAX) * 100, 1),
            "held_fund_share_pct": _r(held / V_LMAX * 100, 1),
            "last_change": ({"signal_close": str(dates[last_change]),
                             "from": _r(sig[last_change - 1], 3),
                             "to": _r(sig[last_change], 3)} if last_change else None),
            "changes_last_year": int(((changes >= lo) & (changes <= t)).sum()),
        },
        "paper": {
            "start": str(dates[i_open]), "sessions": int(t - i_open),
            "value_per_10000": _r(10000.0 * value, 2),          # after the opening trade's cost
            "return_pct": _r((value - 1) * 100, 2),
            "qqq_return_pct": _r((qq[t] - 1) * 100, 2),
            "max_drawdown_pct": pm.get("max_drawdown_pct"),
            "qqq_max_drawdown_pct": qm.get("max_drawdown_pct"),
            "fund_pct_now": _r(100 * h[0] / value, 1) if value > 0 else None,
            "cash_pct_now": _r(100 * h[1] / value, 1) if value > 0 else None,
            "trades": trades,
            "curve": [[str(dates[i]), _r(eq[i], 5), _r(qq[i], 5)] for i in range(i_open, t + 1)],
        },
        "history": [[str(dates[i]), _r(sig[i], 3), _r(vol[i] * 100, 2)] for i in range(lo, t + 1)],
        "last_bars": {k: str(frames[k].index[-1]) for k in TICKERS},
    }


def notify_message(sig: dict, as_of: str) -> str:
    return (f"The leverage signal moved from {sig['held']:.2f}x to {sig['new']:.2f}x at the "
            f"{as_of} close. Paper trade at the next US close: {sig['fund_share_pct']:.0f}% in a "
            f"3x Nasdaq fund, the rest in T-bills.")


def run_leverage_signal(*, offline: bool = False, notify: bool = True,
                        series_path: Path | None = None, state_path: Path | None = None,
                        out_path: Path | None = None, lab_path: Path | None = None,
                        fetch=None, notifier=None, progress=print) -> dict:
    series_path = Path(series_path or SERIES_PATH)
    state_path = Path(state_path or STATE_PATH)
    out_path = Path(out_path or OUT_PATH)
    frames, data = load_series(series_path, offline, fetch=fetch, progress=progress)
    state, state_note = load_state(state_path, str(frames["^NDX"].index[-1]))
    body = build_signal(frames, state)
    notified = False
    if notify and body["signal"]["trade_due"] and state.get("last_notified") != body["as_of"]:
        if notifier is None:
            from .monitor import _notify as notifier
        notifier(NOTIFY_TITLE, notify_message(body["signal"], body["as_of"]))
        state["last_notified"] = body["as_of"]
        write_json_atomic(state_path, state, indent=1)
        notified = True
    payload = {
        "version": 1, "available": True,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule": {"name": "vol30_w60_none", "target_vol_pct": TARGET_VOL * 100, "window": WINDOW,
                 "max_leverage": V_LMAX, "band": V_STEP,
                 "label": "Nasdaq-100 leverage = 30% / the last 60 days' volatility, at most 3x; "
                          "changed only by 0.25 or more; traded at the next close; held as a 3x "
                          "fund (leverage / 3 of the account) plus T-bills"},
        **body,
        "evidence": lab_evidence(Path(lab_path or LAB2_PATH)),
        "data": {"stale": data["stale"], "warning": data["warning"], "note": state_note},
        "notified": notified,
    }
    write_json_atomic(out_path, payload, indent=1)
    s = payload["signal"]
    progress(f"Nasdaq signal {payload['as_of']}: hold {s['held']:.2f}x, signal {s['new']:.2f}x"
             f"{' - TRADE DUE at the next close' if s['trade_due'] else ''}; paper since "
             f"{payload['paper']['start']}: {payload['paper']['return_pct']:+.2f}% "
             f"(QQQ {payload['paper']['qqq_return_pct']:+.2f}%)")
    return payload


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    unknown = [a for a in argv if a not in ("--offline", "--no-notify")]
    if unknown:
        raise SystemExit(f"unknown option(s): {' '.join(unknown)}")
    run_leverage_signal(offline="--offline" in argv, notify="--no-notify" not in argv)


if __name__ == "__main__":
    main()
