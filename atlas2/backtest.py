"""Walk-forward backtest of the pattern strategy over the cached price history.

For each ticker, slide a window through history; when a bullish pattern is
'forming', watch the next 15 bars for a real breakout above the entry level,
then simulate the trade: stop-loss, target, or 40-bar time exit.

Results (per pattern and per market): trades, win rate, average R multiple,
expectancy. Written to data/backtest_latest.json for the dashboard.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import PriceStore
from .indicators import add_indicators
from .patterns import BULLISH_DETECTORS

ROOT = Path(__file__).resolve().parent.parent
STEP = 5            # evaluate every 5th bar
WINDOW = 170        # bars of history each detector sees
BREAKOUT_WAIT = 15  # bars allowed for the breakout to happen
MAX_HOLD = 40       # time exit after this many bars in the trade


def simulate_ticker(df: pd.DataFrame) -> list[dict]:
    n = len(df)
    if n < WINDOW + 60:
        return []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    trades = []
    busy_until = 0
    seen: set[tuple] = set()
    for i in range(WINDOW, n - 5, STEP):
        if i < busy_until:
            continue
        win = df.iloc[i - WINDOW : i + 1]
        for det in BULLISH_DETECTORS:
            try:
                hits = det(win)
            except Exception:
                continue
            for h in hits:
                if h.status != "forming":
                    continue
                key = (h.pattern, round(h.entry, 1))
                if key in seen:
                    continue
                seen.add(key)
                # wait for breakout
                fill = None
                fill_i = None
                for j in range(i + 1, min(i + 1 + BREAKOUT_WAIT, n)):
                    if highs[j] >= h.entry:
                        fill = max(h.entry, opens[j])
                        if fill > h.entry * 1.03:  # gapped too far, skip
                            fill = None
                        fill_i = j
                        break
                    if lows[j] <= h.stop:  # pattern failed before triggering
                        break
                if fill is None or fill_i is None:
                    continue
                risk = fill - h.stop
                if risk <= 0:
                    continue
                exit_price = None
                exit_i = None
                outcome = None
                for j in range(fill_i, min(fill_i + MAX_HOLD, n)):
                    if lows[j] <= h.stop and (j > fill_i or opens[j] <= h.stop):
                        exit_price = min(opens[j], h.stop) if j > fill_i else h.stop
                        exit_i, outcome = j, "stop"
                        break
                    if highs[j] >= h.target:
                        exit_price = h.target
                        exit_i, outcome = j, "target"
                        break
                if exit_price is None:
                    exit_i = min(fill_i + MAX_HOLD, n - 1)
                    exit_price = closes[exit_i]
                    outcome = "time"
                r_multiple = (exit_price - fill) / risk
                trades.append({
                    "pattern": h.pattern,
                    "conf": h.confidence,
                    "signal_date": str(df.index[i])[:10],
                    "entry_date": str(df.index[fill_i])[:10],
                    "exit_date": str(df.index[exit_i])[:10],
                    "outcome": outcome,
                    "r": round(float(r_multiple), 3),
                    "ret_pct": round(float(exit_price / fill - 1) * 100, 2),
                    "hold_bars": int(exit_i - fill_i),
                })
                busy_until = exit_i + 1
                break
            if i < busy_until:
                break
    return trades


def summarize(trades: list[dict]) -> dict:
    by_pattern: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_pattern[t["pattern"]].append(t)
    out = {}
    for pat, ts in sorted(by_pattern.items(), key=lambda kv: -len(kv[1])):
        rs = [t["r"] for t in ts]
        wins = [r for r in rs if r > 0]
        out[pat] = {
            "trades": len(ts),
            "win_rate": round(100 * len(wins) / len(ts), 1),
            "avg_r": round(sum(rs) / len(ts), 2),
            "avg_ret_pct": round(sum(t["ret_pct"] for t in ts) / len(ts), 2),
            "avg_hold_bars": round(sum(t["hold_bars"] for t in ts) / len(ts), 1),
            "outcomes": {
                o: sum(1 for t in ts if t["outcome"] == o)
                for o in ("target", "stop", "time")
            },
        }
    rs = [t["r"] for t in trades]
    total = {
        "trades": len(trades),
        "win_rate": round(100 * sum(1 for r in rs if r > 0) / len(rs), 1) if rs else 0,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else 0,
    }
    return {"per_pattern": out, "total": total}


def run_backtest(markets: list[str], progress=print) -> dict:
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    results = {}
    every_trade: list[dict] = []
    for market in markets:
        uni_path = ROOT / "universe" / f"{market}.csv"
        if not uni_path.exists():
            continue
        tickers = pd.read_csv(uni_path)["ticker"].tolist()
        all_trades = []
        done = 0
        for t in tickers:
            df = store.load(t)
            if len(df) < WINDOW + 60:
                continue
            df = add_indicators(df)
            ticker_trades = simulate_ticker(df)
            for x in ticker_trades:
                x["ticker"] = t
                x["market"] = market
            all_trades.extend(ticker_trades)
            done += 1
            if done % 100 == 0:
                progress(f"[{market}] {done} tickers simulated, {len(all_trades)} trades so far")
        results[market] = summarize(all_trades)
        progress(f"[{market}] backtest: {results[market]['total']}")
        every_trade.extend(all_trades)
    store.close()
    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "available": True, "markets": results,
               "note": ("Simulated on cached daily history (~2 years). Entries on breakout above "
                        "pattern entry, exits at stop/target or after 40 bars. No slippage/fees.")}
    (ROOT / "data" / "backtest_latest.json").write_text(json.dumps(payload, indent=1))
    # full trade stream for portfolio-level simulation (python -m atlas2 portfolio)
    (ROOT / "data" / "backtest_trades.json").write_text(json.dumps(every_trade))
    progress(f"trade stream saved: {len(every_trade)} trades -> data/backtest_trades.json")
    return payload
