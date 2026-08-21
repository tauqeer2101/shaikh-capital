"""Portfolio-level simulation over the backtest trade stream.

Replays the decade as one account: signals arrive chronologically, at most
`max_pos` positions are open at once, each trade risks `risk_pct` of CURRENT
equity (so results compound), and when more signals arrive than slots exist the
highest-confidence setup wins — approximating a selective human trader.

Reports final equity, CAGR, max drawdown (on realized equity), win rate.
Run: python -m atlas2 portfolio
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def regime_multipliers() -> dict:
    """Per market, per date: 1.0 healthy / 0.5 caution / 0.0 downtrend —
    the same rules the live scanner applies via scan.market_regime."""
    from .data import BENCHMARKS, PriceStore
    from .indicators import add_indicators

    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    out: dict[str, dict[str, float]] = {}
    for m, bench in BENCHMARKS.items():
        df = store.load(bench)
        if len(df) < 210:
            continue
        df = add_indicators(df)
        mult: dict[str, float] = {}
        for idx, row in df.iterrows():
            d = str(idx)[:10]
            s50, s200 = row["sma50"], row["sma200"]
            if s200 != s200 or s50 != s50:  # NaN warm-up
                mult[d] = 1.0
                continue
            if row["close"] > s200 and s50 > s200 and row["pct_off_high"] > -8:
                mult[d] = 1.0
            elif row["close"] > s200:
                mult[d] = 0.5
            else:
                mult[d] = 0.0
        out[m] = mult
    store.close()
    return out


def _mult_for(regime: dict, market: str, day: str) -> float:
    table = regime.get(market)
    if not table:
        return 1.0
    if day in table:
        return table[day]
    # holiday mismatch: fall back to the most recent prior session
    prior = [d for d in table if d < day]
    return table[max(prior)] if prior else 1.0


def simulate_portfolio(trades: list[dict], start_capital: float = 25000.0,
                       risk_pct: float = 1.0, max_pos: int = 6,
                       min_conf: float = 0.0, cost_r: float = 0.05,
                       regime: dict | None = None) -> dict:
    """cost_r: friction (fees+slippage) charged per trade, in R units.
    regime: optional {market: {date: multiplier}} — 0 skips, 0.5 halves risk."""
    pool = [t for t in trades if t.get("conf", 1.0) >= min_conf
            and t.get("entry_date") and t.get("exit_date")]
    if not pool:
        return {"error": "no trades pass the filter"}
    entries_by_day: dict[str, list[dict]] = defaultdict(list)
    for t in pool:
        entries_by_day[t["entry_date"]].append(t)
    exits_by_day: dict[str, list[dict]] = defaultdict(list)

    equity = start_capital
    peak = start_capital
    max_dd = 0.0
    open_trades: list[dict] = []
    taken = wins = 0
    yearly: dict[str, float] = {}

    # unified timeline; each day exits are processed first (freeing slots), then entries
    timeline = sorted(set(entries_by_day) | {t["exit_date"] for t in pool})
    for day in timeline:
        # close positions exiting today
        still_open = []
        for pos in open_trades:
            if pos["trade"]["exit_date"] <= day:
                r_net = pos["trade"]["r"] - cost_r
                pnl = pos["risk"] * r_net
                equity += pnl
                if r_net > 0:
                    wins += 1
                year = pos["trade"]["exit_date"][:4]
                yearly[year] = yearly.get(year, 0.0) + pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak)
            else:
                still_open.append(pos)
        open_trades = still_open
        # open new positions, best confidence first
        todays = sorted(entries_by_day.get(day, []), key=lambda t: -t.get("conf", 0))
        for t in todays:
            if len(open_trades) >= max_pos:
                break
            if equity <= 0:
                break
            mult = _mult_for(regime, t.get("market", ""), day) if regime else 1.0
            if mult <= 0:
                continue  # market in downtrend: the system stands aside
            risk = equity * risk_pct / 100.0 * mult
            open_trades.append({"trade": t, "risk": risk})
            taken += 1
    # close anything left at its recorded exit
    for pos in open_trades:
        r_net = pos["trade"]["r"] - cost_r
        equity += pos["risk"] * r_net
        if r_net > 0:
            wins += 1

    first = min(t["entry_date"] for t in pool)
    last = max(t["exit_date"] for t in pool)
    years = max(0.5, (date.fromisoformat(last) - date.fromisoformat(first)).days / 365.25)
    cagr = (equity / start_capital) ** (1 / years) - 1 if equity > 0 else -1.0
    return {
        "risk_pct": risk_pct,
        "min_conf": min_conf,
        "max_pos": max_pos,
        "trades_taken": taken,
        "trades_available": len(pool),
        "win_rate": round(100 * wins / taken, 1) if taken else 0,
        "final_equity": round(equity),
        "cagr_pct": round(cagr * 100, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "years": round(years, 1),
        "yearly_pnl": {y: round(v) for y, v in sorted(yearly.items())},
    }


def run_portfolio_report(progress=print) -> list[dict]:
    path = ROOT / "data" / "backtest_trades.json"
    if not path.exists():
        progress("No trade stream found - run: python -m atlas2 backtest first")
        return []
    trades = json.loads(path.read_text())
    reg = regime_multipliers()
    progress(f"trade stream: {len(trades)} trades\n")
    profiles = [
        ("No regime filter (1% risk) — what trading every market blindly gives", dict(risk_pct=1.0)),
        ("SYSTEM AS DESIGNED (1% risk + regime filter)", dict(risk_pct=1.0, regime=reg)),
        ("Balanced (1.5% risk + regime filter)", dict(risk_pct=1.5, regime=reg)),
        ("Aggressive (2% risk + regime filter)", dict(risk_pct=2.0, regime=reg)),
        ("Very aggressive (3% risk + regime filter)", dict(risk_pct=3.0, regime=reg)),
    ]
    results = []
    for label, kw in profiles:
        r = simulate_portfolio(trades, **kw)
        r["label"] = label
        results.append(r)
        progress(f"{label}\n  CAGR {r['cagr_pct']}%/yr | max drawdown {r['max_drawdown_pct']}% | "
                 f"final €{r['final_equity']:,} from €25,000 in {r['years']}y | "
                 f"{r['trades_taken']} trades | win {r['win_rate']}%")
        progress(f"  yearly P&L: " + ", ".join(f"{y}: {v:+,}" for y, v in r["yearly_pnl"].items()) + "\n")
    (ROOT / "data" / "portfolio_sim.json").write_text(json.dumps(results, indent=1))
    return results
