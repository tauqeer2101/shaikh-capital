"""The owner's own strategy, automated: quality companies in a deep pullback.

Rules (as traded manually since June 2020):
  1. Fundamentally strong company (score >= threshold: growth, profits, balance sheet)
  2. Price 25-45% below its all-time high (all-time = highest high in our 10y store)
  3. A bullish chart pattern has formed (any of the standard detectors)
  4. Entry at the pattern breakout, stop under the pattern structure,
     TARGET = the old all-time high (the recovery thesis)
  5. Only take it when the reward:risk to the old high is >= 2

Run: python -m atlas2 comeback      (also runs automatically after every scan)
Backtest: python -m atlas2 comeback-test
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .data import BENCHMARKS, PriceStore, refresh_prices
from .fundamentals import (FundamentalStore, dividend_summary, fetch_dividends,
                           fetch_fundamentals, fetch_next_earnings,
                           score_fundamentals)
from .indicators import add_indicators, relative_strength_3m
from .patterns import BULLISH_DETECTORS, detect_all
from .scan import (MARKET_CCY, MIN_DOLLAR_VOL, ROOT, _fx_to_eur, load_config,
                   load_universe, market_regime, technical_score)

DD_MIN, DD_MAX = -0.45, -0.25   # how far below the all-time high qualifies
MIN_FUND_SCORE = 55.0
MIN_RR_TO_HIGH = 2.0


def scan_comeback(cfg: dict | None = None, progress=print) -> dict:
    cfg = cfg or load_config()
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    fstore = FundamentalStore(ROOT / "data" / "cache" / "fundamentals.sqlite3")

    fx_cache: dict[str, float] = {}
    regimes: dict[str, dict] = {}
    candidates = []
    for market in ("us", "de", "in"):
        try:
            universe = load_universe(market)
        except FileNotFoundError:
            continue
        names = dict(zip(universe["ticker"], universe["name"]))
        indices = dict(zip(universe["ticker"], universe["index"]))
        bench = store.load(BENCHMARKS[market])
        regimes[market] = market_regime(bench)
        for t in universe["ticker"]:
            df = store.load(t)
            if len(df) < 300:
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            if pd.isna(last["dollar_vol20"]) or last["dollar_vol20"] < MIN_DOLLAR_VOL[market]:
                continue
            ath = float(df["high"].max())
            close = float(last["close"])
            dd = close / ath - 1
            if not (DD_MIN <= dd <= DD_MAX):
                continue
            hits = [h for h in detect_all(df) if h.direction == "bullish"]
            if not hits:
                continue
            best = max(hits, key=lambda h: h.confidence)
            risk = best.entry - best.stop
            if risk <= 0:
                continue
            rr_to_high = (ath - best.entry) / risk
            if rr_to_high < MIN_RR_TO_HIGH:
                continue
            rs3m = relative_strength_3m(df, bench)
            tscore, treasons = technical_score(last, rs3m)
            candidates.append({
                "market": market, "ticker": t, "name": names.get(t, t),
                "index": indices.get(t, ""), "df": df, "last": last,
                "ath": ath, "dd": dd, "hit": best, "rr_to_high": rr_to_high,
                "rs3m": rs3m, "tscore": tscore, "treasons": treasons,
                "ath_date": str(df["high"].idxmax())[:10],
            })
    progress(f"[comeback] {len(candidates)} price-qualified candidates; checking fundamentals...")

    setups = []
    for c in sorted(candidates, key=lambda x: -x["hit"].confidence)[:80]:
        t = c["ticker"]
        market = c["market"]
        f = fetch_fundamentals(fstore, t)
        fscore, fcomponents = score_fundamentals(f)
        if fscore is None or fscore < MIN_FUND_SCORE:
            continue  # the strategy is quality-first: no quality, no trade
        ccy = MARKET_CCY[market]
        if ccy not in fx_cache:
            fx_cache[ccy] = _fx_to_eur(ccy)
        fx = fx_cache[ccy]
        regime = regimes.get(market, {"status": "unknown", "risk_multiplier": 1.0})
        capital_local = cfg["capital_eur"] * fx
        risk_local = capital_local * cfg["risk_pct"] / 100 * regime.get("risk_multiplier", 1.0)
        hit = c["hit"]
        entry, stop = hit.entry, hit.stop
        target = round(c["ath"], 4)  # the recovery thesis: back to the old high
        risk_per_share = entry - stop
        shares = int(risk_local / risk_per_share) if risk_per_share > 0 else 0
        max_pos_local = capital_local * cfg["max_position_pct"] / 100
        cost = shares * entry
        if cost > max_pos_local and entry > 0:
            shares = int(max_pos_local / entry)
            cost = shares * entry
        upside = (target / entry - 1) * 100
        composite = 0.45 * fscore + 0.35 * min(100, hit.confidence * 100) \
            + 0.20 * min(100, upside)
        grade = "A" if composite >= 68 else "B" if composite >= 55 else "C"
        earnings_date = fetch_next_earnings(fstore, t)
        earnings_in_days = None
        if earnings_date:
            earnings_in_days = (date.fromisoformat(earnings_date) - date.today()).days
        earnings_soon = (earnings_in_days is not None
                         and earnings_in_days <= cfg.get("earnings_warn_days", 7))
        last = c["last"]
        df = c["df"]
        reasons = [
            f"quality company {abs(c['dd'])*100:.0f}% below its all-time high of "
            f"{c['ath']:.2f} ({c['ath_date']}) - the recovery play",
            f"{hit.pattern.replace('_', ' ')} ({hit.status}): {hit.note}",
            f"fundamental quality score {fscore:.0f}/100",
            f"target = the old high: +{upside:.0f}% if it fully recovers "
            f"(reward:risk 1:{c['rr_to_high']:.1f})",
        ] + c["treasons"]
        if regime.get("risk_multiplier", 1.0) < 1.0:
            reasons.append(f"market regime '{regime['status']}': suggested size reduced")
        if earnings_soon:
            reasons.append(f"⚠ earnings in {earnings_in_days} day(s) ({earnings_date})")
        setups.append({
            "ticker": t, "name": c["name"], "market": market,
            "market_index": f"{c['index']}",
            "close": round(float(last["close"]), 2), "currency": ccy,
            "pattern": hit.pattern, "pattern_note": hit.note, "status": hit.status,
            "direction": "bullish",
            "entry": hit.entry, "stop": hit.stop, "target": target,
            "risk_reward": round(c["rr_to_high"], 2),
            "upside_pct": round(upside, 1),
            "risk_pct": round((1 - hit.stop / hit.entry) * 100, 1),
            "technical_score": round(c["tscore"], 1),
            "pattern_score": round(hit.confidence * 100, 1),
            "fundamental_score": fscore,
            "fundamental_components": fcomponents,
            "composite_score": round(composite, 1),
            "grade": grade,
            "conflicting_bearish": False,
            "earnings_date": earnings_date,
            "earnings_in_days": earnings_in_days,
            "earnings_soon": earnings_soon,
            "dividend": dividend_summary(fetch_dividends(fstore, t), float(last["close"])),
            "rs_vs_index_3m": None if c["rs3m"] is None else round(c["rs3m"], 1),
            "rsi14": round(float(last["rsi14"]), 1) if pd.notna(last["rsi14"]) else None,
            "pct_off_52w_high": round(float(last["pct_off_high"]), 1) if pd.notna(last["pct_off_high"]) else None,
            "avg_dollar_vol": round(float(last["dollar_vol20"])),
            "position": {
                "shares": shares,
                "cost_local": round(cost, 2),
                "cost_eur": round(cost / fx, 2),
                "risk_eur": round(shares * risk_per_share / fx, 2),
            },
            "reasons": reasons,
            "key_points": [[str(df.index[i])[:10], float(p)] for i, p in hit.key_points],
            "pattern_start": str(df.index[hit.start_idx])[:10],
            "sector": (f or {}).get("sector"),
            "other_patterns": [],
        })
    setups.sort(key=lambda s: s["composite_score"], reverse=True)
    result = {
        "market": "comeback",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "universe_size": sum(1 for _ in candidates),
        "with_data": len(candidates),
        "candidates": len(candidates),
        "benchmark": "-",
        "fx_eur_to_local": 1.0,
        "config": {k: cfg[k] for k in ("capital_eur", "risk_pct", "min_risk_reward")},
        "pattern_tuning": {"disabled": [], "source": "not applied (different strategy)"},
        "setups": setups,
        "warnings": [],
    }
    out_dir = ROOT / "data" / "scans"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comeback_latest.json").write_text(json.dumps(result, indent=1))
    (out_dir / f"comeback_{date.today().isoformat()}.json").write_text(json.dumps(result))
    store.close()
    progress(f"[comeback] done: {len(setups)} quality-recovery setups")
    return result


# ---------------------------------------------------------------- backtest ---

BT_STEP = 5
BT_WINDOW = 170
BT_WAIT = 15
BT_MAX_HOLD = 250   # the owner holds up to ~a year for the recovery


def backtest_comeback(markets: list[str] | None = None, progress=print) -> dict:
    """Decade test of the price-side rules: 25-45% below the (rolling) all-time
    high + bullish pattern breakout, target = that high, stop = pattern stop.
    NOTE: historical fundamentals aren't available, so the quality filter can't
    be applied retroactively - results cover the price mechanics only."""
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    markets = markets or ["us", "de", "in"]
    all_trades = []
    for market in markets:
        try:
            universe = load_universe(market)
        except FileNotFoundError:
            continue
        done = 0
        for t in universe["ticker"]:
            df = store.load(t)
            if len(df) < BT_WINDOW + 100:
                continue
            df = add_indicators(df)
            highs = df["high"].to_numpy()
            lows = df["low"].to_numpy()
            opens = df["open"].to_numpy()
            closes = df["close"].to_numpy()
            ath = df["high"].cummax().to_numpy()
            n = len(df)
            busy_until = 0
            seen = set()
            for i in range(BT_WINDOW, n - 5, BT_STEP):
                if i < busy_until:
                    continue
                dd = closes[i] / ath[i] - 1
                if not (DD_MIN <= dd <= DD_MAX):
                    continue
                win = df.iloc[i - BT_WINDOW : i + 1]
                for det in BULLISH_DETECTORS:
                    try:
                        hits = det(win)
                    except Exception:
                        continue
                    entered = False
                    for h in hits:
                        if h.status != "forming":
                            continue
                        key = (h.pattern, round(h.entry, 1))
                        if key in seen:
                            continue
                        seen.add(key)
                        target = float(ath[i])
                        risk = h.entry - h.stop
                        if risk <= 0 or (target - h.entry) / risk < MIN_RR_TO_HIGH:
                            continue
                        fill = fill_i = None
                        for j in range(i + 1, min(i + 1 + BT_WAIT, n)):
                            if highs[j] >= h.entry:
                                fill = max(h.entry, opens[j])
                                if fill > h.entry * 1.03:
                                    fill = None
                                fill_i = j
                                break
                            if lows[j] <= h.stop:
                                break
                        if fill is None:
                            continue
                        risk = fill - h.stop
                        exit_price = exit_i = outcome = None
                        for j in range(fill_i, min(fill_i + BT_MAX_HOLD, n)):
                            if lows[j] <= h.stop and (j > fill_i or opens[j] <= h.stop):
                                exit_price = min(opens[j], h.stop) if j > fill_i else h.stop
                                exit_i, outcome = j, "stop"
                                break
                            if highs[j] >= target:
                                exit_price, exit_i, outcome = target, j, "recovered"
                                break
                        if exit_price is None:
                            exit_i = min(fill_i + BT_MAX_HOLD, n - 1)
                            exit_price, outcome = closes[exit_i], "time"
                        all_trades.append({
                            "market": market, "ticker": t, "pattern": h.pattern,
                            "entry_date": str(df.index[fill_i])[:10],
                            "exit_date": str(df.index[exit_i])[:10],
                            "outcome": outcome,
                            "r": round(float((exit_price - fill) / risk), 3),
                            "ret_pct": round(float(exit_price / fill - 1) * 100, 2),
                            "hold_bars": int(exit_i - fill_i),
                            "conf": h.confidence,
                        })
                        busy_until = exit_i + 1
                        entered = True
                        break
                    if entered:
                        break
            done += 1
            if done % 150 == 0:
                progress(f"[{market}] {done} tickers, {len(all_trades)} trades so far")
        progress(f"[{market}] done")
    store.close()
    rs = [t["r"] for t in all_trades]
    rets = [t["ret_pct"] for t in all_trades]
    summary = {
        "trades": len(all_trades),
        "win_rate": round(100 * sum(1 for r in rs if r > 0) / len(rs), 1) if rs else 0,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else 0,
        "avg_ret_pct": round(sum(rets) / len(rets), 2) if rets else 0,
        "avg_hold_days": round(sum(t["hold_bars"] for t in all_trades) / len(all_trades), 0) if all_trades else 0,
        "outcomes": {o: sum(1 for t in all_trades if t["outcome"] == o)
                     for o in ("recovered", "stop", "time")},
        "avg_winner_ret": round(sum(t["ret_pct"] for t in all_trades if t["r"] > 0)
                                / max(1, sum(1 for t in all_trades if t["r"] > 0)), 1),
        "avg_loser_ret": round(sum(t["ret_pct"] for t in all_trades if t["r"] <= 0)
                               / max(1, sum(1 for t in all_trades if t["r"] <= 0)), 1),
        "per_market": {},
    }
    for m in markets:
        mts = [t for t in all_trades if t["market"] == m]
        if not mts:
            continue
        mrs = [t["r"] for t in mts]
        summary["per_market"][m] = {
            "trades": len(mts),
            "win_rate": round(100 * sum(1 for r in mrs if r > 0) / len(mrs), 1),
            "avg_r": round(sum(mrs) / len(mrs), 2),
            "avg_ret_pct": round(sum(t["ret_pct"] for t in mts) / len(mts), 2),
        }
    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "summary": summary, "trades": all_trades,
               "note": ("Price-side rules of the owner's manual strategy. Historical "
                        "fundamentals unavailable, so the quality filter is NOT applied "
                        "here; live scans do apply it.")}
    (ROOT / "data" / "comeback_backtest.json").write_text(json.dumps(payload))
    progress(f"comeback strategy backtest: {summary['trades']} trades, "
             f"win {summary['win_rate']}%, avg ret {summary['avg_ret_pct']}%, "
             f"avg R {summary['avg_r']}")
    return payload
