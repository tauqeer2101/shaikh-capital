"""Market scan orchestrator.

Pipeline per market:
  universe -> price refresh -> technical screen -> pattern detection
  -> fundamentals (shortlist only) -> composite score -> ranked setups JSON

Outputs data/scans/{market}_latest.json plus a dated history copy.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from .data import BENCHMARKS, PriceStore, refresh_prices
from .fundamentals import (FundamentalStore, dividend_summary, fetch_dividends,
                           fetch_fundamentals, fetch_next_earnings,
                           score_fundamentals)
from .indicators import add_indicators, relative_strength_3m
from .patterns import detect_all

ROOT = Path(__file__).resolve().parent.parent
MIN_DOLLAR_VOL = {"us": 5e6, "de": 1e6, "in": 5e7}  # in local currency
FX_TICKERS = {"USD": "EURUSD=X", "INR": "EURINR=X", "EUR": None}
MARKET_CCY = {"us": "USD", "de": "EUR", "in": "INR"}


def load_config() -> dict:
    cfg_path = ROOT / "config.json"
    defaults = {
        "capital_eur": 25000,
        "risk_pct": 1.0,
        "min_risk_reward": 1.8,
        "max_position_pct": 25.0,
        "shortlist_size": 45,
        "markets": ["us", "de", "in"],
        "regime_filter": True,
        "earnings_warn_days": 7,
        "dividend_max_age_days": 3,
    }
    if cfg_path.exists():
        defaults.update(json.loads(cfg_path.read_text()))
    return defaults


def load_pattern_tuning(market: str, cfg: dict) -> tuple[set[str], dict, str]:
    """Per-market pattern tuning from the latest backtest.

    Returns (disabled_patterns, {pattern: backtest_stats}, source_note).
    A pattern is auto-disabled for a market when its backtest average R is below
    `min_avg_r` over at least `min_trades` trades; manual entries in
    config.pattern_tuning.disabled[market] are always disabled.
    """
    tuning = cfg.get("pattern_tuning") or {}
    disabled = set(str(p) for p in (tuning.get("disabled") or {}).get(market, []))
    stats: dict = {}
    source = "no backtest available"
    bt_path = ROOT / "data" / "backtest_latest.json"
    if bt_path.exists():
        try:
            bt = json.loads(bt_path.read_text())
            stats = bt.get("markets", {}).get(market, {}).get("per_pattern", {})
            source = f"backtest {bt.get('generated', '')[:10]}"
            if tuning.get("auto_disable", True):
                for pat, s in stats.items():
                    if (s.get("trades", 0) >= tuning.get("min_trades", 30)
                            and s.get("avg_r", 0) < tuning.get("min_avg_r", 0.0)):
                        disabled.add(pat)
        except Exception:
            pass
    return disabled, stats, source


def load_universe(market: str) -> pd.DataFrame:
    path = ROOT / "universe" / f"{market}.csv"
    if not path.exists():
        raise FileNotFoundError(f"universe/{market}.csv missing - run: python -m atlas2 universe")
    return pd.read_csv(path)


def _fx_to_eur(currency: str) -> float:
    """Local-currency units per EUR (e.g. USD -> ~1.08)."""
    fx = FX_TICKERS.get(currency)
    if fx is None:
        return 1.0
    try:
        df = yf.download(fx, period="5d", interval="1d", progress=False, auto_adjust=True)
        return float(df["Close"].iloc[-1].item())
    except Exception:
        return {"USD": 1.10, "INR": 96.0}.get(currency, 1.0)


def market_regime(bench: pd.DataFrame) -> dict:
    """Health of the overall market index: healthy / caution / downtrend.

    Breakout setups fail far more often in weak markets, so the scanner cuts
    suggested risk in 'caution' and recommends standing aside in 'downtrend'.
    """
    if len(bench) < 210:
        return {"status": "unknown", "note": "not enough benchmark history",
                "risk_multiplier": 1.0}
    b = add_indicators(bench)
    last = b.iloc[-1]
    close = float(last["close"])
    sma50 = float(last["sma50"])
    sma200 = float(last["sma200"])
    pct_off = float(last["pct_off_high"])
    above200 = close > sma200
    above50 = close > sma50
    golden = sma50 > sma200
    if above200 and golden and pct_off > -8:
        status, mult = "healthy", 1.0
        note = "index above 50/200-day averages and near its highs - normal trading"
    elif above200:
        status, mult = "caution", 0.5
        note = "index above its 200-day average but under pressure - half size, A-grades only"
    else:
        status, mult = "downtrend", 0.0
        note = "index below its 200-day average - standing aside is recommended; watch, don't chase"
    return {
        "status": status,
        "note": note,
        "risk_multiplier": mult,
        "index_close": round(close, 2),
        "vs_sma200_pct": round((close / sma200 - 1) * 100, 1),
        "above_sma50": above50,
        "pct_off_52w_high": round(pct_off, 1),
    }


def technical_score(row: pd.Series, rs3m: float | None) -> tuple[float, list[str]]:
    """Score 0-100 from trend / momentum / positioning. Returns (score, reasons)."""
    score = 0.0
    reasons = []
    close, sma50, sma200 = row.get("close"), row.get("sma50"), row.get("sma200")
    if pd.notna(sma200) and close > sma200:
        score += 20
        reasons.append("above 200-day average (long-term uptrend)")
    if pd.notna(sma50) and close > sma50:
        score += 15
        reasons.append("above 50-day average")
    if pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200:
        score += 15
        reasons.append("50-day above 200-day (golden alignment)")
    rsi = row.get("rsi14")
    if pd.notna(rsi):
        if 45 <= rsi <= 70:
            score += 15
            reasons.append(f"RSI {rsi:.0f} - healthy momentum, not overbought")
        elif 35 <= rsi < 45 or 70 < rsi <= 78:
            score += 7
    if rs3m is not None:
        if rs3m > 5:
            score += 20
            reasons.append(f"beating its index by {rs3m:.0f}% points over 3 months")
        elif rs3m > 0:
            score += 12
            reasons.append("outperforming its index over 3 months")
    pct_off = row.get("pct_off_high")
    if pd.notna(pct_off) and pct_off > -20:
        score += 15 * (1 - abs(pct_off) / 20)
        if pct_off > -8:
            reasons.append(f"only {abs(pct_off):.0f}% below its 52-week high")
    return min(score, 100.0), reasons


def scan_market(market: str, cfg: dict, progress=print) -> dict:
    universe = load_universe(market)
    tickers = universe["ticker"].tolist()
    names = dict(zip(universe["ticker"], universe["name"]))
    indices = dict(zip(universe["ticker"], universe["index"]))

    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    fstore = FundamentalStore(ROOT / "data" / "cache" / "fundamentals.sqlite3")

    progress(f"[{market}] refreshing prices for {len(tickers)} tickers + benchmark...")
    refresh_prices(store, tickers + [BENCHMARKS[market]],
                   progress=lambda done, total: progress(f"[{market}]   {done}/{total} downloaded"))
    bench = store.load(BENCHMARKS[market])
    regime = market_regime(bench) if cfg.get("regime_filter", True) else {
        "status": "off", "note": "regime filter disabled", "risk_multiplier": 1.0}
    progress(f"[{market}] market regime: {regime['status']} - {regime['note']}")

    disabled_patterns, bt_stats, bt_source = load_pattern_tuning(market, cfg)
    if disabled_patterns:
        progress(f"[{market}] patterns disabled by tuning ({bt_source}): "
                 f"{', '.join(sorted(disabled_patterns))}")

    ccy = MARKET_CCY[market]
    fx = _fx_to_eur(ccy)
    capital_local = cfg["capital_eur"] * fx
    risk_local = capital_local * cfg["risk_pct"] / 100
    max_pos_local = capital_local * cfg["max_position_pct"] / 100

    candidates = []
    warnings = []
    skipped = 0
    for t in tickers:
        df = store.load(t)
        if len(df) < 120:
            skipped += 1
            continue
        df = add_indicators(df)
        last = df.iloc[-1]
        if pd.isna(last["dollar_vol20"]) or last["dollar_vol20"] < MIN_DOLLAR_VOL[market]:
            continue
        rs3m = relative_strength_3m(df, bench)
        tscore, reasons = technical_score(last, rs3m)
        hits = detect_all(df)
        bullish = [h for h in hits
                   if h.direction == "bullish" and h.pattern not in disabled_patterns]
        bearish = [h for h in hits if h.direction == "bearish"]
        for h in bearish:
            if h.confidence >= 0.55:
                warnings.append({
                    "ticker": t, "name": names.get(t, t), "pattern": h.pattern,
                    "status": h.status, "breakdown_level": h.stop if False else h.entry,
                    "note": h.note, "close": round(float(last["close"]), 2),
                })
        if not bullish or tscore < 40:
            continue
        best = max(bullish, key=lambda h: (h.confidence, h.risk_reward or 0))
        rr = best.risk_reward
        if rr is None or rr < cfg["min_risk_reward"]:
            continue
        candidates.append({
            "ticker": t, "df_len": len(df), "tscore": tscore, "reasons": reasons,
            "hit": best, "rs3m": rs3m, "last": last, "all_bullish": bullish,
            # pattern anchor points as (date, price) so the dashboard can draw
            # the detected pattern on the chart
            "key_dates": [[str(df.index[i])[:10], float(p)] for i, p in best.key_points],
            "start_date": str(df.index[best.start_idx])[:10],
        })

    warned_tickers = {w["ticker"] for w in warnings}
    progress(f"[{market}] {len(candidates)} technical candidates; fetching fundamentals for top {cfg['shortlist_size']}...")
    candidates.sort(key=lambda c: (c["hit"].confidence * 50 + c["tscore"]), reverse=True)
    shortlist = candidates[: cfg["shortlist_size"]]

    regime_risk = risk_local * regime["risk_multiplier"]

    setups = []
    for c in shortlist:
        t = c["ticker"]
        f = fetch_fundamentals(fstore, t)
        fscore, fcomponents = score_fundamentals(f)
        earnings_date = fetch_next_earnings(fstore, t)
        earnings_in_days = None
        if earnings_date:
            earnings_in_days = (date.fromisoformat(earnings_date) - date.today()).days
        earnings_soon = (earnings_in_days is not None
                         and earnings_in_days <= cfg.get("earnings_warn_days", 7))
        hit = c["hit"]
        pattern_score = hit.confidence * 100
        rr = hit.risk_reward or 0
        pattern_score = min(100, pattern_score + max(0, (rr - 2) * 5))
        # Nudge the score by this pattern's real backtest record in this market
        bt = bt_stats.get(hit.pattern)
        bt_note = None
        if bt and bt.get("trades", 0) >= 30:
            pattern_score = max(0, min(100, pattern_score + max(-10, min(10, bt["avg_r"] * 20))))
            bt_note = (f"this pattern's {market.upper()} backtest: {bt['win_rate']}% wins, "
                       f"{bt['avg_r']:+.2f} avg R over {bt['trades']} trades")
        if fscore is None:
            composite = 0.55 * c["tscore"] + 0.45 * pattern_score
        else:
            composite = 0.40 * c["tscore"] + 0.35 * pattern_score + 0.25 * fscore
        conflicted = t in warned_tickers
        if conflicted:
            composite -= 10
        last = c["last"]
        entry, stop = hit.entry, hit.stop
        risk_per_share = entry - stop
        shares = int(regime_risk / risk_per_share) if risk_per_share > 0 else 0
        cost = shares * entry
        if cost > max_pos_local and entry > 0:
            shares = int(max_pos_local / entry)
            cost = shares * entry
        grade = "A" if composite >= 72 else "B" if composite >= 58 else "C"
        reasons = list(c["reasons"])
        reasons.insert(0, f"{hit.pattern.replace('_', ' ')} ({hit.status}): {hit.note}")
        if fscore is not None:
            reasons.append(f"fundamental quality score {fscore:.0f}/100")
        if bt_note:
            reasons.append(bt_note)
        if earnings_soon:
            reasons.append(f"⚠ earnings in {earnings_in_days} day(s) ({earnings_date}) - "
                           "gap risk: skip or size down, stops don't protect against gaps")
        if regime["risk_multiplier"] < 1.0:
            reasons.append(f"market regime '{regime['status']}': suggested size reduced "
                           f"to {regime['risk_multiplier']:.0%} of normal risk")
        if conflicted:
            reasons.append("caution: a bearish pattern is also present on this chart (see warnings)")
        setups.append({
            "ticker": t,
            "name": names.get(t, t),
            "market_index": indices.get(t, ""),
            "close": round(float(last["close"]), 2),
            "currency": ccy,
            "pattern": hit.pattern,
            "pattern_note": hit.note,
            "status": hit.status,
            "direction": "bullish",
            "entry": hit.entry,
            "stop": hit.stop,
            "target": hit.target,
            "risk_reward": rr,
            "upside_pct": round((hit.target / hit.entry - 1) * 100, 1),
            "risk_pct": round((1 - hit.stop / hit.entry) * 100, 1),
            "technical_score": round(c["tscore"], 1),
            "pattern_score": round(pattern_score, 1),
            "fundamental_score": fscore,
            "fundamental_components": fcomponents,
            "composite_score": round(composite, 1),
            "grade": grade,
            "conflicting_bearish": conflicted,
            "earnings_date": earnings_date,
            "earnings_in_days": earnings_in_days,
            "earnings_soon": earnings_soon,
            "dividend": dividend_summary(
                fetch_dividends(fstore, t, max_age_days=cfg.get("dividend_max_age_days", 3)),
                float(last["close"])),
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
            "key_points": c["key_dates"],
            "pattern_start": c["start_date"],
            "sector": (f or {}).get("sector"),
            "other_patterns": [h.pattern for h in c["all_bullish"] if h is not hit],
        })

    setups.sort(key=lambda s: s["composite_score"], reverse=True)
    result = {
        "market": market,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(tickers),
        "with_data": len(tickers) - skipped,
        "candidates": len(candidates),
        "benchmark": BENCHMARKS[market],
        "fx_eur_to_local": round(fx, 4),
        "config": {k: cfg[k] for k in ("capital_eur", "risk_pct", "min_risk_reward")},
        "pattern_tuning": {"disabled": sorted(disabled_patterns), "source": bt_source},
        "regime": regime,
        "setups": setups,
        "warnings": sorted(warnings, key=lambda w: w["ticker"]),
    }
    out_dir = ROOT / "data" / "scans"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{market}_latest.json").write_text(json.dumps(result, indent=1))
    (out_dir / f"{market}_{date.today().isoformat()}.json").write_text(json.dumps(result))
    store.close()
    progress(f"[{market}] done: {len(setups)} setups, {len(warnings)} bearish warnings")
    return result


def run_scan(markets: list[str] | None = None, progress=print) -> dict[str, dict]:
    cfg = load_config()
    markets = markets or cfg["markets"]
    results = {}
    for m in markets:
        results[m] = scan_market(m, cfg, progress)
    return results
