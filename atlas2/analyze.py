"""On-demand single-stock analysis for the dashboard search feature.

Search covers the scan universe first (ticker or company name); anything not in
it falls back to Yahoo's symbol search. Analysis runs the same pipeline as the
scanner — indicators, technical score, every pattern detector, fundamentals —
for one ticker, fetching fresh prices if the cache is stale.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data import BENCHMARKS, PriceStore, refresh_prices
from .fundamentals import (FundamentalStore, dividend_summary, fetch_dividends,
                           fetch_fundamentals, score_fundamentals)
from .indicators import add_indicators, relative_strength_3m
from .patterns import detect_all
from .scan import technical_score

ROOT = Path(__file__).resolve().parent.parent
MARKET_LABEL = {"us": "US", "de": "Germany", "in": "India"}


def _market_for(ticker: str) -> str:
    if ticker.endswith(".DE"):
        return "de"
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "in"
    return "us"


@lru_cache(maxsize=1)
def _universe_rows() -> list[dict]:
    rows = []
    for market in ("us", "de", "in"):
        path = ROOT / "universe" / f"{market}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "ticker": str(r["ticker"]),
                "name": str(r["name"]),
                "market": market,
                "index": str(r["index"]),
            })
    return rows


def search_stocks(query: str, limit: int = 12) -> list[dict]:
    q = query.strip().lower()
    if len(q) < 1:
        return []
    starts, contains = [], []
    for r in _universe_rows():
        tick = r["ticker"].lower()
        name = r["name"].lower()
        if tick.startswith(q) or name.startswith(q):
            starts.append(r)
        elif q in tick or q in name:
            contains.append(r)
    results = (starts + contains)[:limit]
    if results:
        return results
    # Not in the universe: ask Yahoo (equities only)
    try:
        import yfinance as yf

        quotes = yf.Search(query, max_results=limit).quotes
        return [
            {
                "ticker": str(x.get("symbol")),
                "name": str(x.get("shortname") or x.get("longname") or x.get("symbol")),
                "market": _market_for(str(x.get("symbol"))),
                "index": str(x.get("exchDisp") or x.get("exchange") or ""),
            }
            for x in quotes
            if x.get("quoteType") == "EQUITY" and x.get("symbol")
        ]
    except Exception:
        return []


def analyze_ticker(ticker: str) -> dict:
    ticker = ticker.strip().upper() if not any(c.islower() for c in ticker) else ticker.strip()
    market = _market_for(ticker)
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    try:
        refresh_prices(store, [ticker, BENCHMARKS[market]], max_age_hours=12.0)
        df = store.load(ticker)
        bench = store.load(BENCHMARKS[market])
    finally:
        store.close()
    if len(df) < 60:
        return {"error": f"Not enough price history for {ticker} "
                         f"({len(df)} daily bars — need 60+)."}

    df = add_indicators(df)
    last = df.iloc[-1]
    rs3m = relative_strength_3m(df, bench)
    tscore, reasons = technical_score(last, rs3m)

    hits = detect_all(df)
    patterns = [
        {
            "pattern": h.pattern,
            "direction": h.direction,
            "status": h.status,
            "entry": h.entry,
            "stop": h.stop,
            "target": h.target,
            "risk_reward": h.risk_reward,
            "confidence": h.confidence,
            "note": h.note,
            "key_points": [[str(df.index[i])[:10], float(p)] for i, p in h.key_points],
            "start": str(df.index[h.start_idx])[:10],
        }
        for h in sorted(hits, key=lambda h: -h.confidence)
    ]
    best_bullish = next((p for p in patterns if p["direction"] == "bullish"), None)

    fstore = FundamentalStore(ROOT / "data" / "cache" / "fundamentals.sqlite3")
    f = fetch_fundamentals(fstore, ticker)
    fscore, fcomponents = score_fundamentals(f)

    close = float(last["close"])
    sma50 = float(last["sma50"]) if pd.notna(last["sma50"]) else None
    sma200 = float(last["sma200"]) if pd.notna(last["sma200"]) else None
    if sma200 and sma50 and close > sma50 and sma50 > sma200:
        trend = "strong uptrend"
    elif sma200 and close > sma200:
        trend = "uptrend"
    elif sma200 and close < sma200 and sma50 and close < sma50:
        trend = "downtrend"
    else:
        trend = "sideways / mixed"

    name = next(
        (r["name"] for r in _universe_rows() if r["ticker"] == ticker), None
    ) or str((f or {}).get("shortName") or ticker).strip()
    return {
        "ticker": ticker,
        "name": name,
        "market": market,
        "market_label": MARKET_LABEL[market],
        "currency": (f or {}).get("currency") or {"us": "USD", "de": "EUR", "in": "INR"}[market],
        "sector": (f or {}).get("sector"),
        "industry": (f or {}).get("industry"),
        "close": round(close, 2),
        "as_of": str(df.index[-1])[:10],
        "trend": trend,
        "technical_score": round(tscore, 1),
        "technical_reasons": reasons,
        "rsi14": round(float(last["rsi14"]), 1) if pd.notna(last["rsi14"]) else None,
        "sma50": None if sma50 is None else round(sma50, 2),
        "sma200": None if sma200 is None else round(sma200, 2),
        "pct_off_52w_high": round(float(last["pct_off_high"]), 1) if pd.notna(last["pct_off_high"]) else None,
        "ret_1m": round(float(last["ret_1m"]), 1) if pd.notna(last["ret_1m"]) else None,
        "ret_3m": round(float(last["ret_3m"]), 1) if pd.notna(last["ret_3m"]) else None,
        "ret_6m": round(float(last["ret_6m"]), 1) if pd.notna(last["ret_6m"]) else None,
        "rs_vs_index_3m": None if rs3m is None else round(rs3m, 1),
        "avg_dollar_vol": round(float(last["dollar_vol20"])) if pd.notna(last["dollar_vol20"]) else None,
        "patterns": patterns,
        "best_bullish": best_bullish,
        "fundamental_score": fscore,
        "fundamental_components": fcomponents,
        "dividend": dividend_summary(fetch_dividends(fstore, ticker), close),
        "market_cap": (f or {}).get("marketCap"),
        "analyst": {
            "recommendation": (f or {}).get("recommendationKey"),
            "count": (f or {}).get("numberOfAnalystOpinions"),
            "target_mean": (f or {}).get("targetMeanPrice"),
        },
    }
