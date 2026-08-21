"""Fundamental data via Yahoo Finance, cached locally, scored 0-100.

Fundamentals are fetched only for tickers that already passed the technical
screen, so a scan touches ~40 tickers per market instead of hundreds.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

FIELDS = [
    "shortName", "sector", "industry", "marketCap", "currency",
    "trailingPE", "forwardPE", "priceToBook",
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
    "returnOnEquity", "profitMargins", "operatingMargins", "grossMargins",
    "debtToEquity", "currentRatio", "freeCashflow",
    "heldPercentInsiders", "recommendationKey", "numberOfAnalystOpinions",
    "targetMeanPrice", "currentPrice",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS fundamentals (
    ticker TEXT PRIMARY KEY,
    fetched TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS earnings (
    ticker TEXT PRIMARY KEY,
    fetched TEXT NOT NULL,
    next_date TEXT
);
CREATE TABLE IF NOT EXISTS dividends (
    ticker TEXT PRIMARY KEY,
    fetched TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


class FundamentalStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def get(self, ticker: str, max_age_days: float = 7.0) -> dict | None:
        row = self.conn.execute(
            "SELECT fetched, payload FROM fundamentals WHERE ticker = ?", (ticker,)
        ).fetchone()
        if row is None:
            return None
        fetched = datetime.fromisoformat(row[0])
        if datetime.utcnow() - fetched > timedelta(days=max_age_days):
            return None
        return json.loads(row[1])

    def put(self, ticker: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO fundamentals VALUES (?,?,?)",
            (ticker, datetime.utcnow().isoformat(), json.dumps(payload)),
        )
        self.conn.commit()


def fetch_next_earnings(store: FundamentalStore, ticker: str,
                        max_age_days: float = 3.0) -> str | None:
    """Next scheduled earnings date (ISO string) or None. Cached a few days."""
    row = store.conn.execute(
        "SELECT fetched, next_date FROM earnings WHERE ticker = ?", (ticker,)
    ).fetchone()
    if row is not None:
        fetched = datetime.fromisoformat(row[0])
        stale = datetime.utcnow() - fetched > timedelta(days=max_age_days)
        past = row[1] is not None and row[1] < datetime.utcnow().date().isoformat()
        if not stale and not past:
            return row[1]
    next_date = None
    try:
        import yfinance as yf

        cal = yf.Ticker(ticker).calendar or {}
        dates = cal.get("Earnings Date") or []
        today = datetime.utcnow().date()
        future = sorted(d for d in dates if d >= today)
        if future:
            next_date = future[0].isoformat()
    except Exception:
        pass
    store.conn.execute(
        "INSERT OR REPLACE INTO earnings VALUES (?,?,?)",
        (ticker, datetime.utcnow().isoformat(), next_date),
    )
    store.conn.commit()
    time.sleep(0.15)
    return next_date


def fetch_dividends(store: FundamentalStore, ticker: str,
                    max_age_days: float = 7.0) -> dict:
    """Dividend history summarized per calendar year (multiple payouts summed),
    plus the trailing-12-month total. {"yearly": {"2021": 36.0, ...}, "ttm": 54.0}"""
    row = store.conn.execute(
        "SELECT fetched, payload FROM dividends WHERE ticker = ?", (ticker,)
    ).fetchone()
    if row is not None:
        if datetime.utcnow() - datetime.fromisoformat(row[0]) <= timedelta(days=max_age_days):
            return json.loads(row[1])
    payload: dict = {"yearly": {}, "ttm": 0.0}
    try:
        import pandas as pd
        import yfinance as yf

        div = yf.Ticker(ticker).dividends
        if div is not None and len(div):
            div.index = div.index.tz_localize(None)
            this_year = datetime.utcnow().year
            by_year = div.groupby(div.index.year).sum()
            payload["yearly"] = {
                str(int(y)): round(float(v), 4)
                for y, v in by_year.items() if int(y) >= this_year - 5
            }
            cutoff = pd.Timestamp(datetime.utcnow() - timedelta(days=365))
            payload["ttm"] = round(float(div[div.index >= cutoff].sum()), 4)
    except Exception:
        pass
    store.conn.execute(
        "INSERT OR REPLACE INTO dividends VALUES (?,?,?)",
        (ticker, datetime.utcnow().isoformat(), json.dumps(payload)),
    )
    store.conn.commit()
    time.sleep(0.15)
    return payload


def dividend_summary(payload: dict, close: float) -> dict | None:
    """Shape dividend data for display; None if the stock pays nothing."""
    yearly = payload.get("yearly") or {}
    ttm = payload.get("ttm") or 0.0
    if not yearly and not ttm:
        return None
    this_year = str(datetime.utcnow().year)
    years = []
    for y in sorted(yearly):
        years.append({"year": y + (" (so far)" if y == this_year else ""),
                      "amount": yearly[y]})
    return {
        "yield_pct": round(ttm / close * 100, 2) if close > 0 and ttm else 0.0,
        "ttm": ttm,
        "yearly": years,
    }


def fetch_fundamentals(store: FundamentalStore, ticker: str) -> dict:
    cached = store.get(ticker)
    if cached is not None:
        return cached
    payload: dict = {}
    try:
        info = yf.Ticker(ticker).info or {}
        payload = {k: info.get(k) for k in FIELDS}
    except Exception as exc:
        payload = {"error": str(exc)[:200]}
    store.put(ticker, payload)
    time.sleep(0.2)  # be polite to Yahoo
    return payload


def _pts(value, anchors: list[tuple[float, float]]) -> float | None:
    """Piecewise-linear score from (value, points) anchors sorted by value."""
    if value is None or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v <= anchors[0][0]:
        return anchors[0][1]
    if v >= anchors[-1][0]:
        return anchors[-1][1]
    for (x1, y1), (x2, y2) in zip(anchors, anchors[1:]):
        if x1 <= v <= x2:
            return y1 + (y2 - y1) * (v - x1) / (x2 - x1)
    return None


def score_fundamentals(f: dict) -> tuple[float | None, list[dict]]:
    """Return (score 0-100 or None if too little data, component breakdown)."""
    if not f or f.get("error"):
        return None, []
    components = [
        ("Revenue growth", f.get("revenueGrowth"), [(-0.1, 0), (0.0, 30), (0.10, 60), (0.25, 90), (0.4, 100)], 0.20),
        ("Earnings growth", f.get("earningsGrowth") if f.get("earningsGrowth") is not None else f.get("earningsQuarterlyGrowth"),
         [(-0.2, 0), (0.0, 30), (0.15, 65), (0.35, 90), (0.6, 100)], 0.20),
        ("Return on equity", f.get("returnOnEquity"), [(0.0, 10), (0.08, 40), (0.15, 70), (0.25, 90), (0.35, 100)], 0.15),
        ("Profit margin", f.get("profitMargins"), [(-0.05, 0), (0.0, 25), (0.08, 55), (0.15, 75), (0.30, 100)], 0.15),
        ("Debt / equity", f.get("debtToEquity"), [(0, 100), (50, 85), (100, 60), (200, 30), (400, 0)], 0.10),
        ("Valuation (fwd P/E)", f.get("forwardPE"), [(5, 90), (12, 100), (20, 80), (35, 50), (60, 20), (100, 0)], 0.20),
    ]
    scored = []
    total_w = 0.0
    total = 0.0
    for label, value, anchors, weight in components:
        s = _pts(value, anchors)
        scored.append({"label": label, "value": value, "score": None if s is None else round(s)})
        if s is not None:
            total += s * weight
            total_w += weight
    if total_w < 0.5:  # too little data to judge
        return None, scored
    return round(total / total_w, 1), scored
