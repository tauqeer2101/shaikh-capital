"""Price data layer: Yahoo Finance batch download with a local SQLite cache."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

BENCHMARKS = {"us": "^GSPC", "de": "^GDAXI", "in": "^NSEI"}
HISTORY_DAYS = 3700  # ~10 years of daily bars; grows further over time as scans accumulate

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS meta (
    ticker TEXT PRIMARY KEY,
    last_fetch TEXT NOT NULL
);
"""


class PriceStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def stale_tickers(self, tickers: list[str], max_age_hours: float = 12.0) -> list[str]:
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()
        rows = self.conn.execute(
            "SELECT ticker FROM meta WHERE last_fetch >= ?", (cutoff,)
        ).fetchall()
        fresh = {r[0] for r in rows}
        return [t for t in tickers if t not in fresh]

    def upsert(self, ticker: str, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        rows = [
            (
                ticker,
                idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10],
                float(r["Open"]), float(r["High"]), float(r["Low"]),
                float(r["Close"]), float(r["Volume"]),
            )
            for idx, r in df.iterrows()
            if pd.notna(r["Close"]) and pd.notna(r["Open"])
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO meta VALUES (?,?)",
            (ticker, datetime.utcnow().isoformat()),
        )
        self.conn.commit()
        return len(rows)

    def load(self, ticker: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM bars "
            "WHERE ticker = ? ORDER BY date",
            self.conn,
            params=(ticker,),
            index_col="date",
            parse_dates=["date"],
        )
        return df

    def tickers_with_data(self, min_bars: int = 200) -> list[str]:
        rows = self.conn.execute(
            "SELECT ticker FROM bars GROUP BY ticker HAVING COUNT(*) >= ?", (min_bars,)
        ).fetchall()
        return [r[0] for r in rows]


def refresh_prices(
    store: PriceStore,
    tickers: list[str],
    batch_size: int = 50,
    max_age_hours: float = 12.0,
    progress=None,
) -> dict[str, int]:
    """Download missing/stale tickers in batches. Returns {ticker: n_bars}."""
    todo = store.stale_tickers(tickers, max_age_hours)
    results: dict[str, int] = {}
    start = (date.today() - timedelta(days=HISTORY_DAYS)).isoformat()
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        try:
            data = yf.download(
                batch,
                start=start,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            print(f"WARN batch download failed ({batch[0]}..): {exc}")
            continue
        for t in batch:
            if isinstance(data.columns, pd.MultiIndex):
                if t in data.columns.get_level_values(0):
                    df = data[t]
                elif t in data.columns.get_level_values(-1):
                    df = data.xs(t, axis=1, level=-1)
                else:
                    results[t] = 0
                    continue
            else:
                df = data
            df = df.dropna(how="all")
            results[t] = store.upsert(t, df)
        if progress:
            progress(min(i + batch_size, len(todo)), len(todo))
    return results
