"""Price data layer: Yahoo Finance batch download with a local SQLite cache."""
from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

BENCHMARKS = {"us": "^GSPC", "de": "^GDAXI", "in": "^NSEI"}
HISTORY_DAYS = 3700  # ~10 years of daily bars; grows further over time as scans accumulate


def utcnow() -> datetime:
    """Naive UTC now - the format the caches store (datetime.utcnow() is deprecated)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
        # wait up to 30 s for a writer instead of failing with "database is locked"
        # (the scan, the dashboard's analyse view and the monitor share this file)
        self.conn = sqlite3.connect(db_path, timeout=30)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def stale_tickers(self, tickers: list[str], max_age_hours: float = 12.0) -> list[str]:
        cutoff = (utcnow() - timedelta(hours=max_age_hours)).isoformat()
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
            (ticker, utcnow().isoformat()),
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


def _frame_for(data, ticker: str, batch: list[str]):
    """One ticker's bars out of a yf.download result (None when it is not there)."""
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            df = data[ticker]
        elif ticker in data.columns.get_level_values(-1):
            df = data.xs(ticker, axis=1, level=-1)
        else:
            return None
    elif len(batch) == 1:
        df = data
    else:                       # flat columns for several tickers: not attributable
        return None
    df = df.dropna(how="all")
    return df if not df.empty else None


def refresh_prices(
    store: PriceStore,
    tickers: list[str],
    batch_size: int = 50,
    max_age_hours: float = 12.0,
    progress=None,
    retry_batch_size: int = 10,
    retry_pause_s: float = 5.0,
    download=None,
) -> dict[str, int]:
    """Download missing/stale tickers in batches. Returns {ticker: n_bars}.

    A batch that fails, and every ticker that came back empty, is tried once more
    in small batches after a short pause (Yahoo drops parts of large requests when
    it is busy); what is still missing then keeps its cached bars, stays stale and
    is tried again on the next run. download(batch) replaces yf.download in tests."""
    todo = store.stale_tickers(tickers, max_age_hours)
    results: dict[str, int] = {}
    start = (date.today() - timedelta(days=HISTORY_DAYS)).isoformat()
    if download is None:
        def download(batch):
            return yf.download(batch, start=start, interval="1d", auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)

    def fetch(batch: list[str]) -> list[str]:
        try:
            data = download(batch)
        except Exception as exc:
            print(f"WARN batch download failed ({batch[0]}..): {exc}")
            return list(batch)
        missing = []
        for t in batch:
            df = _frame_for(data, t, batch)
            results[t] = store.upsert(t, df) if df is not None else 0
            if not results[t]:
                missing.append(t)
        return missing

    retry: list[str] = []
    for i in range(0, len(todo), batch_size):
        retry += fetch(todo[i: i + batch_size])
        if progress:
            progress(min(i + batch_size, len(todo)), len(todo))
    if retry:
        time.sleep(retry_pause_s)
        still: list[str] = []
        for i in range(0, len(retry), retry_batch_size):
            still += fetch(retry[i: i + retry_batch_size])
        print(f"retried {len(retry)} ticker(s) without prices: {len(retry) - len(still)} recovered"
              + (f"; still none for {len(still)} ({', '.join(still[:12])}"
                 f"{' ...' if len(still) > 12 else ''}) - their cached bars are used"
                 if still else ""))
    return results
