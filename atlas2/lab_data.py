"""Data for strategy lab 2: past S&P 500 members' prices, kept apart from the scanner.

The membership list (data/reference/sp500_ticker_start_end.csv, from the public
fja05680/sp500 dataset, MIT licence) gives every stint a ticker spent in the
S&P 500 since 1996. Their daily bars are downloaded from Yahoo Finance
(dividend- and split-adjusted, the same source and setting as the scanner) into
data/cache/lab_sp500.sqlite3 - NOT the scanner's price cache, so former members
never show up in a live scan. Yahoo has no prices for many companies that went
bankrupt or were bought; every ticker's outcome is logged in fetch_log so the
lab can report how much of the index it actually sees.

Run: python -m atlas2 lab-data [--refresh]   (slow on purpose: small batches
with pauses, so Yahoo does not rate-limit the owner's connection). The
membership list is downloaded first when it is missing (data/ is not in git).
"""
from __future__ import annotations

import csv
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MEMBERSHIP_PATH = ROOT / "data" / "reference" / "sp500_ticker_start_end.csv"
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "sp500_ticker_start_end.csv")
SP500_DB_PATH = ROOT / "data" / "cache" / "lab_sp500.sqlite3"
FETCH_START = "1994-01-01"       # two years before the first stint, for 12-month momentum
BATCH = 40
PAUSE_S = 4.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS fetch_log (
    ticker TEXT PRIMARY KEY,
    yahoo_ticker TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    first_date TEXT, last_date TEXT
);
"""


def load_membership(path: Path = MEMBERSHIP_PATH) -> list[dict]:
    """Stints as [{ticker, start, end}] (end None = still a member)."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            t = (r.get("ticker") or "").strip()
            s = (r.get("start_date") or "").strip()[:10]
            e = (r.get("end_date") or "").strip()[:10] or None
            if t and s:
                out.append({"ticker": t, "start": s, "end": e})
    return out


def ensure_membership(path: Path = MEMBERSHIP_PATH, *, fetch=None, progress=print) -> Path:
    """Download the membership list when it is not there yet (an existing file is
    never replaced). fetch(url) -> bytes replaces the download in tests."""
    path = Path(path)
    if path.exists():
        return path
    if fetch is None:
        import urllib.request

        def fetch(url):
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()

    progress(f"downloading the S&P 500 membership list into {path} ...")
    data = fetch(MEMBERSHIP_URL)
    if not data.startswith(b"ticker,start_date,end_date"):
        raise RuntimeError("the membership list did not download as expected")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def yahoo_symbol(ticker: str) -> str:
    """Membership ticker -> Yahoo symbol (share classes use '-': BRK.B -> BRK-B)."""
    return ticker.replace(".", "-")


def _frame_for(data: pd.DataFrame, sym: str) -> pd.DataFrame | None:
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        lv0 = data.columns.get_level_values(0)
        lv1 = data.columns.get_level_values(-1)
        if sym in lv0:
            df = data[sym]
        elif sym in lv1:
            df = data.xs(sym, axis=1, level=-1)
        else:
            return None
    else:
        df = data
    df = df.dropna(subset=[c for c in ("Open", "Close") if c in df.columns], how="any")
    return df if not df.empty else None


def fetch_members(db_path: Path = SP500_DB_PATH, membership_path: Path = MEMBERSHIP_PATH, *,
                  refresh: bool = False, batch: int = BATCH, pause_s: float = PAUSE_S,
                  progress=print, download=None) -> dict:
    """Download every member ticker not yet in fetch_log (all of them with
    refresh). Returns counts. download(symbols) -> DataFrame replaces yf.download
    in tests."""
    if download is None:
        import yfinance as yf

        def download(symbols):
            return yf.download(symbols, start=FETCH_START, interval="1d", auto_adjust=True,
                               progress=False, group_by="ticker", threads=True)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    tickers = sorted({s["ticker"] for s in load_membership(membership_path)})
    done = {r[0] for r in conn.execute("SELECT ticker FROM fetch_log")}
    todo = tickers if refresh else [t for t in tickers if t not in done]
    progress(f"{len(tickers)} member tickers, {len(todo)} to download into {db_path.name}")
    got = empty = 0
    for k in range(0, len(todo), batch):
        part = todo[k:k + batch]
        syms = [yahoo_symbol(t) for t in part]
        data = None
        for attempt in range(3):
            try:
                data = download(syms)
                break
            except Exception as exc:            # network hiccup: wait and retry
                progress(f"WARN batch {k // batch + 1} failed ({exc.__class__.__name__}); "
                         f"retry in {30 * (attempt + 1)} s")
                time.sleep(30 * (attempt + 1))
        if data is None:
            progress(f"WARN batch {k // batch + 1} skipped after 3 failures; run again later")
            continue
        now = datetime.now().isoformat(timespec="seconds")
        for t, sym in zip(part, syms):
            df = _frame_for(data, sym)
            rows = []
            if df is not None:
                for idx, r in df.iterrows():
                    rows.append((t, pd.Timestamp(idx).date().isoformat(), float(r["Open"]),
                                 float(r["High"]), float(r["Low"]), float(r["Close"]),
                                 float(r["Volume"]) if pd.notna(r["Volume"]) else 0.0))
            conn.execute("DELETE FROM bars WHERE ticker = ?", (t,))
            if rows:
                conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)", rows)
                got += 1
            else:
                empty += 1
            conn.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?,?)",
                         (t, sym, now, len(rows), rows[0][1] if rows else None,
                          rows[-1][1] if rows else None))
        conn.commit()
        progress(f"  batch {k // batch + 1}/{(len(todo) + batch - 1) // batch}: "
                 f"{got} with prices, {empty} without so far")
        if k + batch < len(todo):
            time.sleep(pause_s)
    conn.close()
    return {"tickers": len(tickers), "downloaded": len(todo), "with_prices": got,
            "without_prices": empty}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    unknown = [a for a in argv if a != "--refresh"]
    if unknown:
        raise SystemExit(f"unknown option(s): {' '.join(unknown)}")
    ensure_membership()
    print(fetch_members(refresh="--refresh" in argv))


if __name__ == "__main__":
    main()
