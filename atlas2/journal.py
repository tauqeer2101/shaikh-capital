"""Trade journal: log taken setups, track open positions, compute real stats."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    name TEXT DEFAULT '',
    market TEXT NOT NULL,
    currency TEXT NOT NULL,
    pattern TEXT DEFAULT '',
    grade TEXT DEFAULT '',
    entry_date TEXT NOT NULL,
    entry_price REAL NOT NULL,
    shares REAL NOT NULL,
    stop REAL NOT NULL,
    target REAL,
    exit_date TEXT,
    exit_price REAL,
    exit_reason TEXT,
    notes TEXT DEFAULT ''
);
"""


class Journal:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def add(self, t: dict) -> int:
        cur = self.conn.execute(
            """INSERT INTO trades (ticker, name, market, currency, pattern, grade,
               entry_date, entry_price, shares, stop, target, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                t["ticker"], t.get("name", ""), t["market"], t.get("currency", ""),
                t.get("pattern", ""), t.get("grade", ""),
                t.get("entry_date") or date.today().isoformat(),
                float(t["entry_price"]), float(t["shares"]), float(t["stop"]),
                t.get("target"), t.get("notes", ""),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_date: str | None = None, exit_reason: str = "manual",
                    notes: str | None = None) -> bool:
        cur = self.conn.execute(
            """UPDATE trades SET exit_price = ?, exit_date = ?, exit_reason = ?,
               notes = COALESCE(?, notes) WHERE id = ? AND exit_price IS NULL""",
            (float(exit_price), exit_date or date.today().isoformat(),
             exit_reason, notes, trade_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def update_stop(self, trade_id: int, stop: float) -> bool:
        cur = self.conn.execute(
            "UPDATE trades SET stop = ? WHERE id = ? AND exit_price IS NULL",
            (float(stop), trade_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete(self, trade_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def all_trades(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY exit_price IS NOT NULL, entry_date DESC, id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def _r_multiple(t: dict, price: float) -> float | None:
    risk = t["entry_price"] - t["stop"]
    if risk <= 0:
        return None
    return (price - t["entry_price"]) / risk


def enrich_and_summarize(trades: list[dict], last_close: dict[str, float]) -> dict:
    """Attach P&L to each trade (live close for open ones) and compute stats."""
    open_trades, closed = [], []
    for t in trades:
        if t["exit_price"] is None:
            price = last_close.get(t["ticker"])
            t["last_price"] = price
            if price is not None:
                t["pnl"] = round((price - t["entry_price"]) * t["shares"], 2)
                t["pnl_pct"] = round((price / t["entry_price"] - 1) * 100, 2)
                r = _r_multiple(t, price)
                t["r"] = None if r is None else round(r, 2)
            t["open_risk"] = round(max(0.0, (t["entry_price"] - t["stop"])) * t["shares"], 2)
            open_trades.append(t)
        else:
            t["pnl"] = round((t["exit_price"] - t["entry_price"]) * t["shares"], 2)
            t["pnl_pct"] = round((t["exit_price"] / t["entry_price"] - 1) * 100, 2)
            r = _r_multiple(t, t["exit_price"])
            t["r"] = None if r is None else round(r, 2)
            closed.append(t)

    rs = [t["r"] for t in closed if t["r"] is not None]
    wins = [r for r in rs if r > 0]
    by_ccy: dict[str, float] = {}
    for t in closed:
        by_ccy[t["currency"]] = round(by_ccy.get(t["currency"], 0.0) + t["pnl"], 2)
    stats = {
        "open_count": len(open_trades),
        "closed_count": len(closed),
        "win_rate": round(100 * len(wins) / len(rs), 1) if rs else None,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
        "total_r": round(sum(rs), 2) if rs else None,
        "realized_pnl_by_currency": by_ccy,
        "open_risk_by_currency": {},
    }
    for t in open_trades:
        c = t["currency"]
        stats["open_risk_by_currency"][c] = round(
            stats["open_risk_by_currency"].get(c, 0.0) + t["open_risk"], 2
        )
    return {"open": open_trades, "closed": closed, "stats": stats}
