"""Local dashboard server: http://127.0.0.1:8899

Read-only. Serves the latest scan JSON per market and per-ticker chart data
(bars + moving averages + the detected pattern's levels for annotation).
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .data import PriceStore
from .indicators import add_indicators
from .journal import Journal, enrich_and_summarize

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Shaikh Capital")

# ---------------------------------------------------------------- auth ------
# A password in config.json ("dashboard_password") locks the whole dashboard
# behind a login page — required whenever the dashboard is reachable beyond
# this Mac (Tailscale Funnel / public URL). Remove the key to disable.
import hashlib

LOGIN_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Shaikh Capital — sign in</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body { margin:0; min-height:100vh; display:grid; place-items:center;
        background:#0c0e1a; color:#eef0fa;
        font:15px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
 .box { background:#141829; border:1px solid #242b49; border-radius:16px;
        padding:36px 40px; width:320px; text-align:center;
        box-shadow:0 24px 60px rgba(0,0,0,.5); }
 .logo { width:44px; height:44px; border-radius:12px; margin:0 auto 12px;
         background:linear-gradient(135deg,#8b5cf6,#ec4899); display:grid;
         place-items:center; font-weight:800; font-size:18px; color:#fff; }
 h1 { font-size:18px; margin:0 0 2px; } .tag { color:#9aa3c4; font-size:12px;
      margin-bottom:20px; }
 input { width:100%; box-sizing:border-box; background:#0c0e1a; color:#eef0fa;
         border:1px solid #242b49; border-radius:10px; padding:11px 12px;
         font-size:15px; margin-bottom:12px; }
 input:focus { outline:none; border-color:#8b5cf6; }
 button { width:100%; background:linear-gradient(135deg,#8b5cf6,#ec4899);
          color:#fff; border:none; border-radius:10px; padding:11px;
          font-size:15px; font-weight:600; cursor:pointer; }
 .msg { color:#f4527b; font-size:12.5px; min-height:16px; margin-bottom:6px; }
</style></head><body>
<form class="box" method="post" action="/login">
  <div class="logo">SC</div><h1>Shaikh Capital</h1>
  <div class="tag">trade smart</div>
  <div class="msg"><!--msg--></div>
  <input type="password" name="password" placeholder="Password" autofocus>
  <button type="submit">Sign in</button>
</form></body></html>"""


def _dashboard_password() -> str | None:
    try:
        cfg = json.loads((ROOT / "config.json").read_text())
        return cfg.get("dashboard_password") or None
    except Exception:
        return None


@app.middleware("http")
async def auth_middleware(request, call_next):
    from fastapi.responses import HTMLResponse, RedirectResponse

    pw = _dashboard_password()
    if not pw:
        return await call_next(request)
    token = hashlib.sha256(("sc:" + pw).encode()).hexdigest()
    if request.cookies.get("sc_auth") == token:
        return await call_next(request)
    if request.url.path == "/login":
        if request.method == "POST":
            form = await request.form()
            if form.get("password") == pw:
                resp = RedirectResponse("/", status_code=303)
                resp.set_cookie("sc_auth", token, max_age=180 * 24 * 3600,
                                httponly=True, samesite="lax")
                return resp
            return HTMLResponse(
                LOGIN_HTML.replace("<!--msg-->", "Wrong password — try again."),
                status_code=401)
        return HTMLResponse(LOGIN_HTML)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": "authentication required"}, status_code=401)
    return RedirectResponse("/login", status_code=303)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/scan/{market}")
def scan(market: str):
    path = ROOT / "data" / "scans" / f"{market}_latest.json"
    if not path.exists():
        raise HTTPException(404, f"No scan yet for {market}. Run: python -m atlas2 scan {market}")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/scans")
def scans_available():
    out = {}
    for m in ("us", "de", "in", "comeback"):
        path = ROOT / "data" / "scans" / f"{m}_latest.json"
        if path.exists():
            d = json.loads(path.read_text())
            top = d["setups"][0] if d["setups"] else None
            out[m] = {
                "generated": d["generated"],
                "setups": len(d["setups"]),
                "a_grades": sum(1 for s in d["setups"] if s["grade"] == "A"),
                "warnings": len(d["warnings"]),
                "regime": d.get("regime", {}).get("status", "unknown"),
                "top": None if top is None else {
                    "ticker": top["ticker"], "name": top["name"],
                    "grade": top["grade"], "score": top["composite_score"],
                    "pattern": top["pattern"]},
            }
    return out


@app.get("/api/price/{ticker}")
def latest_price(ticker: str):
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    row = store.conn.execute(
        "SELECT date, close FROM bars WHERE ticker = ? ORDER BY date DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    store.close()
    if row is None:
        raise HTTPException(404, f"no cached price for {ticker}")
    return {"ticker": ticker, "date": row[0], "price": round(float(row[1]), 4)}


_news_cache: dict[str, tuple[float, list]] = {}


@app.get("/api/news/{ticker}")
def news(ticker: str, limit: int = 6):
    import time as _time

    cached = _news_cache.get(ticker)
    if cached and _time.time() - cached[0] < 1800:
        return {"news": cached[1][:limit]}
    items = []
    try:
        import yfinance as yf

        for raw in (yf.Ticker(ticker).news or [])[:limit]:
            c = raw.get("content") or {}
            title = c.get("title")
            if not title:
                continue
            items.append({
                "title": title,
                "summary": (c.get("summary") or "")[:220],
                "published": (c.get("pubDate") or "")[:16].replace("T", " "),
                "source": ((c.get("provider") or {}).get("displayName") or ""),
                "url": ((c.get("canonicalUrl") or {}).get("url") or ""),
            })
    except Exception:
        pass
    _news_cache[ticker] = (_time.time(), items)
    return {"news": items[:limit]}


_chart_cache: dict[str, tuple[float, dict]] = {}


@app.get("/api/chart/{ticker}")
def chart(ticker: str, days: int = 260):
    import time as _time

    cache_key = f"{ticker}:{days}"
    hit = _chart_cache.get(cache_key)
    if hit and _time.time() - hit[0] < 600:
        return JSONResponse(hit[1])
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    df = store.load(ticker)
    store.close()
    if df.empty:
        raise HTTPException(404, f"no cached data for {ticker}")
    df = add_indicators(df)
    if days > 0:
        df = df.tail(days)
    bars = [
        {
            "time": idx.date().isoformat(),
            "open": round(r["open"], 4), "high": round(r["high"], 4),
            "low": round(r["low"], 4), "close": round(r["close"], 4),
            "volume": r["volume"],
        }
        for idx, r in df.iterrows()
    ]
    def line(col):
        return [
            {"time": idx.date().isoformat(), "value": round(v, 4)}
            for idx, v in df[col].items()
            if v == v  # not NaN
        ]
    payload = {"bars": bars, "sma50": line("sma50"), "sma200": line("sma200")}
    _chart_cache[cache_key] = (_time.time(), payload)
    return payload


@app.get("/api/search")
def search(q: str = ""):
    from .analyze import search_stocks

    return {"results": search_stocks(q)}


@app.get("/api/analyze/{ticker}")
def analyze(ticker: str):
    from .analyze import analyze_ticker

    result = analyze_ticker(ticker)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return JSONResponse(result)


JOURNAL_DB = ROOT / "data" / "journal.sqlite3"


def _last_closes(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    store = PriceStore(ROOT / "data" / "cache" / "prices.sqlite3")
    out = {}
    for t in set(tickers):
        row = store.conn.execute(
            "SELECT close FROM bars WHERE ticker = ? ORDER BY date DESC LIMIT 1", (t,)
        ).fetchone()
        if row:
            out[t] = float(row[0])
    store.close()
    return out


@app.get("/api/journal")
def journal_list():
    j = Journal(JOURNAL_DB)
    trades = j.all_trades()
    j.close()
    closes = _last_closes([t["ticker"] for t in trades if t["exit_price"] is None])
    return enrich_and_summarize(trades, closes)


@app.post("/api/journal")
def journal_add(trade: dict = Body(...)):
    required = ("ticker", "market", "entry_price", "shares", "stop")
    missing = [k for k in required if trade.get(k) in (None, "")]
    if missing:
        raise HTTPException(422, f"missing fields: {', '.join(missing)}")
    j = Journal(JOURNAL_DB)
    trade_id = j.add(trade)
    j.close()
    return {"id": trade_id}


@app.post("/api/journal/{trade_id}/close")
def journal_close(trade_id: int, body: dict = Body(...)):
    if body.get("exit_price") in (None, ""):
        raise HTTPException(422, "exit_price required")
    j = Journal(JOURNAL_DB)
    ok = j.close_trade(trade_id, float(body["exit_price"]),
                       body.get("exit_date"), body.get("exit_reason", "manual"),
                       body.get("notes"))
    j.close()
    if not ok:
        raise HTTPException(404, "trade not found or already closed")
    return {"ok": True}


@app.post("/api/journal/{trade_id}/stop")
def journal_stop(trade_id: int, body: dict = Body(...)):
    if body.get("stop") in (None, ""):
        raise HTTPException(422, "stop required")
    j = Journal(JOURNAL_DB)
    ok = j.update_stop(trade_id, float(body["stop"]))
    j.close()
    if not ok:
        raise HTTPException(404, "trade not found or already closed")
    return {"ok": True}


@app.delete("/api/journal/{trade_id}")
def journal_delete(trade_id: int):
    j = Journal(JOURNAL_DB)
    ok = j.delete(trade_id)
    j.close()
    if not ok:
        raise HTTPException(404, "trade not found")
    return {"ok": True}


@app.get("/api/alerts")
def alerts(limit: int = 30):
    path = ROOT / "data" / "alerts.jsonl"
    if not path.exists():
        return {"alerts": []}
    lines = path.read_text().strip().splitlines()[-limit:]
    return {"alerts": [json.loads(x) for x in reversed(lines)]}


@app.get("/api/backtest")
def backtest_results():
    path = ROOT / "data" / "backtest_latest.json"
    if not path.exists():
        return JSONResponse({"available": False})
    return JSONResponse(json.loads(path.read_text()))
