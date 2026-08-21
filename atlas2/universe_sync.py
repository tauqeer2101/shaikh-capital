"""Build the scan universe from official/public index-constituent sources.

Writes one CSV per market into universe/: us.csv, de.csv, in.csv
Columns: ticker,name,index
Run: python -m atlas2 universe
"""
from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (Atlas2 personal stock screener)"}

WIKI_SOURCES = {
    "us": [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "S&P 500", "Symbol", "Security"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "S&P 400", "Symbol", "Security"),
    ],
}

NSE_SOURCES = [
    ("https://archives.nseindia.com/content/indices/ind_nifty100list.csv", "NIFTY 100"),
    ("https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv", "NIFTY Midcap 150"),
]

GERMAN_NAME_SOURCES = [
    ("https://en.wikipedia.org/wiki/DAX", "DAX"),
    ("https://de.wikipedia.org/wiki/MDAX", "MDAX"),
    ("https://de.wikipedia.org/wiki/SDAX", "SDAX"),
]


def _fetch_tables(url: str) -> list[pd.DataFrame]:
    html = requests.get(url, headers=UA, timeout=30).text
    return pd.read_html(io.StringIO(html))


def _find_table(tables: list[pd.DataFrame], ticker_col: str, name_col: str):
    best = None
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        t.columns = cols
        tc = next((c for c in cols if ticker_col.lower() in c.lower()), None)
        nc = next((c for c in cols if name_col.lower() in c.lower()), None)
        if tc and nc and len(t) >= 20 and (best is None or len(t) > len(best[0])):
            best = (t, tc, nc)
    if best is None:
        return None
    t, tc, nc = best
    out = t[[tc, nc]].copy()
    out.columns = ["ticker", "name"]
    return out


def _build_us(frames: list[pd.DataFrame]) -> pd.DataFrame:
    for url, index_name, tc, nc in WIKI_SOURCES["us"]:
        t = _find_table(_fetch_tables(url), tc, nc)
        if t is None:
            print(f"WARN {index_name}: constituents table not found")
            continue
        # Yahoo uses '-' where index files use '.' (BRK.B -> BRK-B)
        t["ticker"] = t["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
        t["index"] = index_name
        frames.append(t)
        print(f"  {index_name}: {len(t)} tickers")
    return pd.concat(frames) if frames else pd.DataFrame()


def _build_in() -> pd.DataFrame:
    frames = []
    for url, index_name in NSE_SOURCES:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            df = pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            print(f"WARN {index_name}: fetch failed: {exc}")
            continue
        df = df[df["Series"].astype(str).str.upper().eq("EQ")]
        out = pd.DataFrame({
            "ticker": df["Symbol"].astype(str).str.strip() + ".NS",
            "name": df["Company Name"].astype(str).str.strip(),
            "index": index_name,
        })
        frames.append(out)
        print(f"  {index_name}: {len(out)} tickers")
    return pd.concat(frames) if frames else pd.DataFrame()


def _clean_company_name(raw: str) -> str:
    name = re.sub(r"\[\d+\]", "", str(raw)).strip()
    return re.sub(r"\s+(SE|AG|KGaA|SE & Co\. KGaA|AG & Co\. KGaA|N\.V\.|S\.A\.)$", "", name).strip()


# Names Yahoo search fails to resolve to their Xetra listing
DE_OVERRIDES = {
    "Aroundtown": "AT1.DE",
    "Auto1 Group": "AG1.DE",
    "Elmos Semiconductor": "ELG.DE",
    "K+S": "SDF.DE",
    "Kion Group": "KGX.DE",
    "Knorr-Bremse": "KBX.DE",
    "1&1": "1U1.DE",
    "Carl Zeiss Meditec": "AFX.DE",
    "Dermapharm Holding": "DMP.DE",
    "Deutsche EuroShop": "DEQ.DE",
    "Deutsche Pfandbriefbank": "PBB.DE",
    "Fraport": "FRA.DE",
    "Hypoport": "HYQ.DE",
    "Jenoptik": "JEN.DE",
    "Krones": "KRN.DE",
    "Nordex": "NDX1.DE",
    "Salzgitter": "SZG.DE",
    "Stabilus": "STM.DE",
    "Ströer": "SAX.DE",
    "Wacker Chemie": "WCH.DE",
    "Wacker Neuson": "WAC.DE",
    "Rational": "RAA.DE",
    "RTL Group": "RRTL.DE",
    "TAG Immobilien": "TEG.DE",
    "Traton": "8TRA.DE",
    "Douglas Group": "DOU.DE",
    "Drägerwerk": "DRW3.DE",
    "Dürr": "DUE.DE",
    "Eckert & Ziegler Strahlen- und Medizintechnik": "EUZ.DE",
    "Evotec": "EVT.DE",
    "Grand City Properties": "GYC.DE",
    "Hamborner Reit": "HABA.DE",
    "Indus Holding": "INH.DE",
    "Jungheinrich": "JUN3.DE",
    "Kontron": "KTN.DE",
    "KSB": "KSB3.DE",
    "MLP": "MLP.DE",
    "Patrizia": "PAT.DE",
    "PNE": "PNE3.DE",
    "PVA TePla": "TPE.DE",
    "Sixt": "SIX2.DE",
    "Sto": "STO3.DE",
    "Südzucker": "SZU.DE",
}


def _yahoo_lookup_de(name: str) -> str | None:
    """Map a German company name to its Xetra (.DE) Yahoo symbol."""
    import yfinance as yf

    try:
        quotes = yf.Search(name, max_results=8).quotes
    except Exception:
        return None
    for q in quotes:
        sym = str(q.get("symbol", ""))
        if q.get("exchange") == "GER" and sym.endswith(".DE"):
            return sym
    for q in quotes:  # fallback: any .DE listing
        sym = str(q.get("symbol", ""))
        if sym.endswith(".DE"):
            return sym
    return None


def _build_de(root: Path) -> pd.DataFrame:
    cache_path = root / "universe" / "de_name_map.json"
    cache: dict[str, str | None] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    frames = []
    for url, index_name in GERMAN_NAME_SOURCES:
        try:
            tables = _fetch_tables(url)
        except Exception as exc:
            print(f"WARN {index_name}: fetch failed: {exc}")
            continue
        if index_name == "DAX":
            t = _find_table(tables, "Ticker", "Company")
            if t is None:
                print("WARN DAX: table not found")
                continue
            # Wikipedia mixes venues (e.g. AIR.PA); normalize everything to Xetra .DE
            t["ticker"] = t["ticker"].astype(str).str.strip().str.split(".").str[0] + ".DE"
            t["name"] = t["name"].astype(str).str.strip()
            t["index"] = index_name
            frames.append(t[["ticker", "name", "index"]])
            print(f"  DAX: {len(t)} tickers")
            continue
        # MDAX/SDAX list names only; resolve tickers via Yahoo search (cached)
        table = next(
            (t for t in tables if "Name" in [str(c).strip() for c in t.columns] and len(t) >= 40),
            None,
        )
        if table is None:
            print(f"WARN {index_name}: name table not found")
            continue
        table.columns = [str(c).strip() for c in table.columns]
        rows = []
        misses = []
        for raw in table["Name"].dropna().astype(str):
            name = _clean_company_name(raw)
            if name in DE_OVERRIDES:
                cache[name] = DE_OVERRIDES[name]
            if name not in cache:
                cache[name] = _yahoo_lookup_de(name)
                time.sleep(0.25)
            if cache[name]:
                rows.append({"ticker": cache[name], "name": name, "index": index_name})
            else:
                misses.append(name)
        frames.append(pd.DataFrame(rows))
        print(f"  {index_name}: {len(rows)} mapped, {len(misses)} unresolved" + (f" ({', '.join(misses[:6])}...)" if misses else ""))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return pd.concat(frames) if frames else pd.DataFrame()


def build_universe(root: Path) -> dict[str, int]:
    out_dir = root / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    builders = {
        "us": lambda: _build_us([]),
        "de": lambda: _build_de(root),
        "in": _build_in,
    }
    for market, build in builders.items():
        merged = build()
        if merged.empty:
            print(f"WARN {market}: no sources succeeded, keeping existing file if any")
            continue
        merged = merged.drop_duplicates(subset="ticker").reset_index(drop=True)
        merged.to_csv(out_dir / f"{market}.csv", index=False)
        counts[market] = len(merged)
    return counts


if __name__ == "__main__":
    n = build_universe(Path(__file__).resolve().parent.parent)
    print("Universe built:", n)
