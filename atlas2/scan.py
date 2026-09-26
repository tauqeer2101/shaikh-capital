"""Market scan orchestrator.

Pipeline per market:
  universe -> price refresh -> technical screen -> pattern detection
  -> fundamentals (shortlist only) -> composite score -> ranked setups JSON

Outputs data/scans/{market}_latest.json plus a dated history copy.
"""
from __future__ import annotations

import copy
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

# config.json -> "backtest" (see atlas2/backtest.py and README "Backtest")
BACKTEST_DEFAULTS = {
    # commission + spread + FX (US from a EUR account) / STT etc. (India), percent
    # of trade value, charged on entry AND on exit
    "costs_pct_per_side": {"us": 0.15, "de": 0.10, "in": 0.20},
    # extra slippage (percent) on stop-loss exits and on buy-stop entries (both are
    # market orders once triggered); thinner for the German / Indian mid caps.
    # A single number applies to every market.
    "stop_slippage_pct": {"us": 0.05, "de": 0.15, "in": 0.15},
    "entry_slippage_pct": {"us": 0.05, "de": 0.15, "in": 0.15},
    # estimated dividend yield (%/yr) added to the price-index baseline; de = 0
    # because ^GDAXI is a performance index (dividends already included)
    "index_dividend_yield_pct": {"us": 1.5, "de": 0.0, "in": 1.3},
    "random_entries_per_trade": 10,
    "step": 1,              # check signals every trading day, like the 07:30 scan
    "window": 170,          # pattern look-back (at least 144)
    "breakout_wait": 1,     # a forming setup gets a one-day buy-stop, re-detected daily
    "max_hold": 40,         # time exit: sell at the close of the 40th day after the fill
    "max_gap_above_entry_pct": 3.0,
    "entry_mode": "buy_stop",   # or "close_confirm" (buy at the next open after a close above the entry)
    "seen_on_fill_only": True,  # only a FILLED trade blocks its pattern level
    # known corporate actions the free data leaves unadjusted (backtest only; more
    # can be listed in data/price_artifacts.json). adjust = multiply earlier OHLC
    # by ratio and earlier volume by 1/ratio.
    "price_artifacts": [
        {"ticker": "VEDL.NS", "date": "2026-04-30", "action": "adjust",
         "ratio": 289.5 / 773.6, "note": "demerger"},
        {"ticker": "TMPV.NS", "date": "2025-10-14", "action": "adjust",
         "ratio": 396.709 / 655.314, "note": "demerger"},
        {"ticker": "TRENT.NS", "date": "2026-01-01", "action": "adjust",
         "ratio": 2 / 3, "note": "exact 2/3 open without a volume surge"},
    ],
}
# config.json -> "pattern_tuning" (min_avg_r applies to the after-fees average R)
PATTERN_TUNING_DEFAULTS = {
    "auto_disable": True,
    "min_avg_r": 0.0,
    "min_trades": 30,
    "require_proven_edge": False,
    # chance of at least one false 'proven edge' across all pattern x market
    # cells (two-sided); n_tests null = number of bullish patterns x 3 markets
    "family_alpha": 0.05,
    "n_tests": None,
    "disabled": {"us": [], "de": [], "in": []},
}
VERDICT_WORDS = {
    "proven_edge": "proven edge",
    "unproven": "edge not proven",
    "no_edge": "no edge",
    "insufficient_data": "too few trades to judge",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Copy of `base` with `override` applied; nested dicts merge key by key, so a
    partial user section keeps the default keys it does not mention."""
    out = copy.deepcopy(base)  # never hand out the module-level default dicts / lists
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cfg_path: Path | None = None) -> dict:
    cfg_path = cfg_path or ROOT / "config.json"
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
        "pattern_tuning": PATTERN_TUNING_DEFAULTS,
        "backtest": BACKTEST_DEFAULTS,
    }
    user = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return _deep_merge(defaults, user)


def _history_years(bt: dict, market: str) -> int | None:
    """Length of the backtested period for a market, in whole years."""
    try:
        period = bt["method"]["period"][market]
        days = (date.fromisoformat(period["last_exit"][:10])
                - date.fromisoformat(period["first_entry"][:10])).days
        return max(1, round(days / 365.25))
    except Exception:
        return None


def load_pattern_tuning_info(market: str, cfg: dict, bt_path: Path | None = None) -> dict:
    """Per-market pattern tuning from the latest backtest, with its sources.

    Returns {"disabled": manual | auto, "manual": set, "auto": set,
    "stats": {pattern: backtest_stats}, "source": text, "has_backtest": bool,
    "backtest_max_hold": int | None, "kept_with_different_settings": bool}.

    v2 backtest (after fees): each pattern's verdict is recomputed with the
    CURRENT config via backtest.pattern_verdict and stored in its stats as
    'verdict' / 'verdict_reason'; with auto_disable on, 'no_edge' patterns are
    disabled (and 'unproven' ones too when require_proven_edge is set) - the
    same rule as backtest.disabled_patterns, which the portfolio replay and the
    backtest's live-configuration replay use too.
    v1 file (no fees): the old rule - average R below `min_avg_r` over at least
    `min_trades` trades. Missing or unreadable file: nothing auto-disabled.
    Manual entries in config.pattern_tuning.disabled[market] always apply
    ("manual"); "auto" holds only the patterns the backtest switched off.
    backtest_max_hold: the time exit the stored results used for this market
    (method.settings_by_market, else method.settings), None if unknown.
    """
    tuning = _deep_merge(PATTERN_TUNING_DEFAULTS, cfg.get("pattern_tuning") or {})
    manual = set(str(p) for p in ((tuning.get("disabled") or {}).get(market) or []))
    info = {"manual": manual, "auto": set(), "stats": {}, "source": "no backtest available",
            "has_backtest": False, "backtest_max_hold": None,
            "kept_with_different_settings": False}
    bt_path = bt_path or ROOT / "data" / "backtest_latest.json"
    if bt_path.exists():
        try:
            bt = json.loads(bt_path.read_text())
            from .backtest import disabled_patterns, pattern_verdict  # lazy: backtest imports scan
            markets = bt.get("markets") or {}
            per_pattern = (markets.get(market) or {}).get("per_pattern") or {}
            v2 = int(bt.get("version") or 1) >= 2
            years = _history_years(bt, market) if v2 else None
            new_stats: dict = {}
            for pat, st in per_pattern.items():
                st = dict(st)
                if v2 and "net_avg_r" in st:
                    stored = st.get("verdict_reason")
                    st["verdict"], st["verdict_reason"] = pattern_verdict(st, tuning)
                    # a pattern that never traded keeps the backtest's explanation
                    # of why (it does not depend on the tuning)
                    if not int(st.get("trades") or 0) and st["verdict"] == "insufficient_data" \
                            and stored:
                        st["verdict_reason"] = stored
                    if years:
                        st["history_years"] = years
                new_stats[pat] = st
            new_auto = disabled_patterns(per_pattern, tuning, market) - manual
            info["stats"], info["auto"] = new_stats, new_auto
            date_txt = bt.get("generated", "")[:10]
            method = bt.get("method") if isinstance(bt.get("method"), dict) else {}
            if market not in markets:
                info["source"] = (f"no backtest for {market.upper()} (backtest {date_txt} covers "
                                  f"{', '.join(m.upper() for m in markets) or 'no market'})")
            else:
                info["has_backtest"] = True
                source = f"backtest {date_txt}" + (" (after fees)" if v2 else "")
                carried = (method.get("carried_over") or {}).get(market)
                if carried:
                    source += f", {market.upper()} kept from the run of {str(carried)[:10]}"
                if market in (method.get("kept_with_different_settings") or []):
                    info["kept_with_different_settings"] = True
                    source += " with DIFFERENT settings (--keep-others)"
                info["source"] = source
                sbm = (method.get("settings_by_market") or {}).get(market)
                mh = sbm.get("max_hold") if isinstance(sbm, dict) else None
                if mh is None:
                    mh = (method.get("settings") or {}).get("max_hold")
                info["backtest_max_hold"] = int(mh) if mh is not None else None
        except Exception:
            info.update({"stats": {}, "auto": set(), "backtest_max_hold": None})
    info["disabled"] = info["manual"] | info["auto"]
    return info


def load_pattern_tuning(market: str, cfg: dict,
                        bt_path: Path | None = None) -> tuple[set[str], dict, str]:
    """(disabled_patterns, {pattern: backtest_stats}, source_note) - see
    load_pattern_tuning_info for the rules."""
    info = load_pattern_tuning_info(market, cfg, bt_path)
    return info["disabled"], info["stats"], info["source"]


def _signed(x: float, nd: int) -> str:
    """Signed number that never prints a negative zero."""
    v = round(float(x), nd) + 0.0   # + 0.0 turns -0.0 into 0.0
    return f"{v:+.{nd}f}"


def backtest_note(market: str, bt: dict, max_hold: int | None = None) -> str:
    """One-line summary of a pattern's backtest record for a setup's reasons."""
    if "net_avg_r" not in bt:  # old v1 file
        return (f"this pattern's {market.upper()} backtest: {bt['win_rate']}% wins, "
                f"{bt['avg_r']:+.2f} avg R over {bt['trades']} trades")
    years = bt.get("history_years")
    rule = f"stop/target/{int(max_hold)}-day exit" if max_hold else None
    span = ", ".join(x for x in (f"{years}y" if years else None, "after fees", rule) if x)
    parts = [f"{bt.get('win_rate') or 0:.0f}% wins", f"{_signed(bt.get('net_avg_r') or 0.0, 2)}R avg"]
    ei, eu = bt.get("excess_vs_index_pct"), bt.get("excess_vs_universe_pct")
    if ei is not None:
        parts.append(f"{_signed(ei, 1)}% per trade vs the index")
    if eu is not None:
        parts.append(f"{_signed(eu, 1)}% vs the equal-weight market")
    note = f"this pattern's {market.upper()} backtest ({span}): " + ", ".join(parts)
    verdict = VERDICT_WORDS.get(bt.get("verdict") or "")
    return note + (f" — {verdict}" if verdict else "")


def backtest_score_nudge(bt: dict | None, min_trades: int) -> float:
    """Pattern-score nudge from the pattern's backtest record: net average R x 20,
    capped at +/-10; 0 below min_trades (the same gate the verdict uses)."""
    if not bt or int(bt.get("trades") or 0) < min_trades:
        return 0.0
    r = bt.get("net_avg_r")
    if r is None:
        r = bt.get("avg_r") or 0.0
    return max(-10.0, min(10.0, r * 20))


def backtest_nudge(market: str, bt: dict | None, min_trades: int,
                   max_hold: int | None = None) -> tuple[float, str | None]:
    """(score nudge, setup note) - both gated on the same min_trades as the verdict."""
    if not bt or int(bt.get("trades") or 0) < min_trades:
        return 0.0, None
    return backtest_score_nudge(bt, min_trades), backtest_note(market, bt, max_hold)


def bullish_pattern_names() -> list[str]:
    from .patterns import BULLISH_DETECTORS
    return [d.__name__.removeprefix("detect_") for d in BULLISH_DETECTORS]


def tuning_summary(disabled: set[str], bt_stats: dict, min_trades: int,
                   manual=frozenset()) -> dict:
    """What the market tab needs to explain an empty list: enabled patterns, which
    disabled ones are manual (config) and which the backtest switched off (with
    its reasons), and whether the backtest switched off every judged pattern.
    all_judged_disabled needs backtest stats and at least one backtest disable:
    manual disables alone never make it true."""
    all_bull = bullish_pattern_names()
    manual = set(manual) & set(disabled)
    auto = set(disabled) - manual
    enabled = [p for p in all_bull if p not in disabled]
    judged = [p for p in enabled if int((bt_stats.get(p) or {}).get("trades") or 0) >= min_trades]
    return {
        "enabled": enabled,
        "manual": sorted(manual),
        "auto": sorted(auto),
        "reasons": {p: bt_stats[p].get("verdict_reason") for p in sorted(auto)
                    if p in bt_stats and bt_stats[p].get("verdict_reason")},
        "untested": [p for p in enabled if p not in judged],
        "all_judged_disabled": bool(auto) and bool(bt_stats) and not judged,
    }


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

    tinfo = load_pattern_tuning_info(market, cfg)
    disabled_patterns, bt_stats, bt_source = tinfo["disabled"], tinfo["stats"], tinfo["source"]
    bt_verdicts = {p: s["verdict"] for p, s in bt_stats.items() if s.get("verdict")}
    min_trades = int(_deep_merge(PATTERN_TUNING_DEFAULTS, cfg.get("pattern_tuning") or {})
                     .get("min_trades", 30))
    # max_hold: the time exit you trade (config); bt_max_hold: the one the stored
    # backtest figures used - the setup notes quote that one
    max_hold = int((cfg.get("backtest") or {}).get("max_hold", BACKTEST_DEFAULTS["max_hold"]))
    bt_max_hold = tinfo["backtest_max_hold"]
    max_gap = float((cfg.get("backtest") or {}).get("max_gap_above_entry_pct",
                                                     BACKTEST_DEFAULTS["max_gap_above_entry_pct"]))
    tuning_info = tuning_summary(disabled_patterns, bt_stats, min_trades, manual=tinfo["manual"])
    tuning_note = None
    if bt_max_hold is not None and bt_max_hold != max_hold:
        tuning_note = (f"config max_hold {max_hold} differs from the backtest's {bt_max_hold} - "
                       "re-run the backtest (its figures assume a "
                       f"{bt_max_hold}-day time exit)")
        progress(f"[{market}] WARN {tuning_note}")
    if not tinfo["has_backtest"]:
        progress(f"[{market}] no backtest for this market - nothing is tuned ({bt_source})")
    if tuning_info["manual"]:
        progress(f"[{market}] patterns switched off manually in config.json "
                 f"(pattern_tuning.disabled): {', '.join(tuning_info['manual'])}")
    if tuning_info["auto"]:
        progress(f"[{market}] patterns switched off by the backtest ({bt_source}): "
                 f"{', '.join(tuning_info['auto'])}")
    if tuning_info["all_judged_disabled"]:
        progress(f"[{market}] every pattern with enough backtest history is switched off by "
                 "the backtest's verdicts - expect no setups (set "
                 "pattern_tuning.auto_disable=false to see them)")

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
        # (after fees; gated on the same min_trades as the verdict)
        delta, bt_note = backtest_nudge(market, bt_stats.get(hit.pattern), min_trades, bt_max_hold)
        pattern_score = max(0, min(100, pattern_score + delta))
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
            # the backtest only buys up to this price (fills further above the entry are skipped)
            "max_gap_pct": max_gap,
            "max_buy": round(entry * (1 + max_gap / 100), 4),
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
        "pattern_tuning": {"disabled": sorted(disabled_patterns), "source": bt_source,
                           "verdicts": bt_verdicts, "note": tuning_note,
                           "kept_with_different_settings": tinfo["kept_with_different_settings"],
                           "has_backtest": tinfo["has_backtest"],
                           **tuning_info},
        # the exit rule you trade (config backtest.max_hold; the monitor enforces
        # it) and the one the stored backtest figures assume
        "exit_rule": {"max_hold_days": max_hold, "backtest_max_hold_days": bt_max_hold,
                      "note": tuning_note},
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
