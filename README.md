# Shaikh Capital — Swing Trade Scanner (US · Germany · India)

Shaikh Capital scans ~1,275 liquid stocks across three markets every day, detects classic
chart patterns, scores each stock's technicals and fundamentals, and shows a ranked
list of swing-trade setups with concrete levels: **where to buy, where the stop-loss
goes, and where the target is** — plus a suggested position size for your capital.

## Daily use

1. Double-click **`SCAN.command`** after market close (or any time — it uses the
   latest completed daily bars). First run downloads ~2 years of history and takes
   a few minutes; later runs only refresh what's stale.
2. Double-click **`RUN_DASHBOARD.command`** — the dashboard opens at
   <http://127.0.0.1:8899> with tabs for 🇺🇸 US, 🇩🇪 Germany, 🇮🇳 India.
3. Click any setup to see its chart with entry / stop / target lines, the reasons
   it qualified, its fundamental breakdown, and the suggested share count.

## Reading the dashboard (for non-technical users)

- The **overview cards** at the top show each market at a glance: how many setups,
  how many A-grades, and a health chip (🟢 trade normally / 🟡 careful / 🔴 wait).
- Click any setup and read the **action box first** — it says in plain English
  whether there is anything to do right now, and exactly what.
- The chart draws the **detected pattern itself** in ink-colored lines — Darvas
  box walls, triangle trendlines, flag poles, necklines — so you can verify the
  signal with your own eyes, like an analyst's marked-up chart.
- Every setup is a three-price plan: **Buy above** (only act if price rises past
  it), **Safety stop** (your exit if wrong), **Target** (take profit). The
  suggested share count keeps one loss at ~1% of your capital.
- A **Latest news** panel under each stock shows recent headlines before you commit.
- The **❓ How it works** button in the header explains everything, including a
  glossary of every term used.
- The **☀️/🌙 button** switches between light mode (clean white, Google-style)
  and dark mode. Your choice is remembered.

## What the scanner looks for

**Chart patterns** (bullish): bull flag, cup & handle, inverse head-and-shoulders,
double bottom, ascending triangle, and a true **Darvas box** (box top/bottom each
confirmed by 3 quiet bars, near 52-week highs, quiet volume inside the box and a
volume surge on the breakout, stop under the box bottom).
**Bearish warnings**: head-and-shoulders tops and double tops are listed separately —
avoid new buys there, and review them if you hold the stock.

**Technical screen**: price above the 50/200-day averages, healthy RSI, relative
strength versus the stock's own index (S&P 500 / DAX / NIFTY), liquidity floor,
and proximity to 52-week highs.

**Fundamentals** (fetched for the shortlist): revenue and earnings growth, return
on equity, margins, debt/equity, and forward-P/E valuation, scored 0–100.

**Composite score** = 40% technicals + 35% pattern quality + 25% fundamentals.
Grade A ≥ 72, B ≥ 58. Setups below a 1.8 risk:reward are dropped. A setup whose
chart also carries a bearish pattern is penalized and flagged.

`status = forming` means the pattern is complete but price has not broken out yet —
a buy-stop order above the entry level is the classic way to trade it.
`status = triggered` means the breakout happened within the last ~3%.

## Dividends

Every setup and every searched stock shows a **Dividends** block: the current
dividend yield (trailing 12 months ÷ price), the cash paid per share in the last
12 months, and the **yearly totals for the last 5 years** — a company that pays
quarterly (e.g. HCL Technologies) shows one summed figure per year, not four
entries. Stocks that pay nothing say so explicitly.

Freshness: the yield is recomputed against the current price on **every** scan;
the payout history itself is re-downloaded by the morning scan once it is older
than `dividend_max_age_days` (config.json, default 3), so a newly announced
payment shows up within a few days at most.

## Stock search — analyze anything on demand

The search bar in the header looks up any stock by **name or ticker** — first in
the scan universe, then on Yahoo for anything else in the world (US listings,
`.DE`, `.NS`, …). Picking a result runs the full analysis pipeline for that one
stock immediately:

- price chart with 50/200-day averages,
- trend verdict, RSI, returns, strength vs its index, distance from 52-week high,
- every chart pattern currently detected (bullish and bearish, with levels),
- the fundamental scorecard and analyst consensus.

Use it to vet a stock you heard about, or to check one of the day's bearish
warnings before deciding what to do with a holding.

## Trade journal

The **📒 Journal** tab tracks the trades you actually take:

- On any setup's detail page, click **Log this trade** — fill price, share count and
  date are pre-filled from the setup; adjust to your real fill and save.
- Open positions show the latest cached price, live P&L (%, currency and R-multiple)
  and your total open risk. Use **Close** when you exit (records price + reason),
  **Stop** to trail your stop-loss up, **✕** to delete a mistaken entry.
- Closed trades accumulate your real statistics: win rate, average R and total R —
  compare them against the backtest to see whether your execution matches the edge.

Journal data lives in `data/journal.sqlite3` (back this file up if you care about
the history).

## Market regime filter

Each scan first judges the health of the market index itself (S&P 500 / DAX /
NIFTY) — breakout setups fail far more often in weak markets:

- 🟢 **healthy** — index above its 50/200-day averages, near highs → normal sizing
- 🟡 **caution** — above the 200-day but under pressure → suggested sizes halved,
  stick to A-grades
- 🔴 **downtrend** — below the 200-day → suggested size 0; watch setups, don't chase

The verdict shows as a banner above each market's list. Disable with
`"regime_filter": false` in config.json if you want raw sizing back.

## Earnings warnings

Setups within `earnings_warn_days` (default 7) of a scheduled earnings report are
flagged **⚠E** in the table and in the reasoning. Earnings gaps blow through
stop-losses — skip those trades or size well down.

## Price alerts

`python -m atlas2 monitor` checks hourly (via the `com.atlas2.monitor`
LaunchAgent, weekdays 09:00–22:00) and fires a macOS notification when:

- an **open journal position** touches its stop or target,
- a top-10 A-grade **forming setup** crosses its entry level.

Alerts are logged to `data/alerts.jsonl` and shown in the Journal tab. Each
condition alerts at most once per day. Run `... monitor --force` to test.

## Pattern tuning (evidence-based)

The scanner tunes itself per market from the latest backtest
(`data/backtest_latest.json`): any pattern with a **negative average R over at
least 30 backtested trades in that market is automatically excluded** from new
setups there. Currently that removes ascending triangles in India (−0.18 avg R
over 59 trades); US and Germany run the full pattern set.

Every setup also cites its pattern's real record in that market ("this pattern's
IN backtest: 38.6% wins, +0.05 avg R over 280 trades") and its pattern score is
nudged up or down accordingly. Excluded patterns are listed in the dashboard
header ("tuned out: …").

Settings live in `config.json → pattern_tuning` (`auto_disable`, `min_avg_r`,
`min_trades`, plus a manual `disabled` list per market). Re-run
`python -m atlas2 backtest` every few months so the tuning follows fresh data.

## Position sizing

`config.json` holds your settings (copy `config.example.json` to `config.json`):

```json
{ "capital_eur": 10000, "risk_pct": 1.0, "min_risk_reward": 1.8, ... }
```

Suggested size risks `risk_pct` of `capital_eur` per trade (distance from entry to
stop), converted at the current EUR/USD/INR rate, capped at 25% of capital per
position. Edit the file and re-scan to change it.

## Commands

```sh
.venv/bin/python -m atlas2 scan          # all markets (or: scan us / de / in)
.venv/bin/python -m atlas2 serve         # dashboard at 127.0.0.1:8899
.venv/bin/python -m atlas2 universe      # refresh index constituent lists
.venv/bin/python -m atlas2 backtest     # historical hit-rate per pattern
```

The backtest replays the cached history: every detected pattern, breakout entry,
and stop/target/time exit — results in `data/backtest_latest.json` (win rate,
average R-multiple, expectancy per pattern and market). No slippage or fees.

## Data & honesty notes

- Prices and fundamentals come from Yahoo Finance (free, unofficial). Daily bars,
  auto-adjusted for splits/dividends. Occasionally a ticker is missing or stale —
  the scanner just skips it.
- Universe: S&P 500 + S&P 400 (US), DAX + MDAX + SDAX (Germany, ~120 of 160
  resolved to Xetra tickers), NIFTY 100 + Midcap 150 (India). Refresh with
  `python -m atlas2 universe`.
- This is an **analysis tool, not investment advice**. Patterns fail regularly;
  the edge comes from cutting losers at the stop and letting winners reach the
  target. Nothing here guarantees any return.
