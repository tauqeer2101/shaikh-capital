# Shaikh Capital — Swing Trade Scanner (US · Germany · India)

Shaikh Capital scans ~1,275 liquid stocks across three markets every day, detects classic
chart patterns, scores each stock's technicals and fundamentals, and shows a ranked
list of swing-trade setups with concrete levels: **where to buy, where the stop-loss
goes, and where the target is** — plus a suggested position size for your capital.

## Daily use

1. Double-click **`SCAN.command`** after market close (or any time — it uses the
   latest completed daily bars). First run downloads ~10 years of history and takes
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
- Every setup is a plan of three prices and a time limit: **Buy above** (only
  act if price rises past it — and not if it opens more than 3% higher, the
  **buy-up-to** price), **Safety stop** (your exit if wrong), **Target** (take
  profit), and the **time exit** (sell at the close of the 40th trading day if
  neither stop nor target was hit — the backtest numbers assume you do). The
  suggested share count keeps one loss at ~1% of your capital.
- A **greyed-out setup with a "sector full" chip** is real but held back: two
  setups from its sector already rank higher today (a basket of picks from one
  sector moves as one). In the 📒 Journal, a red **STOP HIT — sell** badge on an
  open position means its price is at or below your safety stop — the plan says
  sell, today. See *Protective guards* below.
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
a buy-stop order above the entry level is the classic way to trade it. The
backtest only buys up to 3% above the entry (the dashboard shows this
buy-up-to price); a plain overnight buy-stop fills at whatever price the stock
opens at, so check the open and skip the trade if it opened higher.
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
  date are pre-filled from the setup; adjust to your real fill and save. The
  setup's time limit is saved with the trade (40 trading days for the pattern
  scanner; none for Comeback setups), so the price monitor can remind you on
  the time-exit day.
- The setup's sector is saved with the trade. If your open positions already
  hold `max_per_sector` (default 2) trades in that sector, the form shows a
  warning naming them before you save — you can still save; the decision is
  yours (see *Protective guards*).
- Open positions show the latest cached price, live P&L (%, currency and R-multiple)
  and your total open risk. A position whose latest price is at or below its
  stop is marked **STOP HIT — sell** with the Close button highlighted; when a
  sector holds `max_per_sector` or more open positions, a per-sector summary
  line appears above the table. Use **Close** when you exit (records price +
  reason: target / stop / time / manual), **Stop** to trail your stop-loss up,
  **✕** to delete a mistaken entry.
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

- an **open journal position** touches its stop or target. A hit stop is an
  **exit instruction**, not a hint: the notification is titled
  `EXIT: <ticker>` and reads *"STOP HIT — sell now. Price X is below your stop
  Y (−5.3% from entry Z). The plan was to exit here; every day held past the
  stop is outside the plan."* It repeats once a day while the position stays
  open, and each alert carries `days_past_stop` (calendar days since the first
  stop alert for that position) so the Journal tab can say "stop hit 3 days
  ago — still open". A hit target stays a "consider taking profit" note.
- an open position reaches its **time exit**: on the morning of the last
  allowed trading day (day 40 after the day you bought, for pattern-scanner
  trades) it reminds you to sell at that day's close unless the stop or target
  is hit first, and it keeps reminding you once a day while the position stays
  open after that day. Trading days are counted from the local price cache
  (the stock's own bars plus the days its exchange traded); weekdays after the
  last cached bar are counted too, so a stale cache does not delay the
  reminder, though an exchange holiday in that stale stretch can bring it one
  day early. Trades logged before this feature, or from an older scan file,
  use the backtest's `max_hold` from `config.json`; Comeback trades have no
  time exit.
- a top-10 A-grade **forming setup** crosses its entry level. The alert repeats
  the setup's **buy-up-to** price: the backtest skips the trade if the stock
  opened above it.

Alerts are logged to `data/alerts.jsonl` and shown in the Journal tab. Each
condition alerts at most once per day. Run `... monitor --force` to test (it
downloads current prices). The dashboard server must be restarted once after
an update so that new journal entries store their time limit.

## Protective guards

Two guards exist because of one paper-traded basket (2026-08-18): five scanner
picks of which four were financials — two custody banks, a card lender and an
insurer — so one sector move hit all of them; and one of them fell through its
stop two days later while the monitor only said "price at/below your stop —
review position", the position stayed open, and the planned 5% loss became
11%. The v2 backtest shows the system's own stops are hit about 54% of the
time, so a hit stop has to be unmistakable, and sector concentration has to be
limited where the picks are made.

- **Stop hit = exit.** When an open position's price is at or below its stop,
  the monitor's alert is the instruction above (`EXIT: <ticker>`, "STOP HIT —
  sell now"), the Journal tab shows a red **STOP HIT — sell** badge with a
  one-line instruction and how many days the position has been held past the
  stop, and the Close button is highlighted. The alert's key and once-a-day
  cadence are unchanged.
- **Sector cap in the scanner.** After the setups are ranked, at most
  `max_per_sector` (default 2) per sector and market stay in the *active* list.
  The rest are **not deleted**: they keep their rank in the list, carry
  `"sector_capped": true` and a `sector_cap_note` ("already 2 Financial
  Services setups ranked higher today — skip unless you sell one"), render
  greyed out with a *sector full* chip, and are still clickable. A setup with
  no known sector sits in an `Unknown` bucket that is never capped. The scan
  JSON adds `sector_counts` (setups per sector, capped ones included) and
  `max_per_sector`; the overview counts (`setups`, `a_grades`) count only
  active setups and add `capped`. The Comeback scan is not capped.
- **Sector warning when logging a trade.** The journal stores each trade's
  sector; `GET /api/journal/exposure` groups the open positions by sector
  against the same cap, and the log form warns when the sector is already
  full — the Save button stays available.

To change the cap, set `"max_per_sector"` in `config.json` (copy the line from
`config.example.json`); `0` or `null` switches the cap and the warning off. The
next scan applies it; restart the dashboard server once after updating so the
journal stores sectors.

## Backtest: does each pattern really have an edge?

`python -m atlas2 backtest` replays the cached daily history (about 10 years,
mid-2016 to today, ~1,300 stocks) and asks, **every trading day** for every
stock: **what would the live scanner have shown today, using only what was
known today — and would that trade have beaten simply holding the market?**

What it simulates:

- **The scanner's own filters**: the liquidity floor, technical score ≥ 40,
  the market regime (no new trades while the index is in a downtrend, checked
  on the signal day and again on the fill day), the highest-confidence pattern
  on the chart, risk:reward ≥ 1.8 for that pattern (never a fall-back to a
  weaker one), and the **top-45 shortlist**: each day the candidates of the
  whole market are ranked the way the scanner ranks them (pattern confidence ×
  50 + technical score) and only the top 45 can be traded.
- **Entries the way you would place them**: a *forming* setup gets a one-day
  buy-stop at that day's entry level. If it does not fill, the scanner looks
  again the next day with fresh levels, exactly like the 07:30 scan. A
  *triggered* setup is bought at the next day's open. A stock is not looked at
  again while its order is waiting or its position is open, and a pattern
  level that was traded is not traded again for 170 days (if that level is
  the scanner's top pick, the stock is skipped that day — the backtest never
  swaps in a runner-up you would not have seen).
- **The buy-up-to price**: fills more than 3% above the entry level are
  skipped. The dashboard shows this buy-up-to price on every setup, and the
  monitor's entry alert repeats it. A plain buy-stop left overnight does
  **not** enforce it — it fills at whatever price the stock opens at — so
  check the open, and skip the trade if it opened higher.
- **Exits**: stop-loss, target, or a **time exit at the close of the 40th
  trading day** after the fill. The stop and target stay live through that
  40th day. If the stop is touched on the entry day it counts as a loss
  (daily bars cannot show which came first); a day that opened between your
  stop and entry and touched both counts as filled and stopped out. A gap
  below the stop exits at the open, not at the stop.
- **Costs on every trade**, charged on entry *and* on exit, as a percent of
  the trade value: US 0.15% (commission, spread and the EUR/USD conversion fee
  from a EUR account), Germany 0.10%, India 0.20% (STT and charges). Buy-stop
  entries and stop-loss exits (both are market orders once triggered) pay
  extra slippage: US 0.05%, Germany and India 0.15% (thinner mid caps). These
  are assumptions; change them in `config.json`.

**The results only apply if you follow the 40-day time exit.** Win rates,
average R, verdicts and the portfolio returns all assume you sell at the close
of the 40th trading day if neither the stop nor the target was hit. Holding
with stop and target only changes the numbers a lot (on the smoke run's
entries: +0.064R to +0.140R per trade, and the average hold from 22 to 34
days), so the rule matters in both directions. Each setup on the dashboard
shows the time limit, and the price monitor reminds you on the time-exit day
for trades you logged in the journal.

Every trade is compared with three alternatives **over the same days**:

1. **Holding the index**: S&P 500 / DAX / NIFTY 50, bought at the open of the
   first index day on or after the entry and sold at the close of the last
   index day on or before the exit. The S&P 500 and NIFTY 50 are price
   indices, so an estimated dividend yield is added (US 1.5% a year, India
   1.3%); the DAX already includes dividends.
2. **Holding the market**: an equal-weight basket of every stock in that
   market's universe that traded across the same days.
3. **Random entries**: 10 random stocks from the same market, bought the same
   day with the same stop/target distances, the same exits and the same
   costs. They are drawn only from stocks that passed the liquidity floor the
   day before and move about as much as the traded stock (14-day ATR% within
   0.8–1.25×); the traded stock itself is never drawn. If the pattern cannot
   beat a coin flip, it is not the pattern doing the work.

Neither the index nor the market basket is charged costs (think of a cheap
index ETF).

A few more rules keep the replay honest:

- **Warm-up**: the first year of price history (July 2016 – July 2017) is only
  used to warm up the 200-day average, the 52-week high and the market regime,
  so signals start around July 2017. Stocks that listed later are scanned from
  their 120th trading day, as the live scanner does.
- **End of the data**: a trade whose 40-day holding window would run past the
  last day of data is left out, whatever happened to it (counted as
  `censored_at_end`). So the last entries are about 40 trading days before the
  data ends; the last exits are near the end.
- **Price data repairs (backtest only)**: the free price data contains a few
  unadjusted splits and bonus issues, demergers, bad listing joins (for
  example old shares at 7 cents glued to a new listing at $16.50), one-day
  spikes and an oddly adjusted 2016 start for some stocks. The backtest
  back-adjusts exact split ratios that come with a volume surge and a few
  known corporate actions (`backtest.price_artifacts` in the config, or
  `data/price_artifacts.json`), drops the history before a bad join, a run of
  flat zero-volume placeholder bars or a stale 2016 start, and removes one-day
  spikes that fully revert. Every repair is listed on the dashboard under
  *Method*. The live scanner still sees the raw data.

### Verdicts

Each pattern gets one verdict per market:

| Verdict | Meaning | What the scanner does |
|---|---|---|
| **proven edge** | Beat the market after fees, and the whole range is above zero: statistically clear in this backtest after adjusting for 18 pattern x market tests (still in-sample) | keeps it |
| **unproven** | Positive on average, but not distinguishable from luck once all 18 pattern x market tests are counted | keeps it (switch off with `require_proven_edge`) |
| **no edge** | Lost money after fees, or did not beat holding the index or the market | **switches it off** in that market |
| **insufficient data** | Fewer than 30 trades | keeps it |

Why "adjusting for 18 tests": with 6 patterns in 3 markets, one of them will
look good by pure luck fairly often (about a 37% chance with plain 95%
ranges). The proven/unproven line therefore uses a wider range so that the
chance of *any* false "proven edge" is about 5% (`pattern_tuning.family_alpha`
and `n_tests` in the config; `n_tests: 1` gives the plain 95% rule). The 95%
ranges shown in the table are the plain ones.

The ranges treat trades opened in the same month — and in the same quarter,
whichever is more cautious — as one block of evidence: a trade lasts up to 40
trading days, so trades opened close together share the same market weather,
and ten trades in one strong month are not ten independent proofs. With
trades spread over fewer than 10 quarters there is no range at all (so at best
"unproven"). When a pattern failed the market comparison, its "no edge"
reason also says whether it was *clearly* worse than the market or just not
better on average, and whether it lost before fees or the fees ate its edge;
when it beat the equal-weight market but failed on the index or on average R,
the reason says so and quotes that test's range instead.

Even the adjusted ranges are optimistic: the pattern set, the settings and the
thresholds were all chosen after looking at earlier results on the same
history. Entries after mid-2024 are not a clean test, and only paper or real
trades made after the 2026-09-25 freeze count as out-of-sample evidence.
Per-pattern results also changed a lot between 2017-21 and 2022-26, so read
the year-by-year figures before trusting a verdict.

### Changing costs and settings

Everything lives in `config.json` (defaults in the code, documented in
`config.example.json`; you only need to list what you change — a partial
section keeps the other default keys):

```json
"backtest": {
  "costs_pct_per_side": { "us": 0.15, "de": 0.10, "in": 0.20 },
  "stop_slippage_pct": { "us": 0.05, "de": 0.15, "in": 0.15 },
  "entry_slippage_pct": { "us": 0.05, "de": 0.15, "in": 0.15 },
  "index_dividend_yield_pct": { "us": 1.5, "de": 0.0, "in": 1.3 },
  "random_entries_per_trade": 10,
  "step": 1, "window": 170, "breakout_wait": 1, "max_hold": 40,
  "max_gap_above_entry_pct": 3.0,
  "entry_mode": "buy_stop"
}
```

For example, if your broker charges more for Indian trades, set
`"costs_pct_per_side": {"in": 0.30}` and re-run the backtest. A single number
for a slippage setting applies to every market. `"entry_mode": "close_confirm"`
models "check back after the close": buy at the next open once a day closes
above the entry (skipped if that close ran more than 3% above it).
`window` must be at least 144 days (the cup-and-handle needs a 140-day base).
The shortlist size is the scanner's own `shortlist_size` (45). Changing the
dividend yields needs a backtest re-run.

### Running it

```sh
.venv/bin/python -m atlas2 backtest                  # all markets, then the portfolio replay
.venv/bin/python -m atlas2 backtest us de            # re-run only these; keeps the other markets' previous results
.venv/bin/python -m atlas2 backtest us --limit 25    # quick smoke run (first 25 tickers)
.venv/bin/python -m atlas2 backtest --raw            # also replay the old unfiltered selection
.venv/bin/python -m atlas2 backtest --workers 4 --no-portfolio
.venv/bin/python -m atlas2 backtest --legacy-grid    # old grid: every 5 days, 15-day frozen buy-stop (labelled files)
.venv/bin/python -m atlas2 backtest --entry-mode close_confirm   # entry variant (labelled files)
.venv/bin/python -m atlas2 portfolio                 # re-run only the portfolio replay
```

A full run uses all but two CPU cores and needs only the local price cache (no
downloads). Every trading day of every stock is replayed, which costs about 1.5
CPU-seconds per stock: on a 10-core Mac (8 workers) expect roughly 5–10
minutes for all three markets, plus under a minute for the portfolio replay.
A 20-stock `--limit` smoke run takes about 15 seconds. It writes:

- `data/backtest_latest.json`: per-pattern results, verdicts, method and
  caveats (read by the scanner and the dashboard).
- `data/backtest_trades.json`: every simulated trade, used by the portfolio
  replay.
- `data/backtest_trades_live.json`: the same replay with today's switched-off
  patterns removed *before* the best pattern is picked, as the live scanner
  does (an enabled pattern then gets extra trades it lost to a switched-off
  one in the headline run). Its per-pattern figures are informational only.
- `data/backtest_trades_walkforward.json`: the same kind of replay, but each
  signal year uses the patterns that would have been switched off at that
  time — judged only on trades that closed before the year started (plus your
  manual `disabled` lists). It feeds the portfolio's walk-forward profile.
- `data/backtest_trades_grade_a.json` and
  `data/backtest_trades_a_in_caution.json`: the replays behind the portfolio's
  grade profiles. After the pick and the shortlist, setups below grade A (for
  the second file: below grade A while the index is in caution) are never
  ordered, so they do not occupy the stock and block a later A-grade setup on
  it. The grade-A replay also leaves out today's switched-off patterns (as in
  the live-configuration file); the A-in-caution replay keeps every pattern.
- `data/backtest_trades_raw.json` (with `--raw` only): the old selection (first
  pattern found, no filters) under the new costs and exits, to show how much
  the scanner's filters matter.
- `data/portfolio_sim.json`: the whole decade replayed as one account, compared
  with holding the markets.

Naming some markets (`backtest us de`) re-runs only those and keeps the other
markets' previous results in `backtest_latest.json` and
`backtest_trades.json` (and the live / walk-forward / grade files), so their
switched-off patterns stay switched off. A kept market keeps the date of the
run that actually computed it and its own settings
(`method.settings_by_market`, plus its own costs, slippage and dividend yield).
The run stops and asks you to choose when the kept results cannot simply be
carried: when they were computed with different settings — simulation
settings, `costs_pct_per_side`, `stop_slippage_pct`, `entry_slippage_pct` or
`index_dividend_yield_pct` for that market (each kept market is compared with
its *own* recorded settings, so a market kept with `--keep-others` makes every
later partial run ask again while its settings still differ), and when the
previous file is in the old v1
format (or unreadable) — v1 results cannot be carried at all. `--keep-others`
keeps results computed with other settings anyway (labelled "DIFFERENT
settings" in the Backtest view and in the market tab's source line),
`--replace` drops them with a warning. Re-running all markets never needs a
choice.

A `--limit` run writes `data/backtest_smoke.json` and
`data/backtest_trades_smoke.json` (plus the `_live_smoke`,
`_walkforward_smoke`, `_grade_a_smoke`, `_a_in_caution_smoke` and, with
`--raw`, `_raw_smoke` files) only. It never overwrites the real results and
skips the portfolio replay. `--limit` needs a whole number of at least 1, and a
mistyped option stops with an error instead of starting a full run.

### Reading the Backtest view on the dashboard

Click **🧪 Backtest** in the tab bar (click a market tab to go back). Until
the new backtest has been run, the tab shows the old results under an **"Old
backtest format"** warning: no fees, no market comparison, no verdicts.

- The **headline** line: *system as designed, X% a year vs holding the
  markets, Y% a year*. Y is an index portfolio weighted like the system's own
  market exposure (the blended index when that is not available). Next to X
  is the range it takes just from which of several equally ranked setups got
  the last free slot (5th–95th percentile of 101 orderings); if Y sits inside
  that range the banner says so instead of calling it a win or a loss. It
  also shows the worst drop along the way, and below it the walk-forward
  result (patterns switched off each year using only earlier trades; it is
  labelled *approximate* when it had to fall back to deleting the switched-off
  patterns' trades instead of re-picking the setups) and the in-sample
  "enabled patterns only" result with its hindsight effect. Both are still
  backtests on the history the settings were chosen on. If your config now
  switches off a different set of patterns than when the portfolio replay
  ran, that line says which patterns the figure left out and asks you to
  re-run.
- One **card per market** counts how many patterns have a proven edge, are
  unproven, have no edge or have too little data, lists the patterns switched
  off in the live scanner, and gives all patterns together: trades, return
  per trade after fees and the edge over holding the index.
- The **per-market table**: trades, wins after fees, average R after fees,
  average return per trade after fees, the edge vs the index and vs the
  equal-weight market (with its 95% range), the same edge per unit of risk,
  the edge vs random entries (with its range), grade-A trades, positive years
  (x of y) and the verdict. Hover over the verdict to read the reason in plain
  English; an **off in scanner** tag marks patterns the live scanner switches
  off. Green means better than the alternative, red means worse. Click a row
  for the details: before and after fees, how the trades ended (target, stop,
  time exit), what holding the index or the market returned over the same
  days, results by grade and a year-by-year table. Under the table: how many
  signal days were checked and why setups were filtered out, the price
  repairs, and (when patterns are switched off) a **Replay without …**
  fold-out with the replay without the patterns switched off at backtest time
  (flagged when your config now switches off a different set).
- The **portfolio table**, one row per profile: yearly return (CAGR, with its
  slot-ordering range), max drawdown (marked to market daily; the small line
  is the closed-trade range), trades, win rate, the average share of the
  account invested, and the difference to the blended index and to the
  exposure-weighted index. Below it, **Holding the market instead** gives each
  index's yearly return (incl. estimated dividends) and worst drawdown and the
  equal-weight market, and a **Year by year** fold-out compares the system
  (booked by exit year), walk-forward and enabled-only profiles with each
  index.
- The **Method** and **Caveats** sections at the bottom list the costs,
  slippage, dividend yields, simulation settings, filters, baselines,
  statistics, period, universe sizes, every price repair and the limitations.
  A market kept from an earlier run is listed with the date of the run that
  computed it and any settings that differ from this run's.

The verdict badges, the "switched off in the live scanner" chips and the
**off in scanner** tags use your *current* `config.json`, the same way the
scanner does on every scan; when a verdict changed since the backtest ran, the
row says what it was at backtest time. (The dashboard server applies this after
it has been restarted once following an update; until then the view shows the
verdicts stored at backtest time and says so.) The headline verdict counts and
the number of pattern x market tests follow the current config too. The
"Enabled patterns only"
note under the portfolio table lists what that profile actually left out when
the portfolio replay ran.

### The portfolio replay

`python -m atlas2 portfolio` (run automatically after a full backtest) replays
the trades as one €25,000 account with at most 6 positions. Each trade risks
1% of current equity and is sized like the dashboard sizes it — shares = risk
÷ (entry level − stop), at most 25% of equity per position and never more than
the free cash. When more setups trigger on a day than there are free slots,
the best-ranked one wins; which of several equally ranked setups gets the last
slot is luck, so every profile is replayed over 101 slot orderings and the
headline is the median one (the p5–p95 range is shown too). Entries fill at
the open while exits happen during the day or at the close, so a position that
exits on a given day keeps its slot and its cash until that day's entries are
placed (all markets share one calendar). Profiles: full
size in every market state (no half-sizing in caution — the backtest already
leaves out signals raised, and orders filled, while the index was in a
downtrend, so this is not a "no regime filter" run; it is labelled that way
only when the backtest itself ran with `"regime_filter": false`), the system
as designed, the walk-forward and in-sample pattern selections (see below),
grade-A only
(a no-fundamentals proxy of the dashboard grade), "A grades only in caution",
1.5% / 2% / 3% risk (the 25% position cap binds on most trades there), and the
old uncapped notional replay for comparison.

**Grade profiles.** Grade-A only trades the backtest's grade-A replay: setups
below grade A are never ordered, so they never block a later A-grade setup on
the same stock, and the patterns the scanner switches off today are removed
before the pick (like the in-sample enabled-only profile, this list was chosen
with hindsight). "A grades only in caution" trades its own replay with every
pattern (as designed). When a replay is missing or was made with another
switched-off list, the profile falls back to deleting the other grades'
trades from the headline stream and its label says *approximate* (B/C trades
still block A setups there, and hidden patterns are traded).

**Walk-forward vs in-sample.** The walk-forward profile trades the
backtest's walk-forward replay: each year the patterns that *earlier* trades
had shown to have no edge are removed before the best setup is picked, as the
live scanner does, so runner-up setups are traded. The in-sample "enabled
patterns only" profile trades the live-configuration replay (today's
switched-off patterns removed before the pick). When a replay does not match
your current `pattern_tuning` (changed config, or a market kept from an older
run), that profile falls back to deleting the switched-off patterns' trades
from the headline stream and is labelled *approximate*. The hindsight effect
(in-sample minus walk-forward) is always computed like for like: both
re-selected, or both deleted — if only one side has a replay, the other side
is re-run with deletion and the selection method's own effect is reported
separately (`selection_effect_pp`).

The equal-weight "market" benchmark uses the backtest's price repairs (known
unadjusted splits and demergers, one-day spikes, bad joins), so it is measured
on the same prices the trades were filled on.

The drawdown is marked to market: every open position is valued at each
day's close, using the same repaired prices the backtest traded on (so a
split or demerger the free data leaves unadjusted does not show up as a paper
gain). The closed-trade drawdown is kept as `max_drawdown_realized_pct`, and
each profile also reports its annual volatility and how much of the account
was invested on average and at most. Yearly returns are booked by exit year.
The marks come from the price cache as it is when the replay runs; a
stand-alone `python -m atlas2 portfolio` months after the backtest can drift
slightly from the fills (Yahoo re-adjusts old prices for new dividends), so
re-run the backtest first if the cache has been refreshed for a long time.

The portfolio replay was corrected on 2026-09-25 and 2026-09-26. It now reads
the market regime at the previous session's close (the old replay used the
entry day's own close, a one-day look-ahead); when more equally ranked setups
than free slots trigger on one day it picks among them in a fixed
pseudo-random order instead of the order they sat in the trade file, and the
headline is the median of 101 such orderings instead of one; and a position
that exits on a day no longer frees its slot or cash for that same day's
entries. The old replay showed 7.7%/yr "as designed" (drawdown 16.8%) on the
old, pre-fees trade stream. Replayed the same way as today's profiles, that
old stream gives a median of 3.4%/yr (p5–p95 0.5–6.5%, highest of the 101
orderings 7.2%; closed-trade drawdown 20.3%); Balanced (1.5% risk) gives 4.6%
(p5–p95 0.3–9.3%) instead of the old 11.0%. Before the same-day change the
median was 5.5% (p5–p95 2.2–7.3%, highest 8.4%). A full backtest run replays the
old stream again and stores the current figures as `reference_pre_costs` in
`portfolio_sim.json` (shown under the portfolio table). Figures from before
the fees update are therefore not comparable one-to-one.

### Honest limitations

- **Survivorship bias**: the universe is *today's* index members, so stocks
  that were dropped or went bust during the decade are missing. That flatters
  the results. The equal-weight market baseline shares this bias; the index
  baseline does not.
- **Fundamentals, earnings dates and news are not backtested**: there is no
  point-in-time data for them. Grades are approximated without the
  fundamentals term, so grade-A figures are approximate.
- **Dividends on the index**: added as an estimated yield accrued daily. Real
  dividends are lumpy and the yield drifted from about 2% to 1.2% over the
  decade, so single-trade figures are approximate (about ±0.05 points);
  averages are reliable.
- **Intraday order is unknown on daily bars**: if the stop and the target are
  both touched on the same day, the stop is assumed first.
- **Shortlist timing**: the live scanner ranks against that day's universe
  and so does the replay, but a stock that is already in a trade still takes
  a shortlist slot. The alert monitor's top-10 A-grade cut, the grades and
  your own choice of setups are not modelled.
- **R is measured from the actual fill**, while the dashboard sizes from the
  entry level; for setups bought at the next open the real risk averages about
  1.27× the planned risk (up to about 1.7×).
- **Small accounts**: a few very high-priced stocks (NVR, AZO, MRF.NS at
  €10,000) get 0 suggested shares on the dashboard, but the backtest still
  trades them.
- **Currency moves are not modelled.** Returns are in local currency.
- **Holiday filler bars**: flat zero-volume bars on exchange holidays and in
  pre-listing placeholders are treated as normal days (measured effect under
  0.001R per trade); the newest Indian bar is intraday, as it is for the live
  07:30 scan.
- **In-sample**: see *Verdicts* above — only trades after 2026-09-25 are a
  clean test.

## Pattern tuning (evidence-based)

The scanner tunes itself per market from the latest backtest
(`data/backtest_latest.json`). With `auto_disable` on, any pattern whose
verdict in that market is **no edge** is excluded from new setups there. Set
`require_proven_edge: true` to also exclude **unproven** patterns. The verdict
is recomputed with your *current* `config.json` every scan, so changing
`min_trades` or `min_avg_r` (the minimum average R *after fees*) takes effect
without re-running the backtest.

The market tab shows which patterns are switched off, and why, in a banner
above the list: **"Switched off by the backtest"** (hover a pattern for the
verdict's reason) and **"Switched off manually in config.json"** for your own
`pattern_tuning.disabled` entries. It also lists patterns that are still
untested or have too few trades. When the list is empty because patterns are
switched off, it says which source did it (and for backtest verdicts, which
verdict) instead of "That's normal"; `auto_disable=false` is only suggested for
the backtest's own switch-offs.

Every setup also cites its pattern's record in that market, for example "this
pattern's US backtest (10y, after fees, stop/target/40-day exit): 44% wins,
+0.03R avg, -0.4% per trade vs the index, -0.2% vs the equal-weight market —
no edge". Its pattern score is nudged up or down by the after-fees average R.
Both the note and the nudge only appear once the pattern has at least
`min_trades` backtest trades — the same number the verdict uses. The time exit
in the note is the one the stored backtest used; if `backtest.max_hold` in
`config.json` has been changed since, the scan log and the banner warn you to
re-run the backtest (the dashboard's sell-by day and the monitor follow the
config).

Settings live in `config.json → pattern_tuning` (`auto_disable`,
`require_proven_edge`, `min_avg_r`, `min_trades`, `family_alpha`, `n_tests`,
plus a manual `disabled` list per market that always applies). A backtest file
from before the fees update still works with the old rule (average R below
`min_avg_r` over at least `min_trades` trades). A market that is missing from
the backtest file is not tuned at all (the banner says "no backtest for …").
Re-run `python -m atlas2 backtest` every few months so the tuning follows fresh
data.

**Heads-up before you re-run the full backtest**: on the last full run of the
fee-aware backtest (before the 2026-09-25 fixes) every US pattern was *no edge*
after fees. If the re-run agrees, with `auto_disable` on the US tab will show
no setups at all, and Germany / India only Darvas boxes (unproven / too few
trades). The double bottom looks enabled but can never
trade: its target is always closer than its stop (best risk:reward about 0.97),
so the 1.8 minimum filters out every signal. Set
`pattern_tuning.auto_disable=false` if you want to see the setups anyway.

## Position sizing

`config.json` holds your settings (copy `config.example.json` to `config.json`):

```json
{ "capital_eur": 10000, "risk_pct": 1.0, "min_risk_reward": 1.8, "max_per_sector": 2, ... }
```

Suggested size risks `risk_pct` of `capital_eur` per trade (distance from entry to
stop), converted at the current EUR/USD/INR rate, capped at 25% of capital per
position. `max_per_sector` limits how many setups per sector make the active
list (see *Protective guards*). Edit the file and re-scan to change it.

## Commands

```sh
.venv/bin/python -m atlas2 scan          # all markets (or: scan us / de / in)
.venv/bin/python -m atlas2 serve         # dashboard at 127.0.0.1:8899
.venv/bin/python -m atlas2 universe      # refresh index constituent lists
.venv/bin/python -m atlas2 backtest      # 10-year replay of the scanner, after fees, vs the market
.venv/bin/python -m atlas2 portfolio     # replay the backtest trades as one account
```

See **Backtest** above for what the backtest does and how to read it.

Unit tests (synthetic data, never touch `data/`):

```sh
.venv/bin/python -m unittest discover -s tests -t . -v
```

## Data & honesty notes

- Prices and fundamentals come from Yahoo Finance (free, unofficial). Daily bars,
  auto-adjusted for splits/dividends. Occasionally a ticker is missing or stale —
  the scanner just skips it. A few stocks carry unadjusted corporate actions or
  bad prints; the backtest repairs them (see *Backtest*), the live scanner does
  not.
- Universe: S&P 500 + S&P 400 (US), DAX + MDAX + SDAX (Germany, ~120 of 160
  resolved to Xetra tickers), NIFTY 100 + Midcap 150 (India). Refresh with
  `python -m atlas2 universe`.
- This is an **analysis tool, not investment advice**. Patterns fail regularly;
  the edge comes from cutting losers at the stop and letting winners reach the
  target. Nothing here guarantees any return.
