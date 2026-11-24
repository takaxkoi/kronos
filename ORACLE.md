# Kronos Oracle

Your stock-market prediction app built on Kronos. GitHub runs the AI on a schedule for free, the dashboard (Netlify) shows the results, and Telegram sends the alerts.

## How it works

```
GitHub Actions (free, on a timer)
  ├─ Daily scan ~5:15 PM ET ─→ forecasts your watchlist, indices, sectors, crypto, forex, gold
  │                             grades old calls · updates the report card · paper trades · recap
  ├─ Pre-market ~8:40 AM ET ──→ Telegram briefing + Alpaca orders (if switched on)
  ├─ S&P 500 scanner ~3 AM ───→ all ~500 stocks with the fast mini model
  ├─ Crypto board every 2h ───→ next-24-hour crypto forecasts
  ├─ Bot every ~10 min ───────→ answers your Telegram commands + entry/stop/target price alerts
  └─ Retrain, monthly ────────→ custom per-stock models (kept only if they beat the base model)
        │ results are committed to  site/data/  and  state/
        ▼
Dashboard (site/index.html on Netlify) reads them straight from this repo
```

## Go-live (about 15 minutes, all from your phone)

1. **Turn on Actions.** Open the repo's **Actions** tab and tap **"I understand my workflows, go ahead and enable them"**. Scheduled jobs on forks stay off until you do this.
2. **Create the Telegram bot.**
   - Message **@BotFather** and send `/newbot`, then copy the token.
   - Send your new bot any message.
   - Open `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `"chat":{"id": …}`.
3. **Add the secrets.** Go to **Settings → Secrets and variables → Actions → New repository secret**.

   | Secret | Required? | Value |
   |---|---|---|
   | `TELEGRAM_BOT_TOKEN` | yes | from BotFather |
   | `TELEGRAM_CHAT_ID` | yes | your chat id (the bot only ever answers this id) |
   | `PORTFOLIO_HOLDINGS` | optional | `AAPL:10,MSFT:5` (kept out of the public files) |
   | `ALPHAVANTAGE_API_KEY` | optional | your Alpha Vantage key: earnings dates for every stock (free). Add `ALPHAVANTAGE_PREMIUM=true` on a paid key to also pull prices |
   | `FINNHUB_API_KEY` | optional | free finnhub.io key: insider trades + sentiment, analyst trends, earnings surprises, news, metrics |
   | `FRED_API_KEY` | optional | free FRED key: rates, yield curve, credit spreads, inflation, jobs |
   | `ALPACA_KEY_ID` / `ALPACA_SECRET_KEY` | optional | free paper account at alpaca.markets: auto-trading and a backup price feed |
   | `ALPACA_LIVE_CONFIRM` | optional | `YES`. Real-money trading also needs `autotrade.live: true` **and** a passing report card |
   | `HF_TOKEN` / `HF_USER` | optional | free huggingface.co account, used to store your monthly custom models |

4. **First run.** Go to **Actions → Daily scan → Run workflow**. The first run takes about 45–70 min while it downloads the model.
5. **Dashboard.** On Netlify, choose **Add new site → Import from GitHub → kronos**. The publish folder is `site` (`netlify.toml` already sets this). You can also drag and drop the `site` folder. The dashboard reads fresh data from GitHub either way.

## Data Vault (daily capture)

`Data vault (daily capture)` runs every weekday at about 6:40 PM ET and saves a snapshot to `data/daily/YYYY-MM-DD/`. Most of this data gets overwritten by providers, so this history can't be downloaded later.

- **Every ticker:** that day's prices. The first run also saves 10 years of daily history to `data/history/`.
- **Every S&P 500 stock:** fundamentals, short interest, analyst rating and price targets, and ownership.
- **Your core list:** options put/call ratios and implied volatility, analyst upgrades and downgrades, insider trades, earnings surprises, institutions, and news.
- **Alpha Vantage** (free key, 25 calls a day, budgeted by the job):
  - the earnings calendar for every stock
  - market movers
  - news sentiment across the market
  - 2- and 10-year Treasury yields
  - a rotating deep-dive into your core list's earnings history, estimate revisions and insider trades

Read any table across all days with `from app.capture import load_days; load_days("fundamentals")`.

## Website + login (Netlify + Supabase)

- **Landing page** (`site/index.html`) with sign-in. **Dashboard** (`site/app.html`) only opens for signed-in, allow-listed members.
- **Data:** forecast files sit in your Supabase table `oracle_docs`, which only emails in `oracle_members` can read. Every GitHub job pushes changed files there through the `oracle_ingest` function, which needs the `ORACLE_SYNC_TOKEN` secret.
- **Add a member:** in the Supabase SQL editor, run `insert into oracle_members(email) values ('friend@example.com');`. They also need a login in your Supabase project.
- **Hosting:** Netlify site `kronos-oracle`, linked to this repo, with `site` as the publish folder.

## AI research ratings (TradingAgents)

The `AI research ratings` job runs your TradingAgents fork on a few watchlist stocks each weeknight. Four AI analysts (chart, fundamentals, news, sentiment) write reports, a bull and a bear debate them, and a portfolio manager issues **Buy / Overweight / Hold / Underweight / Sell** with a thesis, price target and time horizon. Results show on the dashboard's **AI Research** tab.

- **Key:** one AI key as a GitHub secret. `GOOGLE_API_KEY` (Gemini, free tier at aistudio.google.com/apikey) is checked first, then `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
- **Pace:** 3 stocks a night, rotating through the watchlist. Change `research.per_run` in `app/config.yaml`.
- **On demand:** Actions → AI research ratings → Run workflow, and type tickers.
- **Grading:** each rating is logged in `state/research_log.csv` with the price at the time and graded after 5 trading days against SPY. Treat ratings as unproven until the scorecard says otherwise.

## Research tools added Nov 2026

| Tab | What it does | Built on | Job | Key needed |
|---|---|---|---|---|
| Live Prices | Prices every 30 min; sources take turns so free limits last | Yahoo, Finnhub, Tickerbot, Alpaca, Alpha Vantage | Live prices | none (more sources join as keys are added) |
| Terminal | Open any S&P 500 stock: candles, bull case and bear case, same-business peers, stocks that move with and against it, companies named beside it in the news | data vault + all jobs below | Edge Lab + performance | none |
| News | Headlines per stock, each labelled bullish / bearish / neutral | FinBERT (`ProsusAI/finBERT`) | News desk | none |
| WallStreetBets | Most-mentioned stocks, change vs yesterday, popular posts and their mood | ApeWisdom, Reddit archive | News desk | none (`REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` optional for live Reddit) |
| Lenses | Value, Quality, Growth at a fair price, Trend, Wall Street view scores for every stock | rule-based checklists on vault fundamentals | Edge Lab + performance | none |
| Filings Lab | SEC annual reports turned into factors and tested against the next 12 months of returns | SEC EDGAR | Filings Lab (SEC) | set `filings.contact` in `app/config.yaml` to your name and real email; SEC refuses requests without one |
| Edge Lab | 21 trading rules tested on 535 stocks, judged on newer data they were not picked on | vectorbt | Edge Lab + performance | none |
| Performance | What following the app's calls would have returned vs SPY | quantstats | Edge Lab + performance | none |
| Market Scan | Movers and analyst targets | OpenBB, Tickerbot | Research desk, Market scan | `TICKERBOT_API_KEY` for the scans |

Charts use `tradingview/lightweight-charts` (candles) and `apache/echarts` (everything else).

Not wired: `virattt/ai-hedge-fund` and `AI4Finance-Foundation/FinGPT`. The first needs an AI key plus a paid data key beyond five tickers; the second needs a GPU. The Lenses tab and FinBERT cover the same ground for free.

Check which keys are saved: Actions → Secrets check → Run workflow (it prints names only).

## Change settings

Everything is in `app/config.yaml`, including the watchlist, models, samples, horizon, filters, risk, paper trading and autotrade. You can also manage the watchlist from Telegram with `/watch add TSLA`.

## Telegram commands

`/forecast AAPL` · `/top` · `/mood` · `/sectors` · `/report` · `/paper` · `/portfolio` · `/backtest AAPL` · `/replay AAPL 2025-03-14` · `/journal BUY AAPL 10 185.20 note` · `/watch add|remove|list` · `/crypto`

Commands run through GitHub, so replies take about 5–15 min. For instant replies, run `python -m app.cli bot --live` on any computer that has the same secrets set.

## What the numbers mean

- **Agreement (24/30):** Kronos simulates 30 possible futures. This is how many of them end in the predicted direction.
- **Kronos Score (0–100):** made up of 50% agreement, 30% size of the move compared with normal volatility, and 20% how tightly the futures cluster.
- **Expected range:** 80% of the simulated paths stay inside it.
- **Trade plan:**
  - The entry is a buy (or short) zone.
  - The stop sits beyond the worst 10% of paths and at least 1 ATR (average daily price range) away from the entry.
  - The target is the median best price.
  - A plan only counts as valid if the reward is at least 2× the risk.
- **Earnings pause:** signals switch off when an earnings date falls inside the forecast window.
- **Report card:** every call is logged in `state/predictions.csv` and graded automatically when its window ends. Trust the hit rate and the calibration chart before you trust any single call.

## Backtest results so far (Oct 2026)

`tests/diag_backtest.py` replayed about a year of real daily data: 12 large caps and 624 walk-forward forecasts for each setup.

- **400-day windows made forecasts snap back toward the long-run average.** In the first live run AMD came out −60% in 5 days.
- **Kronos-small with a 120-day window had the least bias,** so it is now the default everywhere.
- **No setup beat a coin flip on 5-day stock direction.** All of them landed at about 48–51%, even when 75% or more of the futures agreed.

Treat every signal as unproven until the live report card says otherwise. Results are in `state/diag_bt.json`. To re-run the test, go to **Actions → Kronos diagnostic → Run workflow** and set the script to `diag_backtest.py`.

## Things to know

- **The repo is public,** so your watchlist, paper trades and journal are visible to anyone. Holdings stay in a secret, and the dashboard shows percentages only.
- **Yahoo Finance sometimes blocks GitHub's servers.** If price data goes missing, add the Alpaca keys and the app will fall back to Alpaca for US stocks.
- **The forecasts are probabilities, not guarantees.** Paper-trade until the report card earns real money.

## Test without internet

`python tests/test_oracle.py` runs the whole pipeline on a tiny dummy model with simulated prices. It only proves the plumbing works.

## Added Nov 21, 2026

| Page | Module | What it does |
|---|---|---|
| Daily brief (home page) | `app/brief.py` | One page of plain sentences: futures, rates, oil, gold, bitcoin, the S&P 500, sectors, mood, earnings, your stocks, ideas, the crowd, paper strategies. Rewritten every 30 minutes in market hours. |
| Fear and greed | `app/feargreed.py` | Own 0 to 100 score from six gauges built on saved prices, tested against what the S&P 500 did next. Bitcoin index from alternative.me (credited). |
| S&P 500 | `app/spx.py` | Chart, lines to watch, breadth, sectors, movers, seasonality, past pullbacks. |
| Strategy league | `app/strategies.py` | Seven published strategies backtested on 10 years, each running a $100,000 paper account. Start date kept in `state/strategies.json`. |
| Report card (top section) | `app/accuracy.py` | Compares graded calls with always saying up, shows where calls go wrong, 10-year base rates. |

All five run in `.github/workflows/brief.yml` (nightly, 6:20 AM Eastern, and whenever the price loop starts).

| Calendar | `app/econ.py` | US economic reports for the next 10 days (tries Forex Factory, then Nasdaq, then Finnhub), plus watchlist earnings dates. |
| S&P 500 (regime section) | `app/regime.py` | Labels the market (calm uptrend to bear market) and shows what happened next in each. |
| Insider trades | `app/insiders.py` | Open-market buys and sells by insiders at the 15 watchlist stocks, last 90 days. |
| Earnings moves | `app/earnings.py` | How each watchlist stock moved the day after each past report. |

Call rules since Nov 22, 2026 (`app/ledger.py: follows_rules`): hourly calls are not recorded or scored, and single stocks are never called down.

## Added Nov 23, 2026

- Landing page (`site/index.html`): live brief in a phone, interactive mood, S&P 500, strategy charts, calculator and ask panel. It reads the public data files straight from this repo on GitHub, so it needs no login and no Supabase access.
- The price loop (`prices.yml`) now runs around the clock: full work in market hours, a brief refresh every third hour otherwise, and it restarts itself every 5.3 hours. It also starts the morning rebuild (`brief.yml`) each weekday.
- AI research uses `gemini-3.5-flash-lite` for both roles and covers the whole watchlist nightly (`GOOGLE_API_KEY` secret is set).
- Netlify still builds from `takaxkoi/kronos`. After any change to `site/app.html` or `site/index.html`, copy both files to that repo and push.
- Economy page (`app/economy.py`): rates, inflation, jobs, growth and stress, a recession watch, and a table of today's conditions against what the S&P 500 did next. FRED refuses GitHub's servers without a key, so it falls back to the US Treasury, BLS, the New York Fed and Yahoo. Add a free `FRED_API_KEY` secret to unlock mortgage rates, jobless claims, GDP, consumer mood, the junk bond spread and financial conditions.
- Insider trades page also shows market-wide cluster buys (two or more insiders buying), read once a day from OpenInsider.
- Two models page (`app/second.py`, job `second.yml`): Google's TimesFM 3.0 (from the `takaxkoi/timesfm` fork) forecasts the same 5 days as Kronos for every S&P 500 stock. Calls are saved in `state/second_log.csv` and graded after 5 trading days; the walk-forward backtest re-runs every Saturday. First backtest (Nov 2026): no skill alone (51.0% right against 53.3% for always-up).
- Ticker tape (in `site/app.html`, every page): a terminal-style strip under the top bar showing the "Markets right now" lines from `brief.json` and every symbol in `quotes.json`. Swipe to drag, tap the amber tag to pause, tap a symbol for price, change, the day's range and links to its page and news. It re-reads both files every 2 minutes, so it is as fresh as the price loop.
- Live TV and audio page (`#tv`, in `site/app.html`): free YouTube live streams (Bloomberg TV, Yahoo Finance, CNBC, Fox Business, NBC News NOW, ABC News Live, CBS News 24/7, LiveNOW from FOX, Sky News, Al Jazeera English) in YouTube's own player by channel id, plus market podcasts read from Apple's podcast directory when a show is tapped. Whatever is on stays in a dock above the page while you move around. No backend job and no key.
- Earnings moves now cover the whole S&P 500 (`app/earnings.py`). Report dates are saved in `data/earnings/dates.csv.gz`; each morning run refreshes the 90 stalest tickers. To refresh all at once: Actions → Daily brief + market insights → Run workflow → set the earnings number to 520.
