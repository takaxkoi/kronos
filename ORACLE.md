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
