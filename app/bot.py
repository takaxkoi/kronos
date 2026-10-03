"""Telegram bot. Only answers YOUR chat id. Runs from GitHub every ~10 min, or instantly with `python -m app.cli bot --live`."""
from __future__ import annotations

import os
import re
import time

import requests

from . import alerts
from .core import CONFIG_PATH, SITE_DATA, STATE_DIR, load_config, log, read_json, write_json
from .alerts import e, money, pct

BOT_FILE = STATE_DIR / "bot.json"

HELP = """<b>KRONOS ORACLE</b>
/forecast AAPL — live forecast + chart
/top — today's best long & short setups
/mood — market mood, indices, VIX
/sectors — sector rotation forecast
/report — report card (is it actually right?)
/paper — paper trading account
/portfolio — forecast for your holdings
/backtest AAPL — 1-year walk-forward backtest
/replay AAPL 2025-03-14 — what Kronos would have said that day
/journal BUY AAPL 10 185.20 note — log a real trade
/watch add TSLA · /watch remove TSLA · /watch list
/crypto — hourly crypto board"""


def _updates(offset: int, timeout: int = 0) -> list[dict]:
    r = requests.get(alerts.API.format(os.environ["TELEGRAM_BOT_TOKEN"], "getUpdates"),
                     params={"offset": offset, "timeout": timeout}, timeout=timeout + 15)
    return r.json().get("result", []) if r.ok else []


def poll(live: bool = False) -> int:
    if not alerts.enabled():
        log("telegram secrets missing — bot idle")
        return 0
    owner = str(os.environ["TELEGRAM_CHAT_ID"])
    st = read_json(BOT_FILE, {"offset": 0})
    handled = 0
    while True:
        ups = _updates(st["offset"], timeout=50 if live else 0)
        for u in ups:
            st["offset"] = u["update_id"] + 1
            msg = u.get("message") or u.get("edited_message") or {}
            chat = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if chat != owner or not text.startswith("/"):
                continue
            write_json(BOT_FILE, st)  # save first so a crashed/timed-out command never replays
            try:
                handle(text)
            except Exception as ex:  # never let one bad command kill the bot
                alerts.send(f"⚠️ {e(type(ex).__name__)}: {e(ex)}")
            handled += 1
        write_json(BOT_FILE, st)
        if not live:
            break
    return handled


TICKER = re.compile(r"^[A-Z0-9.^=\-]{1,15}$")


def handle(text: str) -> None:
    parts = text.split()
    cmd, args = parts[0].split("@")[0].lower(), parts[1:]
    if cmd in ("/forecast", "/backtest", "/replay") and args and not TICKER.match(args[0].upper()):
        alerts.send("That doesn't look like a ticker.")
        return
    if cmd == "/journal" and len(args) >= 2 and not TICKER.match(args[1].upper()):
        alerts.send("That doesn't look like a ticker.")
        return
    latest = read_json(SITE_DATA / "latest.json", {})
    if cmd in ("/start", "/help"):
        alerts.send(HELP)
    elif cmd == "/forecast" and args:
        forecast_cmd(args[0].upper())
    elif cmd == "/top":
        lines = ["<b>TOP LONGS</b>"] + [alerts.setup_line(a) for a in latest.get("top_long", [])[:6]]
        lines += ["", "<b>TOP SHORTS</b>"] + [alerts.setup_line(a) for a in latest.get("top_short", [])[:6]]
        alerts.send("\n".join(lines))
    elif cmd == "/mood":
        m = latest.get("mood") or {}
        lines = [f"<b>{m.get('label', '—')}</b> · {m.get('pct_up', 0):.0%} of {m.get('n', 0)} stocks predicted up",
                 f"VIX {money((latest.get('vix') or {}).get('level'))} → {pct((latest.get('vix') or {}).get('median_return'))}"]
        for a in latest.get("indices", []):
            lines.append(f"{a['ticker']} {'▲' if a['direction'] == 'up' else '▼'} {pct(a['median_return'])} "
                         f"({a['agree_count']}/{a['samples']}) range {money(a['ranges']['week']['low'])}–{money(a['ranges']['week']['high'])}")
        alerts.send("\n".join(lines))
    elif cmd == "/sectors":
        alerts.send("<b>SECTOR ROTATION (next week)</b>\n" + "\n".join(
            f"{i + 1}. {s['name']} ({s['etf']}) {pct(s['median_return'])}" for i, s in enumerate(latest.get("sectors", []))))
    elif cmd == "/report":
        r = read_json(SITE_DATA / "report.json", {})
        if not r.get("n"):
            alerts.send(f"No graded calls yet ({r.get('pending', 0)} waiting). Give it a week.")
            return
        cal = " · ".join(f"{c['bucket']}%→{c['actual']:.0%}" for c in r.get("calibration", []))
        lb = ", ".join(f"{x['key']} {x['hit_rate']:.0%}" for x in r.get("leaderboard", [])[:5])
        alerts.send(f"<b>REPORT CARD</b>\nHit rate {r['hit_rate']:.0%} over {r['n']} calls\n"
                    f"High-score calls: {pct(r.get('hit_rate_high')).lstrip('+')} ({r.get('n_high')})\n"
                    f"Calibration: {cal}\nBest: {lb}\nStreak: {r['streak']['current']} {r['streak']['current_type']}")
    elif cmd == "/paper":
        p = read_json(SITE_DATA / "paper.json", {})
        s = p.get("summary") or {}
        if not s:
            alerts.send("Paper account starts after the first daily scan.")
            return
        lines = [f"<b>PAPER</b> ${s['equity']:,.0f} ({pct(s['return'])}) vs SPY {pct(s.get('spy_return'))}"]
        for t, pos in p.get("positions", {}).items():
            ch = (pos["price"] / pos["entry"] - 1) * (1 if pos["side"] == "long" else -1)
            lines.append(f"{t} {pos['side']} {pos['shares']} @ {money(pos['entry'])} → {money(pos['price'])} ({pct(ch)})")
        alerts.send("\n".join(lines))
    elif cmd == "/portfolio":
        portfolio_cmd()
    elif cmd == "/backtest" and args:
        backtest_cmd(args[0].upper())
    elif cmd == "/replay" and len(args) >= 2:
        replay_cmd(args[0].upper(), args[1])
    elif cmd == "/journal" and len(args) >= 4:
        from .scan import add_journal
        row = add_journal(args[0], args[1].upper(), float(args[2]), float(args[3]), " ".join(args[4:]))
        alerts.send(f"📝 Logged {row['side']} {row['qty']} {row['ticker']} @ {row['price']}. "
                    f"Kronos said: {row['kronos_dir'] or 'n/a'} {row['kronos_agree']} · "
                    f"{'✅ with' if row['followed'] else '⚠️ against'} the call")
    elif cmd == "/watch":
        watch_cmd(args)
    elif cmd == "/crypto":
        c = read_json(SITE_DATA / "crypto_hourly.json", {})
        alerts.send("<b>CRYPTO · next 24h</b>\n" + "\n".join(
            f"{a['ticker'].replace('-USD', '')} {'▲' if a['direction'] == 'up' else '▼'} {pct(a['median_return'])} "
            f"· {a['score']} ({a['agree_count']}/{a['samples']})" for a in c.get("rows", [])[:20]) or "Crypto board not run yet.")
    else:
        alerts.send(HELP)


def ensure_torch() -> None:
    """The 10-minute bot job starts light; install PyTorch only when a command needs the model."""
    import importlib.util
    import subprocess
    import sys
    if importlib.util.find_spec("torch") is None:
        alerts.send("⚙️ Warming up the model (first heavy command this run)…")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "--index-url",
                        "https://download.pytorch.org/whl/cpu"], check=True)


def _one(ticker: str):
    from .data import asset_class, fetch_history, next_earnings
    from .engine import Forecaster, Job
    from .signals import analyze, chart_payload
    cfg = load_config()
    m, f = cfg["model"], cfg["forecast"]
    cls = asset_class(ticker, cfg)
    hist = fetch_history([ticker], "1d", cfg)
    if ticker not in hist:
        raise ValueError(f"no price data for {ticker}")
    fc = Forecaster.load(m["name"], m["tokenizer"], m["max_context"])
    p = fc.run([Job(ticker, hist[ticker], f["horizon"], "1d", cls)], samples=f["samples"], lookback=f["lookback"],
               batch_size=f["batch_size"])[ticker]
    a = analyze(p, cfg, earnings=next_earnings(ticker) if cls == "stock" else None)
    return cfg, hist, fc, a, chart_payload(p)


def forecast_cmd(ticker: str) -> None:
    ensure_torch()
    from .charts import forecast_png
    alerts.send(f"⏳ Running Kronos on {e(ticker)}…")
    cfg, _, _, a, ch = _one(ticker)
    r = a["ranges"]
    cap = (alerts.setup_line(a) + f"\nNext: {money(r['next']['low'])}–{money(r['next']['high'])} · "
           f"Week: {money(r['week']['low'])}–{money(r['week']['high'])}\n"
           f"2nd opinion (RSI/trend/MACD): {a['ensemble']['agree']}/3 agree")
    if a.get("earnings"):
        cap += f"\n📅 Earnings {a['earnings']}" + (" — signal paused" if a["paused"] else "")
    if not alerts.send_photo(forecast_png(ch, a), cap):
        alerts.send(cap)


def portfolio_cmd() -> None:
    ensure_torch()
    from .data import asset_class, fetch_history
    from .engine import Forecaster, Job
    from .market import portfolio_forecast
    cfg = load_config()
    if not cfg["holdings"]:
        alerts.send("Add your holdings in the PORTFOLIO_HOLDINGS secret, e.g. <code>AAPL:10,MSFT:5</code>")
        return
    alerts.send("⏳ Forecasting your portfolio…")
    tick = list(cfg["holdings"]) + ["SPY", "QQQ"]
    hist = fetch_history(tick, "1d", cfg)
    m, f = cfg["model"], cfg["forecast"]
    fc = Forecaster.load(m["name"], m["tokenizer"], m["max_context"])
    paths = fc.run([Job(t, hist[t], f["horizon"], "1d", asset_class(t, cfg)) for t in hist], samples=f["samples"],
                   lookback=f["lookback"], batch_size=f["batch_size"])
    p = portfolio_forecast(cfg["holdings"], paths, cfg)
    if not p:
        alerts.send("No data for your holdings.")
        return
    lines = [f"💼 <b>PORTFOLIO</b> ${p['value']:,.0f} → ${p['pred_median']:,.0f} ({pct(p['median_return'])})",
             f"Range ${p['pred_p10']:,.0f} – ${p['pred_p90']:,.0f} · chance of a down week {p['p_down']:.0%}"]
    lines += [f"{h['ticker']}: {pct(h['ret'])}" for h in p["holdings"]]
    if p.get("hedge"):
        lines.append("🛡 " + p["hedge"]["note"])
    alerts.send("\n".join(lines))


def backtest_cmd(ticker: str) -> None:
    ensure_torch()
    from .backtest import backtest
    from .data import asset_class, fetch_history
    from .engine import Forecaster
    alerts.send(f"⏳ Backtesting {e(ticker)} (takes a few minutes)…")
    cfg = load_config()
    hist = fetch_history([ticker], "1d", cfg)
    m = cfg["model"]
    r = backtest(ticker, hist[ticker], Forecaster.load(m["name"], m["tokenizer"], m["max_context"]), cfg, asset_class(ticker, cfg))
    alerts.send(f"<b>BACKTEST {e(ticker)}</b> ({len(r['dates'])} days)\n"
                f"Kronos: {pct(r['kronos']['total_return'])} · max DD {pct(r['kronos']['max_drawdown'])} · {r['n_trades']} trades · "
                f"win {pct(r['win_rate']).lstrip('+') if r['win_rate'] is not None else '—'}\n"
                f"Buy & hold: {pct(r['buy_hold']['total_return'])}\nSMA 20/50: {pct(r['sma_cross']['total_return'])}\n"
                f"Direction hit rate: {pct(r['hit_rate']).lstrip('+') if r['hit_rate'] is not None else '—'}")


def replay_cmd(ticker: str, date: str) -> None:
    ensure_torch()
    from .backtest import replay
    from .charts import forecast_png
    from .data import asset_class, fetch_history
    from .engine import Forecaster
    cfg = load_config()
    hist = fetch_history([ticker], "1d", cfg)
    m = cfg["model"]
    a = replay(ticker, hist[ticker], date, Forecaster.load(m["name"], m["tokenizer"], m["max_context"]), cfg, asset_class(ticker, cfg))
    cap = (f"⏪ <b>{e(ticker)} as of {e(a['asof'][:10])}</b>\nKronos said {a['direction']} {pct(a['median_return'])} "
           f"({a['agree_count']}/{a['samples']})\nActually: {pct(a.get('actual_return'))} "
           f"{'✅' if a.get('correct') else '❌' if 'correct' in a else ''}")
    if not alerts.send_photo(forecast_png(a["chart"], a), cap):
        alerts.send(cap)


def watch_cmd(args: list[str]) -> None:
    cfg = load_config()
    wl = [t.upper() for t in cfg["watchlist"]]
    if not args or args[0] == "list":
        alerts.send("👀 " + ", ".join(wl))
        return
    if len(args) < 2:
        alerts.send("Usage: /watch add TSLA or /watch remove TSLA")
        return
    t = args[1].upper()
    if not TICKER.match(t):
        alerts.send("That doesn't look like a ticker.")
        return
    if args[0] == "add" and t not in wl:
        wl.append(t)
    elif args[0] in ("remove", "rm", "del") and t in wl:
        wl.remove(t)
    text = CONFIG_PATH.read_text()
    text = re.sub(r"^watchlist:.*(?:\n[ \t]*- .*)*", "watchlist: [" + ", ".join(wl) + "]", text, count=1, flags=re.M)
    CONFIG_PATH.write_text(text)
    alerts.send("👀 Watchlist: " + ", ".join(wl) + "\n(takes effect on the next scan)")


if __name__ == "__main__":
    while True:
        poll(live=True)
        time.sleep(1)
