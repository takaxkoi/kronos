"""The jobs GitHub Actions runs on a schedule."""
from __future__ import annotations

import csv
import time
from datetime import datetime, timezone

import numpy as np

from . import alerts, ledger
from .backtest import backtest, battle
from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, safe_name, write_json
from .data import asset_class, fetch_history, headlines, next_earnings, option_idea, sector_map, sp500
from .engine import Forecaster, Job
from .market import correlation_alerts, heatmap, mood, portfolio_forecast, sector_rotation
from .paper import autotrade, paper_update
from .signals import analyze, chart_payload

NAMES_FILE = STATE_DIR / "names.json"


def _fc(cfg, scanner=False) -> Forecaster:
    m = cfg["model"]
    if scanner:
        return Forecaster.load(m["scanner_name"], m["scanner_tokenizer"], m["scanner_max_context"])
    return Forecaster.load(m["name"], m["tokenizer"], m["max_context"])


def _run(fc, cfg, hist, tickers, interval, horizon, samples, classes, seed):
    f = cfg["forecast"]
    jobs = [Job(t, hist[t], horizon, interval, classes[t]) for t in tickers if t in hist]
    return fc.run(jobs, samples=samples, lookback=f["lookback"], temperature=f["temperature"], top_p=f["top_p"],
                  batch_size=f["batch_size"], seed=seed)


def summary(a: dict) -> dict:
    keys = ["ticker", "name", "cls", "sector", "last_close", "direction", "p_up", "agreement", "agree_count", "samples",
            "median_return", "p10_return", "p90_return", "score", "flags", "ranges", "paused", "illiquid", "earnings",
            "timeframes", "aligned", "asof", "ensemble"]
    s = {k: a.get(k) for k in keys}
    pl = a["plan"]
    s["plan"] = {k: pl[k] for k in ("side", "valid", "rr", "entry", "entry_low", "entry_high", "stop", "target", "hold_bars")}
    return s


def _names(cfg) -> dict:
    n = {r["ticker"]: r["name"] for r in sp500()}
    n.update({k: v for k, v in (cfg.get("forex") or {}).items()})
    n.update({k: v for k, v in (cfg.get("commodities") or {}).items()})
    n.update({k: f"{v} sector" for k, v in (cfg.get("sectors") or {}).items()})
    n.update({"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones", "IWM": "Russell 2000", "^VIX": "Volatility Index"})
    return n


# ── DAILY SCAN (after the close) ─────────────────────────────
def daily_scan() -> None:
    t0 = time.time()
    cfg = load_config()
    f = cfg["forecast"]
    watch = [t.upper() for t in cfg["watchlist"]]
    holdings = list(cfg["holdings"])
    core = list(dict.fromkeys(watch + holdings + cfg["indices"]))
    other = list(dict.fromkeys(list(cfg["sectors"]) + cfg["leveraged"] + list(cfg["forex"]) + list(cfg["commodities"])
                               + cfg["crypto"] + [cfg["vix"]]))
    other = [t for t in other if t not in core]
    every = core + other
    classes = {t: asset_class(t, cfg) for t in every}
    hist = fetch_history(every, "1d", cfg)
    sectors = sector_map([t for t in every if t in hist], cfg)
    names = _names(cfg)
    fc = _fc(cfg)
    seed = int(datetime.now().strftime("%Y%m%d"))
    paths = _run(fc, cfg, hist, core, "1d", f["horizon"], f["samples"], classes, seed)
    paths.update(_run(fc, cfg, hist, other, "1d", f["horizon"], max(12, f["samples"] // 2), classes, seed + 1))
    # tickers with their own fine-tuned model (monthly retrain) get re-forecast with it
    for t, mc in (cfg["model"].get("custom") or {}).items():
        if t in hist:
            try:
                cf = Forecaster.load(mc["name"], mc["tokenizer"], mc["max_context"])
                paths.update(_run(cf, cfg, hist, [t], "1d", f["horizon"], f["samples"], classes, seed))
            except Exception as ex:
                log(f"custom model for {t} failed: {ex}")
    log(f"daily forecasts done: {len(paths)} in {time.time() - t0:.0f}s")

    # weekly (month range) + hourly timeframes for the core list
    wk_hist = fetch_history(core, "1wk", cfg)
    wk = _run(fc, cfg, wk_hist, core, "1wk", f["weekly_horizon"], 16, classes, seed + 2)
    hr_hist = fetch_history([t for t in core if classes[t] == "stock"], "1h", cfg)
    hr = _run(fc, cfg, hr_hist, list(hr_hist), "1h", f["hourly_horizon"], 16, classes, seed + 3)
    log(f"multi-timeframe done in {time.time() - t0:.0f}s")

    earn = {t: next_earnings(t) for t in core if classes[t] == "stock"}
    out: dict[str, dict] = {}
    for t, p in paths.items():
        a = analyze(p, cfg, earnings=earn.get(t), sector=sectors.get(t))
        a["name"] = names.get(t, t)
        tf = {"1d": a["direction"]}
        if t in wk:
            w = analyze(wk[t], cfg)
            a["ranges"]["month"] = w["ranges"]["horizon"]
            tf["1wk"] = w["direction"]
            a["weekly"] = {k: w[k] for k in ("direction", "p_up", "agreement", "median_return", "score")}
        if t in hr:
            hh = analyze(hr[t], cfg)
            tf["1h"] = hh["direction"]
            a["hourly"] = {k: hh[k] for k in ("direction", "p_up", "agreement", "median_return", "score")}
        a["timeframes"] = tf
        a["aligned"] = len(tf) == 3 and len(set(tf.values())) == 1
        out[t] = a
    vix = float(hist[cfg["vix"]]["close"].iloc[-1]) if cfg["vix"] in hist else None

    # extras for the watchlist: headlines + option ideas for valid setups
    for t in watch:
        a = out.get(t)
        if not a:
            continue
        a["news"] = headlines(t)
        if a["plan"]["valid"] and a["cls"] == "stock":
            a["option"] = option_idea(t, a["direction"], a["plan"]["target"], a["last_close"], a["plan"]["hold_bars"])

    # ledger: daily + weekly + hourly calls all get graded later
    ledger.record(list(out.values()), vix)
    ledger.record([analyze(p, cfg, sector=sectors.get(t)) for t, p in wk.items()], vix)
    ledger.record([analyze(p, cfg, sector=sectors.get(t)) for t, p in hr.items()], vix)
    ledger.grade(cfg)

    port = portfolio_forecast(cfg["holdings"], paths, cfg) if cfg["holdings"] else None
    paper = paper_update(list(out.values()), hist, cfg)

    # per-ticker detail files
    for t, p in paths.items():
        d = dict(out[t])
        d["chart"] = chart_payload(p)
        if t in wk:
            d["chart_weekly"] = chart_payload(wk[t], bars=60)
        if t in hr:
            d["chart_hourly"] = chart_payload(hr[t], bars=70)
        write_json(SITE_DATA / "t" / f"{safe_name(t)}.json", d)

    scanner = read_json(STATE_DIR / "scanner_latest.json", {})
    universe = list(out.values()) + [a for a in scanner.get("analyses", []) if a["ticker"] not in out]
    corr = correlation_alerts(paths, out, core + list(cfg["sectors"]))
    report = ledger.report(cfg)
    write_json(SITE_DATA / "report.json", report)
    write_json(STATE_DIR / "latest_analyses.json", {"asof": datetime.now(timezone.utc).isoformat(), "vix": vix,
                                                     "analyses": [summary(a) for a in out.values()]})
    _export_latest(cfg, out, universe, scanner, vix, corr, port, paper, report)
    alerts.signal_alerts([out[t] for t in core if t in out], cfg)
    after_close_recap(cfg, out, paper, report, port)
    log(f"daily scan finished in {time.time() - t0:.0f}s")


def _export_latest(cfg, out, universe, scanner, vix, corr, port, paper, report):
    watch = [t.upper() for t in cfg["watchlist"]]
    etfs = set(cfg["indices"]) | set(cfg["sectors"]) | set(cfg["leveraged"])
    universe = [a for a in universe if a["ticker"] not in etfs]  # single stocks only for mood, heatmap, top setups
    stocks = [a for a in universe if a["cls"] == "stock" and not a.get("illiquid") and not a.get("paused")]
    longs = sorted([a for a in stocks if a["direction"] == "up" and a["plan"]["valid"]], key=lambda a: -a["score"])[:12]
    shorts = sorted([a for a in stocks if a["direction"] == "down" and a["plan"]["valid"]], key=lambda a: -a["score"])[:12]

    def pick(lst):
        return [summary(out[t]) for t in lst if t in out]

    def flagged(flag):
        return [summary(a) if "plan" in a else a for a in universe if flag in a.get("flags", []) and not a.get("illiquid")][:20]

    data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "model": cfg["model"]["name"], "horizon": cfg["forecast"]["horizon"], "samples": cfg["forecast"]["samples"],
        "mood": mood(universe), "vix": {"level": vix, **(summary(out[cfg["vix"]]) if cfg["vix"] in out else {})},
        "indices": pick(cfg["indices"]), "watchlist": pick(watch),
        "top_long": [summary(a) if "plan" in a else a for a in longs], "top_short": [summary(a) if "plan" in a else a for a in shorts],
        "sectors": sector_rotation(out, cfg), "leveraged": pick(cfg["leveraged"]),
        "crypto": pick(cfg["crypto"]), "forex": pick(list(cfg["forex"])), "commodities": pick(list(cfg["commodities"])),
        "signals": {k: flagged(k) for k in ["breakout", "breakdown", "reversal_up", "reversal_down", "momentum_up",
                                            "momentum_down", "gap_up", "gap_down", "volume_surge", "volatility"]},
        "earnings_paused": [summary(a) for a in out.values() if a.get("paused")],
        "aligned": [summary(a) for a in out.values() if a.get("aligned")],
        "correlations": corr,
        "heatmap": heatmap(universe), "scanner_asof": scanner.get("asof"),
        "report": {k: report.get(k) for k in ("n", "hit_rate", "hit_rate_high", "pending", "streak")},
        "paper": paper.get("summary"),
        "portfolio": None if not port else {"median_return": port["median_return"], "p_down": port["p_down"],
                                            "holdings": [{"ticker": h["ticker"], "ret": h["ret"],
                                                          "weight": h["value"] / port["value"]} for h in port["holdings"]],
                                            "hedge": {k: v for k, v in (port.get("hedge") or {}).items()
                                                      if k in ("index", "inverse_etf", "beta", "corr")} or None},
        "autotrade": read_json(STATE_DIR / "autotrade.json", {"status": "off"}),
    }
    write_json(SITE_DATA / "latest.json", data)
    write_json(SITE_DATA / "paper.json", {k: paper[k] for k in ("positions", "closed", "equity", "summary") if k in paper})
    write_json(SITE_DATA / "journal.json", read_journal())


# ── S&P 500 OVERNIGHT SCANNER ────────────────────────────────
def sp500_scan() -> None:
    t0 = time.time()
    cfg = load_config()
    rows = sp500()
    if not rows:
        log("no S&P 500 list available")
        return
    tickers = [r["ticker"] for r in rows]
    sectors = {r["ticker"]: r["sector"] for r in rows}
    names = {r["ticker"]: r["name"] for r in rows}
    hist = fetch_history(tickers, "1d", cfg)
    fc = _fc(cfg, scanner=True)
    paths = _run(fc, cfg, hist, list(hist), "1d", cfg["forecast"]["horizon"], cfg["scanner"]["samples"],
                 {t: "stock" for t in hist}, int(datetime.now().strftime("%Y%m%d")))
    res = []
    for t, p in paths.items():
        a = analyze(p, cfg, sector=sectors.get(t))
        a["name"] = names.get(t, t)
        res.append(a)
    ledger.record(res, None, detail=False)
    write_json(STATE_DIR / "scanner_latest.json", {"asof": datetime.now(timezone.utc).isoformat(),
                                                    "analyses": [summary(a) for a in res]})
    ok = [a for a in res if not a["illiquid"]]
    write_json(SITE_DATA / "scanner.json", {
        "generated": datetime.now(timezone.utc).isoformat(), "model": fc.name, "n": len(res),
        "mood": mood(res), "heatmap": heatmap(res),
        "top_long": [summary(a) for a in sorted([a for a in ok if a["direction"] == "up"], key=lambda a: -a["score"])[:25]],
        "top_short": [summary(a) for a in sorted([a for a in ok if a["direction"] == "down"], key=lambda a: -a["score"])[:25]],
    })
    log(f"S&P 500 scan: {len(res)} stocks in {time.time() - t0:.0f}s")


# ── CRYPTO HOURLY ────────────────────────────────────────────
def crypto_hourly() -> None:
    cfg = load_config()
    tick = cfg["crypto"]
    hist = fetch_history(tick, "1h", cfg)
    fc = _fc(cfg, scanner=True)
    paths = _run(fc, cfg, hist, list(hist), "1h", cfg["forecast"]["crypto_hourly_horizon"], 12,
                 {t: "crypto" for t in hist}, int(time.time() // 3600))
    res = [analyze(p, cfg, sector="Crypto") for p in paths.values()]
    ledger.record(res, None, detail=True)
    for t, p in paths.items():
        d = next(a for a in res if a["ticker"] == t)
        write_json(SITE_DATA / "t" / f"{safe_name(t)}_1h.json", {**d, "chart": chart_payload(p, bars=72)})
    write_json(SITE_DATA / "crypto_hourly.json", {"generated": datetime.now(timezone.utc).isoformat(),
                                                   "rows": [summary(a) for a in sorted(res, key=lambda a: -a["score"])]})
    alerts.signal_alerts(res, cfg)


# ── BRIEFINGS ────────────────────────────────────────────────
def premarket() -> None:
    """8:30 AM ET: briefing + Alpaca orders from last night's forecasts (no model needed)."""
    cfg = load_config()
    latest = read_json(SITE_DATA / "latest.json", {})
    if not latest:
        log("no forecasts yet")
        return
    at = autotrade([_full(a) for a in latest.get("top_long", [])], cfg, latest.get("generated"))
    write_json(STATE_DIR / "autotrade.json", at)
    m, ix = latest.get("mood") or {}, latest.get("indices", [])
    lines = [f"☀️ <b>PRE-MARKET BRIEFING</b> · {datetime.now().strftime('%a %b %d')}",
             f"Mood: <b>{m.get('label', '—')}</b> · {m.get('pct_up', 0):.0%} of stocks predicted up · VIX {alerts.money((latest.get('vix') or {}).get('level'))}", ""]
    for a in ix:
        lines.append(f"{a['ticker']}: {'▲' if a['direction'] == 'up' else '▼'} {alerts.pct(a['median_return'])} ({a['agree_count']}/{a['samples']})")
    lines += ["", "<b>TOP 5 LONGS</b>"] + [alerts.setup_line(_full(a)) for a in latest.get("top_long", [])[:5]]
    lines += ["", "<b>TOP 5 SHORTS</b>"] + [alerts.setup_line(_full(a)) for a in latest.get("top_short", [])[:5]]
    gaps = latest.get("signals", {}).get("gap_up", [])[:5] + latest.get("signals", {}).get("gap_down", [])[:5]
    if gaps:
        lines += ["", "<b>GAP CALLS</b>"] + [f"{g['ticker']} {'⬆️' if 'gap_up' in g['flags'] else '⬇️'}" for g in gaps]
    paused = latest.get("earnings_paused", [])
    if paused:
        lines += ["", "📅 Paused for earnings: " + ", ".join(f"{p['ticker']} ({p['earnings']})" for p in paused)]
    if at.get("status") not in ("off", None):
        lines += ["", f"🤖 Auto-trade: {at.get('status')} {at.get('mode', '')} {at.get('reason', '')}"]
    lines.append("\n<i>Forecasts, not guarantees. Check the report card before sizing up.</i>")
    alerts.send("\n".join(lines))


def after_close_recap(cfg, out, paper, report, port=None) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    graded = ledger.graded_on(today)
    lines = [f"🌙 <b>AFTER-CLOSE RECAP</b> · {datetime.now().strftime('%a %b %d')}"]
    if graded:
        ok = sum(str(g["correct"]).lower() == "true" for g in graded)
        lines.append(f"Calls graded today: <b>{ok}/{len(graded)}</b> right")
        best = sorted(graded, key=lambda g: -(float(g["actual_return"]) * (1 if g["direction"] == "up" else -1)))[:3]
        for g in best:
            lines.append(f"  ✓ {g['ticker']} called {g['direction']} → {alerts.pct(float(g['actual_return']))}")
    if report.get("n"):
        st = report["streak"]
        lines.append(f"Report card: {report['hit_rate']:.0%} hit rate over {report['n']} calls · "
                     f"high-score {alerts.pct(report.get('hit_rate_high')).lstrip('+') if report.get('hit_rate_high') is not None else '—'} · "
                     f"streak {st['current']} {st['current_type']}{'s' if st['current'] != 1 else ''}")
    s = paper.get("summary") or {}
    if s:
        lines.append(f"Paper account: ${s['equity']:,.0f} ({alerts.pct(s['return'])}) vs SPY {alerts.pct(s.get('spy_return'))} · "
                     f"{s['open']} open")
    al = [a for a in out.values() if a.get("aligned") and a["score"] >= 60]
    if al:
        lines.append("All timeframes agree: " + ", ".join(f"{a['ticker']} {'▲' if a['direction'] == 'up' else '▼'}" for a in al[:8]))
    if port:
        lines += [f"💼 Portfolio: ${port['value']:,.0f} → ${port['pred_median']:,.0f} predicted "
                  f"({alerts.pct(port['median_return'])}, range ${port['pred_p10']:,.0f}–${port['pred_p90']:,.0f})"]
        if port.get("hedge"):
            lines.append("🛡 " + port["hedge"]["note"])
    alerts.send("\n".join(lines))


def _full(s: dict) -> dict:
    s = dict(s)
    s.setdefault("plan", {})
    s["plan"].setdefault("rr", 0)
    return s


# ── PRICE WATCH (every 15 min in market hours) ───────────────
def price_watch() -> None:
    from .data import latest_prices
    latest = read_json(SITE_DATA / "latest.json", {})
    setups = [a for a in latest.get("watchlist", []) + latest.get("top_long", [])[:8] + latest.get("top_short", [])[:8]
              if a.get("plan", {}).get("valid")]
    if not setups:
        return
    uniq = {a["ticker"]: a for a in setups}
    n = alerts.price_hits(list(uniq.values()), latest_prices(list(uniq)))
    log(f"price watch: {n} alerts")


# ── BACKTEST LAB (manual) ────────────────────────────────────
def backtest_lab(tickers: list[str] | None = None, models: bool = False) -> None:
    cfg = load_config()
    tickers = tickers or [t.upper() for t in cfg["watchlist"]][:8] + ["SPY"]
    classes = {t: asset_class(t, cfg) for t in tickers}
    hist = fetch_history(tickers, "1d", cfg)
    fc = _fc(cfg)
    results = []
    for t in tickers:
        if t in hist:
            try:
                r = backtest(t, hist[t], fc, cfg, classes[t])
            except ValueError as ex:
                log(str(ex))
                continue
            results.append(r)
            write_json(SITE_DATA / "bt" / f"{safe_name(t)}.json", r)
            log(f"backtest {t}: kronos {r['kronos']['total_return']:+.1%} vs buy&hold {r['buy_hold']['total_return']:+.1%}")
    idx = {"generated": datetime.now(timezone.utc).isoformat(), "model": fc.name,
           "rows": [{k: r[k] for k in ("ticker", "n_trades", "win_rate", "hit_rate", "mae", "beats_buy_hold", "n_forecasts")}
                    | {"kronos": r["kronos"]["total_return"], "buy_hold": r["buy_hold"]["total_return"],
                       "sma_cross": r["sma_cross"]["total_return"], "max_dd": r["kronos"]["max_drawdown"],
                       "sharpe": r["kronos"]["sharpe"]} for r in results]}
    old = read_json(SITE_DATA / "bt" / "index.json", {})
    if models:
        idx["battle"] = battle(tickers[:5], hist, cfg, classes)
    elif old.get("battle"):
        idx["battle"] = old["battle"]
    write_json(SITE_DATA / "bt" / "index.json", idx)


# ── JOURNAL ──────────────────────────────────────────────────
JOURNAL = STATE_DIR / "journal.csv"


def add_journal(side: str, ticker: str, qty: float, price: float, note: str = "") -> dict:
    latest = read_json(STATE_DIR / "latest_analyses.json", {})
    call = next((a for a in latest.get("analyses", []) if a["ticker"] == ticker), None)
    row = {"time": datetime.now(timezone.utc).isoformat(), "side": side.upper(), "ticker": ticker, "qty": qty, "price": price,
           "note": note, "kronos_dir": call["direction"] if call else "", "kronos_score": call["score"] if call else "",
           "kronos_agree": f"{call['agree_count']}/{call['samples']}" if call else "",
           "followed": (call is not None and ((side.upper() == "BUY") == (call["direction"] == "up")))}
    new = not JOURNAL.exists()
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    write_json(SITE_DATA / "journal.json", read_journal())
    return row


def read_journal() -> list[dict]:
    if not JOURNAL.exists():
        return []
    with open(JOURNAL) as fh:
        return list(csv.DictReader(fh))[-300:]

