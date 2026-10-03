"""Telegram delivery: signal alerts, price-hit alerts, briefings."""
from __future__ import annotations

import html
import os
import time

import requests

from .core import STATE_DIR, log, read_json, write_json

_JOB = "".join(ch for ch in os.environ.get("GITHUB_WORKFLOW", "local").lower() if ch.isalnum())[:24] or "local"
ALERT_FILE = STATE_DIR / f"alerts_{_JOB}.json"
API = "https://api.telegram.org/bot{}/{}"


def enabled() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"))


def send(text: str, chat_id: str | None = None) -> bool:
    if not enabled():
        log("telegram not configured; message:\n" + text)
        return False
    chat = chat_id or os.environ["TELEGRAM_CHAT_ID"]
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)] or [""]:
        try:
            r = requests.post(API.format(os.environ["TELEGRAM_BOT_TOKEN"], "sendMessage"), timeout=20,
                              data={"chat_id": chat, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": "true"})
            if not r.ok:
                log(f"telegram error: {r.text[:200]}")
                return False
        except Exception as e:
            log(f"telegram failed: {e}")
            return False
    return True


def send_photo(png: bytes, caption: str = "", chat_id: str | None = None) -> bool:
    if not enabled():
        return False
    try:
        r = requests.post(API.format(os.environ["TELEGRAM_BOT_TOKEN"], "sendPhoto"), timeout=30,
                          data={"chat_id": chat_id or os.environ["TELEGRAM_CHAT_ID"], "caption": caption[:1000], "parse_mode": "HTML"},
                          files={"photo": ("chart.png", png, "image/png")})
        return r.ok
    except Exception:
        return False


def once(key: str) -> bool:
    """True the first time a key is seen (dedupes alerts across runs)."""
    seen = read_json(ALERT_FILE, {})
    now = time.time()
    seen = {k: v for k, v in seen.items() if now - v < 14 * 86400}
    if key in seen:
        return False
    seen[key] = now
    write_json(ALERT_FILE, seen)
    return True


def e(s) -> str:
    return html.escape(str(s))


def pct(x) -> str:
    return "—" if x is None else f"{x:+.2%}"


def money(x) -> str:
    if x is None:
        return "—"
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.5f}"


FLAG_TEXT = {"breakout": "🚀 breakout", "breakdown": "🕳 breakdown", "reversal_up": "↩️ bounce", "reversal_down": "↪️ rollover",
             "momentum_up": "📈 momentum", "momentum_down": "📉 momentum", "gap_up": "⬆️ gap up", "gap_down": "⬇️ gap down",
             "volume_surge": "🔊 volume surge", "volatility": "⚡ big swings", "earnings": "📅 earnings — paused",
             "illiquid": "🧊 thin volume", "leveraged": "⚠️ leveraged"}


def setup_line(a: dict) -> str:
    pl = a["plan"]
    arrow = "🟢 LONG" if a["direction"] == "up" else "🔴 SHORT"
    flags = " · ".join(FLAG_TEXT.get(f, f) for f in a["flags"])
    s = (f"<b>{e(a['ticker'])}</b> {arrow}  <b>{a['score']}</b>/100  ({a['agree_count']}/{a['samples']} agree)\n"
         f"   move {pct(a['median_return'])} · entry {money(pl['entry_low'])}–{money(pl['entry_high'])} · "
         f"stop {money(pl['stop'])} · target {money(pl['target'])} · R:R {pl['rr']:.1f}")
    return s + (f"\n   {flags}" if flags else "")


def signal_alerts(analyses: list[dict], cfg: dict) -> int:
    sc = cfg.get("signals", {})
    n = 0
    for a in analyses:
        if a.get("paused") or a.get("illiquid"):
            continue
        if a["score"] < sc.get("alert_min_score", 75) or a["agreement"] < sc.get("alert_min_agreement", 0.75):
            continue
        if not once(f"sig|{a['ticker']}|{a['interval']}|{a['asof']}"):
            continue
        send("🔔 <b>HIGH-CONFIDENCE SIGNAL</b>\n" + setup_line(a))
        n += 1
    return n


def price_hits(setups: list[dict], prices: dict[str, dict]) -> int:
    n = 0
    for a in setups:
        t, pl = a["ticker"], a["plan"]
        q = prices.get(t)
        if not q or not pl["valid"]:
            continue
        lo, hi = pl["entry_low"], pl["entry_high"]
        if lo <= q["price"] <= hi and once(f"zone|{t}|{a['asof']}"):
            send(f"🎯 <b>{e(t)}</b> is IN the entry zone {money(lo)}–{money(hi)} · now {money(q['price'])}\n"
                 f"   stop {money(pl['stop'])} · target {money(pl['target'])}")
            n += 1
        long = pl["side"] == "long"
        if (q["high"] >= pl["target"] if long else q["low"] <= pl["target"]) and once(f"tgt|{t}|{a['asof']}"):
            send(f"✅ <b>{e(t)}</b> touched the TARGET {money(pl['target'])}")
            n += 1
        if (q["low"] <= pl["stop"] if long else q["high"] >= pl["stop"]) and once(f"stp|{t}|{a['asof']}"):
            send(f"🛑 <b>{e(t)}</b> hit the STOP {money(pl['stop'])}")
            n += 1
    return n
