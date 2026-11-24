"""Live market news: the newest stories about your watchlist and the largest US stocks, each tagged
with whether it is bullish or bearish for every company it names.

Source: Tickerbot news (TICKERBOT_API_KEY), which scores each article per ticker. Finnhub's general
market headlines are added when FINNHUB_API_KEY is saved (those are scored by keyword).
Runs with the Live prices job, so the feed refreshes every 30 minutes on weekdays.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import requests

from .core import SITE_DATA, load_config, log, write_json

LABEL = {"bullish": "bull", "somewhat-bullish": "bull", "bearish": "bear", "somewhat-bearish": "bear", "neutral": "neutral"}


def _side(s) -> str:
    return LABEL.get(str(s or "").lower(), "neutral")


def tickerbot(params: dict, note: dict, name: str) -> list[dict]:
    from .tickerbot import _err, call
    ok, data, status = call("GET", "/v2/news", params=params)
    if not ok or not isinstance(data, dict):
        note[name] = f"HTTP {status}: {_err(data)}"[:140]
        return []
    out = []
    for a in data.get("results") or []:
        ments = sorted(a.get("mentions") or [], key=lambda m: -(m.get("relevance") or 0))[:6]
        tick = [{"t": str(m.get("ticker", "")).upper(), "side": _side(m.get("sentiment")), "rel": round(float(m.get("relevance") or 0), 2)} for m in ments if m.get("ticker")]
        if not a.get("title") or not a.get("url"):
            continue
        out.append({"id": f"tb{a.get('id')}", "time": a.get("time_published"), "title": str(a["title"])[:240], "url": a["url"], "source": str(a.get("source") or "")[:40],
                    "tickers": tick, "side": _side(a.get("overall_sentiment_label")) if a.get("overall_sentiment_label") else (tick[0]["side"] if tick else "neutral")})
    note[name] = f"ok, {len(out)} stories"
    return out


def finnhub(note: dict) -> list[dict]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    try:
        from .news import _lexicon
        r = requests.get("https://finnhub.io/api/v1/news", params={"category": "general", "token": key}, timeout=20)
        rows = r.json() if r.ok else []
        rows = [n for n in rows if n.get("headline") and n.get("url")][:40]
        sides = _lexicon([n["headline"] for n in rows])
        note["finnhub"] = f"ok, {len(rows)} stories" if rows else f"HTTP {r.status_code}"
        return [{"id": f"fh{n.get('id')}", "time": datetime.fromtimestamp(n["datetime"], timezone.utc).isoformat().replace("+00:00", "Z") if n.get("datetime") else None,
                 "title": str(n["headline"])[:240], "url": n["url"], "source": str(n.get("source") or "Finnhub")[:40], "tickers": [],
                 "side": {"positive": "bull", "negative": "bear"}.get(s["label"], "neutral")} for n, s in zip(rows, sides)]
    except Exception as e:
        note["finnhub"] = type(e).__name__
        return []


def run() -> dict:
    cfg = load_config()
    watch = [t.upper() for t in cfg.get("watchlist", [])]
    since = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    note: dict = {}
    mine = tickerbot({"ticker": ",".join(watch[:50]), "min_relevance": 0.5, "from": since, "limit": 80}, note, "watchlist news")
    mkt = tickerbot({"universe": "top_100", "min_relevance": 0.6, "from": since, "limit": 80}, note, "market news")
    seen, items = set(), []
    for it in mine + mkt + finnhub(note):
        k = (it["title"] or "").lower()[:70]
        if k in seen or not it.get("time"):
            continue
        seen.add(k)
        it["watch"] = any(m["t"] in watch for m in it["tickers"])
        items.append(it)
    items.sort(key=lambda x: x["time"], reverse=True)
    items = items[:140]
    by = {}
    for it in items:
        for m in it["tickers"]:
            b = by.setdefault(m["t"], {"t": m["t"], "n": 0, "bull": 0, "bear": 0})
            b["n"] += 1
            b[m["side"]] = b.get(m["side"], 0) + 1
    hot = sorted(by.values(), key=lambda b: -b["n"])[:16]
    out = {"generated": datetime.now(timezone.utc).isoformat(), "items": items, "hot": hot, "sources": note,
           "counts": {k: sum(i["side"] == k for i in items) for k in ("bull", "bear", "neutral")},
           "status": "ok" if items else "No news source answered this run."}
    write_json(SITE_DATA / "livenews.json", out)
    log(f"live news: {len(items)} stories, sources {note}")
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
