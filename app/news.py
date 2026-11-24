"""News desk (FinBERT): collects fresh headlines for every watchlist stock and the market as a whole,
then a finance-trained language model (ProsusAI/finbert) labels each one bullish, bearish or neutral.

Free sources, no key: Yahoo Finance and Google News feeds. Finnhub company news is added when
FINNHUB_API_KEY is saved. Each day's score per stock is logged to state/news_sentiment.csv so the
Edge Lab can later test whether the news mood predicts the next move.
"""
from __future__ import annotations

import csv
import html
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from .core import SITE_DATA, STATE_DIR, load_config, log, write_json

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
LOG = STATE_DIR / "news_sentiment.csv"
MARKET = [("Market", "stock market today"), ("Fed & rates", "Federal Reserve interest rates"), ("Economy", "US economy inflation jobs")]


def _rss(url: str, source: str, limit: int) -> list[dict]:
    out = []
    try:
        r = requests.get(url, headers=UA, timeout=20)
        if not r.ok:
            return out
        for it in ET.fromstring(r.content).iter("item"):
            title = html.unescape((it.findtext("title") or "").strip())
            if not title:
                continue
            src = it.findtext("source") or source
            if source == "Google News" and " - " in title:  # Google appends " - Publisher"
                title, src = title.rsplit(" - ", 1)
            try:
                when = parsedate_to_datetime(it.findtext("pubDate")).astimezone(timezone.utc)
            except Exception:
                when = None
            out.append({"title": title[:220], "url": (it.findtext("link") or "").strip(), "source": src.strip()[:40],
                        "time": when.isoformat() if when else None})
            if len(out) >= limit:
                break
    except Exception as e:
        log(f"feed failed ({source}): {type(e).__name__}")
    return out


def gather(ticker: str, name: str | None, limit: int, max_age_days: int) -> list[dict]:
    items = _rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", "Yahoo Finance", limit)
    q = requests.utils.quote(f"{ticker} stock when:{max_age_days}d")
    items += _rss(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en", "Google News", limit)
    key = os.environ.get("FINNHUB_API_KEY")
    if key:
        try:
            today = datetime.now(timezone.utc).date()
            r = requests.get("https://finnhub.io/api/v1/company-news", timeout=20,
                             params={"symbol": ticker, "from": (today - timedelta(days=max_age_days)).isoformat(), "to": today.isoformat(), "token": key})
            for n in (r.json() if r.ok else [])[:limit]:
                items.append({"title": str(n.get("headline", ""))[:220], "url": n.get("url", ""), "source": str(n.get("source", "Finnhub"))[:40],
                              "time": datetime.fromtimestamp(n["datetime"], timezone.utc).isoformat() if n.get("datetime") else None})
            time.sleep(1.05)
        except Exception as e:
            log(f"finnhub news failed for {ticker}: {type(e).__name__}")
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["time"] or "", reverse=True):
        k = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:60]
        if not k or k in seen or (it["time"] and it["time"] < cutoff):
            continue
        seen.add(k)
        out.append(it)
    return out[:limit]


POS = "beat beats surge surges soar soars jump jumps rally rallies record upgrade upgraded raises raised strong growth profit gains gain bullish outperform buy tops".split()
NEG = "miss misses plunge plunges drop drops fall falls slump slumps cut cuts downgrade downgraded lawsuit probe weak loss losses bearish underperform sell warns warning layoffs recall fraud".split()


def _lexicon(titles: list[str]) -> list[dict]:
    out = []
    for t in titles:
        w = re.findall(r"[a-z]+", t.lower())
        s = sum(x in POS for x in w) - sum(x in NEG for x in w)
        out.append({"label": "positive" if s > 0 else "negative" if s < 0 else "neutral", "score": max(-1.0, min(1.0, s / 2))})
    return out


def score(titles: list[str]) -> tuple[list[dict], str]:
    """Returns ([{label, score}], method). score = P(positive) - P(negative), from -1 to +1."""
    if not titles:
        return [], "none"
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
        out = []
        for res in clf(titles, batch_size=32):
            p = {d["label"].lower(): float(d["score"]) for d in res}
            out.append({"label": max(p, key=p.get), "score": round(p.get("positive", 0) - p.get("negative", 0), 4)})
        return out, "FinBERT"
    except Exception as e:
        log(f"FinBERT unavailable ({type(e).__name__}: {str(e)[:120]}); using keyword scoring")
        return _lexicon(titles), "keywords"


def _summ(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0, "score": None, "positive": 0, "negative": 0, "neutral": 0}
    c = {k: sum(i["label"] == k for i in items) for k in ("positive", "negative", "neutral")}
    return {"n": n, "score": round(sum(i["score"] for i in items) / n, 4), **c}


def run() -> dict:
    cfg = load_config()
    nc = cfg.get("news", {})
    limit, age = int(nc.get("per_ticker", 12)), int(nc.get("max_age_days", 3))
    groups = [(t.upper(), gather(t.upper(), None, limit, age)) for t in cfg.get("watchlist", [])]
    market = []
    for label, q in MARKET:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(q + ' when:1d')}&hl=en-US&gl=US&ceid=US:en"
        market.append((label, _rss(url, "Google News", 10)))
    flat = [it for _, items in groups + market for it in items]
    scored, method = score([it["title"] for it in flat])
    for it, s in zip(flat, scored):
        it.update(s)
    out = {"generated": datetime.now(timezone.utc).isoformat(), "method": method, "max_age_days": age,
           "tickers": sorted([{"ticker": t, **_summ(items), "items": items} for t, items in groups],
                             key=lambda r: (r["score"] is None, -(r["score"] or 0))),
           "market": [{"topic": l, **_summ(items), "items": items} for l, items in market]}
    allw = [it for _, items in groups for it in items]
    out["overall"] = _summ(allw)
    out["status"] = "ok" if flat else "No headlines came back from any feed this run."
    today = datetime.now(timezone.utc).date().isoformat()
    rows = []
    if LOG.exists():
        with open(LOG) as fh:
            rows = [r for r in csv.DictReader(fh) if r["date"] != today]
    rows += [{"date": today, "ticker": r["ticker"], "score": r["score"], "n": r["n"], "method": method} for r in out["tickers"] if r["n"]]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "ticker", "score", "n", "method"])
        w.writeheader()
        w.writerows(rows)
    write_json(SITE_DATA / "news.json", out)
    log(f"news: {len(flat)} headlines scored with {method}")
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
