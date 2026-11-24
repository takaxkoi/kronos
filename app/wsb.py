"""WallStreetBets board: what r/wallstreetbets is talking about right now.

- Most-mentioned stocks, with the change in mentions over 24 hours (ApeWisdom, free, no key).
- The hot and top posts of the day straight from Reddit, each labelled bullish / bearish / neutral by FinBERT,
  with the tickers named in the title.
Reddit often blocks cloud servers, so every source is tried in turn and the page says which ones answered.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from .core import ROOT, SITE_DATA, load_config, log, write_json
from .news import _summ, score

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
      "Accept": "application/json, text/xml;q=0.9, */*;q=0.8"}
STOP = set("A I AM AN AT BE BY DO GO IF IN IS IT ME MY NO OF ON OR SO TO UP US WE ALL AND ANY ARE BIG BUY CAN CEO DD EPS ETF FOR GDP HAS IPO ITS NEW NOT NOW OUT PUT SEC THE WAS WHO WHY YOU YOLO CALL CASH EVER FROM GAIN HOLD HUGE JUST LOSS MOON NEXT OPEN PLAY PUTS REAL SELL THIS WHAT WITH WSB USA USD FED CPI ATH AI IMO LOL EOD ITM OTM IV PM AH RH TA FOMO ONE TWO DAY WEEK YEAR BEAR BULL LONG SHORT FREE GOOD BEST EDIT POST TLDR".split())


def universe() -> set[str]:
    u = {f.name[:-7] for f in (ROOT / "data" / "history").glob("*.csv.gz")}
    return {t for t in u if t.isalpha()} | {"GME", "AMC", "PLTR", "HOOD", "SOFI", "RDDT", "MSTR", "RIVN", "NIO", "SPY", "QQQ", "IWM", "TQQQ", "SQQQ", "ASTS", "RKLB", "IONQ", "OKLO", "CRWV"}


NAMES = {"nvidia": "NVDA", "tesla": "TSLA", "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "google": "GOOGL", "alphabet": "GOOGL", "meta": "META",
         "palantir": "PLTR", "gamestop": "GME", "micron": "MU", "broadcom": "AVGO", "netflix": "NFLX", "coinbase": "COIN", "intel": "INTC", "robinhood": "HOOD",
         "supermicro": "SMCI", "super micro": "SMCI", "uber": "UBER", "jpmorgan": "JPM", "boeing": "BA", "disney": "DIS", "walmart": "WMT", "oracle": "ORCL",
         "coreweave": "CRWV", "reddit": "RDDT", "microstrategy": "MSTR", "berkshire": "BRK-B", "costco": "COST", "nike": "NKE", "salesforce": "CRM", "amd": "AMD"}


def find_tickers(text: str, uni: set[str]) -> list[str]:
    out = []
    low = (text or "").lower()
    for name, t in NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", low) and t not in out:
            out.append(t)
    for m in re.finditer(r"\$([A-Za-z]{1,5})\b|\b([A-Z]{2,5})\b", text or ""):
        t = (m.group(1) or m.group(2)).upper()
        if (m.group(1) or t not in STOP) and t in uni and t not in out:
            out.append(t)
    return out


def reddit(kind: str, note: dict) -> list[dict]:
    q = "?limit=40" + ("&t=day" if kind == "top" else "")
    for host in ("https://www.reddit.com", "https://old.reddit.com", "https://api.reddit.com"):
        try:
            r = requests.get(f"{host}/r/wallstreetbets/{kind}.json{q}", headers=UA, timeout=20)
            if r.ok and "json" in r.headers.get("content-type", ""):
                out = []
                for c in r.json()["data"]["children"]:
                    d = c["data"]
                    if d.get("stickied"):
                        continue
                    out.append({"title": html.unescape(d.get("title", ""))[:240], "url": "https://www.reddit.com" + d.get("permalink", ""),
                                "upvotes": int(d.get("score", 0)), "comments": int(d.get("num_comments", 0)), "flair": d.get("link_flair_text") or "",
                                "time": datetime.fromtimestamp(d.get("created_utc", 0), timezone.utc).isoformat(), "list": kind})
                if out:
                    note[f"reddit {kind}"] = "ok"
                    return out
            note[f"reddit {kind}"] = f"HTTP {r.status_code}"
        except Exception as e:
            note[f"reddit {kind}"] = type(e).__name__
    try:  # public feed: titles and links only, no vote counts
        r = requests.get(f"https://www.reddit.com/r/wallstreetbets/{kind}/.rss{q}", headers=UA, timeout=20)
        if r.ok:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            out = [{"title": html.unescape(e.findtext("a:title", "", ns))[:240], "url": (e.find("a:link", ns).get("href") if e.find("a:link", ns) is not None else ""),
                    "upvotes": None, "comments": None, "flair": (e.find("a:category", ns).get("label") if e.find("a:category", ns) is not None else ""),
                    "time": e.findtext("a:updated", None, ns), "list": kind} for e in ET.fromstring(r.content).findall("a:entry", ns)]
            if out:
                note[f"reddit {kind}"] = "ok (feed, no vote counts)"
                return out
        note[f"reddit {kind}"] += f", feed HTTP {r.status_code}"
    except Exception as e:
        note[f"reddit {kind}"] = str(note.get(f"reddit {kind}", "")) + f", feed {type(e).__name__}"
    return []


def archive(note: dict) -> list[dict]:
    """Reddit archives that allow cloud servers. Vote counts can lag the live site by a few hours."""
    since = int(datetime.now(timezone.utc).timestamp()) - 86400
    tries = [("pullpush", f"https://api.pullpush.io/reddit/search/submission/?subreddit=wallstreetbets&after={since}&size=100&sort=desc&sort_type=score"),
             ("arctic shift", f"https://arctic-shift.photon-reddit.com/api/posts/search?subreddit=wallstreetbets&after={since}&limit=100")]
    for name, url in tries:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            rows = (r.json().get("data") or []) if r.ok else []
            out = [{"title": html.unescape(d.get("title", ""))[:240], "url": "https://www.reddit.com" + d.get("permalink", ""), "upvotes": int(d.get("score") or 0),
                    "comments": int(d.get("num_comments") or 0), "flair": d.get("link_flair_text") or "",
                    "time": datetime.fromtimestamp(int(d.get("created_utc") or 0), timezone.utc).isoformat(), "list": name}
                   for d in rows if d.get("title") and d.get("permalink") and not d.get("stickied") and d.get("removed_by_category") is None]
            note[name] = "ok" if out else f"HTTP {r.status_code}, no posts"
            if out:
                return out
        except Exception as e:
            note[name] = type(e).__name__
    return []


def oauth(note: dict) -> list[dict]:
    """Official Reddit API. Used when REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are saved (free 'script' app at reddit.com/prefs/apps)."""
    import os
    cid, sec = os.environ.get("REDDIT_CLIENT_ID"), os.environ.get("REDDIT_CLIENT_SECRET")
    if not cid or not sec:
        return []
    try:
        ua = {"User-Agent": "trading-picker/1.0 by takaxkoi"}
        tok = requests.post("https://www.reddit.com/api/v1/access_token", auth=(cid, sec), data={"grant_type": "client_credentials"}, headers=ua, timeout=20).json()["access_token"]
        out = []
        for kind in ("hot", "top"):
            r = requests.get(f"https://oauth.reddit.com/r/wallstreetbets/{kind}", params={"limit": 40, "t": "day"}, headers={**ua, "Authorization": f"Bearer {tok}"}, timeout=20)
            for c in r.json()["data"]["children"]:
                d = c["data"]
                if not d.get("stickied"):
                    out.append({"title": html.unescape(d.get("title", ""))[:240], "url": "https://www.reddit.com" + d.get("permalink", ""), "upvotes": int(d.get("score", 0)),
                                "comments": int(d.get("num_comments", 0)), "flair": d.get("link_flair_text") or "",
                                "time": datetime.fromtimestamp(d.get("created_utc", 0), timezone.utc).isoformat(), "list": kind})
        note["reddit api"] = "ok" if out else "no posts"
        return out
    except Exception as e:
        note["reddit api"] = type(e).__name__
        return []


def mentions(note: dict) -> list[dict]:
    try:
        r = requests.get("https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/1", headers=UA, timeout=25)
        rows = r.json().get("results", []) if r.ok else []
        note["apewisdom"] = "ok" if rows else f"HTTP {r.status_code}"
        out = []
        for x in rows[:40]:
            now, prev = int(x.get("mentions") or 0), int(x.get("mentions_24h_ago") or 0)
            out.append({"ticker": str(x.get("ticker", "")).upper(), "name": html.unescape(str(x.get("name", "")))[:40], "mentions": now, "mentions_24h_ago": prev,
                        "change": (now / prev - 1) if prev else None, "upvotes": int(x.get("upvotes") or 0), "rank": int(x.get("rank") or 0),
                        "rank_24h_ago": int(x.get("rank_24h_ago") or 0) or None})
        return out
    except Exception as e:
        note["apewisdom"] = type(e).__name__
        return []


def run() -> dict:
    cfg = load_config()
    watch = {t.upper() for t in cfg.get("watchlist", [])}
    uni, note = universe(), {}
    posts, seen = [], set()
    for p in oauth(note):
        if p["url"] not in seen:
            seen.add(p["url"])
            posts.append(p)
    if not posts:
        for kind in ("hot", "top"):
            for p in reddit(kind, note):
                if p["url"] not in seen:
                    seen.add(p["url"])
                    posts.append(p)
    if not posts:
        posts = archive(note)
    labels, method = score([p["title"] for p in posts])
    for p, s in zip(posts, labels):
        p.update(s)
        p["tickers"] = find_tickers(p["title"], uni)
    posts.sort(key=lambda p: (p["upvotes"] or 0, p["time"] or ""), reverse=True)
    tick = mentions(note)
    if not tick and posts:  # fall back to counting tickers in the post titles we did get
        cnt = {}
        for p in posts:
            for t in p["tickers"]:
                c = cnt.setdefault(t, {"ticker": t, "name": "", "mentions": 0, "mentions_24h_ago": None, "change": None, "upvotes": 0, "rank": 0, "rank_24h_ago": None})
                c["mentions"] += 1
                c["upvotes"] += p["upvotes"] or 0
        tick = sorted(cnt.values(), key=lambda c: (c["mentions"], c["upvotes"]), reverse=True)[:40]
        for i, c in enumerate(tick):
            c["rank"] = i + 1
    for t in tick:
        t["on_watchlist"] = t["ticker"] in watch
        mine = [p for p in posts if t["ticker"] in p["tickers"]]
        t["post_mood"] = _summ(mine)["score"] if mine else None
    out = {"generated": datetime.now(timezone.utc).isoformat(), "method": method, "sources": note, "tickers": tick, "posts": posts[:50],
           "mood": _summ(posts), "status": "ok" if tick and posts else ("Reddit blocked this server; showing mention counts only." if tick else
                                                                       "Mention counts unavailable; showing posts only." if posts else "No WallStreetBets source answered this run.")}
    write_json(SITE_DATA / "wsb.json", out)
    log(f"wsb: {len(posts)} posts, {len(tick)} tickers, sources {note}")
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
