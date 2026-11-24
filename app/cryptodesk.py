"""Crypto desk: a CoinMarketCap-style board for the dashboard's Crypto tab.

Every hour it saves one snapshot (site/data/crypto.json):
- the top 250 coins by market value with prices, 1h/24h/7d/30d/1y moves, volume, supply, all-time high
  and a 7-day sparkline (CoinGecko, free; add a COINGECKO_API_KEY "demo" key for higher limits)
- total market value, 24h volume, Bitcoin and Ether dominance (CoinGecko /global)
- trending coins and the largest sectors (CoinGecko)
- the crypto Fear & Greed index, last 30 days (alternative.me)
- crypto headlines from CoinDesk, Cointelegraph, Decrypt, The Block, Bitcoin Magazine and Google News,
  each tagged with the coins it names and labelled bullish / bearish / neutral
- Kronos Oracle's own forecasts for the coins it covers (crypto_hourly.json)

The page also refreshes prices straight from CoinGecko in the browser every minute, so this file is the
fallback and the source of news, sectors and forecasts.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

from .core import SITE_DATA, log, read_json, write_json
from .news import UA, _rss

CG = "https://api.coingecko.com/api/v3"
FEEDS = [("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
         ("https://cointelegraph.com/rss", "Cointelegraph"),
         ("https://decrypt.co/feed", "Decrypt"),
         ("https://www.theblock.co/rss.xml", "The Block"),
         ("https://bitcoinmagazine.com/.rss/full/", "Bitcoin Magazine"),
         ("https://news.google.com/rss/search?q=crypto%20OR%20bitcoin%20when:1d&hl=en-US&gl=US&ceid=US:en", "Google News")]

BULL = set(("surge surges soar soars jump jumps rally rallies record high highs inflow inflows approve approves approval approved "
            "adopt adopts adoption bullish breakout gain gains rebound rebounds rises rise climbs climb buy buys accumulate "
            "partnership launch launches upgrade integrates reserve etf-approval outperform").split())
BEAR = set(("plunge plunges crash crashes drop drops fall falls slump slumps tumble tumbles sink sinks outflow outflows "
            "hack hacked exploit exploited theft stolen liquidation liquidations lawsuit sues probe charges fraud scam "
            "ban bans delist delists bearish selloff sell-off dump dumps fear warns warning collapse insolvent bankrupt "
            "bankruptcy depeg fine fined crackdown losses loss").split())
SKIP_SYMBOLS = {"USDT", "USDC", "DAI", "FDUSD", "USDE", "USDS", "TUSD", "PYUSD", "USD1", "BUSD", "USDD", "FRAX", "RLUSD", "USDX", "GHO"}
AMBIGUOUS = {"ONE", "NEAR", "GAS", "SUN", "BEAM", "CORE", "MOVE", "S", "IP", "ME", "MEW", "OM", "W", "T", "ENA", "PEOPLE", "STORY",
             "TRUMP", "JUP", "ATOM", "POL", "FLOW", "KAS", "SEI", "SAFE", "PRIME", "ACE", "AI", "BLUR", "PYTH", "RAY", "GALA", "SAND"}


def _get(path: str, params: dict | None = None, tries: int = 4):
    key = os.environ.get("COINGECKO_API_KEY")
    hdr = dict(UA, **({"x-cg-demo-api-key": key} if key else {}))
    for i in range(tries):
        try:
            r = requests.get(CG + path, params=params, headers=hdr, timeout=30)
            if r.status_code == 429:
                time.sleep(20 * (i + 1))
                continue
            if r.ok:
                return r.json()
            log(f"coingecko {path}: HTTP {r.status_code}")
        except Exception as e:
            log(f"coingecko {path}: {type(e).__name__}")
        time.sleep(5)
    return None


def _spark(v: list | None, n: int = 56) -> list:
    v = [x for x in (v or []) if x is not None]
    if len(v) <= n:
        return [round(x, 8) for x in v]
    step = (len(v) - 1) / (n - 1)
    return [float(f"{v[round(i * step)]:.6g}") for i in range(n)]


def markets() -> list[dict]:
    rows = _get("/coins/markets", {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1, "sparkline": "true",
                                    "price_change_percentage": "1h,24h,7d,30d,1y"}) or []
    out = []
    for c in rows:
        out.append({"id": c.get("id"), "sym": str(c.get("symbol") or "").upper(), "name": c.get("name"), "img": c.get("image"),
                    "rank": c.get("market_cap_rank"), "price": c.get("current_price"), "mcap": c.get("market_cap"), "fdv": c.get("fully_diluted_valuation"),
                    "vol": c.get("total_volume"), "h24": c.get("high_24h"), "l24": c.get("low_24h"),
                    "c1h": c.get("price_change_percentage_1h_in_currency"), "c24h": c.get("price_change_percentage_24h_in_currency"),
                    "c7d": c.get("price_change_percentage_7d_in_currency"), "c30d": c.get("price_change_percentage_30d_in_currency"),
                    "c1y": c.get("price_change_percentage_1y_in_currency"),
                    "circ": c.get("circulating_supply"), "total": c.get("total_supply"), "max": c.get("max_supply"),
                    "ath": c.get("ath"), "ath_pct": c.get("ath_change_percentage"), "ath_date": (c.get("ath_date") or "")[:10],
                    "atl": c.get("atl"), "spark": _spark((c.get("sparkline_in_7d") or {}).get("price"))})
    return out


def global_stats() -> dict:
    g = (_get("/global") or {}).get("data") or {}
    d = _get("/global/decentralized_finance_defi") or {}
    d = d.get("data") or {}
    return {"mcap": (g.get("total_market_cap") or {}).get("usd"), "vol": (g.get("total_volume") or {}).get("usd"),
            "mcap_chg24": g.get("market_cap_change_percentage_24h_usd"), "coins": g.get("active_cryptocurrencies"), "markets": g.get("markets"),
            "dom": {k.upper(): round(v, 2) for k, v in sorted((g.get("market_cap_percentage") or {}).items(), key=lambda kv: -kv[1])[:6]},
            "defi_mcap": _num(d.get("defi_market_cap")), "defi_dom": _num(d.get("defi_dominance"))}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def trending() -> list[dict]:
    t = _get("/search/trending") or {}
    out = []
    for c in (t.get("coins") or [])[:10]:
        i = c.get("item") or {}
        d = i.get("data") or {}
        ch = (d.get("price_change_percentage_24h") or {}).get("usd")
        out.append({"id": i.get("id"), "sym": str(i.get("symbol") or "").upper(), "name": i.get("name"), "img": i.get("small") or i.get("thumb"),
                    "rank": i.get("market_cap_rank"), "price": _num(d.get("price")), "c24h": ch})
    return out


def categories() -> list[dict]:
    cats = _get("/coins/categories") or []
    out = []
    for c in cats:
        if not c.get("market_cap"):
            continue
        out.append({"id": c.get("id"), "name": c.get("name"), "mcap": c.get("market_cap"), "c24h": c.get("market_cap_change_24h"),
                    "vol": c.get("volume_24h"), "top": [u for u in (c.get("top_3_coins") or [])[:3]]})
        if len(out) >= 40:
            break
    return out


def fear_greed() -> dict:
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 30}, headers=UA, timeout=20)
        rows = (r.json() or {}).get("data") or [] if r.ok else []
        hist = [{"date": datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date().isoformat(), "value": int(x["value"]),
                 "label": x.get("value_classification")} for x in rows]
        return {"now": hist[0] if hist else None, "history": hist[::-1]}
    except Exception as e:
        log(f"fear & greed: {type(e).__name__}")
        return {"now": None, "history": []}


def _side(title: str) -> tuple[str, float]:
    w = re.findall(r"[a-z][a-z-]*", title.lower())
    s = sum(x in BULL for x in w) - sum(x in BEAR for x in w)
    return ("bull" if s > 0 else "bear" if s < 0 else "neutral"), max(-1.0, min(1.0, s / 2))


def news(coins: list[dict]) -> list[dict]:
    items = []
    for url, src in FEEDS:
        items += _rss(url, src, 40)
    names = []
    for c in coins[:150]:
        if c["sym"] in SKIP_SYMBOLS:
            continue
        pats = [re.escape(c["name"])] if c.get("name") and len(c["name"]) > 3 else []
        if c["sym"] not in AMBIGUOUS and len(c["sym"]) >= 3:
            pats.append(r"\$?" + re.escape(c["sym"]))
        if c["sym"] == "BTC":
            pats.append("bitcoin")
        if c["sym"] == "ETH":
            pats += ["ether", "ethereum"]
        if pats:
            names.append((c["sym"], re.compile(r"\b(" + "|".join(pats) + r")\b", re.I)))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x.get("time") or "", reverse=True):
        k = re.sub(r"[^a-z0-9]", "", it["title"].lower())[:70]
        if not k or k in seen or not it.get("url") or (it.get("time") and it["time"] < cutoff):
            continue
        seen.add(k)
        side, sc = _side(it["title"])
        tags = [s for s, rx in names if rx.search(it["title"])][:5]
        out.append({**it, "side": side, "score": sc, "coins": tags})
        if len(out) >= 120:
            break
    return out


def forecasts() -> dict:
    d = read_json(SITE_DATA / "crypto_hourly.json", {}) or {}
    out = {}
    for r in d.get("rows") or []:
        sym = str(r.get("ticker", "")).split("-")[0].upper()
        if not sym:
            continue
        out[sym] = {"dir": r.get("direction"), "p_up": r.get("p_up"), "score": r.get("score"), "agree": r.get("agree_count"), "of": r.get("samples"),
                    "median": r.get("median_return"), "next": (r.get("ranges") or {}).get("next"), "day": (r.get("ranges") or {}).get("week"),
                    "plan": r.get("plan") if (r.get("plan") or {}).get("valid") else None, "asof": r.get("asof")}
    return {"generated": d.get("generated"), "coins": out}


def run() -> dict:
    coins = markets()
    if not coins:
        prev = read_json(SITE_DATA / "crypto.json", None)
        if prev:
            log("crypto desk: CoinGecko unavailable, keeping the previous snapshot's prices")
            coins = prev.get("coins") or []
    res = {"generated": datetime.now(timezone.utc).isoformat(), "global": global_stats(), "coins": coins, "trending": trending(),
           "categories": categories(), "fear_greed": fear_greed(), "news": news(coins), "kronos": forecasts(),
           "sources": {"prices": "CoinGecko", "fear_greed": "alternative.me", "news": [s for _, s in FEEDS]}}
    nb = sum(n["side"] == "bull" for n in res["news"])
    nr = sum(n["side"] == "bear" for n in res["news"])
    res["news_mood"] = {"bull": nb, "bear": nr, "neutral": len(res["news"]) - nb - nr}
    write_json(SITE_DATA / "crypto.json", res)
    log(f"crypto desk: {len(coins)} coins, {len(res['news'])} stories ({nb} bullish, {nr} bearish), {len(res['categories'])} sectors, "
        f"{len(res['kronos']['coins'])} Kronos forecasts")
    return res


if __name__ == "__main__":
    run()
