"""Live price board that takes turns between every price source so no single free limit runs out.

Each run picks a different lead source (by turn), asks it for every ticker in one go where it can,
then fills whatever is missing from the next sources in line. Calls are counted per source per UTC day
against the caps in config.yaml -> quotes.daily_caps, and a source that hits its cap sits out until tomorrow.

Sources: Yahoo (no key) · Finnhub (FINNHUB_API_KEY) · Tickerbot (TICKERBOT_API_KEY) ·
         Alpaca (ALPACA_KEY_ID + ALPACA_SECRET_KEY) · Alpha Vantage (ALPHAVANTAGE_API_KEY, gap-fill only)
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests

from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, write_json

BUDGET = STATE_DIR / "quote_budget.json"
DEFAULT_CAPS = {"yahoo": 1500, "finnhub": 3000, "tickerbot": 40, "alpaca": 2000, "alphavantage": 3}
ROTATION = ["tickerbot", "finnhub", "alpaca", "yahoo"]  # Alpha Vantage only fills gaps (25 calls a day)
UA = {"User-Agent": "Mozilla/5.0 (kronos-oracle)"}


def _f(v):
    try:
        v = float(v)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _row(t, price, prev=None, high=None, low=None, when=None, chg=None):
    price, prev = _f(price), _f(prev)
    if not price or price <= 0:
        return None
    chg = _f(chg) if chg is not None else ((price / prev - 1) * 100 if prev else None)
    return {"ticker": t, "price": price, "prev_close": prev, "change_pct": chg, "high": _f(high), "low": _f(low), "time": when}


def _iso(ts):
    try:
        return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


class Meter:
    """Counts calls per source per UTC day so the free limits last all day."""

    def __init__(self, caps: dict):
        st = read_json(BUDGET, {})
        self.day = datetime.now(timezone.utc).date().isoformat()
        same = st.get("day") == self.day
        self.used = dict(st.get("used", {})) if same else {}
        self.turn = int(st.get("turn", 0)) if same else 0
        self.caps = {**DEFAULT_CAPS, **(caps or {})}
        self.note = {}

    def left(self, s):
        return max(0, int(self.caps.get(s, 0)) - int(self.used.get(s, 0)))

    def spend(self, s, n=1):
        self.used[s] = int(self.used.get(s, 0)) + n

    def save(self):
        write_json(BUDGET, {"day": self.day, "used": self.used, "turn": self.turn + 1})


def has_key(s):
    e = os.environ.get
    return {"yahoo": True, "finnhub": bool(e("FINNHUB_API_KEY")), "tickerbot": bool(e("TICKERBOT_API_KEY")),
            "alpaca": bool(e("ALPACA_KEY_ID") and e("ALPACA_SECRET_KEY")), "alphavantage": bool(e("ALPHAVANTAGE_API_KEY"))}[s]


def yahoo(tickers, m):
    out = {}
    for t in tickers:
        if m.left("yahoo") <= 0:
            break
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", params={"interval": "1d", "range": "5d"},
                             headers=UA, timeout=20)
            m.spend("yahoo")
            if r.status_code == 429:
                m.note["yahoo"] = "Yahoo is rate-limiting this server"
                break
            res = r.json()["chart"]["result"][0]
            meta = res["meta"]
            closes = [c for c in (res.get("indicators", {}).get("quote") or [{}])[0].get("close") or [] if c is not None]
            prev = closes[-2] if len(closes) >= 2 else meta.get("previousClose")  # yesterday's close, not the start of the range
            row = _row(t, meta.get("regularMarketPrice"), prev,
                       meta.get("regularMarketDayHigh"), meta.get("regularMarketDayLow"), _iso(meta.get("regularMarketTime")))
            if row:
                out[t] = row
        except Exception as ex:
            m.note["yahoo"] = f"{type(ex).__name__}"
        time.sleep(0.15)
    return out


def finnhub(tickers, m):
    out, key = {}, os.environ["FINNHUB_API_KEY"]
    for t in tickers:
        if m.left("finnhub") <= 0:
            break
        try:
            r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": t, "token": key}, timeout=20)
            m.spend("finnhub")
            if r.status_code in (401, 403, 429):
                m.note["finnhub"] = f"HTTP {r.status_code}"
                break
            q = r.json()
            row = _row(t, q.get("c"), q.get("pc"), q.get("h"), q.get("l"), _iso(q.get("t")), q.get("dp"))
            if row:
                out[t] = row
        except Exception as ex:
            m.note["finnhub"] = f"{type(ex).__name__}"
        time.sleep(1.05)  # free plan: 60 calls a minute
    return out


def tickerbot(tickers, m):
    from .tickerbot import _err, call
    m.spend("tickerbot")  # one call covers the whole list
    ok, data, status = call("POST", "/v2/scan", body={"q": "ticker IN (" + ",".join(f"'{t}'" for t in tickers) + ")", "limit": 100})
    if not ok or not isinstance(data, dict):
        m.note["tickerbot"] = f"HTTP {status}: {_err(data)}"[:160]
        return {}
    out = {}
    for r in data.get("results") or []:
        t = str(r.get("ticker", "")).upper()
        chg = next((r[k] for k in ("change_1d_pct", "change_pct", "pct_change_1d", "change_percent") if r.get(k) is not None), None)
        row = _row(t, r.get("price") or r.get("close") or r.get("last"), r.get("prev_close") or r.get("previous_close"),
                   r.get("high"), r.get("low"), r.get("asof") or r.get("updated_at") or (data.get("_meta") or {}).get("asof"), chg)
        if row and t in tickers:
            out[t] = row
    return out


def alpaca(tickers, m):
    m.spend("alpaca")
    try:
        r = requests.get("https://data.alpaca.markets/v2/stocks/snapshots", params={"symbols": ",".join(tickers), "feed": "iex"},
                         headers={"APCA-API-KEY-ID": os.environ["ALPACA_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"]},
                         timeout=30)
        if not r.ok:
            m.note["alpaca"] = f"HTTP {r.status_code}"
            return {}
        data = r.json()
        data = data.get("snapshots", data)
    except Exception as ex:
        m.note["alpaca"] = f"{type(ex).__name__}"
        return {}
    out = {}
    for t, s in data.items():
        s = s or {}
        day, tr = s.get("dailyBar") or {}, s.get("latestTrade") or {}
        row = _row(t, tr.get("p") or day.get("c"), (s.get("prevDailyBar") or {}).get("c"), day.get("h"), day.get("l"), tr.get("t"))
        if row:
            out[t] = row
    return out


def alphavantage(tickers, m):
    out, key = {}, os.environ["ALPHAVANTAGE_API_KEY"]
    for t in tickers:
        if m.left("alphavantage") <= 0:
            break
        try:
            r = requests.get("https://www.alphavantage.co/query", params={"function": "GLOBAL_QUOTE", "symbol": t, "apikey": key}, timeout=30)
            m.spend("alphavantage")
            q = r.json().get("Global Quote") or {}
            if not q:
                m.note["alphavantage"] = "daily limit reached"
                break
            row = _row(t, q.get("05. price"), q.get("08. previous close"), q.get("03. high"), q.get("04. low"), q.get("07. latest trading day"))
            if row:
                out[t] = row
        except Exception as ex:
            m.note["alphavantage"] = f"{type(ex).__name__}"
        time.sleep(1.2)
    return out


FETCH = {"yahoo": yahoo, "finnhub": finnhub, "tickerbot": tickerbot, "alpaca": alpaca, "alphavantage": alphavantage}


def run() -> dict:
    cfg = load_config()
    qc = cfg.get("quotes", {})
    tickers = list(dict.fromkeys([t.upper() for t in cfg.get("watchlist", [])] + [t.upper() for t in cfg.get("indices", [])]))
    m = Meter(qc.get("daily_caps"))
    ready = [s for s in ROTATION if has_key(s) and m.left(s) > 0]
    k = m.turn % len(ready) if ready else 0
    order = ready[k:] + ready[:k] + (["alphavantage"] if has_key("alphavantage") else [])
    got = {}
    for s in order:
        missing = [t for t in tickers if t not in got]
        if not missing:
            break
        if m.left(s) <= 0:
            continue
        rows = FETCH[s](missing, m)
        for t, r in rows.items():
            got[t] = {**r, "source": s}
        log(f"prices: {s} gave {len(rows)} of {len(missing)} {m.note.get(s, '')}")
    out = {"generated": datetime.now(timezone.utc).isoformat(), "lead": order[0] if order else None, "turn": m.turn + 1,
           "rows": [got[t] for t in tickers if t in got], "missing": [t for t in tickers if t not in got],
           "sources": [{"name": s, "connected": has_key(s), "used_today": int(m.used.get(s, 0)), "cap": int(m.caps.get(s, 0)),
                        "note": m.note.get(s, "")} for s in ROTATION + ["alphavantage"]]}
    out["status"] = "ok" if not out["missing"] else (f"{len(out['missing'])} tickers had no price this run" if got else "no source returned prices")
    m.save()
    write_json(SITE_DATA / "quotes.json", out)
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
