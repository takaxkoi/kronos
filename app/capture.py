"""DATA VAULT: one snapshot per trading day of everything we can see, saved to data/daily/YYYY-MM-DD/.

Why: most "edge" data (fundamentals, analyst targets, short interest, options positioning, sentiment,
estimate revisions) is overwritten by providers. Capturing it point-in-time every day builds a history
nobody can sell you later, and lets the Edge Lab test signals without look-ahead bias.

Sources
  Yahoo Finance (free): prices for ~550 tickers, fundamentals/short interest/analyst snapshot for every
      stock, plus options positioning, analyst actions, insider trades, earnings surprises, news for your core list.
  Alpha Vantage (free key = 25 calls/day, budgeted): earnings calendar, market movers, news sentiment,
      Treasury yields, and a rotating deep-dive (earnings, estimates, insiders) through your core list.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .core import ROOT, STATE_DIR, SITE_DATA, load_config, log, read_json, write_json
from .data import _av_limited, asset_class, av_gap, av_premium, fetch_history, sp500

VAULT = Path(os.environ.get("KRONOS_VAULT_DIR", ROOT / "data"))

INFO_FIELDS = ["marketCap", "enterpriseValue", "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
               "enterpriseToEbitda", "profitMargins", "grossMargins", "operatingMargins", "returnOnEquity", "returnOnAssets",
               "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth", "debtToEquity", "currentRatio", "freeCashflow",
               "totalCash", "totalDebt", "beta", "shortRatio", "shortPercentOfFloat", "sharesShort", "sharesShortPriorMonth",
               "floatShares", "sharesOutstanding", "heldPercentInsiders", "heldPercentInstitutions", "recommendationMean",
               "recommendationKey", "numberOfAnalystOpinions", "targetMeanPrice", "targetMedianPrice", "targetHighPrice",
               "targetLowPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "fiftyDayAverage", "twoHundredDayAverage",
               "averageVolume", "averageVolume10days", "dividendYield", "payoutRatio", "trailingEps", "forwardEps",
               "earningsTimestamp", "sector", "industry", "currentPrice"]


# ── helpers ──────────────────────────────────────────────────
def _save_csv(df: pd.DataFrame, path: Path) -> int:
    if df is None or df.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return len(df)


def _save_raw(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(obj, f, default=str)


class AVBudget:
    """Alpha Vantage call budget. Free key: 25 calls/day, ~1/sec, counted per UTC day across runs.
    Paid key (ALPHAVANTAGE_PREMIUM=true): no daily cap, throttled under the per-minute cap."""

    def __init__(self, limit: int):
        self.key = os.environ.get("ALPHAVANTAGE_API_KEY")
        self.f = STATE_DIR / "av_budget.json"
        st = read_json(self.f, {})
        today = datetime.now(timezone.utc).date().isoformat()
        self.used = st.get("used", 0) if st.get("day") == today else 0
        self.day, self.limit, self.last, self.dead = today, limit, 0.0, False
        self.premium = av_premium()

    def left(self) -> int:
        if not self.key or self.dead:
            return 0
        return 10**6 if self.premium else max(0, self.limit - self.used)

    def call(self, params: dict, want_json: bool = True):
        if self.left() <= 0:
            return None
        for attempt in range(2):
            wait = av_gap() - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get("https://www.alphavantage.co/query", params={**params, "apikey": self.key}, timeout=60)
            except Exception as e:
                log(f"AV {params.get('function')} failed: {e}")
                return None
            finally:
                self.last = time.time()
                self.used += 1
                write_json(self.f, {"day": self.day, "used": self.used})
            txt = r.text
            if not txt.lstrip().startswith("{"):
                return None if want_json else txt
            try:
                j = r.json()
            except ValueError:
                return None
            msg = j.get("Information") or j.get("Note") or j.get("Error Message")
            if not msg:
                return j if want_json else None
            if _av_limited(msg):
                if self.premium and attempt == 0:
                    log("AV per-minute cap hit — pausing 60s")
                    time.sleep(60)
                    continue
                self.dead = True
            log(f"AV {params.get('function')}: {msg[:120]}")
            return None
        return None


# ── Yahoo snapshots ──────────────────────────────────────────
def yahoo_fundamentals(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf
    rows, fails = [], 0
    for i, t in enumerate(tickers):
        try:
            info = yf.Ticker(t).info or {}
            if not info:
                raise ValueError("empty")
            rows.append({"ticker": t, **{k: info.get(k) for k in INFO_FIELDS}})
            fails = 0
        except Exception:
            fails += 1
            if fails >= 25:
                log(f"fundamentals: Yahoo stopped answering after {i} tickers")
                break
        time.sleep(0.15)
    return pd.DataFrame(rows)


def yahoo_options(tickers: list[str]) -> pd.DataFrame:
    """Put/call ratios (volume + open interest) and ATM implied vol for the nearest 2 expiries."""
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            exps = list(tk.options or [])[:2]
            try:
                px = float(tk.fast_info["lastPrice"])
            except Exception:
                px = np.nan
            for e in exps:
                ch = tk.option_chain(e)
                c, p = ch.calls, ch.puts
                atm = c.iloc[(c["strike"] - px).abs().argsort()[:1]] if len(c) and not np.isnan(px) else c.head(0)
                rows.append({"ticker": t, "expiry": e, "price": px,
                             "call_vol": c["volume"].fillna(0).sum(), "put_vol": p["volume"].fillna(0).sum(),
                             "call_oi": c["openInterest"].fillna(0).sum(), "put_oi": p["openInterest"].fillna(0).sum(),
                             "atm_iv": float(atm["impliedVolatility"].iloc[0]) if len(atm) else np.nan})
        except Exception:
            continue
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["pc_vol_ratio"] = df["put_vol"] / df["call_vol"].replace(0, np.nan)
        df["pc_oi_ratio"] = df["put_oi"] / df["call_oi"].replace(0, np.nan)
    return df


def yahoo_core_details(tickers: list[str], day: str) -> dict[str, pd.DataFrame]:
    """Analyst actions, price targets, insider trades, earnings surprises, institutions, news — core list only."""
    import yfinance as yf
    out = {k: [] for k in ("analyst_actions", "analyst_targets", "insiders", "earnings", "institutions", "news")}
    for t in tickers:
        tk = yf.Ticker(t)
        try:
            ud = tk.upgrades_downgrades
            if ud is not None and len(ud):
                ud = ud.reset_index().head(40)
                ud.insert(0, "ticker", t)
                out["analyst_actions"].append(ud)
        except Exception:
            pass
        try:
            pt = tk.analyst_price_targets
            if pt:
                out["analyst_targets"].append(pd.DataFrame([{"ticker": t, **pt}]))
        except Exception:
            pass
        try:
            it = tk.insider_transactions
            if it is not None and len(it):
                it = it.head(60).copy()
                it.insert(0, "ticker", t)
                out["insiders"].append(it)
        except Exception:
            pass
        try:
            ed = tk.get_earnings_dates(limit=12)
            if ed is not None and len(ed):
                ed = ed.reset_index()
                ed.insert(0, "ticker", t)
                out["earnings"].append(ed)
        except Exception:
            pass
        try:
            ih = tk.institutional_holders
            if ih is not None and len(ih):
                ih = ih.copy()
                ih.insert(0, "ticker", t)
                out["institutions"].append(ih)
        except Exception:
            pass
        try:
            for n in (tk.news or [])[:10]:
                c = n.get("content", n)
                out["news"].append(pd.DataFrame([{"ticker": t, "title": c.get("title"),
                                                  "time": c.get("pubDate") or c.get("providerPublishTime"),
                                                  "source": (c.get("provider") or {}).get("displayName") or c.get("publisher"),
                                                  "url": (c.get("canonicalUrl") or {}).get("url") or c.get("link")}]))
        except Exception:
            pass
        time.sleep(0.2)
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in out.items()}


# ── Alpha Vantage snapshots ──────────────────────────────────
def av_capture(av: AVBudget, core: list[str], folder: Path, cfg: dict) -> dict:
    got: dict[str, int] = {}
    if not av.key:
        return {"skipped": "no ALPHAVANTAGE_API_KEY secret"}
    raw = folder / "av_raw"
    # 1) earnings calendar for every US stock (CSV)
    txt = av.call({"function": "EARNINGS_CALENDAR", "horizon": "3month"}, want_json=False)
    if txt:
        import io
        got["earnings_calendar"] = _save_csv(pd.read_csv(io.StringIO(txt)), folder / "earnings_calendar.csv.gz")
    # 2) market movers
    j = av.call({"function": "TOP_GAINERS_LOSERS"})
    if j:
        _save_raw(j, raw / "top_gainers_losers.json.gz")
        rows = [{"list": k, **r} for k in ("top_gainers", "top_losers", "most_actively_traded") for r in j.get(k, [])]
        got["movers"] = _save_csv(pd.DataFrame(rows), folder / "movers.csv.gz")
    # 3) market-wide news sentiment (one call covers hundreds of tickers)
    j = av.call({"function": "NEWS_SENTIMENT", "sort": "LATEST", "limit": "1000"})
    if j:
        _save_raw(j, raw / "news_sentiment.json.gz")
        rows = []
        for a in j.get("feed", []):
            for ts in a.get("ticker_sentiment", []):
                rows.append({"time": a.get("time_published"), "title": a.get("title"), "source": a.get("source"),
                             "overall_score": a.get("overall_sentiment_score"), "ticker": ts.get("ticker"),
                             "relevance": ts.get("relevance_score"), "ticker_score": ts.get("ticker_sentiment_score"),
                             "ticker_label": ts.get("ticker_sentiment_label")})
        got["news_sentiment"] = _save_csv(pd.DataFrame(rows), folder / "news_sentiment.csv.gz")
    # 4) rates: 2y + 10y Treasury (yield curve)
    for m in ("2year", "10year"):
        j = av.call({"function": "TREASURY_YIELD", "interval": "daily", "maturity": m})
        if j:
            d = pd.DataFrame(j.get("data", [])[:30])
            d.insert(0, "maturity", m)
            got[f"treasury_{m}"] = _save_csv(d, folder / f"treasury_{m}.csv.gz")
    # 5) rotating deep-dive through the core list with whatever budget is left
    keep = cfg.get("capture", {}).get("av_reserve", 4)  # leave a few calls for the rest of the day
    rot_f = STATE_DIR / "capture_rotation.json"
    rot = read_json(rot_f, {"i": 0})
    stocks = [t for t in core if asset_class(t, cfg) == "stock"]
    deep = {"earnings_history": [], "estimates": [], "insiders_av": []}
    while stocks and av.left() > keep:
        t = stocks[rot["i"] % len(stocks)]
        rot["i"] += 1
        for fn, key in (("EARNINGS", "earnings_history"), ("EARNINGS_ESTIMATES", "estimates"),
                        ("INSIDER_TRANSACTIONS", "insiders_av")):
            if av.left() <= keep:
                break
            j = av.call({"function": fn, "symbol": t})
            if not j:
                continue
            _save_raw(j, raw / f"{fn.lower()}_{t}.json.gz")
            if fn == "EARNINGS":
                d = pd.DataFrame(j.get("quarterlyEarnings", [])[:16])
            elif fn == "INSIDER_TRANSACTIONS":
                d = pd.DataFrame(j.get("data", [])[:100])
            else:
                lst = next((v for v in j.values() if isinstance(v, list)), [])
                d = pd.json_normalize(lst)
            if not d.empty:
                d = d.drop(columns=["ticker"], errors="ignore")
                d.insert(0, "ticker", t)
                deep[key].append(d)
        if rot["i"] % len(stocks) == 0:
            break  # one full lap per day at most
    write_json(rot_f, rot)
    for key, parts in deep.items():
        if parts:
            got[key] = _save_csv(pd.concat(parts, ignore_index=True), folder / f"{key}.csv.gz")
    got["av_calls_used_today"] = av.used
    return got


# ── main job ─────────────────────────────────────────────────
def capture() -> dict:
    t0 = time.time()
    cfg = load_config()
    cc = cfg.get("capture", {})
    core = list(dict.fromkeys([t.upper() for t in cfg["watchlist"]] + list(cfg["holdings"]) + cfg["indices"]))
    spx = [r["ticker"] for r in sp500()]
    extra = list(cfg["sectors"]) + cfg["leveraged"] + list(cfg["forex"]) + list(cfg["commodities"]) + cfg["crypto"] + \
        [cfg["vix"], "^TNX", "^IRX", "^FVX", "DX-Y.NYB"]
    universe = list(dict.fromkeys(core + spx + extra))

    hist = fetch_history(universe, "1d", cfg, period="1mo")
    day = max((df.index[-1] for t, df in hist.items() if asset_class(t, cfg) == "stock"), default=pd.Timestamp.now())
    day = pd.Timestamp(day).date().isoformat()
    folder = VAULT / "daily" / day
    log(f"capturing {day} -> {folder}")
    man: dict = {"day": day, "started": datetime.now(timezone.utc).isoformat(), "files": {}}

    # prices: the session's bar for every ticker (+ last 5 sessions so a missed day self-heals)
    rows = []
    for t, df in hist.items():
        for ts, r in df.tail(5).iterrows():
            rows.append({"ticker": t, "date": ts.date().isoformat(), "open": r.open, "high": r.high, "low": r.low,
                         "close": r.close, "volume": r.volume, "cls": asset_class(t, cfg)})
    man["files"]["prices"] = _save_csv(pd.DataFrame(rows), folder / "prices.csv.gz")

    # one-time long backfill of daily history so the Edge Lab has years to test on
    hist_dir = VAULT / "history"
    if cc.get("backfill", True) and not (hist_dir / "_done").exists():
        log("backfilling 10 years of daily prices (one time)…")
        full = fetch_history(universe, "1d", cfg, period="10y")
        for t, df in full.items():
            _save_csv(df.reset_index(), hist_dir / f"{t.replace('^', '_').replace('=', '_')}.csv.gz")
        hist_dir.mkdir(parents=True, exist_ok=True)
        (hist_dir / "_done").write_text(day)
        man["files"]["history_backfill"] = len(full)

    stocks = [t for t in universe if asset_class(t, cfg) == "stock" and t not in cfg["sectors"]
              and t not in cfg["leveraged"] and t not in cfg["indices"]]
    man["files"]["fundamentals"] = _save_csv(yahoo_fundamentals(stocks if cc.get("fundamentals_all", True) else core),
                                             folder / "fundamentals.csv.gz")
    core_stocks = [t for t in core if asset_class(t, cfg) == "stock"]
    man["files"]["options"] = _save_csv(yahoo_options(core_stocks), folder / "options.csv.gz")
    for k, df in yahoo_core_details([t for t in core_stocks if t not in cfg["indices"]], day).items():
        man["files"][k] = _save_csv(df, folder / f"{k}.csv.gz")

    av = AVBudget(cc.get("av_daily_limit", 25))
    man["alpha_vantage"] = av_capture(av, core, folder, cfg)

    # Finnhub (insiders, analyst trends, earnings surprises, news, metrics) + FRED (macro)
    from .sources import finnhub_capture, fred_capture
    fh_list = list(dict.fromkeys([t for t in core_stocks if t not in cfg["indices"]] + cc.get("finnhub_extra", [])))
    for k, df in finnhub_capture(fh_list).items():
        man["files"][k] = _save_csv(df, folder / f"{k}.csv.gz")
    man["files"]["fred_macro"] = _save_csv(fred_capture(), folder / "fred_macro.csv.gz")

    man["seconds"] = round(time.time() - t0)
    man["finished"] = datetime.now(timezone.utc).isoformat()
    write_json(folder / "manifest.json", man)
    # small index for the dashboard
    days = sorted(p.name for p in (VAULT / "daily").iterdir() if p.is_dir())
    write_json(SITE_DATA / "vault.json", {"days": len(days), "first": days[0] if days else None, "last": day,
                                          "latest": man, "size_mb": round(sum(f.stat().st_size for f in VAULT.rglob("*") if f.is_file()) / 1e6, 1)})
    log(f"capture done in {man['seconds']}s: {man['files']} | AV: {man['alpha_vantage']}")
    return man


def load_days(name: str, start: str | None = None) -> pd.DataFrame:
    """Read one table across every captured day, e.g. load_days('fundamentals'). Adds a capture_date column."""
    parts = []
    for d in sorted((VAULT / "daily").glob("*")):
        if start and d.name < start:
            continue
        f = d / f"{name}.csv.gz"
        if f.exists():
            df = pd.read_csv(f)
            df.insert(0, "capture_date", d.name)
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
