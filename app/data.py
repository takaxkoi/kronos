"""Market data: prices (Yahoo Finance, Alpaca fallback), trading calendars,
S&P 500 list, sectors, earnings dates, news headlines and option chains."""
from __future__ import annotations

import io
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)
from pandas.tseries.offsets import CustomBusinessDay

from .core import STATE_DIR, log, read_json, write_json

UA = {"User-Agent": "Mozilla/5.0 (KronosOracle; personal research)"}
PERIODS = {"1d": "5y", "1h": "730d", "1wk": "max"}


# ── calendars ────────────────────────────────────────────────
class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_DAY = CustomBusinessDay(calendar=NYSECalendar())


def asset_class(ticker: str, cfg: dict | None = None) -> str:
    t = ticker.upper()
    if cfg and t in [c.upper() for c in cfg.get("crypto", [])]:
        return "crypto"
    if t.endswith("-USD"):
        return "crypto"
    if t.endswith("=X"):
        return "forex"
    if t.endswith("=F"):
        return "commodity"
    if t.startswith("^"):
        return "index"
    return "stock"


def future_index(last: pd.Timestamp, n: int, interval: str, cls: str) -> pd.DatetimeIndex:
    """Timestamps of the next n bars after `last`."""
    last = pd.Timestamp(last)
    if interval == "1wk":
        return pd.DatetimeIndex([last + pd.Timedelta(weeks=i) for i in range(1, n + 1)])
    if interval == "1d":
        if cls == "crypto":
            return pd.date_range(last + pd.Timedelta(days=1), periods=n, freq="D")
        if cls in ("forex", "commodity"):
            return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
        return pd.date_range(last.normalize() + NYSE_DAY, periods=n, freq=NYSE_DAY)
    if interval == "1h":
        if cls == "crypto":
            return pd.date_range(last + pd.Timedelta(hours=1), periods=n, freq="h")
        # US stock hourly bars: 9:30, 10:30 ... 15:30 on trading days
        sessions = pd.date_range(last.normalize(), periods=n // 7 + 3, freq=NYSE_DAY)
        if last.normalize() not in sessions:
            sessions = sessions.insert(0, last.normalize())
        bars = [d + pd.Timedelta(hours=9 + k, minutes=30) for d in sessions for k in range(7)]
        return pd.DatetimeIndex([b for b in bars if b > last][:n])
    raise ValueError(interval)


# ── prices ───────────────────────────────────────────────────
def _standardize(df: pd.DataFrame, cls: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "close" not in df.columns:
        return pd.DataFrame()
    df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
    df = df.dropna(subset=["close"])
    for c in ["open", "high", "low"]:
        if c not in df.columns:
            df[c] = df["close"]
        df[c] = df[c].fillna(df["close"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        tz = "UTC" if cls == "crypto" else "America/New_York"
        idx = idx.tz_convert(tz).tz_localize(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df.index.name = "timestamps"
    return df.astype(float)


def _drop_partial(df: pd.DataFrame, interval: str, cls: str) -> pd.DataFrame:
    """Drop a bar that is still forming (we only feed Kronos finished candles)."""
    if df.empty:
        return df
    now_utc = datetime.now(timezone.utc)
    last = df.index[-1]
    if interval == "1h":
        tz = "UTC" if cls == "crypto" else "America/New_York"
        end = pd.Timestamp(last).tz_localize(tz) + pd.Timedelta(hours=1)
        if end > pd.Timestamp(now_utc):
            return df.iloc[:-1]
    elif interval == "1d" and cls == "stock":
        ny = pd.Timestamp(now_utc).tz_convert("America/New_York")
        if last.date() == ny.date() and (ny.hour < 16):
            return df.iloc[:-1]
    elif interval == "1wk":
        # weekly bars are stamped with the week's Monday; unfinished until that Friday's close
        ny = pd.Timestamp(now_utc).tz_convert("America/New_York").tz_localize(None)
        if ny < pd.Timestamp(last).normalize() + pd.Timedelta(days=4, hours=16):
            return df.iloc[:-1]
    elif interval == "1d" and cls in ("crypto", "forex", "commodity"):
        if last.date() >= now_utc.date():
            return df.iloc[:-1]
    return df


def fetch_history(tickers: list[str], interval: str = "1d", cfg: dict | None = None,
                  period: str | None = None) -> dict[str, pd.DataFrame]:
    """Download candles for many tickers. Returns {ticker: DataFrame}."""
    import yfinance as yf

    period = period or PERIODS[interval]
    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    out: dict[str, pd.DataFrame] = {}
    # Alpha Vantage first for your core list when you choose it (paid keys can cover everything)
    av = (cfg or {}).get("data", {}).get("alphavantage_first", [])
    if os.environ.get("ALPHAVANTAGE_API_KEY") and av_premium() and av:
        for t in tickers:
            if interval == "1d" and (av == "all" or t in av) and asset_class(t, cfg) == "stock":
                df = _drop_partial(fetch_alphavantage(t, interval), interval, "stock")
                if not df.empty:
                    out[t] = df
        tickers = [t for t in tickers if t not in out]
    for start in range(0, len(tickers), 80):
        chunk = tickers[start:start + 80]
        raw = None
        for attempt in range(4):
            try:
                raw = yf.download(chunk, period=period, interval=interval, auto_adjust=True,
                                  group_by="ticker", threads=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception as e:  # rate limits, network blips
                log(f"yahoo attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
        for t in chunk:
            cls = asset_class(t, cfg)
            try:
                if raw is None or raw.empty:
                    raise KeyError
                if isinstance(raw.columns, pd.MultiIndex):
                    sub = raw[t] if t in raw.columns.get_level_values(0) else pd.DataFrame()
                else:
                    sub = raw
                df = _drop_partial(_standardize(sub, cls), interval, cls)
            except Exception:
                df = pd.DataFrame()
            if df.empty and cls == "stock":
                df = _drop_partial(fetch_alpaca(t, interval), interval, cls)
            if df.empty and cls == "stock" and av_premium():
                df = _drop_partial(fetch_alphavantage(t, interval), interval, cls)
            if not df.empty:
                out[t] = df
    missing = [t for t in list(dict.fromkeys(x.upper() for x in tickers)) if t not in out]
    if missing:
        log(f"no data for {len(missing)}: {', '.join(missing[:15])}")
    return out


_AV = {"exhausted": False, "last": 0.0, "earnings": None, "cache": {}}


def av_premium() -> bool:
    return os.environ.get("ALPHAVANTAGE_PREMIUM", "").lower() in ("1", "true", "yes")


def av_gap() -> float:
    """Seconds between Alpha Vantage calls.
    Free key: ~1 call/sec (and 25/day). Paid key: stay under the plan's per-minute cap
    (AV_CALLS_PER_MIN, default 66 = headroom under the $49.99 plan's 75/min)."""
    if not av_premium():
        return 1.3
    try:
        per_min = max(1, int(os.environ.get("AV_CALLS_PER_MIN", "66")))
    except ValueError:
        per_min = 66
    return 60.0 / per_min


def _av_limited(text: str) -> bool:
    t = text.lower()
    return "rate limit" in t or "requests per day" in t or "per minute" in t or "spreading out" in t


def _av_get(params: dict) -> str | None:
    """One throttled Alpha Vantage call. Returns CSV text or None."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key or _AV["exhausted"]:
        return None
    for attempt in range(2):
        wait = av_gap() - (time.time() - _AV["last"])
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get("https://www.alphavantage.co/query", params={**params, "apikey": key}, timeout=40)
            text = r.text
        except Exception as e:
            log(f"alpha vantage failed: {e}")
            return None
        finally:
            _AV["last"] = time.time()
        if not text.lstrip().startswith("{"):
            return text
        # JSON here means an error, a premium-only notice, or a rate-limit notice
        if not _av_limited(text):
            return None
        if av_premium() and attempt == 0:
            log("alpha vantage per-minute cap hit — pausing 60s")
            time.sleep(60)
            continue
        _AV["exhausted"] = True
        log("alpha vantage limit reached — falling back to Yahoo/Alpaca")
        return None
    return None


def fetch_alphavantage(ticker: str, interval: str, cls: str = "stock") -> pd.DataFrame:
    """US stocks / ETFs from Alpha Vantage (secret ALPHAVANTAGE_API_KEY)."""
    if cls != "stock" or interval not in ("1d", "1wk", "1h"):
        return pd.DataFrame()
    hit = _AV["cache"].get((ticker.upper(), interval))
    if hit is not None:  # same ticker twice in one run costs one call
        return hit.copy()
    premium = av_premium()
    # free keys only get the raw (not split-adjusted) daily series; paid keys get adjusted prices
    fn = {"1d": "TIME_SERIES_DAILY_ADJUSTED" if premium else "TIME_SERIES_DAILY",
          "1wk": "TIME_SERIES_WEEKLY_ADJUSTED", "1h": "TIME_SERIES_INTRADAY"}[interval]
    params = {"function": fn, "symbol": ticker.replace("-", "."), "datatype": "csv", "outputsize": "full" if premium or interval != "1d" else "compact"}
    if interval == "1h":
        params.update({"interval": "60min", "extended_hours": "false"})
    text = _av_get(params)
    if not text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    df = df.rename(columns={"timestamp": "ts", "adjusted close": "adj_close"})
    if "ts" not in df.columns:
        return pd.DataFrame()
    df.index = pd.to_datetime(df["ts"])
    if "adj_close" in df.columns:  # back-adjust OHLC for splits/dividends
        f = df["adj_close"] / df["close"]
        for c in ("open", "high", "low"):
            df[c] = df[c] * f
        df["close"] = df["adj_close"]
    if interval == "1h":  # AV stamps bars at their END (10:30 = 9:30-10:30); match Yahoo's start-stamps
        df.index = df.index - pd.Timedelta(hours=1)
        df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time <= pd.Timestamp("15:30").time())]
    if interval == "1wk":  # AV stamps the week's Friday; use Monday like Yahoo
        df.index = df.index.normalize() - pd.to_timedelta(df.index.weekday, unit="D")
    out = _standardize(df[["open", "high", "low", "close", "volume"]].sort_index(), "stock")
    _AV["cache"][(ticker.upper(), interval)] = out
    return out.copy()


def av_earnings() -> dict[str, str]:
    """Every upcoming US earnings date in ONE call (next 3 months)."""
    if _AV["earnings"] is None:
        text = _av_get({"function": "EARNINGS_CALENDAR", "horizon": "3month"})
        cal: dict[str, str] = {}
        if text:
            try:
                for _, r in pd.read_csv(io.StringIO(text)).iterrows():
                    s = str(r["symbol"]).replace(".", "-")
                    if s not in cal or str(r["reportDate"]) < cal[s]:
                        cal[s] = str(r["reportDate"])
            except Exception:
                pass
        _AV["earnings"] = cal
    return _AV["earnings"]


def fetch_alpaca(ticker: str, interval: str) -> pd.DataFrame:
    """Fallback for US stocks when Yahoo blocks the runner (needs ALPACA keys)."""
    key, sec = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec or interval not in ("1d", "1h", "1wk"):
        return pd.DataFrame()
    tf = {"1d": "1Day", "1h": "1Hour", "1wk": "1Week"}[interval]
    days = {"1d": 1900, "1h": 720, "1wk": 4000}[interval]
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    rows, token = [], None
    try:
        while True:
            params = {"timeframe": tf, "start": start, "limit": 10000, "adjustment": "all", "feed": "iex"}
            if token:
                params["page_token"] = token
            r = requests.get(f"https://data.alpaca.markets/v2/stocks/{ticker}/bars", params=params,
                             headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}, timeout=30)
            r.raise_for_status()
            j = r.json()
            rows += j.get("bars") or []
            token = j.get("next_page_token")
            if not token:
                break
    except Exception as e:
        log(f"alpaca fallback failed for {ticker}: {e}")
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df.index = pd.to_datetime(df["t"], utc=True)
    if interval == "1h":
        df = df[(df.index.tz_convert("America/New_York").time >= pd.Timestamp("09:30").time())
                & (df.index.tz_convert("America/New_York").time <= pd.Timestamp("15:30").time())]
    return _standardize(df[["open", "high", "low", "close", "volume"]], "stock")


def latest_prices(tickers: list[str]) -> dict[str, dict]:
    """Most recent intraday price (for price-hit alerts)."""
    import yfinance as yf
    out = {}
    try:
        raw = yf.download(tickers, period="2d", interval="5m", group_by="ticker", progress=False,
                          auto_adjust=True, threads=True)
    except Exception as e:
        log(f"latest price fetch failed: {e}")
        return out
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            sub = sub.dropna(subset=["Close"])
            today = sub[sub.index.date == sub.index[-1].date()]
            out[t] = {"price": float(sub["Close"].iloc[-1]), "low": float(today["Low"].min()),
                      "high": float(today["High"].max()), "time": sub.index[-1].isoformat()}
        except Exception:
            continue
    return out


# ── universe / reference data ────────────────────────────────
def sp500() -> list[dict]:
    cache = STATE_DIR / "sp500.json"
    c = read_json(cache, {})
    if c and time.time() - c.get("ts", 0) < 7 * 86400:
        return c["rows"]
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30).text
        tbl = pd.read_html(io.StringIO(html))[0]
        rows = [{"ticker": str(r["Symbol"]).replace(".", "-"), "name": r["Security"], "sector": r["GICS Sector"]}
                for _, r in tbl.iterrows()]
        if len(rows) > 400:
            write_json(cache, {"ts": time.time(), "rows": rows})
            return rows
    except Exception as e:
        log(f"S&P 500 list fetch failed: {e}")
    return c.get("rows", [])


def sector_map(tickers: list[str], cfg: dict) -> dict[str, str]:
    m = {r["ticker"]: r["sector"] for r in sp500()}
    cache_p = STATE_DIR / "sectors.json"
    cache = read_json(cache_p, {})
    for etf, name in (cfg.get("sectors") or {}).items():
        m[etf] = f"{name} ETF"
    for t in tickers:
        cls = asset_class(t, cfg)
        if cls != "stock":
            m[t] = {"crypto": "Crypto", "forex": "Forex", "commodity": "Commodities", "index": "Index"}[cls]
    changed = False
    import yfinance as yf
    for t in tickers:
        if t in m:
            continue
        if t in cache:
            m[t] = cache[t]
            continue
        try:
            info = yf.Ticker(t).info or {}
            s = info.get("sector") or ("ETF" if info.get("quoteType") == "ETF" else "Other")
        except Exception:
            s = "Other"
        cache[t] = m[t] = s
        changed = True
    if changed:
        write_json(cache_p, cache)
    return m


def next_earnings(ticker: str) -> str | None:
    if os.environ.get("ALPHAVANTAGE_API_KEY"):
        cal = av_earnings()
        if cal:
            return cal.get(ticker.upper())
    import yfinance as yf
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            today = datetime.now().date()
            future = [d for d in dates if pd.Timestamp(d).date() >= today]
            if future:
                return pd.Timestamp(min(future)).date().isoformat()
    except Exception:
        pass
    return None


def headlines(ticker: str, limit: int = 4) -> list[dict]:
    import yfinance as yf
    out = []
    try:
        for n in (yf.Ticker(ticker).news or [])[:limit]:
            c = n.get("content", n)
            title = c.get("title")
            if not title:
                continue
            url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
            pub = c.get("pubDate") or c.get("providerPublishTime")
            if isinstance(pub, (int, float)):
                pub = datetime.fromtimestamp(pub, timezone.utc).isoformat()
            src = (c.get("provider") or {}).get("displayName") or c.get("publisher")
            out.append({"title": title, "url": url, "time": pub, "source": src})
    except Exception:
        pass
    return out


def option_idea(ticker: str, direction: str, target: float, entry: float, min_days: int) -> dict | None:
    """Pick a listed contract that matches the predicted move (needs live option chain)."""
    import yfinance as yf
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        cutoff = datetime.now().date() + timedelta(days=max(min_days, 1) + 7)
        exp = next((e for e in exps if datetime.strptime(e, "%Y-%m-%d").date() >= cutoff), exps[-1])
        chain = tk.option_chain(exp)
        tbl = chain.calls if direction == "up" else chain.puts
        tbl = tbl[(tbl["openInterest"].fillna(0) > 50) & (tbl["bid"].fillna(0) > 0)]
        if tbl.empty:
            return None
        aim = entry + (target - entry) * 0.4  # slightly out-of-the-money toward the target
        row = tbl.iloc[(tbl["strike"] - aim).abs().argsort().iloc[0]]
        mid = (row["bid"] + row["ask"]) / 2 if row["ask"] > 0 else row["lastPrice"]
        be = row["strike"] + mid if direction == "up" else row["strike"] - mid
        return {"type": "CALL" if direction == "up" else "PUT", "expiry": exp, "strike": float(row["strike"]),
                "bid": float(row["bid"]), "ask": float(row["ask"]), "mid": float(mid),
                "iv": float(row.get("impliedVolatility") or np.nan), "open_interest": int(row["openInterest"]),
                "breakeven": float(be), "target_beats_breakeven": bool((target > be) if direction == "up" else (target < be))}
    except Exception:
        return None
