"""Extra free data sources for the Data Vault and Edge Lab.

  Finnhub (secret FINNHUB_API_KEY, free = 60 calls/min): insider trades + insider sentiment,
      analyst recommendation trends, earnings surprises, company news, key metrics, earnings calendar.
  FRED (secret FRED_API_KEY, free): rates, yield curve, credit spreads, inflation, jobs, VIX, dollar.
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

from .core import log

FRED_SERIES = {
    "DGS10": "10y Treasury", "DGS2": "2y Treasury", "T10Y2Y": "10y-2y curve", "DFF": "Fed funds",
    "BAMLH0A0HYM2": "High-yield spread", "VIXCLS": "VIX", "DTWEXBGS": "Dollar index",
    "CPIAUCSL": "CPI", "UNRATE": "Unemployment", "ICSA": "Jobless claims", "T5YIE": "5y inflation expectations",
}


class Finnhub:
    def __init__(self):
        self.key = os.environ.get("FINNHUB_API_KEY")
        self.last = 0.0
        self.calls = 0
        self.dead = False

    def get(self, path: str, **params):
        if not self.key or self.dead:
            return None
        wait = 1.05 - (time.time() - self.last)  # stay under 60/min
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(f"https://finnhub.io/api/v1/{path}", params={**params, "token": self.key}, timeout=30)
            self.last, self.calls = time.time(), self.calls + 1
            if r.status_code == 429:
                time.sleep(20)
                return None
            if r.status_code in (401, 403):
                log(f"Finnhub {path}: {r.status_code} {r.text[:100]}")
                if r.status_code == 401:
                    self.dead = True
                return None
            return r.json() if r.ok else None
        except Exception as e:
            log(f"Finnhub {path} failed: {e}")
            return None


def finnhub_capture(tickers: list[str]) -> dict[str, pd.DataFrame]:
    fh = Finnhub()
    if not fh.key:
        return {}
    today = date.today()
    out = {k: [] for k in ("fh_insiders", "fh_insider_sentiment", "fh_recommendations", "fh_earnings", "fh_news", "fh_metrics")}
    for t in tickers:
        if fh.dead:
            break
        j = fh.get("stock/insider-transactions", symbol=t, **{"from": (today - timedelta(days=90)).isoformat()})
        if j and j.get("data"):
            out["fh_insiders"].append(pd.DataFrame(j["data"]).assign(ticker=t))
        j = fh.get("stock/insider-sentiment", symbol=t, **{"from": (today - timedelta(days=400)).isoformat(), "to": today.isoformat()})
        if j and j.get("data"):
            out["fh_insider_sentiment"].append(pd.DataFrame(j["data"]).assign(ticker=t))
        j = fh.get("stock/recommendation", symbol=t)
        if isinstance(j, list) and j:
            out["fh_recommendations"].append(pd.DataFrame(j[:6]).assign(ticker=t))
        j = fh.get("stock/earnings", symbol=t)
        if isinstance(j, list) and j:
            out["fh_earnings"].append(pd.DataFrame(j).assign(ticker=t))
        j = fh.get("company-news", symbol=t, **{"from": (today - timedelta(days=2)).isoformat(), "to": today.isoformat()})
        if isinstance(j, list) and j:
            out["fh_news"].append(pd.DataFrame(j[:25])[["datetime", "headline", "source", "url", "summary"]].assign(ticker=t))
        j = fh.get("stock/metric", symbol=t, metric="all")
        if j and j.get("metric"):
            out["fh_metrics"].append(pd.DataFrame([{"ticker": t, **{k: v for k, v in j["metric"].items() if not isinstance(v, (list, dict))}}]))
    res = {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}
    j = fh.get("calendar/earnings", **{"from": today.isoformat(), "to": (today + timedelta(days=30)).isoformat()})
    if j and j.get("earningsCalendar"):
        res["fh_earnings_calendar"] = pd.DataFrame(j["earningsCalendar"])
    log(f"Finnhub: {fh.calls} calls, tables {list(res)}")
    return res


def fred_capture(limit: int = 60) -> pd.DataFrame:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return pd.DataFrame()
    rows = []
    for sid, name in FRED_SERIES.items():
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=30, params={
                "series_id": sid, "api_key": key, "file_type": "json", "sort_order": "desc", "limit": limit})
            if not r.ok:
                log(f"FRED {sid}: {r.status_code} {r.text[:120]}")
                if r.status_code == 400 and "api_key" in r.text:
                    break
                continue
            for o in r.json().get("observations", []):
                rows.append({"series": sid, "name": name, "date": o["date"],
                             "value": pd.to_numeric(o["value"], errors="coerce")})
        except Exception as e:
            log(f"FRED {sid} failed: {e}")
        time.sleep(0.3)
    log(f"FRED: {len(rows)} observations")
    return pd.DataFrame(rows)
