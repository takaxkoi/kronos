"""Filings Lab: free SEC filings for every S&P 500 company, turned into numbers and tested.

1. Downloads each company's reported financials from SEC EDGAR (free, no key): every annual report (10-K)
   on file, using the value as FIRST reported and the date it was filed, so nothing looks into the future.
2. Builds fundamental factors per company per year (growth, margins, returns on capital, accruals, leverage...).
3. Tests each factor: sort companies into five groups by the factor at filing time, then measure how each
   group's stock did over the following 12 months against the average stock.
4. Scores today's companies on the factors that actually worked, and lists the newest filings for your watchlist.

Limits, stated on the page: only companies in the S&P 500 today (so past failures are missing, which flatters
results), annual data only, and no price-based valuation factors yet (share counts are not split-adjusted).
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .core import ROOT, SITE_DATA, load_config, log, read_json, write_json

# SEC asks every automated visitor to identify itself with a name and a real contact email.
# Set yours in config.yaml -> filings.contact (or the SEC_CONTACT secret), e.g. "Lucky Smith lucky@example.com".
UA = {"User-Agent": os.environ.get("SEC_CONTACT") or "Trading-picker research takaxkoi@users.noreply.github.com", "Accept-Encoding": "gzip, deflate"}
_SEEN: set = set()
CACHE = ROOT / "data" / "filings" / "annual.csv.gz"
TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"],
    "rd": ["ResearchAndDevelopmentExpense"],
}
INSTANT = {"assets", "liabilities", "equity"}
FACTORS = {  # name: (plain-English meaning, function of row)
    "Revenue growth": ("Sales this year vs last year.", lambda r: r.revenue / r.revenue_prev - 1),
    "Earnings growth": ("Net income this year vs last year (only when last year was profitable).", lambda r: r.net_income / r.net_income_prev - 1 if r.net_income_prev > 0 else np.nan),
    "Net margin": ("Profit kept from each dollar of sales.", lambda r: r.net_income / r.revenue),
    "Operating margin": ("Operating profit per dollar of sales.", lambda r: r.op_income / r.revenue),
    "Return on equity": ("Profit per dollar of shareholder money.", lambda r: r.net_income / r.equity if r.equity > 0 else np.nan),
    "Return on assets": ("Profit per dollar of everything the company owns.", lambda r: r.net_income / r.assets),
    "Gross profitability": ("Gross profit per dollar of assets.", lambda r: r.gross_profit / r.assets),
    "Cash conversion": ("Operating cash flow minus net income, per dollar of assets. High means profits are backed by cash.", lambda r: (r.cfo - r.net_income) / r.assets),
    "Free cash flow margin": ("Cash left after investment per dollar of sales.", lambda r: (r.cfo - r.capex) / r.revenue),
    "Asset growth": ("How fast the balance sheet grew. Fast growers have tended to disappoint.", lambda r: r.assets / r.assets_prev - 1),
    "Leverage": ("Liabilities as a share of assets.", lambda r: r.liabilities / r.assets),
    "Share count change": ("Rising means dilution; falling means buybacks.", lambda r: (r.shares / r.shares_prev - 1) if 0.75 < r.shares / r.shares_prev < 1.25 else np.nan),
    "R&D intensity": ("Research spending per dollar of sales.", lambda r: r.rd / r.revenue),
    "Margin change": ("Net margin this year minus last year.", lambda r: r.net_income / r.revenue - r.net_income_prev / r.revenue_prev),
}


def sec_get(url: str):
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in _SEEN:
                _SEEN.add(r.status_code)
                log(f"filings: SEC answered HTTP {r.status_code} for {url[:70]}")
            if r.status_code in (403, 429):
                time.sleep(11)  # SEC asks for a pause when you go too fast
                continue
            return None
        except Exception:
            time.sleep(2)
    return None


def annual_rows(ticker: str, facts: dict) -> list[dict]:
    """One row per fiscal year: first-reported values from the 10-K and the date that 10-K was filed."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    by_end: dict[str, dict] = {}
    for field, tags in TAGS.items():
        for tag in tags:
            units = (gaap.get(tag) or {}).get("units") or {}
            for x in units.get("USD") or units.get("shares") or []:
                if x.get("form") not in ("10-K", "10-K405", "20-F", "40-F") or not x.get("end") or x.get("val") is None:
                    continue
                if field not in INSTANT:
                    if not x.get("start"):
                        continue
                    days = (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days
                    if not 340 <= days <= 380:
                        continue
                row = by_end.setdefault(x["end"], {})
                cur = row.get(field)
                if cur is None or (cur[2] == tag and x["filed"] < cur[1]):  # first tag wins; within a tag, earliest filing wins
                    row[field] = (float(x["val"]), x["filed"], tag)
    out = []
    for end, row in by_end.items():
        if "net_income" not in row or "assets" not in row:
            continue
        filed = min(row["net_income"][1], row["assets"][1])
        if (pd.Timestamp(filed) - pd.Timestamp(end)).days > 150:  # a later report restating this year; not point-in-time
            continue
        out.append({"ticker": ticker, "end": end, "filed": filed, **{k: v[0] for k, v in row.items()}})
    return out


def download(tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    m = sec_get("https://www.sec.gov/files/company_tickers.json") or {}
    cik = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in m.values()}
    cik.update({v["ticker"].upper(): int(v["cik_str"]) for v in m.values()})
    rows, missing = [], []
    for i, t in enumerate(tickers):
        c = cik.get(t) or cik.get(t.replace("-", "."))
        if not c:
            missing.append(t)
            continue
        f = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json")
        if f:
            rows += annual_rows(t, f)
        else:
            missing.append(t)
        time.sleep(0.13)  # SEC fair-use limit is 10 requests a second
        if i % 100 == 0:
            log(f"filings: {i}/{len(tickers)} companies")
    return pd.DataFrame(rows), {"cik": cik, "missing": missing}


def recent_filings(watch: list[str], cik: dict) -> list[dict]:
    keep = {"8-K": "Material event", "10-Q": "Quarterly report", "10-K": "Annual report", "4": "Insider trade", "SC 13D": "Activist stake",
            "SC 13G": "Large holder", "SCHEDULE 13D": "Activist stake", "SCHEDULE 13G": "Large holder", "DEF 14A": "Proxy statement", "S-1": "New share offering", "144": "Insider sale notice"}
    out = []
    for t in watch:
        c = cik.get(t)
        j = sec_get(f"https://data.sec.gov/submissions/CIK{c:010d}.json") if c else None
        time.sleep(0.13)
        if not j:
            continue
        r = (j.get("filings") or {}).get("recent") or {}
        n = 0
        for form, date, acc, doc, desc in zip(r.get("form", []), r.get("filingDate", []), r.get("accessionNumber", []), r.get("primaryDocument", []), r.get("primaryDocDescription", [])):
            if form not in keep or n >= 6:
                continue
            n += 1
            out.append({"ticker": t, "form": form, "kind": keep[form], "date": date, "what": (desc or "")[:90],
                        "url": f"https://www.sec.gov/Archives/edgar/data/{c}/{acc.replace('-', '')}/{doc}"})
    return sorted(out, key=lambda x: x["date"], reverse=True)


def add_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "end"]).reset_index(drop=True)
    df["end"], df["filed"] = pd.to_datetime(df["end"]), pd.to_datetime(df["filed"])
    for c in TAGS:
        if c not in df:
            df[c] = np.nan
    df["liabilities"] = df["liabilities"].fillna(df["assets"] - df["equity"])
    g = df.groupby("ticker")
    gap = (df["end"] - g["end"].shift()).dt.days
    for c in ("revenue", "net_income", "assets", "shares"):
        df[c + "_prev"] = g[c].shift().where(gap.between(330, 400))
    with np.errstate(divide="ignore", invalid="ignore"):
        for name, (_, fn) in FACTORS.items():
            vals = []
            for r in df.itertuples(index=False):
                try:
                    v = float(fn(r))
                    vals.append(v if np.isfinite(v) else np.nan)
                except Exception:
                    vals.append(np.nan)
            df[name] = vals
    return df


def forward_returns(df: pd.DataFrame, close: pd.DataFrame, hold: int = 252) -> pd.DataFrame:
    idx = close.index
    mkt = (1 + close.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()  # the average stock, equal weight
    fwd, exc = [], []
    for r in df.itertuples(index=False):
        f, e = np.nan, np.nan
        if r.ticker in close.columns:
            i = idx.searchsorted(r.filed + pd.Timedelta(days=1))
            if i + hold < len(idx):
                a, b = close[r.ticker].iloc[i], close[r.ticker].iloc[i + hold]
                if a > 0 and b > 0:
                    f = b / a - 1
                    e = f - (mkt.iloc[i + hold] / mkt.iloc[i] - 1)
        fwd.append(f)
        exc.append(e)
    df["fwd"], df["excess"] = fwd, exc
    return df


def test_factor(df: pd.DataFrame, name: str) -> dict | None:
    d = df[[name, "excess", "year"]].dropna()
    if len(d) < 400:
        return None
    lo, hi = d[name].quantile([0.01, 0.99])
    d = d.assign(v=d[name].clip(lo, hi))
    spreads, quint = [], []
    for y, grp in d.groupby("year"):
        if len(grp) < 100:
            continue
        q = pd.qcut(grp["v"].rank(method="first"), 5, labels=False)
        m = grp.groupby(q)["excess"].median()  # median: a handful of ten-baggers must not decide the result
        if len(m) == 5:
            quint.append(m.values)
            spreads.append((int(y), float(m.iloc[4] - m.iloc[0])))
    if len(spreads) < 4:
        return None
    s = np.array([x[1] for x in spreads])
    t = float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if s.std(ddof=1) > 0 else 0.0
    pos = float((s > 0).mean())
    return {"name": name, "why": FACTORS[name][0], "spread": float(s.mean()), "t": t, "years": len(s), "years_positive": pos, "n": int(len(d)),
            "quintiles": [float(x) for x in np.mean(quint, axis=0)], "by_year": [{"year": y, "spread": v} for y, v in spreads],
            "better": "high" if s.mean() > 0 else "low", "works": bool(abs(t) >= 2 and max(pos, 1 - pos) >= 0.65)}


def run(cached: bool = False) -> dict:
    """cached=True re-runs the analysis on the last saved download without calling SEC."""
    from .edge import load_closes
    cfg = load_config()
    contact = (cfg.get("filings") or {}).get("contact")
    if contact and not os.environ.get("SEC_CONTACT"):
        UA["User-Agent"] = str(contact)
    close = load_closes(years=11, min_bars=300)
    tickers = [t for t in close.columns if t.replace("-", "").isalpha() and t not in ("SPY", "QQQ", "DIA", "IWM") and not t.startswith("XL")]
    out = {"generated": datetime.now(timezone.utc).isoformat()}
    prev = read_json(SITE_DATA / "filings.json", {})
    raw, meta = (pd.read_csv(CACHE), {}) if cached and CACHE.exists() else download(tickers)
    if cached and CACHE.exists():
        pass
    elif raw.empty:
        if CACHE.exists():
            raw = pd.read_csv(CACHE)
            out["note"] = "SEC did not answer this run; using the last saved download."
        else:
            out["status"] = ("SEC EDGAR refused the download" + (f" (HTTP {sorted(_SEEN)[0]})" if _SEEN else "") + ". SEC requires a real contact email on automated requests: add yours in app/config.yaml under filings.contact, then run this job again.")
            write_json(SITE_DATA / "filings.json", out)
            return out
    else:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(CACHE, index=False)
    df = forward_returns(add_factors(raw), close)
    df["year"] = df["filed"].dt.year
    res = [r for r in (test_factor(df, n) for n in FACTORS) if r]
    res.sort(key=lambda r: abs(r["t"]), reverse=True)
    works = [r for r in res if r["works"]]
    log("filings: " + "; ".join(f"{r['name']} {r['spread']:+.1%} t={r['t']:.1f}" for r in res))

    latest = df.sort_values("end").groupby("ticker").tail(1)
    latest = latest[latest["filed"] >= latest["filed"].max() - pd.Timedelta(days=500)].copy()
    use = works or res[:3]
    for r in res:
        latest["p_" + r["name"]] = latest[r["name"]].rank(pct=True)
    parts = [latest["p_" + r["name"]] if r["better"] == "high" else 1 - latest["p_" + r["name"]] for r in use]
    latest["composite"] = pd.concat(parts, axis=1).mean(axis=1, skipna=True) * 100 if parts else np.nan

    def card(r):
        return {"ticker": r["ticker"], "fiscal_year_end": r["end"].date().isoformat(), "filed": r["filed"].date().isoformat(),
                "composite": None if pd.isna(r["composite"]) else round(float(r["composite"])),
                "factors": {x["name"]: {"value": None if pd.isna(r[x["name"]]) else round(float(r[x["name"]]), 4),
                                        "pct": None if pd.isna(r["p_" + x["name"]]) else round(float(r["p_" + x["name"]]) * 100)} for x in res}}
    watch = [t.upper() for t in cfg.get("watchlist", [])]
    rows = {r["ticker"]: r for _, r in latest.iterrows()}
    out.update({
        "status": "ok", "companies": int(df["ticker"].nunique()), "filings": int(len(df)), "tested": int(df["excess"].notna().sum()),
        "first_year": int(df["year"].min()), "last_year": int(df["year"].max()), "factors": res, "scored_on": [r["name"] for r in use],
        "scored_on_proven": bool(works),
        "watchlist": [card(rows[t]) for t in watch if t in rows],
        "top": [card(r) for _, r in latest.dropna(subset=["composite"]).sort_values("composite", ascending=False).head(20).iterrows()],
        "bottom": [card(r) for _, r in latest.dropna(subset=["composite"]).sort_values("composite").head(10).iterrows()],
        "recent": recent_filings(watch, meta.get("cik", {})) if meta.get("cik") else prev.get("recent", []),
        "missing": len(meta.get("missing", [])) if meta else prev.get("missing", 0),
        "verdict": (f"{len(works)} of {len(res)} filing factors separated winners from losers in most years: " + ", ".join(f"{r['name']} ({r['better']} is better, {abs(r['spread']):.1%} a year)" for r in works) + "."
                    if works else f"None of the {len(res)} filing factors separated winners from losers reliably enough to trust. The score below uses the three strongest and should be read as a starting point for research."),
        "caveats": "Read these with care. The test only includes companies that are in the S&P 500 today, so every past loser that went bust or was dropped is missing. That flatters anything that looks risky at filing time (low margins, heavy R&D, fast growth), because the risky companies still in the index are by definition the ones that made it. Annual reports only. Each group's result is its median stock over the 12 months after filing, against the average stock.",
    })
    write_json(SITE_DATA / "filings.json", out)
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    import sys
    run(cached="--cached" in sys.argv)
