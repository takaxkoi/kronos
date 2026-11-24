"""Economy page: the handful of numbers that drive markets (rates, inflation, jobs, growth, stress),
each with a plain-English read, plus a recession watch and a conditions table tested against what the S&P 500 did next.

Data: FRED (Federal Reserve Bank of St. Louis) public CSV downloads. No key needed.
Every series keeps its last good copy in state/fred/ so one failed download never blanks the page."""
from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from .core import SITE_DATA, STATE_DIR, log, write_json
from .feargreed import build, load_panel, stock_cols

UA = {"User-Agent": "Mozilla/5.0 (kronos-oracle research; personal use)"}
CACHE = STATE_DIR / "fred"
IDS = ["DFF", "DGS2", "DGS10", "T10Y2Y", "T10Y3M", "MORTGAGE30US", "CPIAUCSL", "CPILFESL", "UNRATE", "SAHMREALTIME", "ICSA", "PAYEMS",
       "A191RL1Q225SBEA", "UMCSENT", "BAMLH0A0HYM2", "NFCI", "VIXCLS", "VXVCLS", "DCOILWTICO", "DTWEXBGS"]


SOURCES: set[str] = set()
_fred_dead = {"n": 0}


def _save(sid: str, s: pd.Series) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    s = s.dropna()
    if len(s):
        s.rename(sid).rename_axis("date").to_csv(CACHE / f"{sid}.csv")
    return s


def _saved(sid: str) -> pd.Series:
    f = CACHE / f"{sid}.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_csv(f)
    s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(d.iloc[:, 0])
    return s.dropna()


def fred(sid: str, years: int = 12) -> pd.Series:
    """FRED. With a free FRED_API_KEY it uses the official API (reachable from GitHub). Without one it tries the public CSV,
    which GitHub's servers are often refused, so it gives up after two failures and the fallbacks below take over."""
    start = (datetime.now(timezone.utc) - timedelta(days=365 * years)).date().isoformat()
    key = os.environ.get("FRED_API_KEY")
    try:
        if key:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations", params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": start}, timeout=30)
            r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs}, dtype=float)
        else:
            if _fred_dead["n"] >= 2:
                return pd.Series(dtype=float)
            r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": sid, "cosd": start}, headers=UA, timeout=12)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
            s.index = pd.to_datetime(d.iloc[:, 0])
        s = s.dropna()
        if len(s):
            SOURCES.add("Federal Reserve Bank of St. Louis (FRED)")
        return s
    except Exception as e:
        _fred_dead["n"] += 1
        log(f"FRED {sid}: {type(e).__name__}")
        return pd.Series(dtype=float)


def yahoo(sym: str) -> pd.Series:
    try:
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"interval": "1d", "range": "10y"}, headers=UA, timeout=30)
        res = r.json()["chart"]["result"][0]
        ix = pd.to_datetime(res["timestamp"], unit="s").normalize()
        s = pd.Series(res["indicators"]["quote"][0]["close"], index=ix, dtype=float).dropna()
        s = s[~s.index.duplicated(keep="last")]
        if len(s):
            SOURCES.add("Yahoo Finance")
        return s
    except Exception as e:
        log(f"Yahoo {sym}: {type(e).__name__}")
        return pd.Series(dtype=float)


def treasury(years: int = 9) -> pd.DataFrame:
    """Daily Treasury yield curve from the US Treasury's own site, one small file per year."""
    out, now = [], datetime.now(timezone.utc).year
    for y in range(now - years, now + 1):
        try:
            r = requests.get(f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all",
                             params={"type": "daily_treasury_yield_curve", "field_tdr_date_value": y, "page": "", "_format": "csv"}, headers=UA, timeout=30)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text))
            d["Date"] = pd.to_datetime(d["Date"])
            out.append(d.set_index("Date"))
        except Exception as e:
            log(f"Treasury {y}: {type(e).__name__}")
    if not out:
        return pd.DataFrame()
    SOURCES.add("US Treasury")
    return pd.concat(out).sort_index()


def bls(ids: list[str]) -> dict:
    """Bureau of Labor Statistics public API (no key: 10 years, 25 requests a day)."""
    now = datetime.now(timezone.utc).year
    try:
        r = requests.post("https://api.bls.gov/publicAPI/v1/timeseries/data/", json={"seriesid": ids, "startyear": str(now - 9), "endyear": str(now)}, headers=UA, timeout=40)
        out = {}
        for ser in r.json()["Results"]["series"]:
            rows = {pd.Timestamp(int(x["year"]), int(x["period"][1:]), 1): pd.to_numeric(x["value"], errors="coerce") for x in ser["data"] if x["period"].startswith("M") and x["period"] != "M13"}
            out[ser["seriesID"]] = pd.Series(rows, dtype=float).sort_index().dropna()
        if any(len(v) for v in out.values()):
            SOURCES.add("Bureau of Labor Statistics")
        return out
    except Exception as e:
        log(f"BLS: {type(e).__name__}")
        return {}


def effr() -> pd.Series:
    try:
        end = datetime.now(timezone.utc).date()
        r = requests.get("https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json", params={"startDate": (end - timedelta(days=365 * 10)).isoformat(), "endDate": end.isoformat()}, headers=UA, timeout=30)
        s = pd.Series({pd.Timestamp(x["effectiveDate"]): float(x["percentRate"]) for x in r.json()["refRates"]}, dtype=float).sort_index()
        if len(s):
            SOURCES.add("Federal Reserve Bank of New York")
        return s
    except Exception as e:
        log(f"NY Fed: {type(e).__name__}")
        return pd.Series(dtype=float)


def load() -> dict:
    """Every series from FRED when it answers; otherwise from the agency that publishes it, or a market quote."""
    S = {k: fred(k) for k in IDS}
    need = lambda k: S[k].empty
    if need("DGS10"): S["DGS10"] = yahoo("^TNX")
    if need("VIXCLS"): S["VIXCLS"] = yahoo("^VIX")
    if need("VXVCLS"): S["VXVCLS"] = yahoo("^VIX3M")
    if need("DCOILWTICO"): S["DCOILWTICO"] = yahoo("CL=F")
    if need("DTWEXBGS"): S["DTWEXBGS"] = yahoo("DX-Y.NYB")
    if need("DFF"): S["DFF"] = effr()
    if need("DGS2") or need("T10Y2Y") or need("T10Y3M"):
        t = treasury()
        if len(t):
            col = lambda c: pd.to_numeric(t[c], errors="coerce").dropna() if c in t else pd.Series(dtype=float)
            if need("DGS2"): S["DGS2"] = col("2 Yr")
            if need("DGS10") : S["DGS10"] = col("10 Yr")
            if need("T10Y2Y"): S["T10Y2Y"] = (col("10 Yr") - col("2 Yr")).dropna()
            if need("T10Y3M"): S["T10Y3M"] = (col("10 Yr") - col("3 Mo")).dropna()
        if need("T10Y3M"):
            irx = yahoo("^IRX")
            if len(irx) and len(S["DGS10"]):
                S["T10Y3M"] = (S["DGS10"] - irx).dropna()
    if need("CPIAUCSL") or need("CPILFESL") or need("UNRATE") or need("PAYEMS"):
        b = bls(["CUSR0000SA0", "CUSR0000SA0L1E", "LNS14000000", "CES0000000001"])
        for fid, bid in (("CPIAUCSL", "CUSR0000SA0"), ("CPILFESL", "CUSR0000SA0L1E"), ("UNRATE", "LNS14000000"), ("PAYEMS", "CES0000000001")):
            if need(fid) and len(b.get(bid, [])):
                S[fid] = b[bid]
    if need("SAHMREALTIME") and len(S["UNRATE"]) > 16:
        m3 = S["UNRATE"].rolling(3).mean()
        S["SAHMREALTIME"] = (m3 - m3.rolling(12).min().shift(1)).dropna()      # 3-month average against its low of the prior 12 months
    for k in IDS:
        S[k] = _save(k, S[k]) if len(S[k]) else _saved(k)
    return S


def ago(s: pd.Series, days: int):
    if s.empty:
        return None
    x = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
    return float(x.iloc[-1]) if len(x) else None


def hist(s: pd.Series, years: int = 5) -> list:
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    m = s.resample("ME").last().dropna()
    return [[d.date().isoformat()[:7], round(float(v), 3)] for d, v in m.items()]


def run() -> dict:
    S = load()
    got = [k for k, v in S.items() if len(v)]
    if len(got) < 6:
        res = {"generated": datetime.now(timezone.utc).isoformat(), "groups": [], "status": "The economic data services did not answer. It will retry on the next run."}
        write_json(SITE_DATA / "economy.json", res)
        return res
    items = []

    def add(group, key, name, s, fmt, read, status=None, why=""):
        if s is None or s.empty:
            return
        v = float(s.iloc[-1])
        items.append({"group": group, "key": key, "name": name, "value": v, "show": fmt(v), "date": s.index[-1].date().isoformat(),
                      "month_ago": ago(s, 30), "year_ago": ago(s, 365), "read": read(v, ago(s, 30), ago(s, 365)), "status": status(v) if status else None,
                      "why": why, "hist": hist(s)})

    pc = lambda v: f"{v:.2f}%"
    pc1 = lambda v: f"{v:.1f}%"
    d = lambda a, b: "n/a" if b is None else "unchanged" if abs(a - b) < 0.005 else f"{'up' if a > b else 'down'} {abs(a - b):.2f} points"

    add("Interest rates", "fed", "The Fed's interest rate", S["DFF"], pc,
        lambda v, m, y: f"The Federal Reserve's rate is {v:.2f}%, {d(v, y)} from a year ago. " + ("The Fed has been cutting, which usually helps stocks." if y and v < y - 0.2 else "The Fed has been raising, which usually weighs on stocks." if y and v > y + 0.2 else "The Fed has been holding steady."),
        why="The price of money. It feeds into every loan, mortgage and stock valuation.")
    add("Interest rates", "y10", "10-year Treasury yield", S["DGS10"], pc,
        lambda v, m, y: f"{v:.2f}%, {d(v, m)} in a month. " + ("Rising long rates make stocks, and growth stocks most of all, worth less today." if m and v > m + 0.15 else "Falling long rates take pressure off stock prices." if m and v < m - 0.15 else "Little changed."),
        why="What the government pays to borrow for ten years. The benchmark that stocks are priced against.")
    add("Interest rates", "y2", "2-year Treasury yield", S["DGS2"], pc, lambda v, m, y: f"{v:.2f}%, {d(v, m)} in a month. It tracks where traders think the Fed's rate is heading.",
        why="The market's guess at the Fed's next moves.")
    add("Interest rates", "curve", "Yield curve (10-year minus 2-year)", S["T10Y2Y"], lambda v: f"{v:+.2f} pts",
        lambda v, m, y: (f"Long rates are {v:.2f} points above short rates: the normal shape." if v > 0 else f"Short rates are {abs(v):.2f} points above long rates. That upside-down shape has come before every US recession since the 1970s, usually a year or more ahead."),
        lambda v: "ok" if v > 0.25 else "watch" if v > 0 else "warn", "When short-term rates are higher than long-term ones, lenders pull back and recessions tend to follow.")
    add("Interest rates", "mort", "30-year mortgage rate", S["MORTGAGE30US"], pc, lambda v, m, y: f"{v:.2f}%, {d(v, y)} from a year ago.", why="Drives housing, the most rate-sensitive part of the economy.")

    cpi = (S["CPIAUCSL"].pct_change(12) * 100).dropna()
    core = (S["CPILFESL"].pct_change(12) * 100).dropna()
    infl = lambda v: "ok" if v < 2.6 else "watch" if v < 3.6 else "warn"
    add("Inflation", "cpi", "Inflation (consumer prices)", cpi, pc1, lambda v, m, y: f"Prices are up {v:.1f}% from a year ago; a year earlier the figure was {y:.1f}%. The Fed aims for about 2%." if y is not None else f"Prices are up {v:.1f}% from a year ago.", infl,
        "High inflation keeps interest rates high. Falling inflation gives the Fed room to cut.")
    add("Inflation", "core", "Core inflation (without food and energy)", core, pc1, lambda v, m, y: f"{v:.1f}% from a year ago. This is the steadier measure the Fed watches for the trend.", infl, "Strips out the jumpy parts to show the underlying trend.")

    add("Jobs", "unemp", "Unemployment rate", S["UNRATE"], pc1, lambda v, m, y: f"{v:.1f}% of people looking for work cannot find it, {d(v, y)} from a year ago.",
        lambda v: None, "A rising rate means the economy is losing steam.")
    add("Jobs", "sahm", "Recession trigger (Sahm rule)", S["SAHMREALTIME"].clip(lower=0), lambda v: f"{v:.2f}",
        lambda v, m, y: f"Reading {v:.2f}. " + ("At 0.50 or above. This has marked the start of every US recession since 1970." if v >= 0.5 else "Below 0.50, the level that has marked the start of every US recession since 1970."),
        lambda v: "warn" if v >= 0.5 else "watch" if v >= 0.3 else "ok", "How far unemployment has risen from its low of the past year. A fast rise is the classic recession signal.")
    claims = S["ICSA"].rolling(4).mean().dropna()
    add("Jobs", "claims", "New jobless claims (4-week average)", claims, lambda v: f"{v / 1000:.0f}K",
        lambda v, m, y: f"About {v / 1000:.0f},000 people a week are filing for unemployment" + (f", {'up' if v > y else 'down'} {abs(v / y - 1):.0%} from a year ago." if y else "."),
        lambda v: None, "The fastest jobs number there is: weekly layoffs. A steady climb is an early warning.")
    jobs = S["PAYEMS"].diff().dropna()
    add("Jobs", "payrolls", "Jobs added last month", jobs, lambda v: f"{v:+,.0f}K",
        lambda v, m, y: f"Employers {'added' if v >= 0 else 'cut'} {abs(v):,.0f},000 jobs. The three-month average is {jobs.tail(3).mean():+,.0f},000.",
        lambda v: "ok" if jobs.tail(3).mean() > 75 else "watch" if jobs.tail(3).mean() > 0 else "warn", "The monthly jobs report, the single biggest scheduled market mover.")

    add("Growth", "gdp", "Economic growth (GDP)", S["A191RL1Q225SBEA"], pc1, lambda v, m, y: f"The economy grew at a {v:.1f}% yearly pace last quarter." if v >= 0 else f"The economy shrank at a {abs(v):.1f}% yearly pace last quarter.",
        lambda v: "ok" if v > 1 else "watch" if v > 0 else "warn", "Two shrinking quarters in a row is the rule of thumb for a recession.")
    add("Growth", "sent", "Consumer mood (Michigan survey)", S["UMCSENT"], lambda v: f"{v:.1f}", lambda v, m, y: f"{v:.1f}, {d(v, y)} from a year ago. Shoppers are most of the economy, so their mood matters.", why="How households feel about their finances and the economy.")
    add("Growth", "oil", "Oil (US crude)", S["DCOILWTICO"], lambda v: f"${v:.0f}", lambda v, m, y: f"${v:.0f} a barrel" + (f", {'up' if v > m else 'down'} {abs(v / m - 1):.0%} in a month." if m else "."), why="Expensive oil acts like a tax on everyone and pushes inflation up.")
    add("Growth", "usd", "US dollar (against other currencies)", S["DTWEXBGS"], lambda v: f"{v:.1f}", lambda v, m, y: f"Index {v:.1f}" + (f", {'up' if v > y else 'down'} {abs(v / y - 1):.1%} from a year ago. A strong dollar hurts US companies that sell abroad." if y else "."), why="Big US companies earn a lot overseas; a strong dollar shrinks those sales.")

    hy = S["BAMLH0A0HYM2"]
    add("Stress", "hy", "Junk bond spread", hy, lambda v: f"{v:.2f} pts",
        lambda v, m, y: f"Risky companies pay {v:.2f} points more than the government to borrow, {d(v, m)} in a month. " + ("Above 5 points means lenders are scared." if v >= 5 else "Under 5 points means lenders are calm."),
        lambda v: "ok" if v < 4 else "watch" if v < 5 else "warn", "When lenders get nervous they demand more from weak borrowers. This often moves before stocks do.")
    add("Stress", "nfci", "Financial conditions (Chicago Fed)", S["NFCI"], lambda v: f"{v:+.2f}",
        lambda v, m, y: f"{v:+.2f}. " + ("Below zero means money is easier to get than usual." if v < 0 else "Above zero means money is tighter than usual, which is when things break."),
        lambda v: "ok" if v < -0.2 else "watch" if v < 0 else "warn", "One number for how easy it is to borrow across the whole financial system.")
    vix, vx3 = S["VIXCLS"], S["VXVCLS"]
    if len(vix) and len(vx3):
        ratio = (vix / vx3).dropna()
        add("Stress", "vixterm", "Near-term fear against longer-term fear", ratio, lambda v: f"{v:.2f}",
            lambda v, m, y: f"The VIX is {float(vix.reindex(ratio.index).iloc[-1]):.1f} and the three-month VIX is {float(vx3.reindex(ratio.index).iloc[-1]):.1f}. " + ("Fear about right now is higher than fear about later. That only happens in a panic." if v > 1 else "Fear about later is higher than fear about now, the normal calm shape."),
            lambda v: "ok" if v < 0.92 else "watch" if v <= 1 else "warn", "Options traders normally pay more to insure further out. When that flips, they are scrambling for protection today.")

    # recession watch
    watch = []
    c3 = S["T10Y3M"]
    if len(c3):
        inv_days = int((c3.tail(378) < 0).sum())
        watch.append({"name": "Yield curve (10-year minus 3-month)", "on": bool(c3.iloc[-1] < 0 or inv_days > 60),
                      "text": f"Now {float(c3.iloc[-1]):+.2f} points. " + ("Upside down today." if c3.iloc[-1] < 0 else f"It was upside down on {inv_days} of the last 378 trading days; recessions have usually started after it turns back up." if inv_days > 60 else "Not upside down, and has not been lately.")})
    if len(S["SAHMREALTIME"]):
        v = max(0.0, float(S["SAHMREALTIME"].iloc[-1]))
        watch.append({"name": "Unemployment rising fast (Sahm rule)", "on": v >= 0.5, "text": f"Reading {v:.2f}; the trigger is 0.50."})
    if len(claims) and ago(claims, 365):
        ch = float(claims.iloc[-1] / ago(claims, 365) - 1)
        watch.append({"name": "Layoffs climbing", "on": ch > 0.2, "text": f"Jobless claims are {'up' if ch > 0 else 'down'} {abs(ch):.0%} from a year ago; a rise of more than 20% is the warning level."})
    if len(hy):
        watch.append({"name": "Lenders pulling back", "on": bool(hy.iloc[-1] >= 5), "text": f"Junk bond spread {float(hy.iloc[-1]):.2f} points; 5 or more is stress."})
    if len(S["NFCI"]):
        watch.append({"name": "Money getting tight", "on": bool(S["NFCI"].iloc[-1] > 0), "text": f"Financial conditions index {float(S['NFCI'].iloc[-1]):+.2f}; above zero is tighter than normal."})
    on = sum(w["on"] for w in watch)
    short = 5 - len(watch)
    watch_text = (f"{on} of {len(watch)} recession warnings are lit" + (f" ({short} more cannot be measured without a free FRED key). " if short > 0 else ". ") + ("That is a clear warning. Recessions are when the deepest stock declines happen." if on >= 3 else "Worth watching, not yet a clear signal." if on >= 1 else "None of the classic warnings is flashing."))

    # conditions today, and what followed the same conditions in the past
    cond, cond_text = [], ""
    try:
        close, vol = load_panel()
        _, _, fg, _ = build(close, vol)
        spy = close["SPY"]
        c = close[stock_cols(close)]
        fwd = spy.shift(-63) / spy - 1
        al = lambda s: s.reindex(spy.index.union(s.index)).ffill().reindex(spy.index)
        b200 = (c > c.rolling(200).mean()).sum(axis=1) / c.notna().sum(axis=1).replace(0, np.nan)
        tests = [("Trend", "S&P 500 above its 200-day average", "S&P 500 below its 200-day average", spy > spy.rolling(200).mean(), spy.rolling(200).mean().notna()),
                 ("Breadth", "More than half of stocks in an uptrend", "Fewer than half of stocks in an uptrend", b200 > 0.5, b200.notna()),
                 ("Mood", "Extreme fear (score under 25)", "Not in extreme fear", fg.reindex(spy.index) < 25, fg.reindex(spy.index).notna())]
        if len(hy):
            h = al(hy)
            tests.append(("Credit", "Junk bond spread above its one-year average (lenders nervous)", "Junk bond spread below its one-year average (lenders calm)", h > h.rolling(252).mean(), h.rolling(252).mean().notna()))
        if len(vix) and len(vx3):
            r = al((vix / vx3).dropna())
            tests.append(("Panic", "Near-term fear above longer-term fear", "Near-term fear below longer-term fear", r > 1, r.notna()))
        if len(c3):
            k = al(c3)
            tests.append(("Yield curve", "Upside down (10-year below 3-month)", "Normal shape", k < 0, k.notna()))
        base = fwd.dropna()
        for name, yes, no, mask, ok in tests:
            mask = mask.reindex(spy.index).fillna(False) & ok
            now_on = bool(mask.iloc[-1])
            same = (mask if now_on else (~mask & ok))
            x = fwd[same].dropna()
            other = fwd[(~mask & ok) if now_on else mask].dropna()
            if len(x) < 30:
                continue
            cond.append({"name": name, "state": yes if now_on else no, "avg": float(x.mean()), "up": float((x > 0).mean()), "days": int(len(x)),
                         "other": float(other.mean()) if len(other) >= 30 else None, "typical": float(base.mean()),
                         "verdict": "better" if x.mean() > base.mean() + 0.005 else "worse" if x.mean() < base.mean() - 0.005 else "same"})
        nb, nw = sum(q["verdict"] == "better" for q in cond), sum(q["verdict"] == "worse" for q in cond)
        cond_text = (f"Of {len(cond)} conditions in place today, {nb} have been followed by better-than-usual returns over the next three months and {nw} by worse. "
                     f"A typical three months for the S&P 500: {base.mean():+.1%}.")
    except Exception as e:
        log(f"conditions table skipped: {type(e).__name__}: {e}")

    order = ["Interest rates", "Inflation", "Jobs", "Growth", "Stress"]
    warn = [i for i in items if i["status"] == "warn"]
    head = ("At a warning level: " + " and ".join(i["name"].lower().split(" (")[0] for i in warn[:3]) + "." if warn else "No economic number is at a warning level.")
    res = {"generated": datetime.now(timezone.utc).isoformat(), "status": "ok", "headline": head,
           "groups": [{"name": g, "items": [i for i in items if i["group"] == g]} for g in order if any(i["group"] == g for i in items)],
           "watch": watch, "watch_on": on, "watch_text": watch_text, "conditions": cond, "conditions_text": cond_text,
           "conditions_note": "From about 9 years of daily data, one mostly rising stretch. Days overlap, so read these as leanings. The pattern in this period: scared, beaten-down markets were followed by better returns than calm ones.",
           "source": ", ".join(sorted(SOURCES)) or "saved copies", "missing": [k for k in IDS if k not in got]}
    write_json(SITE_DATA / "economy.json", res)
    log(f"economy: {len(items)} indicators, {on}/{len(watch)} recession warnings, {len(cond)} conditions; missing {res['missing'] or 'none'}")
    return res


if __name__ == "__main__":
    run()
