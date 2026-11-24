"""Government data page: free public records that move markets, in one place (site/data/gov.json).

- Fed liquidity (FRED): the Fed's balance sheet minus the Treasury's cash account and reverse repo, a common
  read on how much money is sloshing around markets, set against the S&P 500.
- Hedge-fund positioning (CFTC Commitments of Traders, weekly): how speculators are positioned in S&P 500,
  Nasdaq and Treasury futures, bitcoin, gold and oil, and how extreme that is against the past three years.
- Big investors (SEC 13F, quarterly): what Berkshire, Bridgewater, Pershing Square and others bought and sold.
- Material events (SEC 8-K): CEO changes, deals, results, cyber incidents and more at your watchlist companies.
- Short selling (FINRA, daily): the share of each stock's trading that was short sales.
- Congress trades (Alpha Vantage): stock trades disclosed by members of Congress in your watchlist names.

Each source keeps its last good copy in state/gov/, so one failed download never blanks the page.
"""
from __future__ import annotations

import io
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, write_json

CACHE = STATE_DIR / "gov"
SOURCES: set[str] = set()


def _sec_ua() -> dict:
    c = os.environ.get("SEC_CONTACT") or (load_config().get("filings") or {}).get("contact") or "Trading-picker research takaxkoi@users.noreply.github.com"
    return {"User-Agent": str(c), "Accept-Encoding": "gzip, deflate"}


def _keep(name: str, fresh, ok) -> object:
    """Use fresh data when it looks right, else the last good copy."""
    p = CACHE / f"{name}.json"
    if ok(fresh):
        write_json(p, fresh)
        return fresh
    old = read_json(p, None)
    if old is not None:
        log(f"gov: {name} unavailable, showing the last good copy")
    return old


# ── Fed liquidity ──────────────────────────────────────────
def fred(sid: str, start: str) -> pd.Series:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return pd.Series(dtype=float)
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": sid, "api_key": key, "file_type": "json", "observation_start": start}, timeout=30)
        r.raise_for_status()
        s = pd.Series({pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in r.json()["observations"]}, dtype=float).dropna()
        if len(s):
            SOURCES.add("Federal Reserve Bank of St. Louis (FRED)")
        return s
    except Exception as e:
        log(f"FRED {sid}: {type(e).__name__}")
        return pd.Series(dtype=float)


def liquidity() -> dict | None:
    start = (date.today() - timedelta(days=365 * 4)).isoformat()
    walcl, tga, rrp, spx = fred("WALCL", start), fred("WTREGEN", start), fred("RRPONTSYD", start), fred("SP500", start)
    if walcl.empty or tga.empty:
        return None
    wk = pd.DataFrame({"fed": walcl / 1e6, "tga": tga / 1e6}).resample("W-WED").last().ffill()     # $ trillions
    wk["rrp"] = (rrp / 1e3).resample("W-WED").mean().reindex(wk.index).ffill().fillna(0)          # RRP is in $ billions
    wk["net"] = wk["fed"] - wk["tga"] - wk["rrp"]
    wk["spx"] = spx.resample("W-WED").last().reindex(wk.index).ffill()
    wk = wk.dropna(subset=["net"])
    last, m3 = wk.iloc[-1], wk.iloc[-14] if len(wk) > 14 else wk.iloc[0]
    corr = float(wk["net"].diff(13).corr(wk["spx"].pct_change(13))) if wk["spx"].notna().sum() > 30 else None
    chg = float(last["net"] - m3["net"])
    return {"asof": wk.index[-1].date().isoformat(), "net": round(float(last["net"]), 3), "fed": round(float(last["fed"]), 3),
            "tga": round(float(last["tga"]), 3), "rrp": round(float(last["rrp"]), 3), "chg_3m": round(chg, 3), "corr": None if corr is None else round(corr, 2),
            "read": (f"Net liquidity is ${last['net']:.2f} trillion, {'up' if chg >= 0 else 'down'} ${abs(chg)*1000:,.0f} billion in three months. "
                     + ("Rising liquidity has tended to support stocks." if chg > 0 else "Falling liquidity has tended to be a headwind for stocks.")),
            "series": {"dates": [d.date().isoformat() for d in wk.index], "net": [round(float(x), 3) for x in wk["net"]],
                       "spx": [None if pd.isna(x) else round(float(x), 1) for x in wk["spx"]]}}


# ── CFTC Commitments of Traders ────────────────────────────
COT = [  # (name, contract code, dataset, who)
    ("S&P 500 futures", "13874A", "gpe5-46if", "lev"), ("Nasdaq-100 futures", "209742", "gpe5-46if", "lev"),
    ("10-year Treasury futures", "043602", "gpe5-46if", "lev"), ("Bitcoin futures", "133741", "gpe5-46if", "lev"),
    ("Gold", "088691", "72hh-3qpy", "mm"), ("Crude oil (WTI)", "067651", "72hh-3qpy", "mm")]


def _col(row: dict, pat: str) -> float | None:
    for k, v in row.items():
        if re.fullmatch(pat, k):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def cot() -> list[dict] | None:
    out = []
    for name, code, ds, who in COT:
        try:
            r = requests.get(f"https://publicreporting.cftc.gov/resource/{ds}.json",
                             params={"cftc_contract_market_code": code, "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": 160}, timeout=40)
            r.raise_for_status()
            rows = r.json()
        except Exception as e:
            log(f"CFTC {name}: {type(e).__name__}")
            continue
        pts = []
        for x in rows:
            if who == "lev":
                lo, sh = _col(x, r"lev_money_positions_long(_all)?"), _col(x, r"lev_money_positions_short(_all)?")
            else:
                lo, sh = _col(x, r"m_money_positions_long_all"), _col(x, r"m_money_positions_short_all")
            oi = _col(x, r"open_interest_all")
            if lo is None or sh is None or not oi:
                continue
            pts.append((str(x.get("report_date_as_yyyy_mm_dd", ""))[:10], (lo - sh) / oi))
        if len(pts) < 20:
            continue
        pts = sorted(pts)
        vals = [v for _, v in pts]
        now, prev = vals[-1], vals[-2]
        pctl = sum(v <= now for v in vals) / len(vals)
        lbl = "hedge funds" if who == "lev" else "money managers"
        out.append({"name": name, "who": lbl, "date": pts[-1][0], "net_pct": round(now, 4), "chg": round(now - prev, 4), "pctile": round(pctl, 3),
                    "hist": [[d, round(v, 4)] for d, v in pts],
                    "read": (f"{lbl.capitalize()} are net {'long' if now > 0 else 'short'} {abs(now):.1%} of open interest, "
                             + (f"more bullish than in {pctl:.0%} of the past {len(vals)} weeks." if pctl >= .5 else f"more bearish than in {1 - pctl:.0%} of the past {len(vals)} weeks.")
                             + (" Positioning this extreme has often come before reversals." if pctl >= .9 or pctl <= .1 else ""))})
        time.sleep(0.3)
    if out:
        SOURCES.add("CFTC Commitments of Traders")
    return out


# ── SEC: 13F big investors and 8-K events ──────────────────
FUNDS = [("Berkshire Hathaway", 1067983, "Warren Buffett"), ("Bridgewater Associates", 1350694, "Ray Dalio's firm"),
         ("Pershing Square", 1336528, "Bill Ackman"), ("Scion Asset Management", 1649339, "Michael Burry"),
         ("Appaloosa", 1656456, "David Tepper"), ("Duquesne Family Office", 1536411, "Stanley Druckenmiller"),
         ("Tiger Global", 1167483, "Chase Coleman"), ("Third Point", 1040273, "Dan Loeb")]


def sec(url: str, as_json: bool = True):
    for i in range(3):
        try:
            r = requests.get(url, headers=_sec_ua(), timeout=40)
            time.sleep(0.15)
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            if r.ok:
                return r.json() if as_json else r.text
            return None
        except Exception:
            time.sleep(2)
    return None


def _holdings(cik: int, acc: str) -> pd.DataFrame | None:
    folder = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}"
    idx = sec(folder + "/index.json")
    items = ((idx or {}).get("directory") or {}).get("item") or []
    xmls = [i["name"] for i in items if i.get("name", "").lower().endswith(".xml") and "primary_doc" not in i["name"].lower()]
    best: list = []
    for x in xmls[:4]:                       # the information table is usually the only other XML, but not always
        txt = sec(f"{folder}/{x}", as_json=False)
        if not txt or "infotable" not in txt.lower():
            continue
        # filers spell the tags differently (namespaces, prefixes, capitals): strip and lower-case them all
        txt = re.sub(r'\s[\w:]*xmlns(:\w+)?="[^"]*"', "", txt)
        txt = re.sub(r"\s\w+:\w+=\"[^\"]*\"", "", txt)
        txt = re.sub(r"<(/?)(?:[\w.-]+:)?([\w.-]+)", lambda m: "<" + m.group(1) + m.group(2).lower(), txt)
        try:
            root = ET.fromstring(txt.encode())
        except ET.ParseError as e:
            log(f"13F {cik} {x}: XML error {e}")
            continue
        rows = []
        for it in root.iter("infotable"):
            g = lambda t: (it.findtext(t) or "").strip()
            if g("putcall"):
                continue
            try:
                rows.append({"name": g("nameofissuer"), "cusip": g("cusip").upper(), "value": float(g("value").replace(",", "") or 0),
                             "shares": float((it.findtext("shrsorprnamt/sshprnamt") or "0").replace(",", ""))})
            except ValueError:
                continue
        if len(rows) > len(best):
            best = rows
    if not best:
        log(f"13F {cik} {acc}: no information table among {xmls[:4]}")
        return None
    return pd.DataFrame(best).groupby("cusip").agg(name=("name", "first"), value=("value", "sum"), shares=("shares", "sum"))


def thirteen_f() -> list[dict] | None:
    out = []
    for name, cik, who in FUNDS:
        j = sec(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        r = ((j or {}).get("filings") or {}).get("recent") or {}
        f = [(d, a, p) for fm, d, a, p in zip(r.get("form", []), r.get("filingDate", []), r.get("accessionNumber", []), r.get("reportDate", [])) if fm == "13F-HR"]
        f = [x for i, x in enumerate(f) if x[2] not in {y[2] for y in f[:i]}]       # one filing per quarter (the newest)
        if len(f) < 2:
            log(f"13F {name}: {len(f)} quarterly filings found")
            continue
        new, old = _holdings(cik, f[0][1]), _holdings(cik, f[1][1])
        if new is None or old is None:
            log(f"13F {name}: could not read {'latest' if new is None else 'previous'} holdings")
            continue
        tot = new["value"].sum()
        # values were reported in $ thousands before 2023 and in dollars since; normalise to the newer report's scale
        scale = 1000 if old["value"].sum() * 50 < tot else 1
        old = old.assign(value=old["value"] * scale)
        m = new.join(old, how="outer", lsuffix="", rsuffix="_old").fillna({"value": 0, "shares": 0, "value_old": 0, "shares_old": 0})
        m["name"] = m["name"].fillna(m["name_old"])
        m["dsh"] = m["shares"] - m["shares_old"]
        m["pct"] = m["value"] / tot
        def pick(mask, by, asc, n=5):
            s = m[mask].sort_values(by, ascending=asc).head(n)
            return [{"name": str(x["name"]).title()[:40], "value": float(x["value"]), "old": float(x["value_old"]), "pct": round(float(x["pct"]), 4),
                     "chg_shares": round(float(x["dsh"] / x["shares_old"]), 3) if x["shares_old"] else None} for _, x in s.iterrows()]
        out.append({"fund": name, "who": who, "filed": f[0][0], "period": f[0][2], "prev_period": f[1][2], "total": float(tot), "positions": int((m["value"] > 0).sum()),
                    "top": pick(m["value"] > 0, "value", False),
                    "new": pick((m["shares_old"] == 0) & (m["shares"] > 0), "value", False),
                    "sold": pick((m["shares"] == 0) & (m["shares_old"] > 0), "value_old", False),
                    "added": pick((m["shares_old"] > 0) & (m["dsh"] > 0), "value", False, 4),
                    "trimmed": pick((m["shares"] > 0) & (m["dsh"] < 0), "value_old", False, 4),
                    "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR"})
    if out:
        SOURCES.add("SEC EDGAR")
    return out


ITEMS = {"1.01": "Signed a major agreement", "1.02": "Ended a major agreement", "1.03": "Bankruptcy", "1.05": "Cybersecurity incident",
         "2.01": "Completed an acquisition or sale", "2.02": "Quarterly results", "2.03": "Took on new debt", "2.05": "Restructuring or job cuts",
         "2.06": "Write-down", "3.01": "Listing problem", "3.02": "Sold new shares", "3.03": "Changed shareholder rights", "2.04": "Debt came due early", "5.08": "Set a shareholder-proposal deadline", "4.01": "Changed auditor", "4.02": "Past results can't be relied on",
         "5.01": "Change of control", "5.02": "Executive or board change", "5.03": "Changed bylaws", "5.07": "Shareholder vote results",
         "7.01": "Investor update", "8.01": "Other event", "9.01": "Exhibits"}
IMPORTANT = {"1.01", "1.02", "1.03", "1.05", "2.01", "2.03", "2.05", "2.06", "3.01", "4.01", "4.02", "5.01", "5.02"}


def events(watch: list[str]) -> list[dict] | None:
    tk = sec("https://www.sec.gov/files/company_tickers.json") or {}
    cik = {v["ticker"].upper(): int(v["cik_str"]) for v in tk.values()} if isinstance(tk, dict) else {}
    out, cutoff = [], (date.today() - timedelta(days=120)).isoformat()
    for t in watch:
        c = cik.get(t.replace("-", "."), cik.get(t))
        j = sec(f"https://data.sec.gov/submissions/CIK{c:010d}.json") if c else None
        r = ((j or {}).get("filings") or {}).get("recent") or {}
        for fm, d, a, doc, it in zip(r.get("form", []), r.get("filingDate", []), r.get("accessionNumber", []), r.get("primaryDocument", []), r.get("items", [])):
            if d < cutoff:
                break
            if fm not in ("8-K", "8-K/A"):
                continue
            codes = [x.strip() for x in str(it or "").split(",") if x.strip() and x.strip() != "9.01"]
            out.append({"ticker": t, "date": d, "codes": codes, "what": "; ".join(ITEMS.get(x, x) for x in codes) or "Filing",
                        "important": any(x in IMPORTANT for x in codes), "url": f"https://www.sec.gov/Archives/edgar/data/{c}/{a.replace('-', '')}/{doc}"})
    if out:
        SOURCES.add("SEC EDGAR")
    return sorted(out, key=lambda x: x["date"], reverse=True)[:80]


# ── FINRA short volume ─────────────────────────────────────
def short_volume(watch: list[str]) -> dict | None:
    frames, d, tries = [], date.today(), 0
    while len(frames) < 10 and tries < 20:
        d -= timedelta(days=1)
        tries += 1
        if d.weekday() >= 5:
            continue
        try:
            r = requests.get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d:%Y%m%d}.txt", timeout=30)
            if not r.ok or "|" not in r.text[:200]:
                continue
            x = pd.read_csv(io.StringIO(r.text), sep="|").dropna(subset=["Symbol"])
            x = x[x["TotalVolume"] > 0]
            x["date"] = d.isoformat()
            frames.append(x[["date", "Symbol", "ShortVolume", "TotalVolume"]])
        except Exception as e:
            log(f"FINRA {d}: {type(e).__name__}")
    if not frames:
        return None
    SOURCES.add("FINRA Reg SHO daily short sale volume")
    df = pd.concat(frames)
    g = df.groupby("Symbol").agg(short=("ShortVolume", "sum"), total=("TotalVolume", "sum"), days=("date", "nunique"))
    g["pct"] = g["short"] / g["total"]
    last = df[df["date"] == df["date"].max()].set_index("Symbol")
    g["today"] = (last["ShortVolume"] / last["TotalVolume"]).reindex(g.index)
    try:
        from .feargreed import load_panel, stock_cols
        close, _ = load_panel()
        big = set(stock_cols(close))
    except Exception:
        big = set()
    rows = lambda idx: [{"t": s, "pct10": round(float(g.at[s, "pct"]), 3), "today": None if pd.isna(g.at[s, "today"]) else round(float(g.at[s, "today"]), 3),
                         "vol": int(g.at[s, "total"] / max(g.at[s, "days"], 1))} for s in idx if s in g.index]
    sp = g[g.index.isin(big) & (g["days"] >= 5)] if big else g[g["total"] > 5e7]
    return {"days": sorted(df["date"].unique().tolist()), "watch": rows(watch), "most_shorted": rows(sp.sort_values("pct", ascending=False).head(15).index),
            "jump": rows(sp.assign(j=sp["today"] - sp["pct"]).sort_values("j", ascending=False).head(10).index),
            "market_avg": round(float(sp["short"].sum() / sp["total"].sum()), 3) if len(sp) else None}


# ── Congress trades (Alpha Vantage) ────────────────────────
def congress(watch: list[str]) -> dict | None:
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    store = read_json(CACHE / "congress_store.json", {}) or {}
    if key:
        order = sorted(watch, key=lambda t: (store.get(t) or {}).get("fetched", ""))[:6]   # oldest first, 6 a run
        for t in order:
            try:
                r = requests.get("https://www.alphavantage.co/query", params={"function": "CONGRESS_TRADES", "symbol": t, "apikey": key}, timeout=40)
                j = r.json() if r.ok else {}
                if j.get("trades") is not None:
                    store[t] = {"fetched": datetime.now(timezone.utc).isoformat(), "trades": j["trades"][:60]}
                else:
                    log(f"Congress {t}: {str(j)[:120]}")
            except Exception as e:
                log(f"Congress {t}: {type(e).__name__}")
            time.sleep(1.2)
        write_json(CACHE / "congress_store.json", store)
    if not store:
        return None
    SOURCES.add("US House and Senate disclosures via Alpha Vantage")
    cutoff = (date.today() - timedelta(days=180)).isoformat()
    trades, summ = [], []
    for t, v in store.items():
        b = s = 0
        for x in v.get("trades") or []:
            d = str(x.get("transaction_date") or "")[:10]
            if d < cutoff:
                continue
            side = str(x.get("transaction_type") or "").upper()
            lo, hi = float(x.get("amount_min") or 0), float(x.get("amount_max") or 0)
            b += side.startswith("BUY") or side.startswith("PURCHASE")
            s += side.startswith("SELL") or side.startswith("SALE")
            trades.append({"t": t, "who": x.get("politician_canonical") or x.get("politician"), "party": x.get("party"), "state": x.get("state"),
                           "chamber": str(x.get("chamber") or "").title(), "side": "Buy" if "BUY" in side or "PURCHASE" in side else "Sell" if "SELL" in side or "SALE" in side else side.title(),
                           "date": d, "filed": str(x.get("filed_date") or "")[:10], "lo": lo, "hi": hi})
        summ.append({"t": t, "buys": b, "sells": s, "updated": v.get("fetched")})
    trades.sort(key=lambda x: x["date"], reverse=True)
    return {"trades": trades[:120], "summary": sorted(summ, key=lambda x: -(x["buys"] + x["sells"]))}


def run() -> dict:
    cfg = load_config()
    watch = [t.upper() for t in cfg["watchlist"]]
    CACHE.mkdir(parents=True, exist_ok=True)
    res = {"generated": datetime.now(timezone.utc).isoformat()}
    for name, fn, ok in [("liquidity", liquidity, lambda x: bool(x)), ("cot", cot, lambda x: bool(x)), ("funds", thirteen_f, lambda x: bool(x)),
                         ("events", lambda: events(watch), lambda x: x is not None and len(x) > 0), ("shorts", lambda: short_volume(watch), lambda x: bool(x)),
                         ("congress", lambda: congress(watch), lambda x: bool(x))]:
        try:
            fresh = fn()
        except Exception as e:
            log(f"gov {name}: {type(e).__name__}: {e}")
            fresh = None
        res[name] = _keep(name, fresh, ok)
    res["sources"] = sorted(SOURCES)
    write_json(SITE_DATA / "gov.json", res)
    log("gov: " + ", ".join(f"{k} {'ok' if res.get(k) else 'missing'}" for k in ["liquidity", "cot", "funds", "events", "shorts", "congress"]))
    return res


if __name__ == "__main__":
    run()
