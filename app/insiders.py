"""Insider trades for the watchlist: what executives and directors did with their own money.
Only open-market buys and sells count. Gifts, option exercises, share awards and tax withholding are left out,
because they say nothing about what the insider thinks of the price."""
from __future__ import annotations

import glob
import re
from html.parser import HTMLParser
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from .core import ROOT, SITE_DATA, STATE_DIR, log, read_json, write_json


class _Table(HTMLParser):
    """Rows of the first table with class 'tinytable' (OpenInsider's results table)."""
    def __init__(self):
        super().__init__(); self.on = False; self.rows = []; self.row = None; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == "table" and "tinytable" in (dict(attrs).get("class") or ""):
            self.on = True
        elif self.on and tag == "tr":
            self.row = []
        elif self.on and tag in ("td", "th") and self.row is not None:
            self.cell = []
    def handle_endtag(self, tag):
        if tag == "table" and self.on:
            self.on = False
        elif self.on and tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip()); self.cell = None
        elif self.on and tag == "tr" and self.row:
            self.rows.append(self.row); self.row = None
    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def clusters() -> dict | None:
    """Market-wide cluster buys: companies where two or more insiders bought on the open market. Source: OpenInsider (SEC Form 4)."""
    try:
        r = requests.get("http://openinsider.com/latest-cluster-buys", headers={"User-Agent": "Mozilla/5.0 (kronos-oracle research; one request a day)"}, timeout=40)
        r.raise_for_status()
        t = _Table(); t.feed(r.text)
    except Exception as e:
        log(f"cluster buys unavailable: {type(e).__name__}")
        return None
    num = lambda x: float(re.sub(r"[^0-9.\-]", "", x) or 0)
    sp = {str(x.get("ticker")) for x in (read_json(STATE_DIR / "sp500.json") or {}).get("rows", []) if isinstance(x, dict)}
    out = []
    for c in t.rows:
        if len(c) < 13 or not re.match(r"\d{4}-\d\d-\d\d", c[1]):
            continue
        if any(k in c[5] for k in ("Closed-End Funds", "Blank Checks")) or "Purchase" not in c[7]:
            continue
        out.append({"filed": c[1][:10], "traded": c[2][:10], "ticker": c[3], "company": c[4].title() if c[4].isupper() else c[4], "industry": c[5],
                    "insiders": int(num(c[6])), "price": num(c[8]), "shares": int(num(c[9])), "own_change": c[11], "value": num(c[12]), "sp500": c[3] in sp})
    if not out:
        return None
    week = [x for x in out if x["filed"] >= (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()]
    big = sorted(week or out, key=lambda x: -x["value"])[:3]
    return {"rows": out[:80], "week": len(week), "week_value": float(sum(x["value"] for x in week)),
            "headline": (f"{len(week)} companies had several insiders buying in the last week. Largest: " + ", ".join(f"{x['ticker']} (${x['value'] / 1e6:.1f} million, {x['insiders']} insiders)" for x in big) + "."),
            "sp500": [x["ticker"] for x in week if x["sp500"]], "credit": "OpenInsider, from SEC Form 4 filings", "url": "http://openinsider.com/latest-cluster-buys"}


def run() -> dict:
    days = sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "fh_insiders.csv.gz")))
    if not days:
        res = {"generated": datetime.now(timezone.utc).isoformat(), "rows": [], "status": "No insider data saved yet.", "clusters": clusters()}
        write_json(SITE_DATA / "insiders.json", res)
        return res
    d = pd.concat([pd.read_csv(f) for f in days[-10:]]).drop_duplicates(["id", "name", "transactionDate", "change", "transactionCode"])
    d = d[(d["transactionCode"].isin(["P", "S"])) & (d["isDerivative"].astype(str).str.lower() != "true")].copy()
    d["date"] = pd.to_datetime(d["transactionDate"], errors="coerce")
    d = d.dropna(subset=["date"])
    d["value"] = d["change"].abs() * d["transactionPrice"].fillna(0)
    asof = pd.Timestamp(days[-1].split("/")[-2])
    recent = d[d["date"] >= asof - timedelta(days=90)]
    rows = []
    for t, g in recent.groupby("ticker"):
        b, s = g[g["transactionCode"] == "P"], g[g["transactionCode"] == "S"]
        b30 = b[b["date"] >= asof - timedelta(days=30)]
        cluster = b30["name"].nunique() >= 2
        net = float(b["value"].sum() - s["value"].sum())
        if cluster:
            read = f"{b30['name'].nunique()} different insiders bought in the last 30 days. Several insiders buying together is the strongest insider signal there is."
        elif len(b):
            read = f"{b['name'].nunique()} insider{'s' if b['name'].nunique() != 1 else ''} bought with their own money. Buys are rare at large companies and worth noticing."
        elif len(s):
            read = f"{s['name'].nunique()} insider{'s' if s['name'].nunique() != 1 else ''} sold and none bought. Selling is common (taxes, diversifying) and says much less than buying does."
        else:
            read = "No open-market trades."
        trades = [{"date": r["date"].date().isoformat(), "name": str(r["name"]).title(), "buy": r["transactionCode"] == "P", "shares": int(abs(r["change"])),
                   "price": float(r["transactionPrice"] or 0), "value": float(r["value"])} for _, r in g.sort_values("date", ascending=False).head(15).iterrows()]
        rows.append({"ticker": t, "buys": int(len(b)), "buyers": int(b["name"].nunique()), "buy_value": float(b["value"].sum()),
                     "sells": int(len(s)), "sellers": int(s["name"].nunique()), "sell_value": float(s["value"].sum()), "net": net, "cluster": bool(cluster), "read": read, "trades": trades})
    rows.sort(key=lambda r: (-r["cluster"], -r["buy_value"], r["net"]))
    buyers = [r["ticker"] for r in rows if r["buys"]]
    head = (f"Insiders bought at {', '.join(buyers)} in the last 90 days." if buyers else "No insider on your watchlist bought shares on the open market in the last 90 days.")
    res = {"generated": datetime.now(timezone.utc).isoformat(), "asof": asof.date().isoformat(), "headline": head, "rows": rows,
           "total_sold": float(sum(r["sell_value"] for r in rows)), "total_bought": float(sum(r["buy_value"] for r in rows)),
           "note": "Covers the 15 stocks on your watchlist, from SEC filings collected through Finnhub. Insiders must report trades within two business days."}
    old = read_json(SITE_DATA / "insiders.json") or {}
    res["clusters"] = clusters() or old.get("clusters")
    write_json(SITE_DATA / "insiders.json", res)
    log(f"insiders: {len(rows)} stocks, buyers at {buyers or 'none'}")
    return res


if __name__ == "__main__":
    run()
