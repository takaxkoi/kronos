"""Earnings reactions for every S&P 500 stock: how each one has actually moved the day after reporting.

Report dates come from Yahoo (through yfinance) and are kept in data/earnings/dates.csv.gz. A few dozen of the
stalest tickers are refreshed on each run, so the whole index stays current without hammering the source.
The move is the first close that could react against the close before it: next day for an after-hours report."""
from __future__ import annotations

import glob
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import ROOT, SITE_DATA, STATE_DIR, load_config, log, read_json, write_json
from .feargreed import load_panel, stock_cols

STORE = ROOT / "data" / "earnings" / "dates.csv.gz"
COLS = ["ticker", "when", "eps_est", "eps", "surprise", "fetched"]


def store() -> pd.DataFrame:
    if STORE.exists():
        try:
            return pd.read_csv(STORE)
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def fetch(n: int) -> int:
    """Refresh report dates for the n tickers whose saved copy is oldest (never-fetched first)."""
    import yfinance as yf
    close, _ = load_panel()
    names = stock_cols(close)
    d = store()
    last = d.groupby("ticker")["fetched"].max().to_dict() if len(d) else {}
    todo = sorted(names, key=lambda t: last.get(t, ""))[:n]
    today, got, fails = datetime.now(timezone.utc).date().isoformat(), [], 0
    for t in todo:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=28)
            if ed is None or not len(ed):
                raise ValueError("empty")
            ed = ed.reset_index()
            got.append(pd.DataFrame({"ticker": t, "when": ed.iloc[:, 0].astype(str), "eps_est": pd.to_numeric(ed.get("EPS Estimate"), errors="coerce"),
                                     "eps": pd.to_numeric(ed.get("Reported EPS"), errors="coerce"), "surprise": pd.to_numeric(ed.get("Surprise(%)"), errors="coerce"), "fetched": today}))
            fails = 0
        except Exception:
            fails += 1
            if fails >= 12:                      # the source is refusing us; stop and keep what we have
                log("earnings dates: source stopped answering, keeping saved copies")
                break
        time.sleep(0.35)
    if got:
        new = pd.concat(got)
        d = pd.concat([d[~d["ticker"].isin(new["ticker"].unique())], new])
        STORE.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(STORE, index=False)
    log(f"earnings dates: refreshed {len(got)} of {len(todo)} tickers; {d['ticker'].nunique() if len(d) else 0} stocks on file")
    return len(got)


def dates() -> pd.DataFrame:
    """Saved dates for the whole index, plus the nightly vault copy for the watchlist (whichever is newer per ticker)."""
    d = store()
    files = sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "earnings.csv.gz")))
    if files:
        v = pd.read_csv(files[-1]).rename(columns={"Earnings Date": "when", "EPS Estimate": "eps_est", "Reported EPS": "eps", "Surprise(%)": "surprise"})
        v["fetched"] = files[-1].split("/")[-2]
        have = d.groupby("ticker")["fetched"].max().to_dict() if len(d) else {}
        keep = [t for t in v["ticker"].unique() if have.get(t, "") < v["fetched"].iloc[0]]
        d = pd.concat([d[~d["ticker"].isin(keep)], v[v["ticker"].isin(keep)][COLS]])
    if len(d):
        d["when"] = pd.to_datetime(d["when"], utc=True).dt.tz_convert("America/New_York")
    return d


def run() -> dict:
    e = dates()
    if not len(e):
        res = {"generated": datetime.now(timezone.utc).isoformat(), "rows": [], "status": "No earnings dates saved yet."}
        write_json(SITE_DATA / "earnings.json", res)
        return res
    close, _ = load_panel()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    watch = {t.upper() for t in load_config().get("watchlist", [])}
    meta = {r.get("ticker"): r for r in (read_json(STATE_DIR / "sp500.json") or {}).get("rows", []) if isinstance(r, dict)}
    rows = []
    for t, g in e.groupby("ticker"):
        if t not in close:
            continue
        c = close[t].dropna()
        past, nxt = [], None
        for _, r in g.sort_values("when").iterrows():
            day = pd.Timestamp(r["when"].date())
            if day >= today:
                nxt = nxt or r["when"]
                continue
            if pd.isna(r.get("eps")):
                continue
            after = r["when"].hour >= 16
            i = c.index.searchsorted(day, side="right" if after else "left")
            if i <= 0 or i >= len(c):
                continue
            sur = r.get("surprise")
            past.append([day.date().isoformat(), round(float(c.iloc[i] / c.iloc[i - 1] - 1), 4), None if pd.isna(sur) else int(sur > 0), None if pd.isna(sur) else round(float(sur), 1)])
        if len(past) < 4:
            continue
        mv = np.array([p[1] for p in past])
        beats = [p for p in past if p[2] == 1]
        misses = [p for p in past if p[2] == 0]
        fell = sum(p[1] < 0 for p in beats)
        m = meta.get(t) or {}
        row = {"ticker": t, "name": m.get("name"), "sector": m.get("sector"), "watch": t in watch, "n": len(past), "typical": float(np.median(np.abs(mv))), "avg": float(mv.mean()),
               "up": int((mv > 0).sum()), "best": float(mv.max()), "worst": float(mv.min()), "beats": len(beats), "misses": len(misses),
               "after_beat": float(np.mean([p[1] for p in beats])) if beats else None, "after_miss": float(np.mean([p[1] for p in misses])) if misses else None,
               "fell_on_beat": fell, "history": past[::-1][:12],
               "next": nxt.date().isoformat() if nxt is not None else None, "next_days": int((pd.Timestamp(nxt.date()) - today).days) if nxt is not None else None,
               "next_when": ("after the close" if nxt.hour >= 16 else "before the open") if nxt is not None else None}
        row["read"] = (f"Moves about {row['typical']:.1%} the day after reporting, up {row['up']} of {row['n']} times. Beat estimates {row['beats']} times"
                       + (f" and still fell after {fell} of those." if beats else "."))
        rows.append(row)
    rows.sort(key=lambda r: (r["next_days"] is None, r["next_days"] if r["next_days"] is not None else 999, r["ticker"]))
    mine = [r for r in rows if r["watch"]]
    soon_mine = [r for r in mine if r["next_days"] is not None and r["next_days"] <= 14]
    week = [r for r in rows if r["next_days"] is not None and r["next_days"] <= 7]
    allb = sum(r["beats"] for r in rows)
    res = {"generated": datetime.now(timezone.utc).isoformat(), "rows": rows, "stocks": len(rows),
           "headline": (", ".join(f"{r['ticker']} in {r['next_days']} days" for r in soon_mine) + " report next." if soon_mine else
                        f"{len(week)} S&P 500 companies report in the next week; none are on your watchlist." if week else "No watchlist stock reports in the next two weeks."),
           "market": {"typical": float(np.median([r["typical"] for r in rows])), "beat_fell": (sum(r["fell_on_beat"] for r in rows) / allb) if allb else None,
                      "up_share": float(sum(r["up"] for r in rows) / sum(r["n"] for r in rows)), "reports": int(sum(r["n"] for r in rows)), "week": len(week)},
           "note": f"Covers {len(rows)} S&P 500 stocks with at least four past reports on file. The move is the close after the report against the close before it. Dates from Yahoo Finance."}
    write_json(SITE_DATA / "earnings.json", res)
    log(f"earnings reactions: {len(rows)} stocks, {len(week)} reporting within a week")
    return res


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch(int(sys.argv[sys.argv.index("--fetch") + 1]))
    run()
