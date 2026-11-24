"""Fear and greed, built from prices the app already saves, so every part can be checked.

Six gauges, each scored 0 (fear) to 100 (greed) by where today sits against its own last two years,
then averaged. Same recipe idea as CNN's index, minus the options put/call gauge (no free history).
The whole 10-year history is rebuilt on every run and tested: what did the S&P 500 do after each zone?
Also fetches the Bitcoin Fear & Greed Index from alternative.me (shown with credit, as their terms ask).
"""
from __future__ import annotations

import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .core import ROOT, SITE_DATA, load_config, log, write_json

SPECIAL = {"SPY", "QQQ", "DIA", "IWM", "HYG", "LQD", "TLT", "TIP", "GLD", "USO", "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS"}
ZONES = [(25, "Extreme fear"), (45, "Fear"), (55, "Neutral"), (75, "Greed"), (101, "Extreme greed")]
WINDOW = 504  # two years of trading days


def zone(v: float) -> str:
    for top, name in ZONES:
        if v < top:
            return name
    return "Extreme greed"


def load_panel(years: int = 10):
    """Close and volume for everything saved (stocks, ETFs, indexes), newest vault days merged in."""
    close, vol = {}, {}
    for f in sorted(glob.glob(str(ROOT / "data" / "history" / "*.csv.gz"))):
        t = f.split("/")[-1][:-7]
        if "-USD" in t or "=" in t:
            continue
        try:
            df = pd.read_csv(f, usecols=["timestamps", "close", "volume"])
            ix = pd.to_datetime(df["timestamps"]).dt.tz_localize(None).dt.normalize()
            keep = ~ix.duplicated(keep="last").values
            close[t] = pd.Series(df["close"].values[keep], index=ix[keep])
            vol[t] = pd.Series(df["volume"].values[keep], index=ix[keep])
        except Exception:
            continue
    close, vol = pd.DataFrame(close).sort_index(), pd.DataFrame(vol).sort_index()
    extra = []
    for f in sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "prices.csv.gz"))):
        try:
            extra.append(pd.read_csv(f, usecols=["ticker", "date", "close", "volume"]))
        except Exception:
            continue
    if extra:
        e = pd.concat(extra).drop_duplicates(["ticker", "date"], keep="last")
        e["ticker"] = e["ticker"].str.replace("^", "_", regex=False)
        e["date"] = pd.to_datetime(e["date"]).dt.tz_localize(None).dt.normalize()
        for name, frame in (("close", close), ("volume", vol)):
            p = e.pivot(index="date", columns="ticker", values=name)
            p = p[[c for c in p.columns if c in frame.columns]]
            if name == "close":
                close = p.combine_first(close)
            else:
                vol = p.combine_first(vol)
    close = close[close.index.dayofweek < 5]
    close = close[close["SPY"].notna()] if "SPY" in close else close
    close = close[close.index >= close.index[-1] - pd.DateOffset(years=years)]
    vol = vol.reindex(close.index)
    return close.ffill(limit=3), vol


def stock_cols(close: pd.DataFrame) -> list[str]:
    cfg = load_config()
    skip = SPECIAL | set(cfg.get("sectors", {})) | set(cfg.get("leveraged", []) or [])
    return [c for c in close.columns if "_" not in c and "." not in c and c not in skip and close[c].notna().sum() > 300]


def gauges(close: pd.DataFrame, vol: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Raw gauge values, oriented so that higher always means greedier."""
    st = stock_cols(close)
    c, v = close[st], vol[st]
    spy = close["SPY"]
    g = pd.DataFrame(index=close.index)
    g["momentum"] = spy / spy.rolling(125).mean() - 1
    hi = (c >= c.rolling(252, min_periods=200).max()).sum(axis=1)
    lo = (c <= c.rolling(252, min_periods=200).min()).sum(axis=1)
    n = c.notna().sum(axis=1).replace(0, np.nan)
    g["strength"] = ((hi - lo) / n).rolling(5).mean()
    dollars = (c * v).fillna(0)
    up = dollars.where(c.diff() > 0, 0).sum(axis=1)
    g["breadth"] = (up.rolling(20).sum() / dollars.sum(axis=1).rolling(20).sum().replace(0, np.nan))
    if "_VIX" in close:
        vix = close["_VIX"]
        g["volatility"] = -(vix / vix.rolling(50).mean() - 1)
    if "TLT" in close:
        g["safe_haven"] = spy.pct_change(20) - close["TLT"].pct_change(20)
    if "HYG" in close and "LQD" in close:
        g["junk_bonds"] = close["HYG"].pct_change(20) - close["LQD"].pct_change(20)
    above50 = (c > c.rolling(50).mean()).sum(axis=1) / n
    above200 = (c > c.rolling(200).mean()).sum(axis=1) / n
    last = c.iloc[-1], c.iloc[-2]
    inside = {"stocks": int(n.iloc[-1]), "new_highs": int(hi.iloc[-1]), "new_lows": int(lo.iloc[-1]),
              "above_50d": float(above50.iloc[-1]), "above_200d": float(above200.iloc[-1]),
              "above_50d_week_ago": float(above50.iloc[-6]),
              "advancers": int((last[0] > last[1]).sum()), "decliners": int((last[0] < last[1]).sum())}
    return g, inside


ABOUT = {
    "momentum": ("Market momentum", "The S&P 500 against its own 125-day average. Far above it means buyers are confident; below it means they are backing away.",
                 lambda x: f"S&P 500 is {abs(x):.1%} {'above' if x >= 0 else 'below'} its 125-day average"),
    "strength": ("New highs against new lows", "How many big stocks are at a one-year high compared with a one-year low. More highs is greed, more lows is fear.",
                 lambda x: f"{abs(x):.1%} more stocks at one-year {'highs than lows' if x >= 0 else 'lows than highs'} this week"),
    "breadth": ("Money in rising stocks", "The share of the last 20 days of trading, in dollars, that went into stocks that rose that day.",
                lambda x: f"{x:.0%} of the money traded went into rising stocks"),
    "volatility": ("Volatility (VIX)", "The VIX against its 50-day average. A VIX jumping above its average is fear; a VIX sinking below it is calm.",
                   lambda x: f"VIX is {abs(x):.0%} {'below' if x >= 0 else 'above'} its 50-day average"),
    "safe_haven": ("Stocks against safe bonds", "The S&P 500's last 20 days against long Treasury bonds. Money running to bonds is fear.",
                   lambda x: f"stocks {'beat' if x >= 0 else 'trailed'} Treasury bonds by {abs(x):.1%} over 20 days"),
    "junk_bonds": ("Appetite for risky bonds", "Junk bonds against safer company bonds over 20 days. Buying junk for the extra yield is greed.",
                   lambda x: f"junk bonds {'beat' if x >= 0 else 'trailed'} safer bonds by {abs(x):.1%} over 20 days"),
}


def build(close: pd.DataFrame, vol: pd.DataFrame):
    g, inside = gauges(close, vol)
    score = g.rolling(WINDOW, min_periods=252).rank(pct=True) * 100
    index = score.mean(axis=1, skipna=False).dropna()
    return g, score, index, inside


def after(index: pd.Series, spy: pd.Series) -> dict:
    """What the S&P 500 did after days in each zone. Days overlap, so treat these as a guide, not a guarantee."""
    out = {"horizons": [5, 21, 63], "rows": []}
    fwd = {h: spy.shift(-h) / spy - 1 for h in out["horizons"]}
    z = index.map(zone)
    for name in ["Extreme fear", "Fear", "Neutral", "Greed", "Extreme greed", "All days"]:
        days = index.index if name == "All days" else z.index[z == name]
        row = {"zone": name, "days": int(len(days)), "share": float(len(days) / len(index))}
        for h, f in fwd.items():
            r = f.reindex(days).dropna()
            row[f"d{h}"] = {"avg": float(r.mean()) if len(r) else None, "up": float((r > 0).mean()) if len(r) else None, "n": int(len(r))}
        # separate episodes: runs of days in the zone, so 40 fearful days in one month count once
        if name != "All days":
            inz = (z == name).astype(int)
            row["episodes"] = int(((inz.diff() == 1) | ((inz == 1) & (inz.shift().isna()))).sum())
        out["rows"].append(row)
    base = out["rows"][-1]
    ef, eg = out["rows"][0], out["rows"][4]
    bits = []
    for r, word in ((ef, "extreme fear"), (eg, "extreme greed")):
        a, b = r["d63"]["avg"], base["d63"]["avg"]
        if a is None or r["days"] < 20:
            bits.append(f"There were too few {word} days to say anything.")
        else:
            bits.append(f"After {word} days the S&P 500 averaged {a:+.1%} over the next three months, against {b:+.1%} for a typical day "
                        f"({r['days']} days in {r.get('episodes', 0)} separate stretches).")
    out["verdict"] = " ".join(bits) + (" The days overlap and all of it comes from one mostly rising decade, so treat it as a lean, not a rule. "
                                       "Published tests of CNN's index reach the same place: a real but small effect.")
    return out


def crypto() -> dict | None:
    try:
        r = requests.get("https://api.alternative.me/fng/", params={"limit": 31}, timeout=20)
        d = r.json()["data"]
        rows = [{"date": datetime.fromtimestamp(int(x["timestamp"]), timezone.utc).date().isoformat(),
                 "value": int(x["value"]), "label": x["value_classification"]} for x in d]
        return {"now": rows[0], "week_ago": rows[7] if len(rows) > 7 else None, "month_ago": rows[30] if len(rows) > 30 else None,
                "history": rows[::-1], "credit": "Crypto Fear & Greed Index by alternative.me", "url": "https://alternative.me/crypto/fear-and-greed-index/"}
    except Exception as e:
        log(f"crypto fear and greed unavailable: {type(e).__name__}")
        return None


def sectors(close: pd.DataFrame) -> list[dict]:
    out = []
    for etf, name in (load_config().get("sectors") or {}).items():
        if etf in close and close[etf].notna().sum() > 30:
            s = close[etf].dropna()
            out.append({"etf": etf, "name": name, "d1": float(s.iloc[-1] / s.iloc[-2] - 1), "d5": float(s.iloc[-1] / s.iloc[-6] - 1),
                        "d21": float(s.iloc[-1] / s.iloc[-22] - 1)})
    return sorted(out, key=lambda r: -r["d5"])


def run() -> dict:
    close, vol = load_panel()
    g, score, index, inside = build(close, vol)
    now = float(index.iloc[-1])
    parts = []
    for k in score.columns:
        title, why, say = ABOUT[k]
        parts.append({"key": k, "name": title, "score": float(score[k].iloc[-1]), "zone": zone(float(score[k].iloc[-1])),
                      "now": say(float(g[k].iloc[-1])), "why": why, "week_ago": float(score[k].iloc[-6])})
    parts.sort(key=lambda p: p["score"])
    hist = index.tail(260)
    res = {"generated": datetime.now(timezone.utc).isoformat(), "asof": index.index[-1].date().isoformat(),
           "score": now, "zone": zone(now), "yesterday": float(index.iloc[-2]), "week_ago": float(index.iloc[-6]),
           "month_ago": float(index.iloc[-22]), "year_ago": float(index.iloc[-253]) if len(index) > 253 else None,
           "parts": parts, "history": [{"date": d.date().isoformat(), "v": round(float(v), 1)} for d, v in hist.items()],
           "spy": [round(float(x), 2) for x in close["SPY"].reindex(hist.index)],
           "first": index.index[0].date().isoformat(), "after": after(index, close["SPY"]),
           "internals": inside, "sectors": sectors(close), "crypto": crypto(),
           "method": "Six gauges, each scored 0 to 100 against its own last two years, then averaged. Built from saved daily prices for "
                     f"{inside['stocks']} large US stocks plus SPY, VIX, TLT, HYG and LQD. CNN's version adds an options gauge that has no free history."}
    write_json(SITE_DATA / "feargreed.json", res)
    log(f"fear and greed: {now:.0f} ({zone(now)}) as of {res['asof']}")
    return res


if __name__ == "__main__":
    run()
