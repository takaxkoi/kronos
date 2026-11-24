"""S&P 500 page: everything the saved prices can say about the index right now, in plain numbers.
Chart, where it stands against its averages and highs, what is underneath (breadth, sectors, movers),
how past pullbacks of this size played out, and what this month usually does."""
from __future__ import annotations

import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import ROOT, SITE_DATA, STATE_DIR, load_config, log, read_json, write_json
from .feargreed import build, load_panel, stock_cols, zone

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def ohlc(t: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "history" / f"{t.replace('^', '_')}.csv.gz", usecols=["timestamps", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None).dt.normalize()
    df = df.drop(columns="timestamps")
    extra = []
    for f in sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "prices.csv.gz"))):
        try:
            d = pd.read_csv(f)
            extra.append(d[d["ticker"] == t][["date", "open", "high", "low", "close", "volume"]])
        except Exception:
            continue
    if extra:
        e = pd.concat(extra)
        e["date"] = pd.to_datetime(e["date"]).dt.tz_localize(None).dt.normalize()
        df = pd.concat([df, e])
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index().dropna(subset=["close"])


def pullbacks(s: pd.Series, depth: float = 0.05) -> list[dict]:
    """Every fall of `depth` or more from a high: how deep it got and how long until a new high."""
    out, peak, peak_d, low, low_d, inside = [], s.iloc[0], s.index[0], s.iloc[0], s.index[0], False
    for d, v in s.items():
        if v >= peak:
            if inside:
                out.append({"peak": peak_d, "low": low_d, "depth": low / peak - 1, "recovered": d, "days": int((d - peak_d).days)})
            peak, peak_d, low, low_d, inside = v, d, v, d, False
        elif v < low:
            low, low_d = v, d
            inside = inside or (low / peak - 1 <= -depth)
    if inside:
        out.append({"peak": peak_d, "low": low_d, "depth": low / peak - 1, "recovered": None, "days": None})
    return out


def run() -> dict:
    close, vol = load_panel()
    _, _, fg, inside = build(close, vol)
    # the index itself (6,900) when the vault has it; the SPY fund (690) until then
    src = "^GSPC" if (ROOT / "data" / "history" / "_GSPC.csv.gz").exists() else "SPY"
    o = ohlc(src)
    s = o["close"]
    st = stock_cols(close)
    c = close[st]
    ret = lambda n: float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n else None
    year = s[s.index.year == s.index[-1].year]
    prev_year_end = s[s.index.year < s.index[-1].year]
    ma50, ma200 = s.rolling(50).mean(), s.rolling(200).mean()
    d = s.diff()
    up14 = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn14 = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float(100 - 100 / (1 + up14.iloc[-1] / dn14.iloc[-1]))
    sign = np.sign(d.dropna().values)
    streak = 1
    while streak < len(sign) and sign[-1 - streak] == sign[-1]:
        streak += 1
    high = float(s.max())
    stats = {"last": float(s.iloc[-1]), "date": s.index[-1].date().isoformat(), "d1": ret(1), "d5": ret(5), "d21": ret(21), "d63": ret(63), "y1": ret(252),
             "ytd": float(s.iloc[-1] / prev_year_end.iloc[-1] - 1) if len(prev_year_end) else None,
             "high": high, "high_date": s.idxmax().date().isoformat(), "from_high": float(s.iloc[-1] / high - 1),
             "high52": float(s.tail(252).max()), "low52": float(s.tail(252).min()),
             "ma50": float(ma50.iloc[-1]), "ma200": float(ma200.iloc[-1]), "vs50": float(s.iloc[-1] / ma50.iloc[-1] - 1), "vs200": float(s.iloc[-1] / ma200.iloc[-1] - 1),
             "hi20": float(o["high"].tail(20).max()), "lo20": float(o["low"].tail(20).min()),
             "rsi": rsi, "vol20": float(s.pct_change().tail(20).std() * np.sqrt(252)), "vol_year": float(s.pct_change().tail(252).std() * np.sqrt(252)),
             "streak": int(streak), "streak_dir": "up" if sign[-1] > 0 else "down",
             "avg_volume_ratio": float(o["volume"].iloc[-1] / o["volume"].tail(21).iloc[:-1].mean()) if o["volume"].iloc[-1] else None}

    n = c.notna().sum(axis=1).replace(0, np.nan)
    b50, b200 = (c > c.rolling(50).mean()).sum(axis=1) / n, (c > c.rolling(200).mean()).sum(axis=1) / n
    hi = (c >= c.rolling(252, min_periods=200).max()).sum(axis=1)
    lo = (c <= c.rolling(252, min_periods=200).min()).sum(axis=1)
    tail = close.index[-260:]
    breadth = {"dates": [x.date().isoformat() for x in tail], "above50": [round(float(x), 3) for x in b50.reindex(tail)],
               "above200": [round(float(x), 3) for x in b200.reindex(tail)], "highs": [int(x) for x in hi.reindex(tail)], "lows": [int(x) for x in lo.reindex(tail)],
               "fear_greed": [round(float(x), 1) if x == x else None for x in fg.reindex(tail)], "spy": [round(float(x), 2) for x in o["close"].reindex(tail)]}

    sec_of = {r.get("ticker"): r.get("sector") for r in (read_json(STATE_DIR / "sp500.json") or {}).get("rows", []) if isinstance(r, dict)}
    names = {r.get("ticker"): r.get("name") for r in (read_json(STATE_DIR / "sp500.json") or {}).get("rows", []) if isinstance(r, dict)}
    mv = pd.DataFrame({"d1": c.iloc[-1] / c.iloc[-2] - 1, "d5": c.iloc[-1] / c.iloc[-6] - 1, "d21": c.iloc[-1] / c.iloc[-22] - 1, "price": c.iloc[-1]}).dropna()
    pick = lambda col, asc: [{"t": t, "name": names.get(t), "sector": sec_of.get(t), "v": float(r[col]), "price": float(r["price"])}
                             for t, r in mv.sort_values(col, ascending=asc).head(8).iterrows()]
    movers = {k: {"up": pick(k, False), "down": pick(k, True)} for k in ("d1", "d5", "d21")}

    sectors = []
    for etf, name in (load_config().get("sectors") or {}).items():
        if etf not in close:
            continue
        x = close[etf].dropna()
        py = x[x.index.year < x.index[-1].year]
        members = [t for t in st if sec_of.get(t) == name]
        sectors.append({"etf": etf, "name": name, "d1": float(x.iloc[-1] / x.iloc[-2] - 1), "d5": float(x.iloc[-1] / x.iloc[-6] - 1), "d21": float(x.iloc[-1] / x.iloc[-22] - 1),
                        "ytd": float(x.iloc[-1] / py.iloc[-1] - 1) if len(py) else None, "vs200": float(x.iloc[-1] / x.rolling(200).mean().iloc[-1] - 1),
                        "members": len(members), "above50": float((c[members].iloc[-1] > c[members].rolling(50).mean().iloc[-1]).mean()) if members else None})

    m = s.resample("ME").last().pct_change().dropna()
    m = m[m.index < s.index[-1].to_period("M").to_timestamp()]          # finished months only
    season = [{"month": MONTHS[k - 1], "avg": float(g.mean()), "up": float((g > 0).mean()), "years": int(len(g)), "best": float(g.max()), "worst": float(g.min())}
              for k, g in m.groupby(m.index.month)]
    now_m = s.index[-1].month
    before = s[s.index < s.index[-1].to_period("M").to_timestamp()]
    mtd = float(s.iloc[-1] / before.iloc[-1] - 1) if len(before) else None

    pb = pullbacks(s)
    done = [p for p in pb if p["recovered"] is not None]
    cur = stats["from_high"]
    deeper = [p for p in done if p["depth"] <= cur] if cur <= -0.05 else []
    pull = {"count": len(pb), "years": round((s.index[-1] - s.index[0]).days / 365.25, 1), "median_depth": float(np.median([p["depth"] for p in pb])) if pb else None,
            "median_days": float(np.median([p["days"] for p in done])) if done else None, "current": cur,
            "list": [{"peak": p["peak"].date().isoformat(), "low": p["low"].date().isoformat(), "depth": float(p["depth"]),
                      "recovered": p["recovered"].date().isoformat() if p["recovered"] is not None else None, "days": p["days"]} for p in pb][-12:],
            "stopped_here": int(len(done) - len(deeper)) if cur <= -0.05 else None, "went_deeper": len(deeper) if cur <= -0.05 else None}

    f = float(fg.iloc[-1])
    S, ins = stats, []
    nm = "The S&P 500" if src == "^GSPC" else "The S&P 500 fund (SPY)"
    ins.append(f"{nm} closed at {S['last']:,.2f}, {abs(S['from_high']):.1%} below its record of {S['high']:,.2f} set on {pd.Timestamp(S['high_date']).strftime('%b %-d')}."
               if S["from_high"] < -0.002 else f"{nm} closed at a record {S['last']:,.2f}.")
    at200 = abs(S["vs200"]) < 0.005
    ins.append(f"It is {abs(S['vs50']):.1%} {'above' if S['vs50'] >= 0 else 'below'} its 50-day average and "
               + (f"sitting right on its 200-day average ({S['ma200']:.2f}). Holding that line keeps the long-term trend up; losing it is when deep declines usually happen." if at200 else
                  f"{abs(S['vs200']):.1%} {'above' if S['vs200'] >= 0 else 'below'} its 200-day average. " + ("The long-term trend is still up." if S["vs200"] >= 0 else "Below the 200-day average is where most deep declines happen.")))
    ins.append(f"{inside['above_50d']:.0%} of big stocks are above their own 50-day average, {'down' if inside['above_50d'] < inside['above_50d_week_ago'] else 'up'} from {inside['above_50d_week_ago']:.0%} a week ago. "
               f"{inside['new_highs']} are at one-year highs and {inside['new_lows']} at one-year lows.")
    if cur <= -0.05 and pull["went_deeper"] is not None:
        ins.append(f"In the last {pull['years']:.0f} years there were {len(done)} finished pullbacks of 5% or more. {pull['stopped_here']} bottomed before falling this far; {pull['went_deeper']} went deeper. "
                   f"A typical one took about {pull['median_days']:.0f} days to get back to its high.")
    else:
        ins.append(f"In the last {pull['years']:.0f} years there were {len(pb)} pullbacks of 5% or more, typically {abs(pull['median_depth']):.0%} deep and about {pull['median_days']:.0f} days from high back to high.")
    sm = next((x for x in season if x["month"] == MONTHS[now_m - 1]), None)
    if sm:
        ins.append(f"{sm['month']} has been up in {round(sm['up'] * sm['years'])} of the last {sm['years']} years, averaging {sm['avg']:+.1%}. So far this month: {mtd:+.1%}." if mtd is not None else "")
    ins.append(f"Daily swings are running at {S['vol20']:.0%} a year, {'above' if S['vol20'] > S['vol_year'] else 'below'} the {S['vol_year']:.0%} of the past year. Fear and greed score: {f:.0f} of 100 ({zone(f).lower()}).")
    best, worst = max(sectors, key=lambda x: x["d5"]), min(sectors, key=lambda x: x["d5"])
    ins.append(f"Over five days {best['name']} led ({best['d5']:+.1%}) and {worst['name']} lagged ({worst['d5']:+.1%}).")

    res = {"generated": datetime.now(timezone.utc).isoformat(), "stats": stats, "source": src, "insights": [x for x in ins if x],
           "chart": [[d.date().isoformat(), round(float(r.open), 2), round(float(r.high), 2), round(float(r.low), 2), round(float(r.close), 2), float(r.volume or 0)] for d, r in o.tail(750).iterrows()],
           "breadth": breadth, "internals": inside, "sectors": sorted(sectors, key=lambda x: -x["d5"]), "movers": movers,
           "season": season, "month_to_date": mtd, "this_month": MONTHS[now_m - 1], "pullbacks": pull,
           "fear_greed": {"score": f, "zone": zone(f)}, "stocks": int(n.iloc[-1])}
    write_json(SITE_DATA / "spx.json", res)
    log(f"S&P 500 page: {stats['date']} {stats['last']:.2f}, {stats['from_high']:.1%} from high")
    return res


if __name__ == "__main__":
    run()
