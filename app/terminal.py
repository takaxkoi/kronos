"""Stock terminal: one profile page per stock, built from everything the app has collected.

For every S&P 500 stock and everything on the watchlist it writes site/data/p/<TICKER>.json with:
snapshot numbers, a year of daily candles, who it trades like (industry peers, stocks that move with it and
against it, companies named beside it in the news), the bull case and bear case in plain words, news mood,
WallStreetBets chatter, investor-lens scores and five years of reported financials from SEC filings.
Also writes p/index.json so the dashboard can search every stock.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import ROOT, SITE_DATA, load_config, log, read_json, safe_name, write_json
from .lenses import LENSES, score_row


def ohlc(ticker: str, extra: pd.DataFrame | None) -> pd.DataFrame | None:
    f = ROOT / "data" / "history" / f"{ticker}.csv.gz"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f, usecols=["timestamps", "open", "high", "low", "close", "volume"]).rename(columns={"timestamps": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        if extra is not None and ticker in extra.index.get_level_values(0):
            e = extra.loc[ticker].reset_index()
            df = pd.concat([df, e[e["date"] > df["date"].max()]], ignore_index=True)
        return df.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")
    except Exception:
        return None


def vault_prices() -> pd.DataFrame | None:
    parts = []
    for f in sorted((ROOT / "data" / "daily").glob("*/prices.csv.gz")):
        try:
            parts.append(pd.read_csv(f, usecols=["ticker", "date", "open", "high", "low", "close", "volume"]))
        except Exception:
            continue
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates(["ticker", "date"], keep="last")
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return d.set_index(["ticker", "date"]).sort_index()


def num(v, nd=4):
    try:
        v = float(v)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def story(t, f, rets, sector_ret, news, wsb, lens, corr_mkt) -> dict:
    """Plain-English bull and bear points from the facts on file. No AI; every line is a measured fact."""
    bull, bear = [], []

    def add(ok, good, bad=None):
        if ok is True:
            bull.append(good)
        elif ok is False and bad:
            bear.append(bad)

    g = lambda k: num(getattr(f, k, None)) if f is not None else None
    px, m50, m200, hi = g("currentPrice"), g("fiftyDayAverage"), g("twoHundredDayAverage"), g("fiftyTwoWeekHigh")
    if px and m200:
        add(px > m200, f"Trading {abs(px / m200 - 1):.0%} above its 200-day average: the long-term trend is up.", f"Trading {abs(px / m200 - 1):.0%} below its 200-day average: the long-term trend is down.")
    if px and m50:
        add(px > m50, "Above its 50-day average: recent momentum is positive.", "Below its 50-day average: recent momentum has turned negative.")
    if px and hi and px < 0.75 * hi:
        bear.append(f"Down {1 - px / hi:.0%} from its 52-week high.")
    r1m = rets.get("1m")
    if r1m is not None and sector_ret is not None:
        d = r1m - sector_ret
        if d > 0.04:
            bull.append(f"Beat its sector by {d:.0%} over the past month, so buyers are picking this stock in particular.")
        elif d < -0.04:
            bear.append(f"Trailed its sector by {abs(d):.0%} over the past month, so the weakness is specific to this company.")
    rg, eg = g("revenueGrowth"), g("earningsGrowth")
    if rg is not None:
        add(rg > 0.10 or None if rg >= 0 else False, f"Revenue is growing {rg:.0%} a year.", f"Revenue is shrinking {abs(rg):.0%} a year.")
    if eg is not None:
        add(eg > 0.15 or None if eg >= 0 else False, f"Earnings are growing {eg:.0%} a year.", f"Earnings are falling {abs(eg):.0%} a year.")
    pm = g("profitMargins")
    if pm is not None:
        add(pm > 0.15 or None if pm >= 0 else False, f"Keeps {pm:.0%} of sales as profit.", "The company is losing money.")
    pe, fpe = g("trailingPE"), g("forwardPE")
    if fpe and fpe > 40:
        bear.append(f"Priced at {fpe:.0f} times next year's earnings: a lot of good news is already in the price.")
    elif fpe and 0 < fpe < 15:
        bull.append(f"Priced at {fpe:.0f} times next year's earnings, cheaper than most of the market.")
    tgt, rec, na = g("targetMeanPrice"), g("recommendationMean"), g("numberOfAnalystOpinions")
    if tgt and px and na and na >= 5:
        up = tgt / px - 1
        add(up > 0.10 or None if up >= -0.03 else False, f"Analysts' average target is {up:.0%} above the price ({na:.0f} analysts).", f"The price is already {abs(up):.0%} above analysts' average target.")
    if rec and na and na >= 5 and rec > 3:
        bear.append(f"Analysts lean Hold or Sell (rating {rec:.1f}, where 1 is Strong Buy).")
    sh = g("shortPercentOfFloat")
    if sh and sh > 0.08:
        bear.append(f"{sh:.0%} of the tradable shares are sold short: a large group is betting against it.")
    de = g("debtToEquity")
    if de and de > 200:
        bear.append(f"Heavy debt: {de / 100:.1f} times shareholder equity.")
    if news and news.get("n"):
        s = news["score"]
        if s > 0.10:
            bull.append(f"The news flow is positive: {news['positive']} bullish against {news['negative']} bearish headlines in the last few days.")
        elif s < -0.10:
            bear.append(f"The news flow is negative: {news['negative']} bearish against {news['positive']} bullish headlines in the last few days.")
    if wsb:
        ch = wsb.get("change")
        line = f"#{wsb['rank']} most-mentioned stock on WallStreetBets ({wsb['mentions']} mentions"
        line += f", {ch:+.0%} vs yesterday)." if ch is not None else ")."
        (bull if (ch or 0) > 0.25 else bear if (ch or 0) < -0.25 else bull).append(line + " Crowd attention cuts both ways.")
    score = len(bull) - len(bear)
    r1w = rets.get("1w")
    who = "Bulls are in control" if score >= 2 and (r1m or 0) >= 0 else "Bears are in control" if score <= -2 and (r1m or 0) <= 0 else "Bulls and bears are evenly matched"
    move = None
    if r1w is not None:
        move = f"{'Up' if r1w >= 0 else 'Down'} {abs(r1w):.1%} over the past week"
        if sector_ret is not None and rets.get("1w_sector") is not None:
            d = r1w - rets["1w_sector"]
            move += (", in line with its sector, so the move is mostly about the group." if abs(d) < 0.015 else
                     f", {abs(d):.1%} {'better' if d > 0 else 'worse'} than its sector, so the move is specific to this stock.")
        else:
            move += "."
    return {"verdict": who, "bull": bull[:7], "bear": bear[:7], "move": move}


def run() -> dict:
    cfg = load_config()
    days = sorted((ROOT / "data" / "daily").glob("*/fundamentals.csv.gz"))
    fund = pd.read_csv(days[-1]).drop_duplicates("ticker").set_index("ticker") if days else pd.DataFrame()
    watch = [t.upper() for t in cfg.get("watchlist", [])]
    tickers = sorted({t for t in list(fund.index) + watch + [t.upper() for t in cfg.get("indices", [])] if isinstance(t, str) and (ROOT / "data" / "history" / f"{t}.csv.gz").exists()})
    extra = vault_prices()
    bars = {t: d for t in tickers if (d := ohlc(t, extra)) is not None and len(d) > 60}
    raw = pd.DataFrame({t: d.set_index("date")["close"] for t, d in bars.items()}).sort_index()
    close = raw.ffill()
    ret = raw.pct_change(fill_method=None)
    corr = ret.tail(126).corr(min_periods=60)

    def chg(t, n):
        s = raw[t].dropna()
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1) if len(s) > n else None

    sector = fund["sector"].to_dict() if "sector" in fund else {}
    industry = fund["industry"].to_dict() if "industry" in fund else {}
    r1m = {t: chg(t, 21) for t in bars}
    r1w = {t: chg(t, 5) for t in bars}
    sec1m = pd.Series(r1m).groupby(pd.Series(sector)).median().to_dict()
    sec1w = pd.Series(r1w).groupby(pd.Series(sector)).median().to_dict()

    news = {r["ticker"]: r for r in (read_json(SITE_DATA / "news.json", {}).get("tickers") or [])}
    wsb = {r["ticker"]: r for r in (read_json(SITE_DATA / "wsb.json", {}).get("tickers") or [])}
    quotes = {r["ticker"]: r for r in (read_json(SITE_DATA / "quotes.json", {}).get("rows") or [])}
    fil = {}
    f = ROOT / "data" / "filings" / "annual.csv.gz"
    if f.exists():
        try:
            a = pd.read_csv(f).sort_values(["ticker", "end"])
            for t, grp in a.groupby("ticker"):
                fil[t] = [{"year": str(r.end)[:4], "revenue": num(r.revenue, 0), "net_income": num(r.net_income, 0),
                           "margin": num(r.net_income / r.revenue) if r.revenue else None, "cash_from_ops": num(getattr(r, "cfo", None), 0),
                           "filed": str(r.filed)[:10]} for r in grp.tail(6).itertuples(index=False)]
        except Exception as e:
            log(f"terminal: filings cache unreadable: {e}")
    lens_all = {t: score_row(r) for t, r in zip(fund.index, fund.itertuples(index=False))} if len(fund) else {}
    try:
        from .wsb import find_tickers, universe
        uni = universe()
    except Exception:
        find_tickers, uni = None, set()

    index = []
    for t in bars:
        fr = fund.loc[t] if t in fund.index else None
        rets = {"1d": chg(t, 1), "1w": r1w.get(t), "1m": r1m.get(t), "3m": chg(t, 63), "1y": chg(t, 252), "1w_sector": sec1w.get(sector.get(t))}
        ind = industry.get(t)
        peers = [p for p in bars if p != t and ind and industry.get(p) == ind]
        if len(peers) < 4:
            peers += [p for p in bars if p != t and p not in peers and sector.get(t) and sector.get(p) == sector.get(t)]
        mc = fund["marketCap"].to_dict() if "marketCap" in fund else {}
        peers = sorted(peers, key=lambda p: (industry.get(p) != ind, -(mc.get(p) or 0)))[:8]

        def brief(p, extra_kv=None):
            return {"ticker": p, "sector": sector.get(p), "industry": industry.get(p), "ret_1m": num(r1m.get(p)), "market_cap": num(mc.get(p), 0),
                    "pe": num(fund.at[p, "forwardPE"], 1) if p in fund.index and "forwardPE" in fund else None,
                    "lens": (lens_all.get(p) or {}).get("overall"), **(extra_kv or {})}
        c = corr[t].drop(t).dropna() if t in corr else pd.Series(dtype=float)
        c = c[[p for p in c.index if sector.get(p)]]  # companies only, not index funds
        with_ = [brief(p, {"corr": num(v, 2)}) for p, v in c.sort_values(ascending=False).head(6).items()]
        against = [brief(p, {"corr": num(v, 2)}) for p, v in c.sort_values().head(4).items()]
        n = news.get(t)
        named = []
        if n and find_tickers:
            cnt = {}
            for it in n.get("items", []):
                for o in find_tickers(it["title"], uni):
                    if o != t:
                        cnt[o] = cnt.get(o, 0) + 1
            named = [brief(p, {"headlines": k}) if p in bars else {"ticker": p, "headlines": k} for p, k in sorted(cnt.items(), key=lambda x: -x[1])[:6]]
        d = bars[t].tail(260)
        q = quotes.get(t)
        snap = {k: num(fr[k]) for k in ("marketCap", "trailingPE", "forwardPE", "priceToSalesTrailing12Months", "profitMargins", "operatingMargins", "returnOnEquity",
                                         "revenueGrowth", "earningsGrowth", "debtToEquity", "beta", "shortPercentOfFloat", "dividendYield", "recommendationMean",
                                         "numberOfAnalystOpinions", "targetMeanPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "heldPercentInstitutions") if fr is not None and k in fr.index} if fr is not None else {}
        prof = {"ticker": t, "generated": datetime.now(timezone.utc).isoformat(), "sector": sector.get(t), "industry": ind,
                "price": (q or {}).get("price") or num(close[t].dropna().iloc[-1]), "price_time": (q or {}).get("time") or close[t].dropna().index[-1].date().isoformat(),
                "live_change_pct": (q or {}).get("change_pct"), "returns": {k: num(v) for k, v in rets.items()}, "snapshot": snap,
                "chart": [[r.date.date().isoformat(), num(r.open, 2), num(r.high, 2), num(r.low, 2), num(r.close, 2), num(r.volume, 0)] for r in d.itertuples(index=False)],
                "peers": [brief(p) for p in peers], "moves_with": with_, "moves_against": against, "named_together": named,
                "story": story(t, fr, rets, sec1m.get(sector.get(t)), n, wsb.get(t), lens_all.get(t), None),
                "news": {k: n[k] for k in ("n", "score", "positive", "negative", "neutral", "items")} if n else None,
                "wsb": wsb.get(t), "lenses": lens_all.get(t), "financials": fil.get(t), "on_watchlist": t in watch}
        write_json(SITE_DATA / "p" / f"{safe_name(t)}.json", prof)
        index.append({"t": t, "s": sector.get(t), "i": ind, "p": prof["price"], "d": num(rets["1d"]), "m": num(rets["1m"]), "y": num(rets["1y"]),
                      "l": (lens_all.get(t) or {}).get("overall"), "v": prof["story"]["verdict"].split(" ")[0], "c": num(mc.get(t), 0), "w": t in watch})
    out = {"generated": datetime.now(timezone.utc).isoformat(), "asof": close.index[-1].date().isoformat(), "stocks": index, "lens_names": list(LENSES)}
    write_json(SITE_DATA / "p" / "index.json", out)
    log(f"terminal: wrote {len(index)} stock profiles, prices as of {out['asof']}")
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
