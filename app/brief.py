"""Daily brief: one page with what you need to know before the day starts, written as plain sentences.
Built from the other pages' data plus a fresh look at futures, rates, the dollar, oil, gold and bitcoin.
Nothing here is invented: a section with no data behind it is left out."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from .core import SITE_DATA, load_config, log, read_json, write_json

UA = {"User-Agent": "Mozilla/5.0 (kronos-oracle)"}
NY = ZoneInfo("America/New_York")
SNAP = [("ES=F", "S&P 500 futures", "pct"), ("NQ=F", "Nasdaq futures", "pct"), ("^VIX", "VIX (fear gauge)", "lvl"), ("^TNX", "10-year Treasury yield", "yld"),
        ("DX-Y.NYB", "US dollar", "pct"), ("CL=F", "Oil", "usd"), ("GC=F", "Gold", "usd"), ("BTC-USD", "Bitcoin", "usd")]


def snapshot() -> list[dict]:
    out = []
    for t, name, kind in SNAP:
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", params={"interval": "1d", "range": "5d"}, headers=UA, timeout=20)
            res = r.json()["chart"]["result"][0]
            meta = res["meta"]
            closes = [c for c in (res.get("indicators", {}).get("quote") or [{}])[0].get("close") or [] if c is not None]
            px = meta.get("regularMarketPrice")
            prev = closes[-2] if len(closes) >= 2 else meta.get("chartPreviousClose")
            if px and prev:
                out.append({"t": t, "name": name, "kind": kind, "price": float(px), "chg": float(px / prev - 1), "diff": float(px - prev)})
        except Exception:
            continue
    return out


def earnings(watch: list[str], days: int = 5) -> list[dict]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    today = datetime.now(NY).date()
    try:
        r = requests.get("https://finnhub.io/api/v1/calendar/earnings", params={"from": today.isoformat(), "to": (today + timedelta(days=days)).isoformat(), "token": key}, timeout=25)
        rows = r.json().get("earningsCalendar") or []
    except Exception:
        return []
    rows = [x for x in rows if x.get("symbol")]
    rows.sort(key=lambda x: (x["symbol"] not in watch, -(x.get("revenueEstimate") or 0)))
    out = [{"t": x["symbol"], "date": x.get("date"), "when": {"bmo": "before the open", "amc": "after the close"}.get(x.get("hour"), ""),
            "watch": x["symbol"] in watch} for x in rows[:14]]
    return sorted(out, key=lambda x: (x["date"] or "", not x["watch"]))


def pc(v, d=1):
    return f"{v * 100:+.{d}f}%".replace("-", "−") if v is not None else "n/a"


def age_days(iso) -> float:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(str(iso).replace("Z", "+00:00"))).total_seconds() / 86400
    except Exception:
        return 99.0


def run() -> dict:
    cfg = load_config()
    watch = [t.upper() for t in cfg.get("watchlist", [])]
    J = lambda f: read_json(SITE_DATA / f) or {}
    F, X, L, SC, Q, N, W, ST, A = (J(f) for f in ("feargreed.json", "spx.json", "latest.json", "scanner.json", "quotes.json", "livenews.json", "wsb.json", "strategies.json", "accuracy.json"))
    EC, RG, INS, ER, ECO, S2 = (J(f) for f in ("econ.json", "regime.json", "insiders.json", "earnings.json", "economy.json", "second.json"))
    two = {r["t"]: r for r in (S2.get("rows") or [])}
    now = datetime.now(NY)
    weekend = now.weekday() >= 5
    sections, tiles = [], []
    add = lambda key, title, go, lines: sections.append({"key": key, "title": title, "go": go, "lines": [x for x in lines if x]}) if any(lines) else None
    line = lambda t, tone=None, go=None: {"t": t, "tone": tone, "go": go}
    S = X.get("stats") or {}

    # before the open
    snap = snapshot()
    ls = []
    for x in snap:
        tone = "up" if x["chg"] > 0 else "dn"
        if x["kind"] == "lvl":
            ls.append(line(f"{x['name']} {x['price']:.1f}, {pc(x['chg'])} from the last close", "dn" if x["chg"] > 0 else "up"))
        elif x["kind"] == "yld":
            ls.append(line(f"{x['name']} {x['price']:.2f}%, {'up' if x['diff'] > 0 else 'down'} {abs(x['diff']):.2f} points"))
        elif x["kind"] == "usd":
            ls.append(line(f"{x['name']} ${x['price']:,.0f}, {pc(x['chg'])}", tone))
        else:
            ls.append(line(f"{x['name']} {pc(x['chg'], 2)}", tone))
    add("open", "Markets right now" if not weekend else "Where things stand (US stocks reopen Monday)", "live", ls)
    fut = next((x for x in snap if x["t"] == "ES=F"), None)

    # last session
    if S:
        ins = X.get("insights") or []
        add("spx", "The S&P 500", "spx", [line(f"Last close {S['last']:.2f}: {pc(S.get('d1'), 2)} on the day, {pc(S.get('d5'))} over five days, {pc(S.get('ytd'))} this year.", "up" if (S.get("d1") or 0) > 0 else "dn")]
            + [line(t) for t in ins[1:4]])
        tiles.append({"label": "S&P 500 last close", "value": pc(S.get("d1"), 2), "sub": f"{abs(S['from_high']):.1%} below its high" if S["from_high"] < -0.002 else "at a record", "tone": "up" if (S.get("d1") or 0) > 0 else "dn"})
    sec = X.get("sectors") or []
    if sec:
        by1 = sorted(sec, key=lambda x: -x["d1"])
        add("sectors", "Sectors", "spx", [line(f"Strongest last session: {by1[0]['name']} {pc(by1[0]['d1'])}, {by1[1]['name']} {pc(by1[1]['d1'])}.", "up"),
                                           line(f"Weakest: {by1[-1]['name']} {pc(by1[-1]['d1'])}, {by1[-2]['name']} {pc(by1[-2]['d1'])}.", "dn"),
                                           line(f"Over five days money moved toward {sec[0]['name']} and away from {sec[-1]['name']}.")])

    # mood
    if F.get("score") is not None:
        sc, wk = F["score"], F.get("week_ago")
        parts = F.get("parts") or []
        aft = next((r for r in (F.get("after") or {}).get("rows", []) if r["zone"] == F["zone"]), None)
        ls = [line(f"Fear and greed score: {sc:.0f} of 100, {F['zone'].lower()}. A week ago it was {wk:.0f}.", "dn" if sc < 45 else "up" if sc > 55 else None)]
        if parts:
            ls.append(line(f"Most fearful gauge: {parts[0]['name'].lower()} ({parts[0]['now']}). Least fearful: {parts[-1]['name'].lower()}."))
        if aft and aft["d21"]["avg"] is not None:
            ls.append(line(f"After past {F['zone'].lower()} days the S&P 500 was higher a month later {aft['d21']['up']:.0%} of the time, averaging {pc(aft['d21']['avg'])}."))
        if (F.get("crypto") or {}).get("now"):
            c = F["crypto"]["now"]
            ls.append(line(f"Bitcoin fear and greed (alternative.me): {c['value']}, {c['label'].lower()}."))
        add("mood", "Mood", "fear", ls)
        tiles.append({"label": "Fear and greed", "value": f"{sc:.0f}", "sub": F["zone"], "tone": "dn" if sc < 45 else "up" if sc > 55 else None})

    # economic calendar: today and the next few days, big ones only
    ev = [x for x in (EC.get("events") or []) if x.get("date") and 0 <= (datetime.fromisoformat(x["date"]).date() - now.date()).days <= 5]
    big = [x for x in ev if x.get("impact") == "High"] or ev
    add("econ", "Economic reports this week", "calendar",
        [line(f"{x['day']}{' ' + x['time'] if x.get('time') else ''}: {x['title']}" + (f" (expected {x['forecast']}, last {x['previous']})" if x.get("forecast") and x.get("previous") else "")
              + (f". {x['why']}" if x.get("why") else ""), "dn" if x.get("impact") == "High" and x["date"] == now.date().isoformat() else None) for x in big[:6]])
    if RG.get("now"):
        add("regime", "What kind of market this is", "spx", [line(RG["text"], "dn" if RG["now"] in ("Correction", "Bear market") else None), line(RG.get("note", ""))])

    if ECO.get("watch_text"):
        pick = {i["key"]: i for g in ECO.get("groups", []) for i in g["items"]}
        add("economy", "The economy", "economy", [line(ECO["watch_text"], "dn" if ECO.get("watch_on", 0) >= 2 else None)]
            + [line(pick[k]["name"] + ": " + pick[k]["read"], "dn" if pick[k].get("status") == "warn" else None) for k in ("fed", "cpi", "unemp", "hy") if k in pick]
            + ([line(ECO["conditions_text"])] if ECO.get("conditions_text") else []))

    # calendar
    typical = {r["ticker"]: r for r in (ER.get("rows") or [])}
    er = earnings(watch)
    ls = []
    for x in er:
        d = datetime.fromisoformat(x["date"]).strftime("%a %b %-d") if x.get("date") else ""
        ls.append(line(f"{x['t']} reports {d} {x['when']}".strip() + (" (on your watchlist)" if x["watch"] else "")
                       + (f". It usually moves about {typical[x['t']]['typical']:.0%} on the day." if x["t"] in typical else ""), "up" if x["watch"] else None, f"t/{x['t']}"))
    soon = [w for w in (L.get("watchlist") or []) if w.get("earnings") and 0 <= (datetime.fromisoformat(w["earnings"]).date() - now.date()).days <= 10 and w["ticker"] not in {x["t"] for x in er}]
    ls += [line(f"{w['ticker']} reports {datetime.fromisoformat(w['earnings']).strftime('%a %b %-d')} (on your watchlist)", "up", f"t/{w['ticker']}") for w in soon]
    add("calendar", "Earnings in the next few days", None, ls[:10])

    # your stocks
    rows = {r["ticker"]: r for r in (Q.get("rows") or [])}
    mine = sorted([rows[t] for t in watch if t in rows and rows[t].get("change_pct") is not None], key=lambda r: -r["change_pct"])
    ls = []
    if mine:
        fresh = age_days(Q.get("generated")) < 1.5
        up_n = sum(r["change_pct"] > 0 for r in mine)
        ls.append(line(f"{up_n} of your {len(mine)} stocks were up at the last price check{'' if fresh else ' (prices are from the last trading day)'}."))
        ls.append(line(f"Best: {mine[0]['ticker']} {pc(mine[0]['change_pct'] / 100)}, {mine[1]['ticker']} {pc(mine[1]['change_pct'] / 100)}.", "up", f"t/{mine[0]['ticker']}"))
        ls.append(line(f"Worst: {mine[-1]['ticker']} {pc(mine[-1]['change_pct'] / 100)}, {mine[-2]['ticker']} {pc(mine[-2]['change_pct'] / 100)}.", "dn", f"t/{mine[-1]['ticker']}"))
    fc = [w for w in (L.get("watchlist") or []) if w.get("direction")]
    if fc:
        ups = [w for w in fc if w["direction"] == "up"]
        top = max(fc, key=lambda w: w.get("median_return") or -9)
        ls.append(line(f"The model expects {len(ups)} of {len(fc)} to rise over the next 5 days. Highest forecast: {top['ticker']} {pc(top.get('median_return'))}.", None, "watch"))
    hot = [h for h in (N.get("hot") or []) if h["t"] in watch][:3]
    if hot and age_days(N.get("generated")) < 2:
        ls.append(line("In the news: " + ", ".join(f"{h['t']} ({h['n']} stories, {h['bull']} bullish, {h['bear']} bearish)" for h in hot) + ".", None, "livenews"))
    buyers = [r for r in (INS.get("rows") or []) if r.get("buys")]
    if buyers:
        ls.append(line("Insiders bought their own stock at " + ", ".join(r["ticker"] + (" (several at once)" if r["cluster"] else "") for r in buyers) + " in the last 90 days.", "up", "insiders"))
    add("watch", "Your stocks", "watch", ls)

    # ideas (up calls only, see the accuracy study)
    use = SC if SC.get("top_long") and age_days(SC.get("generated")) <= age_days(L.get("generated")) else L
    ideas = [a for a in (use.get("top_long") or []) if a.get("direction", "up") == "up"][:3]
    stale = age_days(use.get("generated"))
    ls = []
    for a in ideas:
        p = a.get("plan") or {}
        t2 = two.get(a["ticker"])
        ls.append(line(f"{a['ticker']} ({a.get('name') or a.get('sector') or ''}): model sees {pc(a.get('median_return'))} in 5 days, score {a.get('score')} of 100."
                       + ((" The second model agrees." if t2["t_dir"] == "up" else " The second model disagrees.") if t2 else "")
                       + (f" Stop {p['stop']:.2f}, target {p['target']:.2f}." if p.get("stop") and p.get("target") else ""), "up", f"t/{a['ticker']}"))
    if ideas and stale > 3:
        ls.insert(0, line(f"These ideas are {stale:.0f} days old. Check the price before acting on any of them.", "dn"))
    if ideas:
        hit = (A.get("log") or {}).get("hit")
        ls.append(line(f"Keep sizes small: calls have been right {hit:.0%} of the time so far, which is no better than always guessing up." if hit else "Keep sizes small: the track record is still short."))
        add("ideas", "Ideas worth a look", "scanner", ls)

    # crowd
    wt = [t for t in (W.get("tickers") or []) if t.get("ticker")][:4]
    if wt and age_days(W.get("generated")) < 3:
        ris = [t for t in (W.get("tickers") or [])[:25] if (t.get("change") or 0) > 0.5][:3]
        add("crowd", "What the crowd is watching", "wsb", [line("Most talked about on WallStreetBets: " + ", ".join(f"{t['ticker']} ({t['mentions']:,} mentions)" for t in wt) + "."),
                                                          line("Chatter rising fastest: " + ", ".join(f"{t['ticker']} ({t['change']:+.0%})" for t in ris) + ".") if ris else None])
    K = INS.get("clusters") or {}
    if K.get("headline"):
        add("clusters", "Insider buying across the market", "insiders", [line(K["headline"], "up"), line("In the S&P 500: " + ", ".join(K["sp500"][:8]) + ".") if K.get("sp500") else None])
    c = N.get("counts") or {}
    if c and age_days(N.get("generated")) < 2:
        add("news", "News flow", "livenews", [line(f"{c.get('bull', 0)} bullish stories against {c.get('bear', 0)} bearish in the latest batch.", "up" if c.get("bull", 0) > c.get("bear", 0) else "dn")])

    if S2.get("compared"):
        mine2 = [r for r in (S2.get("watch") or []) if r.get("agree")]
        add("second", "Where both models agree", "second",
            [line(f"Kronos and TimesFM agree on {S2['agree']} of {S2['compared']} S&P 500 stocks for the next 5 days."),
             line("Both say up on your list: " + ", ".join(r["t"] for r in mine2 if r["t_dir"] == "up") + ".", "up") if any(r["t_dir"] == "up" for r in mine2) else None,
             line("Strongest agreement across the index: " + ", ".join(r["t"] for r in (S2.get("both_up") or [])[:6]) + ".") if S2.get("both_up") else None])

    # paper strategies
    rows_s = [r for r in (ST.get("rows") or [])]
    if rows_s:
        ls = []
        for r in rows_s:
            if r["kind"] == "Benchmark":
                continue
            held = ", ".join(h["ticker"] for h in r["holdings"][:5]) + (f" and {len(r['holdings']) - 5} more" if len(r["holdings"]) > 5 else "")
            txt = f"{r['name']}: holds {held or 'cash'}"
            if r["buys"] or r["sells"]:
                txt += ". New today: " + ", ".join([f"buy {t}" for t in r["buys"][:4]] + [f"sell {t}" for t in r["sells"][:4]])
            ls.append(line(txt + ".", None, "league"))
        add("league", "Paper strategies", "league", ls)

    # trust
    lg = A.get("log") or {}
    if lg:
        add("trust", "How much to trust this", "report", [line(f"{lg['hit']:.0%} of {lg['n']:,} graded calls were right. Saying up every time would have scored {lg['always_up']:.0%}. Use the forecasts as one input, not the decision.")])

    zone = (F.get("zone") or "").lower()
    head = []
    if fut:
        head.append(f"Futures point {'higher' if fut['chg'] > 0 else 'lower'} ({pc(fut['chg'], 2)})" if not weekend else "Markets are closed today")
    if S:
        head.append(f"the S&P 500 is {abs(S['from_high']):.1%} below its high" if S["from_high"] < -0.002 else "the S&P 500 is at a record")
    if RG.get("now"):
        head.append(f"this is a {RG['now'].lower()}")
    if zone:
        head.append(f"the mood is {zone}")
    headline = (", ".join(head) + ".") if head else "No market data yet."
    headline = headline[0].upper() + headline[1:]
    summary = []
    if S:
        summary.append("The long-term trend is still up." if S["vs200"] >= 0.005 else "The index is sitting right on its 200-day average, the line most investors watch." if S["vs200"] > -0.005 else "The index is below its 200-day average, which is when to be most careful.")
    if F.get("score") is not None:
        summary.append("Fear this deep has usually been a better time to buy than to sell, but it can get worse first." if F["score"] < 25 else
                       "Greed this high has usually meant smaller gains ahead." if F["score"] > 75 else "Mood is not at an extreme, so it says little by itself.")
    wl_er = [x["t"] for x in er if x["watch"]] + [w["ticker"] for w in soon]
    if wl_er:
        summary.append("On your list, " + ", ".join(dict.fromkeys(wl_er)) + f" {'report' if len(set(wl_er)) > 1 else 'reports'} earnings soon.")
    res = {"generated": datetime.now(timezone.utc).isoformat(), "day": now.strftime("%A, %B %-d"), "weekend": weekend, "headline": headline,
           "summary": " ".join(summary), "tiles": tiles, "sections": sections, "data_date": S.get("date")}
    write_json(SITE_DATA / "brief.json", res)
    log(f"daily brief: {len(sections)} sections; {headline}")
    return res


if __name__ == "__main__":
    run()
