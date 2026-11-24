"""Economic calendar: the scheduled US releases that move markets (inflation, jobs, the Fed, growth).
Tries free sources in turn and keeps the last good copy if all of them fail, marked with its age."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from .core import SITE_DATA, log, read_json, write_json

NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36", "Accept": "application/json, text/plain, */*"}
WHY = [
    (("cpi", "consumer price"), "Inflation", "The main inflation number. Hotter than expected usually hurts stocks because it keeps interest rates high."),
    (("pce",), "Inflation", "The inflation measure the Fed itself watches most."),
    (("ppi", "producer price"), "Inflation", "Prices at the factory gate, an early read on inflation."),
    (("non-farm", "nonfarm", "payroll"), "Jobs", "The monthly jobs report, the biggest scheduled market mover."),
    (("unemployment rate",), "Jobs", "Share of people looking for work. A jump raises recession worries."),
    (("unemployment claims", "jobless claims"), "Jobs", "Weekly layoffs. A steady rise is an early warning."),
    (("jolts", "job openings"), "Jobs", "How many jobs are open. Shows how tight hiring is."),
    (("adp",), "Jobs", "A private preview of the jobs report."),
    (("fomc", "federal funds", "fed interest rate", "rate decision"), "The Fed", "The Federal Reserve on interest rates. Lower rates help stocks; a surprise either way moves everything."),
    (("powell", "fed chair"), "The Fed", "The Fed chair speaking. Markets trade on every hint about rates."),
    (("gdp",), "Growth", "How fast the economy is growing."),
    (("retail sales",), "Growth", "How much shoppers are spending, which is most of the economy."),
    (("ism", "pmi"), "Growth", "A survey of businesses. Above 50 means growing, below 50 shrinking."),
    (("consumer confidence", "consumer sentiment"), "Growth", "How households feel about the economy."),
    (("durable goods",), "Growth", "Orders for big-ticket items, a sign of business confidence."),
    (("home sales", "housing starts", "building permits"), "Housing", "The housing market, which is sensitive to interest rates."),
    (("crude oil inventories",), "Energy", "Weekly oil stockpiles, which move oil prices."),
    (("bond auction", "note auction"), "Rates", "Government borrowing. Weak demand pushes interest rates up."),
]


def explain(title: str) -> tuple[str, str]:
    t = title.lower()
    for keys, topic, why in WHY:
        if any(k in t for k in keys):
            return topic, why
    return "Other", ""


def _ff() -> list[dict]:
    out = []
    for name in ("thisweek", "nextweek"):
        try:
            r = requests.get(f"https://nfs.faireconomy.media/ff_calendar_{name}.json", headers=UA, timeout=25)
            if r.status_code != 200:
                continue
            for x in r.json():
                if x.get("country") != "USD" or x.get("impact") not in ("High", "Medium"):
                    continue
                when = datetime.fromisoformat(x["date"]).astimezone(NY)
                out.append({"when": when.isoformat(), "title": x.get("title", ""), "impact": x["impact"], "forecast": x.get("forecast") or "", "previous": x.get("previous") or ""})
        except Exception:
            continue
    return out


def _nasdaq() -> list[dict]:
    out, day = [], datetime.now(NY).date()
    for i in range(8):
        d = day + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        try:
            r = requests.get("https://api.nasdaq.com/api/calendar/economicevents", params={"date": d.isoformat()}, headers=UA, timeout=20)
            rows = ((r.json().get("data") or {}).get("rows")) or []
        except Exception:
            continue
        for x in rows:
            if x.get("country") != "United States":
                continue
            title = x.get("eventName", "")
            topic, _ = explain(title)
            if topic == "Other":
                continue
            hh = (x.get("gmt") or "").strip()
            try:
                t = datetime.strptime(f"{d.isoformat()} {hh}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).astimezone(NY)
            except ValueError:
                t = datetime(d.year, d.month, d.day, 8, 30, tzinfo=NY)
            out.append({"when": t.isoformat(), "title": title, "impact": "High" if topic in ("Inflation", "Jobs", "The Fed") else "Medium",
                        "forecast": (x.get("consensus") or "").replace("&nbsp;", "").strip(), "previous": (x.get("previous") or "").replace("&nbsp;", "").strip()})
    return out


def _finnhub() -> list[dict]:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    try:
        r = requests.get("https://finnhub.io/api/v1/calendar/economic", params={"token": key}, timeout=25)
        rows = r.json().get("economicCalendar") or []
    except Exception:
        return []
    out = []
    for x in rows:
        if x.get("country") != "US" or x.get("impact") not in ("high", "medium"):
            continue
        try:
            t = datetime.strptime(x["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone(NY)
        except Exception:
            continue
        out.append({"when": t.isoformat(), "title": x.get("event", ""), "impact": x["impact"].title(),
                    "forecast": "" if x.get("estimate") is None else str(x["estimate"]), "previous": "" if x.get("prev") is None else str(x["prev"])})
    return out


def _upcoming(events: list[dict]) -> list[dict]:
    today = datetime.now(NY).date()
    seen, rows = set(), []
    for e in sorted(events, key=lambda e: e["when"]):
        when = datetime.fromisoformat(e["when"])
        if when.date() < today or (when.date() - today).days > 10 or (e["title"], e["when"]) in seen:
            continue
        seen.add((e["title"], e["when"]))
        topic, why = explain(e["title"])
        rows.append({**e, "day": when.strftime("%a %b %-d"), "time": when.strftime("%-I:%M %p") if (when.hour, when.minute) != (0, 0) else "", "date": when.date().isoformat(), "topic": topic, "why": why})
    return rows


def run() -> dict:
    rows, used = [], None
    for name, fn in (("Forex Factory", _ff), ("Nasdaq", _nasdaq), ("Finnhub", _finnhub)):
        try:
            rows = _upcoming(fn())
        except Exception:
            rows = []
        log(f"economic calendar: {name} gave {len(rows)} upcoming US events")
        if rows:
            used = name
            break
    if not rows:
        old = read_json(SITE_DATA / "econ.json")
        if old and old.get("events"):
            old["status"] = "Could not refresh the calendar on the last try. Showing the last copy."
            write_json(SITE_DATA / "econ.json", old)
            return old
        res = {"generated": datetime.now(timezone.utc).isoformat(), "events": [], "source": None, "status": "No free calendar source had upcoming events. It will retry on the next run."}
        write_json(SITE_DATA / "econ.json", res)
        return res
    res = {"generated": datetime.now(timezone.utc).isoformat(), "events": rows, "source": used, "status": "ok"}
    write_json(SITE_DATA / "econ.json", res)
    return res


if __name__ == "__main__":
    run()
