"""Why calls are wrong, and what would make more of them right.

Part 1 reads the graded forecast log and compares it with the dumbest possible forecaster ("always say up").
Part 2 uses 10 years of prices to measure how often a big stock rose over the next 5 days under simple,
checkable conditions. Those base rates are what a forecast has to beat.
Part 3 replays the log with a short list of rules (which calls to stop making) and shows the hit rate that is left."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import SITE_DATA, STATE_DIR, log, write_json
from .feargreed import build, load_panel, stock_cols


def _grp(g: pd.DataFrame, key, label) -> list[dict]:
    out = []
    for k, x in g.groupby(key, observed=True):
        out.append({"key": label(k), "n": int(len(x)), "hit": float(x["ok"].mean()), "always_up": float((x["actual_return"] > 0).mean())})
    return out


def log_study() -> dict | None:
    f = STATE_DIR / "predictions.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, low_memory=False)
    g = d[d["graded"].astype(str) == "True"].copy()
    if len(g) < 50:
        return None
    g["ok"] = g["correct"].astype(str).eq("True")
    g["day"] = g["asof"].astype(str).str[:10]
    daily = g[g["interval"] == "1d"]
    kinds = {"stock": "Stocks", "crypto": "Crypto", "forex": "Currencies", "commodity": "Commodities", "index": "Indexes"}
    spans = {"1d": "Daily bars, 5 days ahead", "1h": "Hourly bars", "1wk": "Weekly bars"}
    res = {"n": int(len(g)), "hit": float(g["ok"].mean()), "always_up": float((g["actual_return"] > 0).mean()),
           "days": int(daily["day"].nunique()), "first": g["day"].min(), "last": g["day"].max(),
           "by_direction": _grp(g, "direction", lambda k: "Said up" if k == "up" else "Said down"),
           "by_span": _grp(g, "interval", lambda k: spans.get(k, k)),
           "by_kind": _grp(g, "cls", lambda k: kinds.get(k, k)),
           "by_sure": _grp(g, pd.cut(g["agreement"], [0, 0.65, 0.8, 1.01]), lambda k: {0.65: "Model barely agreed with itself", 0.8: "Mostly agreed", 1.01: "Nearly all runs agreed"}[k.right])}
    # replay with rules
    keep = g[(g["interval"] != "1h") & ~((g["cls"] == "stock") & (g["direction"] == "down")) & (g["cls"] != "crypto")]
    res["rules"] = {"n": int(len(keep)), "hit": float(keep["ok"].mean()), "always_up": float((keep["actual_return"] > 0).mean()), "dropped": int(len(g) - len(keep))}
    return res


def base_rates() -> dict:
    close, vol = load_panel()
    _, _, fg, _ = build(close, vol)
    c = close[stock_cols(close)]
    fwd = c.shift(-5) / c - 1
    ma200, ma50 = c.rolling(200).mean(), c.rolling(50).mean()
    spy_up = (close["SPY"] > close["SPY"].rolling(200).mean())
    d = c.diff()
    r2 = 100 - 100 / (1 + d.clip(lower=0).ewm(alpha=0.5, adjust=False).mean() / (-d.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean().replace(0, np.nan))
    f = fg.reindex(c.index)
    ok = ma200.notna() & fwd.notna()
    rows = lambda s: pd.DataFrame(np.repeat(s.values[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
    tests = [
        ("Any stock, any day", ok, "The starting point: big stocks drift up."),
        ("Stock above its 200-day average", ok & (c > ma200), "Uptrend."),
        ("Stock below its 200-day average", ok & (c < ma200), "Downtrend."),
        ("Sharp two-day drop while in an uptrend", ok & (c > ma200) & (r2 < 10), "The dip-buying setup."),
        ("New 20-day high in an uptrend", ok & (c > ma200) & (c >= c.rolling(20).max()), "Breakout."),
        ("Market in extreme fear (score under 25)", ok & rows(f < 25), "From the fear and greed page."),
        ("Extreme fear, and the stock is still in an uptrend", ok & rows(f < 25) & (c > ma200), "Strong stock, scared market."),
        ("Market in extreme greed (score over 75)", ok & rows(f > 75), "Everyone is already in."),
        ("Stock and market both below their 200-day averages", ok & (c < ma200) & rows(~spy_up), "The one setup where calling down is close to even."),
    ]
    out = []
    for name, mask, note in tests:
        x = fwd.where(mask).stack()
        out.append({"name": name, "note": note, "n": int(len(x)), "up": float((x > 0).mean()), "avg": float(x.mean())})
    return {"rows": out, "first": c.index[200].date().isoformat(), "last": c.index[-6].date().isoformat(), "stocks": int(c.shape[1])}


def run() -> dict:
    L, B = log_study(), base_rates()
    base = B["rows"][0]["up"]
    find = []
    if L:
        dn = next((x for x in L["by_direction"] if x["key"] == "Said down"), None)
        hr = next((x for x in L["by_span"] if x["key"] == "Hourly bars"), None)
        find.append({"title": "The app is right less often than saying up every time",
                     "text": f"{L['hit']:.0%} of {L['n']:,} graded calls were right. Saying up on every one of them would have been right {L['always_up']:.0%} of the time. So far the forecasts have not shown skill beyond the market's normal upward drift."})
        if dn:
            find.append({"title": "Down calls are where most of the wrongs come from",
                         "text": f"Calls that said down were right {dn['hit']:.0%} of the time ({dn['n']:,} calls). Over 10 years a big stock rose over the next 5 days {base:.0%} of the time, so a down call starts behind."})
        if hr:
            find.append({"title": "Hourly calls are worse than a coin flip", "text": f"{hr['hit']:.0%} right on {hr['n']:,} hourly calls. Short windows are mostly noise."})
        sure = L["by_sure"]
        if len(sure) >= 2:
            find.append({"title": "The model's confidence does not mean much yet",
                         "text": f"When nearly all model runs agreed, calls were right {sure[-1]['hit']:.0%}; when it barely agreed, {sure[0]['hit']:.0%}. A higher score has not meant a better call."})
        find.append({"title": "The record is short", "text": f"All {L['n']:,} graded calls come from {L['days']} trading days ({L['first']} to {L['last']}). Calls made on the same day rise and fall together, so this is closer to {L['days']} tests than {L['n']:,}."})
    R = {r["name"]: r for r in B["rows"]}
    low = min(B["rows"], key=lambda r: r["up"])
    fear_row = R["Market in extreme fear (score under 25)"]
    fixes = [
        {"rule": "Grade against always-up, not against a coin flip", "why": "50% is the wrong bar for stocks, which rise more often than they fall. The report card now shows the always-up number beside the hit rate.", "status": "Done"},
        {"rule": "Stop making down calls on single stocks",
         "why": f"In 10 years no simple setup had big stocks falling more often than rising over 5 days. The weakest one ({low['name'].lower()}) still rose {low['up']:.0%} of the time. A down call fights the drift every time.", "status": "On since Nov 22"},
        {"rule": "Stop counting hourly calls", "why": "They have been right less than half the time. Short windows are mostly noise.", "status": "On since Nov 22"},
        {"rule": "Lean in when the market is scared, ease off when it is greedy",
         "why": f"Stocks rose {fear_row['up']:.0%} of the time over the next 5 days when the fear and greed score was under 25, against {base:.0%} on a normal day and {R['Market in extreme greed (score over 75)']['up']:.0%} when it was over 75.", "status": "In the daily brief"},
        {"rule": "Wait for more days before trusting the score", "why": f"The record covers {L['days'] if L else 0} trading days. About 60 is the minimum to tell skill from luck.", "status": "Waiting"},
    ]
    res = {"generated": datetime.now(timezone.utc).isoformat(), "log": L, "base": B, "findings": find, "fixes": fixes,
           "bottom_line": (f"With those rules the same log scores {L['rules']['hit']:.0%} on {L['rules']['n']:,} calls. Always-up on those calls scores {L['rules']['always_up']:.0%}, "
                           "so the rules remove bad calls but do not create skill. The honest way to more right calls is fewer, better-chosen calls, then waiting for proof." if L else "")}
    write_json(SITE_DATA / "accuracy.json", res)
    log(f"accuracy study: log {L['hit']:.1%} vs always-up {L['always_up']:.1%}" if L else "accuracy study: no graded calls")
    return res


if __name__ == "__main__":
    run()
