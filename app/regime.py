"""Market regime: one label for what kind of market this is, and what has worked in each kind.
Labels use only the S&P 500's distance from its high, its 200-day average and the VIX, so anyone can check them."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import SITE_DATA, log, write_json
from .feargreed import build, load_panel
from .strategies import make, returns

ORDER = ["Calm uptrend", "Nervous uptrend", "Pullback", "Correction", "Bear market"]
SAY = {
    "Calm uptrend": ("Within 5% of the high, above the 200-day average, VIX under 20.", "The most common state. Staying invested has been the right call; fighting it has not."),
    "Nervous uptrend": ("Within 5% of the high and above the 200-day average, but VIX at 20 or more.", "Prices are holding while investors pay up for protection. Swings are bigger in both directions."),
    "Pullback": ("5% to 10% below the high.", "Normal and frequent. Most end here, some turn into corrections."),
    "Correction": ("10% to 20% below the high.", "Painful but historically a better time to add than to sell, with big swings on the way."),
    "Bear market": ("20% or more below the high.", "Rare. Recoveries from here have been large but the bottom is only clear afterwards."),
}


def label(dd: float, above200: bool, vix: float) -> str:
    if dd <= -0.20:
        return "Bear market"
    if dd <= -0.10:
        return "Correction"
    if dd <= -0.05:
        return "Pullback"
    return "Calm uptrend" if (above200 and vix < 20) else "Nervous uptrend"


def run() -> dict:
    close, vol = load_panel()
    _, _, fg, _ = build(close, vol)
    spy = close["SPY"]
    dd = spy / spy.cummax() - 1
    above = spy > spy.rolling(200).mean()
    vix = close["_VIX"].ffill() if "_VIX" in close else pd.Series(18.0, index=close.index)
    reg = pd.Series([label(a, b, c) for a, b, c in zip(dd, above, vix)], index=close.index).iloc[200:]
    rets = close.pct_change(fill_method=None)
    strat = {s["name"]: returns(s["w"], rets) for s in make(close, fg) if s["key"] not in ("6040",)}
    rows = []
    for name in ORDER:
        days = reg.index[reg == name]
        if not len(days):
            continue
        row = {"name": name, "rule": SAY[name][0], "note": SAY[name][1], "days": int(len(days)), "share": float(len(days) / len(reg))}
        for h in (21, 63):
            f = (spy.shift(-h) / spy - 1).reindex(days).dropna()
            row[f"d{h}"] = {"avg": float(f.mean()) if len(f) else None, "up": float((f > 0).mean()) if len(f) else None}
        # what each strategy earned over the NEXT month, starting from days in this regime
        perf = sorted(({"name": k, "month": float(r.rolling(21).sum().shift(-21).reindex(days).mean())} for k, r in strat.items()), key=lambda x: -x["month"])
        row["worked"] = perf[:3]
        row["hold"] = next((p["month"] for p in perf if p["name"] == "Just hold the S&P 500"), None)
        row["swing"] = float(spy.pct_change().reindex(days).std() * np.sqrt(252))
        rows.append(row)
    now = reg.iloc[-1]
    since = reg.index[(reg != now)].max() if (reg != now).any() else reg.index[0]
    days_in = int((reg.index > since).sum())
    prev = reg[reg != now].iloc[-1] if (reg != now).any() else None
    cur = next(r for r in rows if r["name"] == now)
    text = (f"{now}. {SAY[now][0]} It has been this way for {days_in} trading day{'s' if days_in != 1 else ''}"
            + (f", after a stretch of {prev.lower()}" if prev else "") + ". "
            + (f"From days like this the S&P 500 was higher three months later {cur['d63']['up']:.0%} of the time, averaging {cur['d63']['avg']:+.1%}." if cur["d63"]["avg"] is not None else ""))
    res = {"generated": datetime.now(timezone.utc).isoformat(), "asof": reg.index[-1].date().isoformat(), "now": now, "days_in": days_in, "previous": prev,
           "text": text, "note": SAY[now][1], "drawdown": float(dd.iloc[-1]), "vix": float(vix.iloc[-1]), "above200": bool(above.iloc[-1]),
           "rows": rows, "first": reg.index[0].date().isoformat(),
           "history": [{"date": d.date().isoformat(), "r": ORDER.index(v)} for d, v in reg.tail(260).items()],
           "caveat": "Strategy results by regime come from the same 10-year backtest as the Strategy league, with the same limits."}
    write_json(SITE_DATA / "regime.json", res)
    log(f"regime: {now} for {days_in} days")
    return res


if __name__ == "__main__":
    run()
