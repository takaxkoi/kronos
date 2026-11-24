"""Performance report (quantstats): what would have happened if you had followed the app's calls.

Builds a daily return stream from every graded forecast (long the "up" calls, short the "down" calls,
equal weight, held for the forecast window) and measures it against SPY: Sharpe, drawdown, win rate,
monthly returns. Also reports the paper-trading account. Runs in the Edge Lab job.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import ROOT, SITE_DATA, STATE_DIR, log, read_json, write_json


def daily_stream(g: pd.DataFrame) -> pd.Series:
    """Each call earns its signed return spread evenly over its holding days; overlapping calls are averaged."""
    g = g.copy()
    g["signed"] = g["actual_return"] * np.where(g["direction"] == "up", 1.0, -1.0)
    rows = []
    for (asof, h), grp in g.groupby(["asof", "horizon"]):
        days = pd.bdate_range(pd.Timestamp(asof) + pd.offsets.BDay(1), periods=int(h))
        r = grp["signed"].mean() / int(h)
        rows += [(d, r) for d in days]
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=["d", "r"]).groupby("d")["r"].mean().sort_index()
    return s


def spy_returns() -> pd.Series:
    try:
        df = pd.read_csv(ROOT / "data" / "history" / "SPY.csv.gz", usecols=["timestamps", "close"])
        s = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamps"]))
        for f in sorted((ROOT / "data" / "daily").glob("*/prices.csv.gz")):
            d = pd.read_csv(f)
            d = d[d["ticker"] == "SPY"]
            for _, r in d.iterrows():
                s.loc[pd.Timestamp(r["date"]).tz_localize(None).normalize()] = r["close"]
        return s[~s.index.duplicated(keep="last")].sort_index().pct_change().dropna()
    except Exception as e:
        log(f"SPY history unavailable: {e}")
        return pd.Series(dtype=float)


def measure(r: pd.Series, bench: pd.Series | None = None) -> dict:
    import quantstats as qs
    r = r.dropna()
    if len(r) < 2:
        return {"days": int(len(r))}

    def num(fn, *a, **k):
        try:
            v = float(fn(*a, **k))
            return v if np.isfinite(v) else None
        except Exception:
            return None

    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    out = {"days": int(len(r)), "start": r.index[0].date().isoformat(), "end": r.index[-1].date().isoformat(),
           "total_return": float(eq.iloc[-1] - 1), "sharpe": num(qs.stats.sharpe, r), "sortino": num(qs.stats.sortino, r),
           "max_drawdown": num(qs.stats.max_drawdown, r), "volatility": num(qs.stats.volatility, r),
           "win_rate": num(qs.stats.win_rate, r), "best_day": num(qs.stats.best, r), "worst_day": num(qs.stats.worst, r),
           "profit_factor": num(qs.stats.profit_factor, r),
           "curve": [[d.date().isoformat(), round(float(v) - 1, 5)] for d, v in eq.items()],
           "drawdown": [[d.date().isoformat(), round(float(v), 5)] for d, v in dd.items()]}
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    out["monthly"] = [{"month": f"{y}-{mo:02d}", "ret": float(v)} for (y, mo), v in m.items()]
    if bench is not None and len(bench):
        b = bench.reindex(r.index).fillna(0)
        out["spy_return"] = float((1 + b).prod() - 1)
        out["spy_curve"] = [[d.date().isoformat(), round(float(v) - 1, 5)] for d, v in (1 + b).cumprod().items()]
    return out


def run() -> dict:
    out = {"generated": datetime.now(timezone.utc).isoformat(), "streams": []}
    spy = spy_returns()
    f = STATE_DIR / "predictions.csv"
    if f.exists():
        p = pd.read_csv(f)
        g = p[(p["graded"].astype(str).str.lower() == "true") & (p["interval"] == "1d") & p["actual_return"].notna()
              & p["direction"].isin(["up", "down"])]
        cuts = [("Every stock call", "Long every up call and short every down call on stocks and ETFs.", g[g["cls"].isin(["stock", "index"])]),
                ("Up calls only", "Only the calls that said up (no shorting).", g[(g["cls"].isin(["stock", "index"])) & (g["direction"] == "up")]),
                ("High-score calls (75+)", "Only calls with a Kronos Score of 75 or higher.", g[(g["cls"].isin(["stock", "index"])) & (g["score"] >= 75)]),
                ("Calls with a valid trade plan", "Only calls where reward was at least twice the risk.", g[(g["cls"].isin(["stock", "index"])) & (g["plan_valid"].astype(str).str.lower() == "true")]),
                ("Crypto calls", "Daily crypto calls.", g[g["cls"] == "crypto"])]
        for name, why, sub in cuts:
            if len(sub) < 5:
                continue
            m = measure(daily_stream(sub), spy)
            hit = sub["correct"].astype(str).str.lower().eq("true").mean()
            out["streams"].append({"name": name, "why": why, "calls": int(len(sub)), "hit_rate": float(hit), **m})
            log(f"perf: {name}: {len(sub)} calls, hit {hit:.1%}, return {m.get('total_return')}")
    paper = read_json(SITE_DATA / "paper.json", {})
    eq = paper.get("equity") or []
    if len(eq) >= 3:
        s = pd.Series([e["equity"] for e in eq], index=pd.to_datetime([e["date"] for e in eq])).pct_change().dropna()
        out["paper"] = measure(s, spy)
    days = max([s.get("days", 0) for s in out["streams"]] or [0])
    out["note"] = (f"Only {days} trading days of graded calls so far. Sharpe and drawdown mean little until there are a few months of history."
                   if days < 60 else "Measured after the fact on graded calls; no trading costs included.")
    write_json(SITE_DATA / "perf.json", out)
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
