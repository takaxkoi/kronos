"""Strategy league: well-known, published strategies tested on the app's own 10 years of prices,
then each one runs its own $100,000 paper account from the day it was added.

Every strategy is a table of target weights. Signals use only data up to the close; trades happen the next day.
Costs: 5 bps on every dollar traded. Backtest is split 60/40: the newer 40% is the honest half.
Known weakness, stated on the page: the stock list is today's S&P 500 members, which flatters stock-picking
strategies (losers that left the index are missing). The ETF strategies do not have that problem.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, write_json
from .feargreed import build, load_panel, stock_cols

FEE = 0.0005
START_CASH = 100_000


def month_end(sig: pd.DataFrame) -> pd.DataFrame:
    """Keep the signal from the last trading day of each month, hold it until the next one."""
    last = list(sig.index.to_series().groupby(sig.index.to_period("M")).max().values)
    end = pd.Timestamp(last[-1])
    if (end + pd.offsets.BDay(1)).month == end.month:   # the newest month is not finished yet
        last = last[:-1]
    return sig[sig.index.isin(last)].reindex(sig.index).ffill()


def top_n(score: pd.DataFrame, n: int) -> pd.DataFrame:
    r = score.rank(axis=1, ascending=False, method="first")
    return (r <= n).astype(float).div(n).where(score.notna().sum(axis=1) >= n, 0.0)


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def hold_state(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    held = np.zeros_like(entry, dtype=bool)
    for i in range(1, len(entry)):
        held[i] = (held[i - 1] & ~exit_[i]) | entry[i]
    return held


def make(close: pd.DataFrame, fg: pd.Series) -> list[dict]:
    cfg = load_config()
    spy = close["SPY"]
    up_trend = (spy > spy.rolling(200).mean())
    st = stock_cols(close)
    c = close[st]
    out = []

    def add(key, name, kind, rule, why, source, w, bias=False, cadence="monthly"):
        out.append({"key": key, "name": name, "kind": kind, "rule": rule, "why": why, "source": source, "bias": bias, "cadence": cadence,
                    "w": w.reindex(close.index).fillna(0.0)})

    sect = [s for s in (cfg.get("sectors") or {}) if s in close]
    mom6 = close[sect] / close[sect].shift(126) - 1
    add("sectors", "Sector rotation", "ETFs", "Each month-end, hold the 3 sector funds that rose most over six months. Hold cash while the S&P 500 is below its 200-day average.",
        "Sectors that have been leading tend to keep leading for a few months. The 200-day rule steps aside in bear markets.",
        "Faber, Relative Strength Strategies for Investing (2010)", month_end(top_n(mom6, 3).mul(up_trend.astype(float), axis=0)))

    five = [t for t in ["SPY", "QQQ", "IWM", "TLT", "GLD"] if t in close]
    tr = (close[five] > close[five].rolling(200).mean()).astype(float) / len(five)
    add("trend", "Trend following, five funds", "ETFs", "Split the money evenly across US large stocks, tech, small stocks, Treasury bonds and gold. Hold each one only while it is above its 200-day average; otherwise that slice sits in cash.",
        "Cuts losses in long declines. It gives up some gain in exchange for much smaller drops.",
        "Faber, A Quantitative Approach to Tactical Asset Allocation (2007)", month_end(tr))

    r12 = close[["SPY", "TLT"]] / close[["SPY", "TLT"]].shift(252) - 1
    dm = pd.DataFrame(0.0, index=close.index, columns=["SPY", "TLT"])
    dm.loc[r12["SPY"] > 0, "SPY"] = 1.0
    dm.loc[(r12["SPY"] <= 0) & (r12["TLT"] > 0), "TLT"] = 1.0
    add("dual", "Dual momentum", "ETFs", "Each month-end: if the S&P 500 is up over the last 12 months, hold it. If not, hold Treasury bonds if they are up. If neither, hold cash.",
        "One simple switch between stocks, bonds and cash, based on which has been working for a year.",
        "Antonacci, Dual Momentum Investing (2014)", month_end(dm.where(r12["SPY"].notna(), 0.0)))

    mom = c.shift(21) / c.shift(252) - 1
    add("stockmom", "Momentum stocks, top 20", "Stocks", "Each month-end, hold the 20 big stocks that rose most over the past year (skipping the latest month). Cash while the S&P 500 is below its 200-day average.",
        "Last year's winners have tended to keep winning for several more months. It is the most studied pattern in stock returns.",
        "Jegadeesh and Titman (1993)", month_end(top_n(mom, 20).mul(up_trend.astype(float), axis=0)), bias=True)

    volat = c.pct_change().rolling(252, min_periods=200).std()
    add("lowvol", "Calm stocks, 30 steadiest", "Stocks", "Each month-end, hold the 30 big stocks whose prices moved least over the past year.",
        "Steady stocks have earned about as much as wild ones with far smaller drops.",
        "Baker, Bradley and Wurgler (2011)", month_end(top_n(-volat, 30)), bias=True)

    r2 = rsi(c, 2)
    held = hold_state(((r2 < 10) & (c > c.rolling(200).mean())).fillna(False).values, (c > c.rolling(5).mean()).fillna(False).values)
    held = pd.DataFrame(held, index=c.index, columns=c.columns).astype(float)
    cnt = held.sum(axis=1).clip(lower=10)
    add("dip", "Buy sharp dips in uptrends", "Stocks", "Buy a stock that is above its 200-day average after a sharp two-day drop (2-day RSI under 10). Sell when it closes back above its 5-day average. At most 10% of the money per stock.",
        "Strong stocks that fall hard for a day or two have tended to bounce within the week.",
        "Connors and Alvarez, Short Term Trading Strategies That Work (2008)", held.div(cnt, axis=0), bias=True, cadence="daily")

    f = fg.reindex(close.index)
    hf = hold_state((f <= 25).fillna(False).values, (f >= 55).fillna(False).values)
    add("fear", "Buy when others are fearful", "ETFs", "Buy the S&P 500 when the app's fear and greed score drops to 25 or lower. Sell when it recovers to 55. Cash the rest of the time.",
        "Tests whether the fear and greed score is worth trading on by itself.",
        "This app's fear and greed index", pd.DataFrame({"SPY": hf.astype(float)}, index=close.index), cadence="daily")

    add("6040", "60/40 stocks and bonds", "Benchmark", "60% S&P 500, 40% long Treasury bonds, always.", "The classic do-nothing mix, here for comparison.", "Benchmark",
        pd.DataFrame({"SPY": 0.6, "TLT": 0.4}, index=close.index))
    add("spy", "Just hold the S&P 500", "Benchmark", "100% S&P 500, always.", "The bar every strategy has to clear.", "Benchmark",
        pd.DataFrame({"SPY": 1.0}, index=close.index))
    return out


def returns(w: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    live = w.shift(1).fillna(0.0)                      # yesterday's target is today's holding
    cost = live.diff().abs().sum(axis=1).fillna(0.0) * FEE
    return (live * rets[w.columns].fillna(0.0)).sum(axis=1) - cost


def stats(r: pd.Series) -> dict:
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    sd = r.std()
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "total": float(eq.iloc[-1] - 1), "dd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else None, "vol": float(sd * np.sqrt(252))}


def run() -> dict:
    close, vol = load_panel()
    _, _, fg, _ = build(close, vol)
    rets = close.pct_change(fill_method=None)
    strat = make(close, fg)
    start = close.index[260]                           # first day every signal has a year of history
    idx = close.index[close.index >= start]
    split = idx[int(len(idx) * 0.6)]
    st = read_json(STATE_DIR / "strategies.json") or {}
    st.setdefault("started", close.index[-1].date().isoformat())
    live_from = pd.Timestamp(st["started"])
    rows = []
    for s in strat:
        r = returns(s["w"], rets).loc[idx]
        w = s["w"]
        now = w.iloc[-1][w.iloc[-1] > 0].sort_values(ascending=False)
        prev = w.iloc[-2]
        px = close.iloc[-1]
        eq_all = (1 + r).cumprod()
        lv = r[r.index > live_from]
        live_eq = START_CASH * float((1 + lv).prod())
        monthly = eq_all.resample("ME").last()
        row = {k: s[k] for k in ("key", "name", "kind", "rule", "why", "source", "bias", "cadence")}
        row.update({"all": stats(r), "early": stats(r[r.index < split]), "late": stats(r[r.index >= split]),
                    "invested": float((w.loc[idx].sum(axis=1) > 0.01).mean()),
                    "trades_per_year": float(w.loc[idx].diff().abs().sum(axis=1).sum() / 2 / (len(idx) / 252)),
                    "curve": [round(float(x), 4) for x in monthly], "curve_dates": [d.date().isoformat() for d in monthly.index],
                    "live": {"equity": live_eq, "return": live_eq / START_CASH - 1, "days": int(len(lv)),
                             "curve": [{"date": d.date().isoformat(), "equity": round(START_CASH * float(v), 2)} for d, v in (1 + lv).cumprod().items()][-260:]},
                    "holdings": [{"ticker": t.replace("_", "^"), "weight": float(v), "dollars": float(v * live_eq), "price": float(px[t]),
                                  "shares": int(v * live_eq // px[t]) if px[t] > 0 else 0} for t, v in now.items()],
                    "cash": float(max(0.0, 1 - now.sum())),
                    "buys": sorted(t for t in now.index if prev.get(t, 0) <= 0), "sells": sorted(t for t in w.columns if prev.get(t, 0) > 0 and w.iloc[-1][t] <= 0)})
        rows.append(row)
    base = next(r for r in rows if r["key"] == "spy")
    for r in rows:
        e, l = r["early"], r["late"]
        r["beat_return"] = bool(l and l["cagr"] > base["late"]["cagr"] and e["cagr"] > base["early"]["cagr"])
        r["beat_risk"] = bool(l and (l["sharpe"] or 0) > (base["late"]["sharpe"] or 0) and (e["sharpe"] or 0) > (base["early"]["sharpe"] or 0))
        r["smaller_drop"] = bool(r["all"]["dd"] > base["all"]["dd"])
    real = [r for r in rows if r["kind"] != "Benchmark"]
    n_ret, n_risk = sum(r["beat_return"] for r in real), sum(r["beat_risk"] for r in real)
    clean = [r for r in real if not r["bias"]]
    best = max(clean, key=lambda r: r["all"].get("sharpe") or -9)
    verdict = (f"{n_ret} of {len(real)} strategies earned more than simply holding the S&P 500 in both halves of the test, and {n_risk} beat it once the size of the drops is counted. "
               f"The smoothest ride with no known flaw in the test was {best['name']}: {best['all']['cagr']:+.1%} a year with a worst drop of {best['all']['dd']:.0%}, "
               f"against {base['all']['cagr']:+.1%} a year and a worst drop of {base['all']['dd']:.0%} for the S&P 500. Less gain, much less pain.")
    res = {"generated": datetime.now(timezone.utc).isoformat(), "asof": close.index[-1].date().isoformat(), "start": start.date().isoformat(),
           "split": split.date().isoformat(), "live_from": st["started"], "cash": START_CASH, "fee_bps": FEE * 1e4, "best": best["key"],
           "verdict": verdict, "rows": rows,
           "caveat": "Stock strategies are tested on today's S&P 500 members, so companies that fell out of the index are missing and the results look better than they would have been. The fund strategies do not have that problem. Past results do not promise future ones; the live paper accounts are the real test."}
    write_json(SITE_DATA / "strategies.json", res)
    write_json(STATE_DIR / "strategies.json", st)
    log("strategies: " + "; ".join(f"{r['key']} {r['late'].get('cagr', 0):+.1%}/{r['all'].get('dd', 0):.0%}" for r in rows))
    return res


if __name__ == "__main__":
    run()
