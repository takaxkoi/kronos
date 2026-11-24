"""Strategy Test: six stock-picking rules tested month by month over the app's 10 years of saved prices.

Each month-end, every rule buys its top 20 stocks using only what was known that day, holds them for one
month, and is compared with the equal-weight average of every stock in the test. This repeats the Tickerbot
test from Nov 2026 (22 months) on about 9 years of data, for free.

Data: prices from data/history (Yahoo, split- and dividend-adjusted) plus the data vault, and fundamentals
from SEC annual reports (data/filings/annual.csv.gz), each used only after the date it was filed.
Beta is worked out from the past year of daily prices on each date, so it is point-in-time too.

Known limits, also shown on the page:
- Only today's S&P 500 members, so companies that failed or were bought out are missing (flatters results).
- Market value in the past is estimated from today's market value scaled by the price change and the change
  in reported share count (splits detected and removed), so P/E is approximate.
- Banks and insurers often report no operating income, so the quality rules mostly skip them.
- No earnings-surprise history, so the "buy after an earnings beat" rule cannot be repeated here.
"""
from __future__ import annotations

import glob
from datetime import datetime, timezone

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

from .core import ROOT, SITE_DATA, log, write_json
from .feargreed import load_panel, stock_cols

TOP = 20
MIN_CAP = 10e9         # estimated market value on the date, in dollars
FEE = 0.0005           # per side, on the share of the portfolio that changes
SPLITS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50]

# Same rules on Tickerbot, Jan 2025 - Oct 2026, every US stock over $10B (including ones not in the S&P 500 today).
# Monthly excess return vs the average stock, in points. Used to measure how much testing on today's members flatters.
TICKERBOT = {"mom": 1.10, "qm": 0.22, "qv": -0.10, "lowb": -0.87}

RULES = {
    "mom": ("Momentum", "The 20 stocks that rose most over the past six months.",
            "Recent winners tend to keep winning for a while, but crash when the trend turns."),
    "qm": ("Quality momentum", "The 20 biggest six-month gainers that are above their 200-day average and earn over 15% on equity with operating margins over 15%.",
           "Momentum limited to profitable companies, to avoid the most speculative winners."),
    "qv": ("Quality value", "Of the companies earning over 15% on equity with operating margins over 15%, the 20 with the lowest P/E.",
           "Good businesses at cheap prices. Tends to hold up when rates rise."),
    "lowb": ("Low beta", "The 20 stocks that moved least with the market over the past year (lowest beta).",
             "Calm stocks fall less in sell-offs and lag in rallies."),
    "defq": ("Defensive quality", "Stocks with beta under 0.7, return on equity and operating margin over 15%, P/E between 0 and 25, and the price above its 200-day average; the 20 strongest over six months.",
             "The screen behind the Nov 2026 watchlist."),
    "switch": ("Regime switch", "Low beta when the S&P 500 is below its 200-day average or fell last month; quality momentum otherwise.",
               "Plays defense only when the market is weak."),
}


def month_ends(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = idx.to_series()
    return [pd.Timestamp(x) for x in s.groupby(idx.to_period("M")).max().values]


def load_fundamentals(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Annual filings with a split-safe share count, plus today's market value per ticker."""
    a = pd.read_csv(ROOT / "data" / "filings" / "annual.csv.gz", parse_dates=["end", "filed"])
    a = a[a["ticker"].isin(close.columns)].sort_values(["ticker", "filed"])
    mcap = pd.Series(dtype=float)
    for f in sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "fundamentals.csv.gz")))[::-1]:
        try:
            d = pd.read_csv(f, usecols=["ticker", "marketCap"]).dropna()
            mcap = mcap.combine_first(d.set_index("ticker")["marketCap"])
            if len(mcap) > 450:
                break
        except Exception:
            continue
    latest = a.groupby("ticker")["shares"].last()
    q = latest.reindex(a["ticker"]).values / a["shares"].values      # newest share count / this filing's
    adj = np.ones(len(a))
    for i, r in enumerate(q):
        if not np.isfinite(r) or r <= 0:
            adj[i] = np.nan
            continue
        for s, inv in ((r, False), (1 / r, True)):
            if s >= 1.7:
                best = min(SPLITS, key=lambda k: abs(s / k - 1))
                if abs(s / best - 1) < 0.15:
                    adj[i] = (1 / best) if inv else best
    a["shares_now_basis"] = a["shares"] * adj         # this filing's share count in today's split terms
    return a, mcap


def snapshot(a: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """The newest annual report filed on or before t (and no older than 18 months)."""
    s = a[(a["filed"] <= t) & (a["filed"] > t - pd.DateOffset(months=18))]
    return s.groupby("ticker").last()


def factors(close, spy_ret, rets, a, mcap, t) -> pd.DataFrame:
    i = close.index.get_loc(t)
    c = close.iloc[: i + 1]
    px = c.iloc[-1]
    f = pd.DataFrame(index=close.columns)
    f["ret6"] = px / c.iloc[-127] - 1 if i >= 126 else np.nan
    f["above200"] = px > c.iloc[-200:].mean()
    r = rets.iloc[max(0, i - 251): i + 1]
    m = spy_ret.iloc[max(0, i - 251): i + 1]
    ok = r.notna().sum() >= 200
    f["beta"] = (r.apply(lambda col: col.cov(m)) / m.var()).where(ok)
    snap = snapshot(a, t).reindex(close.columns)
    f["roe"] = (snap["net_income"] / snap["equity"]).where(snap["equity"] > 0)
    f["opm"] = snap["op_income"] / snap["revenue"]
    latest = a.groupby("ticker")["shares"].last().reindex(close.columns)
    drift = (snap["shares_now_basis"] / latest).clip(0.5, 2.0)        # buybacks / issuance since then
    now_px = close.ffill().iloc[-1]
    mc_t = mcap.reindex(close.columns) * (px / now_px) * drift.fillna(1.0)
    f["pe"] = (mc_t / snap["net_income"]).where(snap["net_income"] > 0)
    f["mcap"] = mc_t
    return f


def picks(f: pd.DataFrame, regime_weak: bool) -> dict[str, list[str]]:
    f = f[f["mcap"] >= MIN_CAP]                        # big companies on that date, as in the Tickerbot test
    good = (f["roe"] > 0.15) & (f["opm"] > 0.15)
    out = {
        "mom": f["ret6"].dropna().nlargest(TOP).index.tolist(),
        "qm": f.loc[good & f["above200"], "ret6"].dropna().nlargest(TOP).index.tolist(),
        "qv": f.loc[good & (f["pe"] > 0), "pe"].dropna().nsmallest(TOP).index.tolist(),
        "lowb": f.loc[f["beta"] > 0, "beta"].dropna().nsmallest(TOP).index.tolist(),
        "defq": f.loc[good & (f["beta"] > 0) & (f["beta"] < 0.7) & (f["pe"] > 0) & (f["pe"] < 25) & f["above200"], "ret6"].dropna().nlargest(TOP).index.tolist(),
    }
    out["switch"] = out["lowb"] if regime_weak else out["qm"]
    return out


def summarize(df: pd.DataFrame, key: str) -> dict:
    x = df[key] - df["base"]
    n = int(x.notna().sum())
    sd = float(x.std()) if n > 1 else np.nan
    down, up = df["base"] < 0, df["base"] >= 0
    eq = (1 + df[key].fillna(0)).cumprod()
    yrs = n / 12 if n else np.nan
    split = int(n * 0.6)
    return {"months": n, "avg": float(df[key].mean()), "excess": float(x.mean()), "t": float(x.mean() / (sd / np.sqrt(n))) if sd and sd > 0 else None,
            "beat": int((x > 0).sum()), "down_months": int(down.sum()), "down_excess": float(x[down].mean()), "down_beat": int((x[down] > 0).sum()),
            "up_excess": float(x[up].mean()), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if n else None,
            "dd": float((eq / eq.cummax() - 1).min()), "worst_month": float(x.min()), "best_month": float(x.max()),
            "early_excess": float(x.iloc[:split].mean()), "late_excess": float(x.iloc[split:].mean())}


def run() -> dict:
    close, _ = load_panel()
    stocks = stock_cols(close)
    if "SPY" not in close:
        raise SystemExit("SPY history missing")
    spy = close["SPY"]
    cl = close[stocks]
    rets = cl.pct_change(fill_method=None)
    spy_ret = spy.pct_change(fill_method=None)
    a, mcap = load_fundamentals(cl)
    ends = [t for t in month_ends(cl.index) if cl.index.get_loc(t) >= 260]
    rows, prev, dates = [], {}, []
    for t0, t1 in zip(ends[:-1], ends[1:]):
        weak = bool(spy[t0] < spy.loc[:t0].iloc[-200:].mean() or spy[t0] / spy.loc[:t0].iloc[-22] - 1 < 0)
        f = factors(cl, spy_ret, rets, a, mcap, t0)
        p = picks(f, weak)
        r = (cl.loc[t1] / cl.loc[t0] - 1)
        big = f.index[f["mcap"] >= MIN_CAP]
        row = {"date": t1.date().isoformat(), "from": t0.date().isoformat(), "weak": weak,
               "base": float(r.reindex(big).dropna().mean()), "base_n": int(r.reindex(big).notna().sum()), "spy": float(spy[t1] / spy[t0] - 1)}
        for k, lst in p.items():
            got = r.reindex(lst).dropna()
            turn = 1.0 if k not in prev else len(set(lst) - set(prev[k])) / max(len(lst), 1)
            row[k] = float(got.mean() - 2 * FEE * turn) if len(got) >= 5 else np.nan
            row[k + "_n"] = int(len(got))
        prev = p
        rows.append(row)
    df = pd.DataFrame(rows)
    stats = {k: summarize(df, k) for k in RULES}
    base_cagr = float((1 + df["base"]).prod() ** (12 / len(df)) - 1)
    spy_cagr = float((1 + df["spy"]).prod() ** (12 / len(df)) - 1)

    # today's picks
    t = cl.index[-1]
    weak_now = bool(spy[t] < spy.iloc[-200:].mean() or spy[t] / spy.iloc[-22] - 1 < 0)
    f_now = factors(cl, spy_ret, rets, a, mcap, t)
    p_now = picks(f_now, weak_now)
    now = {k: [{"ticker": x, "ret6": _f(f_now.at[x, "ret6"]), "beta": _f(f_now.at[x, "beta"]), "roe": _f(f_now.at[x, "roe"]),
                "opm": _f(f_now.at[x, "opm"]), "pe": _f(f_now.at[x, "pe"])} for x in lst] for k, lst in p_now.items()}

    ov = df[(df["date"].str[:7] >= "2025-01") & (df["date"].str[:7] <= "2026-10")]
    bias = [{"key": k, "name": RULES[k][0], "here": float((ov[k] - ov["base"]).mean() * 100), "tickerbot": v, "months": int(len(ov))}
            for k, v in TICKERBOT.items()]
    sig = [k for k in RULES if (stats[k]["t"] or 0) >= 2]
    best = max(RULES, key=lambda k: stats[k]["excess"])
    verdict = (f"Over {len(df)} months ({df['from'].iloc[0][:7]} to {df['date'].iloc[-1][:7]}), "
               + (f"{len(sig)} of {len(RULES)} rules beat the average stock by a margin unlikely to be luck: " + ", ".join(RULES[k][0] for k in sig) + ". " if sig
                  else "no rule beat the average stock by a margin you could tell apart from luck. ")
               + f"The best average was {RULES[best][0]} at {stats[best]['excess']*100:+.2f} points a month, beating it in {stats[best]['beat']} of {stats[best]['months']} months. "
               f"The average stock returned {base_cagr:+.1%} a year and the S&P 500 {spy_cagr:+.1%}.")
    mb = next(b for b in bias if b["key"] == "mom")
    if mb["months"] >= 12 and mb["here"] - mb["tickerbot"] > 1:
        verdict += (f" Treat the momentum result with suspicion: on the {mb['months']} months both tests cover, it shows {mb['here']:+.1f} points a month here "
                    f"but {mb['tickerbot']:+.1f} in the Tickerbot test, which included companies not in the S&P 500 today. Stocks join the index after rising, "
                    "so testing on today's members hands momentum winners it could not have known about.")
    res = {"generated": datetime.now(timezone.utc).isoformat(), "asof": t.date().isoformat(), "stocks": len(stocks),
           "months": len(df), "start": df["from"].iloc[0], "end": df["date"].iloc[-1], "top": TOP, "fee_bps": FEE * 1e4,
           "base_cagr": base_cagr, "spy_cagr": spy_cagr, "verdict": verdict, "regime_weak_now": weak_now, "bias_check": bias,
           "rules": [{"key": k, "name": v[0], "rule": v[1], "why": v[2], **stats[k],
                      "biased": any(b["key"] == k and b["months"] >= 12 and b["here"] - b["tickerbot"] > 1 for b in bias)} for k, v in RULES.items()],
           "monthly": {"dates": df["date"].tolist(), "base": _r(df["base"]), "spy": _r(df["spy"]), **{k: _r(df[k]) for k in RULES}},
           "today": now,
           "caveats": ["Only today's S&P 500 members are tested, so companies that failed or were bought out are missing. That flatters every rule.",
                       "Past market values are estimated from today's, scaled by price and reported share count, so P/E is approximate.",
                       "Banks and insurers often report no operating income, so the quality rules mostly leave them out.",
                       "There is no earnings-surprise history here, so the 'buy after an earnings beat' rule from the Tickerbot test is not repeated.",
                       f"Costs of {FEE*1e4:.0f} bps a side are charged on the share of each portfolio that changes."]}
    write_json(SITE_DATA / "factorlab.json", res)
    log("factorlab: " + "; ".join(f"{k} {stats[k]['excess']*100:+.2f}pt t={stats[k]['t'] or 0:.1f} beat {stats[k]['beat']}/{stats[k]['months']}" for k in RULES))
    return res


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _r(s: pd.Series):
    return [None if not np.isfinite(v) else round(float(v), 5) for v in s]


if __name__ == "__main__":
    run()
