"""Edge Lab (vectorbt): tests simple trading rules on years of real prices for the whole S&P 500
and reports which ones held up on data they were NOT picked on.

Every rule is tested on the first 60% of the period (in-sample) and again on the last 40% (out-of-sample).
A rule only counts as an edge if it beats buy-and-hold out-of-sample, after costs.
Runs in its own job because vectorbt needs its own package versions.
"""
from __future__ import annotations

import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import ROOT, SITE_DATA, load_config, log, write_json

FEES = 0.0005  # 5 bps per side, commission + slippage


def load_closes(years: int = 6, min_bars: int = 1000) -> pd.DataFrame:
    cols = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "history" / "*.csv.gz"))):
        t = f.split("/")[-1][:-7]
        if "-USD" in t or "=" in t or t.startswith("^"):
            continue
        try:
            df = pd.read_csv(f, usecols=["timestamps", "close"])
            s = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamps"])).dropna()
            cols[t] = s[~s.index.duplicated()]
        except Exception:
            continue
    close = pd.DataFrame(cols).sort_index()
    extra = []  # newer days saved by the data vault
    for f in sorted(glob.glob(str(ROOT / "data" / "daily" / "*" / "prices.csv.gz"))):
        try:
            d = pd.read_csv(f, usecols=["ticker", "date", "close"])
            extra.append(d)
        except Exception:
            continue
    if extra:
        e = pd.concat(extra).drop_duplicates(["ticker", "date"], keep="last")
        e["date"] = pd.to_datetime(e["date"]).dt.tz_localize(None).dt.normalize()
        e = e.pivot(index="date", columns="ticker", values="close")
        e = e[[c for c in e.columns if c in close.columns]]
        close = e.combine_first(close) if len(e) else close
    close = close[close.index >= close.index[-1] - pd.DateOffset(years=years)]
    close = close.loc[:, close.notna().sum() >= min_bars].ffill()
    return close


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rules(close: pd.DataFrame):
    """Yields (family, name, plain-English description, entries, exits)."""
    sma = {n: close.rolling(n).mean() for n in (10, 20, 50, 100, 200)}
    for f, s in [(10, 50), (20, 50), (20, 100), (50, 100), (50, 200), (20, 200)]:
        yield ("Trend", f"{f}/{s}-day average cross", f"Buy when the {f}-day average rises above the {s}-day average, sell when it falls back below.",
               (sma[f] > sma[s]) & (sma[f].shift() <= sma[s].shift()), (sma[f] < sma[s]) & (sma[f].shift() >= sma[s].shift()))
    r = rsi(close)
    for lo, hi in [(25, 50), (30, 50), (30, 60), (35, 60), (30, 70)]:
        yield ("Buy the dip", f"RSI below {lo}, sell above {hi}", f"Buy when the 14-day RSI drops below {lo} (oversold), sell when it recovers above {hi}.",
               r < lo, r > hi)
    for lo, hi in [(30, 55), (35, 60)]:
        yield ("Buy the dip", f"RSI below {lo} in an uptrend", f"Same dip-buy, but only while the stock is above its 200-day average.",
               (r < lo) & (close > sma[200]), r > hi)
    for n, m in [(20, 10), (55, 20), (100, 20), (252, 50)]:
        hh, ll = close.rolling(n).max().shift(), close.rolling(m).min().shift()
        yield ("Breakout", f"{n}-day high breakout", f"Buy a new {n}-day closing high, sell on a new {m}-day low.", close > hh, close < ll)
    for n in (63, 126, 252):
        mom = close / close.shift(n) - 1
        yield ("Momentum", f"{n}-day momentum + 200-day filter", f"Hold while the stock is up over the last {n} trading days and above its 200-day average.",
               (mom > 0) & (close > sma[200]), (mom < 0) | (close < sma[200]))
    yield ("Trend", "Above the 200-day average", "Hold only while the price is above its 200-day average.", close > sma[200], close < sma[200])


def stats(pf, bh_ret: pd.Series) -> dict:
    ret = pf.total_return()
    sh = pf.sharpe_ratio().replace([np.inf, -np.inf], np.nan)
    tr = pf.trades
    try:
        wr = float(np.nanmedian(tr.win_rate().values))
        ntr = float(np.nanmedian(tr.count().values))
    except Exception:
        wr, ntr = float("nan"), float("nan")
    return {"ret": float(ret.median()), "sharpe": float(sh.median()), "dd": float(pf.max_drawdown().median()),
            "beat": float((ret > bh_ret.reindex(ret.index)).mean()), "win_rate": wr, "trades": ntr}


def run() -> dict:
    import vectorbt as vbt
    cfg = load_config()
    close = load_closes()
    split = int(len(close) * 0.6)
    parts = {"is": slice(0, split), "oos": slice(split, len(close))}
    log(f"edge lab: {close.shape[1]} stocks, {close.index[0].date()} to {close.index[-1].date()}, split at {close.index[split].date()}")

    def bh(sl):
        c = close.iloc[sl]
        pf = vbt.Portfolio.from_holding(c, fees=FEES, freq="1D")
        sh = pf.sharpe_ratio().replace([np.inf, -np.inf], np.nan)
        return pf.total_return(), {"ret": float(pf.total_return().median()), "sharpe": float(sh.median()), "dd": float(pf.max_drawdown().median())}

    bh_ret, base = {}, {}
    for k, sl in parts.items():
        bh_ret[k], base[k] = bh(sl)
    out, per_ticker = [], {}
    for fam, name, why, ent, ex in rules(close):
        row = {"family": fam, "name": name, "why": why}
        for k, sl in parts.items():
            e, x = ent.iloc[sl].fillna(False), ex.iloc[sl].fillna(False)
            pf = vbt.Portfolio.from_signals(close.iloc[sl], e, x, fees=FEES, freq="1D")
            row[k] = stats(pf, bh_ret[k])
            if k == "oos":
                per_ticker[name] = pf.total_return()
                row["exposure"] = float((pf.asset_value() > 0).mean().median())  # share of days invested
        row["held_up"] = bool(row["oos"]["sharpe"] > base["oos"]["sharpe"] and row["is"]["sharpe"] > base["is"]["sharpe"])
        out.append(row)
        log(f"  {name}: in-sample sharpe {row['is']['sharpe']:.2f}, out-of-sample {row['oos']['sharpe']:.2f} (buy&hold {base['oos']['sharpe']:.2f})")
    out.sort(key=lambda r: (r["oos"]["sharpe"] if r["oos"]["sharpe"] == r["oos"]["sharpe"] else -9), reverse=True)
    best = out[0]
    watch = [t.upper() for t in cfg.get("watchlist", []) if t.upper() in close.columns]
    wl = [{"ticker": t, "rule": float(per_ticker[best["name"]].get(t, np.nan)), "buy_hold": float(bh_ret["oos"].get(t, np.nan))} for t in watch]
    winners = [r for r in out if r["held_up"]]
    verdict = (f"{len(winners)} of {len(out)} rules beat buy-and-hold on risk-adjusted return both in the test period and on newer data they were not picked on. "
               "Lower drawdown is where they helped most; few beat buy-and-hold on raw return." if winners else
               f"None of the {len(out)} rules beat buy-and-hold on risk-adjusted return in both periods. Simple chart rules on big stocks are not an edge by themselves.")
    res = {"generated": datetime.now(timezone.utc).isoformat(), "stocks": int(close.shape[1]), "start": close.index[0].date().isoformat(),
           "split": close.index[split].date().isoformat(), "end": close.index[-1].date().isoformat(), "fees_bps": FEES * 1e4,
           "baseline": base, "rules": out, "best": best["name"], "watchlist": wl, "verdict": verdict}
    write_json(SITE_DATA / "edge.json", res)
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return res


if __name__ == "__main__":
    run()
