"""Backtest lab, what-if replay and model battle (walk-forward: Kronos only ever sees the past)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Forecaster, Job
from .signals import analyze, chart_payload


def _metrics(daily: pd.Series) -> dict:
    eq = (1 + daily.fillna(0)).cumprod()
    dd = eq / eq.cummax() - 1
    sd = daily.std()
    return {"total_return": float(eq.iloc[-1] - 1), "max_drawdown": float(dd.min()),
            "sharpe": float(daily.mean() / sd * np.sqrt(252)) if sd and sd > 0 else 0.0,
            "equity": [round(float(x), 5) for x in eq.values]}


def backtest(ticker: str, df: pd.DataFrame, fc: Forecaster, cfg: dict, cls: str = "stock") -> dict:
    bc, fcfg = cfg.get("backtest", {}), cfg.get("forecast", {})
    H = fcfg.get("horizon", 5)
    step = max(bc.get("step", 5), H)
    thr = bc.get("threshold", 0.65)
    cost = bc.get("cost_bps", 5) / 1e4
    allow_short = cfg.get("paper", {}).get("allow_short", False)
    n = len(df)
    start = max(n - bc.get("days", 250), fcfg.get("lookback", 400) // 2)
    asofs = list(range(start, n - H + 1, step))
    if not asofs:
        raise ValueError(f"{ticker}: not enough history to backtest ({n} bars)")
    jobs = [Job(ticker, df, H, "1d", cls, asof=i) for i in asofs]
    res = fc.run(jobs, samples=bc.get("samples", 16), lookback=fcfg.get("lookback", 400),
                 temperature=fcfg.get("temperature", 1.0), top_p=fcfg.get("top_p", 0.9),
                 batch_size=fcfg.get("batch_size", 64), seed=7)
    o, c = df["open"].values, df["close"].values
    strat = np.zeros(n)
    trades, hits, errs = [], [], []
    for i in asofs:
        p = res.get(f"{ticker}@{i}")
        if p is None:
            continue
        a = analyze(p, cfg)
        actual = c[i + H - 1] / c[i - 1] - 1
        hits.append((actual > 0) == (a["direction"] == "up"))
        errs.append(abs(a["median_return"] - actual))
        side = 1 if a["direction"] == "up" else -1
        if a["agreement"] < thr or (side < 0 and not allow_short):
            continue
        for k in range(i, i + H):
            r = c[k] / o[k] - 1 if k == i else c[k] / c[k - 1] - 1
            strat[k] += side * r
        strat[i] -= cost
        strat[i + H - 1] -= cost
        tr = side * (c[i + H - 1] / o[i] - 1) - 2 * cost
        trades.append({"date": df.index[i].date().isoformat(), "side": "long" if side > 0 else "short",
                       "entry": float(o[i]), "exit": float(c[i + H - 1]), "return": float(tr), "agreement": a["agreement"]})
    win = slice(asofs[0], asofs[-1] + H) if asofs else slice(0, 0)
    dates = [d.date().isoformat() for d in df.index[win]]
    k = pd.Series(strat[win])
    bh = pd.Series(c[win] / np.r_[o[win.start], c[win][:-1]] - 1) if asofs else pd.Series(dtype=float)
    s20, s50 = df["close"].rolling(20).mean().values, df["close"].rolling(50).mean().values
    pos = np.r_[0, (s20[:-1] > s50[:-1]).astype(float)]
    sma = pd.Series((pos * np.r_[0, c[1:] / c[:-1] - 1])[win])
    out = {"ticker": ticker, "dates": dates, "horizon": H, "threshold": thr, "model": fc.name,
           "kronos": _metrics(k), "buy_hold": _metrics(bh), "sma_cross": _metrics(sma),
           "trades": trades, "n_trades": len(trades),
           "win_rate": float(np.mean([t["return"] > 0 for t in trades])) if trades else None,
           "hit_rate": float(np.mean(hits)) if hits else None, "mae": float(np.mean(errs)) if errs else None,
           "n_forecasts": len(hits)}
    out["beats_buy_hold"] = out["kronos"]["total_return"] > out["buy_hold"]["total_return"]
    return out


def replay(ticker: str, df: pd.DataFrame, date: str, fc: Forecaster, cfg: dict, cls: str = "stock") -> dict:
    fcfg = cfg.get("forecast", {})
    H = fcfg.get("horizon", 5)
    i = int(df.index.searchsorted(pd.Timestamp(date), side="right"))
    i = min(i, len(df))
    res = fc.run([Job(ticker, df, H, "1d", cls, asof=i)], samples=fcfg.get("samples", 30),
                 lookback=fcfg.get("lookback", 400), batch_size=fcfg.get("batch_size", 64), seed=11)
    p = res.get(f"{ticker}@{i}")
    if p is None:
        raise ValueError(f"{ticker}: not enough history before {date} (needs ~120 bars)")
    a = analyze(p, cfg)
    actual = df.iloc[i:i + H]
    a["chart"] = chart_payload(p)
    a["actual"] = [[t.isoformat(), r.open, r.high, r.low, r.close] for t, r in actual.iterrows()]
    if len(actual):
        ar = float(actual["close"].iloc[-1] / p.last_close - 1)
        a["actual_return"] = ar
        a["correct"] = (ar > 0) == (a["direction"] == "up")
    return a


def battle(tickers: list[str], hist: dict[str, pd.DataFrame], cfg: dict, classes: dict[str, str]) -> list[dict]:
    rows = []
    for m in cfg.get("model", {}).get("battle", []):
        try:
            fc = Forecaster.load(m["name"], m["tokenizer"], m["max_context"])
        except Exception as e:
            rows.append({"model": m["name"], "error": str(e)})
            continue
        res = [backtest(t, hist[t], fc, cfg, classes.get(t, "stock")) for t in tickers if t in hist]
        res = [r for r in res if r["hit_rate"] is not None]
        if not res:
            continue
        rows.append({"model": m["name"], "tickers": len(res),
                     "hit_rate": float(np.mean([r["hit_rate"] for r in res])),
                     "mae": float(np.mean([r["mae"] for r in res])),
                     "avg_return": float(np.mean([r["kronos"]["total_return"] for r in res])),
                     "avg_buy_hold": float(np.mean([r["buy_hold"]["total_return"] for r in res])),
                     "beats_buy_hold": int(sum(r["beats_buy_hold"] for r in res))})
    return sorted(rows, key=lambda r: -(r.get("hit_rate") or 0))
