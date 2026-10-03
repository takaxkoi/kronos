"""Turns Kronos's simulated futures into scores, ranges, signals and trade plans."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import Paths


# ── classic indicators (for context + the ensemble second opinion) ──
def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    v = 100 - 100 / (1 + rs)
    return float(v.iloc[-1]) if not np.isnan(v.iloc[-1]) else 50.0


def atr(df: pd.DataFrame, n: int = 14) -> float:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def indicators(df: pd.DataFrame, cfg: dict) -> dict:
    c = df["close"]
    rets = np.log(c).diff().dropna()
    gaps = (df["open"] / c.shift(1) - 1).dropna()
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    n = cfg.get("signals", {}).get("breakout_lookback", 20)
    return {
        "sma20": float(c.tail(20).mean()), "sma50": float(c.tail(50).mean()),
        "rsi": rsi(c, cfg.get("ensemble", {}).get("rsi_period", 14)),
        "macd_hist": float((macd - macd.ewm(span=9).mean()).iloc[-1]),
        "atr": atr(df), "vol": float(rets.tail(60).std() or 1e-4),
        "avg_abs_ret": float(rets.tail(60).abs().mean() or 1e-4),
        "avg_abs_gap": float(gaps.tail(60).abs().mean()) if len(gaps) else 0.0,
        "ret10": float(c.iloc[-1] / c.iloc[-11] - 1) if len(c) > 11 else 0.0,
        "ret20": float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0.0,
        "hi_n": float(df["high"].tail(n).max()), "lo_n": float(df["low"].tail(n).min()),
        "avg_vol20": float(df["volume"].tail(20).mean()),
        "dollar_vol20": float((df["volume"] * df["close"]).tail(20).mean()),
    }


def _q(a, q):
    return float(np.nanpercentile(a, q))


def analyze(p: Paths, cfg: dict, earnings: str | None = None, sector: str | None = None) -> dict:
    sc, rk = cfg.get("signals", {}), cfg.get("risk", {})
    df, arr = p.history, p.arr.copy()
    arr[:, :, :4] = np.clip(arr[:, :, :4], 1e-9, None)
    S, H = arr.shape[0], arr.shape[1]
    o, h, l, c, v = (arr[:, :, i] for i in range(5))
    c0 = p.last_close
    ind = indicators(df, cfg)

    fin = c[:, -1] / c0 - 1
    p_up = float(np.mean(fin > 0))
    direction = "up" if p_up >= 0.5 else "down"
    agreement = max(p_up, 1 - p_up)
    n_agree = int(round(agreement * S))
    med = float(np.median(fin))

    # Kronos Score 0-100: confidence (50%), size of move vs normal (30%), tightness of paths (20%)
    sigma_h = ind["vol"] * math.sqrt(H)
    conf = (agreement - 0.5) / 0.5
    mag = min(1.0, abs(med) / (0.75 * sigma_h + 1e-9))
    tight = float(np.clip(1 - (_q(fin, 75) - _q(fin, 25)) / (2.5 * sigma_h + 1e-9), 0, 1))
    score = int(round(100 * (0.5 * conf + 0.3 * mag + 0.2 * tight)))

    def rng(k):
        k = min(k, H - 1)
        return {"low": _q(l[:, :k + 1].min(axis=1), 10), "high": _q(h[:, :k + 1].max(axis=1), 90),
                "mid": float(np.median(c[:, k])), "date": p.future[k].isoformat()}

    ranges = {"next": rng(0), "week": rng(4 if p.interval == "1d" else H - 1), "horizon": rng(H - 1)}

    flags, signals = [], {}
    # breakout / breakdown vs recent N-bar high/low
    p_brk = float(np.mean(c.max(axis=1) > ind["hi_n"]))
    p_bdn = float(np.mean(c.min(axis=1) < ind["lo_n"]))
    if p_brk >= 0.6 and direction == "up":
        flags.append("breakout"); signals["breakout"] = {"level": ind["hi_n"], "prob": p_brk}
    if p_bdn >= 0.6 and direction == "down":
        flags.append("breakdown"); signals["breakdown"] = {"level": ind["lo_n"], "prob": p_bdn}
    # reversals
    if (ind["ret10"] <= -0.07 or ind["rsi"] < 32) and p_up >= 0.65:
        flags.append("reversal_up"); signals["reversal_up"] = {"ret10": ind["ret10"], "rsi": ind["rsi"]}
    if (ind["ret10"] >= 0.10 or ind["rsi"] > 72) and p_up <= 0.35:
        flags.append("reversal_down"); signals["reversal_down"] = {"ret10": ind["ret10"], "rsi": ind["rsi"]}
    # momentum confirm
    if c0 > ind["sma20"] > ind["sma50"] and ind["ret20"] > 0 and p_up >= 0.62:
        flags.append("momentum_up")
    if c0 < ind["sma20"] < ind["sma50"] and ind["ret20"] < 0 and p_up <= 0.38:
        flags.append("momentum_down")
    # gap prediction (markets that close overnight)
    if p.cls != "crypto" and p.interval == "1d":
        gap = float(np.median(o[:, 0]) / c0 - 1)
        p_gu = float(np.mean(o[:, 0] > c0))
        signals["gap"] = {"pct": gap, "p_up": p_gu}
        if abs(gap) >= max(0.5 * ind["avg_abs_gap"], 0.004) and max(p_gu, 1 - p_gu) >= 0.7:
            flags.append("gap_up" if gap > 0 else "gap_down")
    # volume surge
    if ind["avg_vol20"] > 0:
        vr = float(np.median(v.mean(axis=1)) / ind["avg_vol20"])
        signals["volume_ratio"] = vr
        if vr >= sc.get("volume_surge_ratio", 1.5):
            flags.append("volume_surge")
    # volatility warning
    path_c = np.concatenate([np.full((S, 1), c0), c], axis=1)
    pred_abs = float(np.median(np.abs(np.diff(np.log(path_c), axis=1)).mean(axis=1)))
    vratio = pred_abs / (ind["avg_abs_ret"] + 1e-9)
    signals["volatility_ratio"] = vratio
    if vratio >= sc.get("volatility_warn_ratio", 1.6):
        flags.append("volatility")
    # earnings inside the window -> pause
    paused = False
    if earnings and p.cls == "stock":
        ed = pd.Timestamp(earnings)
        if df.index[-1] < ed <= p.future[-1] + pd.Timedelta(days=1):
            paused = sc.get("earnings_pause", True)
            flags.append("earnings")
    # liquidity / penny filter
    flt = cfg.get("filters", {})
    illiquid = p.cls == "stock" and (c0 < flt.get("min_price", 3) or ind["dollar_vol20"] < flt.get("min_avg_dollar_volume", 5e6))
    if illiquid:
        flags.append("illiquid")
    if p.ticker in (cfg.get("leveraged") or []):
        flags.append("leveraged")

    # ensemble second opinion (RSI, trend, MACD)
    votes = [ind["rsi"] < 50 and direction == "down" or ind["rsi"] >= 50 and direction == "up",
             (ind["sma20"] > ind["sma50"]) == (direction == "up"),
             (ind["macd_hist"] > 0) == (direction == "up")]
    ensemble = {"agree": int(sum(votes)), "of": 3, "rsi": ind["rsi"], "trend": "up" if ind["sma20"] > ind["sma50"] else "down",
                "macd": "up" if ind["macd_hist"] > 0 else "down"}

    plan = trade_plan(p, direction, ind, rk, sc, arr)
    return {
        "ticker": p.ticker, "cls": p.cls, "interval": p.interval, "sector": sector, "model": p.model,
        "asof": df.index[-1].isoformat(), "last_close": c0, "horizon": H, "samples": S,
        "direction": direction, "p_up": p_up, "agreement": agreement, "agree_count": n_agree,
        "median_return": med, "mean_return": float(np.mean(fin)), "p10_return": _q(fin, 10), "p90_return": _q(fin, 90),
        "score": score, "ranges": ranges, "flags": flags, "signals": signals, "plan": plan,
        "ensemble": ensemble, "earnings": earnings, "paused": paused, "illiquid": illiquid,
        "indicators": {k: ind[k] for k in ("rsi", "sma20", "sma50", "atr", "ret10", "ret20", "dollar_vol20")},
        "step_median": [float(x) for x in np.median(c, axis=0)],
        "future": [t.isoformat() for t in p.future],
    }


def trade_plan(p: Paths, direction: str, ind: dict, rk: dict, sc: dict, arr: np.ndarray) -> dict:
    o, h, l, c = (arr[:, :, i] for i in range(4))
    c0, a = p.last_close, ind["atr"]
    med_c = np.median(c, axis=0)
    if direction == "up":
        zone = [min(float(np.median(l[:, 0])), c0 - 0.25 * a), c0]
        entry = sum(zone) / 2
        stop = min(_q(l.min(axis=1), 10) - 0.25 * a, zone[0] - 0.25 * a, entry - a)
        target = float(np.median(h.max(axis=1)))
        risk, reward = entry - stop, target - entry
        peak = int(np.argmax(med_c))
        best = int(np.argmin(np.median(l, axis=0)[: peak + 1]))
    else:
        zone = [c0, max(float(np.median(h[:, 0])), c0 + 0.25 * a)]
        entry = sum(zone) / 2
        stop = max(_q(h.max(axis=1), 90) + 0.25 * a, zone[1] + 0.25 * a, entry + a)
        target = float(np.median(l.min(axis=1)))
        risk, reward = stop - entry, entry - target
        peak = int(np.argmin(med_c))
        best = int(np.argmax(np.median(h, axis=0)[: peak + 1]))
    rr = reward / risk if risk > 0 else 0.0
    acct, rpt = rk.get("account_size", 10000), rk.get("risk_per_trade", 0.01)
    shares = int(min(acct * rpt / risk, acct / entry)) if risk > 0 and entry > 0 else 0
    return {"side": "long" if direction == "up" else "short", "entry_low": zone[0], "entry_high": zone[1],
            "entry": entry, "stop": stop, "target": target, "rr": rr, "valid": bool(rr >= sc.get("min_rr", 2.0) and reward > 0),
            "hold_bars": peak + 1, "best_entry_bar": best + 1, "best_entry_date": p.future[best].isoformat(),
            "exit_date": p.future[peak].isoformat(), "shares": shares, "risk_dollars": shares * risk,
            "reward_dollars": shares * reward}


def chart_payload(p: Paths, bars: int = 90, spaghetti: int = 12) -> dict:
    df = p.history.tail(bars)
    arr = p.arr
    med = np.median(arr[:, :, :4], axis=0)
    med[:, 1] = np.maximum(med[:, 1], med[:, [0, 3]].max(axis=1))
    med[:, 2] = np.minimum(med[:, 2], med[:, [0, 3]].min(axis=1))
    closes = arr[:, :, 3]
    return {
        "history": [[t.isoformat(), r.open, r.high, r.low, r.close, r.volume] for t, r in df.iterrows()],
        "future": [t.isoformat() for t in p.future],
        "median": med.tolist(),
        "volume": np.median(arr[:, :, 4], axis=0).tolist(),
        "bands": {str(q): np.percentile(closes, q, axis=0).tolist() for q in (10, 25, 50, 75, 90)},
        "paths": closes[: spaghetti].tolist(),
    }
