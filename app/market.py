"""Market-wide views: heatmap, sector rotation, mood gauge, correlation splits, portfolio + hedges."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import Paths


def mood(analyses: list[dict]) -> dict:
    stocks = [a for a in analyses if a["cls"] == "stock" and not a.get("illiquid")]
    if not stocks:
        return {}
    up = sum(a["direction"] == "up" for a in stocks)
    pct = up / len(stocks)
    avg = float(np.mean([a["median_return"] for a in stocks]))
    label = "RISK-ON" if pct >= 0.6 else "RISK-OFF" if pct <= 0.4 else "NEUTRAL"
    return {"pct_up": pct, "n": len(stocks), "avg_return": avg, "label": label,
            "breakouts": sum("breakout" in a["flags"] for a in stocks),
            "breakdowns": sum("breakdown" in a["flags"] for a in stocks)}


def heatmap(analyses: list[dict]) -> list[dict]:
    return [{"t": a["ticker"], "s": a.get("sector") or "Other", "r": a["median_return"], "sc": a["score"],
             "d": a["direction"], "p": a["p_up"], "f": a["flags"]}
            for a in analyses if a["cls"] == "stock" and not a.get("illiquid")]


def sector_rotation(analyses: dict[str, dict], cfg: dict) -> list[dict]:
    rows = []
    for etf, name in (cfg.get("sectors") or {}).items():
        a = analyses.get(etf)
        if a:
            rows.append({"etf": etf, "name": name, "median_return": a["median_return"], "p_up": a["p_up"],
                         "score": a["score"], "direction": a["direction"]})
    return sorted(rows, key=lambda r: -r["median_return"])


def correlation_alerts(paths: dict[str, Paths], analyses: dict[str, dict], tickers: list[str],
                       min_corr: float = 0.75, min_agree: float = 0.65) -> list[dict]:
    closes = {t: paths[t].history["close"] for t in tickers if t in paths and t in analyses}
    if len(closes) < 2:
        return []
    rets = pd.DataFrame(closes).pct_change().tail(90)
    corr = rets.corr()
    out = []
    names = list(closes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = corr.at[a, b]
            if np.isnan(c) or c < min_corr:
                continue
            A, B = analyses[a], analyses[b]
            if A["direction"] != B["direction"] and A["agreement"] >= min_agree and B["agreement"] >= min_agree:
                out.append({"a": a, "b": b, "corr": float(c), "a_dir": A["direction"], "b_dir": B["direction"],
                            "a_ret": A["median_return"], "b_ret": B["median_return"]})
    return sorted(out, key=lambda r: -r["corr"])


def portfolio_forecast(holdings: dict[str, float], paths: dict[str, Paths], cfg: dict) -> dict | None:
    """Predicted value of your holdings at the forecast horizon (sums simulated paths)."""
    held = {t: q for t, q in holdings.items() if t in paths}
    if not held:
        return None
    S = min(paths[t].arr.shape[0] for t in held)
    now_val = sum(q * paths[t].last_close for t, q in held.items())
    sims = np.zeros(S)
    rows = []
    for t, q in held.items():
        fin = paths[t].arr[:S, -1, 3]
        sims += q * fin
        rows.append({"ticker": t, "shares": q, "value": q * paths[t].last_close,
                     "pred_value": float(q * np.median(fin)), "ret": float(np.median(fin) / paths[t].last_close - 1)})
    ret = sims / now_val - 1
    out = {"value": now_val, "pred_median": float(np.median(sims)), "pred_p10": float(np.percentile(sims, 10)),
           "pred_p90": float(np.percentile(sims, 90)), "median_return": float(np.median(ret)),
           "p_down": float(np.mean(ret < 0)), "holdings": sorted(rows, key=lambda r: -r["value"])}
    hc = cfg.get("hedge", {})
    if out["median_return"] <= hc.get("trigger_return", -0.02) or out["p_down"] >= hc.get("trigger_prob", 0.65):
        out["hedge"] = hedge_suggestion(held, paths, now_val, hc.get("ratio", 0.5))
    return out


def hedge_suggestion(held: dict[str, float], paths: dict[str, Paths], value: float, ratio: float) -> dict:
    best = None
    for idx, inv in (("SPY", "SH"), ("QQQ", "PSQ")):
        if idx not in paths:
            continue
        ir = paths[idx].history["close"].pct_change().tail(120)
        port = sum(q * paths[t].history["close"] for t, q in held.items()).pct_change().tail(120)
        df = pd.concat([port, ir], axis=1).dropna()
        if len(df) < 30:
            continue
        cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
        beta = cov[0, 1] / cov[1, 1]
        corr = float(df.corr().iloc[0, 1])
        if beta <= 0.2:  # holdings barely move with this index; it would not hedge anything
            continue
        if best is None or corr > best["corr"]:
            px = paths[idx].last_close
            notional = value * beta * ratio
            best = {"index": idx, "inverse_etf": inv, "beta": float(beta), "corr": corr, "notional": float(notional),
                    "short_shares": int(notional / px), "index_price": px,
                    "note": f"Hedge ~{int(ratio * 100)}% of market exposure: short {int(notional / px)} {idx}, "
                            f"or buy ${notional:,.0f} of {inv}, or buy {idx} puts."}
    return best or {}
