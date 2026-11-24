"""Second opinion: Google's TimesFM 3.0 forecasts the same 5 trading days Kronos does, on every S&P 500 stock.

Two models trained by different teams on different data make different mistakes. Where they agree is the
only part worth a second look, and whether agreement actually helps is measured, not assumed:
  * a walk-forward backtest of TimesFM alone (direction, and whether its favourites beat its least-liked)
  * a live log (state/second_log.csv) graded after 5 trading days, joined to Kronos' graded calls.
Model code comes from the user's fork: github.com/takaxkoi/timesfm (package `timesfm3`)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, write_json
from .feargreed import load_panel, stock_cols

H = 5                 # trading days ahead, same as Kronos
CONTEXT = 512         # about two years of daily closes
LOG = STATE_DIR / "second_log.csv"
DECILES = np.arange(0.1, 1.0, 0.1)
_model = None


def model():
    global _model
    if _model is None:
        from timesfm3 import TimesFM3Forecaster
        _model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cpu")
    return _model


def forecast(series: dict[str, np.ndarray], batch: int = 48) -> dict[str, dict]:
    """series: ticker -> closes (oldest first). Returns expected 5-day return and chance of a rise for each."""
    out, keys = {}, list(series)
    m = model()
    for i in range(0, len(keys), batch):
        ks = keys[i:i + batch]
        ctx = [np.log(series[k][-CONTEXT:]).astype(np.float32) for k in ks]
        res = list(m.predict_batch(contexts=ctx, horizon=H, return_quantiles=True))
        for k, c, r in zip(ks, ctx, res):
            f = np.asarray(r.forecast, dtype=float).reshape(-1)[H - 1]
            q = np.sort(np.asarray(r.quantiles, dtype=float).reshape(-1, 9)[H - 1])
            last = float(c[-1])
            # chance the price ends above today's: where today's price falls among the nine forecast deciles
            p_below = float(np.interp(last, q, DECILES, left=0.05, right=0.95))
            out[k] = {"ret": float(np.exp(f - last) - 1), "p_up": 1 - p_below, "lo": float(np.exp(q[0] - last) - 1), "hi": float(np.exp(q[-1] - last) - 1)}
    return out


def universe(close: pd.DataFrame) -> list[str]:
    extra = [t for t in ["SPY", "QQQ", "IWM", "DIA"] if t in close]
    return [t for t in stock_cols(close) + extra if close[t].notna().sum() >= 300]


def backtest(close: pd.DataFrame, dates: int = 50, every: int = 5, take: int = 4) -> dict:
    """Walk forward: on each past date, forecast with only the prices known that day, then look at what happened."""
    names = universe(close)
    names = [t for i, t in enumerate(sorted(names)) if i % take == 0]            # every 4th stock, alphabetically
    idx = close.index
    ends = [len(idx) - 1 - H - k * every for k in range(dates)]
    rows = []
    for n, e in enumerate(ends):
        if e < CONTEXT:
            break
        ser = {t: close[t].iloc[:e + 1].dropna().values for t in names}
        ser = {t: v for t, v in ser.items() if len(v) >= 300}
        fc = forecast(ser)
        for t, f in fc.items():
            a, b = close[t].iloc[e], close[t].iloc[e + H]
            if a == a and b == b and a > 0:
                rows.append({"date": idx[e], "t": t, "ret": f["ret"], "p_up": f["p_up"], "actual": b / a - 1})
        if n % 10 == 0:
            log(f"second-model backtest: {n + 1}/{len(ends)} dates")
    d = pd.DataFrame(rows)
    d["ok"] = (d["ret"] > 0) == (d["actual"] > 0)
    by = d.groupby("date")
    # do its favourites beat its least-liked? top and bottom fifth by forecast, each date
    spread = by.apply(lambda g: g.nlargest(max(1, len(g) // 5), "ret")["actual"].mean() - g.nsmallest(max(1, len(g) // 5), "ret")["actual"].mean())
    top = by.apply(lambda g: g.nlargest(max(1, len(g) // 5), "ret")["actual"].mean())
    allm = by["actual"].mean()
    sure = d[(d["p_up"] > 0.6) | (d["p_up"] < 0.4)]
    res = {"made": datetime.now(timezone.utc).isoformat(), "calls": int(len(d)), "dates": int(by.ngroups), "stocks": int(d["t"].nunique()),
           "first": d["date"].min().date().isoformat(), "last": d["date"].max().date().isoformat(),
           "hit": float(d["ok"].mean()), "always_up": float((d["actual"] > 0).mean()), "said_up": float((d["ret"] > 0).mean()),
           "hit_up": float(d[d["ret"] > 0]["ok"].mean()) if (d["ret"] > 0).any() else None, "hit_down": float(d[d["ret"] <= 0]["ok"].mean()) if (d["ret"] <= 0).any() else None,
           "sure_n": int(len(sure)), "sure_hit": float(sure["ok"].mean()) if len(sure) else None,
           "spread": float(spread.mean()), "spread_t": float(spread.mean() / spread.std() * np.sqrt(len(spread))) if spread.std() > 0 else 0.0,
           "spread_wins": int((spread > 0).sum()), "top_avg": float(top.mean()), "all_avg": float(allm.mean()),
           "date_hit": float((by["ok"].mean() > 0.5).mean())}
    write_json(STATE_DIR / "second_backtest.json", res)
    return res


def grade(close: pd.DataFrame) -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame()
    d = pd.read_csv(LOG, parse_dates=["date"])
    pos = {dt: i for i, dt in enumerate(close.index)}
    act = []
    for _, r in d.iterrows():
        i = pos.get(r["date"])
        t = r["ticker"]
        if i is None or t not in close or i + H >= len(close.index):
            act.append(np.nan)
            continue
        a, b = close[t].iloc[i], close[t].iloc[i + H]
        act.append(b / a - 1 if a == a and b == b and a > 0 else np.nan)
    d["actual"] = act
    return d


def kronos_now() -> tuple[dict, str | None]:
    """Kronos' latest call for every stock it has one for: scanner heatmap first, then the watchlist run."""
    out, when = {}, None
    sc, la = read_json(SITE_DATA / "scanner.json") or {}, read_json(SITE_DATA / "latest.json") or {}
    for r in sc.get("heatmap") or []:
        out[r["t"]] = {"dir": r.get("d"), "ret": r.get("r"), "p_up": r.get("p"), "sector": r.get("s")}
    when = sc.get("generated")
    for key in ("indices", "watchlist"):
        for r in la.get(key) or []:
            out[r["ticker"]] = {"dir": r.get("direction"), "ret": r.get("median_return"), "p_up": r.get("p_up"), "sector": r.get("sector")}
    return out, when or la.get("generated")


def run(do_backtest: bool = False) -> dict:
    close, _ = load_panel()
    names = universe(close)
    asof = close.index[-1]
    fc = forecast({t: close[t].dropna().values for t in names})
    log(f"second model: {len(fc)} forecasts as of {asof.date()}")

    # log today's calls once per date
    new = pd.DataFrame([{"date": asof.date().isoformat(), "ticker": t, "price": float(close[t].iloc[-1]), "ret": round(f["ret"], 5), "p_up": round(f["p_up"], 4)} for t, f in fc.items()])
    if LOG.exists():
        old = pd.read_csv(LOG)
        new = pd.concat([old[old["date"] != asof.date().isoformat()], new])
    new.to_csv(LOG, index=False)

    bt = backtest(close) if do_backtest else (read_json(STATE_DIR / "second_backtest.json") or None)

    # live record, and the joint record with Kronos
    live, joint = None, None
    g = grade(close).dropna(subset=["actual"]) if LOG.exists() else pd.DataFrame()
    if len(g):
        g["ok"] = (g["ret"] > 0) == (g["actual"] > 0)
        live = {"calls": int(len(g)), "days": int(g["date"].nunique()), "hit": float(g["ok"].mean()), "always_up": float((g["actual"] > 0).mean())}
        try:
            p = pd.read_csv(STATE_DIR / "predictions.csv", low_memory=False)
            p = p[(p["interval"] == "1d") & (p["graded"].astype(str) == "True")].copy()
            p["date"] = pd.to_datetime(p["asof"].astype(str).str[:10])
            j = g.merge(p[["date", "ticker", "direction", "correct"]], on=["date", "ticker"])
            if len(j):
                j["k_ok"] = j["correct"].astype(str).str.lower().eq("true")
                j["same"] = (j["ret"] > 0) == (j["direction"] == "up")
                joint = {"calls": int(len(j)), "agree_n": int(j["same"].sum()), "agree_hit": float(j[j["same"]]["k_ok"].mean()) if j["same"].any() else None,
                         "split_n": int((~j["same"]).sum()), "kronos_hit": float(j["k_ok"].mean()), "timesfm_hit": float(j["ok"].mean())}
        except Exception as e:
            log(f"joint record skipped: {type(e).__name__}")

    K, k_when = kronos_now()
    watch = [t.upper() for t in load_config().get("watchlist", [])]
    rows = []
    for t, f in fc.items():
        k = K.get(t) or {}
        t_dir = "up" if f["ret"] > 0 else "down"
        rows.append({"t": t, "sector": k.get("sector"), "price": float(close[t].iloc[-1]), "t_dir": t_dir, "t_ret": f["ret"], "t_p": f["p_up"], "t_lo": f["lo"], "t_hi": f["hi"],
                     "k_dir": k.get("dir"), "k_ret": k.get("ret"), "k_p": k.get("p_up"), "agree": (k.get("dir") == t_dir) if k.get("dir") else None, "watch": t in watch})
    both = [r for r in rows if r["agree"]]
    up = sorted([r for r in both if r["t_dir"] == "up"], key=lambda r: -(r["t_p"] + (r["k_p"] or 0.5)))
    have = [r for r in rows if r["agree"] is not None]
    res = {"generated": datetime.now(timezone.utc).isoformat(), "asof": asof.date().isoformat(), "kronos_asof": k_when, "horizon": H, "model": "TimesFM 3.0 (Google Research)",
           "n": len(rows), "compared": len(have), "agree": len(both), "agree_share": (len(both) / len(have)) if have else None,
           "t_up_share": float(np.mean([r["t_dir"] == "up" for r in rows])), "k_up_share": float(np.mean([r["k_dir"] == "up" for r in have])) if have else None,
           "watch": [r for r in rows if r["watch"]], "both_up": up[:25], "rows": rows, "backtest": bt, "live": live, "joint": joint}
    write_json(SITE_DATA / "second.json", res)
    log(f"second model: agrees with Kronos on {len(both)} of {len(have)}; " + (f"backtest hit {bt['hit']:.1%} vs always-up {bt['always_up']:.1%}, spread {bt['spread']:+.2%} (t {bt['spread_t']:.1f})" if bt else "no backtest yet"))
    return res


if __name__ == "__main__":
    run(do_backtest="--backtest" in sys.argv)
