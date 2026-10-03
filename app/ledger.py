"""Prediction log + report card: every call is saved, then graded against what really happened."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import STATE_DIR, log
from .data import fetch_history

PRED_FILE = STATE_DIR / "predictions.csv"
GRADE_PERIOD = {"1d": "6mo", "1h": "60d", "1wk": "2y"}


def load() -> pd.DataFrame:
    if not PRED_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_FILE, dtype={"id": str})
    df = df[df["id"] != "id"]  # stray headers from union merges
    if df["id"].duplicated().any():  # union merge can keep an ungraded + graded copy: keep the graded one
        df["_g"] = df["graded"].astype(str).str.lower().eq("true")
        df = df.sort_values("_g").drop_duplicates("id", keep="last").drop(columns="_g").sort_index()
    return df.reset_index(drop=True)


def save(df: pd.DataFrame) -> None:
    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRED_FILE, index=False)


def record(analyses: list[dict], vix: float | None, detail: bool = True) -> int:
    """Append new predictions (one per ticker/bar/model)."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for a in analyses:
        pl = a["plan"]
        rows.append({
            "id": f"{a['asof']}|{a['ticker']}|{a['interval']}|{a['model']}",
            "run_ts": now, "asof": a["asof"], "ticker": a["ticker"], "cls": a["cls"], "interval": a["interval"],
            "sector": a.get("sector") or "", "model": a["model"], "horizon": a["horizon"], "last_close": a["last_close"],
            "direction": a["direction"], "p_up": a["p_up"], "agreement": a["agreement"], "score": a["score"],
            "median_return": a["median_return"], "flags": "|".join(a["flags"]),
            "side": pl["side"], "entry": pl["entry"], "stop": pl["stop"], "target": pl["target"], "plan_valid": pl["valid"],
            "paused": a.get("paused", False), "vix": vix if vix is not None else np.nan,
            "step_median": json.dumps([round(x, 4) for x in a["step_median"]]) if detail else "",
            "graded": False, "graded_ts": "", "actual": "", "actual_return": np.nan, "correct": np.nan, "outcome": "", "trade_return": np.nan,
        })
    if not rows:
        return 0
    old = load()
    new = pd.DataFrame(rows)
    if not old.empty:
        new = new[~new["id"].isin(set(old["id"]))]
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    save(df)
    return len(new)


def simulate_plan(bars: pd.DataFrame, long: bool, entry: float, stop: float, target: float) -> tuple[str, float]:
    """Limit entry must actually be touched; gaps through the stop exit at the open (worse than the stop)."""
    filled = False
    for _, bar in bars.iterrows():
        if not filled:
            if (bar["low"] <= entry) if long else (bar["high"] >= entry):
                filled = True
                fill = min(entry, bar["open"]) if long else max(entry, bar["open"])
                entry = fill
            else:
                continue
        if (bar["low"] <= stop) if long else (bar["high"] >= stop):  # stop wins a same-bar tie
            px = min(stop, bar["open"]) if long else max(stop, bar["open"])
            return "stop", (px / entry - 1) * (1 if long else -1)
        if (bar["high"] >= target) if long else (bar["low"] <= target):
            return "target", (target / entry - 1) * (1 if long else -1)
    if not filled:
        return "no_fill", np.nan
    return "time", (float(bars["close"].iloc[-1]) / entry - 1) * (1 if long else -1)


def grade(cfg: dict) -> int:
    df = load()
    if df.empty:
        return 0
    df["graded"] = df["graded"].astype(str).str.lower().eq("true")
    todo = df[~df["graded"]]
    if todo.empty:
        return 0
    n = 0
    now = datetime.now(timezone.utc).isoformat()
    if "graded_ts" not in df.columns:
        df["graded_ts"] = ""
    for col in ("actual", "outcome", "graded_ts", "correct"):
        df[col] = df[col].astype(object)
    for interval, grp in todo.groupby("interval"):
        hist = fetch_history(sorted(grp["ticker"].unique()), interval, cfg, period=GRADE_PERIOD.get(interval, "6mo"))
        for i, r in grp.iterrows():
            hd = hist.get(r["ticker"])
            if hd is None:
                continue
            after = hd[hd.index > pd.Timestamp(r["asof"])]
            H = int(r["horizon"])
            if len(after) < H:
                continue
            after = after.iloc[:H]
            c0 = float(r["last_close"])
            fin = float(after["close"].iloc[-1]) / c0 - 1
            df.at[i, "actual"] = json.dumps([round(x, 4) for x in after["close"]])
            df.at[i, "actual_return"] = fin
            df.at[i, "correct"] = bool((fin > 0) == (r["direction"] == "up"))
            outcome = "time"
            if str(r["plan_valid"]).lower() == "true":
                outcome, tr = simulate_plan(after, r["side"] == "long", float(r["entry"]), float(r["stop"]), float(r["target"]))
                df.at[i, "trade_return"] = tr
            df.at[i, "outcome"] = outcome
            df.at[i, "graded"] = True
            df.at[i, "graded_ts"] = now
            n += 1
    save(df)
    log(f"graded {n} predictions")
    return n


def _rate(s: pd.Series) -> float | None:
    s = s.dropna()
    return float(s.astype(bool).mean()) if len(s) else None


def report(cfg: dict) -> dict:
    df = load()
    if df.empty:
        return {"n": 0}
    df["graded"] = df["graded"].astype(str).str.lower().eq("true")
    pending = int((~df["graded"]).sum())
    g = df[df["graded"]].copy()
    if g.empty:
        return {"n": 0, "pending": pending}
    g["correct"] = g["correct"].astype(str).str.lower().eq("true")
    g["asof_d"] = pd.to_datetime(g["asof"]).dt.date.astype(str)
    g["abs_err"] = (g["median_return"] - g["actual_return"]).abs()
    alert = cfg.get("signals", {}).get("alert_min_score", 75)

    def grp(col, min_n=3):
        rows = []
        for k, s in g.groupby(col):
            if len(s) >= min_n:
                rows.append({"key": k, "n": len(s), "hit_rate": float(s["correct"].mean()),
                             "avg_move": float((s["actual_return"] * np.where(s["direction"] == "up", 1, -1)).mean())})
        return sorted(rows, key=lambda r: -r["hit_rate"])

    # calibration: when agreement says X%, is it right X% of the time?
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    g["bucket"] = pd.cut(g["agreement"], bins, right=False, labels=["50-60", "60-70", "70-80", "80-90", "90-100"])
    calib = [{"bucket": str(b), "n": len(s), "predicted": float(s["agreement"].mean()), "actual": float(s["correct"].mean())}
             for b, s in g.groupby("bucket", observed=True) if len(s)]
    # signal decay: hit rate at each step ahead
    decay = []
    d = g[(g["interval"] == "1d") & g["step_median"].fillna("").astype(str).str.startswith("[") & g["actual"].fillna("").astype(str).str.startswith("[")]
    if not d.empty:
        steps = d.apply(lambda r: list(zip(json.loads(r["step_median"]), json.loads(r["actual"]))), axis=1)
        maxh = int(d["horizon"].max())
        for k in range(maxh):
            hits = [(pm > c0) == (ac > c0) for st, c0 in zip(steps, d["last_close"]) if len(st) > k for pm, ac in [st[k]]]
            if hits:
                decay.append({"step": k + 1, "n": len(hits), "hit_rate": float(np.mean(hits))})
    # regimes by VIX
    vix = pd.to_numeric(g["vix"], errors="coerce")
    g["regime"] = np.select([vix < 15, vix < 25, vix >= 25], ["calm (VIX<15)", "normal (15-25)", "wild (VIX 25+)"], "unknown")
    # signal types
    sig = []
    for f in ["breakout", "breakdown", "reversal_up", "reversal_down", "momentum_up", "momentum_down", "gap_up",
              "gap_down", "volume_surge", "volatility", "earnings"]:
        s = g[g["flags"].fillna("").str.split("|").apply(lambda x: f in x)]
        if len(s):
            sig.append({"signal": f, "n": len(s), "hit_rate": float(s["correct"].mean())})
    # streaks on high-score calls
    hs = g[g["score"] >= alert].sort_values(["asof", "ticker"])
    cur = longest_w = longest_l = run = 0
    last = None
    for ok in hs["correct"]:
        run = run + 1 if ok == last else 1
        last = ok
        if ok:
            longest_w = max(longest_w, run)
        else:
            longest_l = max(longest_l, run)
    cur = run if last is not None else 0
    # trade plans
    tp = g[g["outcome"].isin(["target", "stop", "time"]) & g["plan_valid"].astype(str).str.lower().eq("true")]
    plans = {"n": len(tp), "target": int((tp["outcome"] == "target").sum()), "stop": int((tp["outcome"] == "stop").sum()),
             "time": int((tp["outcome"] == "time").sum()),
             "no_fill": int((g["outcome"] == "no_fill").sum()),
             "avg_trade_return": float(tp["trade_return"].mean()) if len(tp) else None,
             "win_rate": float((tp["trade_return"] > 0).mean()) if len(tp) else None}
    daily = g.groupby("asof_d")["correct"].agg(["mean", "count"]).reset_index()
    daily["rolling"] = daily["mean"].rolling(10, min_periods=1).mean()
    cols = ["asof", "ticker", "interval", "direction", "agreement", "score", "median_return", "actual_return", "correct", "outcome"]
    return {
        "n": len(g), "pending": pending, "hit_rate": float(g["correct"].mean()),
        "hit_rate_high": _rate(g[g["score"] >= alert]["correct"]), "n_high": int((g["score"] >= alert).sum()),
        "median_abs_error": float(g["abs_err"].median()),
        "leaderboard": grp("ticker", 5), "by_sector": grp("sector"), "by_class": grp("cls"), "by_regime": grp("regime"),
        "by_model": grp("model"), "by_interval": grp("interval"), "calibration": calib, "decay": decay, "signals": sig,
        "streak": {"current": cur, "current_type": "win" if last else "loss", "longest_win": longest_w, "longest_loss": longest_l},
        "plans": plans,
        "daily": daily.rename(columns={"asof_d": "date", "mean": "hit_rate", "count": "n"}).to_dict("records"),
        "recent": g.sort_values("asof", ascending=False).head(40)[cols].to_dict("records"),
    }


def graded_on(date_str: str) -> list[dict]:
    """Predictions graded on a given day (for the after-close recap)."""
    df = load()
    if df.empty or "graded_ts" not in df.columns:
        return []
    g = df[df["graded_ts"].fillna("").astype(str).str.startswith(date_str)]
    return g.to_dict("records")


def hit_rate_overall() -> tuple[float | None, int]:
    """Gate for live trading: only the kind of call that gets traded (daily, US stock, long, valid plan, score 60+)."""
    df = load()
    if df.empty:
        return None, 0
    g = df[df["graded"].astype(str).str.lower().eq("true") & (df["cls"] == "stock") & (df["interval"] == "1d")
           & (df["side"] == "long") & df["plan_valid"].astype(str).str.lower().eq("true")
           & (pd.to_numeric(df["score"], errors="coerce") >= 60)]
    if g.empty:
        return None, 0
    return float(g["correct"].astype(str).str.lower().eq("true").mean()), len(g)
