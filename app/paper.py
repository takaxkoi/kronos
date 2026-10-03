"""Paper trading (fake money) + Alpaca auto-trading (paper by default, live only when the report card earns it)."""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from .core import STATE_DIR, log, read_json, write_json
from .ledger import hit_rate_overall

PAPER_FILE = STATE_DIR / "paper.json"


def _pick(analyses: list[dict], cfg: dict, allow_short: bool) -> list[dict]:
    out = []
    for a in analyses:
        if a["cls"] != "stock" or a.get("paused") or a.get("illiquid") or not a["plan"]["valid"]:
            continue
        if a["direction"] == "down" and not allow_short:
            continue
        if a["agreement"] < 0.65 or a["score"] < 60:
            continue
        out.append(a)
    return sorted(out, key=lambda a: -a["score"])


def paper_update(analyses: list[dict], hist: dict[str, pd.DataFrame], cfg: dict) -> dict:
    pc = cfg.get("paper", {})
    st = read_json(PAPER_FILE) or {"cash": pc.get("starting_cash", 100000), "start_value": pc.get("starting_cash", 100000),
                                    "positions": {}, "closed": [], "equity": []}
    # 1) manage open positions with the newest finished bar
    for t, pos in list(st["positions"].items()):
        df = hist.get(t)
        if df is None or df.empty:
            continue
        bars = df[df.index > pd.Timestamp(pos["last_bar"])]
        exit_px, why = None, None
        for ts, bar in bars.iterrows():
            pos["bars_held"] += 1
            pos["last_bar"] = ts.isoformat()
            long = pos["side"] == "long"
            if (bar["low"] <= pos["stop"]) if long else (bar["high"] >= pos["stop"]):
                exit_px = min(pos["stop"], float(bar["open"])) if long else max(pos["stop"], float(bar["open"]))
                why = "stop"; break
            if (bar["high"] >= pos["target"]) if long else (bar["low"] <= pos["target"]):
                exit_px, why = pos["target"], "target"; break
            if pos["bars_held"] >= pos["hold_bars"]:
                exit_px, why = float(bar["close"]), "time"; break
        if exit_px is not None:
            sh = pos["shares"]
            pnl = (exit_px - pos["entry"]) * sh if pos["side"] == "long" else (pos["entry"] - exit_px) * sh
            st["cash"] += exit_px * sh if pos["side"] == "long" else -exit_px * sh
            st["closed"].append({**pos, "ticker": t, "exit": exit_px, "exit_reason": why, "pnl": pnl,
                                 "return": pnl / (pos["entry"] * sh), "closed": pos["last_bar"]})
            del st["positions"][t]
    # 2) mark to market
    def value():
        v = st["cash"]
        for t, pos in st["positions"].items():
            px = float(hist[t]["close"].iloc[-1]) if t in hist else pos["entry"]
            pos["price"] = px
            v += px * pos["shares"] if pos["side"] == "long" else -px * pos["shares"]
        return v
    eq = value()
    # 3) open new positions from today's best setups
    slots = pc.get("max_positions", 10) - len(st["positions"])
    for a in _pick(analyses, cfg, pc.get("allow_short", False)):
        if slots <= 0:
            break
        t = a["ticker"]
        if t in st["positions"]:
            continue
        px = a["last_close"]
        sh = math.floor(eq * pc.get("position_pct", 0.10) / px)
        if sh <= 0 or (a["direction"] == "up" and sh * px > st["cash"]):
            continue
        side = "long" if a["direction"] == "up" else "short"
        st["cash"] += -sh * px if side == "long" else sh * px
        st["positions"][t] = {"side": side, "shares": sh, "entry": px, "stop": a["plan"]["stop"],
                              "target": a["plan"]["target"], "hold_bars": a["plan"]["hold_bars"], "bars_held": 0,
                              "opened": a["asof"], "last_bar": a["asof"], "score": a["score"], "price": px}
        slots -= 1
    eq = value()
    spy = float(hist["SPY"]["close"].iloc[-1]) if "SPY" in hist else None
    day = max((df.index[-1] for df in hist.values()), default=pd.Timestamp.now()).date().isoformat()
    st["equity"] = [e for e in st["equity"] if e["date"] != day] + [{"date": day, "equity": eq, "spy": spy}]
    closed = st["closed"]
    st["summary"] = {"equity": eq, "return": eq / st["start_value"] - 1, "open": len(st["positions"]),
                     "trades": len(closed), "win_rate": (sum(c["pnl"] > 0 for c in closed) / len(closed)) if closed else None,
                     "spy_return": (spy / st["equity"][0]["spy"] - 1) if spy and st["equity"][0].get("spy") else None}
    write_json(PAPER_FILE, st)
    return st


# ── Alpaca ───────────────────────────────────────────────────
def _alpaca(live: bool):
    key, sec = os.environ.get("ALPACA_KEY_ID"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        return None, None
    base = "https://api.alpaca.markets" if live else "https://paper-api.alpaca.markets"
    return base, {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def autotrade(analyses: list[dict], cfg: dict, generated: str | None = None) -> dict:
    ac = cfg.get("autotrade", {})
    if not ac.get("enabled"):
        return {"status": "off"}
    # never trade on stale forecasts (e.g. the daily scan failed last night)
    from .data import NYSE_DAY
    prev_session = (pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize() - NYSE_DAY)
    newest = max((pd.Timestamp(a["asof"]) for a in analyses if a.get("asof")), default=None)
    if newest is None or newest.normalize() < prev_session:
        return {"status": "blocked", "reason": f"forecasts are stale (newest {newest}, need {prev_session.date()})"}
    live = bool(ac.get("live"))
    if live:
        rate, n = hit_rate_overall()
        if os.environ.get("ALPACA_LIVE_CONFIRM") != "YES":
            return {"status": "blocked", "reason": "live trading needs secret ALPACA_LIVE_CONFIRM=YES"}
        if n < ac.get("min_graded", 100) or (rate or 0) < ac.get("min_hit_rate", 0.58):
            return {"status": "blocked", "reason": f"report card not earned yet ({n} graded, hit rate {rate})"}
    base, hdr = _alpaca(live)
    if not base:
        return {"status": "blocked", "reason": "ALPACA_KEY_ID / ALPACA_SECRET_KEY secrets missing"}
    try:
        acct = requests.get(f"{base}/v2/account", headers=hdr, timeout=20).json()
        held = {p["symbol"] for p in requests.get(f"{base}/v2/positions", headers=hdr, timeout=20).json()}
        open_orders = {o["symbol"] for o in requests.get(f"{base}/v2/orders?status=open", headers=hdr, timeout=20).json()}
        equity = float(acct.get("equity", 0))
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    slots = ac.get("max_positions", 5) - len(held | open_orders)
    placed = []
    for a in _pick(analyses, cfg, allow_short=False):
        if slots <= 0:
            break
        t = a["ticker"]
        if t in held or t in open_orders:
            continue
        pl = a["plan"]
        qty = math.floor(equity * ac.get("position_pct", 0.10) / a["last_close"])
        if qty <= 0:
            continue
        order = {"symbol": t, "qty": str(qty), "side": "buy", "type": "limit", "time_in_force": "day",
                 "limit_price": str(round(pl["entry"], 2)), "order_class": "bracket",
                 "take_profit": {"limit_price": str(round(pl["target"], 2))},
                 "stop_loss": {"stop_price": str(round(pl["stop"], 2))}}
        try:
            r = requests.post(f"{base}/v2/orders", json=order, headers=hdr, timeout=20)
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"message": r.text[:200]}
            placed.append({"ticker": t, "qty": qty, "ok": r.ok, "msg": body.get("id") if r.ok else body.get("message")})
        except Exception as e:
            placed.append({"ticker": t, "qty": qty, "ok": False, "msg": str(e)})
        slots -= 1
    log(f"autotrade ({'LIVE' if live else 'paper'}): {placed}")
    return {"status": "ok", "mode": "live" if live else "paper", "orders": placed,
            "time": datetime.now(timezone.utc).isoformat()}
