"""Tickerbot market scanner: screens every US stock with SQL-style conditions (secret TICKERBOT_API_KEY).

Runs the scans listed in config.yaml -> tickerbot.scans, saves matches to site/data/tickerbot.json for the
dashboard's Market Scan tab, and keeps the signal catalog in state/tickerbot_signals.json so new scans can
be written against real signal names. Self-contained: needs only requests + app.core.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests

from .core import SITE_DATA, STATE_DIR, load_config, log, write_json

BASE = os.environ.get("TICKERBOT_API_URL", "https://api.tickerbot.io")


def call(method: str, path: str, params: dict | None = None, body: dict | None = None):
    """Returns (ok, json_or_text, status). Never raises, so one bad scan can't stop the rest."""
    key = os.environ.get("TICKERBOT_API_KEY")
    if not key:
        return False, "TICKERBOT_API_KEY secret is missing", 0
    try:
        r = requests.request(method, BASE + path, params=params, json=body, timeout=60,
                             headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                                      "User-Agent": "kronos-oracle"})
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", 0
    try:
        data = r.json()
    except ValueError:
        data = r.text[:500]
    return r.ok, data, r.status_code


def _err(data) -> str:
    if isinstance(data, dict):
        e = data.get("error", data)
        if isinstance(e, dict):
            return str(e.get("message") or e.get("code") or e)[:300]
        return str(e)[:300]
    return str(data)[:300]


def catalog() -> list[dict]:
    """Every signal name Tickerbot can screen on (paged)."""
    out, cursor = [], None
    for _ in range(40):
        ok, data, _s = call("GET", "/v2/signals", {"limit": 100, **({"cursor": cursor} if cursor else {})})
        if not ok or not isinstance(data, dict):
            break
        rows = data.get("results") or data.get("data") or data.get("signals") or []
        out += [{k: r.get(k) for k in ("name", "type", "category", "group", "description", "grain", "refresh") if k in r}
                for r in rows if isinstance(r, dict)]
        cursor = data.get("next_cursor")
        if not cursor or not rows:
            break
        time.sleep(0.3)
    return out


def run() -> dict:
    cfg = load_config()
    tc = cfg.get("tickerbot", {})
    out = {"generated": datetime.now(timezone.utc).isoformat(), "scans": [], "status": "ok"}
    ok, data, status = call("POST", "/v2/scan", body={"q": "price > 0", "universe": "top_10", "limit": 3})
    if not ok:
        out["status"] = f"Tickerbot refused the key or plan (HTTP {status}): {_err(data)}"
        log(out["status"])
        write_json(SITE_DATA / "tickerbot.json", out)
        return out
    out["plan_meta"] = data.get("_meta") if isinstance(data, dict) else None
    cat = catalog()
    if cat:
        write_json(STATE_DIR / "tickerbot_signals.json", cat)
    out["signals_available"] = len(cat)
    watch = [t.upper() for t in cfg.get("watchlist", [])]
    for sc in tc.get("scans", []):
        body = {k: sc[k] for k in ("q", "universe", "select", "order", "dir") if sc.get(k)}
        body["limit"] = int(sc.get("limit", 15))
        ok, data, status = call("POST", "/v2/scan", body=body)
        entry = {"name": sc["name"], "why": sc.get("why", ""), "q": sc["q"]}
        if ok and isinstance(data, dict):
            rows = data.get("results") or []
            for r in rows:
                if isinstance(r, dict):
                    r["on_watchlist"] = str(r.get("ticker", "")).upper() in watch
            entry.update(results=rows, total=data.get("total"), n=len(rows))
        else:
            entry.update(results=[], error=f"HTTP {status}: {_err(data)}")
            out["status"] = "some scans failed"
        out["scans"].append(entry)
        log(f"scan '{sc['name']}': {entry.get('n', 0)} matches {entry.get('error', '')}")
        time.sleep(0.5)
    if watch:  # snapshot of your own list with a few core signals
        sel = tc.get("watchlist_select", "ticker,price,change_1d_pct,relative_volume,rsi_14")
        ok, data, status = call("POST", "/v2/scan", body={"q": "ticker IN (" + ",".join(f"'{t}'" for t in watch) + ")",
                                                          "select": sel, "limit": 100, "order": "change_1d_pct"})
        out["watchlist"] = (data.get("results") or []) if ok and isinstance(data, dict) else []
        if not ok:
            out["watchlist_error"] = f"HTTP {status}: {_err(data)}"
    write_json(SITE_DATA / "tickerbot.json", out)
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
