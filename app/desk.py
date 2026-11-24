"""Research desk (OpenBB): market movers, the Treasury yield curve and the latest SEC filings for
your watchlist, pulled through the OpenBB platform from sources that need no key (Yahoo, SEC, Federal Reserve).
Every call is wrapped so one failing source never blanks the page; failures are listed in the output.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .core import SITE_DATA, load_config, log, write_json


def _records(res, limit: int = 25) -> list[dict]:
    df = res.to_df().reset_index()
    df = df.loc[:, ~df.columns.duplicated()].head(limit)
    out = []
    for row in df.to_dict("records"):
        clean = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and v != v):
                continue
            clean[str(k)] = v.isoformat() if hasattr(v, "isoformat") else (v if isinstance(v, (int, float, bool, str)) else str(v))
        out.append(clean)
    return out


def run() -> dict:
    cfg = load_config()
    out = {"generated": datetime.now(timezone.utc).isoformat(), "errors": [], "movers": {}, "filings": [], "curve": [], "targets": []}
    try:
        from openbb import obb
    except Exception as e:
        out["status"] = f"OpenBB did not load: {type(e).__name__}: {str(e)[:200]}"
        write_json(SITE_DATA / "desk.json", out)
        return out

    log('desk: OpenBB sections: ' + ', '.join(a for a in dir(obb) if not a.startswith('_')))

    try:
        cmds = sorted(obb.coverage.commands)
        for pre in (".yfinance", ".sec", ".ustreasury", ".federal_reserve", ".news", ".finra", ".nasdaq"):
            log("desk: commands " + pre + ": " + ", ".join(c for c in cmds if c.startswith(pre)))
    except Exception as e:
        log(f"desk: coverage failed: {e}")

    def tryit(label, fn):
        try:
            return _records(fn())
        except Exception as e:
            out["errors"].append(f"{label}: {type(e).__name__}: {str(e)[:160]}")
            log(f"desk: {label} failed: {type(e).__name__}: {str(e)[:200]}")
            return []

    def first(label, *paths):
        """OpenBB moved commands between versions; try each known location."""
        for path, kw in paths:
            fn = obb
            try:
                for part in path.split("."):
                    fn = getattr(fn, part)
            except AttributeError:
                continue
            rows = tryit(f"{label} via {path}", lambda fn=fn, kw=kw: fn(**kw))
            if rows:
                return rows
        return []

    for kind in ("gainers", "losers", "active"):
        out["movers"][kind] = first(kind, (f"equity.discovery.{kind}", {"provider": "yfinance"}), (f"yfinance.equity.discovery.{kind}", {}),
                                    ("nasdaq.markets.movers", {"category": {"gainers": "most_advanced", "losers": "most_declined", "active": "most_active_dollar_volume"}[kind]}))[:15]
    out["targets"] = []
    for t in [x.upper() for x in cfg.get("watchlist", [])]:
        rows = first(f"consensus {t}", ("equity.estimates.consensus", {"symbol": t, "provider": "yfinance"}), ("yfinance.equity.estimates.consensus", {"symbol": t}))
        if rows:
            out["targets"].append({"ticker": t, **{k: v for k, v in rows[0].items() if k not in ("index", "symbol", "name")}})
    out["filings"].sort(key=lambda r: r["date"], reverse=True)
    out["status"] = "ok" if not out["errors"] else f"{len(out['errors'])} source calls failed"
    write_json(SITE_DATA / "desk.json", out)
    log(f"desk: {sum(len(v) for v in out['movers'].values())} movers, {len(out['filings'])} filings, {len(out['curve'])} curve points")
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
