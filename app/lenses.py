"""Investor lenses: scores every S&P 500 stock the way five different schools of investing would,
using the fundamentals the data vault saves each day. Rule-based, no AI key needed.

Value · Quality · Growth at a fair price · Trend · Wall Street view
Each lens is a short checklist; the score is the share of checks passed.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .core import ROOT, SITE_DATA, load_config, log, write_json


def _pct(v):
    return f"{v * 100:.0f}%"


LENSES = {
    "Value": ("Is it cheap against what the business earns and owns?", [
        ("P/E under 20", lambda r: r.trailingPE < 20, lambda r: f"P/E {r.trailingPE:.1f}"),
        ("Forward P/E under 16", lambda r: r.forwardPE < 16, lambda r: f"fwd P/E {r.forwardPE:.1f}"),
        ("Price/book under 3", lambda r: r.priceToBook < 3, lambda r: f"P/B {r.priceToBook:.1f}"),
        ("EV/EBITDA under 12", lambda r: r.enterpriseToEbitda < 12, lambda r: f"EV/EBITDA {r.enterpriseToEbitda:.1f}"),
        ("Pays a dividend", lambda r: r.dividendYield > 0, lambda r: f"yield {r.dividendYield:.2f}%"),
    ]),
    "Quality": ("Is it a durable, profitable business with little debt?", [
        ("Return on equity over 15%", lambda r: r.returnOnEquity > 0.15, lambda r: f"ROE {_pct(r.returnOnEquity)}"),
        ("Operating margin over 15%", lambda r: r.operatingMargins > 0.15, lambda r: f"op margin {_pct(r.operatingMargins)}"),
        ("Net margin over 10%", lambda r: r.profitMargins > 0.10, lambda r: f"net margin {_pct(r.profitMargins)}"),
        ("Debt/equity under 80", lambda r: r.debtToEquity < 80, lambda r: f"D/E {r.debtToEquity:.0f}"),
        ("Positive free cash flow", lambda r: r.freeCashflow > 0, lambda r: f"FCF ${r.freeCashflow / 1e9:.1f}B"),
        ("Revenue growing", lambda r: r.revenueGrowth > 0, lambda r: f"revenue {_pct(r.revenueGrowth)}"),
    ]),
    "Growth at a fair price": ("Is it growing fast without being priced for perfection?", [
        ("Earnings growth over 15%", lambda r: r.earningsGrowth > 0.15, lambda r: f"earnings {_pct(r.earningsGrowth)}"),
        ("Revenue growth over 10%", lambda r: r.revenueGrowth > 0.10, lambda r: f"revenue {_pct(r.revenueGrowth)}"),
        ("PEG under 1.5", lambda r: 0 < r.forwardPE / (r.earningsGrowth * 100) < 1.5, lambda r: f"PEG {r.forwardPE / (r.earningsGrowth * 100):.2f}"),
        ("Forward P/E below trailing", lambda r: r.forwardPE < r.trailingPE, lambda r: f"{r.forwardPE:.0f} vs {r.trailingPE:.0f}"),
        ("Debt/equity under 100", lambda r: r.debtToEquity < 100, lambda r: f"D/E {r.debtToEquity:.0f}"),
    ]),
    "Trend": ("Is the price already moving the right way?", [
        ("Above 200-day average", lambda r: r.currentPrice > r.twoHundredDayAverage, lambda r: f"{_pct(r.currentPrice / r.twoHundredDayAverage - 1)} vs 200d"),
        ("Above 50-day average", lambda r: r.currentPrice > r.fiftyDayAverage, lambda r: f"{_pct(r.currentPrice / r.fiftyDayAverage - 1)} vs 50d"),
        ("50-day above 200-day", lambda r: r.fiftyDayAverage > r.twoHundredDayAverage, lambda r: "golden cross" if r.fiftyDayAverage > r.twoHundredDayAverage else "death cross"),
        ("Within 15% of 52-week high", lambda r: r.currentPrice > 0.85 * r.fiftyTwoWeekHigh, lambda r: f"{_pct(r.currentPrice / r.fiftyTwoWeekHigh - 1)} from high"),
    ]),
    "Wall Street view": ("What do analysts and short sellers think?", [
        ("Analyst rating Buy or better", lambda r: r.recommendationMean < 2.3, lambda r: f"rating {r.recommendationMean:.1f} (1 = strong buy)"),
        ("Target 10%+ above price", lambda r: r.targetMeanPrice > 1.10 * r.currentPrice, lambda r: f"target {_pct(r.targetMeanPrice / r.currentPrice - 1)}"),
        ("Covered by 10+ analysts", lambda r: r.numberOfAnalystOpinions >= 10, lambda r: f"{r.numberOfAnalystOpinions:.0f} analysts"),
        ("Short interest under 5% of float", lambda r: r.shortPercentOfFloat < 0.05, lambda r: f"short {r.shortPercentOfFloat * 100:.1f}%"),
    ]),
}


def score_row(r) -> dict:
    out, total = {}, []
    for lens, (_, checks) in LENSES.items():
        passed, known, notes = 0, 0, []
        for label, test, show in checks:
            try:
                ok = bool(test(r))
                txt = show(r)
                if "nan" in txt.lower():
                    continue
            except Exception:
                continue
            known += 1
            passed += ok
            notes.append({"check": label, "ok": ok, "value": txt})
        sc = round(100 * passed / known) if known else None
        out[lens] = {"score": sc, "passed": passed, "of": known, "checks": notes}
        if sc is not None:
            total.append(sc)
    out["overall"] = round(sum(total) / len(total)) if total else None
    return out


def run() -> dict:
    cfg = load_config()
    days = sorted((ROOT / "data" / "daily").glob("*/fundamentals.csv.gz"))
    out = {"generated": datetime.now(timezone.utc).isoformat(), "lenses": {k: v[0] for k, v in LENSES.items()}}
    if not days:
        out["status"] = "No fundamentals captured yet. Run Actions → Data vault (daily capture)."
        write_json(SITE_DATA / "lenses.json", out)
        return out
    df = pd.read_csv(days[-1])
    out["asof"] = days[-1].parent.name
    rows = []
    for r in df.itertuples(index=False):
        s = score_row(r)
        rows.append({"ticker": r.ticker, "sector": getattr(r, "sector", None) if isinstance(getattr(r, "sector", None), str) else None,
                     "price": float(r.currentPrice) if r.currentPrice == r.currentPrice else None, **s})
    watch = [t.upper() for t in cfg.get("watchlist", [])]
    by = {r["ticker"]: r for r in rows}
    out["watchlist"] = [by[t] for t in watch if t in by]
    out["top"] = {}
    for lens in LENSES:
        ranked = sorted([r for r in rows if r[lens]["score"] is not None and r[lens]["of"] >= 3],
                        key=lambda r: (r[lens]["score"], r["overall"] or 0), reverse=True)[:12]
        out["top"][lens] = [{"ticker": r["ticker"], "sector": r["sector"], "score": r[lens]["score"], "overall": r["overall"],
                             "why": ", ".join(c["value"] for c in r[lens]["checks"] if c["ok"])[:140]} for r in ranked]
    allr = sorted([r for r in rows if all(r[k]["score"] is not None for k in LENSES)], key=lambda r: r["overall"], reverse=True)[:15]
    out["top"]["All five lenses"] = [{"ticker": r["ticker"], "sector": r["sector"], "score": r["overall"], "overall": r["overall"],
                                     "why": " · ".join(f"{k} {r[k]['score']}" for k in LENSES if r[k]["score"] is not None)} for r in allr]
    out["status"] = "ok"
    out["stocks"] = len(rows)
    write_json(SITE_DATA / "lenses.json", out)
    log(f"lenses: scored {len(rows)} stocks as of {out['asof']}")
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")
    return out


if __name__ == "__main__":
    run()
