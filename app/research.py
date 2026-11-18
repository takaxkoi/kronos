"""AI research ratings (TradingAgents): a team of AI analysts researches each stock, debates it,
and a portfolio manager issues Buy / Overweight / Hold / Underweight / Sell with a written thesis.

Runs in its own GitHub job (workflow "AI research ratings") because TradingAgents needs newer
packages than the Kronos forecaster. Self-contained: imports only app.core + app.publish.

Needs ONE AI key as a GitHub secret. Checked in this order:
  GOOGLE_API_KEY (Gemini, free tier) · GROQ_API_KEY · OPENROUTER_API_KEY · ANTHROPIC_API_KEY · OPENAI_API_KEY

Every rating is logged with the price at the time, then graded after `holding_days` trading days
against SPY, so the dashboard shows whether the ratings are actually worth following.
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone

from .core import SITE_DATA, STATE_DIR, load_config, log, read_json, safe_name, write_json

LOG_FILE = STATE_DIR / "research_log.csv"
ROT_FILE = STATE_DIR / "research_rotation.json"
LOG_COLS = ["date", "ticker", "rating", "price", "target", "horizon", "model", "graded", "exit_date",
            "ret", "spy_ret", "alpha", "correct"]
BULLISH, BEARISH = {"Buy", "Overweight"}, {"Sell", "Underweight"}

# provider -> (env var, quick model, deep model). Override any of it in config.yaml -> research:
PROVIDERS = [
    ("google", "GOOGLE_API_KEY", "gemini-3.5-flash-lite", "gemini-3.8-flash"),
    ("groq", "GROQ_API_KEY", "llama-3.3-70b-versatile", "llama-3.3-70b-versatile"),
    ("openrouter", "OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct:free", "meta-llama/llama-3.3-70b-instruct:free"),
    ("anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5", "claude-sonnet-5"),
    ("openai", "OPENAI_API_KEY", "gpt-5.4-mini", "gpt-5.4"),
]


def pick_provider(rc: dict):
    want = rc.get("provider")
    for name, env, quick, deep in PROVIDERS:
        if (want and name != want) or not os.environ.get(env):
            continue
        return name, rc.get("quick_model") or quick, rc.get("deep_model") or deep
    return None


class Usage:
    """Counts AI calls and tokens so we know how much of a free tier each stock uses."""

    def __init__(self):
        from langchain_core.callbacks import BaseCallbackHandler

        outer = self

        class _H(BaseCallbackHandler):
            def on_llm_end(self, response, **kw):
                outer.calls += 1
                try:
                    for gens in response.generations:
                        for g in gens:
                            um = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                            outer.tokens_in += int(um.get("input_tokens", 0) or 0)
                            outer.tokens_out += int(um.get("output_tokens", 0) or 0)
                except Exception:
                    pass

            def on_tool_end(self, output, **kw):
                outer.tools += 1

        self.calls = self.tools = self.tokens_in = self.tokens_out = 0
        self.handler = _H()

    def snapshot(self):
        return {"ai_calls": self.calls, "tool_calls": self.tools, "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


def _field(md: str, name: str) -> str:
    m = re.search(rf"\*\*{name}\*\*\s*:\s*(.*?)(?=\n\s*\*\*[A-Z][A-Za-z ]+\*\*\s*:|\Z)", md or "", re.S)
    return m.group(1).strip() if m else ""


def _num(s: str):
    m = re.search(r"-?\d[\d,]*\.?\d*", s or "")
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def _closes(tickers: list[str], period: str = "6mo"):
    import yfinance as yf
    out = {}
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period=period, auto_adjust=True)["Close"].dropna()
            if len(h):
                h.index = h.index.tz_localize(None).normalize()
                out[t] = h
        except Exception as e:
            log(f"price fetch failed for {t}: {e}")
    return out


def read_log() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as fh:
        return list(csv.DictReader(fh))


def write_log(rows: list[dict]) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def grade(rows: list[dict], holding: int) -> int:
    """Score ratings whose holding window has finished: stock return vs SPY from the first close after the rating."""
    import pandas as pd
    todo = [r for r in rows if str(r.get("graded")).lower() != "true"]
    if not todo:
        return 0
    px = _closes(sorted({r["ticker"] for r in todo} | {"SPY"}))
    n = 0
    for r in todo:
        s, spy = px.get(r["ticker"]), px.get("SPY")
        if s is None or spy is None:
            continue
        after = s[s.index > pd.Timestamp(r["date"])]
        if len(after) <= holding:
            continue
        start, end = after.index[0], after.index[holding]
        ret = float(after.iloc[holding] / after.iloc[0] - 1)
        sp = spy[(spy.index >= start) & (spy.index <= end)]
        if len(sp) < 2:
            continue
        spy_ret = float(sp.iloc[-1] / sp.iloc[0] - 1)
        alpha = ret - spy_ret
        correct = "" if r["rating"] not in BULLISH | BEARISH else str((alpha > 0) == (r["rating"] in BULLISH))
        r.update(graded="True", exit_date=end.date().isoformat(), ret=round(ret, 5), spy_ret=round(spy_ret, 5),
                 alpha=round(alpha, 5), correct=correct)
        n += 1
    return n


def scorecard(rows: list[dict]) -> dict:
    g = [r for r in rows if str(r.get("graded")).lower() == "true"]
    by = {}
    for r in g:
        b = by.setdefault(r["rating"], {"n": 0, "ret": 0.0, "alpha": 0.0})
        b["n"] += 1
        b["ret"] += float(r["ret"])
        b["alpha"] += float(r["alpha"])
    calls = [r for r in g if r.get("correct") in ("True", "False")]
    return {"graded": len(g), "pending": len(rows) - len(g),
            "hit_rate": (sum(r["correct"] == "True" for r in calls) / len(calls)) if calls else None, "n_calls": len(calls),
            "by_rating": [{"rating": k, "n": v["n"], "avg_return": v["ret"] / v["n"], "avg_alpha": v["alpha"] / v["n"]}
                          for k, v in by.items()],
            "recent": sorted(g, key=lambda r: r["date"], reverse=True)[:30]}


def run(tickers: list[str] | None = None) -> None:
    cfg = load_config()
    rc = cfg.get("research", {})
    rows = read_log()
    holding = int(rc.get("holding_days", 5))
    try:
        log(f"graded {grade(rows, holding)} earlier ratings")
    except Exception as e:
        log(f"grading skipped: {e}")

    chosen = pick_provider(rc)
    index = read_json(SITE_DATA / "research.json", {"ratings": {}})
    index.setdefault("ratings", {})
    if not chosen:
        index["status"] = "No AI key found. Add GOOGLE_API_KEY (free at aistudio.google.com/apikey) as a GitHub secret."
        log(index["status"])
    else:
        provider, quick, deep = chosen
        os.environ.setdefault("TRADINGAGENTS_RESULTS_DIR", "/tmp/ta_results")
        os.environ.setdefault("TRADINGAGENTS_CACHE_DIR", "/tmp/ta_cache")
        os.environ.setdefault("TRADINGAGENTS_MEMORY_LOG_PATH", str(STATE_DIR / "research_memory.md"))
        from tradingagents.default_config import build_default_config
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        tcfg = build_default_config()
        tcfg.update(llm_provider=provider, quick_think_llm=quick, deep_think_llm=deep,
                    max_debate_rounds=int(rc.get("debate_rounds", 1)), max_risk_discuss_rounds=int(rc.get("risk_rounds", 1)),
                    llm_max_retries=int(rc.get("llm_retries", 6)))
        analysts = tuple(rc.get("analysts", ["market", "social", "news", "fundamentals"]))

        watch = [t.upper() for t in cfg["watchlist"]]
        if not tickers:  # rotate through the watchlist a few at a time to stay inside free limits
            k = int(rc.get("per_run", 3))
            rot = read_json(ROT_FILE, {"i": 0})
            tickers = [watch[(rot["i"] + j) % len(watch)] for j in range(min(k, len(watch)))]
            write_json(ROT_FILE, {"i": (rot["i"] + len(tickers)) % len(watch)})
        today = datetime.now(timezone.utc).date().isoformat()
        prices = _closes(tickers, "5d")
        pause = float(rc.get("pause_seconds", 20))
        errors = []
        for n, t in enumerate(tickers):
            use, t0 = Usage(), time.time()
            try:
                log(f"researching {t} with {provider}:{deep}")
                ta = TradingAgentsGraph(selected_analysts=analysts, debug=False, config=tcfg, callbacks=[use.handler])
                state, rating = ta.propagate(t, today)
                md = state.get("final_trade_decision", "")
                price = float(prices[t].iloc[-1]) if t in prices else None
                target = _num(_field(md, "Price Target"))
                rec = {"ticker": t, "date": today, "rating": rating, "price": price, "target": target,
                       "upside": (target / price - 1) if target and price else None,
                       "horizon": _field(md, "Time Horizon"), "summary": _field(md, "Executive Summary"),
                       "model": f"{provider}:{deep}", "seconds": round(time.time() - t0), "usage": use.snapshot()}
                index["ratings"][t] = rec
                write_json(SITE_DATA / "research" / f"{safe_name(t)}.json", {
                    **rec, "thesis": _field(md, "Investment Thesis"), "decision": md,
                    "reports": {"Market & chart": state.get("market_report", ""), "Fundamentals": state.get("fundamentals_report", ""),
                                "News": state.get("news_report", ""), "Sentiment": state.get("sentiment_report", "")},
                    "bull_case": (state.get("investment_debate_state") or {}).get("bull_history", ""),
                    "bear_case": (state.get("investment_debate_state") or {}).get("bear_history", ""),
                    "research_plan": state.get("investment_plan", ""), "trader_plan": state.get("trader_investment_plan", ""),
                    "risk_debate": (state.get("risk_debate_state") or {}).get("history", "")})
                rows = [r for r in rows if not (r["ticker"] == t and r["date"] == today)]
                rows.append({"date": today, "ticker": t, "rating": rating, "price": price or "", "target": target or "",
                             "horizon": rec["horizon"], "model": rec["model"], "graded": "False"})
                log(f"{t}: {rating} in {rec['seconds']}s, usage {rec['usage']}")
            except Exception as e:
                errors.append({"ticker": t, "error": f"{type(e).__name__}: {str(e)[:400]}", "usage": use.snapshot()})
                log(f"{t} FAILED after {use.calls} AI calls: {type(e).__name__}: {str(e)[:600]}")
                traceback.print_exc()
                if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                    log("AI provider limit hit; stopping this run")
                    break
            if n < len(tickers) - 1:
                time.sleep(pause)
        index["status"] = "ok" if not errors else f"{len(errors)} of {len(tickers)} failed"
        index["last_errors"] = errors
        index["model"] = f"{provider}:{deep}"

    write_log(rows)
    index["generated"] = datetime.now(timezone.utc).isoformat()
    index["holding_days"] = holding
    index["scorecard"] = scorecard(rows)
    write_json(SITE_DATA / "research.json", index)
    try:
        from .publish import publish
        publish()
    except Exception as e:
        log(f"publish failed: {e}")


if __name__ == "__main__":
    run([a.upper() for a in sys.argv[1:] if re.fullmatch(r"[A-Za-z0-9.^=\-]{1,15}", a)] or None)
