"""Offline end-to-end test of the Oracle app.

Uses a tiny randomly-initialised Kronos and simulated prices so it runs anywhere with no
internet. The numbers it produces are meaningless — it only proves the plumbing works.
Run:  python -m pytest tests/test_oracle.py -q     (or: python tests/test_oracle.py)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(os.environ.get("ORACLE_TEST_DIR") or tempfile.mkdtemp())
os.environ["KRONOS_STATE_DIR"] = str(TMP / "state")
os.environ["KRONOS_SITE_DIR"] = str(TMP / "site")
os.environ.pop("TELEGRAM_BOT_TOKEN", None)
os.environ["PORTFOLIO_HOLDINGS"] = "AAPL:10,MSFT:5"
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from app import data as D  # noqa: E402
from app import engine as E  # noqa: E402
from app import scan as S  # noqa: E402
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402


def tiny_forecaster(name="test/tiny"):
    torch.manual_seed(0)
    tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1, ffn_dropout_p=0,
                          attn_dropout_p=0, resid_dropout_p=0, s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1,
                          zeta=0.05, group_size=4).eval()
    mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0, attn_dropout_p=0,
                 resid_dropout_p=0, token_dropout_p=0, learn_te=True).eval()
    return E.Forecaster(KronosPredictor(mdl, tok, device="cpu", max_context=512), name)


def fake_history(tickers, interval="1d", cfg=None, period=None):
    out = {}
    n = {"1d": 700, "1h": 500, "1wk": 450}[interval]
    for i, t in enumerate(tickers):
        rng = np.random.default_rng(abs(hash(t)) % 2**32)
        cls = D.asset_class(t, cfg)
        if interval == "1d":
            idx = (pd.date_range("2023-01-02", periods=n, freq="D") if cls == "crypto"
                   else pd.date_range("2023-01-03", periods=n, freq=D.NYSE_DAY))
        elif interval == "1wk":
            idx = pd.date_range("2017-01-02", periods=n, freq="W-MON")
        else:
            idx = D.future_index(pd.Timestamp("2025-01-02 15:30"), n, "1h", cls)
        price = 50 + 150 * rng.random()
        r = rng.normal(0.0004, 0.018, n)
        c = price * np.exp(np.cumsum(r))
        o = np.r_[price, c[:-1]] * (1 + rng.normal(0, 0.003, n))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.008, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.008, n)))
        v = rng.integers(2e6, 9e6, n).astype(float) * (0 if cls == "forex" else 1)
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
        out[t.upper()] = D._standardize(df, cls)
    return out


def patch():
    tiny = tiny_forecaster()
    E.Forecaster.load = classmethod(lambda cls, *a, **k: tiny)
    for mod in (S, D):
        mod.fetch_history = fake_history
    import app.ledger as L
    L.fetch_history = fake_history
    rows = [{"ticker": t, "name": f"{t} Inc", "sector": s} for t, s in
            [("AAPL", "Information Technology"), ("MSFT", "Information Technology"), ("XOM", "Energy"),
             ("JPM", "Financials"), ("PFE", "Health Care"), ("KO", "Consumer Staples")]]
    S.sp500 = lambda: rows
    S.sector_map = lambda tickers, cfg: {t: "Information Technology" for t in tickers}
    nv_end = fake_history(["NVDA"])["NVDA"].index[-1]
    S.next_earnings = lambda t: (nv_end + pd.Timedelta(days=2)).date().isoformat() if t == "NVDA" else None
    S.headlines = lambda t: [{"title": f"{t} headline", "url": "https://example.com", "time": None, "source": "Test"}]
    S.option_idea = lambda *a, **k: {"type": "CALL", "expiry": "2026-11-20", "strike": 200.0, "bid": 3.1, "ask": 3.3,
                                     "mid": 3.2, "iv": 0.31, "open_interest": 1200, "breakeven": 203.2,
                                     "target_beats_breakeven": True}


def small_config(tmp: Path):
    import yaml
    cfg = yaml.safe_load((ROOT / "app" / "config.yaml").read_text())
    cfg["watchlist"] = ["AAPL", "MSFT", "NVDA", "TSLA"]
    cfg["leveraged"] = ["TQQQ"]
    cfg["crypto"] = ["BTC-USD", "ETH-USD"]
    cfg["forex"] = {"EURUSD=X": "EUR/USD"}
    cfg["commodities"] = {"GC=F": "Gold"}
    cfg["sectors"] = {"XLK": "Technology", "XLE": "Energy", "XLF": "Financials"}
    cfg["forecast"].update({"samples": 8, "lookback": 200, "batch_size": 128})
    cfg["backtest"].update({"days": 60, "samples": 4})
    cfg["paper"]["starting_cash"] = 100000
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    os.environ["KRONOS_CONFIG"] = str(p)
    import app.core as C
    C.CONFIG_PATH = p
    import app.bot as B
    B.CONFIG_PATH = p


def test_end_to_end():
    TMP.mkdir(parents=True, exist_ok=True)
    small_config(TMP)
    patch()
    S.sp500_scan()
    S.daily_scan()
    S.crypto_hourly()
    S.premarket()
    S.backtest_lab(["AAPL", "SPY"], models=True)
    S.add_journal("BUY", "AAPL", 10, 190.5, "test")
    from app import bot
    for cmd in ["/help", "/top", "/mood", "/sectors", "/report", "/paper", "/crypto", "/watch list", "/forecast AAPL",
                "/portfolio", "/replay AAPL 2024-06-03", "/watch add AMD"]:
        bot.handle(cmd)
    site = TMP / "site"
    latest = json.loads((site / "latest.json").read_text())
    assert latest["watchlist"] and latest["indices"] and latest["mood"]
    a = latest["watchlist"][0]
    assert 0 <= a["score"] <= 100 and 0.5 <= a["agreement"] <= 1
    assert a["plan"]["stop"] < a["plan"]["entry"] < a["plan"]["target"] or a["plan"]["side"] == "short"
    det = json.loads((site / "t" / "AAPL.json").read_text())
    assert len(det["chart"]["future"]) == 5 and "chart_weekly" in det and "chart_hourly" in det
    assert det["timeframes"].keys() == {"1d", "1wk", "1h"}
    assert (site / "scanner.json").exists() and (site / "crypto_hourly.json").exists()
    assert (site / "bt" / "index.json").exists() and (site / "paper.json").exists()
    nvda = json.loads((site / "t" / "NVDA.json").read_text())
    assert nvda["paused"] and "earnings" in nvda["flags"]
    # future calendar sanity
    fi = D.future_index(pd.Timestamp("2025-12-24"), 3, "1d", "stock")
    assert pd.Timestamp("2025-12-25") not in fi
    hi = D.future_index(pd.Timestamp("2025-01-03 15:30"), 3, "1h", "stock")
    assert hi[0] == pd.Timestamp("2025-01-06 09:30")
    # grading: pretend time passed by grading against longer fake history
    from app import ledger
    assert len(ledger.load()) > 0

    def extended(tickers, interval="1d", cfg=None, period=None):
        out = {}
        for t, df in fake_history(tickers, interval, cfg).items():
            cls = D.asset_class(t, cfg)
            fut = D.future_index(df.index[-1], 30, interval, cls)
            rng = np.random.default_rng(1)
            c = df["close"].iloc[-1] * np.exp(np.cumsum(rng.normal(0, 0.02, 30)))
            ext = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6, "amount": 1e6 * c}, index=fut)
            out[t] = pd.concat([df, ext])
        return out
    ledger.fetch_history = extended
    import yaml
    n = ledger.grade(yaml.safe_load(open(os.environ["KRONOS_CONFIG"])))
    rep = ledger.report(yaml.safe_load(open(os.environ["KRONOS_CONFIG"])))
    assert n > 0 and rep["n"] == n and rep["calibration"] and rep["decay"]
    (TMP / "site" / "report.json").write_text(json.dumps(__import__("app.core", fromlist=["clean"]).clean(rep)))
    # trade-plan simulation: no fill, gap through stop, target
    bars = pd.DataFrame({"open": [101, 100, 95], "high": [102, 101, 96], "low": [100.5, 99, 94], "close": [101, 99.5, 95]})
    assert ledger.simulate_plan(bars.iloc[:1], True, 100, 98, 110)[0] == "no_fill"
    out, r = ledger.simulate_plan(bars, True, 100, 98, 110)
    assert out == "stop" and abs(r - (95 / 100 - 1)) < 1e-9  # gapped below the stop -> exit at the open
    # live trading gates
    from app import paper
    cfg = yaml.safe_load(open(os.environ["KRONOS_CONFIG"]))
    cfg["autotrade"].update({"enabled": True, "live": True})
    a = {"asof": pd.Timestamp.now().isoformat()}
    os.environ.pop("ALPACA_LIVE_CONFIRM", None)
    assert paper.autotrade([a], cfg)["status"] == "blocked"
    os.environ["ALPACA_LIVE_CONFIRM"] = "YES"
    assert "report card" in paper.autotrade([a], cfg)["reason"]
    assert "stale" in paper.autotrade([{"asof": "2020-01-02"}], cfg)["reason"]
    os.environ.pop("ALPACA_LIVE_CONFIRM", None)
    from app import bot
    bot.handle("/watch add ]")
    assert "]" not in yaml.safe_load(open(os.environ["KRONOS_CONFIG"]))["watchlist"]
    print("OK —", TMP)


if __name__ == "__main__":
    test_end_to_end()
