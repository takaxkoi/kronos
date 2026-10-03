"""Diagnostic: how do lookback length and batching change Kronos's daily forecasts? Writes state/diag.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
from app.core import load_config, write_json, STATE_DIR
from app.data import fetch_history, future_index
from model import Kronos, KronosTokenizer, KronosPredictor

cfg = load_config()
tick = ["AMD", "AAPL", "SPY", "NFLX", "META", "KO"]
hist = fetch_history(tick, "1d", cfg)
out = {}
for mname, tname, ctx in [("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
                          ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
                          ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048)]:
    tok = KronosTokenizer.from_pretrained(tname).eval(); mdl = Kronos.from_pretrained(mname).eval()
    pr = KronosPredictor(mdl, tok, device="cpu", max_context=ctx)
    for t in tick:
        df = hist[t]
        for lb in (400, 200, 100, 60):
            x = df.tail(lb)
            fut = future_index(x.index[-1], 5, "1d", "stock")
            cols = ["open", "high", "low", "close", "volume", "amount"]
            z = float((x["close"].iloc[-1] - x["close"].mean()) / x["close"].std())
            torch.manual_seed(0)
            single = pr.predict(x[cols].reset_index(drop=True), pd.Series(x.index), pd.Series(fut), pred_len=5,
                                T=1.0, top_p=0.9, sample_count=8, verbose=False)
            torch.manual_seed(0)
            batch = pr.predict_batch([x[cols].reset_index(drop=True)] * 8, [pd.Series(x.index)] * 8, [pd.Series(fut)] * 8,
                                     pred_len=5, T=1.0, top_p=0.9, sample_count=1, verbose=False)
            bc = np.median([b["close"].values for b in batch], axis=0)
            nov = pr.predict(x[["open", "high", "low", "close"]].reset_index(drop=True), pd.Series(x.index), pd.Series(fut),
                             pred_len=5, T=1.0, top_p=0.9, sample_count=8, verbose=False)
            c0 = float(x["close"].iloc[-1])
            out[f"{mname.split('/')[-1]}|{t}|{lb}"] = {
                "last": c0, "z_last": z, "mean": float(x["close"].mean()),
                "single_path": [round(v / c0 - 1, 4) for v in single["close"]],
                "batch_median": [round(v / c0 - 1, 4) for v in bc],
                "no_volume": [round(v / c0 - 1, 4) for v in nov["close"]]}
            print(mname, t, lb, "z", round(z, 2), "single", out[f"{mname.split('/')[-1]}|{t}|{lb}"]["single_path"][-1],
                  "batch", out[f"{mname.split('/')[-1]}|{t}|{lb}"]["batch_median"][-1],
                  "novol", out[f"{mname.split('/')[-1]}|{t}|{lb}"]["no_volume"][-1], flush=True)
write_json(STATE_DIR / "diag.json", out)
