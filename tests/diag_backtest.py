"""Walk-forward bake-off of model x lookback on real daily data. Writes state/diag_bt.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from app.core import load_config, write_json, STATE_DIR
from app.data import fetch_history
from app.engine import Forecaster, Job

cfg = load_config()
tick = ["AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "JPM", "KO", "XOM", "SPY", "QQQ", "NFLX"]
hist = fetch_history(tick, "1d", cfg)
H, DAYS, STEP, S = 5, 260, 5, 8
configs = [("mini", "NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048, 400),
           ("mini", "NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048, 200),
           ("small", "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512, 60),
           ("small", "NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512, 120),
           ("base", "NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512, 60)]
out = {}
for short, name, tok, ctx, lb in configs:
    t0 = time.time()
    fc = Forecaster.load(name, tok, ctx)
    rows = []
    for t in tick:
        df = hist.get(t)
        if df is None:
            continue
        n = len(df)
        asofs = list(range(n - DAYS, n - H + 1, STEP))
        res = fc.run([Job(t, df, H, "1d", "stock", asof=i) for i in asofs], samples=S, lookback=lb, batch_size=64,
                     seed=1, min_bars=min(lb, 60))
        c = df["close"].values
        for i in asofs:
            p = res.get(f"{t}@{i}")
            if p is None:
                continue
            fin = p.arr[:, -1, 3] / c[i - 1] - 1
            pred, act = float(np.median(fin)), float(c[i + H - 1] / c[i - 1] - 1)
            agree = max(np.mean(fin > 0), np.mean(fin <= 0))
            rows.append((t, pred, act, agree))
    pred = np.array([r[1] for r in rows]); act = np.array([r[2] for r in rows]); ag = np.array([r[3] for r in rows])
    hit = (np.sign(pred) == np.sign(act))
    hi = ag >= 0.75
    key = f"{short}-lb{lb}"
    out[key] = {"n": len(rows), "hit_rate": float(hit.mean()), "hit_rate_conf75": float(hit[hi].mean()) if hi.any() else None,
                "n_conf75": int(hi.sum()), "mae": float(np.mean(np.abs(pred - act))), "bias": float(pred.mean() - act.mean()),
                "avg_abs_pred": float(np.mean(np.abs(pred))), "avg_abs_actual": float(np.mean(np.abs(act))),
                "pct_pred_up": float((pred > 0).mean()), "pct_actual_up": float((act > 0).mean()),
                "corr": float(np.corrcoef(pred, act)[0, 1]), "secs": round(time.time() - t0)}
    print(key, out[key], flush=True)
    write_json(STATE_DIR / "diag_bt.json", out)
