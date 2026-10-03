"""Monthly fine-tune: trains a custom Kronos per ticker on its own history using the repo's
finetune_csv pipeline, then keeps it ONLY if it beats the stock model on recent unseen data.

Needs secrets HF_TOKEN + HF_USER (free Hugging Face account) to store the trained weights."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from .core import ROOT, load_config, log

CUSTOM_FILE = ROOT / "app" / "custom_models.yaml"


def retrain() -> None:
    cfg = load_config()
    rc = cfg.get("retrain", {})
    tickers = [t.upper() for t in rc.get("tickers", cfg["watchlist"][:3])]
    hf_token, hf_user = os.environ.get("HF_TOKEN"), os.environ.get("HF_USER")
    if not hf_token or not hf_user:
        log("HF_TOKEN / HF_USER secrets missing — retrain skipped")
        return
    from .backtest import backtest
    from .data import asset_class, fetch_history
    from .engine import Forecaster
    hist = fetch_history(tickers, "1d", cfg, period="max")
    custom = (yaml.safe_load(CUSTOM_FILE.read_text()) if CUSTOM_FILE.exists() else None) or {}
    m = cfg["model"]
    base = Forecaster.load(m["name"], m["tokenizer"], m["max_context"])
    for t in tickers:
        df = hist.get(t)
        if df is None or len(df) < 1500:
            log(f"{t}: not enough history to fine-tune")
            continue
        train = df.iloc[:-130]  # hold out the last ~6 months for the head-to-head
        work = Path(tempfile.mkdtemp())
        csv = work / f"{t}.csv"
        train.reset_index().to_csv(csv, index=False)
        conf = {
            "data": {"data_path": str(csv), "lookback_window": 400, "predict_window": 5, "max_context": 512, "clip": 5.0,
                     "train_ratio": 0.9, "val_ratio": 0.1, "test_ratio": 0.0},
            "training": {"tokenizer_epochs": 0, "basemodel_epochs": rc.get("epochs", 3), "batch_size": 16, "log_interval": 50,
                         "num_workers": 0, "seed": 42, "tokenizer_learning_rate": 2e-4,
                         "predictor_learning_rate": rc.get("learning_rate", 2e-6), "adam_beta1": 0.9, "adam_beta2": 0.95,
                         "adam_weight_decay": 0.1, "accumulation_steps": 1},
            "model_paths": {"pretrained_tokenizer": m["tokenizer"], "pretrained_predictor": m["name"], "exp_name": t,
                            "base_path": str(work / "out"), "base_save_path": "", "finetuned_tokenizer": m["tokenizer"],
                            "tokenizer_save_name": "tokenizer", "basemodel_save_name": "basemodel"},
            "experiment": {"name": f"oracle_{t}", "description": "monthly retrain", "use_comet": False,
                           "train_tokenizer": False, "train_basemodel": True, "skip_existing": False},
            "device": {"use_cuda": False, "device_id": 0},
        }
        cpath = work / "config.yaml"
        cpath.write_text(yaml.safe_dump(conf))
        log(f"{t}: fine-tuning…")
        r = subprocess.run([sys.executable, "train_sequential.py", "--config", str(cpath), "--skip-tokenizer"],
                           cwd=ROOT / "finetune_csv", capture_output=True, text=True)
        best = next((p for p in (work / "out").rglob("best_model") if "basemodel" in str(p)), None)
        if r.returncode != 0 or best is None:
            log(f"{t}: training failed\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
            continue
        # head-to-head on the held-out months
        tune_cfg = {**cfg, "backtest": {**cfg["backtest"], "days": 125}}
        from model import Kronos, KronosPredictor, KronosTokenizer
        tok = KronosTokenizer.from_pretrained(m["tokenizer"]).eval()
        mine = Forecaster(KronosPredictor(Kronos.from_pretrained(str(best)).eval(), tok, max_context=m["max_context"]), f"custom:{t}")
        cls = asset_class(t, cfg)
        b = backtest(t, df, base, tune_cfg, cls)
        c = backtest(t, df, mine, tune_cfg, cls)
        log(f"{t}: base hit {b['hit_rate']:.2%} vs custom {c['hit_rate']:.2%}")
        if (c["hit_rate"] or 0) <= (b["hit_rate"] or 0):
            log(f"{t}: custom model did not beat the base model — discarded")
            continue
        from huggingface_hub import HfApi
        repo = f"{hf_user}/kronos-oracle-{t.lower().replace('^', '').replace('=', '')}"
        api = HfApi(token=hf_token)
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(best), repo_id=repo)
        custom[t] = {"name": repo, "tokenizer": m["tokenizer"], "max_context": m["max_context"],
                     "hit_rate": c["hit_rate"], "base_hit_rate": b["hit_rate"]}
        CUSTOM_FILE.write_text(yaml.safe_dump(custom))
        log(f"{t}: custom model adopted -> {repo}")
