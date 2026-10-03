"""Shared settings, paths and small helpers."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATE_DIR = Path(os.environ.get("KRONOS_STATE_DIR", ROOT / "state"))
SITE_DATA = Path(os.environ.get("KRONOS_SITE_DIR", ROOT / "site" / "data"))
CONFIG_PATH = Path(os.environ.get("KRONOS_CONFIG", ROOT / "app" / "config.yaml"))


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    custom = ROOT / "app" / "custom_models.yaml"
    if custom.exists():
        cfg.setdefault("model", {})["custom"] = yaml.safe_load(custom.read_text()) or {}
    holdings = os.environ.get("PORTFOLIO_HOLDINGS", "").strip()
    cfg["holdings"] = parse_holdings(holdings)
    return cfg


def parse_holdings(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.replace(";", ",").split(","):
        if ":" in part:
            t, q = part.split(":", 1)
            try:
                out[t.strip().upper()] = float(q)
            except ValueError:
                continue
    return out


def clean(obj: Any) -> Any:
    """Make numpy / pandas values JSON-safe (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(clean(data), f, separators=(",", ":"))
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def log(*args) -> None:
    print("[oracle]", *args, flush=True)


def safe_name(ticker: str) -> str:
    return ticker.replace("^", "_").replace("=", "_").replace("/", "_")
