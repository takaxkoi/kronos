"""Kronos wrapper that returns MANY simulated futures per stock instead of one average.

KronosPredictor averages its samples internally, so we feed the same history N times
(sample_count=1 each) to get N independent paths. Those paths power the agreement meter,
expected ranges, confidence bands and every signal built on top.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import log
from .data import future_index

COLS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass
class Job:
    ticker: str
    df: pd.DataFrame          # finished candles, index = timestamps
    horizon: int
    interval: str = "1d"
    cls: str = "stock"
    asof: int | None = None   # use history up to this row (backtests); None = all


@dataclass
class Paths:
    ticker: str
    interval: str
    cls: str
    history: pd.DataFrame     # the candles Kronos saw
    future: pd.DatetimeIndex
    arr: np.ndarray           # (samples, horizon, 6)  open high low close volume amount
    model: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def last_close(self) -> float:
        return float(self.history["close"].iloc[-1])

    def col(self, name: str) -> np.ndarray:
        return self.arr[:, :, COLS.index(name)]


class Forecaster:
    _cache: dict = {}

    def __init__(self, predictor, name: str):
        self.predictor = predictor
        self.name = name

    @classmethod
    def load(cls, name: str, tokenizer: str, max_context: int) -> "Forecaster":
        key = (name, tokenizer, max_context)
        if key not in cls._cache:
            import torch
            from model import Kronos, KronosPredictor, KronosTokenizer
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            log(f"loading {name}")
            tok = KronosTokenizer.from_pretrained(tokenizer).eval()
            mdl = Kronos.from_pretrained(name).eval()
            cls._cache[key] = Forecaster(KronosPredictor(mdl, tok, max_context=max_context), name)
        return cls._cache[key]

    def run(self, jobs: list[Job], samples: int = 30, lookback: int = 400, temperature: float = 1.0,
            top_p: float = 0.9, batch_size: int = 64, seed: int | None = None, min_bars: int = 120) -> dict[str, Paths]:
        import torch
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        prepared = []
        for j in jobs:
            df = j.df if j.asof is None else j.df.iloc[: j.asof]
            df = df.tail(lookback)
            if len(df) < min_bars or df[["open", "high", "low", "close"]].isnull().values.any():
                continue
            fut = future_index(df.index[-1], j.horizon, j.interval, j.cls)
            prepared.append((j, df, fut))
        # group by (history length, horizon) because predict_batch needs equal shapes
        groups: dict[tuple, list] = {}
        for p in prepared:
            groups.setdefault((len(p[1]), p[0].horizon), []).append(p)
        out: dict[str, Paths] = {}
        for (_, horizon), items in groups.items():
            seqs = []
            for idx, (j, df, fut) in enumerate(items):
                x_ts = pd.Series(df.index)
                y_ts = pd.Series(fut)
                xdf = df[COLS].reset_index(drop=True)
                seqs += [(idx, xdf, x_ts, y_ts)] * samples
            results: dict[int, list] = {}
            for s in range(0, len(seqs), batch_size):
                chunk = seqs[s:s + batch_size]
                preds = self.predictor.predict_batch(
                    [c[1] for c in chunk], [c[2] for c in chunk], [c[3] for c in chunk],
                    pred_len=horizon, T=temperature, top_p=top_p, sample_count=1, verbose=False)
                for c, p in zip(chunk, preds):
                    results.setdefault(c[0], []).append(p[COLS].values)
            for idx, (j, df, fut) in enumerate(items):
                arr = np.stack(results[idx]).astype(float)
                arr[:, :, 4:] = np.clip(arr[:, :, 4:], 0, None)  # no negative volume
                # keep candles internally consistent
                arr[:, :, 1] = np.maximum(arr[:, :, 1], arr[:, :, [0, 3]].max(axis=2))
                arr[:, :, 2] = np.minimum(arr[:, :, 2], arr[:, :, [0, 3]].min(axis=2))
                key = j.ticker if j.asof is None else f"{j.ticker}@{j.asof}"
                out[key] = Paths(j.ticker, j.interval, j.cls, df, fut, arr, self.name)
        return out
