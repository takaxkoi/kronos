"""Forecast chart image for Telegram (dark + gold)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BG, GOLD, UP, DN, MUTED, TXT = "#0B0B0D", "#BF9A3F", "#3FBF8A", "#E0565B", "#2A2A30", "#E9E4D8"


def forecast_png(ch: dict, a: dict) -> bytes:
    hist = ch["history"][-60:]
    H = len(ch["future"])
    n = len(hist)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for i, (_, o, h, l, c, _v) in enumerate(hist):
        col = UP if c >= o else DN
        ax.vlines(i, l, h, color=col, lw=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-9), color=col))
    x = np.arange(n, n + H)
    b = ch["bands"]
    ax.fill_between(x, b["10"], b["90"], color=GOLD, alpha=0.12, lw=0)
    ax.fill_between(x, b["25"], b["75"], color=GOLD, alpha=0.22, lw=0)
    for path in ch["paths"]:
        ax.plot(np.r_[n - 1, x], np.r_[hist[-1][4], path], color=GOLD, alpha=0.18, lw=0.7)
    for i, (o, h, l, c) in enumerate(ch["median"]):
        ax.vlines(n + i, l, h, color=GOLD, lw=0.9)
        ax.add_patch(plt.Rectangle((n + i - 0.3, min(o, c)), 0.6, max(abs(c - o), 1e-9), facecolor=BG if c >= o else GOLD,
                                   edgecolor=GOLD, lw=0.9))
    pl = a["plan"]
    if pl["valid"]:
        for y, lab, col in ((pl["target"], "TARGET", UP), (pl["entry"], "ENTRY", TXT), (pl["stop"], "STOP", DN)):
            ax.axhline(y, color=col, lw=0.6, ls="--", alpha=0.7)
            ax.text(n + H - 0.5, y, f" {lab} {y:,.2f}", color=col, fontsize=7, va="center")
    ax.axvline(n - 0.5, color=MUTED, lw=1)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors="#8A8577", labelsize=7)
    ax.set_xticks([])
    ax.grid(axis="y", color=MUTED, lw=0.4)
    arrow = "▲" if a["direction"] == "up" else "▼"
    ax.set_title(f"{a['ticker']}  {arrow} {a['median_return']:+.2%}  ·  score {a['score']}  ·  {a['agree_count']}/{a['samples']} agree",
                 color=GOLD, fontsize=11, loc="left", fontweight="bold")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()
