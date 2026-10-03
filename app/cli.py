"""Command line entry point.

  python -m app.cli daily        # after the close: full forecast, grading, paper trading, recap
  python -m app.cli premarket    # 8:30 AM ET briefing + Alpaca orders
  python -m app.cli sp500        # overnight S&P 500 scanner
  python -m app.cli crypto       # hourly crypto board
  python -m app.cli live         # bot commands + price-hit alerts
  python -m app.cli bot --live   # instant bot replies (run on your own computer)
  python -m app.cli backtest [TICKERS...] [--battle]
  python -m app.cli capture      # DATA VAULT: daily point-in-time snapshot of everything -> data/daily/
  python -m app.cli grade        # grade finished predictions only
  python -m app.cli retrain      # monthly fine-tune of custom per-ticker models
"""
from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def market_open_now() -> bool:
    ny = datetime.now(ZoneInfo("America/New_York"))
    if ny.weekday() >= 5:
        return False
    mins = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= mins <= 16 * 60 + 5


def main(argv: list[str]) -> None:
    from . import scan
    cmd = argv[0] if argv else "help"
    if cmd == "daily":
        scan.daily_scan()
    elif cmd == "premarket":
        scan.premarket()
    elif cmd == "sp500":
        scan.sp500_scan()
    elif cmd == "crypto":
        scan.crypto_hourly()
    elif cmd == "live":
        from .bot import poll
        poll()
        if market_open_now():
            scan.price_watch()
    elif cmd == "bot":
        from .bot import poll
        poll(live="--live" in argv)
    elif cmd == "backtest":
        tick = [a.upper() for a in argv[1:] if not a.startswith("--")]
        scan.backtest_lab(tick or None, models="--battle" in argv)
    elif cmd == "capture":
        from .capture import capture
        capture()
    elif cmd == "grade":
        from . import ledger
        from .core import SITE_DATA, load_config, write_json
        cfg = load_config()
        ledger.grade(cfg)
        write_json(SITE_DATA / "report.json", ledger.report(cfg))
    elif cmd == "retrain":
        from .retrain import retrain
        retrain()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
