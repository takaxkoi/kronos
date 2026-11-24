#!/usr/bin/env bash
# One-time setup for Kronos Oracle on a fresh Ubuntu 24.04 server. Safe to run again.
#   curl -fsSL <url>/server/install.sh | bash
set -euo pipefail
REPO="${ORACLE_REPO:-https://github.com/takaxkoi/kronos}"
BRANCH="${ORACLE_BRANCH:-master}"
[ "$(id -u)" = 0 ] || { echo "Run this as root."; exit 1; }
say() { echo; echo ">>> $*"; }

say "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git cron util-linux curl ca-certificates build-essential python3-dev >/dev/null

say "Adding 4 GB of swap (spare memory for the model jobs)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

say "Fetching the app"
if [ -d /opt/oracle/.git ]; then git -C /opt/oracle pull -q --ff-only || true
else git clone -q --depth 1 -b "$BRANCH" "$REPO" /opt/oracle; fi
chmod +x /opt/oracle/server/*.sh
mkdir -p /opt/oracle-cache/hf /opt/oracle-envs /var/log/oracle

say "Main Python environment (this is the slow part: several minutes)"
[ -d /opt/oracle/.venv ] || python3 -m venv /opt/oracle/.venv
P=/opt/oracle/.venv/bin/pip
$P install -q --upgrade pip
$P install -q torch --index-url https://download.pytorch.org/whl/cpu
$P install -q -r /opt/oracle/requirements-app.txt transformers
$P install -q "git+https://github.com/takaxkoi/timesfm" || echo "second model (TimesFM) could not be installed; that one job will be skipped"

say "Side environments (labs, market scan, AI research)"
mk() { [ -d "/opt/oracle-envs/$1" ] || python3 -m venv "/opt/oracle-envs/$1"; "/opt/oracle-envs/$1/bin/pip" install -q --upgrade pip; shift; }
mk edge;     /opt/oracle-envs/edge/bin/pip install -q vectorbt quantstats "pandas<2.3" "numpy<2.3" "plotly<6" pyyaml requests || echo "labs environment failed"
mk desk;     /opt/oracle-envs/desk/bin/pip install -q openbb openbb-yfinance pyyaml requests numpy && /opt/oracle-envs/desk/bin/python -c "import openbb; openbb.build()" >/dev/null 2>&1 || echo "market scan environment failed"
mk research; /opt/oracle-envs/research/bin/pip install -q "git+https://github.com/takaxkoi/TradingAgents" pyyaml numpy requests || echo "AI research environment failed"

say "Keys file"
if [ ! -f /etc/oracle.env ]; then
  cat > /etc/oracle.env <<'K'
# Kronos Oracle keys. One per line, no spaces around the = sign, no quotes.
# Required: without the sync token nothing reaches the dashboard.
ORACLE_SYNC_TOKEN=
# Data sources
FINNHUB_API_KEY=
TICKERBOT_API_KEY=
GOOGLE_API_KEY=
# Optional
FRED_API_KEY=
ALPHAVANTAGE_API_KEY=
SEC_CONTACT=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
K
  chmod 600 /etc/oracle.env
fi

say "Schedule"
install -m 644 /opt/oracle/server/oracle.cron /etc/cron.d/oracle
systemctl enable --now cron >/dev/null 2>&1 || service cron start
ln -sf /opt/oracle/server/oracle.sh /usr/local/bin/oracle

say "Done. Next:"
echo "  1. Put your keys in:   nano /etc/oracle.env      (save with Ctrl+O, Enter, Ctrl+X)"
echo "  2. First run:          oracle first"
echo "  3. Check on it:        oracle status"
