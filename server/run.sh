#!/usr/bin/env bash
# Runs one scheduled job on the server: the same steps the GitHub job used to run, then publishes to Supabase.
# Usage: run.sh <job>     Logs: /var/log/oracle/<job>.log
set -u
JOB="${1:?job name}"; shift || true
ROOT=/opt/oracle
cd "$ROOT" || exit 1
set -a; [ -f /etc/oracle.env ] && . /etc/oracle.env; set +a
export PYTHONUNBUFFERED=1 HF_HOME=/opt/oracle-cache/hf
mkdir -p /var/log/oracle /opt/oracle-cache/hf
LOG=/var/log/oracle/$JOB.log
py()   { "$ROOT/.venv/bin/python" "$@"; }            # main environment
pyin() { local v="$1"; shift; "/opt/oracle-envs/$v/bin/python" "$@"; }   # side environments with their own package versions
publish() { py -m app.cli publish || echo "publish to Supabase failed (dashboard keeps the previous data)"; }
weekday() { [ "$(date -u +%u)" -le 5 ]; }

job() {
  case "$JOB" in
    prices)     py -m app.quotes; py -m app.livenews; py -m app.brief ;;
    brief)      py -m app.feargreed; py -m app.spx; py -m app.strategies; py -m app.regime; py -m app.accuracy; py -m app.insiders
                py -m app.earnings --fetch "${1:-90}"; py -m app.econ; py -m app.economy; py -m app.brief ;;
    daily)      py -m app.cli daily ;;
    premarket)  py -m app.cli premarket ;;
    sp500)      py -m app.cli sp500 ;;
    crypto)     py -m app.cli crypto ;;
    cryptodesk) py -m app.cryptodesk ;;
    capture)    py -m app.cli capture ;;
    live)       py -m app.cli live ;;
    retrain)    py -m app.cli retrain ;;
    news)       py -m app.news; py -m app.wsb ;;
    second)     if [ "${1:-}" = "backtest" ] || [ "$(date -u +%u)" = "6" ]; then py -m app.second --backtest; else py -m app.second; fi ;;
    gov)        py -m app.govdata ;;
    tickerbot)  py -m app.tickerbot ;;
    edge)       pyin edge -m app.lenses; pyin edge -m app.terminal; pyin edge -m app.perf; pyin edge -m app.edge; pyin edge -m app.factorlab ;;
    filings)    pyin edge -m app.filings ;;
    desk)       pyin desk -m app.desk ;;
    research)   pyin research -m app.research ${TICKERS:-} ;;
    *) echo "unknown job: $JOB"; return 2 ;;
  esac
}

# one copy of a job at a time; the heavy model jobs also wait for each other so memory is never shared
HEAVY=""; case "$JOB" in daily|sp500|crypto|retrain|second|news|research) HEAVY=1 ;; esac
(
  flock -n 9 || { echo "$(date -u +%FT%TZ) $JOB already running, skipped"; exit 0; }
  if [ -n "$HEAVY" ]; then exec 8>/var/lock/oracle-heavy.lock; flock -w 14400 8 || { echo "gave up waiting for another model job"; exit 1; }; fi
  echo "===== $(date -u +%FT%TZ) start $JOB"
  job "$@"; rc=$?
  publish
  echo "===== $(date -u +%FT%TZ) end $JOB (exit $rc)"
  echo "$(date -u +%FT%TZ) $JOB $rc" >> /var/log/oracle/history.log
) 9>"/var/lock/oracle-$JOB.lock" >>"$LOG" 2>&1
