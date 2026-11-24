#!/usr/bin/env bash
# Small helper: `oracle status`, `oracle run <job>`, `oracle log <job>`, `oracle first`, `oracle update`.
R=/opt/oracle/server/run.sh
case "${1:-status}" in
  status)
    echo "Keys set: $(grep -cE '^[A-Z_]+=.+' /etc/oracle.env 2>/dev/null) of $(grep -cE '^[A-Z_]+=' /etc/oracle.env 2>/dev/null)"
    echo "Memory:   $(free -h | awk '/Mem:/{print $3" used of "$2}')   Disk: $(df -h /opt | awk 'NR==2{print $3" used of "$2}')"
    echo "Last runs (UTC, job, exit code; 0 means it worked):"
    [ -f /var/log/oracle/history.log ] && tac /var/log/oracle/history.log | awk '!seen[$2]++' | sort -k2 | column -t || echo "  none yet" ;;
  run)   shift; nohup "$R" "$@" >/dev/null 2>&1 & echo "started $1 in the background; watch it with: oracle log $1" ;;
  log)   tail -n "${3:-60}" "/var/log/oracle/${2:?job}.log" ;;
  first) # fill the dashboard in a sensible order the first time; takes a while because it re-downloads 10 years of prices
    nohup bash -c "$R capture; $R brief 520; $R prices; $R news; $R edge; $R sp500; $R second; $R daily" >/dev/null 2>&1 &
    echo "First run started in the background (an hour or more). Check progress with: oracle status" ;;
  update) git -C /opt/oracle pull --ff-only && install -m 644 /opt/oracle/server/oracle.cron /etc/cron.d/oracle && echo updated ;;
  *) echo "usage: oracle status | run <job> | log <job> | first | update" ;;
esac
