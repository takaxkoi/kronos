#!/usr/bin/env bash
# Small helper: `oracle status`, `oracle keys`, `oracle token`, `oracle run <job>`, `oracle log <job>`, `oracle first`, `oracle update`.
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
  keys)  # asks for each key in turn; what you type is hidden; press Enter to leave one as it is
    for k in FINNHUB_API_KEY TICKERBOT_API_KEY GOOGLE_API_KEY FRED_API_KEY ALPHAVANTAGE_API_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID; do
      cur=$(grep -E "^$k=" /etc/oracle.env | cut -d= -f2-)
      read -r -s -p "$k $([ -n "$cur" ] && echo '(already set, Enter keeps it)' || echo '(empty, Enter skips)'): " v; echo
      [ -n "$v" ] && { grep -q "^$k=" /etc/oracle.env && sed -i "s|^$k=.*|$k=$v|" /etc/oracle.env || echo "$k=$v" >> /etc/oracle.env; }
    done
    grep -q '^SEC_CONTACT=.\+' /etc/oracle.env || sed -i 's|^SEC_CONTACT=.*|SEC_CONTACT=Ishmam Hossain takaxkoi@gmail.com|' /etc/oracle.env
    echo "Saved. Keys set: $(grep -cE '^[A-Z_]+=.+' /etc/oracle.env)" ;;
  token) # makes a new dashboard sync token on this server and prints its fingerprint (safe to share; the token itself never leaves)
    t=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    sed -i "s|^ORACLE_SYNC_TOKEN=.*|ORACLE_SYNC_TOKEN=$t|" /etc/oracle.env
    echo; echo "Send this fingerprint to Claude:"; echo; printf '%s' "$t" | sha256sum | cut -d' ' -f1; echo ;;
  update) git -C /opt/oracle pull --ff-only && install -m 644 /opt/oracle/server/oracle.cron /etc/cron.d/oracle && echo updated ;;
  *) echo "usage: oracle status | keys | token | run <job> | log <job> | first | update" ;;
esac
