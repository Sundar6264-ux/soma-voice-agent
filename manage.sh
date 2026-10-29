#!/usr/bin/env bash
# ./manage.sh {start|stop|restart|status|test}
# Runs the app detached on 127.0.0.1:${PORT:-8790}; logs in logs/web.log.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8790}"
PIDFILE="logs/web.pid"
mkdir -p logs

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    setsid nohup ./venv/bin/uvicorn voiceagent.main:_app_factory --factory \
      --host 127.0.0.1 --port "$PORT" --proxy-headers --forwarded-allow-ips 127.0.0.1 \
      >> logs/web.log 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 20); do
      if curl -fsS "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; then
        echo "started on 127.0.0.1:$PORT (pid $(cat "$PIDFILE"))"; exit 0
      fi
      sleep 0.5
    done
    echo "did not come up; tail logs/web.log"; exit 1
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")"; rm -f "$PIDFILE"; echo stopped; else echo "not running"; fi
    ;;
  restart)
    "$0" stop; sleep 1; "$0" start
    ;;
  status)
    if running; then echo "running (pid $(cat "$PIDFILE"))"; curl -fsS "http://127.0.0.1:$PORT/healthz"; echo; else echo "not running"; fi
    ;;
  test)
    ./venv/bin/python -m unittest discover -s tests -t .
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|test}"; exit 2
    ;;
esac
