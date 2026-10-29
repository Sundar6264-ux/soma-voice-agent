#!/usr/bin/env bash
# ./manage.sh {start|stop|restart|status|test}
#             {asterisk-build|asterisk-start|asterisk-stop|asterisk-status|asterisk-cli}
# Runs the app detached on 127.0.0.1:${PORT:-8790}; logs in logs/web.log.
# Asterisk runs in Docker (host network, restarts with the Docker daemon); logs: docker logs rva-asterisk.
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
  asterisk-build)
    docker build -t rva-asterisk:22 asterisk
    ;;
  asterisk-start)
    ./venv/bin/python asterisk/render.py
    docker rm -f rva-asterisk > /dev/null 2>&1 || true
    mounts=()
    for f in asterisk/etc/*.conf; do mounts+=(-v "$PWD/$f:/etc/asterisk/$(basename "$f"):ro"); done
    docker run -d --name rva-asterisk --network host --restart unless-stopped "${mounts[@]}" rva-asterisk:22 > /dev/null
    sleep 4
    docker exec rva-asterisk asterisk -rx "pjsip show registrations" || { docker logs --tail 30 rva-asterisk; exit 1; }
    ;;
  asterisk-stop)
    docker rm -f rva-asterisk > /dev/null 2>&1 && echo "asterisk stopped" || echo "asterisk not running"
    ;;
  asterisk-status)
    docker exec rva-asterisk asterisk -rx "pjsip show registrations"
    docker exec rva-asterisk asterisk -rx "core show channels count"
    ;;
  asterisk-cli)
    docker exec -it rva-asterisk asterisk -rvvv
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|test|asterisk-build|asterisk-start|asterisk-stop|asterisk-status|asterisk-cli}"; exit 2
    ;;
esac
