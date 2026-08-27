#!/bin/sh
# Start OSRM, wait for it to answer, then start the API in the foreground so the
# container's health follows the process that serves traffic.
set -e

osrm-routed --algorithm mld --port 5000 --max-matching-size 2000 \
    /graph/bayarea.osrm &
osrm_pid=$!

i=0
until curl -fsS -o /dev/null "http://127.0.0.1:5000/route/v1/driving/-122.14,37.44;-122.38,37.62" 2>/dev/null; do
    i=$((i + 1))
    if [ $i -gt 60 ]; then
        echo "osrm-routed did not come up" >&2
        exit 1
    fi
    kill -0 "$osrm_pid" 2>/dev/null || { echo "osrm-routed died" >&2; exit 1; }
    sleep 1
done
echo "osrm-routed ready after ${i}s"

exec python3 site/server.py \
    --host "$HOST" --port "$PORT" \
    --serve "$SERVE_DIR" --data "$STATIONS_DIR" \
    --osrm "$OSRM_URL" \
    ${ALLOW_ORIGIN:+--allow-origin "$ALLOW_ORIGIN"} \
    ${BEHIND_PROXY:+--behind-proxy}
