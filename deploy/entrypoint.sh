#!/bin/sh
# Start OSRM, wait for it to answer, then start the API in the foreground so the
# container's health follows the process that serves traffic.
set -e

osrm-routed --algorithm mld --port 5000 --max-matching-size 2000 \
    /graph/bayarea.osrm &
osrm_pid=$!

# The graph is ~930 MB and a cold machine reads it off storage before the server
# answers anything. On a warm local cache that takes about 4s, which is where the
# old 60s budget came from; on a cold shared-cpu-1x it does not, and the machine
# was killing itself and rebooting into the same cold read forever. Wait minutes,
# not seconds, and say so while waiting -- the log line is how we learn the real
# number. OSRM_WAIT keeps it tunable without another image build.
: "${OSRM_WAIT:=300}"
i=0
until curl -fsS -o /dev/null "http://127.0.0.1:5000/route/v1/driving/-122.14,37.44;-122.38,37.62" 2>/dev/null; do
    i=$((i + 1))
    if [ $i -gt "$OSRM_WAIT" ]; then
        echo "osrm-routed did not answer within ${OSRM_WAIT}s" >&2
        exit 1
    fi
    kill -0 "$osrm_pid" 2>/dev/null || { echo "osrm-routed died" >&2; exit 1; }
    [ $((i % 30)) -eq 0 ] && echo "still loading the graph, ${i}s elapsed" >&2
    sleep 1
done
echo "osrm-routed ready after ${i}s"

exec python3 site/server.py \
    --host "$HOST" --port "$PORT" \
    --serve "$SERVE_DIR" --data "$STATIONS_DIR" \
    --osrm "$OSRM_URL" \
    ${ALLOW_ORIGIN:+--allow-origin "$ALLOW_ORIGIN"} \
    ${BEHIND_PROXY:+--behind-proxy}
