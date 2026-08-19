#!/bin/bash
# scripts/nightly.sh
#
# One run of the whole serving pipeline: pull yesterday, refresh the forward
# inputs, predict the horizon, rebuild the site data.
#
# Ordered so that a failure early stops the run rather than publishing a
# forecast built on stale inputs. `set -e` is the whole safety model: a site
# that silently serves last week's numbers is worse than a site that visibly
# did not update.
#
# The seasonal table is rebuilt on Sundays only. It is a four-year average, so
# one more day moves it by nothing, and the rebuild is 40 seconds of streaming
# over 1,500 files that there is no reason to spend nightly.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${TRAFFIC_DATA:-$HOME/traffic-data}"
PY="${PYTHON:-/Users/Tom/miniforge3/bin/python}"
LOG="$DATA/logs/nightly-$(date +%Y-%m-%d).log"
mkdir -p "$DATA/logs"
exec >>"$LOG" 2>&1

echo "=== $(date -u +%FT%TZ) nightly start ==="
cd "$REPO"
set -a; . "$HOME/.pems_env"; set +a

echo "--- PeMS: any missing days this year"
$PY -m forecast.backfill_stations --years "$(date +%Y)" --out "$DATA/stations"

# Roadwork is NOT collected here. Caltrans publishes current state only, so a
# missed day is gone forever -- it gets its own launchd agent at 03:05 so a
# failure anywhere in this pipeline cannot cost a day of history that can never
# be backfilled.

echo "--- venue calendars, forward events"
$PY -m collector.venue_events --live --out "$DATA/events"
$PY -m collector.events_normalize --inputs "$DATA"/events/events_*.jsonl \
    --out "$DATA/events/events_clean.jsonl"
$PY -m collector.merge_league_times --events "$DATA/events/events_clean.jsonl" \
    --out "$DATA/events/events_merged.jsonl"

echo "--- weather: forward forecast on the station grid"
$PY -m collector.weather --live --forecast-days 7 --grid-step 0.25 \
    --name live_forecast --meta "$DATA/stations/_meta/d04_meta.txt" --out "$DATA/weather"

if [ "$(date +%u)" = "7" ]; then
  echo "--- Sunday: rebuild the seasonal table"
  $PY -m forecast.build_seasonal --data "$DATA/stations"   # through today, serving build
fi

echo "--- predict the horizon"
$PY -m forecast.predict_network --days 7 --out "$DATA/serve"

echo "--- rebuild site data"
$PY site/build_data.py --serve "$DATA/serve" --out "$REPO/site/data"

echo "=== $(date -u +%FT%TZ) nightly ok ==="
