# One image, two processes: osrm-routed and the Python API.
#
# They live together because the API calls OSRM on every route and the prepared
# graph is 880 MB. Splitting them would mean shipping that graph twice or paying
# a network hop on a call that happens once or twice per page view.
#
# Built on the OSRM image with Python added, rather than the reverse. OSRM
# publishes on Debian trixie and python:3.12-slim is bookworm, so copying the
# binary across gives an unresolvable boost version mismatch.
#
# The graph is prepared at image build time so the runtime does no graph work.
# Starting is still not instant: osrm-routed reads the ~930 MB result off cold
# storage and answers 76s after boot on Fly. entrypoint.sh waits that out.
#
# The base is pinned by digest, not :latest. A silent rebuild of the upstream
# tag is what broke this image once already: curl arrived without a CA bundle
# and every HTTPS fetch failed. The digest also guarantees both stages share a
# base, which :latest does not once upstream republishes mid-build. To move up,
# repoint both FROM lines at the new digest deliberately.

FROM ghcr.io/project-osrm/osrm-backend@sha256:3ac496ff8fd7e1af53846179d73d06a97f719c8ad2217d008ed868942398665c AS graph
# Geofabrik is the primary because its NorCal extract is the smallest input that
# covers the bbox. It goes down: a multi-hour 502 across the whole site blocked
# this build once. The fallback is the OSM France mirror, which only publishes
# statewide, so it trades roughly twice the download for a build that still
# finishes. The bbox lies entirely within California, so osmium yields the same
# graph either way and nothing downstream can tell which source was used.
ARG PBF_URL=https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf
ARG PBF_URL_FALLBACK=https://download.openstreetmap.fr/extracts/north-america/us-west/california-latest.osm.pbf
ARG BBOX=-123.1,36.9,-121.3,38.9
WORKDIR /graph
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl osmium-tool \
    && rm -rf /var/lib/apt/lists/*
RUN echo "builder: $(nproc) cpu, $(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo)MB ram, \
cgroup-max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo n/a), \
$(df -Pm /graph | awk 'NR==2{print $4}')MB disk" >&2 \
    && { curl -fsSL --connect-timeout 20 --retry 2 "$PBF_URL" -o src.osm.pbf \
         && osmium extract --bbox "$BBOX" src.osm.pbf -o bayarea.osm.pbf \
         && rm -f src.osm.pbf; } \
       || { echo "primary source failed; streaming from the mirror instead" >&2; \
            rm -f src.osm.pbf bayarea.osm.pbf; \
            curl -fsSL --connect-timeout 20 --retry 2 "$PBF_URL_FALLBACK" \
              | osmium extract --bbox "$BBOX" -F pbf -s simple -o bayarea.osm.pbf -; } \
    && osrm-extract -p /opt/car.lua bayarea.osm.pbf \
    && osrm-partition bayarea.osrm \
    && osrm-customize bayarea.osrm \
    && rm bayarea.osm.pbf

FROM ghcr.io/project-osrm/osrm-backend@sha256:3ac496ff8fd7e1af53846179d73d06a97f719c8ad2217d008ed868942398665c
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv tini curl \
    && rm -rf /var/lib/apt/lists/*

# The API reads the forecast table and prices routes. It trains nothing, so
# lightgbm and the collectors' dependencies stay out of the image.
RUN python3 -m venv /venv \
    && /venv/bin/pip install --no-cache-dir pandas numpy pyarrow

COPY --from=graph /graph/ /graph/

WORKDIR /app
COPY forecast/ forecast/
COPY site/ site/
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Station metadata is small, static and required at startup, so it is baked in.
# The forecast table is not: it changes nightly and is fetched from SERVE_URL.
COPY deploy/d04_meta.txt /data/stations/_meta/d04_meta.txt

ENV PATH=/venv/bin:$PATH \
    OSRM_URL=http://127.0.0.1:5000 \
    SERVE_DIR=/data/serve \
    STATIONS_DIR=/data/stations \
    HOST=0.0.0.0 \
    PORT=8080 \
    BEHIND_PROXY=1
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/entrypoint.sh"]
