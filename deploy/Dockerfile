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
# The graph is prepared at image build time, so the runtime starts in seconds.

FROM ghcr.io/project-osrm/osrm-backend:latest AS graph
ARG PBF_URL=https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf
ARG BBOX=-123.1,36.9,-121.3,38.9
WORKDIR /graph
RUN apt-get update && apt-get install -y --no-install-recommends curl osmium-tool \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "$PBF_URL" -o norcal.osm.pbf \
    && osmium extract --bbox "$BBOX" norcal.osm.pbf -o bayarea.osm.pbf \
    && rm norcal.osm.pbf \
    && osrm-extract -p /opt/car.lua bayarea.osm.pbf \
    && osrm-partition bayarea.osrm \
    && osrm-customize bayarea.osrm \
    && rm bayarea.osm.pbf

FROM ghcr.io/project-osrm/osrm-backend:latest
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv tini curl \
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
