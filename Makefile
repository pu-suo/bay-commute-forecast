# Makefile — the whole project as a handful of verbs.
#
# Targets are ordered the way the data flows. Everything reads and writes
# $(DATA) (outside the repo, since it is 1.7 GB) and nothing here needs a GPU,
# a cloud account, or a database.

PY   ?= /Users/Tom/miniforge3/bin/python
DATA ?= $(HOME)/traffic-data
OSM  ?= $(DATA)/osm

.PHONY: help osrm seasonal train validate serve-data site nightly clean-osrm

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

osrm: ## build the routing graph and start OSRM on :5001
	docker run --rm -v "$(OSM):/data" ghcr.io/project-osrm/osrm-backend \
	  osrm-extract -p /opt/car.lua /data/bayarea.osm.pbf
	docker run --rm -v "$(OSM):/data" ghcr.io/project-osrm/osrm-backend \
	  osrm-partition /data/bayarea.osrm
	docker run --rm -v "$(OSM):/data" ghcr.io/project-osrm/osrm-backend \
	  osrm-customize /data/bayarea.osrm
	-docker rm -f osrm-bay
	docker run -d --name osrm-bay -p 5001:5000 -v "$(OSM):/data" \
	  ghcr.io/project-osrm/osrm-backend osrm-routed --algorithm mld /data/bayarea.osrm

seasonal: ## rebuild both seasonal profiles (train-only and serving)
	$(PY) -m forecast.build_seasonal --through 2025-01-01 \
	  --out $(DATA)/stations/_seasonal_trainonly.parquet
	$(PY) -m forecast.build_seasonal --data $(DATA)/stations

train: ## train the network model (CPU, ~25 min)
	$(PY) -m forecast.train_stations --out models/network

validate: ## score the model in minutes on the nine corridors
	$(PY) -m forecast.validate_routes --model models/network/model.txt

serve-data: ## predict the horizon and rebuild the site's JSON
	$(PY) -m forecast.predict_network --days 7 --out $(DATA)/serve
	$(PY) site/build_data.py --serve $(DATA)/serve --out site/data

site: ## run the local site on :8000 (needs OSRM for routing)
	$(PY) site/server.py --port 8000 --serve $(DATA)/serve

nightly: ## one full nightly run, exactly as launchd invokes it
	bash scripts/nightly.sh

clean-osrm: ## drop the routing container
	-docker rm -f osrm-bay
