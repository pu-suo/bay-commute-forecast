# forecast/surface.py
"""
Surface-street travel time by spreading freeway congestion onto local roads.

PeMS covers freeways only — there is no arterial speed data in it, and no free
source of Bay Area surface-street speeds exists. But local streets and the
freeways beside them share their demand drivers: the same commute peaks, the
same events, the same weather. So freeway congestion is used here as a *proxy*
signal for nearby local roads.

The model is deliberately simple and explicit:

    local_speed = maxspeed x (1 - ALPHA x (1 - freeway_ratio))
    freeway_ratio = inverse-distance-weighted mean of (predicted / free-flow)
                    over mainline stations within RADIUS_MI

ALPHA dampens the transfer: arterials do not slow as hard as the freeway beside
them, partly because they are already slower and partly because freeway
congestion diverts traffic *onto* them. ALPHA = 1.0 would mean a local road
halves in speed exactly when the freeway does.

HONEST LIMITATIONS — these belong on the site, not just in this docstring:

  * ALPHA is an assumption, not a fitted parameter. There is no arterial ground
    truth to fit it against, so it is a stated prior that should be revised by
    periodic manual spot-checks.
  * The sign is not guaranteed. Freeway congestion can push traffic onto
    parallel arterials (slowing them) or hold it on the freeway (freeing them).
    Which dominates almost certainly varies by location.
  * Surface estimates must be reported separately from freeway forecasts and
    EXCLUDED from the published accuracy statistics. Mixing an unvalidated
    component into a scored number is how an accuracy page stops meaning
    anything.
"""
import json
import logging
import math
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger("surface")

OVERPASS = "https://overpass-api.de/api/interpreter"
ROAD_CLASSES = "motorway|trunk|primary|secondary|tertiary"

# How far a freeway station can plausibly speak for a local road.
RADIUS_MI = 2.0
MAX_STATIONS = 6
# Dampening. 0.5 = arterials absorb half the freeway's proportional slowdown.
# A stated prior, not a fitted value. Revise from spot-checks.
ALPHA = 0.5
# Never claim a road is faster than its limit, or slower than a crawl.
MIN_SPEED_MPH = 5.0

DEFAULT_MAXSPEED = {"motorway": 65, "trunk": 45, "primary": 35,
                    "secondary": 35, "tertiary": 30}


def fetch_ways(bbox, cache_path=None):
    """Fetch drivable ways in a bbox (south, west, north, east) from Overpass."""
    if cache_path and os.path.exists(cache_path):
        return json.load(open(cache_path))["elements"]
    s, w, n, e = bbox
    query = (f'[out:json][timeout:180];'
             f'way["highway"~"^({ROAD_CLASSES})$"]({s},{w},{n},{e});'
             f'out tags geom;')
    req = urllib.request.Request(
        OVERPASS, data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "bay-commute-forecast/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=240) as r:
        payload = json.load(r)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump(payload, open(cache_path, "w"))
    logger.info("overpass: %d ways", len(payload.get("elements", [])))
    return payload.get("elements", [])


def parse_maxspeed(tags):
    """'35 mph' -> 35.0; fall back to a class default when untagged."""
    raw = tags.get("maxspeed")
    if raw:
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        if digits:
            mph = float(digits)
            return mph if "mph" in str(raw).lower() else mph * 0.621371
    return float(DEFAULT_MAXSPEED.get(tags.get("highway"), 30))


def ways_to_frame(elements):
    """One row per way: midpoint, length in miles, free-flow speed limit."""
    rows = []
    for el in elements:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        lats = np.array([p["lat"] for p in geom])
        lons = np.array([p["lon"] for p in geom])
        miles = float(np.sum(np.hypot(np.diff(lats) * 69.0,
                                      np.diff(lons) * 54.6)))
        if miles <= 0:
            continue
        tags = el.get("tags", {})
        rows.append({"way_id": el["id"], "name": tags.get("name", ""),
                     "highway": tags.get("highway"),
                     "lat": float(lats.mean()), "lon": float(lons.mean()),
                     "miles": miles, "maxspeed_mph": parse_maxspeed(tags)})
    return pd.DataFrame(rows)


def link_stations(ways, stations, radius_mi=RADIUS_MI, max_stations=MAX_STATIONS):
    """
    Precompute, per way, the nearby freeway stations and their weights.

    Geometry does not change, so this is computed once and reused every night.
    Weights are inverse-distance-squared, normalised, over the nearest stations
    inside the radius. Ways with no freeway within the radius get no link and
    fall back to free-flow.
    """
    slat = stations["Latitude"].to_numpy()
    slon = stations["Longitude"].to_numpy()
    sid = stations["sensor_id"].astype(int).to_numpy()

    links = []
    for w in ways.itertuples():
        d = np.hypot((slat - w.lat) * 69.0, (slon - w.lon) * 54.6)
        near = np.where(d <= radius_mi)[0]
        if near.size == 0:
            continue
        near = near[np.argsort(d[near])][:max_stations]
        wt = 1.0 / np.maximum(d[near], 0.15) ** 2
        wt = wt / wt.sum()
        for station, weight, dist in zip(sid[near], wt, d[near]):
            links.append({"way_id": w.way_id, "station": int(station),
                          "weight": float(weight), "dist_mi": float(dist)})
    out = pd.DataFrame(links)
    linked = out["way_id"].nunique() if len(out) else 0
    logger.info("linked %d/%d ways to freeway stations (%.0f%%)",
                linked, len(ways), linked / max(len(ways), 1) * 100)
    return out


def freeway_ratio(links, station_state):
    """
    Weighted (predicted / free-flow) per way. 1.0 means freeways at free-flow.

    `station_state` must carry station, speed and freeflow columns.
    """
    s = station_state.copy()
    s["ratio"] = (s["speed"] / s["freeflow"]).clip(0.15, 1.05)
    m = links.merge(s[["station", "ratio"]], on="station", how="inner")
    if m.empty:
        return pd.Series(dtype=float)
    m["wr"] = m["weight"] * m["ratio"]
    agg = m.groupby("way_id").agg(num=("wr", "sum"), den=("weight", "sum"))
    return (agg["num"] / agg["den"]).rename("freeway_ratio")


def surface_minutes(ways, links, station_state, alpha=ALPHA):
    """
    Estimated travel time per way, in minutes.

    Returns the estimate alongside the free-flow time and the ratio that drove
    it, so the site can show how much of the number is assumption rather than
    measurement.
    """
    ratio = freeway_ratio(links, station_state)
    out = ways.merge(ratio, on="way_id", how="left")
    # no nearby freeway -> no signal -> assume free-flow rather than invent one
    out["freeway_ratio"] = out["freeway_ratio"].fillna(1.0)
    factor = 1.0 - alpha * (1.0 - out["freeway_ratio"])
    out["est_speed_mph"] = np.maximum(out["maxspeed_mph"] * factor, MIN_SPEED_MPH)
    out["freeflow_minutes"] = out["miles"] / out["maxspeed_mph"] * 60.0
    out["est_minutes"] = out["miles"] / out["est_speed_mph"] * 60.0
    out["is_estimate"] = True          # never let this be mistaken for a forecast
    return out
