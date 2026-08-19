# site/site_geocode.py
"""
Turn typed text into a point on the map.

Google Places is the requested provider and the better one -- it understands
"the ferry building" and "1 Hacker Way" alike, and it ranks by what people
actually search for. It needs a billed API key, so the module treats the
provider as a choice rather than a hard dependency:

  GOOGLE_MAPS_API_KEY set    Google Places Text Search
  otherwise                  Photon, an OSM geocoder that needs no key

The key stays on the server. A browser-side Places call would put it in page
source where anyone can lift it and spend the account's quota, and restricting
a key by HTTP referrer does not survive someone copying the page. One proxy
endpoint is less code than the client library anyway.

Results are biased to the Bay Area in both providers, because a commute
forecast asking about "Springfield" means the one down the road.
"""
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("geocode")

GOOGLE_TEXT = "https://places.googleapis.com/v1/places:searchText"
PHOTON = "https://photon.komoot.io/api"

# The forecast covers PeMS district 4. Searching outside it returns a point the
# map can show but the model cannot forecast, so results are biased here and the
# caller is told when a hit lands outside.
BAY_AREA = {"lat": 37.75, "lon": -122.2, "radius_m": 90000}
BOUNDS = (36.9, -123.1, 38.9, -121.3)          # south, west, north, east
TIMEOUT = 8
MAX_RESULTS = 6


def _in_area(lat, lon):
    s, w, n, e = BOUNDS
    return s <= lat <= n and w <= lon <= e


def _google(query, key):
    body = json.dumps({
        "textQuery": query,
        "maxResultCount": MAX_RESULTS,
        "locationBias": {"circle": {
            "center": {"latitude": BAY_AREA["lat"], "longitude": BAY_AREA["lon"]},
            "radius": BAY_AREA["radius_m"]}},
    }).encode()
    req = urllib.request.Request(GOOGLE_TEXT, data=body, headers={
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        payload = json.load(r)
    out = []
    for pl in payload.get("places", []):
        loc = pl.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is None or lon is None:
            continue
        out.append({"name": (pl.get("displayName") or {}).get("text", ""),
                    "address": pl.get("formattedAddress", ""),
                    "lat": round(lat, 6), "lon": round(lon, 6),
                    "in_area": _in_area(lat, lon), "source": "google"})
    return out


def _photon(query):
    s, w, n, e = BOUNDS
    params = {"q": query, "limit": MAX_RESULTS, "lang": "en",
              "lat": BAY_AREA["lat"], "lon": BAY_AREA["lon"],
              "bbox": f"{w},{s},{e},{n}"}
    url = f"{PHOTON}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "bay-commute-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        payload = json.load(r)
    out = []
    for feat in payload.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        p = feat.get("properties", {})
        parts = [p.get(k) for k in ("housenumber", "street", "city", "state")]
        out.append({"name": p.get("name") or p.get("street") or query,
                    "address": ", ".join(x for x in parts if x),
                    "lat": round(lat, 6), "lon": round(lon, 6),
                    "in_area": _in_area(lat, lon), "source": "photon"})
    return out


def parse_latlon(text):
    """A pasted "37.44,-122.14" should not cost a network round trip."""
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return [{"name": f"{lat:.4f}, {lon:.4f}", "address": "coordinates",
             "lat": lat, "lon": lon, "in_area": _in_area(lat, lon),
             "source": "literal"}]


def _dedupe(results):
    """Photon in particular returns the same place several times at one point."""
    seen, out = set(), []
    for r in results:
        key = (round(r["lat"], 4), round(r["lon"], 4), r["name"][:24].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def geocode(query):
    direct = parse_latlon(query)
    if direct:
        return direct
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    try:
        return _dedupe(_google(query, key) if key else _photon(query))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
        logger.warning("geocode(%r) failed on %s: %s", query,
                       "google" if key else "photon", e)
        # A keyed setup still falls back rather than showing the user nothing:
        # a quota error or a billing lapse should degrade, not break the page.
        if key:
            try:
                return _dedupe(_photon(query))
            except Exception:                       # noqa: BLE001
                return []
        return []
