# site/build_geometry.py
"""
Turn the OSM extract into the road geometry the map draws.

The first map drew straight lines between consecutive detectors, so freeways
arrived as chains of chords across every curve and stopped wherever a detector
had no forecast, leaving holes mid-freeway. Geometry now comes from OSM and
forecasts are attached to it:

  freeway   real centrelines cut into ~0.4 mi chunks, each governed by the
            nearest detector whose direction of travel agrees with the chunk's
            heading
  surface   primary and secondary roads within reach of a detector, carrying
            the inverse-distance weights the spread model uses

Speeds are not baked in. The browser already holds a speed series per detector,
so a chunk ships one id instead of 168 numbers and the scrubber stays instant.

    python site/build_geometry.py --osm ~/traffic-data/osm/bayarea.osm.pbf
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("build_geometry")

FREEWAY_TAGS = "w/highway=motorway,trunk"
SURFACE_TAGS = "w/highway=primary,secondary"

SIMPLIFY_M = 25.0          # Douglas-Peucker tolerance; motorways are smooth
CHUNK_MI = 0.4             # a little under detector spacing, so no chunk spans two
MAX_ASSIGN_MI = 4.0        # beyond this a detector does not speak for a chunk
# The OSM extract reaches into the Central Valley, where district 4 has no
# detectors: I-5 and CA-99 are a fifth of all chunks and none can be coloured.
# Roads further than this from any detector are not drawn.
MAX_RENDER_MI = 8.0
MAX_BEARING_DIFF = 70.0
SURFACE_RADIUS_MI = 2.0    # matches forecast.surface.RADIUS_MI
# Must match forecast.surface.MAX_STATIONS. Measured over the network, the
# nearest three carry only 69% of the inverse-distance weight (p10 = 53%), so
# cutting to three to save payload would shade a street differently from the
# price the route engine puts on it. Coordinate precision gives the bytes back:
# 4 dp is 11 m, finer than these lines are drawn.
SURFACE_MAX_STATIONS = 6
COORD_DP = 5               # ~1.1 m, freeway
SURFACE_DP = 4             # ~11 m, plenty for a 1-2 px line

MI_PER_DEG_LAT = 69.0
MI_PER_DEG_LON = 54.6      # at Bay Area latitude

DEFAULT_MAXSPEED = {"primary": 35, "secondary": 30,
                    "primary_link": 25, "secondary_link": 25}


def osmium_export(pbf, tags):
    """Run osmium once and stream back LineString features."""
    with tempfile.NamedTemporaryFile(suffix=".osm.pbf", delete=False) as f:
        filtered = f.name
    with tempfile.NamedTemporaryFile(suffix=".geojsonseq", delete=False) as f:
        seq = f.name
    try:
        subprocess.run(["osmium", "tags-filter", pbf, tags, "-o", filtered,
                        "--overwrite"], check=True, capture_output=True)
        subprocess.run(["osmium", "export", filtered, "-f", "geojsonseq",
                        "--geometry-types=linestring", "-o", seq, "--overwrite"],
                       check=True, capture_output=True)
        with open(seq) as fh:
            for line in fh:
                line = line.strip().lstrip("\x1e")
                if line:
                    yield json.loads(line)
    finally:
        for p in (filtered, seq):
            os.path.exists(p) and os.unlink(p)


def simplify(pts, tol_m=SIMPLIFY_M):
    """
    Iterative Douglas-Peucker on [lat, lon] in metres-ish local coordinates.

    Iterative rather than recursive: a few OSM ways are long enough to blow the
    default recursion limit, and a stack is three extra lines.
    """
    if len(pts) <= 2:
        return pts
    lat = pts[:, 0] * MI_PER_DEG_LAT * 1609.344
    lon = pts[:, 1] * MI_PER_DEG_LON * 1609.344
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        ax, ay, bx, by = lon[a], lat[a], lon[b], lat[b]
        dx, dy = bx - ax, by - ay
        norm = np.hypot(dx, dy)
        px, py = lon[a + 1:b], lat[a + 1:b]
        if norm < 1e-9:
            d = np.hypot(px - ax, py - ay)
        else:
            d = np.abs(dy * px - dx * py + bx * ay - by * ax) / norm
        i = int(np.argmax(d))
        if d[i] > tol_m:
            k = a + 1 + i
            keep[k] = True
            stack.extend([(a, k), (k, b)])
    return pts[keep]


def chunk(pts, chunk_mi=CHUNK_MI):
    """Cut a polyline into runs of roughly chunk_mi, preserving continuity."""
    if len(pts) < 2:
        return []
    seg = np.hypot(np.diff(pts[:, 0]) * MI_PER_DEG_LAT,
                   np.diff(pts[:, 1]) * MI_PER_DEG_LON)
    out, start, run = [], 0, 0.0
    for i, d in enumerate(seg):
        run += d
        if run >= chunk_mi and i + 1 > start:
            out.append(pts[start:i + 2])
            start, run = i + 1, 0.0
    if start < len(pts) - 1:
        out.append(pts[start:])
    return out


def bearing(a, b):
    dlon = np.radians(b[1] - a[1])
    y = np.sin(dlon) * np.cos(np.radians(b[0]))
    x = (np.cos(np.radians(a[0])) * np.sin(np.radians(b[0]))
         - np.sin(np.radians(a[0])) * np.cos(np.radians(b[0])) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def angular_diff(a, b):
    d = np.abs(a - b) % 360.0
    return min(d, 360.0 - d)


def local_xy(lat, lon):
    return np.column_stack([lat * MI_PER_DEG_LAT, lon * MI_PER_DEG_LON])


def build_freeway(pbf, stations, tree, sxy, known):
    """Chunked motorway/trunk centrelines, each with a governing detector."""
    from forecast.matching import station_bearings
    # The direction of travel, not the letter on the shield. See the note in
    # forecast.matching. Using the signed direction here left a third of the
    # network grey, concentrated on I-580 and I-80.
    sbrg = station_bearings(stations).to_numpy(dtype=float)
    sid = stations["sensor_id"].astype(int).to_numpy()

    chunks, mids, heads = [], [], []
    for feat in osmium_export(pbf, FREEWAY_TAGS):
        pts = np.array(feat["geometry"]["coordinates"], dtype=float)[:, ::-1]
        for c in chunk(simplify(pts)):
            if len(c) < 2:
                continue
            chunks.append(c)
            mids.append(c[len(c) // 2])
            heads.append(bearing(c[0], c[-1]))
    logger.info("freeway: %s chunks", f"{len(chunks):,}")

    mid_xy = local_xy(np.array([m[0] for m in mids]), np.array([m[1] for m in mids]))
    dists, idxs = tree.query(mid_xy, k=min(14, len(sxy)))

    out, unassigned, off_map = [], 0, 0
    for i, c in enumerate(chunks):
        nearest = float(np.atleast_1d(dists[i])[0])
        if nearest > MAX_RENDER_MI:
            off_map += 1
            continue
        pick = None
        for d, j in zip(np.atleast_1d(dists[i]), np.atleast_1d(idxs[i])):
            if d > MAX_ASSIGN_MI:
                break
            if np.isnan(sbrg[j]) or int(sid[j]) not in known:
                continue
            if angular_diff(heads[i], sbrg[j]) <= MAX_BEARING_DIFF:
                pick = (int(sid[j]), float(d))
                break
        if pick is None:
            unassigned += 1
        out.append({"s": pick[0] if pick else None,
                    "c": [[round(float(p[0]), COORD_DP), round(float(p[1]), COORD_DP)]
                          for p in c]})
    logger.info("freeway: %s chunks kept, %s dropped beyond %.0f mi from any detector",
                f"{len(out):,}", f"{off_map:,}", MAX_RENDER_MI)
    logger.info("freeway: %s kept chunks are uninstrumented and render neutral (%.1f%%)",
                f"{unassigned:,}", 100 * unassigned / max(len(out), 1))
    return out


def build_surface(pbf, stations, tree, sxy, known):
    """Primary/secondary roads near a detector, with spread-model weights."""
    sid = stations["sensor_id"].astype(int).to_numpy()
    ways, mids = [], []
    for feat in osmium_export(pbf, SURFACE_TAGS):
        pts = np.array(feat["geometry"]["coordinates"], dtype=float)[:, ::-1]
        pts = simplify(pts, tol_m=40.0)
        if len(pts) < 2:
            continue
        tags = feat.get("properties", {})
        ways.append((pts, parse_maxspeed(tags)))
        mids.append(pts[len(pts) // 2])
    logger.info("surface: %s ways before the radius filter", f"{len(ways):,}")

    mid_xy = local_xy(np.array([m[0] for m in mids]), np.array([m[1] for m in mids]))
    dists, idxs = tree.query(mid_xy, k=min(SURFACE_MAX_STATIONS + 2, len(sxy)))

    out = 0
    result = []
    for i, (pts, maxspeed) in enumerate(ways):
        refs = []
        for d, j in zip(np.atleast_1d(dists[i]), np.atleast_1d(idxs[i])):
            if d > SURFACE_RADIUS_MI or len(refs) >= SURFACE_MAX_STATIONS:
                break
            if int(sid[j]) in known:
                refs.append((int(sid[j]), 1.0 / max(float(d), 0.15) ** 2))
        if not refs:
            out += 1
            continue
        total = sum(w for _, w in refs)
        result.append({
            "w": [[s, round(w / total, 3)] for s, w in refs],
            "m": maxspeed,
            "c": [[round(float(p[0]), SURFACE_DP), round(float(p[1]), SURFACE_DP)]
                  for p in pts],
        })
    logger.info("surface: %s ways kept, %s dropped for having no detector within %.1f mi",
                f"{len(result):,}", f"{out:,}", SURFACE_RADIUS_MI)
    return result


def parse_maxspeed(tags):
    raw = tags.get("maxspeed")
    if raw:
        digits = "".join(ch for ch in str(raw) if ch.isdigit())
        if digits:
            mph = float(digits)
            return int(mph if "mph" in str(raw).lower() else mph * 0.621371)
    return int(DEFAULT_MAXSPEED.get(tags.get("highway"), 30))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--osm", default="~/traffic-data/osm/bayarea.osm.pbf")
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--serve", default="~/traffic-data/serve")
    p.add_argument("--out", default="site/data")
    p.add_argument("--skip-surface", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    from scipy.spatial import cKDTree
    pbf = os.path.expanduser(a.osm)
    meta = pd.read_csv(os.path.join(os.path.expanduser(a.data), "_meta", "d04_meta.txt"),
                       sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    meta = meta.dropna(subset=["Latitude", "Longitude"])

    # Only detectors that actually have a forecast can colour anything. Filtering
    # here rather than in the browser is what removes the holes: a chunk whose
    # nearest sensor is silent is handed the next one along instead of nothing.
    fc = pd.read_parquet(os.path.join(os.path.expanduser(a.serve), "forecast.parquet"),
                         columns=["station"])
    known = set(fc["station"].unique().tolist())
    logger.info("%d mainline detectors, %d with a forecast", len(meta), len(known))

    sxy = local_xy(meta["Latitude"].to_numpy(), meta["Longitude"].to_numpy())
    tree = cKDTree(sxy)

    payload = {"freeway": build_freeway(pbf, meta, tree, sxy, known)}
    payload["surface"] = [] if a.skip_surface else build_surface(
        pbf, meta, tree, sxy, known)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "geometry.json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    logger.info("geometry.json: %.1f MB  (%s freeway chunks, %s surface ways)",
                os.path.getsize(path) / 1e6,
                f"{len(payload['freeway']):,}", f"{len(payload['surface']):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
