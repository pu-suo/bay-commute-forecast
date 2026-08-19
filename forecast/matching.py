# forecast/matching.py
"""
Match a route polyline onto the PeMS detectors it traverses.

Two tests, not one. Proximity alone matches the opposing carriageway, which sits
about 200 ft away: on a Palo Alto to SFO route that gave 39 northbound and 41
southbound detectors, claiming 134% of the route as instrumented. So a detector
must also be heading-aligned.

"Heading" is the direction traffic moves, not the letter on the shield. Caltrans
signs I-580 east/west while it runs north/south through Oakland, and I-80 east
while it runs due north at Vallejo. 19.4% of detectors sit more than 70 degrees
from their signed direction. Bearings therefore come from each detector's
neighbours along its own freeway, flipped for S and W because postmiles count
the other way.

Detectors are returned ordered along the route, so forecast.route can walk them
with the clock advancing.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("matching")

TOLERANCE_MI = 0.10          # ~530 ft
MAX_BEARING_DIFF = 65.0      # degrees
DIR_BEARING = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
# Postmiles increase northbound and eastbound, so travel runs with increasing
# postmile on N/E and against it on S/W.
_REVERSED = ("S", "W")


def _bearing(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, degrees clockwise from north."""
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(np.radians(lat2))
    x = (np.cos(np.radians(lat1)) * np.sin(np.radians(lat2))
         - np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(dlon))
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def _angular_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def station_bearings(stations):
    """
    True direction of travel at each detector, in degrees clockwise from north.

    Taken from the neighbouring detectors on the same freeway and direction,
    ordered by absolute postmile, and flipped for southbound and westbound so it
    points the way traffic goes rather than the way postmiles count.

    Falls back to the signed direction only where a detector has no neighbour to
    take a bearing from.
    """
    out = pd.Series(stations["Dir"].map(DIR_BEARING).to_numpy(dtype=float),
                    index=stations.index)
    if "Abs_PM" not in stations.columns:
        return out
    for (_, direction), grp in stations.groupby(["Fwy", "Dir"], sort=False):
        g = grp.dropna(subset=["Abs_PM", "Latitude", "Longitude"]).sort_values("Abs_PM")
        if len(g) < 2:
            continue
        lat, lon = g["Latitude"].to_numpy(), g["Longitude"].to_numpy()
        prev_lat, prev_lon = np.r_[lat[0], lat[:-1]], np.r_[lon[0], lon[:-1]]
        next_lat, next_lon = np.r_[lat[1:], lat[-1]], np.r_[lon[1:], lon[-1]]
        b = _bearing(prev_lat, prev_lon, next_lat, next_lon)
        if direction in _REVERSED:
            b = (b + 180.0) % 360.0
        out.loc[g.index] = b
    return out


def match_route(coords, stations, tolerance_mi=TOLERANCE_MI,
                max_bearing_diff=MAX_BEARING_DIFF, cum_mi=None):
    """
    coords    (N,2) array of [lon, lat] from an OSRM geojson geometry
    stations  metadata frame with sensor_id, Latitude, Longitude, Dir, Length
    cum_mi    optional cumulative distance per coordinate. Pass OSRM's own
              annotation distances when you have them: the caller then measures
              spans on exactly the same scale this function assigns positions
              on, and a span cannot be priced against a length it never had.

    Returns the traversed stations ordered along the route, with `along_mi`
    (distance from origin) and `off_mi` (perpendicular offset).
    """
    coords = np.asarray(coords, dtype=float)
    rlat, rlon = coords[:, 1], coords[:, 0]
    if cum_mi is not None:
        cum = np.asarray(cum_mi, dtype=float)
    else:
        seg = np.hypot(np.diff(rlat) * 69.0, np.diff(rlon) * 54.6)
        cum = np.concatenate([[0.0], np.cumsum(seg)])

    # bearing of the route at each vertex (last inherits the previous)
    brg = np.empty(len(rlat))
    brg[:-1] = _bearing(rlat[:-1], rlon[:-1], rlat[1:], rlon[1:])
    brg[-1] = brg[-2] if len(brg) > 1 else 0.0

    slat = stations["Latitude"].to_numpy()
    slon = stations["Longitude"].to_numpy()

    best_d = np.full(len(stations), np.inf)
    best_i = np.zeros(len(stations), dtype=int)
    for a in range(0, len(rlat), 200):          # chunked to bound memory
        b = min(a + 200, len(rlat))
        d = np.hypot((slat[:, None] - rlat[None, a:b]) * 69.0,
                     (slon[:, None] - rlon[None, a:b]) * 54.6)
        j = d.argmin(axis=1)
        v = d[np.arange(len(stations)), j]
        upd = v < best_d
        best_d[upd] = v[upd]
        best_i[upd] = a + j[upd]

    near = best_d <= tolerance_mi
    station_brg = station_bearings(stations).to_numpy(dtype=float)
    aligned = _angular_diff(brg[best_i], station_brg) <= max_bearing_diff
    keep = near & aligned & np.isfinite(station_brg)

    out = stations[keep].copy()
    out["along_mi"] = cum[best_i[keep]]
    out["off_mi"] = best_d[keep]
    out = out.sort_values("along_mi").reset_index(drop=True)

    logger.info("matched %d stations (%d near, %d rejected on heading)",
                len(out), int(near.sum()), int((near & ~aligned).sum()))
    return out, float(cum[-1])
