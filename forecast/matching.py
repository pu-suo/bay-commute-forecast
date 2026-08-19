# forecast/matching.py
"""
Match a route polyline onto the PeMS stations it actually traverses.

The naive version — nearest station within a tolerance — is wrong in a way that
is easy to miss and doubles every answer: opposing carriageways of a divided
freeway sit within a couple of hundred feet of each other, so a northbound route
matches the southbound detectors too. On a Palo Alto to SFO route that produced
39 northbound and 41 southbound stations, and 134% of the route's own length as
"instrumented".

So matching is two tests, not one:

  proximity   the station is within TOLERANCE_MI of the route line, and
  heading     the route's local bearing agrees with the station's direction
              of travel to within MAX_BEARING_DIFF degrees.

Stations are then ordered by distance along the route, which is what makes
time-dependent traversal possible: `forecast.route` accumulates each segment's
travel time and advances the clock as it goes, so a long trip is evaluated
against the forecast for the time the driver will actually *be* there rather
than the time they left.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("matching")

TOLERANCE_MI = 0.10          # ~530 ft
MAX_BEARING_DIFF = 65.0      # degrees
DIR_BEARING = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


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
    station_brg = stations["Dir"].map(DIR_BEARING).to_numpy(dtype=float)
    aligned = _angular_diff(brg[best_i], station_brg) <= max_bearing_diff
    keep = near & aligned & np.isfinite(station_brg)

    out = stations[keep].copy()
    out["along_mi"] = cum[best_i[keep]]
    out["off_mi"] = best_d[keep]
    out = out.sort_values("along_mi").reset_index(drop=True)

    logger.info("matched %d stations (%d near, %d rejected on heading)",
                len(out), int(near.sum()), int((near & ~aligned).sum()))
    return out, float(cum[-1])
