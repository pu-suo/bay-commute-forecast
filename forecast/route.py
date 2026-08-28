# forecast/route.py
"""
Origin and destination in, day-ahead travel time out.

A route is cut into spans and each span is priced by whatever evidence covers
it, with the clock advancing across them:

  freeway   a PeMS detector governs it. Priced from the forecast table and
            scored on the accuracy page.
  surface   no detector within tolerance. Priced by scaling OSRM's own
            free-flow duration by the spread model. Labelled as an estimate
            everywhere it appears and excluded from accuracy figures.

The two are kept apart throughout. A single number blending a measured forecast
with an unvalidated proxy leaves nobody able to tell which part failed.

Spans are walked in order, so a segment forty minutes into a trip is priced at
the forecast for the time the driver reaches it.

    python -m forecast.route --from 37.4419,-122.1430 --to 37.6213,-122.3790 \
        --depart "2026-08-19 08:00"
"""
import argparse
import json
import logging
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.matching import match_route          # noqa: E402
from forecast.surface import ALPHA, MIN_SPEED_MPH  # noqa: E402

logger = logging.getLogger("route")

OSRM = os.environ.get("OSRM_URL", "http://localhost:5001")
# Mainline detectors sit ~0.6 mi apart. A gap much larger than that means the
# route left the instrumented freeway (an off-ramp, an arterial, a rural
# stretch), and one detector cannot speak for six miles of surface street.
MAX_FREEWAY_SPAN_MI = 2.5
SURFACE_RADIUS_MI = 2.0
SURFACE_MAX_STATIONS = 6


def osrm_route(origin, dest, base=OSRM):
    """(coords[lon,lat], cumulative_miles, cumulative_freeflow_minutes)."""
    coords = f"{origin[1]},{origin[0]};{dest[1]},{dest[0]}"
    q = urllib.parse.urlencode({"overview": "full", "geometries": "geojson",
                                "annotations": "duration,distance"})
    url = f"{base}/route/v1/driving/{coords}?{q}"
    with urllib.request.urlopen(url, timeout=30) as r:
        payload = json.load(r)
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM: {payload.get('code')} {payload.get('message','')}")
    route = payload["routes"][0]
    pts = np.array(route["geometry"]["coordinates"], dtype=float)

    ann_d, ann_t = [], []
    for leg in route["legs"]:
        ann_d.extend(leg["annotation"]["distance"])
        ann_t.extend(leg["annotation"]["duration"])
    # annotations are per edge between consecutive points; prepend the origin
    cum_mi = np.concatenate([[0.0], np.cumsum(np.array(ann_d) / 1609.344)])
    cum_min = np.concatenate([[0.0], np.cumsum(np.array(ann_t) / 60.0)])
    return pts, cum_mi, cum_min


class Forecast:
    """The nightly table, indexed for lookup by (station, timestamp)."""

    def __init__(self, serve_dir, stations_meta):
        d = os.path.expanduser(serve_dir)
        fc = pd.read_parquet(os.path.join(d, "forecast.parquet"))
        self.slot = int(pd.Series(sorted(fc["ts"].unique()[:2]))
                        .diff().dropna().dt.total_seconds().iloc[0] // 60) or 15
        # Convert through datetime64[s] rather than dividing raw ints. Parquet
        # round-trips these as microseconds while Timestamp.value is always
        # nanoseconds, so //10**9 keys the two sides a thousandfold apart and
        # every lookup misses.
        fc["key"] = (fc["station"].astype(np.int64) * 10 ** 12 +
                     fc["ts"].to_numpy().astype("datetime64[s]").astype(np.int64))
        self.mph = dict(zip(fc["key"], fc["mph"]))
        self.seasonal = dict(zip(fc["key"], fc["seasonal_speed"]))
        # Tables written before the spread was published have no column; the
        # band is then simply unavailable rather than invented.
        self.sd = (dict(zip(fc["key"], fc["seasonal_sd"]))
                   if "seasonal_sd" in fc.columns else {})
        self.freeflow = (pd.read_parquet(os.path.join(d, "freeflow.parquet"))
                           .set_index("station")["freeflow"].to_dict())
        self.meta = stations_meta
        self.start = fc["ts"].min()
        self.end = fc["ts"].max()

    def _key(self, station, ts):
        snapped = pd.Timestamp(ts).floor(f"{self.slot}min")
        return int(station) * 10 ** 12 + int(snapped.timestamp())

    def speed(self, station, ts):
        return self.mph.get(self._key(station, ts))

    def spread(self, station, ts):
        """Historical sd of this detector at this weekday and time, in mph."""
        return self.sd.get(self._key(station, ts))

    def ratio(self, station, ts):
        """predicted / free-flow for one station: the spread model's input."""
        mph = self.speed(station, ts)
        ff = self.freeflow.get(int(station))
        if mph is None or not ff:
            return None
        return float(np.clip(mph / ff, 0.15, 1.05))


def _spread_factor(fc, lat, lon, ts, radius_mi=SURFACE_RADIUS_MI,
                   max_stations=SURFACE_MAX_STATIONS, alpha=ALPHA):
    """
    Inverse-distance-squared weighted freeway ratio near a point, damped by alpha.

    Returns (factor, n_stations, mean_ratio). factor of 1.0 means free-flow, so
    a point with no freeway within the radius gets no adjustment rather than an
    invented one.
    """
    m = fc.meta
    d = np.hypot((m["Latitude"].to_numpy() - lat) * 69.0,
                 (m["Longitude"].to_numpy() - lon) * 54.6)
    near = np.where(d <= radius_mi)[0]
    if near.size == 0:
        return 1.0, 0, None
    near = near[np.argsort(d[near])][:max_stations]
    ids = m["sensor_id"].astype(int).to_numpy()[near]

    ratios, weights = [], []
    for sid, dist in zip(ids, d[near]):
        r = fc.ratio(sid, ts)
        if r is None:
            continue
        ratios.append(r)
        weights.append(1.0 / max(dist, 0.15) ** 2)
    if not ratios:
        return 1.0, 0, None
    w = np.array(weights) / np.sum(weights)
    ratio = float(np.dot(w, ratios))
    return 1.0 - alpha * (1.0 - ratio), len(ratios), ratio


def build_spans(matched, route_mi):
    """
    Cut the route into freeway spans (one detector each) and surface spans.

    Each matched station governs from where it sits to the next one. A span
    longer than MAX_FREEWAY_SPAN_MI is reclassified as surface, since the
    detector behind it stopped being evidence somewhere in the middle.
    """
    spans = []
    if matched.empty:
        return [{"kind": "surface", "start_mi": 0.0, "end_mi": route_mi,
                 "sensor_id": None}]

    along = matched["along_mi"].to_numpy()
    if along[0] > 0.01:
        spans.append({"kind": "surface", "start_mi": 0.0,
                      "end_mi": float(along[0]), "sensor_id": None})

    edges = np.concatenate([along, [route_mi]])
    for i, row in enumerate(matched.itertuples()):
        a, b = float(edges[i]), float(edges[i + 1])
        if b - a <= 0:
            continue
        kind = "freeway" if (b - a) <= MAX_FREEWAY_SPAN_MI else "surface"
        spans.append({"kind": kind, "start_mi": a, "end_mi": b,
                      "sensor_id": int(row.sensor_id) if kind == "freeway" else None,
                      "lat": float(row.Latitude), "lon": float(row.Longitude),
                      "freeway": row.Fwy, "direction": row.Dir})
    return spans


# The band is a spread of days, not a confidence interval on the model. Each
# detector carries seasonal_sd, the historical variation at this weekday and
# time; converted to minutes through the local derivative of miles/speed
# (d_min = miles * 60 * sd / mph^2) rather than by repricing at speed-sd, which
# explodes as speed falls and produced a 78-minute upper edge on a drive whose
# worst real Friday was 57.
#
# Summing those per-span deviations as if perfectly correlated still overstates
# the route, so the sum is scaled. The two factors were fitted on 20 route-days
# against the p10-p90 of real matching weekdays: mean edge error 3.4 min on the
# slow side and 2.5 on the fast, over drives of 27 to 70 minutes. They are
# asymmetric because the real spread is: a day can go far more wrong than right.
#
# Known weakness: the Bay Bridge runs wider than this band. Its bottleneck is
# the toll plaza, upstream of every detector, the same reason it is the one
# corridor in the accuracy table that loses to the seasonal baseline.
BAND_SLOW = 0.40
BAND_FAST = 0.25


def price(spans, fc, pts, cum_mi, cum_min, depart):
    """Walk the spans in order, advancing the clock, pricing each by its evidence."""
    t = pd.Timestamp(depart)
    out = []
    for sp in spans:
        miles = sp["end_mi"] - sp["start_mi"]
        if miles <= 0:
            continue
        mid = (sp["start_mi"] + sp["end_mi"]) / 2
        j = int(np.clip(np.searchsorted(cum_mi, mid), 0, len(pts) - 1))
        lat, lon = float(pts[j][1]), float(pts[j][0])

        if sp["kind"] == "freeway":
            mph = fc.speed(sp["sensor_id"], t)
            if mph:
                minutes = miles / mph * 60.0
                sd = fc.spread(sp["sensor_id"], t) or 0.0
                out.append({**sp, "miles": miles, "mph": float(mph),
                            "minutes": minutes, "arrive": t, "estimate": False,
                            "minutes_sd": miles * 60.0 * sd / max(float(mph), MIN_SPEED_MPH) ** 2,
                            "lat": lat, "lon": lon})
                t += pd.Timedelta(minutes=minutes)
                continue
            sp = {**sp, "kind": "surface", "sensor_id": None}   # no forecast: fall back

        # free-flow duration for exactly this stretch, from OSRM's own profile
        ff_min = float(np.interp(sp["end_mi"], cum_mi, cum_min) -
                       np.interp(sp["start_mi"], cum_mi, cum_min))
        factor, n, ratio = _spread_factor(fc, lat, lon, t)
        ff_speed = miles / ff_min * 60.0 if ff_min > 0 else 30.0
        est_speed = max(ff_speed * factor, MIN_SPEED_MPH)
        minutes = miles / est_speed * 60.0
        out.append({**sp, "miles": miles, "mph": est_speed, "minutes": minutes,
                    "minutes_sd": 0.0,
                    "arrive": t, "estimate": True, "spread_factor": factor,
                    "spread_stations": n, "freeway_ratio": ratio,
                    "lat": lat, "lon": lon})
        t += pd.Timedelta(minutes=minutes)
    return pd.DataFrame(out)


class PreparedRoute:
    """
    The part of a route that does not depend on departure time.

    OSRM's path, the detectors it passes and the spans between them are the same
    whatever time you leave; only the prices change. Splitting them apart lets an
    all-day profile call OSRM once instead of once per slot.
    """

    __slots__ = ("pts", "cum_mi", "cum_min", "matched", "spans", "route_mi")

    def __init__(self, origin, dest, meta, base=OSRM):
        self.pts, self.cum_mi, self.cum_min = osrm_route(origin, dest, base)
        self.route_mi = float(self.cum_mi[-1])
        self.matched, _ = match_route(self.pts, meta, cum_mi=self.cum_mi)
        self.spans = build_spans(self.matched, self.route_mi)


def price_at(prepared, depart, fc):
    """Price an already-prepared route for one departure time."""
    segs = price(prepared.spans, fc, prepared.pts, prepared.cum_mi,
                 prepared.cum_min, depart)
    fwy = segs[~segs["estimate"]] if len(segs) else segs
    srf = segs[segs["estimate"]] if len(segs) else segs
    total = float(segs["minutes"].sum()) if len(segs) else 0.0
    summary = {
        "depart": str(pd.Timestamp(depart)),
        "arrive": str(pd.Timestamp(depart) + pd.Timedelta(minutes=total)),
        "total_minutes": total,
        "freeway_minutes": float(fwy["minutes"].sum()) if len(fwy) else 0.0,
        "surface_minutes": float(srf["minutes"].sum()) if len(srf) else 0.0,
        "route_miles": prepared.route_mi,
        "freeway_miles": float(fwy["miles"].sum()) if len(fwy) else 0.0,
        "surface_miles": float(srf["miles"].sum()) if len(srf) else 0.0,
        "osrm_freeflow_minutes": float(prepared.cum_min[-1]),
        "stations_used": int(len(fwy)),
    }
    summary["measured_share"] = (summary["freeway_minutes"] / total) if total else 0.0

    # The band, scaled onto the surface share too. Surface speed is derived from
    # the freeway ratio, so a day that is slow on the freeway is slow on the
    # arterials feeding it; holding surface fixed would understate the spread on
    # exactly the city routes where surface is most of the trip.
    fw = summary["freeway_minutes"]
    if len(fwy) and fw > 0:
        sd_min = float(fwy["minutes_sd"].sum()) * (1.0 + summary["surface_minutes"] / fw)
        summary["typical_slow"] = total + BAND_SLOW * sd_min
        summary["typical_fast"] = max(total - BAND_FAST * sd_min, 1.0)
    else:
        # No detector on the route: there is no measured spread to report, and a
        # made-up one would be worse than none.
        summary["typical_slow"] = summary["typical_fast"] = None
    return segs, summary


def plan(origin, dest, depart, fc, base=OSRM):
    """Full route forecast for one departure time. Returns (segments, summary)."""
    return price_at(PreparedRoute(origin, dest, fc.meta, base), depart, fc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="origin", required=True, help="lat,lon")
    p.add_argument("--to", dest="dest", required=True, help="lat,lon")
    p.add_argument("--depart", required=True)
    p.add_argument("--serve", default="~/traffic-data/serve")
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--osrm", default=OSRM)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    meta = pd.read_csv(os.path.join(os.path.expanduser(a.data), "_meta", "d04_meta.txt"),
                       sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    fc = Forecast(a.serve, meta)

    origin = tuple(float(x) for x in a.origin.split(","))
    dest = tuple(float(x) for x in a.dest.split(","))
    segs, s = plan(origin, dest, a.depart, fc, a.osrm)

    logger.info("\n%s -> %s   depart %s", a.origin, a.dest, s["depart"])
    logger.info("%.1f min   arrive %s", s["total_minutes"], s["arrive"][11:16])
    logger.info("  freeway  %5.1f mi  %5.1f min   (%d detectors, forecast)",
                s["freeway_miles"], s["freeway_minutes"], s["stations_used"])
    logger.info("  surface  %5.1f mi  %5.1f min   (estimate, not scored)",
                s["surface_miles"], s["surface_minutes"])
    logger.info("  OSRM free-flow reference: %.1f min", s["osrm_freeflow_minutes"])
    logger.info("  %.0f%% of the journey time is measured forecast",
                s["measured_share"] * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
