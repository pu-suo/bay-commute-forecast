# site/build_data.py
"""
Turn the nightly forecast table into the two files the browser reads.

The site is static by construction. A day-ahead forecast is fully knowable the
night before, so there is no query a server needs to answer at page load, and
the whole map is a file:

  network.json    an hourly speed series per detector, plus its free-flow speed
  corridors.json  the nine named commutes, minute-by-minute across the week

Road geometry is NOT here -- it comes from `build_geometry.py`, which reads real
OSM centrelines. The two are keyed to each other by detector id, so geometry is
rebuilt only when the road network changes while speeds are rebuilt nightly.

Hourly, not five-minute. The serving table holds 15-minute resolution and the
route planner uses all of it, but a map that animates across a week needs 168
frames rather than 672, and the difference is a megabyte the visitor waits on
for no visual gain. Resolution is a rendering decision here, not a modelling one.

"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger("build_data")

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", default="~/traffic-data/serve")
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--events", default="~/traffic-data/events/events_merged.jsonl")
    p.add_argument("--models", default="models/network",
                   help="metrics are copied next to the data so the accuracy "
                        "page is served from the same static directory")
    p.add_argument("--out", default="site/data")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from forecast.corridors import CORRIDORS, VENUES_BY_SLUG

    serve = os.path.expanduser(a.serve)
    fc = pd.read_parquet(os.path.join(serve, "forecast.parquet"))
    ff = pd.read_parquet(os.path.join(serve, "freeflow.parquet"))
    meta = pd.read_csv(os.path.join(os.path.expanduser(a.data), "_meta", "d04_meta.txt"),
                       sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})

    hourly = fc[fc["ts"].dt.minute == 0].copy()
    slots = sorted(hourly["ts"].unique())
    slot_index = {t: i for i, t in enumerate(slots)}
    logger.info("%d hourly slots, %s -> %s", len(slots), slots[0], slots[-1])

    h = hourly

    # dense (station, slot) speed matrix, so the browser gets a flat array per
    # station rather than 400k little objects to parse
    stations = np.sort(h["station"].unique())
    sidx = {int(s): i for i, s in enumerate(stations)}
    grid = np.full((len(stations), len(slots)), -1, dtype=np.int16)
    si = h["station"].map(sidx).to_numpy()
    ti = h["ts"].map(slot_index).to_numpy()
    grid[si, ti] = np.rint(h["mph"].to_numpy()).astype(np.int16)

    freeflow = ff.set_index("station")["freeflow"].to_dict()
    network = {
        "slots": [pd.Timestamp(t).isoformat() for t in slots],
        "speeds": {str(int(s)): grid[sidx[int(s)]].tolist() for s in stations},
        "freeflow": {str(int(s)): round(float(freeflow.get(int(s), 65.0)), 1)
                     for s in stations},
    }
    out = a.out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "network.json"), "w") as f:
        json.dump(network, f, separators=(",", ":"))
    logger.info("network.json: %d detectors x %d slots, %.1f MB", len(stations),
                len(slots), os.path.getsize(os.path.join(out, "network.json")) / 1e6)

    # ---- the nine named commutes, as a week of travel times -----------------
    speed = fc.set_index(["station", "ts"])["mph"]
    base = fc.set_index(["station", "ts"])["seasonal_speed"]
    corridors = []
    for c in CORRIDORS:
        st = c.stations(meta)
        if st.empty:
            continue
        ids = st["sensor_id"].astype(int).tolist()
        # spacing tiles the corridor; the last detector governs to the end
        pm = st["Abs_PM"].to_numpy()
        spans = np.abs(np.diff(np.concatenate([pm, [pm[-1] + (pm[-1] - pm[-2])]])))
        sub = fc[fc["station"].isin(ids)]
        piv = sub.pivot_table(index="ts", columns="station", values="mph")
        pivb = sub.pivot_table(index="ts", columns="station", values="seasonal_speed")
        order = [i for i in ids if i in piv.columns]
        w = pd.Series(spans, index=ids).reindex(order).to_numpy()
        total_mi = float(np.abs(spans).sum())

        # Missing detectors must never shorten the trip. A sum over whatever
        # happens to be present reads as free-flowing traffic rather than as
        # missing data -- the same trap the historical pipeline hit -- so the
        # covered portion is priced and then scaled up to the full corridor.
        def scaled(matrix):
            mph = matrix.to_numpy()
            ok = np.isfinite(mph) & (mph > 0)
            covered = (np.where(ok, w, 0.0)).sum(axis=1)
            mins = np.nansum(np.where(ok, w / np.where(ok, mph, 1) * 60, 0.0), axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(covered > 0, mins * total_mi / covered, np.nan), covered

        minutes, covered = scaled(piv[order])
        base_min, _ = scaled(pivb[order])
        cov_pct = float(np.nanmean(covered) / total_mi * 100)
        corridors.append({
            "slug": c.slug, "name": c.name, "miles": round(total_mi, 1),
            "coverage_pct": round(cov_pct, 1),
            "ts": [pd.Timestamp(t).isoformat() for t in piv.index],
            "minutes": [round(float(x), 1) for x in minutes],
            "typical": [round(float(x), 1) for x in base_min],
            "coords": [[round(float(r.Latitude), 5), round(float(r.Longitude), 5)]
                       for r in st.itertuples()],
        })
        logger.info("  %-34s %5.1f mi  %5.1f-%5.1f min   %3.0f%% instrumented",
                    c.slug, total_mi, np.nanmin(minutes), np.nanmax(minutes), cov_pct)

    # upcoming events, so the site can explain an unusual evening
    events = []
    horizon_end = fc["ts"].max()
    for line in open(os.path.expanduser(a.events)):
        if not line.strip():
            continue
        e = json.loads(line)
        start = e["start"]
        ts = pd.Timestamp(start if "T" in start else start + "T19:00:00")
        if not (fc["ts"].min() <= ts <= horizon_end):
            continue
        v = VENUES_BY_SLUG.get(e["venue"])
        if not v:
            continue
        events.append({"ts": ts.isoformat(), "title": e.get("title", ""),
                       "venue": v.name, "lat": v.lat, "lon": v.lon,
                       "capacity": v.capacity})
    events.sort(key=lambda x: x["ts"])

    with open(os.path.join(out, "corridors.json"), "w") as f:
        json.dump({"corridors": corridors, "events": events,
                   "generated": pd.Timestamp.now().isoformat()}, f,
                  separators=(",", ":"))
    logger.info("corridors.json: %d corridors, %d upcoming events",
                len(corridors), len(events))

    # The accuracy page is the product's central claim, so its inputs travel
    # with the site rather than being re-derived by the browser.
    import shutil
    for name in ("metrics.json", "route_metrics.json"):
        src = os.path.join(a.models, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, name))
            logger.info("copied %s", name)
        else:
            logger.warning("missing %s -- accuracy page will show a gap", src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
