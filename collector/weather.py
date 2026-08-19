# collector/weather.py
"""
Weather features from Open-Meteo — free, no API key.

Three endpoints for three different jobs, and picking the wrong one is the
classic way to build a model that works offline and fails in production:

  Historical Forecast API   the forecast that *was issued* on a past date.
                            This is what you TRAIN on, because it is the kind
                            of input the model will actually receive at serve
                            time. Coverage starts ~2022.
  Forecast API              tomorrow's forecast. What you SERVE with.
  Historical Weather (ERA5) what actually happened. Analysis only — training on
                            it gives the model perfect weather knowledge it will
                            never have, and the model silently over-relies on it.

Corridor midpoints are rounded before deduplication: paired directions on the
same freeway sit within a few hundred metres of each other, so nine corridors
collapse to about five distinct weather locations and five times fewer calls.

    python -m collector.weather --start 2022-01-01 --end 2026-08-15 \
        --out ~/traffic-data/weather
"""
import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LIVE_FORECAST = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = ["temperature_2m", "precipitation", "wind_speed_10m", "visibility"]
TIMEZONE = "America/Los_Angeles"

REQUEST_TIMEOUT = 90
RETRY_ATTEMPTS = 3
POLITENESS_SECONDS = 1.5
# ~2 decimal places is a little over 1 km — well inside a weather grid cell, and
# enough to collapse the paired-direction corridors onto one request each.
COORD_PRECISION = 2

logger = logging.getLogger("weather")


def _get(url, params):
    q = f"{url}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "commute-forecast/1.0"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                payload = json.loads(r.read())
            time.sleep(POLITENESS_SECONDS)
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            logger.warning("open-meteo attempt %d/%d failed: %s", attempt, RETRY_ATTEMPTS, e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(5 * attempt)
    raise RuntimeError(f"open-meteo unreachable: {last}")


def corridor_points(meta, corridors):
    """{(lat,lon): [corridor_slug, ...]} at reduced precision."""
    points = {}
    for c in corridors:
        stations = c.stations(meta)
        if stations.empty:
            continue
        mid = stations.iloc[len(stations) // 2]
        key = (round(float(mid["Latitude"]), COORD_PRECISION),
               round(float(mid["Longitude"]), COORD_PRECISION))
        points.setdefault(key, []).append(c.slug)
    return points


def grid_points(meta, step):
    """
    {(lat,lon): [cell_label]} for every grid cell that actually contains a
    mainline station.

    Corridor midpoints were enough for nine corridors, but the network spans
    Gilroy to Ukiah -- 130 miles -- and nearest-point assignment from five
    distinct locations would hand a Santa Rosa station the weather in Oakland.
    A regular grid over the occupied cells is coarse in the right way: it is
    wrong by at most half a cell everywhere, instead of nearly right in five
    places and badly wrong elsewhere.

    Empty cells are skipped. Of the 90 cells in the bounding box only 30 hold a
    station, so the grid costs a third of what its extent suggests.
    """
    import numpy as np
    ml = meta[meta["Type"] == "ML"]
    lat = (np.floor(ml["Latitude"] / step) * step + step / 2).round(3)
    lon = (np.floor(ml["Longitude"] / step) * step + step / 2).round(3)
    cells = {}
    for la, lo in set(zip(lat, lon)):
        cells[(float(la), float(lo))] = [f"cell_{la}_{lo}"]
    return dict(sorted(cells.items()))


def fetch_archived_forecast(lat, lon, start, end):
    """Hourly archived-forecast rows for one point and date range."""
    data = _get(HIST_FORECAST, {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    })
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    return [
        {"ts": times[i], "lat": lat, "lon": lon,
         **{v: (hourly.get(v) or [None] * len(times))[i] for v in HOURLY_VARS}}
        for i in range(len(times))
    ]


def fetch_live_forecast(lat, lon, days=7):
    """Hourly forecast going forward -- what the nightly job actually serves on."""
    data = _get(LIVE_FORECAST, {
        "latitude": lat, "longitude": lon,
        "forecast_days": days,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    })
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    return [
        {"ts": times[i], "lat": lat, "lon": lon,
         **{v: (hourly.get(v) or [None] * len(times))[i] for v in HOURLY_VARS}}
        for i in range(len(times))
    ]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-15")
    p.add_argument("--out", default="~/traffic-data/weather")
    p.add_argument("--meta", default="~/traffic-data/corridors/_meta/d04_meta.txt")
    p.add_argument("--grid-step", type=float, default=None,
                   help="degrees; when set, cover occupied grid cells instead of "
                        "corridor midpoints (network-scale mode)")
    p.add_argument("--live", action="store_true",
                   help="fetch the forward forecast instead of the archive")
    p.add_argument("--forecast-days", type=int, default=7)
    p.add_argument("--name", default="archived_forecast",
                   help="output basename, so grid and corridor sets coexist")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from forecast.corridors import CORRIDORS
    from collector.pems_client import parse_station_meta
    import pandas as pd

    meta = parse_station_meta(os.path.expanduser(a.meta))
    if a.grid_step:
        points = grid_points(meta, a.grid_step)
        logger.info("%.2f deg grid -> %d occupied cells", a.grid_step, len(points))
    else:
        points = corridor_points(meta, CORRIDORS)
        logger.info("%d corridors -> %d distinct weather points",
                    len(CORRIDORS), len(points))

    rows = []
    if a.live:
        for (lat, lon), slugs in points.items():
            try:
                got = fetch_live_forecast(lat, lon, a.forecast_days)
                rows.extend(got)
            except RuntimeError as ex:
                logger.error("  %.2f,%.2f live FAILED: %s", lat, lon, ex)
        logger.info("live forecast: %d hourly rows over %d points",
                    len(rows), len(points))
    for (lat, lon), slugs in (() if a.live else points.items()):
        # chunk by year: one multi-year request is fine for the API but a failure
        # mid-range would otherwise cost the whole point
        for year in range(int(a.start[:4]), int(a.end[:4]) + 1):
            s = max(f"{year}-01-01", a.start)
            e = min(f"{year}-12-31", a.end)
            if s > e:
                continue
            try:
                got = fetch_archived_forecast(lat, lon, s, e)
                rows.extend(got)
                logger.info("  %.2f,%.2f %s: %d hourly rows  (%s)",
                            lat, lon, year, len(got), ",".join(slugs))
            except RuntimeError as ex:
                logger.error("  %.2f,%.2f %s FAILED: %s", lat, lon, year, ex)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    path = os.path.join(out, f"{a.name}.parquet")
    df.to_parquet(path, index=False)

    mapping = {f"{lat},{lon}": slugs for (lat, lon), slugs in points.items()}
    with open(os.path.join(out, f"{a.name}_points.json"
                           if a.name != "archived_forecast" else "points.json"), "w") as f:
        json.dump(mapping, f, indent=1)

    logger.info("wrote %d rows -> %s", len(df), path)
    logger.info("rain hours: %d (%.1f%%)",
                int((df["precipitation"] > 0).sum()),
                (df["precipitation"] > 0).mean() * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
