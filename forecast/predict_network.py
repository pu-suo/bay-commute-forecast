# forecast/predict_network.py
"""
Nightly batch: predict every detector's speed for every slot in the horizon.

A day-ahead forecast is knowable the night before, so there is nothing to serve
on demand. That turns a hosted inference API into a table:

    detector x timestamp -> mph    2,291 x 7 days x 96 slots = 1.5M rows

which is a few megabytes of Parquet, costs nothing to host, and makes a route
query a dictionary lookup.

Features must be assembled by the same functions training used, or the model
receives inputs it has never seen and fails quietly. The one legitimate
difference is the weather source: training used the archived forecast, serving
uses the live forecast, which is the same kind of object.

    python -m forecast.predict_network --days 7 --out ~/traffic-data/serve
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.features_stations import (          # noqa: E402
    CATEGORICAL, NUMERIC, attach_calendar, attach_events, attach_station_attrs,
    attach_weather)

logger = logging.getLogger("predict_network")

SLOT_MINUTES = 15


def horizon_frame(seasonal, start_date, days, slot_minutes=SLOT_MINUTES):
    """
    Cross stations with future slots, then attach each slot's seasonal profile.

    Built as a merge rather than a loop: the seasonal table is already keyed by
    (station, dow, tod), so the horizon is a small frame of dates joined to it.
    Stations with no profile for a slot drop out here rather than reaching the
    model with a missing baseline. The seasonal median is the strongest feature
    by a wide margin, so imputing it would be inventing the answer.
    """
    tods = np.arange(0, 24 * 60, slot_minutes, dtype=np.int16)
    dates = pd.date_range(start_date, periods=days, freq="D")
    cal = pd.DataFrame({"date": np.repeat(dates, len(tods)),
                        "tod": np.tile(tods, len(dates))})
    cal["dow"] = cal["date"].dt.dayofweek.astype(np.int8)

    s = seasonal[seasonal["tod"].isin(tods)]
    df = cal.merge(s, on=["dow", "tod"], how="inner")
    df["ts"] = df["date"] + pd.to_timedelta(df["tod"], unit="m")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--model", default="models/network/model.txt")
    p.add_argument("--weather", default="~/traffic-data/weather")
    p.add_argument("--weather-name", default="live_forecast")
    p.add_argument("--events", default="~/traffic-data/events/events_merged.jsonl")
    p.add_argument("--start", default=None, help="default: tomorrow")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out", default="~/traffic-data/serve")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    import lightgbm as lgb
    data = os.path.expanduser(a.data)
    seasonal = pd.read_parquet(os.path.join(data, "_seasonal.parquet"))
    meta = pd.read_csv(os.path.join(data, "_meta", "d04_meta.txt"), sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    model = lgb.Booster(model_file=os.path.expanduser(a.model))

    start = a.start or str((pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).date())
    started = time.time()
    df = horizon_frame(seasonal, start, a.days)
    logger.info("horizon %s +%dd -> %s station-slots", start, a.days, f"{len(df):,}")

    df = attach_station_attrs(df, meta)
    df = df[df["freeway"].notna()].copy()
    df = attach_calendar(df)
    df = attach_weather(df, a.weather, name=a.weather_name)
    df = attach_events(df, a.events)

    # Categories must be the model's own, in the model's own order. LightGBM
    # stores category codes, so a frame containing a different set of freeways
    # would renumber them.
    import json
    meta_path = os.path.join(os.path.dirname(os.path.expanduser(a.model)), "metrics.json")
    cats = json.load(open(meta_path))["categories"]
    for c in CATEGORICAL:
        vals = df[c].astype(str)
        unknown = ~vals.isin(cats[c])
        if unknown.any():
            # A value the model never saw becomes NaN, which LightGBM treats as
            # a missing category. Acceptable, but log it: this is how a whole
            # freeway loses its identity without anything failing.
            logger.warning("%s: %s rows carry a category the model never saw (%s)",
                           c, f"{int(unknown.sum()):,}",
                           ", ".join(sorted(vals[unknown].unique())[:6]))
        df[c] = pd.Categorical(vals.where(~unknown), categories=cats[c])

    feats = NUMERIC + CATEGORICAL
    df["mph"] = model.predict(df[feats]).astype(np.float32)
    # A tree can extrapolate outside anything physical when features combine in
    # ways the training data never showed; clamp rather than publish 3 mph or
    # 120 mph on a freeway.
    df["mph"] = df["mph"].clip(5.0, 80.0)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    cols = ["station", "ts", "mph", "seasonal_speed"]
    path = os.path.join(out, "forecast.parquet")
    df[cols].to_parquet(path, index=False, compression="zstd")

    # Free-flow per station, for the surface spread model's ratio. Taken from the
    # seasonal profile's upper decile rather than the posted limit: a detector's
    # own quiet-hour speed is what "uncongested here" actually means.
    ff = (seasonal.groupby("station")["seasonal_speed"].quantile(0.9)
                  .rename("freeflow").reset_index())
    ff.to_parquet(os.path.join(out, "freeflow.parquet"), index=False)

    logger.info("wrote %s rows -> %s (%.0fs, %.1f MB)", f"{len(df):,}", path,
                time.time() - started, os.path.getsize(path) / 1e6)
    logger.info("mph: p05 %.1f  median %.1f  p95 %.1f",
                *np.percentile(df["mph"], [5, 50, 95]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
