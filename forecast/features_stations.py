# forecast/features_stations.py
"""
Feature assembly at network scale — all 2,291 stations rather than nine corridors.

The corridor pipeline loads everything into pandas. That does not survive the
jump to 1.35 billion station-intervals, so two things change:

1. STATION IS NOT A CATEGORICAL. The corridor model used `corridor` as a
   category, which cannot generalise to a station it never saw. Here the model
   is given station *attributes* instead — freeway, direction, lane count, and
   the station's own seasonal median, which carries its typical level. That
   means it can be trained on a sample of stations and applied to all of them,
   which is the only affordable way to cover the network.

2. THE SEASONAL TABLE IS PRECOMPUTED AND SMALL. Rather than a rolling window
   over every row, the median and sd per (station, weekday, time-of-day) are
   accumulated streaming across files into a 4.6M-row lookup. Crucially it is
   fitted on the TRAINING period only and applied forward, which is both causal
   with respect to the test set and exactly what a production system does:
   fit the seasonal profile on history, then serve it.

Row sampling happens after the lookup is built, so sampling never corrupts the
statistics the way sampling before a rolling computation would.
"""
import glob
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("features_stations")

TRAIN_END = "2025-01-01"
SAMPLE_EVERY_NTH_INTERVAL = 3      # 15-minute rows for training; serving stays 5-min
MIN_OBS_FOR_SEASONAL = 4


def _day_paths(data_dir):
    return sorted(glob.glob(os.path.join(os.path.expanduser(data_dir),
                                         "year=*", "*.parquet")))


def _date_of(path):
    return pd.Timestamp(os.path.basename(path)[9:19].replace("_", "-"))


def build_seasonal(data_dir, train_end=TRAIN_END, min_pct_observed=20):
    """
    Streaming mean/sd per (station, weekday, time-of-day) over the train span.

    Accumulated into dense numpy arrays rather than by adding pandas Series.
    The key space is bounded -- 2,291 stations x 7 weekdays x 288 intervals, so
    ~4.6M cells, about 110 MB across the three accumulators -- while aligning
    660k-entry Series 1,461 times is quadratic-ish in practice and does not
    finish. np.add.at on flat indices is O(rows) per day.

    Exact medians would need every value retained, so this keeps sums and
    sums-of-squares and reports the mean. On 5-minute speeds the two are close,
    and the sd is passed to the model separately so it can learn where they
    diverge.
    """
    paths = [p for p in _day_paths(data_dir) if _date_of(p) < pd.Timestamp(train_end)]
    logger.info("seasonal table from %d training days", len(paths))

    # station id -> dense index, discovered from the newest training day
    probe = pd.read_parquet(paths[-1], columns=["station"])
    stations = np.sort(probe["station"].unique())
    sidx = pd.Series(np.arange(len(stations)), index=stations)
    n_st, n_tod = len(stations), 288

    shape = (n_st, 7, n_tod)
    cnt = np.zeros(shape, dtype=np.int32)
    ssum = np.zeros(shape, dtype=np.float64)
    sqsum = np.zeros(shape, dtype=np.float64)

    for i, path in enumerate(paths, 1):
        d = pd.read_parquet(path, columns=["station", "tod", "speed", "pct_observed"])
        d = d[(d["pct_observed"] >= min_pct_observed) & d["station"].isin(sidx.index)]
        if d.empty:
            continue
        si = sidx.reindex(d["station"]).to_numpy()
        ti = (d["tod"].to_numpy() // 5).astype(np.int64)
        dow = _date_of(path).dayofweek
        flat = (si * 7 + dow) * n_tod + ti
        sp = d["speed"].to_numpy(dtype=np.float64)
        np.add.at(cnt.reshape(-1), flat, 1)
        np.add.at(ssum.reshape(-1), flat, sp)
        np.add.at(sqsum.reshape(-1), flat, sp * sp)
        if i % 250 == 0:
            logger.info("  %d/%d days", i, len(paths))

    keep = cnt >= MIN_OBS_FOR_SEASONAL
    st_i, dw_i, td_i = np.nonzero(keep)
    n = cnt[keep].astype(np.float64)
    mean = ssum[keep] / n
    var = (sqsum[keep] / n) - mean ** 2
    out = pd.DataFrame({
        "station": stations[st_i].astype(np.int32),
        "dow": dw_i.astype(np.int8),
        "tod": (td_i * 5).astype(np.int16),
        "seasonal_speed": mean.astype(np.float32),
        "seasonal_sd": np.sqrt(np.clip(var, 0, None)).astype(np.float32),
    })
    logger.info("seasonal table: %d (station,dow,tod) cells", len(out))
    return out


def sample_rows(data_dir, seasonal, start, end, every_nth=SAMPLE_EVERY_NTH_INTERVAL,
                station_frac=1.0, min_pct_observed=20, seed=0):
    """Load a subsampled row set for one date range and attach seasonal stats."""
    rng = np.random.default_rng(seed)
    keep_stations = None
    if station_frac < 1.0:
        allst = seasonal["station"].unique()
        keep_stations = set(rng.choice(allst, int(len(allst) * station_frac), replace=False))
        logger.info("sampling %d of %d stations", len(keep_stations), len(allst))

    frames = []
    paths = [p for p in _day_paths(data_dir)
             if pd.Timestamp(start) <= _date_of(p) < pd.Timestamp(end)]
    for p in paths:
        d = pd.read_parquet(p, columns=["station", "tod", "speed", "pct_observed"])
        d = d[(d["pct_observed"] >= min_pct_observed) & (d["tod"] % (5 * every_nth) == 0)]
        if keep_stations is not None:
            d = d[d["station"].isin(keep_stations)]
        if d.empty:
            continue
        date = _date_of(p)
        d["date"] = date
        d["dow"] = date.dayofweek
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(seasonal, on=["station", "dow", "tod"], how="inner")
    logger.info("%s..%s -> %s rows", start, end, f"{len(df):,}")
    return df


def attach_station_attrs(df, meta):
    """Generalisable station attributes, so unseen stations still get a prediction."""
    m = meta.copy()
    m["station"] = m["sensor_id"].astype(int)
    cols = ["station", "Fwy", "Dir", "Lanes", "Length", "Latitude", "Longitude"]
    df = df.merge(m[cols], on="station", how="left")
    return df.rename(columns={"Fwy": "freeway", "Dir": "direction",
                              "Lanes": "lanes", "Length": "seg_miles"})


def attach_calendar(df):
    import holidays
    years = sorted(df["date"].dt.year.unique())
    us = holidays.US(years=list(years), state="CA")
    hset = set(us)
    adj = {h + pd.Timedelta(days=k) for h in hset for k in (-1, 1)}
    adj = {a.date() if hasattr(a, "date") else a for a in adj} - hset
    d = df["date"].dt.date
    df["holiday_class"] = np.where(d.isin(hset), "holiday",
                          np.where(d.isin(adj), "adjacent", "none"))
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


NUMERIC = ["seasonal_speed", "seasonal_sd", "tod", "dow", "month", "is_weekend",
           "lanes", "seg_miles", "Latitude", "Longitude"]
CATEGORICAL = ["freeway", "direction", "holiday_class"]
TARGET = "speed"
