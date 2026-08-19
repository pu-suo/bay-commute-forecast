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


def split_stations(seasonal, holdout_frac=0.30, seed=7):
    """
    Partition stations into a training pool and a spatial holdout.

    The holdout is the point of the whole design. Training on a sample of
    stations is only defensible if the model then works on stations whose rows
    it never saw, and the only way to know that is to withhold some and look.
    A purely temporal split cannot answer it: every station appears in both
    halves, so the model can memorise station-specific level and still score
    well.

    The holdout stations keep their seasonal profile, because in production
    every station has one -- the profile is computed from history for all 2,291
    regardless of which rows the model trained on. What is withheld is the
    chance to learn that station's idiosyncrasies.
    """
    allst = np.sort(seasonal["station"].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(allst)
    n_hold = int(len(allst) * holdout_frac)
    holdout = set(shuffled[:n_hold].tolist())
    pool = set(shuffled[n_hold:].tolist())
    logger.info("stations: %d train pool / %d spatial holdout", len(pool), len(holdout))
    return pool, holdout


def sample_rows(data_dir, seasonal, start, end, every_nth=SAMPLE_EVERY_NTH_INTERVAL,
                station_frac=1.0, min_pct_observed=20, seed=0, stations=None):
    """Load a subsampled row set for one date range and attach seasonal stats."""
    rng = np.random.default_rng(seed)
    keep_stations = set(stations) if stations is not None else None
    if station_frac < 1.0:
        allst = np.sort(np.array(sorted(keep_stations))
                        if keep_stations is not None
                        else seasonal["station"].unique())
        keep_stations = set(rng.choice(allst, int(len(allst) * station_frac),
                                       replace=False).tolist())
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
        d["dow"] = np.int8(date.dayofweek)
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


def attach_weather(df, weather_dir, name="archived_forecast_grid"):
    """
    Nearest gridded weather cell, joined on the hour.

    Each station is assigned once to the nearest cell centre; the join is then
    (cell, hour), which keeps it a single merge rather than a spatial lookup per
    row. Missing weather stays "unknown" rather than becoming dry -- the archive
    starts in 2022 and asserting sunshine before that is how a model learns that
    2021 never rained.
    """
    base = os.path.expanduser(weather_dir)
    path = os.path.join(base, f"{name}.parquet")
    if not os.path.exists(path):
        logger.warning("no gridded weather at %s -- weather features disabled", path)
        df["wx_class"] = "unknown"
        df["precip_mm"] = np.nan
        df["wind_kmh"] = np.nan
        df["precip_3h"] = np.nan
        return df

    wx = pd.read_parquet(path)
    wx["hour"] = pd.to_datetime(wx["ts"]).dt.floor("h")
    cells = wx[["lat", "lon"]].drop_duplicates().to_numpy()

    # rain that has been falling for hours behaves differently from a first
    # burst, so accumulate per cell before the join rather than after -- after
    # the join the same cell-hour appears on hundreds of stations.
    wx = wx.sort_values(["lat", "lon", "hour"])
    wx["precip_3h"] = (wx.groupby(["lat", "lon"])["precipitation"]
                         .rolling(3, min_periods=1).sum()
                         .reset_index(level=[0, 1], drop=True))

    st = df[["station", "Latitude", "Longitude"]].drop_duplicates("station")
    d2 = ((st["Latitude"].to_numpy()[:, None] - cells[None, :, 0]) ** 2 +
          (st["Longitude"].to_numpy()[:, None] - cells[None, :, 1]) ** 2)
    nearest = cells[d2.argmin(axis=1)]
    st = st.assign(lat=nearest[:, 0], lon=nearest[:, 1])

    df = df.merge(st[["station", "lat", "lon"]], on="station", how="left")
    df["hour"] = df["ts"].dt.floor("h")
    grid = (wx.drop_duplicates(["lat", "lon", "hour"])
              .set_index(["lat", "lon", "hour"])
              [["precipitation", "wind_speed_10m", "precip_3h"]])
    joined = df.join(grid, on=["lat", "lon", "hour"])

    pr = joined["precipitation"]
    df["wx_class"] = np.where(pr.isna(), "unknown",
                     np.where(pr >= 2.5, "rain_heavy",
                     np.where(pr >= 0.2, "rain_light", "dry")))
    df["precip_mm"] = pr.astype(np.float32)
    df["wind_kmh"] = joined["wind_speed_10m"].astype(np.float32)
    df["precip_3h"] = joined["precip_3h"].astype(np.float32)
    return df.drop(columns=["lat", "lon", "hour"])


def attach_events(df, events_path, max_miles=12.0):
    """
    Nearest venue event in time, for the venue nearest each station in space.

    The corridor model hard-coded which corridors a venue could affect. That
    does not scale and it bakes in an assumption the data should decide, so here
    the model gets `event_miles` -- distance from station to venue -- and learns
    the decay itself. A station four miles from Levi's and one twenty-five miles
    away receive the same event with different distances, and the split falls
    where the data puts it.

    Beyond `max_miles` no event is attached at all: those rows are the vast
    majority, and letting them carry a far-away concert would drown the signal
    that does exist near the venue.
    """
    from forecast.corridors import VENUES

    path = os.path.expanduser(events_path)
    if not os.path.exists(path):
        logger.warning("no events at %s -- event features disabled", path)
        df["hours_since_event"] = np.nan
        df["event_capacity"] = 0.0
        df["event_miles"] = np.nan
        return df

    import json
    events = [json.loads(l) for l in open(path) if l.strip()]
    by_venue = {}
    for e in events:
        start = e["start"]
        ts = pd.Timestamp(start if "T" in start else start + "T19:00:00")
        by_venue.setdefault(e["venue"], []).append((ts, e))

    venues = {v.slug: v for v in VENUES if v.slug in by_venue}
    if not venues:
        logger.warning("no events matched known venues")
        df["hours_since_event"] = np.nan
        df["event_capacity"] = 0.0
        df["event_miles"] = np.nan
        return df

    slugs = list(venues)
    vlat = np.array([venues[s].lat for s in slugs])
    vlon = np.array([venues[s].lon for s in slugs])

    st = df[["station", "Latitude", "Longitude"]].drop_duplicates("station")
    # equirectangular is accurate to well under a percent at Bay Area latitudes
    dy = (st["Latitude"].to_numpy()[:, None] - vlat[None, :]) * 69.0
    dx = ((st["Longitude"].to_numpy()[:, None] - vlon[None, :]) * 69.0 *
          np.cos(np.radians(st["Latitude"].to_numpy()[:, None])))
    dist = np.sqrt(dx ** 2 + dy ** 2)
    j = dist.argmin(axis=1)
    st = st.assign(venue=[slugs[k] for k in j],
                   event_miles=dist[np.arange(len(st)), j])
    st.loc[st["event_miles"] > max_miles, "venue"] = None

    df = df.merge(st[["station", "venue", "event_miles"]], on="station", how="left")
    df["hours_since_event"] = np.nan
    df["event_capacity"] = 0.0

    for slug, grp in df.groupby("venue", sort=False, dropna=True):
        recs = sorted(by_venue[slug], key=lambda r: r[0])
        starts = np.array([r[0].to_datetime64() for r in recs])
        caps = np.full(len(recs), float(venues[slug].capacity))
        target = grp["ts"].to_numpy()

        prev = np.clip(np.searchsorted(starts, target) - 1, 0, len(starts) - 1)
        d_prev = (target - starts[prev]) / np.timedelta64(1, "h")
        after = (d_prev >= 0) & (d_prev <= EVENT_WINDOW_AFTER_H)

        nxt = np.clip(np.searchsorted(starts, target), 0, len(starts) - 1)
        d_next = (target - starts[nxt]) / np.timedelta64(1, "h")
        before = (d_next >= -EVENT_WINDOW_BEFORE_H) & (d_next < 0) & ~after

        use = np.where(before, nxt, prev)
        hours = np.where(before, d_next, d_prev)
        active = after | before
        if not active.any():
            continue
        idx = grp.index[active]
        df.loc[idx, "hours_since_event"] = hours[active]
        df.loc[idx, "event_capacity"] = caps[use][active]

    df["event_miles"] = df["event_miles"].astype(np.float32)
    n = int(df["hours_since_event"].notna().sum())
    logger.info("event window touches %s rows (%.3f%%)", f"{n:,}", 100 * n / max(len(df), 1))
    return df.drop(columns=["venue"])


EVENT_WINDOW_BEFORE_H = 4.0
EVENT_WINDOW_AFTER_H = 6.0

NUMERIC = ["seasonal_speed", "seasonal_sd", "tod", "dow", "month", "is_weekend",
           "lanes", "seg_miles", "Latitude", "Longitude",
           "precip_mm", "wind_kmh", "precip_3h",
           "hours_since_event", "event_capacity", "event_miles"]
CATEGORICAL = ["freeway", "direction", "holiday_class", "wx_class"]
TARGET = "speed"
