# forecast/features.py
"""
Build the modelling feature matrix.

Every feature here is causal: computable from information available strictly
before the day being predicted. That constraint is the whole point of the
module — it is easy to write a feature that quietly peeks at the target and
produces beautiful offline numbers that evaporate in production.

The seasonal median is included as a *feature* rather than treated as a rival.
It already achieves a median error of 18 seconds, and a model asked to predict
travel time from scratch would spend nearly all its capacity rediscovering
corridor x weekday x time-of-day structure it could simply be handed. Given the
prior, the model can spend its capacity on the ~20% of error that sits in
identifiable contexts instead.
"""
import json
import logging
import os

import numpy as np
import pandas as pd

from forecast.corridors import VENUES_BY_SLUG

logger = logging.getLogger("features")

SEASONAL_WEEKS = 8
MIN_SEASONAL_OBS = 3
EVENT_WINDOW_BEFORE_H = 4.0
EVENT_WINDOW_AFTER_H = 6.0

CATEGORICAL = ["corridor", "holiday_class", "wx_class", "event_type"]


def add_seasonal(df):
    """
    Seasonal median and its dispersion, from the last N same-slot observations.

    shift(1) before the rolling window is what keeps this causal: every row's
    statistics are computed from strictly earlier occurrences of the same
    (corridor, weekday, time-of-day), never including itself.
    """
    key = ["corridor", "dow", "tod"]
    grp = df.groupby(key, sort=False)["minutes"]
    shifted = grp.shift(1)
    roll = (shifted.groupby([df[k] for k in key], sort=False)
                   .rolling(SEASONAL_WEEKS, min_periods=MIN_SEASONAL_OBS))
    lvl = list(range(len(key)))
    df["seasonal_median"] = roll.median().reset_index(level=lvl, drop=True)
    # How volatile this slot usually is. Directly useful for the tail, and the
    # basis for prediction intervals later.
    df["seasonal_sd"] = roll.std().reset_index(level=lvl, drop=True)
    return df


def add_calendar(df):
    import holidays
    years = sorted(df["date"].dt.year.unique())
    us = holidays.US(years=list(years), state="CA")
    hset = set(us)
    adjacent = {h + pd.Timedelta(days=k) for h in hset for k in (-1, 1)}
    adjacent = {a.date() if hasattr(a, "date") else a for a in adjacent} - hset

    d = df["date"].dt.date
    df["holiday_class"] = np.where(
        d.isin(hset), d.map(lambda x: us.get(x, "holiday")),
        np.where(d.isin(adjacent), "adjacent", "none"))
    df["month"] = df["ts"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_yesterday_residual(df):
    """
    How far yesterday ran from its own baseline, at the corridor-day level.

    Catches ongoing disruption that nothing else in the feature set knows about:
    a lane closure that started yesterday is still there today. Shifted by one
    day so today never sees itself.
    """
    daily = (df.assign(resid=df["minutes"] - df["seasonal_median"])
               .groupby(["corridor", "date"])["resid"].mean()
               .rename("yday_residual").reset_index())
    daily["date"] = daily["date"] + pd.Timedelta(days=1)
    return df.merge(daily, on=["corridor", "date"], how="left")


def classify_event(title):
    t = (title or "").lower()
    if "49ers" in t or "week-" in t:
        return "nfl"
    if "earthquakes" in t or " sc" in t or "fc " in t:
        return "soccer"
    if "sharks" in t or "barracuda" in t:
        return "hockey"
    if "warriors" in t or "nba" in t:
        return "basketball"
    if "giants" in t or " at san francisco giants" in t:
        return "baseball"
    if "stanford" in t:
        return "college"
    return "concert"


def add_events(df, events_path):
    """
    Attach the nearest venue event in time, plus its context.

    hours_since_event is signed and continuous rather than bucketed: ingress
    builds before the start and egress spikes hours after, and those are
    different shapes that a flag cannot express. Date-only events are assumed to
    start at 19:00 (the modal evening slot) and flagged so their contribution can
    be audited separately.
    """
    with open(os.path.expanduser(events_path)) as f:
        events = [json.loads(line) for line in f if line.strip()]

    by_corridor = {}
    for e in events:
        venue = VENUES_BY_SLUG.get(e["venue"])
        if not venue:
            continue
        start = e["start"]
        ts = pd.Timestamp(start if "T" in start else start + "T19:00:00")
        rec = (ts, venue.capacity, classify_event(e.get("title")),
               int(bool(e.get("has_time"))))
        for slug in venue.corridors:
            by_corridor.setdefault(slug, []).append(rec)

    df["hours_since_event"] = np.nan
    df["event_capacity"] = 0.0
    df["event_type"] = "none"
    df["event_time_known"] = 0
    df["events_today"] = 0

    for slug, recs in by_corridor.items():
        mask = (df["corridor"] == slug).to_numpy()
        if not mask.any():
            continue
        recs.sort(key=lambda r: r[0])
        starts = np.array([r[0].to_datetime64() for r in recs])
        target = df.loc[mask, "ts"].to_numpy()

        idx = np.clip(np.searchsorted(starts, target) - 1, 0, len(starts) - 1)
        delta = (target - starts[idx]) / np.timedelta64(1, "h")
        inwin = (delta >= -EVENT_WINDOW_BEFORE_H) & (delta <= EVENT_WINDOW_AFTER_H)

        # also look forward, so pre-event build-up attaches to the coming event
        nxt = np.clip(np.searchsorted(starts, target), 0, len(starts) - 1)
        dnext = (target - starts[nxt]) / np.timedelta64(1, "h")
        pre = (dnext >= -EVENT_WINDOW_BEFORE_H) & (dnext < 0) & ~inwin
        use = np.where(pre, nxt, idx)
        hours = np.where(pre, dnext, delta)
        active = inwin | pre

        caps = np.array([r[1] for r in recs], dtype=float)
        types = np.array([r[2] for r in recs], dtype=object)
        known = np.array([r[3] for r in recs])

        sub = df.index[mask]
        df.loc[sub[active], "hours_since_event"] = hours[active]
        df.loc[sub[active], "event_capacity"] = caps[use][active]
        df.loc[sub[active], "event_type"] = types[use][active]
        df.loc[sub[active], "event_time_known"] = known[use][active]

        # day-level count: several venues firing at once is a different regime
        day_counts = pd.Series([r[0].date() for r in recs]).value_counts()
        df.loc[mask, "events_today"] = (df.loc[mask, "date"].dt.date
                                        .map(day_counts).fillna(0).to_numpy())
    return df


def add_weather(df, weather_dir):
    path = os.path.join(os.path.expanduser(weather_dir), "archived_forecast.parquet")
    if not os.path.exists(path):
        logger.warning("no weather at %s", path)
        df["wx_class"], df["precip_mm"], df["wind_kmh"] = "unknown", np.nan, np.nan
        return df

    wx = pd.read_parquet(path)
    points = json.load(open(os.path.join(os.path.expanduser(weather_dir), "points.json")))
    slug_to_point = {slug: key for key, slugs in points.items() for slug in slugs}

    wx["hour"] = pd.to_datetime(wx["ts"]).dt.floor("h")
    wx["key"] = wx["lat"].round(2).astype(str) + "," + wx["lon"].round(2).astype(str)
    grid = (wx.drop_duplicates(["key", "hour"])
              .set_index(["key", "hour"])[["precipitation", "wind_speed_10m"]])

    df["hour"] = df["ts"].dt.floor("h")
    df["key"] = df["corridor"].map(slug_to_point)
    joined = df.join(grid, on=["key", "hour"])

    p = joined["precipitation"]
    # missing weather is NOT dry weather: the archive starts in 2022
    df["wx_class"] = np.where(p.isna(), "unknown",
                     np.where(p >= 2.5, "rain_heavy",
                     np.where(p >= 0.2, "rain_light", "dry")))
    df["precip_mm"] = p
    df["wind_kmh"] = joined["wind_speed_10m"]

    # rain that has been falling a while behaves differently from a first burst
    df = df.sort_values(["corridor", "ts"])
    df["precip_3h"] = (df.groupby("corridor")["precip_mm"]
                         .rolling(36, min_periods=1).sum()
                         .reset_index(level=0, drop=True))
    return df.drop(columns=["hour", "key"])


NUMERIC = ["seasonal_median", "seasonal_sd", "tod", "dow", "month", "is_weekend",
           "yday_residual", "hours_since_event", "event_capacity",
           "event_time_known", "events_today", "precip_mm", "wind_kmh", "precip_3h"]


def build(traffic_dir, events_path, weather_dir):
    from forecast.baseline import load_corridors
    df = load_corridors(traffic_dir)
    df = add_seasonal(df)
    df = df[df["seasonal_median"].notna()].copy()
    df = add_calendar(df)
    df = add_yesterday_residual(df)
    df = add_events(df, events_path)
    df = add_weather(df, weather_dir)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    logger.info("features built: %d rows x %d features",
                len(df), len(NUMERIC) + len(CATEGORICAL))
    return df
