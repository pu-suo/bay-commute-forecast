# forecast/predict.py
"""
Generate day-ahead forecasts and write the payload the site renders.

This is the serving path, and the constraint that shapes it is that every
feature must be computable for a date that has not happened yet:

  seasonal_median / sd   from history, per (corridor, weekday, time-of-day)
  calendar               deterministic
  yday_residual          from the most recent completed day. PeMS runs a day
                         behind, so "yesterday" is genuinely the freshest actual
                         available — which is fine, because this predicts
                         tomorrow, not now.
  weather                Open-Meteo *Forecast* API, not the archive
  events                 the live venue crawl

Nothing here may read the target. If a feature cannot be produced for a future
date it does not belong in the model, which is why the training feature set was
constrained to exactly these.

    python -m forecast.predict --days 7 --out site/data/forecast.json
"""
import argparse
import json
import logging
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.baseline import load_corridors  # noqa: E402
from forecast.corridors import CORRIDORS, VENUES_BY_SLUG  # noqa: E402
from forecast.features import (NUMERIC, CATEGORICAL, SEASONAL_WEEKS,  # noqa: E402
                               classify_event)

logger = logging.getLogger("predict")

LIVE_FORECAST = "https://api.open-meteo.com/v1/forecast"
INTERVAL_MINUTES = 5
AM_PEAK = (7 * 60, 9 * 60 + 55)
PM_PEAK = (16 * 60, 18 * 60 + 55)


def seasonal_table(hist, weeks=SEASONAL_WEEKS):
    """Median and sd of the most recent N same-slot observations, per corridor."""
    recent = hist.sort_values("ts").groupby(["corridor", "dow", "tod"]).tail(weeks)
    return (recent.groupby(["corridor", "dow", "tod"])["minutes"]
                  .agg(seasonal_median="median", seasonal_sd="std")
                  .reset_index())


def yesterday_residuals(hist, seasonal):
    """Mean residual on the latest completed day, per corridor."""
    last_day = hist["date"].max()
    day = hist[hist["date"] == last_day].merge(seasonal, on=["corridor", "dow", "tod"], how="left")
    resid = ((day["minutes"] - day["seasonal_median"])
             .groupby(day["corridor"]).mean().rename("yday_residual"))
    logger.info("yday_residual from %s", last_day.date())
    return resid, last_day


def future_grid(days, start_date):
    rows = []
    for c in CORRIDORS:
        for d in range(days):
            date = start_date + timedelta(days=d)
            for minute in range(0, 24 * 60, INTERVAL_MINUTES):
                rows.append((c.slug, date, minute))
    df = pd.DataFrame(rows, columns=["corridor", "date", "tod"])
    df["date"] = pd.to_datetime(df["date"])
    df["ts"] = df["date"] + pd.to_timedelta(df["tod"], unit="m")
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["ts"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def add_calendar(df):
    import holidays
    years = sorted(df["date"].dt.year.unique())
    us = holidays.US(years=list(years), state="CA")
    hset = set(us)
    adjacent = {h + timedelta(days=k) for h in hset for k in (-1, 1)} - hset
    d = df["date"].dt.date
    df["holiday_class"] = np.where(
        d.isin(hset), d.map(lambda x: us.get(x, "holiday")),
        np.where(d.isin(adjacent), "adjacent", "none"))
    return df


def fetch_live_weather(points, start, end):
    """Hourly forecast per weather point, keyed the same way as the archive."""
    frames = []
    for key in points:
        lat, lon = (float(v) for v in key.split(","))
        q = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": "precipitation,wind_speed_10m",
            "timezone": "America/Los_Angeles"})
        req = urllib.request.Request(f"{LIVE_FORECAST}?{q}",
                                     headers={"User-Agent": "commute-forecast/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        h = data.get("hourly") or {}
        frames.append(pd.DataFrame({
            "key": key,
            "hour": pd.to_datetime(h.get("time", [])),
            "precip_mm": h.get("precipitation", []),
            "wind_kmh": h.get("wind_speed_10m", []),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_weather(df, weather_dir):
    points = json.load(open(os.path.join(os.path.expanduser(weather_dir), "points.json")))
    slug_to_point = {slug: key for key, slugs in points.items() for slug in slugs}
    wx = fetch_live_weather(list(points),
                            df["date"].min().strftime("%Y-%m-%d"),
                            df["date"].max().strftime("%Y-%m-%d"))
    df["hour"] = df["ts"].dt.floor("h")
    df["key"] = df["corridor"].map(slug_to_point)
    if wx.empty:
        df["precip_mm"], df["wind_kmh"], df["wx_class"] = np.nan, np.nan, "unknown"
    else:
        df = df.merge(wx, on=["key", "hour"], how="left")
        p = df["precip_mm"]
        df["wx_class"] = np.where(p.isna(), "unknown",
                         np.where(p >= 2.5, "rain_heavy",
                         np.where(p >= 0.2, "rain_light", "dry")))
    df = df.sort_values(["corridor", "ts"])
    df["_p"] = df["precip_mm"].fillna(0.0)
    df["precip_3h"] = (df.groupby("corridor")["_p"]
                         .rolling(36, min_periods=1).sum()
                         .reset_index(level=0, drop=True))
    return df.drop(columns=["hour", "key", "_p"])


def add_events(df, events_path):
    """Attach upcoming events, and keep a human-readable label for the site."""
    with open(os.path.expanduser(events_path)) as f:
        events = [json.loads(line) for line in f if line.strip()]

    by_corridor = {}
    for e in events:
        venue = VENUES_BY_SLUG.get(e["venue"])
        if not venue:
            continue
        s = e["start"]
        ts = pd.Timestamp(s if "T" in s else s + "T19:00:00")
        for slug in venue.corridors:
            by_corridor.setdefault(slug, []).append(
                (ts, venue.capacity, classify_event(e.get("title")),
                 int(bool(e.get("has_time"))), e.get("title", ""), venue.name))

    df["hours_since_event"] = np.nan
    df["event_capacity"] = 0.0
    df["event_type"] = "none"
    df["event_time_known"] = 0
    df["events_today"] = 0
    df["event_label"] = ""

    for slug, recs in by_corridor.items():
        mask = (df["corridor"] == slug).to_numpy()
        if not mask.any():
            continue
        recs.sort(key=lambda r: r[0])
        starts = np.array([r[0].to_datetime64() for r in recs])
        target = df.loc[mask, "ts"].to_numpy()
        idx = np.clip(np.searchsorted(starts, target) - 1, 0, len(starts) - 1)
        delta = (target - starts[idx]) / np.timedelta64(1, "h")
        nxt = np.clip(np.searchsorted(starts, target), 0, len(starts) - 1)
        dnext = (target - starts[nxt]) / np.timedelta64(1, "h")
        inwin = (delta >= -4) & (delta <= 6)
        pre = (dnext >= -4) & (dnext < 0) & ~inwin
        use = np.where(pre, nxt, idx)
        hours = np.where(pre, dnext, delta)
        active = inwin | pre
        sub = df.index[mask]
        df.loc[sub[active], "hours_since_event"] = hours[active]
        df.loc[sub[active], "event_capacity"] = np.array([r[1] for r in recs], float)[use][active]
        df.loc[sub[active], "event_type"] = np.array([r[2] for r in recs], object)[use][active]
        df.loc[sub[active], "event_time_known"] = np.array([r[3] for r in recs])[use][active]
        labels = np.array([f"{r[5]}: {r[4]}" for r in recs], dtype=object)
        df.loc[sub[active], "event_label"] = labels[use][active]
        # The label is for humans, so it must name events happening on *that*
        # day. The feature window runs six hours past the start, so a Saturday
        # evening event stays active into Sunday's small hours — correct for the
        # model, wrong on a page that would then claim Sunday has a game.
        start_dates = np.array([r[0].date() for r in recs], dtype=object)
        same_day = np.zeros(len(sub), dtype=bool)
        same_day[active] = (start_dates[use][active]
                            == df.loc[sub[active], "date"].dt.date.to_numpy())
        df.loc[sub[~same_day], "event_label"] = ""
        counts = pd.Series([r[0].date() for r in recs]).value_counts()
        df.loc[mask, "events_today"] = (df.loc[mask, "date"].dt.date
                                        .map(counts).fillna(0).to_numpy())
    return df


def peak_summary(day_rows, lo, hi):
    window = day_rows[(day_rows["tod"] >= lo) & (day_rows["tod"] <= hi)]
    if window.empty:
        return None
    worst = window.loc[window["pred"].idxmax()]
    return {
        "worst_time": f"{int(worst['tod']) // 60:02d}:{int(worst['tod']) % 60:02d}",
        "minutes": round(float(worst["pred"]), 1),
        "typical": round(float(worst["seasonal_median"]), 1),
        "delta": round(float(worst["pred"] - worst["seasonal_median"]), 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traffic", default="~/traffic-data/corridors")
    p.add_argument("--events", default="~/traffic-data/events/events_merged.jsonl")
    p.add_argument("--weather", default="~/traffic-data/weather")
    p.add_argument("--model", default="models/model.txt")
    p.add_argument("--metrics", default="models/metrics.json")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--out", default="site/data/forecast.json")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    import lightgbm as lgb
    model = lgb.Booster(model_file=os.path.expanduser(a.model))

    hist = load_corridors(a.traffic)
    seasonal = seasonal_table(hist)
    resid, last_day = yesterday_residuals(hist, seasonal)

    start = (last_day + pd.Timedelta(days=1)).date()
    grid = future_grid(a.days, start)
    logger.info("forecasting %s .. %s", start, start + timedelta(days=a.days - 1))

    grid = grid.merge(seasonal, on=["corridor", "dow", "tod"], how="left")
    grid["yday_residual"] = grid["corridor"].map(resid).fillna(0.0)
    grid = add_calendar(grid)
    grid = add_weather(grid, a.weather)
    grid = add_events(grid, a.events)
    grid = grid[grid["seasonal_median"].notna()].copy()

    for c in CATEGORICAL:
        grid[c] = grid[c].astype("category")
    grid["pred"] = model.predict(grid[NUMERIC + CATEGORICAL])

    metrics = {}
    if os.path.exists(os.path.expanduser(a.metrics)):
        metrics = json.load(open(os.path.expanduser(a.metrics)))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_through": str(last_day.date()),
        "model": metrics,
        "corridors": [],
    }
    names = {c.slug: c.name for c in CORRIDORS}
    for slug, csub in grid.groupby("corridor", observed=True):
        days = []
        for date, dsub in csub.groupby("date"):
            labels = [x for x in dsub["event_label"].unique() if x]
            days.append({
                "date": str(date.date()),
                "weekday": date.strftime("%a"),
                "am_peak": peak_summary(dsub, *AM_PEAK),
                "pm_peak": peak_summary(dsub, *PM_PEAK),
                "why": labels[:3],
                "series": [
                    {"t": f"{int(r.tod) // 60:02d}:{int(r.tod) % 60:02d}",
                     "m": round(float(r.pred), 1),
                     "b": round(float(r.seasonal_median), 1)}
                    for r in dsub.sort_values("tod").itertuples()
                    if int(r.tod) % 15 == 0          # 15-min resolution for the page
                ],
            })
        payload["corridors"].append({"slug": slug, "name": names.get(slug, slug), "days": days})

    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    logger.info("wrote %d corridors x %d days -> %s (%.0f KB)",
                len(payload["corridors"]), a.days, out, os.path.getsize(out) / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
