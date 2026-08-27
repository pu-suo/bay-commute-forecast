# forecast/enhanced.py
"""
Layered residual corrections on top of the T0 seasonal baseline, and an
attribution of how much error each layer actually removes.

The seasonal median already predicts ordinary traffic to within about 18
seconds, so the useful question is not "what is the MAE" but "what explains the
tail". Each layer here is a multiplicative correction learned only on the
training period and applied unchanged to the test period:

    corrected = baseline * factor(context)

where factor is the median ratio of actual to baseline observed for that
context during training. Multiplicative rather than additive because congestion
scales with trip length: two minutes lost on the Bay Bridge is a different
event from two minutes lost on 880 north.

Layers, in the order the residual analysis said they matter:

  holiday   day-of-year effects a day-of-week baseline cannot see. Holidays run
            2x the normal error and, with their adjacent travel days, carry
            about 15% of all error.
  event     venue events, bucketed by hours since start. Large where they
            apply and rare enough to be invisible in a global average. The
            attribution below reports both so neither number misleads.
  weather   rain and wind, from archived forecasts rather than observations,
            so training input matches what serving will actually have.

Every factor is estimated on data strictly before the test window. A factor
fitted on the whole span would leak the answer and inflate every number here.
"""
import argparse
import glob
import json
import logging
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.baseline import load_corridors, backtest  # noqa: E402
from forecast.corridors import VENUES_BY_SLUG  # noqa: E402

logger = logging.getLogger("enhanced")

MIN_SAMPLES = 20        # below this a learned factor is noise, so fall back to 1.0
TOD_BUCKET_MINUTES = 60


def add_holiday_features(df):
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


def add_event_features(df, events_path):
    """
    Mark each interval with the venue event it is closest to in time.

    Bucketed by hours since start because the effect is a wave, not a flag:
    ingress builds before, egress spikes after, and the two look nothing alike.
    Date-only events are treated as starting at 19:00, the modal evening slot,
    and flagged so their contribution can be checked separately.
    """
    with open(os.path.expanduser(events_path)) as f:
        events = [json.loads(line) for line in f if line.strip()]

    per_corridor = {}
    for e in events:
        venue = VENUES_BY_SLUG.get(e["venue"])
        if not venue:
            continue
        start = e["start"]
        ts = pd.Timestamp(start) if "T" in start else pd.Timestamp(start + "T19:00:00")
        for slug in venue.corridors:
            per_corridor.setdefault(slug, []).append((ts, e["venue"]))

    df["event_bucket"] = "none"
    for slug, entries in per_corridor.items():
        mask = df["corridor"] == slug
        if not mask.any():
            continue
        starts = pd.Series(sorted(t for t, _ in entries))
        target = df.loc[mask, "ts"]
        idx = np.searchsorted(starts.values, target.values) - 1
        idx = np.clip(idx, 0, len(starts) - 1)
        delta = (target.values - starts.values[idx]) / np.timedelta64(1, "h")
        # -3h before to +6h after is the window where a venue can plausibly matter
        bucket = np.where((delta >= -3) & (delta <= 6),
                          np.floor(delta).astype(int).astype(str), "none")
        df.loc[mask, "event_bucket"] = bucket
    return df


def add_weather_features(df, weather_dir):
    path = os.path.join(os.path.expanduser(weather_dir), "archived_forecast.parquet")
    if not os.path.exists(path):
        logger.warning("no weather at %s, skipping weather layer", path)
        df["wx_class"] = "unknown"
        return df, False

    wx = pd.read_parquet(path)
    points = json.load(open(os.path.join(os.path.expanduser(weather_dir), "points.json")))
    slug_to_point = {slug: key for key, slugs in points.items() for slug in slugs}

    wx["hour"] = pd.to_datetime(wx["ts"]).dt.floor("h")
    wx["key"] = wx["lat"].round(2).astype(str) + "," + wx["lon"].round(2).astype(str)
    grid = wx.set_index(["key", "hour"])[["precipitation", "wind_speed_10m"]]

    df["hour"] = df["ts"].dt.floor("h")
    df["key"] = df["corridor"].map(slug_to_point)
    joined = df.join(grid, on=["key", "hour"])

    # "no row in the weather archive" is not "it did not rain". The archive
    # starts ~2022, so filling nulls with zero asserts that 2021 was
    # permanently dry: 15% of the corpus, and a model trained on it would learn
    # exactly that.
    precip = joined["precipitation"]
    df["wx_class"] = np.where(precip.isna(), "unknown",
                     np.where(precip >= 2.5, "rain_heavy",
                     np.where(precip >= 0.2, "rain_light", "dry")))
    df["precip_mm"] = precip                      # continuous, for real models
    df["wind_kmh"] = joined["wind_speed_10m"]
    return df, True


def learn_factor(train, keys):
    """Median actual/baseline ratio per context, from training data only."""
    t = train[train["pred_seasonal"] > 0].copy()
    t["ratio"] = t["minutes"] / t["pred_seasonal"]
    g = t.groupby(keys)["ratio"].agg(["median", "size"])
    g = g[g["size"] >= MIN_SAMPLES]["median"]
    return g


def apply_factor(df, factor, keys, col):
    idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
    df[col] = pd.Series(idx.map(factor), index=df.index).astype(float).fillna(1.0)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/traffic-data/corridors")
    p.add_argument("--events", default="~/traffic-data/events/events_merged.jsonl")
    p.add_argument("--weather", default="~/traffic-data/weather")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-08-01")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    df = load_corridors(a.data)
    _, scored = backtest(df, a.test_start, a.test_end)

    # rebuild predictions over the whole span so factors can be fit on train
    full = df.copy()
    key = ["corridor", "dow", "tod"]
    g = full.groupby(key, sort=False)["minutes"]
    full["pred_seasonal"] = (g.shift(1)
                             .groupby([full[k] for k in key], sort=False)
                             .rolling(8, min_periods=3).median()
                             .reset_index(level=list(range(len(key))), drop=True))
    full = full[full["pred_seasonal"].notna()]

    full["tod_bucket"] = (full["tod"] // TOD_BUCKET_MINUTES).astype(int)
    full = add_holiday_features(full)
    full = add_event_features(full, a.events)
    full, have_wx = add_weather_features(full, a.weather)

    train = full[full["date"] < pd.Timestamp(a.test_start)]
    test = full[(full["date"] >= pd.Timestamp(a.test_start))
                & (full["date"] < pd.Timestamp(a.test_end))].copy()
    logger.info("train rows %d | test rows %d", len(train), len(test))

    # Each layer's correction applies ONLY where its context is active. Letting
    # the "none" context carry a learned factor turns every layer into a global
    # recalibration, which both inflates its apparent credit and lets later
    # layers undo earlier ones.
    layers = [("holiday", ["corridor", "holiday_class", "tod_bucket"], "holiday_class"),
              ("event", ["corridor", "event_bucket", "tod_bucket"], "event_bucket")]
    if have_wx:
        layers.append(("weather", ["corridor", "wx_class", "tod_bucket"], "wx_class"))
    INACTIVE = {"none", "dry", "unknown"}

    base_pred = test["pred_seasonal"].copy()
    base_mae = (base_pred - test["minutes"]).abs().mean()
    total_err = (base_pred - test["minutes"]).abs().sum()

    logger.info("\n%-10s%9s%9s%10s%11s%11s%9s", "layer", "rows", "share",
                "MAE@rows", "corrected", "gain@rows", "global")
    logger.info("%-10s%9s%9s%10.3f%11s%11s%9s", "baseline", f"{len(test):,}", "100%",
                base_mae, "-", "-", "-")

    combined = base_pred.copy()
    for name, keys, ctx in layers:
        active_train = train[~train[ctx].isin(INACTIVE)]
        factor = learn_factor(active_train, keys)
        test = apply_factor(test, factor, keys, f"f_{name}")
        active = ~test[ctx].isin(INACTIVE)
        test.loc[~active, f"f_{name}"] = 1.0

        adjusted = base_pred * test[f"f_{name}"]
        before = (base_pred[active] - test.loc[active, "minutes"]).abs()
        after = (adjusted[active] - test.loc[active, "minutes"]).abs()
        global_mae = (adjusted - test["minutes"]).abs().mean()

        logger.info("%-10s%9s%8.1f%%%10.3f%11.3f%10.1f%%%8.2f%%",
                    name, f"{int(active.sum()):,}", active.mean() * 100,
                    before.mean(), after.mean(),
                    (1 - after.mean() / before.mean()) * 100,
                    (base_mae - global_mae) / base_mae * 100)
        combined = combined * test[f"f_{name}"]

    final = (combined - test["minutes"]).abs().mean()
    logger.info("\nall layers combined: %.4f -> %.4f  (%.2f%% global improvement)",
                base_mae, final, (base_mae - final) / base_mae * 100)
    logger.info("share of total baseline error sitting on rows each layer touches:")
    for name, _, ctx in layers:
        act = ~test[ctx].isin(INACTIVE)
        share = (base_pred[act] - test.loc[act, "minutes"]).abs().sum() / total_err * 100
        logger.info("  %-10s %5.1f%%", name, share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
