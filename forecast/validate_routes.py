# forecast/validate_routes.py
"""
Score the network model in minutes, on real corridors, over a held-out period.

Detector MAE in mph is what the model optimises, but nobody experiences mph. A
corridor's travel time is a sum of length/speed terms, so errors partly cancel
and partly compound; composing and measuring is the only way to know which.

Three series per corridor and interval, all through the same composition so the
comparison isolates the model rather than the method:

  actual     composed from observed detector speeds
  baseline   composed from the seasonal profile, fitted on training days only
  model      composed from the model's predictions

Spans use detector spacing, not each detector's `Length`. Length averages
0.34 mi against 0.58 mi spacing, so summing it understates a corridor by roughly
40%.

    python -m forecast.validate_routes --model models/network/model.txt
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.corridors import CORRIDORS                     # noqa: E402
from forecast.features_stations import (                     # noqa: E402
    CATEGORICAL, NUMERIC, attach_calendar, attach_events, attach_station_attrs,
    attach_weather, _date_of, _day_paths)

logger = logging.getLogger("validate_routes")


def spacing_miles(stations):
    """Miles each detector governs: distance to the next, last mirrors the previous."""
    pm = stations["Abs_PM"].to_numpy(dtype=float)
    if len(pm) < 2:
        return np.array([float(stations["Length"].fillna(0.5).iloc[0])])
    gaps = np.abs(np.diff(pm))
    return np.concatenate([gaps, [gaps[-1]]])


def compose(pivot, weights, total_mi):
    """
    Sum length/speed over whatever is reporting, then scale to full length.

    A raw sum over present detectors looks like a faster trip rather than like
    missing data, so scaling matters.
    """
    mph = pivot.to_numpy(dtype=float)
    ok = np.isfinite(mph) & (mph > 0)
    covered = np.where(ok, weights, 0.0).sum(axis=1)
    minutes = np.where(ok, weights / np.where(ok, mph, 1.0) * 60.0, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(covered > 0, minutes * total_mi / covered, np.nan)
    return scaled, covered / total_mi


def load_window(data_dir, seasonal, meta, weather_dir, events_path,
                start, end, stations, every_nth, min_coverage):
    """Observed station rows for the test window, with model features attached."""
    paths = [p for p in _day_paths(data_dir)
             if pd.Timestamp(start) <= _date_of(p) < pd.Timestamp(end)]
    keep = set(int(s) for s in stations)
    frames = []
    for p in paths:
        d = pd.read_parquet(p, columns=["station", "tod", "speed", "pct_observed"])
        d = d[d["station"].isin(keep) & (d["tod"] % (5 * every_nth) == 0)
              & (d["pct_observed"] >= min_coverage)]
        if d.empty:
            continue
        date = _date_of(p)
        d["date"] = date
        d["dow"] = np.int8(date.dayofweek)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.merge(seasonal, on=["station", "dow", "tod"], how="inner")
    df = attach_station_attrs(df, meta)
    df = df[df["freeway"].notna()].copy()
    df["ts"] = df["date"] + pd.to_timedelta(df["tod"], unit="m")
    df = attach_calendar(df)
    df = attach_weather(df, weather_dir)
    df = attach_events(df, events_path)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--seasonal", default="_seasonal_trainonly.parquet",
                   help="the train-only profile; never the serving one, which "
                        "is refitted nightly through today and would leak")
    p.add_argument("--model", default="models/network/model.txt")
    p.add_argument("--weather", default="~/traffic-data/weather")
    p.add_argument("--events", default="~/traffic-data/events/events_merged.jsonl")
    p.add_argument("--start", default="2025-07-01")
    p.add_argument("--end", default="2026-08-01")
    p.add_argument("--every-nth", type=int, default=3, help="3 -> 15-minute intervals")
    p.add_argument("--min-coverage", type=float, default=0.80,
                   help="RELATIVE to the corridor's own median coverage, not "
                        "absolute; see the note at the filter")
    p.add_argument("--out", default="models/network/route_metrics.json")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    import lightgbm as lgb
    data = os.path.expanduser(a.data)
    seasonal = pd.read_parquet(os.path.join(data, a.seasonal))
    meta = pd.read_csv(os.path.join(data, "_meta", "d04_meta.txt"), sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    model = lgb.Booster(model_file=os.path.expanduser(a.model))
    sidecar = json.load(open(os.path.join(os.path.dirname(os.path.expanduser(a.model)),
                                          "metrics.json")))
    cats = sidecar["categories"]
    # Models fitted on pace answer in min/mile; everything below speaks mph.
    target = sidecar.get("target", "speed")

    corridor_stations, weights, totals = {}, {}, {}
    wanted = set()
    for c in CORRIDORS:
        st = c.stations(meta)
        ids = st["sensor_id"].astype(int).tolist()
        w = spacing_miles(st)
        corridor_stations[c.slug] = ids
        weights[c.slug] = pd.Series(w, index=ids)
        totals[c.slug] = float(w.sum())
        wanted.update(ids)
    logger.info("%d corridors, %d distinct detectors", len(CORRIDORS), len(wanted))

    started = time.time()
    df = load_window(data, seasonal, meta, a.weather, a.events,
                     a.start, a.end, wanted, a.every_nth, min_coverage=20)
    logger.info("test window %s..%s -> %s station-intervals (%.0fs)",
                a.start, a.end, f"{len(df):,}", time.time() - started)

    for c in CATEGORICAL:
        vals = df[c].astype(str)
        df[c] = pd.Categorical(vals.where(vals.isin(cats[c])), categories=cats[c])
    raw = model.predict(df[NUMERIC + CATEGORICAL])
    if target == "pace":
        raw = 60.0 / np.clip(raw, 60.0 / 80.0, 60.0 / 5.0)
    df["pred"] = raw.clip(5.0, 80.0)

    rows = []
    for c in CORRIDORS:
        ids = corridor_stations[c.slug]
        sub = df[df["station"].isin(ids)]
        if sub.empty:
            continue
        piv = {k: sub.pivot_table(index="ts", columns="station", values=k)
               for k in ("speed", "seasonal_speed", "pred")}
        order = [i for i in ids if i in piv["speed"].columns]
        w = weights[c.slug].reindex(order).to_numpy()
        tot = totals[c.slug]

        actual, cov = compose(piv["speed"][order], w, tot)
        base, _ = compose(piv["seasonal_speed"][order], w, tot)
        pred, _ = compose(piv["pred"][order], w, tot)

        # Coverage is near-constant within a corridor: the same detectors
        # report all year and the rest never do. An absolute threshold therefore
        # deletes whole corridors rather than selecting good intervals. The Bay
        # Bridge runs at 0.50 coverage every hour of every day and disappears
        # entirely at a 0.80 cut.
        #
        # The threshold is relative to the corridor's own median instead, which
        # catches real outages and keeps the structural gaps. Actual, baseline
        # and prediction share the same detectors and the same scaling, so only
        # the absolute minutes shift.
        med_cov = float(np.nanmedian(cov))
        ok = (np.isfinite(actual) & np.isfinite(base) & np.isfinite(pred)
              & (cov >= a.min_coverage * med_cov))
        if ok.sum() == 0:
            continue
        idx = piv["speed"].index[ok]
        hour = idx.hour
        peak = ((hour >= 7) & (hour < 10)) | ((hour >= 15) & (hour < 19))

        def mae(x, y, m=None, _ok=ok):
            # `m` indexes the already-filtered rows, so its default has to be
            # the filtered length, not the raw one.
            xs, ys = x[_ok], y[_ok]
            m = np.ones(len(xs), bool) if m is None else m
            return float(np.abs(xs[m] - ys[m]).mean())

        rows.append({
            "corridor": c.slug, "name": c.name, "miles": round(tot, 1),
            "n": int(ok.sum()), "coverage": round(med_cov, 2),
            "mean_minutes": round(float(actual[ok].mean()), 1),
            "baseline_mae": round(mae(base, actual), 3),
            "model_mae": round(mae(pred, actual), 3),
            "baseline_peak": round(mae(base, actual, peak), 3),
            "model_peak": round(mae(pred, actual, peak), 3),
        })

    rep = pd.DataFrame(rows)
    rep["gain_pct"] = (rep["baseline_mae"] - rep["model_mae"]) / rep["baseline_mae"] * 100
    rep["gain_peak"] = (rep["baseline_peak"] - rep["model_peak"]) / rep["baseline_peak"] * 100

    logger.info("\n%-20s%7s%6s%8s%9s%8s%8s%9s%9s%8s", "corridor", "mi", "cov",
                "n", "mean", "base", "model", "gain", "pk base", "pk gain")
    for _, r in rep.iterrows():
        logger.info("%-20s%7.1f%5.0f%%%8s%9.1f%8.3f%8.3f%8.1f%%%9.3f%7.1f%%",
                    r["corridor"], r["miles"], r["coverage"] * 100,
                    f"{int(r['n']):,}", r["mean_minutes"], r["baseline_mae"],
                    r["model_mae"], r["gain_pct"], r["baseline_peak"], r["gain_peak"])

    wsum = rep["n"].to_numpy()
    overall = {
        "weighted_baseline_mae": float(np.average(rep["baseline_mae"], weights=wsum)),
        "weighted_model_mae": float(np.average(rep["model_mae"], weights=wsum)),
        "weighted_baseline_peak": float(np.average(rep["baseline_peak"], weights=wsum)),
        "weighted_model_peak": float(np.average(rep["model_peak"], weights=wsum)),
        "intervals": int(rep["n"].sum()),
        "window": [a.start, a.end],
    }
    overall["gain_pct"] = ((overall["weighted_baseline_mae"] - overall["weighted_model_mae"])
                           / overall["weighted_baseline_mae"] * 100)
    overall["gain_peak_pct"] = ((overall["weighted_baseline_peak"] - overall["weighted_model_peak"])
                                / overall["weighted_baseline_peak"] * 100)
    logger.info("\nweighted   baseline %.3f min   model %.3f min   gain %.1f%%",
                overall["weighted_baseline_mae"], overall["weighted_model_mae"],
                overall["gain_pct"])
    logger.info("peak only  baseline %.3f min   model %.3f min   gain %.1f%%",
                overall["weighted_baseline_peak"], overall["weighted_model_peak"],
                overall["gain_peak_pct"])

    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"overall": overall, "corridors": rep.to_dict("records")}, f, indent=1)
    logger.info("\nsaved -> %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
