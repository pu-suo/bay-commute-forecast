# forecast/train_stations.py
"""
Train the network-scale speed model.

The corridor model answers "how long is this drive". This one answers "how fast
will this detector read", which composes into any route. Three consequences:

Detector attributes, not identity. 2,291 detectors cannot each get an embedding
from a sampled training set, and a category the model never saw is useless at
serve time.

The holdout is spatial as well as temporal. A temporal split cannot detect
detector memorisation, because every detector sits on both sides of it. 30% are
withheld entirely and scored separately.

The seasonal baseline is fitted on training days only and applied forward, not
recomputed over the test window.

    python -m forecast.train_stations --out models/network
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
from forecast.features_stations import (           # noqa: E402
    CATEGORICAL, HOLIDAY_CLASSES, NUMERIC, TARGET, WX_CLASSES, attach_calendar,
    attach_events, attach_station_attrs, attach_weather, sample_rows,
    split_stations)

logger = logging.getLogger("train_stations")

PARAMS = {
    "objective": "l1",          # MAE: a few badly-wrong sensors should not
    "learning_rate": 0.06,      # drag the fit the way squared error lets them
    "num_leaves": 255,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "max_bin": 127,             # 10M rows: smaller bins, much less memory
    "verbosity": -1,
    "num_threads": 0,
}
NUM_ROUNDS = 1200
EARLY_STOPPING = 60


def load(data_dir, seasonal, meta, weather_dir, events_path,
         start, end, stations, station_frac, every_nth, seed, label):
    started = time.time()
    df = sample_rows(data_dir, seasonal, start, end, every_nth=every_nth,
                     station_frac=station_frac, seed=seed, stations=stations)
    if df.empty:
        return df
    df = attach_station_attrs(df, meta)
    df = df[df["freeway"].notna()].copy()
    df["ts"] = df["date"] + pd.to_timedelta(df["tod"], unit="m")
    df = attach_calendar(df)
    df = attach_weather(df, weather_dir)
    df = attach_events(df, events_path)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    logger.info("%-6s %s rows x %d features  (%.0fs)",
                label, f"{len(df):,}", len(NUMERIC) + len(CATEGORICAL),
                time.time() - started)
    return df


FIXED_VOCAB = {"wx_class": WX_CLASSES, "holiday_class": HOLIDAY_CLASSES}


def align_categories(frames):
    """
    Categories must match across train/valid/test or LightGBM sees different codes.

    Classes with a fixed vocabulary are unioned in even when no frame contains
    them, so the model always has a slot for a value that serving can produce.
    """
    for c in CATEGORICAL:
        cats = set().union(*[set(f[c].cat.categories) for f in frames if len(f)])
        cats = sorted(cats | set(FIXED_VOCAB.get(c, ())))
        for f in frames:
            if len(f):
                f[c] = f[c].cat.set_categories(cats)


def evaluate(test, pred, name):
    """MAE in mph, overall and on the slices a commute forecast is judged on."""
    base = test["seasonal_speed"].to_numpy()
    actual = test[TARGET].to_numpy()
    peak = (test["tod"].between(7 * 60, 9 * 60 + 55) |
            test["tod"].between(15 * 60, 18 * 60 + 55)).to_numpy()
    congested = (base < 45)
    slices = [
        ("all", np.ones(len(test), bool)),
        ("peak", peak),
        ("congested", congested),
        ("peak+congested", peak & congested),
        ("holiday", (test["holiday_class"].astype(str) != "none").to_numpy()),
        ("rain", test["wx_class"].astype(str).isin(["rain_light", "rain_heavy"]).to_numpy()),
        ("near event", test["hours_since_event"].notna().to_numpy()),
        ("worst 5%", np.abs(base - actual) > np.quantile(np.abs(base - actual), 0.95)),
    ]
    rows = []
    for label, m in slices:
        if m.sum() == 0:
            continue
        b = float(np.abs(base[m] - actual[m]).mean())
        p = float(np.abs(pred[m] - actual[m]).mean())
        rows.append({"stations": name, "slice": label, "n": int(m.sum()),
                     "baseline_mph": b, "model_mph": p,
                     "gain_pct": (b - p) / b * 100 if b else 0.0})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--seasonal", default="_seasonal_trainonly.parquet",
                   help="the train-only profile; never the serving one, which "
                        "is refitted nightly through today and would leak")
    p.add_argument("--weather", default="~/traffic-data/weather")
    p.add_argument("--events", default="~/traffic-data/events/events_merged.jsonl")
    p.add_argument("--train-end", default="2025-01-01")
    p.add_argument("--valid-end", default="2025-07-01")
    p.add_argument("--test-end", default="2026-08-01")
    p.add_argument("--train-start", default="2022-01-01",
                   help="weather archive starts 2022; earlier days would train "
                        "the model on a year it is told had no weather")
    p.add_argument("--train-station-frac", type=float, default=0.20)
    p.add_argument("--test-station-frac", type=float, default=0.12)
    p.add_argument("--every-nth", type=int, default=6, help="6 -> 30-minute rows")
    p.add_argument("--out", default="models/network")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    import lightgbm as lgb
    data = os.path.expanduser(a.data)
    seasonal = pd.read_parquet(os.path.join(data, a.seasonal))
    meta = pd.read_csv(os.path.join(data, "_meta", "d04_meta.txt"), sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    logger.info("seasonal %s cells | %d mainline stations in meta",
                f"{len(seasonal):,}", len(meta))

    pool, holdout = split_stations(seasonal)

    common = dict(data_dir=data, seasonal=seasonal, meta=meta,
                  weather_dir=a.weather, events_path=a.events, every_nth=a.every_nth)
    train = load(**common, start=a.train_start, end=a.train_end, stations=pool,
                 station_frac=a.train_station_frac, seed=1, label="train")
    valid = load(**common, start=a.train_end, end=a.valid_end, stations=pool,
                 station_frac=a.test_station_frac, seed=1, label="valid")
    seen = load(**common, start=a.valid_end, end=a.test_end, stations=pool,
                station_frac=a.test_station_frac, seed=1, label="seen")
    unseen = load(**common, start=a.valid_end, end=a.test_end, stations=holdout,
                  station_frac=a.test_station_frac, seed=1, label="unseen")
    align_categories([train, valid, seen, unseen])

    feats = NUMERIC + CATEGORICAL
    dtrain = lgb.Dataset(train[feats], label=train[TARGET],
                         categorical_feature=CATEGORICAL, free_raw_data=True)
    dvalid = lgb.Dataset(valid[feats], label=valid[TARGET],
                         categorical_feature=CATEGORICAL, reference=dtrain)
    started = time.time()
    model = lgb.train(PARAMS, dtrain, num_boost_round=NUM_ROUNDS, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                                 lgb.log_evaluation(100)])
    logger.info("trained %d rounds in %.0fs", model.best_iteration, time.time() - started)

    rows = []
    for name, frame in (("seen", seen), ("unseen", unseen)):
        pred = model.predict(frame[feats], num_iteration=model.best_iteration)
        rows += evaluate(frame, pred, name)
    report = pd.DataFrame(rows)

    logger.info("\n%-9s%-16s%12s%11s%10s%8s", "stations", "slice", "n",
                "baseline", "model", "gain")
    for _, r in report.iterrows():
        logger.info("%-9s%-16s%12s%11.3f%10.3f%7.1f%%", r["stations"], r["slice"],
                    f"{int(r['n']):,}", r["baseline_mph"], r["model_mph"], r["gain_pct"])

    logger.info("\ntop features:")
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    total = imp.sum()
    for k, v in imp.head(14).items():
        logger.info("  %-20s%12.0f  %5.1f%%", k, v, 100 * v / total)

    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    model.save_model(os.path.join(out, "model.txt"))
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"rounds": model.best_iteration,
                   "train_rows": len(train), "features": feats,
                   "categories": {c: list(map(str, train[c].cat.categories))
                                  for c in CATEGORICAL},
                   "slices": report.to_dict("records")}, f, indent=1)
    logger.info("\nsaved -> %s/model.txt", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
