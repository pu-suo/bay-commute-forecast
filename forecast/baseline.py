# forecast/baseline.py
"""
T0 seasonal baseline forecaster, and an honest rolling-origin backtest.

The baseline predicts corridor travel time from the median of the last K
occurrences of the same (corridor, weekday, time-of-day). Traffic is highly
regular, so this is a strong forecaster and the bar any model has to clear.
publishing a fancy model that loses to it would be worse than shipping nothing.

The backtest is rolling-origin: predicting day D uses only observations
strictly before D. That is the whole ballgame for a forecasting product. Fit a
median over the full history and evaluate on days inside it and the numbers
look wonderful and mean nothing, because at serve time the future isn't there.

Imputed days are excluded from both training and scoring. PeMS fills whole days
with modelled values that look completely normal; scoring against them measures
agreement with PeMS's imputation model, not with reality.

    python -m forecast.baseline --data ~/traffic-data/corridors \
        --test-start 2025-01-01 --test-end 2026-08-01
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger("baseline")

LOOKBACK_WEEKS = 8          # how many same-weekday observations feed the median
MIN_COVERAGE = 0.90         # corridor intervals below this are partial, not slow
MIN_OBSERVATIONS = 3        # fewer than this and the median is noise


def load_corridors(data_dir, drop_imputed=True, min_coverage=MIN_COVERAGE):
    """Load every per-day Parquet into one frame, filtered for usable rows."""
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir),
                                          "year=*", "*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no parquet under {data_dir}")
    df = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    before = len(df)

    if drop_imputed:
        df = df[~df["imputed_day"]]
    df = df[df["coverage"] >= min_coverage]
    df = df[df["minutes"].notna() & (df["minutes"] > 0)]

    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.normalize()
    df["dow"] = df["ts"].dt.dayofweek
    df["tod"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    logger.info("loaded %d files, %d rows -> %d usable (%.1f%% dropped)",
                len(paths), before, len(df), (1 - len(df) / before) * 100)
    return df.sort_values("ts")


def backtest(df, test_start, test_end, lookback_weeks=LOOKBACK_WEEKS):
    """
    Rolling-origin evaluation of the seasonal median, against two references.

    For each target row the prediction is the median of the same
    (corridor, weekday, time-of-day) over the preceding `lookback_weeks`
    occurrences, all strictly earlier than the target date.

    Reference models, both also causal:
      last_week   the same slot exactly 7 days earlier
      overall     the corridor's median travel time, ignoring time entirely
    """
    df = df.copy()
    key = ["corridor", "dow", "tod"]

    # Same slot, previous weeks. shift(1) inside the group guarantees the window
    # ends before the target row, so no observation ever sees itself.
    g = df.groupby(key, sort=False)["minutes"]
    df["pred_seasonal"] = (g.shift(1)
                            .groupby([df[k] for k in key], sort=False)
                            .rolling(lookback_weeks, min_periods=MIN_OBSERVATIONS)
                            .median()
                            .reset_index(level=list(range(len(key))), drop=True))
    df["pred_last_week"] = g.shift(1)

    overall = (df[df["date"] < pd.Timestamp(test_start)]
               .groupby("corridor")["minutes"].median())
    df["pred_overall"] = df["corridor"].map(overall)

    mask = ((df["date"] >= pd.Timestamp(test_start))
            & (df["date"] < pd.Timestamp(test_end))
            & df["pred_seasonal"].notna())
    test = df[mask]
    if test.empty:
        raise ValueError("no test rows; check the date range")

    rows = []
    for corridor, sub in test.groupby("corridor"):
        row = {"corridor": corridor, "n": len(sub),
               "mean_minutes": sub["minutes"].mean()}
        for name in ("seasonal", "last_week", "overall"):
            err = (sub[f"pred_{name}"] - sub["minutes"]).abs()
            row[f"mae_{name}"] = err.mean()
        # peak hours are what a commuter actually cares about
        peak = sub[sub["tod"].between(7 * 60, 9 * 60 + 55)
                   | sub["tod"].between(15 * 60, 18 * 60 + 55)]
        row["mae_seasonal_peak"] = ((peak["pred_seasonal"] - peak["minutes"])
                                    .abs().mean() if len(peak) else np.nan)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("corridor")
    result["skill_vs_lastweek"] = (1 - result["mae_seasonal"] / result["mae_last_week"])
    result["mae_pct_of_mean"] = result["mae_seasonal"] / result["mean_minutes"] * 100
    return result, test


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/traffic-data/corridors")
    p.add_argument("--test-start", default="2025-01-01")
    p.add_argument("--test-end", default="2026-08-01")
    p.add_argument("--lookback-weeks", type=int, default=LOOKBACK_WEEKS)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    df = load_corridors(a.data)
    logger.info("date span: %s .. %s", df["date"].min().date(), df["date"].max().date())

    result, test = backtest(df, a.test_start, a.test_end, a.lookback_weeks)
    logger.info("\ntest rows: %d  (%s .. %s)", len(test), a.test_start, a.test_end)
    logger.info("\n%-18s%8s%9s%9s%9s%9s%8s", "corridor", "mean", "MAE",
                "peakMAE", "lastwk", "overall", "skill")
    for _, r in result.iterrows():
        logger.info("%-18s%8.1f%9.2f%9.2f%9.2f%9.2f%7.0f%%",
                    r.corridor, r.mean_minutes, r.mae_seasonal, r.mae_seasonal_peak,
                    r.mae_last_week, r.mae_overall, r.skill_vs_lastweek * 100)
    logger.info("\nweighted MAE: %.2f min (%.1f%% of mean travel time)",
                np.average(result.mae_seasonal, weights=result.n),
                np.average(result.mae_pct_of_mean, weights=result.n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
