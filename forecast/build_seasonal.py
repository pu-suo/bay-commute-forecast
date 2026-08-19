# forecast/build_seasonal.py
"""
Build the (station, weekday, time-of-day) speed profile that everything else
leans on.

Two builds exist and they must not be confused:

  _seasonal_trainonly.parquet   fitted on days before the evaluation split.
                                Used by training and by validation, so that
                                nothing in a scored window can influence the
                                baseline it is scored against.
  _seasonal.parquet             fitted on everything up to today. Used by the
                                nightly serving job, because in production the
                                right profile is the freshest causal one.

Writing them to separate files rather than regenerating one in place is what
keeps a nightly job from silently invalidating a published accuracy figure.

    python -m forecast.build_seasonal --through 2025-01-01 \
        --out ~/traffic-data/stations/_seasonal_trainonly.parquet
"""
import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.features_stations import build_seasonal   # noqa: E402

logger = logging.getLogger("build_seasonal")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--through", default=None,
                   help="exclusive upper bound; default today (serving build)")
    p.add_argument("--out", default=None,
                   help="default: <data>/_seasonal.parquet")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        stream=sys.stdout)

    data = os.path.expanduser(a.data)
    through = a.through or str(pd.Timestamp.now().normalize().date())
    out = os.path.expanduser(a.out or os.path.join(data, "_seasonal.parquet"))

    s = build_seasonal(data, train_end=through)
    tmp = out + ".tmp"
    s.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out)          # atomic: a reader never sees a partial table
    logger.info("wrote %s cells through %s -> %s (%.1f MB)",
                f"{len(s):,}", through, out, os.path.getsize(out) / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
