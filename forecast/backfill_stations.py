# forecast/backfill_stations.py
"""
Backfill PeMS station_5min at network scale, streaming and discarding.

Same shape as forecast.backfill but keeps every mainline detector instead of
collapsing to nine corridors. One file per day:

    station int32 | tod int16 | speed float32 | pct_observed int8

zstd, about 740 KB a day, 1.4 GB for 2021-2026. PeMS ships ~29 MB per
district-day with no server-side filter, so each day is downloaded whole,
reduced, written, and deleted. Peak disk is one file rather than 156 GB.
"""
import argparse, logging, os, sys, time, traceback
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collector.pems_client import (PemsClient, DATASET_5MIN, DATASET_META,
                                   parse_station_5min, parse_station_meta)
from forecast.corridors import SPEED_LANE_TYPES

logger = logging.getLogger("backfill_stations")


def load_meta(client, cache_path):
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        entries = client.list_files(year=datetime.utcnow().year, dataset=DATASET_META)
        got = client.download(entries[-1], dest_dir=os.path.dirname(cache_path) or ".")
        os.replace(got, cache_path)
    return parse_station_meta(cache_path)


def day_path(out_dir, filename):
    stamp = filename.replace("d04_text_station_5min_", "").replace(".txt.gz", "")
    return os.path.join(out_dir, f"year={stamp[:4]}", f"stations_{stamp}.parquet")


def reduce_day(raw_path, keep_ids):
    readings = parse_station_5min(raw_path, mainline_only=True)
    if readings.empty:
        return None
    df = readings[readings["sensor_id"].astype(str).isin(keep_ids)].copy()
    df = df[df["avg_speed"].notna() & (df["avg_speed"] > 0)]
    if df.empty:
        return None
    ts = pd.to_datetime(df["ts"])
    out = pd.DataFrame({
        "station": df["sensor_id"].astype(np.int32),
        "tod": (ts.dt.hour * 60 + ts.dt.minute).astype(np.int16),
        "speed": df["avg_speed"].astype(np.float32),
        "pct_observed": df["pct_observed"].fillna(0).clip(0, 100).astype(np.int8),
    })
    return out.sort_values(["station", "tod"], ignore_index=True)


def run(years, out_dir, meta_cache, limit=None):
    client = PemsClient(); client.login()
    meta = load_meta(client, meta_cache)
    ml = meta[meta["Type"].isin(SPEED_LANE_TYPES)].dropna(subset=["Length"])
    keep = set(ml["sensor_id"].astype(str))
    logger.info("keeping %d speed-reporting stations (%.0f centreline miles)",
                len(keep), ml["Length"].sum())

    tasks = []
    for y in years:
        tasks += client.list_files(year=y, dataset=DATASET_5MIN)
    tasks.sort(key=lambda e: e.filename, reverse=True)
    pending = [e for e in tasks if not os.path.exists(day_path(out_dir, e.filename))]
    logger.info("%d days total, %d pending", len(tasks), len(pending))
    if limit:
        pending = pending[:limit]

    raw_dir = os.path.join(out_dir, "_raw"); os.makedirs(raw_dir, exist_ok=True)
    done = failed = 0; t0 = time.time()
    for i, entry in enumerate(pending, 1):
        dest = day_path(out_dir, entry.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        raw = None
        try:
            raw = client.download(entry, dest_dir=raw_dir)
            frame = reduce_day(raw, keep)
            if frame is None or frame.empty:
                failed += 1; continue
            tmp = dest + ".tmp"
            frame.to_parquet(tmp, index=False, compression="zstd")
            os.replace(tmp, dest)
            done += 1
            rate = (time.time() - t0) / max(done, 1)
            logger.info("[%d/%d] %s -> %d rows, %.1f KB | %.1fs/day | ETA %.1fh",
                        i, len(pending), entry.filename, len(frame),
                        os.path.getsize(dest) / 1024, rate,
                        rate * (len(pending) - i) / 3600)
        except Exception:
            failed += 1
            logger.error("[%d/%d] FAILED %s\n%s", i, len(pending), entry.filename,
                         traceback.format_exc())
        finally:
            if raw and os.path.exists(raw):
                os.remove(raw)
    logger.info("done: %d written, %d failed, %.2f h", done, failed, (time.time()-t0)/3600)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    out = os.path.expanduser(a.out)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)-8s] %(message)s")
    return run(a.years, out, os.path.join(out, "_meta", "d04_meta.txt"), a.limit)


if __name__ == "__main__":
    sys.exit(main())
