# forecast/backfill.py
"""
Stream-and-discard backfill: PeMS station_5min -> corridor travel time -> Parquet.

PeMS ships ~29 MB of gzipped text per district-day and there is no server-side
station filter, so every corridor-day requires downloading the whole file. But
almost none of it needs keeping: 660k mainline rows collapse to 288 intervals x
9 corridors. This fetches a day, reduces it, writes the result, and deletes the
raw file, so peak disk is one file rather than 156 GB.

Output layout, one small Parquet per day so the job is resumable at any point
and an interrupted run can never corrupt an existing file:

    <out>/year=2026/corridors_2026_08_13.parquet

Columns: ts, corridor, minutes, n_stations, coverage, pct_observed, imputed_day

Days are processed newest-first, because recent years are the ones the model
actually trains on and a run stopped early should leave the useful data behind.

Credentials come from the environment (see pems_client):
    set -a; . ~/.pems_env; set +a
    python -m forecast.backfill --years 2026 2025 2024 --out ~/traffic-data/corridors
"""
import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "collector"))

from pems_client import (  # noqa: E402
    PemsClient, DATASET_5MIN, DATASET_META,
    parse_station_5min, parse_station_meta,
)
from forecast.corridors import CORRIDORS, corridor_travel_time, is_imputed_day  # noqa: E402

logger = logging.getLogger("backfill")


def load_meta(client, cache_path):
    """Fetch (once) and cache the station metadata that defines corridors."""
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        entries = client.list_files(year=datetime.utcnow().year, dataset=DATASET_META)
        if not entries:
            raise RuntimeError("No station metadata files available.")
        newest = entries[-1]
        logger.info("Fetching station metadata: %s", newest.filename)
        downloaded = client.download(newest, dest_dir=os.path.dirname(cache_path) or ".")
        os.replace(downloaded, cache_path)
    return parse_station_meta(cache_path)


def day_output_path(out_dir, filename):
    """<out>/year=YYYY/corridors_YYYY_MM_DD.parquet from a PeMS filename."""
    stamp = filename.replace("d04_text_station_5min_", "").replace(".txt.gz", "")
    year = stamp[:4]
    return os.path.join(out_dir, f"year={year}", f"corridors_{stamp}.parquet")


def reduce_day(raw_path, meta):
    """Collapse one day of station readings into per-corridor travel time."""
    import pandas as pd

    readings = parse_station_5min(raw_path)
    if readings.empty:
        return None
    imputed = is_imputed_day(readings)

    frames = []
    for corridor in CORRIDORS:
        tt = corridor_travel_time(readings, corridor, meta)
        if tt.empty:
            continue
        tt = tt.reset_index()
        tt["corridor"] = corridor.slug
        frames.append(tt)
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    # Flagged rather than dropped: a fully-imputed day looks entirely normal and
    # is only identifiable here, so the flag has to travel with the rows.
    out["imputed_day"] = imputed
    return out[["ts", "corridor", "minutes", "n_stations",
                "coverage", "pct_observed", "imputed_day"]]


def run(years, out_dir, meta_cache, keep_raw=False, limit=None):
    client = PemsClient()
    client.login()
    meta = load_meta(client, meta_cache)
    logger.info("Station metadata loaded: %d stations", len(meta))

    tasks = []
    for year in years:
        for entry in client.list_files(year=year, dataset=DATASET_5MIN):
            tasks.append(entry)
    # newest first, so an interrupted run leaves the most useful data behind
    tasks.sort(key=lambda e: e.filename, reverse=True)

    pending = [e for e in tasks if not os.path.exists(day_output_path(out_dir, e.filename))]
    logger.info("%d day-files total, %d already done, %d pending",
                len(tasks), len(tasks) - len(pending), len(pending))
    if limit:
        pending = pending[:limit]
        logger.info("Limited to %d for this run.", len(pending))

    raw_dir = os.path.join(out_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)
    done = failed = 0
    started = time.time()

    for i, entry in enumerate(pending, 1):
        dest = day_output_path(out_dir, entry.filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        raw_path = None
        try:
            raw_path = client.download(entry, dest_dir=raw_dir)
            frame = reduce_day(raw_path, meta)
            if frame is None or frame.empty:
                logger.warning("[%d/%d] %s produced no rows", i, len(pending), entry.filename)
                failed += 1
                continue
            # write to a temp name then rename, so a kill mid-write can't leave
            # a truncated Parquet that later looks "already done"
            tmp = dest + ".tmp"
            frame.to_parquet(tmp, index=False)
            os.replace(tmp, dest)
            done += 1
            rate = (time.time() - started) / max(done, 1)
            eta = rate * (len(pending) - i) / 3600
            logger.info("[%d/%d] %s -> %d rows (imputed=%s) | %.1fs/day | ETA %.1fh",
                        i, len(pending), entry.filename, len(frame),
                        bool(frame["imputed_day"].iloc[0]), rate, eta)
        except Exception:
            failed += 1
            logger.error("[%d/%d] FAILED %s\n%s", i, len(pending), entry.filename,
                         traceback.format_exc())
        finally:
            if raw_path and os.path.exists(raw_path) and not keep_raw:
                os.remove(raw_path)

    logger.info("Backfill finished: %d written, %d failed, %.2f h elapsed",
                done, failed, (time.time() - started) / 3600)
    return 0 if failed == 0 else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--meta-cache", default=None)
    p.add_argument("--keep-raw", action="store_true",
                   help="do not delete the .txt.gz after reducing (debug only)")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N pending days (for smoke tests)")
    a = p.parse_args(argv)

    out = os.path.expanduser(a.out)
    meta_cache = os.path.expanduser(a.meta_cache or os.path.join(out, "_meta", "d04_meta.txt"))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)-8s] %(message)s",
                        stream=sys.stdout)
    return run(a.years, out, meta_cache, keep_raw=a.keep_raw, limit=a.limit)


if __name__ == "__main__":
    sys.exit(main())
