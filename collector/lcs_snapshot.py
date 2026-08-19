# collector/lcs_snapshot.py
"""
Daily snapshot of the Caltrans Lane Closure System for district 4.

Caltrans publishes current state only. There is no historical archive and no way
to ask for last week's planned closures, so the archive starts the day you start
collecting and a missed day cannot be recovered. Runs under its own launchd
agent at 03:05 for that reason.

No authentication. Each run writes one gzipped JSON file named for the UTC date,
so re-running the same day is idempotent.

As of 2026-08-14: 1,456 district-4 records, 854 freeway-relevant, 380 mainline,
86% future-dated with a median lead time of 10 days. Mainline records carry
begin and end postmiles, which join to corridors via Abs_PM.
"""
import argparse
import gzip
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

FEED_URL = "https://cwwp2.dot.ca.gov/data/d4/lcs/lcsStatusD04.json"
USER_AGENT = "Mozilla/5.0 (commute-forecast LCS archiver)"
REQUEST_TIMEOUT = 120
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 15

# Facilities that can affect freeway travel time. "Conventional Hwy" and
# "Surface Street" closures are excluded from the summary count because they
# don't touch the corridors, though they are still archived in the raw file.
FREEWAY_FACILITIES = {
    "Mainline", "On Ramp", "Off Ramp", "Connector",
    "Toll Bridge", "Tunnels/Tubes", "Toll Plaza",
}

logger = logging.getLogger(__name__)


def fetch_feed(url=FEED_URL):
    """Fetch the LCS feed with retry/backoff. Returns parsed JSON."""
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read()
            logger.info("Fetched %d bytes from LCS feed.", len(raw))
            return json.loads(raw)
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            last_error = e
            logger.warning("LCS fetch failed (attempt %d/%d): %s",
                           attempt, RETRY_ATTEMPTS, e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"LCS feed unreachable after retries: {last_error}")


def iter_closures(payload):
    """
    Yield individual closure records.

    The feed nests as {"<key>": [ {"<key2>": {...record...}}, ... ]}, with the
    outer and inner key names undocumented and subject to change, so unwrap
    structurally rather than by name.
    """
    items = payload if isinstance(payload, list) else next(iter(payload.values()))
    for item in items:
        if isinstance(item, dict) and len(item) == 1:
            inner = next(iter(item.values()))
            yield inner if isinstance(inner, dict) else item
        else:
            yield item


def summarize(payload):
    """Count records by facility class so the log line is worth reading."""
    total = freeway = mainline = 0
    for rec in iter_closures(payload):
        total += 1
        facility = (rec.get("closure") or {}).get("facility")
        if facility in FREEWAY_FACILITIES:
            freeway += 1
        if facility == "Mainline":
            mainline += 1
    return total, freeway, mainline


def write_snapshot(payload, out_dir, stamp=None):
    """Write one gzipped snapshot named for the UTC date. Idempotent per day."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = stamp or datetime.now(timezone.utc)
    path = os.path.join(out_dir, f"lcs_d04_{stamp:%Y%m%d}.json.gz")

    envelope = {
        # Captured-at is why the archive exists: it records what was
        # knowable on this date, which is what point-in-time features need.
        "captured_at_utc": stamp.isoformat(),
        "source_url": FEED_URL,
        "payload": payload,
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(envelope, f)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/lcs",
                        help="directory for dated snapshots (default: data/lcs)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        stream=sys.stdout,
    )

    payload = fetch_feed()
    total, freeway, mainline = summarize(payload)
    path = write_snapshot(payload, args.out)
    logger.info("Archived %d closures (%d freeway-relevant, %d mainline) -> %s (%d bytes)",
                total, freeway, mainline, path, os.path.getsize(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
