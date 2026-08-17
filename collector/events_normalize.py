# collector/events_normalize.py
"""
Normalise raw crawled events into a modelling-ready table.

The crawlers deliberately over-collect: it is cheaper to filter junk afterwards
than to re-run a 30-minute Wayback replay because a parser was too strict. This
pass does the cleanup that must happen before anything trains on the rows.

Four fixes, each from a real defect observed in the 2022-2026 crawl:

  1. Times embedded in titles. Chase Center renders
     "Wednesday April 27 - 07:00 PM - Chase Center - Game 5: Warriors ..." so the
     start time exists but arrives as prose. Recovered and promoted to the
     timestamp, which matters because event effects are timed, not daily.

  2. Non-events. "2026 Premium Season Tickets Wait List", "Protocol Update" and
     "Events Calendar - Filter by" are ticket products and site chrome. Left in,
     they become phantom event-days that teach the model events do nothing.

  3. Implausible times. Stanford's feed yields 10:41 and 10:45 kickoffs, which
     are not real slots — almost certainly placeholders. Times at odd minutes
     are demoted to date-only rather than trusted.

  4. Duplicates across snapshots. Every Wayback snapshot re-lists the same
     upcoming events. Dedupe keeps the EARLIEST observation, so captured_at
     records when an event was first announced — which is the point-in-time
     fact a forecast issued on a given date is allowed to know.
"""
import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime

logger = logging.getLogger("events_normalize")

# Rows that are site furniture or merchandise rather than something people drive to.
NON_EVENT = re.compile(
    r"(season tickets?|wait\s*list|protocol update|events? calendar|filter by|"
    r"expand_more|parking|gift card|newsletter|subscribe|group tickets?|"
    r"suite (rental|holder)|private events?|^more events?$|^calendar$|^events?$)", re.I)

# "Wednesday April 27 • 07:00 PM • Chase Center - <real title>"
EMBEDDED = re.compile(
    r"^\s*(?:mon|tues|wednes|thurs|fri|satur|sun)day\s+[a-z]+\s+\d{1,2}\s*[•·|-]\s*"
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s*[•·|-]\s*(?:[^-•·|]{0,40}[•·|-]\s*)?(.+)$", re.I)

# A leading weekday/date/venue run with no time, e.g. "Saturday May 06 - Chase Center - X"
LEADING_DATE = re.compile(
    r"^\s*(?:mon|tues|wednes|thurs|fri|satur|sun)day\s+[a-z]+\s+\d{1,2}\s*[•·|-]\s*", re.I)

# Real venues start events on the hour, half hour, or quarter hour. Anything else
# (10:41, 10:45 is borderline but 10:41 is not) is a placeholder, not a schedule.
PLAUSIBLE_MINUTES = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}


def _recover_embedded_time(title):
    """Return (clean_title, hh, mm) pulling any time out of the title prose."""
    m = EMBEDDED.match(title)
    if m:
        hh, mm = int(m.group(1)) % 12, int(m.group(2))
        if m.group(3).lower() == "pm":
            hh += 12
        return m.group(4).strip(" -•·|"), hh, mm
    return LEADING_DATE.sub("", title).strip(" -•·|"), None, None


def _plausible_time(dt):
    """Reject placeholder timestamps: midnight, or minutes off the 5-minute grid."""
    if dt.hour == 0 and dt.minute == 0:
        return False
    return dt.minute in PLAUSIBLE_MINUTES


def normalize(rows):
    """Clean, retime and dedupe raw crawler rows. Returns (events, stats)."""
    stats = {"in": len(rows), "non_event": 0, "time_recovered": 0,
             "time_demoted": 0, "deduped": 0}
    cleaned = []
    for r in rows:
        title = (r.get("title") or "").strip()
        if not title or NON_EVENT.search(title):
            stats["non_event"] += 1
            continue

        start, has_time = r["start"], bool(r.get("has_time"))
        title, hh, mm = _recover_embedded_time(title)
        if not title:
            stats["non_event"] += 1
            continue

        if hh is not None and not has_time:
            start = f"{start[:10]}T{hh:02d}:{mm:02d}:00"
            has_time = True
            stats["time_recovered"] += 1

        if has_time:
            try:
                dt = datetime.fromisoformat(start)
            except ValueError:
                dt = None
            if dt is None or not _plausible_time(dt):
                start, has_time = start[:10], False
                stats["time_demoted"] += 1

        cleaned.append({**r, "title": title, "start": start, "has_time": has_time})

    # Keep the earliest observation of each event: captured_at then means
    # "first announced by", which is what a point-in-time feature needs.
    best = {}
    for e in cleaned:
        key = (e["venue"], e["start"][:10], e["title"].lower()[:50])
        prior = best.get(key)
        if prior is None or e["captured_at"] < prior["captured_at"]:
            best[key] = e
        # a later snapshot may have gained a time the first one lacked
        elif e["has_time"] and not prior["has_time"]:
            best[key] = {**prior, "start": e["start"], "has_time": True}
    stats["deduped"] = len(cleaned) - len(best)
    stats["out"] = len(best)
    return sorted(best.values(), key=lambda e: (e["venue"], e["start"])), stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    rows = []
    for path in a.inputs:
        with open(os.path.expanduser(path)) as f:
            rows.extend(json.loads(line) for line in f if line.strip())

    events, stats = normalize(rows)
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    logger.info("in=%(in)d  dropped_non_event=%(non_event)d  "
                "time_recovered=%(time_recovered)d  time_demoted=%(time_demoted)d  "
                "deduped=%(deduped)d  out=%(out)d", stats)
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
