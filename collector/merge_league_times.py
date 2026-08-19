# collector/merge_league_times.py
"""
Fill event start times from league APIs and add home games the crawl missed.

Two jobs:

  fill  a crawled event on a date the league also has a home game takes the
        league's exact start time.
  add   a league home game the crawl never saw is inserted. Wayback only
        captured what was on screen that day, so its coverage of a 162-game
        season is partial. This added 572 games.

Covers Oracle Park (MLB StatsAPI) and SAP Center (api-web.nhle.com). Levi's
Stadium and Oakland Arena stay time-less: no free NFL schedule API exists and
Oakland Arena has no single league behind it.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collector.sports_api import mlb_home_games, nhl_home_games  # noqa: E402

logger = logging.getLogger("merge_league_times")

# Venue slug -> how to pull its authoritative schedule.
MLB_VENUE = "oracle-park"
NHL_VENUE = "sap-center"


def collect_league_games(years):
    """Return {venue_slug: {date: game}} for every league-backed venue."""
    by_venue = {MLB_VENUE: {}, NHL_VENUE: {}}

    for year in years:
        try:
            for g in mlb_home_games(f"{year}-01-01", f"{year}-12-31"):
                by_venue[MLB_VENUE][g["start"][:10]] = g
        except Exception as e:
            logger.warning("MLB %s failed: %s", year, e)

    for start_year in years:
        season = f"{start_year}{start_year + 1}"
        try:
            for g in nhl_home_games(season):
                by_venue[NHL_VENUE][g["start"][:10]] = g
        except Exception as e:
            logger.warning("NHL %s failed: %s", season, e)

    for slug, games in by_venue.items():
        logger.info("%s: %d league home games collected", slug, len(games))
    return by_venue


def merge(events, league, source_tag="league-api"):
    """Fill times on matching events, then add unmatched league games."""
    stats = {"filled": 0, "already_timed": 0, "added": 0}
    seen_dates = {slug: set() for slug in league}

    out = []
    for e in events:
        slug = e["venue"]
        if slug in league:
            seen_dates[slug].add(e["start"][:10])
            game = league[slug].get(e["start"][:10])
            if game and not e.get("has_time"):
                # Keep the crawled title, which names the actual event and for a
                # mixed-use arena may not be the league fixture, but take the
                # league's timestamp, which is authoritative.
                e = {**e, "start": game["start"], "has_time": True,
                     "time_source": source_tag}
                stats["filled"] += 1
            elif e.get("has_time"):
                stats["already_timed"] += 1
        out.append(e)

    for slug, games in league.items():
        for date, game in games.items():
            if date in seen_dates[slug]:
                continue
            out.append({
                "venue": slug,
                "start": game["start"],
                "has_time": True,
                "title": game["title"],
                "source_url": source_tag,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "time_source": source_tag,
            })
            stats["added"] += 1

    return sorted(out, key=lambda e: (e["venue"], e["start"])), stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", required=True, help="clean events jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--years", nargs="+", type=int,
                   default=[2022, 2023, 2024, 2025, 2026])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    src = os.path.expanduser(a.events)
    events = [json.loads(l) for l in open(src) if l.strip()]
    logger.info("loaded %d events", len(events))

    league = collect_league_games(a.years)
    merged, stats = merge(events, league)

    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for e in merged:
            f.write(json.dumps(e) + "\n")

    logger.info("filled=%(filled)d already_timed=%(already_timed)d added=%(added)d",
                stats)
    logger.info("%d events -> %s", len(merged), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
