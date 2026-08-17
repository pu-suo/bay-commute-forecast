# collector/sports_api.py
"""
Official league schedule APIs — the reliable source of event *start times*.

Venue calendars give reliable dates but are inconsistent about times: Shoreline
and Stanford publish them, Levi's and Oakland Arena don't. Start time is not
optional here. A measured 49ers home game moved SR-237 East by +20 min, but the
spike lands at 17:55 for an afternoon kickoff and 21:35 for a night one, so a
date-only event cannot place its own effect.

Where a league publishes a free API, prefer it over the venue page: it carries
exact kickoff times, an unambiguous venue name for home/away filtering, and
years of history.

Verified working 2026-08-15:
    MLB  statsapi.mlb.com          -> Oracle Park       (times, venue, history)
    NHL  api-web.nhle.com          -> SAP Center        (times, venue, history)

Verified NOT usable:
    ESPN site.api.espn.com         -> HTTP 403, Akamai block
    TheSportsDB                    -> HTTP 503

Still unsolved: **NFL**, which is the league that matters most here because
Levi's Stadium is the only venue with a confirmed large traffic effect. The
Levi's calendar supplies game dates; kickoff times need another source, and
Wikipedia season pages are the most likely fallback since they tabulate them.
Until that is closed, Levi's NFL events carry has_time=False.
"""
import gzip
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
USER_AGENT = "Mozilla/5.0 (commute-forecast schedule reader)"
REQUEST_TIMEOUT = 45
POLITENESS_SECONDS = 1.0

MLB_SCHEDULE = ("https://statsapi.mlb.com/api/v1/schedule"
                "?sportId=1&teamId={team}&startDate={start}&endDate={end}&hydrate=venue")
NHL_SEASON = "https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"

# team ids / abbreviations for the venues in the registry
MLB_GIANTS = 137
NHL_SHARKS = "SJS"

logger = logging.getLogger("sports_api")


def _get_json(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    time.sleep(POLITENESS_SECONDS)
    return json.loads(raw)


def _utc_to_local(stamp):
    """'2025-07-08T01:45:00Z' -> naive local ISO. Leagues publish UTC."""
    dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return dt.astimezone(LOCAL_TZ).replace(tzinfo=None).isoformat()


def mlb_home_games(start_date, end_date, team=MLB_GIANTS, venue_match="Oracle Park"):
    """Giants home games at Oracle Park between two ISO dates."""
    data = _get_json(MLB_SCHEDULE.format(team=team, start=start_date, end=end_date))
    out = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            venue = (game.get("venue") or {}).get("name") or ""
            if venue_match.lower() not in venue.lower():
                continue          # away game, or a neutral site
            out.append({
                "start": _utc_to_local(game["gameDate"]),
                "has_time": True,
                "title": (f'{game["teams"]["away"]["team"]["name"]} at '
                          f'{game["teams"]["home"]["team"]["name"]}'),
                "day_night": game.get("dayNight"),
                "venue": venue,
            })
    logger.info("MLB: %d home games %s..%s", len(out), start_date, end_date)
    return out


def nhl_home_games(season, team=NHL_SHARKS, venue_match="SAP Center"):
    """
    Sharks home games for a season.

    `season` is the NHL's concatenated form, e.g. 20252026.
    """
    data = _get_json(NHL_SEASON.format(team=team, season=season))
    out = []
    for game in data.get("games", []):
        venue = (game.get("venue") or {}).get("default") or ""
        if venue_match.lower() not in venue.lower():
            continue
        away = (game.get("awayTeam") or {}).get("abbrev", "?")
        home = (game.get("homeTeam") or {}).get("abbrev", "?")
        out.append({
            "start": _utc_to_local(game["startTimeUTC"]),
            "has_time": True,
            "title": f"{away} at {home}",
            "day_night": None,
            "venue": venue,
        })
    logger.info("NHL: %d home games in %s", len(out), season)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("MLB / Oracle Park, July 2025:")
    for g in mlb_home_games("2025-07-01", "2025-07-31")[:5]:
        print(f"  {g['start']}  {g['day_night']:<6} {g['title']}")
    print("\nNHL / SAP Center, 2025-26:")
    for g in nhl_home_games("20252026")[:5]:
        print(f"  {g['start']}  {g['title']}")
