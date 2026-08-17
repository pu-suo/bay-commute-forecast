# collector/venue_events.py
"""
Venue event calendar crawler — live pages and Wayback backfill.

One venue calendar carries every event type at that venue. The SAP Center page
lists "Sharks vs. Bruins", "Bellator" and "Disney On Ice" together, so there is
no need for separate sports and concert feeds; crawl the venue and you get both.

Two extraction strategies, tried in order:

  1. schema.org JSON-LD  — structured, and crucially carries a full ISO
     startDate *with time of day*. Shoreline publishes this.
  2. pipe-token text walk — for sites that render events as plain markup.
     Levi's emits "Aug | 21 | Karol G | ... | BUY TICKETS"; SAP Center emits
     "Sep. | 10 | , 2026 | Soda Stereo | Event Starts | 8:00 PM".

Start time matters more than it looks. A measured 49ers game at Levi's moved
SR-237 East by +20 min, but the spike lands at 17:55 for an afternoon kickoff
and 21:35 for a night one — a day-level flag cannot place it. Events without a
parsed time are marked has_time=False so downstream can treat them differently
rather than silently assuming an afternoon start.

The same parsers run against Wayback snapshots, which is how history is
recovered for sites that publish no archive of their own.

    python -m collector.venue_events --live --out ~/traffic-data/events
    python -m collector.venue_events --wayback --from 2023 --to 2026 --out ~/traffic-data/events
"""
import argparse
import gzip
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forecast.corridors import VENUES  # noqa: E402

USER_AGENT = "Mozilla/5.0 (commute-forecast venue archiver)"
REQUEST_TIMEOUT = 60
POLITENESS_SECONDS = 2.0
RETRY_ATTEMPTS = 3

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# Everything downstream — corridor timestamps, kickoff-relative features — is
# Bay Area wall-clock. Venues are inconsistent about this: Shoreline publishes
# "-07:00" offsets while Stanford publishes UTC with a Z suffix, so a raw
# comparison lands a Friday night kickoff on Saturday morning.
LOCAL_TZ = ZoneInfo("America/Los_Angeles")

# Tokens that terminate an event title in the pipe-delimited layouts.
STOP_TOKENS = re.compile(
    r"^(buy tickets|tickets|details|more info|event starts|buy|info|"
    r"group tickets|learn more|sold out|on sale.*)$", re.I)

MONTH_TOKEN = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?$", re.I)
DAY_TOKEN = re.compile(r"^(\d{1,2})$")
YEAR_TOKEN = re.compile(r"^,?\s*(20\d{2})$")
TIME_TOKEN = re.compile(r"^(\d{1,2}):(\d{2})\s*(am|pm)$", re.I)

logger = logging.getLogger("venue_events")


@dataclass
class Event:
    venue: str
    start: str          # ISO 8601; date-only when no time was published
    has_time: bool
    title: str
    source_url: str
    captured_at: str    # when THIS observation was made — the point-in-time key


def _fetch(url):
    """GET with retry and gzip handling. Returns decoded text."""
    last = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
            return raw.decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < RETRY_ATTEMPTS:
                time.sleep(4 * attempt)
    raise RuntimeError(f"fetch failed for {url}: {last}")


def _location_text(node):
    """Flatten a schema.org location into searchable text."""
    loc = node.get("location")
    if isinstance(loc, str):
        return loc
    if isinstance(loc, dict):
        parts = [str(loc.get("name") or "")]
        addr = loc.get("address")
        if isinstance(addr, dict):
            parts += [str(addr.get(k) or "") for k in
                      ("streetAddress", "addressLocality", "addressRegion")]
        elif isinstance(addr, str):
            parts.append(addr)
        return " ".join(parts)
    return ""


def events_from_jsonld(html, venue=None):
    """
    Pull schema.org Event objects out of any ld+json blocks. Best case.

    When the markup declares a location, it is checked against the venue. Team
    schedule pages list home *and* away fixtures — Stanford's page happily
    advertises "Stanford at California", which happens in Berkeley. Counting an
    away game as a venue event injects pure noise into the event coefficient,
    so anything whose stated location clearly isn't this venue is dropped.
    """
    found = []
    venue_words = set()
    if venue is not None:
        venue_words = {w for w in re.split(r'\W+', venue.name.lower())
                       if len(w) > 3 and w not in {"stadium", "center", "centre",
                                                   "arena", "park", "amphitheatre"}}

    for m in re.finditer(r'<script[^>]*application/ld\+json[^>]*>(.*?)</script>',
                         html, re.S | re.I):
        try:
            doc = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        stack = [doc]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                if "Event" in str(node.get("@type", "")) and node.get("startDate"):
                    loc = _location_text(node).lower()
                    away = bool(loc) and bool(venue_words) and not (venue_words & set(
                        re.split(r'\W+', loc)))
                    if not away:
                        start = str(node["startDate"])
                        found.append((start, str(node.get("name") or "").strip(),
                                      "T" in start))
                stack.extend(v for v in node.values() if isinstance(v, (list, dict)))
    return found


def _tokenize(html):
    """Strip markup to a pipe-delimited token stream, preserving order."""
    t = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.S | re.I)
    t = re.sub(r'<[^>]+>', ' | ', t)
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&")
           .replace("&#038;", "&").replace("&#039;", "'").replace("&times;", "x"))
    t = re.sub(r'\s+', ' ', t)
    return [tok.strip() for tok in t.split("|") if tok.strip()]


def events_from_text(tokens, default_year):
    """
    Walk the token stream looking for <month> <day> [, year] <title...> patterns.

    Titles run until a stop token or the next date. Year is taken from an
    explicit token when present, otherwise inferred from the capture date with
    a wrap forward when the month has already passed (calendars list upcoming
    events, so a January entry seen in November means next January).
    """
    out = []
    i = 0
    while i < len(tokens) - 1:
        if not MONTH_TOKEN.match(tokens[i]):
            i += 1
            continue
        month = MONTHS[tokens[i][:3].lower()]
        dm = DAY_TOKEN.match(tokens[i + 1])
        if not dm:
            i += 1
            continue
        day = int(dm.group(1))
        j = i + 2
        year = default_year
        if j < len(tokens):
            ym = YEAR_TOKEN.match(tokens[j])
            if ym:
                year = int(ym.group(1))
                j += 1
        else:
            j = len(tokens)

        title_parts, hh, mm = [], None, None
        while j < len(tokens) and len(title_parts) < 6:
            tok = tokens[j]
            if MONTH_TOKEN.match(tok):
                break
            tmatch = TIME_TOKEN.match(tok)
            if tmatch:
                hh, mm = int(tmatch.group(1)) % 12, int(tmatch.group(2))
                if tmatch.group(3).lower() == "pm":
                    hh += 12
                j += 1
                continue
            if STOP_TOKENS.match(tok):
                j += 1
                if title_parts:
                    break
                continue
            if re.fullmatch(r'[\d\s,.\-]+', tok):   # stray date fragments
                j += 1
                continue
            title_parts.append(tok)
            j += 1

        title = " - ".join(title_parts).strip()
        if title and 1 <= day <= 31:
            try:
                if hh is not None:
                    start = datetime(year, month, day, hh, mm).isoformat()
                    out.append((start, title, True))
                else:
                    start = datetime(year, month, day).date().isoformat()
                    out.append((start, title, False))
            except ValueError:
                pass
        i = max(j, i + 1)
    return out


# Calendars repeat the date inside the title cell ("Karol G - ... - Aug 21") and
# some prefix a stray year fragment ("/ 2026 - Oakland Roots"). Both are layout
# artefacts, not part of the event name.
_TRAILING_DATE = re.compile(
    r"\s*[-–]\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\s*$", re.I)
_LEADING_FRAGMENT = re.compile(r"^[\s/,\-]*(20\d{2})?\s*[-–]\s*")


def to_local_iso(start):
    """
    Normalise a published startDate to Bay Area wall-clock.

    Accepts date-only strings (returned unchanged), offset-aware timestamps,
    and UTC 'Z' forms with optional fractional seconds. Naive timestamps are
    assumed already local, which is what the text parser produces.
    """
    s = start.strip()
    if "T" not in s:
        return s, False
    iso = re.sub(r"\.\d+", "", s).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return s, False
    if dt.tzinfo is None:
        return dt.isoformat(), True
    return dt.astimezone(LOCAL_TZ).replace(tzinfo=None).isoformat(), True


def _clean_title(title):
    title = _LEADING_FRAGMENT.sub("", title)
    prev = None
    while prev != title:                     # strip repeated trailing dates
        prev = title
        title = _TRAILING_DATE.sub("", title)
    return title.strip(" -–/,")


def parse_events(html, venue, source_url, captured_at, default_year):
    """JSON-LD first; fall back to the token walk. Deduplicated on (start, title)."""
    raw = events_from_jsonld(html, venue)
    strategy = "jsonld"
    if len(raw) < 3:
        raw = events_from_text(_tokenize(html), default_year)
        strategy = "text"

    seen, events = set(), []
    for start, title, has_time in raw:
        title = _clean_title(title)
        local_start, parsed_time = to_local_iso(start)
        has_time = has_time and parsed_time
        key = (local_start[:16], title.lower()[:60])
        if not title or key in seen:
            continue
        seen.add(key)
        events.append(Event(venue.slug, local_start, has_time, title[:200],
                            source_url, captured_at))
    return events, strategy


def fetch_live(venue):
    """Current calendar page: forward-looking events, captured now."""
    now = datetime.now(timezone.utc)
    html = _fetch(venue.calendar_url)
    time.sleep(POLITENESS_SECONDS)
    return parse_events(html, venue, venue.calendar_url, now.isoformat(), now.year)


def wayback_snapshots(url, year_from, year_to, limit=400):
    """Monthly-collapsed 200-status snapshot timestamps for a URL."""
    q = ("https://web.archive.org/cdx/search/cdx?url="
         + urllib.parse.quote(url.replace("https://", "").replace("http://", ""), safe="")
         + f"&matchType=prefix&output=json&fl=timestamp&filter=statuscode:200"
         f"&from={year_from}&to={year_to}&collapse=timestamp:6&limit={limit}")
    raw = _fetch(q)
    rows = json.loads(raw) if raw.strip() else []
    return sorted({r[0] for r in rows[1:]}) if len(rows) > 1 else []


def fetch_wayback(venue, year_from, year_to, max_snapshots=40):
    """
    Replay archived calendar pages to recover history.

    Each snapshot's captured_at is the archive timestamp, not now, so the
    resulting rows remain point-in-time correct: they record what was
    announced as of that date.
    """
    stamps = wayback_snapshots(venue.calendar_url, year_from, year_to)
    if len(stamps) > max_snapshots:
        step = len(stamps) / max_snapshots
        stamps = [stamps[int(i * step)] for i in range(max_snapshots)]
    logger.info("%s: %d wayback snapshots to replay", venue.slug, len(stamps))

    all_events, strategies = [], set()
    for stamp in stamps:
        url = f"https://web.archive.org/web/{stamp}id_/{venue.calendar_url}"
        try:
            html = _fetch(url)
        except RuntimeError as e:
            logger.warning("  %s %s: %s", venue.slug, stamp, e)
            continue
        captured = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc).isoformat()
        evs, strategy = parse_events(html, venue, url, captured, int(stamp[:4]))
        strategies.add(strategy)
        all_events.extend(evs)
        logger.info("  %s %s -> %d events (%s)", venue.slug, stamp[:8], len(evs), strategy)
        time.sleep(POLITENESS_SECONDS)
    return all_events, strategies


def write_events(events, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"events_{tag}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(asdict(e)) + "\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live", action="store_true")
    p.add_argument("--wayback", action="store_true")
    p.add_argument("--from", dest="year_from", type=int, default=2023)
    p.add_argument("--to", dest="year_to", type=int, default=2026)
    p.add_argument("--venues", nargs="*", default=None, help="venue slugs; default all")
    p.add_argument("--out", default="data/events")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)-8s] %(message)s")
    out = os.path.expanduser(a.out)
    venues = [v for v in VENUES if not a.venues or v.slug in a.venues]

    if a.live:
        allev = []
        for v in venues:
            try:
                evs, strategy = fetch_live(v)
                logger.info("%-18s %3d events via %s", v.slug, len(evs), strategy)
                allev.extend(evs)
            except Exception as e:
                logger.error("%-18s FAILED: %s", v.slug, e)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        logger.info("live total %d -> %s", len(allev), write_events(allev, out, f"live_{stamp}"))

    if a.wayback:
        allev = []
        for v in venues:
            try:
                evs, strategies = fetch_wayback(v, a.year_from, a.year_to)
                logger.info("%-18s %4d events from wayback (%s)",
                            v.slug, len(evs), ",".join(strategies) or "none")
                allev.extend(evs)
            except Exception as e:
                logger.error("%-18s wayback FAILED: %s", v.slug, e)
        logger.info("wayback total %d -> %s", len(allev),
                    write_events(allev, out, f"wayback_{a.year_from}_{a.year_to}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
