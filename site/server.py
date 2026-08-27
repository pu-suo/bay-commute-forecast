# site/server.py
"""
Local server for the parts of the site that cannot be a static file.

Almost all of it can. The map, week view and corridors are JSON on disk and
would deploy to any static host. What needs a process is routing an arbitrary
origin and destination, which calls OSRM, matches the polyline to detectors and
prices the spans.

  GET /api/route?from=lat,lon&to=lat,lon&depart=ISO
  GET /api/route?from=...&to=...&arrive=ISO       solves for departure time
  GET /api/profile?from=...&to=...&date=YYYY-MM-DD
  GET /api/geocode?q=<text>

Everything else is served from site/ as files, gzipped: the road geometry is
6 MB of JSON that compresses to 1.3.

    python site/server.py --port 8000
"""
import argparse
import gzip
import json
import logging
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)     # so the geocoder resolves however this is invoked
from forecast.route import Forecast, PreparedRoute, price_at   # noqa: E402
from site_geocode import geocode              # noqa: E402

logger = logging.getLogger("server")
STATE = {}

# Resolution of the all-day profile and of the arrive-by search. 15 minutes is
# the serving table's own resolution; asking for finer would interpolate rather
# than inform.
PROFILE_STEP_MIN = 15

# Rate limiting, in units of work rather than requests, because the endpoints
# are not equally expensive: a profile prices 96 departure slots and a geocode
# spends someone's API quota. Refilled continuously, so a burst is allowed and a
# sustained loop is not.
BUCKET_CAPACITY = 60
BUCKET_REFILL_PER_SEC = 1.0
COST = {"/api/route": 1, "/api/profile": 8, "/api/geocode": 2,
        "/api/horizon": 0, "/api/health": 0}


def current_forecast():
    """
    Reload the table when the nightly job replaces it.

    The server holds the whole forecast in memory, which makes a route query a
    dictionary lookup and also means it would serve yesterday's numbers at 04:00
    without noticing. The file's mtime is checked per request: one stat call
    against a rebuild that happens once a day.
    """
    path = os.path.join(os.path.expanduser(STATE["serve"]), "forecast.parquet")
    mtime = os.path.getmtime(path)
    if mtime != STATE.get("mtime"):
        logger.info("forecast table changed; reloading")
        STATE["fc"] = Forecast(STATE["serve"], STATE["meta"])
        STATE["mtime"] = mtime
    return STATE["fc"]


SERVE_FILES = ("forecast.parquet", "freeflow.parquet")


def refresh_from_url(base_url, serve_dir, timeout=120):
    """
    Pull the serving tables from a URL if the published copy is newer.

    The nightly job runs on a laptop and the API runs in a container, so
    something has to carry 12 MB between them. Fetching over HTTPS from the
    published site keeps the container stateless: no volume, no inbound SSH, and
    a restart re-reads whatever is current rather than whatever was last pushed
    to it.
    """
    serve_dir = os.path.expanduser(serve_dir)
    os.makedirs(serve_dir, exist_ok=True)
    changed = False
    for name in SERVE_FILES:
        dest = os.path.join(serve_dir, name)
        req = urllib.request.Request(f"{base_url.rstrip('/')}/{name}")
        if os.path.exists(dest):
            stamp = time.strftime("%a, %d %b %Y %H:%M:%S GMT",
                                  time.gmtime(os.path.getmtime(dest)))
            req.add_header("If-Modified-Since", stamp)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 304:
                continue
            logger.warning("refresh %s failed: %s", name, e)
            continue
        except (urllib.error.URLError, TimeoutError) as e:
            logger.warning("refresh %s failed: %s", name, e)
            continue
        tmp = dest + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, dest)
        changed = True
        logger.info("refreshed %s (%.1f MB)", name, len(body) / 1e6)
    return changed


def start_refresher(base_url, serve_dir, every_min):
    def loop():
        while True:
            time.sleep(every_min * 60)
            try:
                refresh_from_url(base_url, serve_dir)
            except Exception:                       # noqa: BLE001
                logger.exception("refresh loop")
    t = threading.Thread(target=loop, daemon=True)
    t.start()


class RateLimiter:
    """Token bucket per client address."""

    def __init__(self, capacity=BUCKET_CAPACITY, refill=BUCKET_REFILL_PER_SEC):
        self.capacity, self.refill = capacity, refill
        self._buckets = {}
        self._lock = threading.Lock()

    def take(self, key, cost):
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens < cost:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - cost, now)
            if len(self._buckets) > 10000:        # bound memory under a spray
                cutoff = now - self.capacity / self.refill
                self._buckets = {k: v for k, v in self._buckets.items()
                                 if v[1] > cutoff}
            return True


def _point(raw):
    lat, lon = (float(x) for x in raw.split(","))
    return lat, lon


def route_response(origin, dest, depart, prepared=None):
    fc = current_forecast()
    prepared = prepared or PreparedRoute(origin, dest, fc.meta, STATE["osrm"])
    segs, summary = price_at(prepared, depart, fc)
    keep = ["kind", "miles", "mph", "minutes", "arrive", "estimate",
            "sensor_id", "freeway", "direction", "lat", "lon"]
    rows = []
    for r in segs.to_dict("records"):
        row = {k: r.get(k) for k in keep}
        row["arrive"] = pd.Timestamp(row["arrive"]).isoformat()
        row["sensor_id"] = int(row["sensor_id"]) if pd.notna(row.get("sensor_id")) else None
        row["freeway"] = None if pd.isna(row.get("freeway")) else int(row["freeway"])
        row["direction"] = None if pd.isna(row.get("direction")) else row["direction"]
        rows.append(row)
    return {"summary": summary, "segments": rows}


def profile(origin, dest, date, step=PROFILE_STEP_MIN, prepared=None):
    """Travel time for every departure slot on one day."""
    fc = current_forecast()
    prepared = prepared or PreparedRoute(origin, dest, fc.meta, STATE["osrm"])
    day = pd.Timestamp(date).normalize()
    out = []
    for m in range(0, 24 * 60, step):
        t = day + pd.Timedelta(minutes=m)
        if not (fc.start <= t <= fc.end):
            continue
        _, s = price_at(prepared, t, fc)
        out.append({"depart": t.isoformat(), "minutes": round(s["total_minutes"], 1),
                    "measured_share": round(s["measured_share"], 3)})
    return out


def solve_arrive_by(origin, dest, arrive, step=PROFILE_STEP_MIN, prepared=None):
    """
    Latest departure that still arrives by the deadline.

    Travel time depends on when you leave, so this cannot be a subtraction:
    leaving 30 minutes earlier can save 45. Walks candidate departures backwards
    from the deadline and takes the last one that still lands in time.
    """
    fc = current_forecast()
    prepared = prepared or PreparedRoute(origin, dest, fc.meta, STATE["osrm"])
    target = pd.Timestamp(arrive)
    best = None
    for back in range(0, 24 * 60 + 1, step):
        t = target - pd.Timedelta(minutes=back)
        if t < fc.start:
            break
        _, s = price_at(prepared, t, fc)
        landing = t + pd.Timedelta(minutes=s["total_minutes"])
        if landing <= target:
            best = {"depart": t.isoformat(), "arrive": landing.isoformat(),
                    "minutes": round(s["total_minutes"], 1),
                    "slack_minutes": round((target - landing).total_seconds() / 60, 1)}
            break
    return best, prepared


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s", fmt % args)

    GZIP_MIN = 4096
    protocol_version = "HTTP/1.1"
    timeout = 20                  # drop a client that stops reading mid-response

    def _allow(self, path):
        cost = COST.get(path, 1)
        if not cost:
            return True
        # X-Forwarded-For only when a proxy we control set it; behind Fly or
        # Pages the socket address is the proxy's, not the visitor's.
        client = self.headers.get("X-Forwarded-For", "").split(",")[0].strip() \
            if STATE.get("behind_proxy") else ""
        return STATE["limiter"].take(client or self.client_address[0], cost)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _cors(self):
        origin = STATE.get("allow_origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Max-Age", "86400")

    def _send(self, code, body, ctype="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {}
        accepts = self.headers.get("Accept-Encoding", "")
        if len(payload) >= self.GZIP_MIN and "gzip" in accepts:
            payload = gzip.compress(payload, 6)
            headers["Content-Encoding"] = "gzip"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        if parsed.path.startswith("/api/") and not self._allow(parsed.path):
            return self._send(429, {"error": "rate limited"})
        try:
            if parsed.path == "/api/route":
                origin, dest = _point(q["from"][0]), _point(q["to"][0])
                if "arrive" in q:
                    sol, prepared = solve_arrive_by(origin, dest, q["arrive"][0])
                    if not sol:
                        return self._send(200, {"error": "no departure inside the horizon"})
                    body = route_response(origin, dest, sol["depart"], prepared)
                    body["arrive_by"] = sol
                    return self._send(200, body)
                return self._send(200, route_response(origin, dest, q["depart"][0]))

            if parsed.path == "/api/profile":
                origin, dest = _point(q["from"][0]), _point(q["to"][0])
                return self._send(200, {"profile": profile(origin, dest, q["date"][0])})

            if parsed.path == "/api/geocode":
                q = q.get("q", [""])[0].strip()
                if len(q) < 3:
                    return self._send(200, {"results": []})
                return self._send(200, {"results": geocode(q)})

            if parsed.path == "/api/health":
                fc = current_forecast()
                age_h = (time.time() - STATE.get("mtime", 0)) / 3600
                stale = age_h > 36
                return self._send(503 if stale else 200, {
                    "ok": not stale,
                    "forecast_age_hours": round(age_h, 1),
                    "horizon_end": fc.end.isoformat(),
                })

            if parsed.path == "/api/horizon":
                fc = current_forecast()
                return self._send(200, {"start": fc.start.isoformat(),
                                        "end": fc.end.isoformat(), "slot": fc.slot})
            return self._static(parsed.path)
        except (KeyError, ValueError) as e:
            self._send(400, {"error": f"{type(e).__name__}: {e}"})
        except Exception as e:                       # noqa: BLE001
            logger.exception("request failed")
            self._send(500, {"error": str(e)})

    def _static(self, path):
        root = os.path.dirname(os.path.abspath(__file__))
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.normpath(os.path.join(root, rel))
        if not full.startswith(root) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        types = {".html": "text/html", ".js": "application/javascript",
                 ".css": "text/css", ".json": "application/json"}
        ctype = types.get(os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as f:
            self._send(200, f.read(), ctype)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--serve", default="~/traffic-data/serve")
    p.add_argument("--data", default="~/traffic-data/stations")
    p.add_argument("--osrm", default=os.environ.get("OSRM_URL", "http://localhost:5001"))
    p.add_argument("--allow-origin", default=os.environ.get("ALLOW_ORIGIN"),
                   help="site origin allowed to call the API cross-origin, e.g. "
                        "https://pu-suo.github.io. Unset means same-origin only.")
    p.add_argument("--behind-proxy", action="store_true",
                   default=bool(os.environ.get("BEHIND_PROXY")),
                   help="trust X-Forwarded-For for rate limiting")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--serve-url", default=os.environ.get("SERVE_URL"),
                   help="base URL to pull forecast.parquet from, for a container "
                        "that has no access to the machine building it")
    p.add_argument("--refresh-min", type=int,
                   default=int(os.environ.get("REFRESH_MIN", "60")))
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if a.serve_url:
        logger.info("pulling serving tables from %s", a.serve_url)
        refresh_from_url(a.serve_url, a.serve)

    meta = pd.read_csv(os.path.join(os.path.expanduser(a.data), "_meta", "d04_meta.txt"),
                       sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    STATE["serve"], STATE["meta"], STATE["osrm"] = a.serve, meta, a.osrm
    STATE["allow_origin"] = a.allow_origin
    STATE["behind_proxy"] = a.behind_proxy
    STATE["limiter"] = RateLimiter()
    fc = current_forecast()
    logger.info("forecast horizon %s -> %s", fc.start, fc.end)
    if a.allow_origin:
        logger.info("cross-origin allowed from %s", a.allow_origin)
    if a.serve_url:
        start_refresher(a.serve_url, a.serve, a.refresh_min)
    logger.info("serving on http://%s:%d", a.host, a.port)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
