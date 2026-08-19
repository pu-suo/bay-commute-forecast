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
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)     # so the geocoder resolves however this is invoked
from forecast.route import Forecast, plan     # noqa: E402
from site_geocode import geocode              # noqa: E402

logger = logging.getLogger("server")
STATE = {}

# Resolution of the all-day profile and of the arrive-by search. 15 minutes is
# the serving table's own resolution; asking for finer would interpolate rather
# than inform.
PROFILE_STEP_MIN = 15


def current_forecast():
    """
    Reload the table when the nightly job replaces it.

    The server holds the whole forecast in memory, which is what makes a route
    query a dictionary lookup. That is also how it ends up serving yesterday's
    numbers at 04:00 without noticing, so the file's mtime is checked per
    request -- a stat call, against a rebuild that happens once a day.
    """
    path = os.path.join(os.path.expanduser(STATE["serve"]), "forecast.parquet")
    mtime = os.path.getmtime(path)
    if mtime != STATE.get("mtime"):
        logger.info("forecast table changed; reloading")
        STATE["fc"] = Forecast(STATE["serve"], STATE["meta"])
        STATE["mtime"] = mtime
    return STATE["fc"]


def _point(raw):
    lat, lon = (float(x) for x in raw.split(","))
    return lat, lon


def route_response(origin, dest, depart):
    segs, summary = plan(origin, dest, depart, current_forecast(), STATE["osrm"])
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


def profile(origin, dest, date, step=PROFILE_STEP_MIN):
    """Travel time for every departure slot on one day."""
    fc = current_forecast()
    day = pd.Timestamp(date).normalize()
    out = []
    for m in range(0, 24 * 60, step):
        t = day + pd.Timedelta(minutes=m)
        if not (fc.start <= t <= fc.end):
            continue
        _, s = plan(origin, dest, t, fc, STATE["osrm"])
        out.append({"depart": t.isoformat(), "minutes": round(s["total_minutes"], 1),
                    "measured_share": round(s["measured_share"], 3)})
    return out


def solve_arrive_by(origin, dest, arrive, step=PROFILE_STEP_MIN):
    """
    Latest departure that still arrives by the deadline.

    Travel time depends on when you leave, so this cannot be a subtraction --
    leaving 30 minutes earlier can save 45. Walk candidate departures backwards
    from the deadline and take the last one that still lands in time.
    """
    fc = current_forecast()
    target = pd.Timestamp(arrive)
    best = None
    for back in range(0, 24 * 60 + 1, step):
        t = target - pd.Timedelta(minutes=back)
        if t < fc.start:
            break
        _, s = plan(origin, dest, t, fc, STATE["osrm"])
        landing = t + pd.Timedelta(minutes=s["total_minutes"])
        if landing <= target:
            best = {"depart": t.isoformat(), "arrive": landing.isoformat(),
                    "minutes": round(s["total_minutes"], 1),
                    "slack_minutes": round((target - landing).total_seconds() / 60, 1)}
            break
    return best


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("%s", fmt % args)

    GZIP_MIN = 4096

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
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/api/route":
                origin, dest = _point(q["from"][0]), _point(q["to"][0])
                if "arrive" in q:
                    sol = solve_arrive_by(origin, dest, q["arrive"][0])
                    if not sol:
                        return self._send(200, {"error": "no departure inside the horizon"})
                    body = route_response(origin, dest, sol["depart"])
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
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    meta = pd.read_csv(os.path.join(os.path.expanduser(a.data), "_meta", "d04_meta.txt"),
                       sep="\t")
    meta = meta[meta["Type"] == "ML"].rename(columns={"ID": "sensor_id"})
    STATE["serve"], STATE["meta"], STATE["osrm"] = a.serve, meta, a.osrm
    fc = current_forecast()
    logger.info("forecast horizon %s -> %s", fc.start, fc.end)
    logger.info("serving http://localhost:%d", a.port)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
