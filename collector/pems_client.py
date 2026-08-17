# collector/pems_client.py
"""
Caltrans PeMS ingestion client.

PeMS publishes no documented API, but its Data Clearinghouse UI is driven by a
JSON endpoint that is perfectly usable once you hold a session cookie:

    GET /?srq=clearinghouse&district_id=<n>&geotag=&yy=<year>
        &type=<dataset>&returnformat=text

That endpoint returns in well under a second, while the equivalent HTML pages
take 20-30s. Never scrape the UI - list via JSON, then fetch files by id.

Credentials come from the environment, never from a file:

    export PEMS_USERNAME='you@example.com'
    export PEMS_PASSWORD='...'

Register at https://pems.dot.ca.gov/?dnode=apply (approval takes 1-2 days).

All of the below was verified end-to-end against district 4 on 2026-08-14:
login, JSON listing, file download, and parsing of both dataset types.
"""
import gzip
import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import requests

BASE_URL = "https://pems.dot.ca.gov/"
LOGIN_FAILED_MARKER = "Incorrect username or password"
LOGIN_FORM_MARKER = 'class="login_form"'

DEFAULT_DISTRICT = 4  # Bay Area

# Datasets confirmed available for district 4. `meta` carries station
# direction and lat/lon; `station_5min` carries the speeds.
DATASET_META = "meta"
DATASET_5MIN = "station_5min"

REQUEST_TIMEOUT = 300  # PeMS HTML pages routinely take 20-30s
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
INTER_REQUEST_DELAY_SECONDS = 1.0  # shared public resource; do not hammer it

logger = logging.getLogger(__name__)


class PemsAuthError(RuntimeError):
    """Raised when PeMS rejects the supplied credentials."""


@dataclass(frozen=True)
class ClearinghouseFile:
    """One downloadable artifact listed in the Data Clearinghouse."""

    file_id: str
    filename: str
    size_bytes: int
    month: str

    @property
    def url(self):
        return f"{BASE_URL}?download={self.file_id}&dnode=Clearinghouse"


class PemsClient:
    """Authenticated PeMS session with Data Clearinghouse access."""

    def __init__(self, username=None, password=None):
        self.username = username or os.environ.get("PEMS_USERNAME")
        self.password = password or os.environ.get("PEMS_PASSWORD")
        if not self.username or not self.password:
            raise PemsAuthError(
                "Set PEMS_USERNAME and PEMS_PASSWORD in the environment."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self._authenticated = False

    def login(self):
        """Establish a PHPSESSID session. Idempotent."""
        if self._authenticated:
            return

        logger.info("Authenticating to PeMS as %s", self.username)
        resp = self.session.post(
            BASE_URL,
            data={
                "redirect": "",
                "username": self.username,
                "password": self.password,
                "login": "Login",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        if LOGIN_FAILED_MARKER in resp.text:
            raise PemsAuthError("PeMS rejected the credentials.")
        if LOGIN_FORM_MARKER in resp.text:
            raise PemsAuthError(
                "Still on the login page after POST; the form contract changed."
            )
        if "PHPSESSID" not in self.session.cookies:
            raise PemsAuthError("No PHPSESSID issued; cannot hold a session.")

        self._authenticated = True
        logger.info("PeMS authentication successful.")

    def _get(self, params, stream=False):
        """GET with retry/backoff, keeping the session cookie attached."""
        self.login()
        last_error = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                resp = self.session.get(
                    BASE_URL, params=params,
                    timeout=REQUEST_TIMEOUT, stream=stream,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_error = e
                logger.warning("PeMS request failed (attempt %d/%d): %s",
                               attempt, RETRY_ATTEMPTS, e)
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"PeMS request failed after retries: {last_error}")

    def list_files(self, year, dataset=DATASET_5MIN, district=DEFAULT_DISTRICT):
        """
        Return ClearinghouseFile entries for a district/year/dataset,
        sorted by filename (which sorts chronologically).
        """
        resp = self._get({
            "srq": "clearinghouse",
            "district_id": district,
            "geotag": "",
            "yy": year,
            "type": dataset,
            "returnformat": "text",
        })
        time.sleep(INTER_REQUEST_DELAY_SECONDS)

        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Clearinghouse did not return JSON (session expired?): {e}"
            ) from e

        files = [
            ClearinghouseFile(
                file_id=entry["file_id"],
                filename=entry["file_name"],
                size_bytes=int(entry["bytes"].replace(",", "")),
                month=month,
            )
            for month, entries in payload.get("data", {}).items()
            for entry in entries
        ]
        files.sort(key=lambda f: f.filename)
        logger.info("Found %d %s files for district %s in %s.",
                    len(files), dataset, district, year)
        return files

    def download(self, entry, dest_dir="."):
        """Stream one clearinghouse file to disk. Skips if already present."""
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, entry.filename)
        if os.path.exists(dest) and os.path.getsize(dest) == entry.size_bytes:
            logger.info("Already have %s, skipping.", entry.filename)
            return dest

        self.login()
        logger.info("Downloading %s (%.1f MB)",
                    entry.filename, entry.size_bytes / 1e6)
        resp = self._get(
            {"download": entry.file_id, "dnode": "Clearinghouse"}, stream=True
        )
        # A session-expiry page would arrive as HTML; refuse to save it as data.
        if "text/html" in resp.headers.get("Content-Type", ""):
            raise RuntimeError(
                f"Got HTML instead of data for {entry.filename}; session expired."
            )
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

        logger.info("Saved %s (%d bytes)", dest, os.path.getsize(dest))
        time.sleep(INTER_REQUEST_DELAY_SECONDS)
        return dest


# station_5min is headerless CSV with positional columns. The first 12 are
# fixed; per-lane triples repeat afterwards for as many lanes as the station
# has, which is why we only ever read the first 12.
STATION_5MIN_COLUMNS = [
    "timestamp", "station", "district", "freeway", "direction",
    "lane_type", "station_length", "samples", "pct_observed",
    "total_flow", "avg_occupancy", "avg_speed",
]

# Only mainline and HOV detectors report speed at all; ramps (OR/FR/FF) are
# always null, so including them silently halves apparent coverage.
SPEED_BEARING_LANE_TYPES = ("ML", "HV")


def parse_station_5min(path, stations=None, mainline_only=True,
                       min_pct_observed=None):
    """
    Parse a station_5min .txt.gz into a tidy speed frame.

    Returns: ts, sensor_id, freeway, direction, lane_type, total_flow,
             avg_occupancy, avg_speed, pct_observed

    `min_pct_observed` filters out heavily imputed rows. PeMS backfills missing
    detector samples, and district 4 averages only ~50% directly observed, so
    set this when you need genuinely measured ground truth.
    """
    import pandas as pd

    df = pd.read_csv(path, header=None, usecols=range(len(STATION_5MIN_COLUMNS)),
                     names=STATION_5MIN_COLUMNS, compression="infer")
    df["ts"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M:%S")
    df["sensor_id"] = df["station"].astype(str)

    if mainline_only:
        df = df[df["lane_type"].isin(SPEED_BEARING_LANE_TYPES)]
    if stations is not None:
        df = df[df["sensor_id"].isin({str(s) for s in stations})]
    if min_pct_observed is not None:
        df = df[df["pct_observed"] >= min_pct_observed]

    return df[["ts", "sensor_id", "freeway", "direction", "lane_type",
               "total_flow", "avg_occupancy", "avg_speed", "pct_observed"]]


def parse_station_meta(path):
    """
    Parse a station metadata .txt (tab-separated, with header).

    Renames ID -> sensor_id so the frame drops straight into the existing
    mapping pipeline, which expects sensor_id / Latitude / Longitude / Dir.
    """
    import pandas as pd

    meta = pd.read_csv(path, sep="\t")
    return meta.rename(columns={"ID": "sensor_id"})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        stream=sys.stdout,
    )
    year = time.gmtime().tm_year
    client = PemsClient()
    for dataset in (DATASET_META, DATASET_5MIN):
        entries = client.list_files(year=year, dataset=dataset)
        for entry in entries[-3:]:
            print(f"  {dataset:<13} {entry.filename:<40} "
                  f"{entry.size_bytes:>12,}B  id={entry.file_id}")
