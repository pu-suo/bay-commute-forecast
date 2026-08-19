# forecast/corridors.py
"""
Corridor and venue definitions, and the travel-time primitive.

A corridor is an ordered run of mainline detectors along one freeway in one
direction between two named endpoints. Travel time is the sum of
segment_length / speed over those detectors.

Postmile bounds were derived by resolving landmarks to their nearest mainline
detector, then checked for coverage: no corridor has a gap wider than ~2 miles.

Note that this module uses each detector's `Length` attribute. That understates
absolute travel time by roughly 40%, because Length averages 0.34 mi while
detectors sit 0.58 mi apart. The figures stay internally consistent, since
actual and baseline share the convention. Routing uses spacing instead; see
forecast.route.
"""
from dataclasses import dataclass, field

# Only mainline and HOV detectors report speed at all. Ramps (OR/FR/FF) are
# always null, so including them halves apparent coverage for no benefit.
SPEED_LANE_TYPES = ("ML", "HV")

# PeMS marks imputed samples via pct_observed. District 4 averages only ~51%
# directly observed, and fully-imputed days are indistinguishable from real
# ones except by this column, and they corrupt training if left in.
DEFAULT_MIN_PCT_OBSERVED = 20.0
IMPUTED_DAY_THRESHOLD = 10.0


@dataclass(frozen=True)
class Corridor:
    slug: str
    name: str
    freeway: int
    direction: str
    pm_min: float
    pm_max: float

    def stations(self, meta):
        """Ordered station metadata for this corridor, south/west to north/east."""
        sel = meta[
            (meta["Fwy"] == self.freeway)
            & (meta["Dir"] == self.direction)
            & (meta["Type"] == "ML")
            & (meta["Abs_PM"] >= self.pm_min)
            & (meta["Abs_PM"] <= self.pm_max)
        ]
        return sel.dropna(subset=["Abs_PM", "Length"]).sort_values("Abs_PM")


CORRIDORS = [
    Corridor("bay-bridge-wb", "Bay Bridge WB: Berkeley to SF", 80, "W", 4.5, 12.0),
    Corridor("bay-bridge-eb", "Bay Bridge EB: SF to Berkeley", 80, "E", 4.5, 12.0),
    Corridor("101-nb-peninsula", "US-101 NB: San Jose to SFO", 101, "N", 389.2, 421.6),
    Corridor("101-sb-peninsula", "US-101 SB: SFO to San Jose", 101, "S", 389.2, 421.6),
    Corridor("880-nb", "I-880 NB: San Jose to Oakland", 880, "N", 4.5, 41.6),
    Corridor("880-sb", "I-880 SB: Oakland to San Jose", 880, "S", 4.5, 41.6),
    Corridor("580-wb-altamont", "I-580 WB: Altamont to Dublin", 580, "W", 17.8, 36.5),
    Corridor("580-eb-altamont", "I-580 EB: Dublin to Altamont", 580, "E", 17.8, 36.5),
    Corridor("237-eb", "SR-237 EB: Sunnyvale to Milpitas", 237, "E", 0.0, 12.0),
]

BY_SLUG = {c.slug: c for c in CORRIDORS}


@dataclass(frozen=True)
class Venue:
    slug: str
    name: str
    lat: float
    lon: float
    capacity: int
    calendar_url: str
    # Corridors whose sensors fall within 3 miles. Empty means the venue cannot
    # influence any modelled corridor and contributes nothing until one is added.
    corridors: tuple = field(default_factory=tuple)


# Every venue large enough to plausibly move a freeway is included. Which of
# them *actually* do is a coefficient the model estimates, not a filter applied
# here: coarse before/after tests at n<10 detect a Levi's-sized effect and would
# miss anything conditional on sellout, weather, or peak overlap.
VENUES = [
    Venue("levis-stadium", "Levi's Stadium", 37.4030, -121.9698, 68500,
          "https://www.levisstadium.com/events/",
          ("237-eb", "101-nb-peninsula", "101-sb-peninsula")),
    Venue("oakland-arena", "Oakland Arena / Coliseum", 37.7516, -122.2005, 46800,
          "https://www.theoaklandarena.com/events",
          ("880-nb", "880-sb")),
    Venue("stanford-stadium", "Stanford Stadium", 37.4347, -122.1611, 50000,
          "https://gostanford.com/sports/football/schedule",
          ("101-sb-peninsula", "101-nb-peninsula")),
    Venue("shoreline-amph", "Shoreline Amphitheatre", 37.4267, -122.0806, 22500,
          "https://www.shorelineamphitheatre.com/shows",
          ("101-nb-peninsula", "101-sb-peninsula")),
    Venue("oracle-park", "Oracle Park", 37.7786, -122.3893, 41900,
          "https://www.mlb.com/giants/schedule",
          ("bay-bridge-wb", "bay-bridge-eb")),
    Venue("sap-center", "SAP Center", 37.3327, -121.9010, 17500,
          "https://www.sapcenter.com/events",
          ("101-nb-peninsula", "880-nb")),
    # Mission Bay: no freeway detectors on any current corridor. Kept so its
    # calendar still accumulates; needs an SF corridor before it can contribute.
    Venue("chase-center", "Chase Center", 37.7680, -122.3877, 18000,
          "https://www.chasecenter.com/events", ()),
]

VENUES_BY_SLUG = {v.slug: v for v in VENUES}


def corridor_travel_time(readings, corridor, meta):
    """
    Collapse 5-minute station readings into corridor travel time in minutes.

    `readings` is a parsed station_5min frame (see pems_client.parse_station_5min)
    carrying ts, sensor_id, avg_speed and pct_observed.

    Travel time is **scaled to the full corridor length**. A raw sum over only
    the stations that happen to be reporting understates travel time whenever
    coverage is partial, and because it looks like a faster trip
    rather than like missing data, that error is invisible downstream. Scaling by
    total_miles / observed_miles keeps intervals comparable.

    Stations are never dropped for low pct_observed. It is bimodal rather than a
    smooth quality gradient (about a quarter of station-intervals sit at exactly
    0 and the median is 100), so thresholding it removes whole segments from the
    sum and corrupts the very number it was meant to protect. Quality is
    *reported* here and filtered at training time instead.

    Returns a frame indexed by timestamp with:
        minutes         travel time, scaled to full corridor length
        n_stations      stations reporting this interval
        coverage        observed_miles / total_miles (filter on this, not on
                        pct_observed, when you need complete intervals)
        pct_observed    mean, for discounting imputation-heavy intervals
    """
    import pandas as pd  # imported lazily so the module stays importable bare

    stations = corridor.stations(meta)
    if stations.empty:
        raise ValueError(f"Corridor {corridor.slug} matched no stations.")
    lengths = dict(zip(stations["sensor_id"].astype(str), stations["Length"]))
    total_miles = float(stations["Length"].sum())

    df = readings[readings["sensor_id"].astype(str).isin(lengths)].copy()
    df = df[df["avg_speed"].notna() & (df["avg_speed"] > 0)]
    if df.empty:
        return pd.DataFrame(
            columns=["minutes", "n_stations", "coverage", "pct_observed"]
        )

    seg_miles = df["sensor_id"].astype(str).map(lengths)
    df["seg_miles"] = seg_miles
    df["seg_minutes"] = seg_miles / df["avg_speed"] * 60.0

    out = df.groupby("ts").agg(
        raw_minutes=("seg_minutes", "sum"),
        observed_miles=("seg_miles", "sum"),
        n_stations=("sensor_id", "nunique"),
        pct_observed=("pct_observed", "mean"),
    )
    out["coverage"] = out["observed_miles"] / total_miles
    out["minutes"] = out["raw_minutes"] / out["coverage"].where(out["coverage"] > 0)
    return out[["minutes", "n_stations", "coverage", "pct_observed"]]


def is_imputed_day(readings, threshold=IMPUTED_DAY_THRESHOLD):
    """
    True when a whole day is PeMS-imputed rather than measured.

    These days look normal in every other way (full station counts, plausible
    speeds, 288 intervals) and are only detectable here. A fully-imputed day covering
    a real event will teach a model that the event had no effect, which is
    worse than having no data for it at all.
    """
    return float(readings["pct_observed"].mean()) < threshold
