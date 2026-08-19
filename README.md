# Bay Commute Forecast

Day-ahead travel-time forecasts for the Bay Area freeway network, any origin to
any destination, with the accuracy published alongside them.

This is not a real-time traffic map; Google Maps does that better than anyone
could. It answers a different question: what will your drive look like tomorrow,
or next Thursday, and why? Forecasts are in minutes, because that is the unit
people plan in.

![the map](docs/screenshot.png)

## What it does

Search for a place by name, or click the map. OSRM finds the path, the path is
matched onto the 2,291 PeMS detectors it traverses, and each stretch is priced
from that detector's forecast. The clock advances as the route is walked, so a
segment forty minutes into a trip is priced at the forecast for the time you
reach it.

Arrive-by is solved by search rather than subtraction, since travel time depends
on departure time and leaving 30 minutes earlier can save 45.

The week view shows seven days by 24 hours per corridor. Side streets with no
detector are drawn as thin lines coloured by the spread model and can be
switched off, so what the inference contributes is visible rather than asserted.

## Status

| Component | State |
|---|---|
| Traffic ingestion, corridors | done, 2,050 days, 2021-2026 |
| Traffic ingestion, all detectors | done, 2,052 days x 2,291 detectors, 1.4 GB |
| Event ingestion | done, 1,329 events, 7 venues |
| Weather ingestion | done, 1.22M archived-forecast rows on a 30-cell grid |
| Roadwork ingestion | running daily, accumulating |
| Seasonal baseline | done, 3.96M (detector, weekday, slot) cells, 36 s build |
| Network model | trained, spatial and temporal holdout |
| Routing | self-hosted OSRM, direction-aware detector matching |
| Serving | nightly batch to a 1.3M-row forecast table |
| Site | map, week view, route planner, accuracy page |
| Deployment | not started, still local |

## How it works

```
PeMS station_5min ---> detector Parquet ---> seasonal profile ----+
   (2,291 detectors,     (one file/day,       (detector x dow     |
    5-min, 2021-)         740 KB zstd)         x slot)            |
                                                                  v
Open-Meteo archived forecast ----------------------------->   LightGBM  ---> nightly
   (30-cell grid, hourly)                                      (speed)       forecast
                                                                  ^          table
Venue calendars + league APIs ------------------------------------+         (1.3M rows)
   (7 venues, distance to nearest)                                              |
                                                                                v
                    OSRM ---> direction-aware detector matching --------->    route
                 (self-hosted)     + surface spread model                    minutes
```

### The model predicts detector speed, not corridor time

The first version forecast nine hand-defined corridors with `corridor` as a
categorical feature. That does not scale: a category the model has never seen is
useless, and there is no affordable way to train on 2,291 of them.

So the unit is a detector, described by its attributes: freeway, direction, lane
count, position, segment length, and its own seasonal profile. Nothing
identifies it. That permits training on a sample of the network and serving all
of it, and the assumption is checked by withholding 30% of detectors from
training and scoring them separately.

### The seasonal profile is a feature, not a rival

Mean speed for a given (detector, weekday, slot) is already a strong day-ahead
forecast. A model asked to predict speed from scratch would spend most of its
capacity rediscovering that structure, so it is handed over as a feature and the
model works on the deviation.

Two profiles are maintained:

| File | Fitted on | Used by |
|---|---|---|
| `_seasonal_trainonly.parquet` | days before the evaluation split | training, validation |
| `_seasonal.parquet` | everything up to today | the nightly serving job |

Keeping them separate stops a nightly refit invalidating a published accuracy
number.

### The map draws two kinds of claim

Road geometry comes from the OSM extract, not from the detectors. The first
version drew straight lines between consecutive detectors, which arrived as
chains of chords across every curve and stopped wherever a detector had no
forecast, leaving holes mid-freeway.

| | drawn as | coloured by | scored |
|---|---|---|---|
| freeway with a detector | thick, opaque | that detector's forecast | yes |
| freeway with none within 4 mi | thin, grey | nothing | n/a |
| side street | thin, translucent, switchable | the spread model | no |

10,131 freeway chunks of about 0.4 mi, 87% with a detector, plus 49,389 primary
and secondary roads. Speeds are not baked into the geometry: the browser holds a
speed series per detector, so a chunk ships one detector id instead of 168
numbers and the scrubber stays responsive.

The map stops where the data does. The OSM extract reaches into the Central
Valley, where district 4 has no detectors; I-5 and CA-99 alone were a fifth of
every chunk and none could be coloured.

### Direction of travel is not the direction on the shield

Matching a route to detectors tests heading as well as proximity, to avoid
matching the opposing carriageway. The first version compared against the
compass reading of the signed direction, but Caltrans signs I-580 east/west
while it runs north/south through Oakland, and I-80 east while it runs almost
due north at Vallejo.

19.4% of detectors sit more than 70 degrees from their signed direction, worst
on the freeways a Bay Area commute uses. Each detector's bearing now comes from
its neighbours along its own freeway, flipped for southbound and westbound
because postmiles count the other way.

| route | detectors matched, signed | using true bearing |
|---|---:|---:|
| I-80, Berkeley to Vallejo | 28 | 52 |
| US-101, Palo Alto to SFO | 31 | 39 |
| I-580, Livermore to Oakland | 46 | 50 |
| I-880, San Jose to Oakland | 77 | 76 |

Those journeys were being handed to the surface estimator because the detector
standing on the road was rejected for pointing the wrong way.

### A route is spans, priced by their evidence

| Span | Evidence | Scored |
|---|---|---|
| freeway | a PeMS detector within 530 ft, heading-aligned | yes |
| surface | none; OSRM free-flow scaled by the spread model | no |

Segment length is the spacing between consecutive detectors, not each
detector's `Length` attribute. Length is the stretch a detector nominally
represents (about 0.34 mi) and it does not tile the road (about 0.58 mi
spacing), so summing it understates freeway time by roughly 40%.

The surface spread model estimates local-street speed from nearby freeway
conditions:

```
local_speed    = free_flow * (1 - ALPHA * (1 - freeway_ratio))
freeway_ratio  = inverse-distance-squared weighted mean of (forecast / free-flow)
                 over up to 6 mainline detectors within 2 mi
ALPHA          = 0.5     stated prior, not fitted
```

There is no public source of Bay Area arterial speeds, so ALPHA cannot be fitted
and the model cannot be scored. It is a stated assumption, reported separately,
and excluded from every accuracy figure. The sign is not even guaranteed:
freeway congestion can divert traffic onto parallel arterials, slowing them, or
hold it on the freeway, freeing them.

## Accuracy

Everything below is measured on data the model never trained on. The split is
temporal, training ends before validation starts and validation ends before the
test window, and 30% of detectors are withheld from training entirely.

### Nine named commutes, in minutes

Travel time composed from detector-level forecasts over 268,100 held-out
15-minute intervals across 13 months. The baseline is the seasonal profile: same
detector, weekday and time of day averaged over history. At a day-ahead horizon
that is a strong opponent, because current traffic carries almost no signal and
calendar effects carry nearly all of it.

```
weighted MAE      1.28 min   vs 1.46 min baseline    +12.1%
peak MAE          2.59 min   vs 3.12 min baseline    +17.2%
```

| Corridor | mi | detector cov. | mean trip | baseline | model | gain | peak gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bay Bridge WB, Berkeley to SF | 7.4 | 60% | 8.7 | 1.227 | 1.091 | +11.1% | +15.0% |
| Bay Bridge EB, SF to Berkeley | 7.8 | 49% | 8.9 | 1.112 | 1.158 | -4.1% | -7.3% |
| US-101 NB, San Jose to SFO | 32.3 | 57% | 31.7 | 1.618 | 1.412 | +12.7% | +20.5% |
| US-101 SB, SFO to San Jose | 31.1 | 59% | 31.7 | 1.825 | 1.534 | +15.9% | +22.4% |
| I-880 NB, San Jose to Oakland | 38.5 | 66% | 39.8 | 2.304 | 1.942 | +15.7% | +21.5% |
| I-880 SB, Oakland to San Jose | 36.7 | 70% | 38.5 | 2.837 | 2.469 | +13.0% | +19.1% |
| I-580 WB, Altamont to Dublin | 14.0 | 86% | 13.1 | 0.449 | 0.430 | +4.2% | +6.1% |
| I-580 EB, Dublin to Altamont | 14.6 | 83% | 14.8 | 0.933 | 0.822 | +11.9% | +18.0% |
| SR-237 EB, Sunnyvale to Milpitas | 4.5 | 94% | 5.1 | 0.766 | 0.615 | +19.7% | +23.5% |

The Bay Bridge eastbound is worse than the baseline and is left in the table. It
has the lowest detector coverage of the nine at 49%, and a bottleneck the
sensors cannot see: the toll plaza and its metering lights sit upstream of the
instrumented stretch. Dropping the corridor would make the other eight read as a
general result rather than a conditional one.

### Detector level, in mph

The number to look at is the gap between the two right-hand columns. The network
has 2,291 detectors and the model trains on a sample; if skill came from
memorising individual detectors, the unseen column would collapse.

| Slice | n (seen) | baseline | seen | gain | unseen | gain |
|---|---:|---:|---:|---:|---:|---:|
| all | 1,685,219 | 3.048 | 2.880 | +5.5% | 2.700 | +6.1% |
| peak | 490,440 | 4.625 | 4.298 | +7.1% | 4.147 | +6.3% |
| congested (<45 mph typical) | 46,721 | 11.594 | 9.655 | +16.7% | 11.859 | +9.8% |
| peak and congested | 41,091 | 11.504 | 9.543 | +17.0% | 11.915 | +11.0% |
| holiday | 152,721 | 3.533 | 2.942 | +16.7% | 2.778 | +18.2% |
| rain | 39,182 | 3.967 | 3.524 | +11.2% | 3.158 | +9.9% |
| near an event | 34,370 | 3.889 | 3.650 | +6.1% | 3.667 | +6.9% |
| worst 5% of intervals | 84,261 | 20.308 | 19.467 | +4.1% | 19.110 | +4.6% |

There is no holdout penalty. Detectors the model has never seen gain 6.1%
against 5.5% for detectors it trained on, so describing a detector by its
attributes and seasonal profile rather than its identity works.

The gains land where the baseline is weakest: holidays at +17 to +18%, and
congested peak intervals at +11 to +17%. On ordinary free-flowing intervals the
model adds almost nothing, which is correct, because an average is already right
there.

### Feature importance

```
seasonal_speed   69.2%
seasonal_sd      10.0%
tod               5.2%
month             2.6%
freeway           2.2%
lanes             1.5%
event_miles       1.3%   distance to the nearest venue, learned not hard-coded
holiday_class     1.2%
lat / lon         2.4%
precip_3h         0.8%   accumulated rain beats instantaneous rain
```

Events reach the model as distance to the nearest venue plus hours since its
start, rather than a hand-drawn list of which corridors each venue may affect.
The decay with distance is left to the data.

### Selecting the model

Two candidates were trained: A on 275 detectors (9.5M readings) and B on 620
detectors (22.4M readings). On the training objective B is better, with
validation MAE 2.493 mph against A's 2.527.

Scored on the product metric over the same validation window, the ranking flips:

| Candidate | detectors | rows | valid MAE (mph) | valid route MAE (min) | peak (min) |
|---|---:|---:|---:|---:|---:|
| A | 275 | 9.5M | 2.527 | 1.283 | 2.441 |
| B | 620 | 22.4M | 2.493 | 1.291 | 2.489 |

Corridor time is a sum of `length / speed`, so error on a slow detector costs
far more minutes than the same error on a fast one, and uniform mph MAE does not
know that. Selecting on the training loss would have shipped the worse product.

A is shipped, selected on route MAE over the validation window. The test window
was scored once, afterwards. B's metrics are kept in `models/network_b/`.

## Data sources

| Signal | Source | Auth | History | Forward |
|---|---|---|---|---|
| Traffic speed | Caltrans PeMS `station_5min`, district 4 | account | 2010 to yesterday | n/a |
| Detector metadata | PeMS `meta` | account | snapshots | n/a |
| Road geometry, routing | OpenStreetMap, self-hosted OSRM | none | n/a | n/a |
| Weather (training) | Open-Meteo Historical Forecast | none | 2022 on | n/a |
| Weather (serving) | Open-Meteo Forecast | none | n/a | 16 days |
| Place search | Google Places, or Photon without a key | optional | n/a | n/a |
| Events | 7 venue calendars, Wayback | none | 2021 on | live |
| Baseball, hockey | MLB StatsAPI, api-web.nhle.com | none | years | full seasons |
| Roadwork | Caltrans LCS | none | none | ~10-day lead |
| Holidays | `holidays` package | none | yes | yes |

### Why these

PeMS over HERE or Google. PeMS is public domain under California's use policy,
so derived datasets can be republished. HERE caps retention at 30 days and
restricts use to enabling an end user's use of their service. That difference
decides whether this project can exist in public.

Open-Meteo's Historical Forecast API, not ERA5 observations. Training on what
actually happened gives the model weather knowledge it will never have at serve
time; training on the forecast that was issued keeps train and serve inputs the
same kind of thing. This is the easiest decision in the project to get wrong.

A weather grid, not corridor midpoints. The first pull sampled nine corridor
midpoints, which collapsed to five locations. The network spans Gilroy to Ukiah,
130 miles, so nearest-point assignment would hand a Santa Rosa detector the
weather in Oakland. A 0.25 degree grid over the bounding box has 90 cells and
only 30 contain a detector, so covering the network costs a third of what its
extent suggests.

Place search degrades rather than breaks. Google Places needs a billed key, so
the provider is a choice: set `GOOGLE_MAPS_API_KEY` and it is used, leave it
unset and Photon answers instead. Either way the call is proxied server-side,
since a browser-side Places call puts the key in page source. A keyed setup
falls back to Photon on a quota error.

Venue calendars over an events vendor. One venue page lists every event type at
that venue; the SAP Center calendar carries "Sharks vs. Bruins", "Bellator" and
"Disney On Ice" together. Ticketmaster's Discovery API would be cleaner but
self-serve registration is closed and their site is JS-rendered with no event
data in the HTML.

League APIs where they exist. MLB and NHL publish free schedule APIs with exact
start times and complete seasons. Merging them added 572 games the Wayback crawl
never saw, because archived calendar pages only show a dozen upcoming events and
capture a fraction of an 81-game home season.

### Known gaps

- NFL kickoff times. No free source. ESPN 403s, pro-football-reference 403s,
  TheSportsDB 503s, Wikipedia's schedule table has no time column, and Levi's
  own event pages omit them. Needs a hand-maintained table, about 10 rows a
  season.
- Chase Center's live calendar is a JS SPA. Wayback recovered 2022-2024, when
  the site was server-rendered; nothing since.
- School calendars are not collected. Likely a real AM-peak driver.
- Roadwork is collected but unused. Caltrans publishes current state only, so
  the archive began the day collection started and is too short to train on.

## What the data says

Findings that shaped the design, all measured rather than assumed.

Only about 20% of forecast error sits in identifiable contexts: holidays 14.8%,
weather 3.0%, events 2.8%. The other 80% is ordinary day-to-day variation and
may be irreducible. This is why the target was never "beat the baseline by 50%";
that number does not exist in this data.

Holidays are the biggest available win. Twice the normal error, 1.4x on adjacent
travel days, and invisible to a day-of-week baseline. The model recovers 17-18%
of error on the intervals they touch.

Events are more a UX feature than an accuracy feature. A 49ers home game adds
+12 to +20 minutes on SR-237 East, which is measured, large and real, but five
in-season game Sundays by about 36 affected intervals is a rounding error in a
global metric. Worth building for the explanation line ("Sunday 5pm on 237 will
run 13 min instead of 9, 49ers home game"), not for MAE.

Venue effects do not generalise. Same metro, same method:

| Venue | Capacity | Egress | Effect |
|---|---|---|---|
| Levi's Stadium | 68,500 | ~17:00 | +20 min (+82%) |
| Shoreline Amphitheatre | 22,500 | ~23:00 | +0.32 min |
| SAP Center | 17,500 | ~22:00 | +0.08 min |

It takes a very large crowd leaving at once into a network that is already
loaded. Late egress into an empty freeway does nothing. This is why the model
gets distance-to-venue and finds the decay itself.

Rain matters and the model handles it, at +10 to +11% on rain intervals, where
an earlier multiplicative lookup factor made them worse. The effect depends on
intensity, duration, and probably whether it is the season's first storm;
`precip_3h` outranks instantaneous precipitation in the model.

Only 2.6% of Bay Area hours have any rain, so the ceiling here is low regardless
of method.

## Data quality traps

Each of these corrupts a model quietly rather than failing loudly. All were hit
during development.

PeMS imputes whole days. `pct_observed` averages about 51% in district 4 and
some days are 100% modelled. An imputed day looks normal in every other way and
is only detectable via that column. An imputed day covering a real event teaches
the model the event did nothing.

`pct_observed` is bimodal, not a gradient. About a quarter of detector-intervals
sit at exactly 0 and the median is 100, so thresholding it removes whole segments
from a corridor sum and corrupts the number it was meant to protect. Coverage is
reported; filtering happens at training time.

Partial coverage reads as a faster trip. Travel time is a sum over detectors, so
missing sensors produce a lower number, which looks like free-flowing traffic
rather than missing data. Every travel time is scaled to full length.

Coverage thresholds delete corridors rather than intervals. Coverage is
near-constant within a corridor, so an absolute 0.80 cut removed six of nine
corridors from the accuracy report, including the Bay Bridge, which runs at 0.50
every hour of every day. The threshold is relative to each corridor's own median.

Detector `Length` does not tile the road. It averages 0.34 mi while detectors sit
0.58 mi apart, so summing it understates freeway time by about 40%. Routing uses
spacing between consecutive matched detectors.

Proximity matches both carriageways. Opposing directions sit about 200 ft apart,
so nearest-detector matching claimed 39 northbound and 41 southbound detectors on
one route, 134% of its own length as instrumented.

A heading test against the signed direction rejects a fifth of all detectors.
Freeway direction letters describe the route, not the compass; covered above.

Parquet round-trips timestamps as microseconds while `Timestamp.value` is
nanoseconds. A hand-rolled `// 10**9` on both sides keyed the forecast table and
its lookups a thousandfold apart, and every route fell back to the surface model,
producing a plausible number rather than an error.

A left join promotes an integer column to float. One unmatched detector turned
`freeway` into "101.0" against a model that learned "101", so the feature was
void at serve time and only at serve time.

Venues disagree about time zones. Stanford publishes UTC (02:30:00Z), Shoreline
publishes -07:00. Unnormalised, every Friday night game lands on Saturday.

Missing weather is not dry weather. The archive starts in 2022; filling nulls
with zero asserted that 2021 was permanently dry, 15% of the corpus.

Team schedule pages list away games. Stanford's calendar advertises "Stanford at
California", which happens in Berkeley. Verified against schema.org `location`.

Cancelled events are not small events. Venues keep the listing and edit the
title. 23 were being fed to the model as ordinary sell-outs that moved no traffic.

A `*.jsonl` glob matches the pipeline's own output. The event normaliser was
re-ingesting yesterday's cleaned file and compounding its dedupe every run.

`groupby.apply(lambda)` over 660k groups is 880x slower than an aggregated
column, and accumulating the result across 1,461 days as pandas Series never
finished. Dense numpy accumulation over the bounded key space took the seasonal
build from 8.7 hours to 36 seconds.

macOS `cron` skips jobs while the machine sleeps and never catches up. Two days
of roadwork history, which Caltrans does not publish retrospectively, were lost
before this was noticed. Everything scheduled now runs under `launchd`.

## Layout

```
collector/
  pems_client.py        PeMS auth and Data Clearinghouse JSON endpoint
  lcs_snapshot.py       Caltrans roadwork, daily snapshot
  venue_events.py       venue calendars, live and Wayback replay
  sports_api.py         MLB StatsAPI, NHL API
  weather.py            Open-Meteo: archived forecast, live forecast, grid mode
  events_normalize.py   dedupe, junk and cancellation filter, time recovery
  merge_league_times.py fill event times from league APIs
forecast/
  corridors.py          corridor and venue registry, travel-time primitive
  backfill.py           corridor-level stream-and-discard
  backfill_stations.py  network-level stream-and-discard
  build_seasonal.py     the (detector, weekday, slot) profile, both variants
  features_stations.py  network-scale feature assembly
  train_stations.py     LightGBM plus the spatial and temporal holdout report
  validate_routes.py    scores the model in minutes, on real corridors
  predict_network.py    nightly batch over the whole horizon
  matching.py           route polyline to detectors, direction-aware
  surface.py            the spread model for local streets
  route.py              origin and destination to minutes
site/
  build_data.py         forecast table to per-detector speed series, corridors
  build_geometry.py     OSM extract to road centrelines keyed to detectors
  site_geocode.py       place search: Google Places when keyed, Photon when not
  server.py             stdlib server for /api/route, /api/profile, /api/geocode
  index.html app.js     map, week grid, route planner
  accuracy.html         the scores, from the JSON the model wrote
scripts/nightly.sh      one full run, invoked by launchd at 03:20
```

### Storage

Files, no database. About 1.5 GB, nearly all of it the detector archive.

| Data | Format | Size |
|---|---|---|
| Traffic, all detectors | Parquet, one file per day | 1.4 GB |
| Traffic, nine corridors | Parquet, one file per day | 64 MB |
| Seasonal profiles | Parquet | 31 MB each |
| Weather | Parquet | 7.4 MB |
| Events | JSONL | 1.1 MB |
| Roadwork | gzipped JSON per day | 200 KB |
| Serving forecast | Parquet | 12 MB |
| Road geometry (site) | JSON | 7.8 MB, 1.6 MB gzipped |

Postgres would be overhead for append-only daily writes with one reader. DuckDB
reads the same files if it outgrows pandas.

PeMS ships about 29 MB per district-day with no server-side detector filter, so
every day requires downloading the whole file. The backfill fetches a day,
reduces it, writes the result and deletes the raw file, so peak disk is one file
instead of 156 GB.

## Running it

```bash
pip install -r requirements.txt

# PeMS credentials, never committed
cat > ~/.pems_env <<'EOF'
export PEMS_USERNAME='you@example.com'
export PEMS_PASSWORD='...'
EOF
chmod 600 ~/.pems_env
set -a; . ~/.pems_env; set +a

# one-time: the routing graph, needs Docker and a Bay Area OSM extract
make osrm

make seasonal      # both profiles, about 40 s each
make train         # LightGBM on 9.5M detector readings, about 2 min on CPU
make validate      # score it in minutes on the nine corridors
make serve-data    # predict the horizon, rebuild the site JSON
make geometry      # road centrelines from the OSM extract, 5 s
make site          # http://localhost:8000

# optional: better place search
export GOOGLE_MAPS_API_KEY='...'   # falls back to Photon when unset
```

Nightly, under `launchd` rather than `cron`, since cron skips jobs while the
machine sleeps:

```
~/Library/LaunchAgents/com.bayforecast.lcs.plist       03:05  roadwork
~/Library/LaunchAgents/com.bayforecast.nightly.plist   03:20  everything else
```

Roadwork gets its own agent because Caltrans publishes current state only, so a
missed day can never be backfilled and it must not sit downstream of anything
that can fail.

## Cost

Training is about 2 minutes on CPU, with no GPU and no cloud. Inference is a
nightly batch, so there is no hosted model. All data is free; PeMS needs an
account, Open-Meteo and OSM need nothing.

Hosting is not settled. The static half (map, week view, accuracy page) is
plain JSON and costs nothing on any static host. Routing needs a process: OSRM's
Bay Area graph is 880 MB and the route endpoint needs the forecast table. The
plan is static assets on GitHub Pages plus one small container for `/api/*`,
which is a few dollars a month rather than zero.

## Attribution

Traffic and roadwork data courtesy of the California Department of
Transportation (Caltrans) Performance Measurement System. Weather from
[Open-Meteo](https://open-meteo.com/). Map data (c) OpenStreetMap contributors,
tiles (c) CARTO. Routing by [OSRM](https://project-osrm.org/). None of them
endorse this project.
