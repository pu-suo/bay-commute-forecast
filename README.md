# Bay Commute Forecast

Day-ahead travel-time forecasts for nine Bay Area freeway corridors, with the
accuracy published alongside them.

Not a real-time traffic map — Google Maps already does that better than anyone
could. This answers a different question: **what will your commute look like
tomorrow, or next Thursday, and why?** Nothing else answers that, and it is the
question people actually plan around.

Forecasts are in **minutes**, not mph, because minutes are the unit people think in.

---

## Current status

The data pipeline is complete and validated. The forecaster works. Feature
selection, model training, and the UI are deliberately still open.

| Component | State |
|---|---|
| Traffic ingestion | ✅ 2,050 days, 2021–2026, zero failures |
| Event ingestion | ✅ 1,339 events, 7 venues |
| Weather ingestion | ✅ 364,608 archived-forecast rows |
| Roadwork ingestion | ✅ daily cron, accumulating |
| Baseline forecaster | ✅ MAE 1.09 min, rolling-origin validated |
| Feature engineering | ⬜ open |
| Model training | ⬜ open |
| Site / UI | ⬜ open |

---

## How well it works

Rolling-origin backtest — every prediction uses only data from strictly before
the day being predicted. 1.33M held-out rows across 19 months.

```
weighted MAE          1.09 min   (5.7% of mean travel time)
median error          0.30 min   (18 seconds)
p99 error            10.65 min
skill vs "same time last week"   +17% to +26%
```

Per corridor:

| Corridor | Mean trip | MAE | Peak MAE |
|---|---|---|---|
| Bay Bridge WB — Berkeley→SF | 9.5 | 1.08 | 1.87 |
| Bay Bridge EB — SF→Berkeley | 8.6 | 0.62 | 1.33 |
| US-101 NB — San Jose→SFO | 32.0 | 1.03 | 1.98 |
| US-101 SB — SFO→San Jose | 32.9 | 1.29 | 2.76 |
| I-880 NB — San Jose→Oakland | 38.5 | 1.61 | 2.96 |
| I-880 SB — Oakland→San Jose | 39.3 | 1.95 | 3.62 |
| I-580 WB — Altamont→Dublin | 15.5 | 0.53 | 0.80 |
| I-580 EB — Dublin→Altamont | 17.3 | 0.94 | 1.62 |
| SR-237 EB — Sunnyvale→Milpitas | 10.3 | 0.78 | 1.77 |

The baseline is a **seasonal median** — the median of the last 8 same-weekday,
same-time-of-day observations. Not a neural network. At a day-ahead horizon
current traffic carries almost no signal and calendar effects carry nearly all
of it, so this is the right architecture, and any model has to beat it to earn
its place.

---

## Data sources

| Signal | Source | Auth | History | Forward |
|---|---|---|---|---|
| Traffic speed | Caltrans PeMS `station_5min`, district 4 | account | 2010→yesterday | — |
| Corridor geometry | PeMS `meta` | account | snapshots | — |
| Weather (training) | Open-Meteo **Historical Forecast** | none | 2022→ | — |
| Weather (serving) | Open-Meteo Forecast | none | — | 16 days |
| Events | 7 venue calendars + Wayback | none | 2021→ | live |
| Baseball / hockey | MLB StatsAPI, api-web.nhle.com | none | years | full seasons |
| Roadwork | Caltrans LCS | none | ✗ none | ~10-day lead |
| Holidays | `holidays` package | none | ✓ | ✓ |

### Why these and not others

**PeMS over HERE or Google.** PeMS is public domain under California's use
policy, so derived datasets can be republished. HERE caps retention at 30 days
and restricts use to enabling an end user's use of their service. That
difference decides whether this project can exist in public.

**Open-Meteo's Historical *Forecast* API, not ERA5 observations.** Training on
what actually happened gives the model weather knowledge it will never have at
serve time. Training on the forecast that *was issued* keeps train and serve
inputs the same kind of thing. This is the single most important data decision
in the project and the easiest to get wrong.

**Venue calendars over an events vendor.** One venue page lists every event type
at that venue — the SAP Center calendar carries "Sharks vs. Bruins", "Bellator"
and "Disney On Ice" together. Ticketmaster's Discovery API would be cleaner but
self-serve registration is closed, and their website is JS-rendered with no
event data in the HTML.

**League APIs where they exist.** MLB and NHL publish free schedule APIs with
exact start times and complete seasons. Merging them added **572 games the
Wayback crawl never saw** — archived calendar pages only ever show a dozen
upcoming events, so they capture a fraction of an 81-game home season.

### Known gaps

- **NFL kickoff times.** No free source exists. ESPN 403s, pro-football-reference
  403s, TheSportsDB 503s, Wikipedia's schedule table has no time column, and
  Levi's own event pages omit them. Needs a hand-maintained table (~10 rows a
  season) or a Ticketmaster key.
- **Chase Center** live calendar is a JS SPA. Wayback recovered 2022–2024, when
  the site was server-rendered; nothing since.
- **School calendars** not yet collected. Likely a real AM-peak driver.

---

## What the data says

Findings that shaped the design, all measured rather than assumed.

**Only ~20% of forecast error sits in identifiable contexts.** Holidays 14.8%,
weather 3.0%, events 2.8%. The other 80% is ordinary day-to-day variation and
may be irreducible.

**Holidays are the biggest available win.** 2× normal error, 1.4× on adjacent
travel days, and a simple correction recovers **16.8%** of error on the days it
touches. Invisible to a day-of-week baseline.

**Events are a UX feature, not an accuracy feature.** A 49ers home game adds
**+12 to +20 minutes** on SR-237 East — measured, large, real. But five in-season
game Sundays × ~36 affected intervals is 180 rows out of 1.33 million, so it
cannot move a global metric. Worth building for the explanation line
("Sunday 5pm on 237 will run 13 min instead of 9 — 49ers home game"), not for MAE.

**Venue effects do not generalise.** Same metro, same method:

| Venue | Capacity | Egress | Effect |
|---|---|---|---|
| Levi's Stadium | 68,500 | ~17:00 | **+20 min (+82%)** |
| Shoreline Amphitheatre | 22,500 | ~23:00 | +0.32 min |
| SAP Center | 17,500 | ~22:00 | +0.08 min |

It takes a very large crowd leaving at once into a network that is *already*
loaded. Late egress into an empty freeway does nothing.

**Rain matters but resists simple correction.** Rain intervals have MAE 1.695 vs
1.093 — 55% harder to predict — yet a multiplicative lookup factor makes them
*worse*. Effect depends on intensity, duration, and probably whether it's the
season's first storm. This is where a real model should earn its keep.

**Only 2.0% of Bay Area hours have any rain**, so the ceiling here is low
regardless.

---

## Data quality traps

Each of these silently corrupts a model rather than failing loudly. All were hit
during development.

**PeMS imputes whole days.** `pct_observed` averages ~51% in district 4, and some
days are 100% modelled. An imputed day looks completely normal — full station
counts, plausible speeds, 288 intervals — and is only detectable via that column.
An imputed day covering a real event teaches the model the event did nothing.
9.5% of rows are dropped for this.

**`pct_observed` is bimodal, not a gradient.** About a quarter of station-intervals
sit at exactly 0 and the median is 100. Thresholding it removes whole segments
from a corridor sum and corrupts the number it was meant to protect. Coverage is
*reported*, and filtering happens at training time.

**Partial coverage reads as a faster trip.** Travel time is a sum over stations,
so missing sensors produce a *lower* number, which looks like free-flowing
traffic rather than missing data. All travel times are scaled to full corridor
length.

**Venues disagree about time zones.** Stanford publishes UTC (`02:30:00Z`),
Shoreline publishes local (`-07:00`). Unnormalised, every Friday night game lands
on Saturday morning.

**Missing weather is not dry weather.** The archive starts in 2022; filling nulls
with zero asserted that 2021 was permanently dry — 15% of the corpus.

**Team schedule pages list away games.** Stanford's calendar advertises
"Stanford at California", which happens in Berkeley. Verified against
schema.org `location`.

---

## Layout

```
collector/
  pems_client.py        PeMS auth + Data Clearinghouse (undocumented JSON endpoint)
  lcs_snapshot.py       Caltrans roadwork, daily snapshot (no history published)
  venue_events.py       venue calendars, live + Wayback replay
  sports_api.py         MLB StatsAPI, NHL API
  weather.py            Open-Meteo archived forecasts
  events_normalize.py   dedupe, junk filter, time recovery
  merge_league_times.py fill event times from league APIs
forecast/
  corridors.py          corridor + venue registry, travel-time primitive
  backfill.py           stream-and-discard: PeMS → corridor Parquet
  baseline.py           seasonal median + rolling-origin backtest
  enhanced.py           layered corrections + error attribution
site/                   (open)
```

### Storage

Files, no database. 69 MB total.

| Data | Format | Size |
|---|---|---|
| Traffic | Parquet, one file per day, `year=YYYY/` | 64 MB |
| Weather | single Parquet | 2.2 MB |
| Events | JSONL | 1.1 MB |
| Roadwork | gzipped JSON per day | 96 KB |

Postgres would be overhead: append-only daily writes, one reader, 69 MB.
DuckDB reads the same files if it ever outgrows pandas.

**Stream-and-discard.** PeMS ships ~29 MB per district-day with no server-side
station filter, so every corridor-day requires downloading the whole file. But
660k mainline rows reduce to 288 intervals × 9 corridors — **31 KB**. The backfill
fetches a day, reduces it, writes the result, deletes the raw file. Peak disk is
one file instead of 156 GB.

---

## Running it

```bash
pip install -r requirements.txt

# PeMS credentials — never committed
cat > ~/.pems_env <<'EOF'
export PEMS_USERNAME='you@example.com'
export PEMS_PASSWORD='...'
EOF
chmod 600 ~/.pems_env
set -a; . ~/.pems_env; set +a

python -m forecast.backfill --years 2026 2025 2024 2023 2022 2021 --out ~/traffic-data/corridors
python -m collector.weather --start 2022-01-01 --end 2026-08-15 --out ~/traffic-data/weather
python -m collector.venue_events --live    --out ~/traffic-data/events
python -m collector.venue_events --wayback --from 2022 --to 2026 --out ~/traffic-data/events
python -m collector.events_normalize --inputs ~/traffic-data/events/events_*.jsonl \
    --out ~/traffic-data/events/events_clean.jsonl
python -m collector.merge_league_times --events ~/traffic-data/events/events_clean.jsonl \
    --out ~/traffic-data/events/events_merged.jsonl

python -m forecast.baseline --data ~/traffic-data/corridors
python -m forecast.enhanced --data ~/traffic-data/corridors
```

Roadwork must run daily — Caltrans publishes only current state, so the archive
exists only from the day collection starts:

```
5 3 * * * cd /path/to/repo && python -m collector.lcs_snapshot --out ~/traffic-data/lcs
```

---

## Planned stack

- **Training** — LightGBM on tabular features. 2.7M rows × ~15 features trains in
  minutes on CPU. No GPU, no cloud.
- **Scheduling** — GitHub Actions cron (free, always on; laptop cron misses days
  when the machine sleeps).
- **Site** — static HTML regenerated nightly, served from GitHub Pages or
  Cloudflare Pages. The forecast is a nightly batch, so there is no API to host.
- **Running cost** — $0.

---

## Attribution

Traffic and roadwork data courtesy of the California Department of
Transportation (Caltrans) Performance Measurement System. Weather from
[Open-Meteo](https://open-meteo.com/). Neither endorses this project.
