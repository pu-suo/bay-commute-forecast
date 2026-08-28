# Deploying

Two halves. Static files go to GitHub Pages; the routing API runs in one
container because it needs OSRM and the forecast table.

## What needs an account

Fly, and nothing else. No API keys are required: the basemap (Esri), the
geocoder (Photon) and the routing engine (OSRM, in the container) are all
keyless, and the PeMS credentials stay on the machine that collects data.

`GOOGLE_MAPS_API_KEY` is optional. Set it for better place search; leave it
unset and Photon answers.

## One-time

```bash
brew install flyctl
fly auth signup                 # or: fly auth login

cd deploy
fly launch --no-deploy --copy-config --name bay-commute-forecast

# Deploy from the REPO ROOT, not from deploy/. The Dockerfile copies forecast/
# and site/, which live at the root, so the build context has to be the root.
# Running `fly deploy` inside deploy/ fails on COPY forecast/.
cd ..
fly deploy --config deploy/fly.toml --dockerfile deploy/Dockerfile
```

The machine stops when idle and starts on the first request, so it costs a few
dollars a month rather than about twelve. The landing page never touches it, so
the wake is only paid by someone clicking "Forecast this drive" on a cold app.
To keep it always warm instead, set `min_machines_running = 1` and
`auto_stop_machines = false`.

The first deploy takes 15-20 minutes because it downloads a Geofabrik extract
and prepares the OSRM graph. Later ones reuse that layer.

`fly.toml` and `site/config.json` already carry the right hostnames. If the app
name is taken, pick another and update `api_base` in `site/config.json` and
`ALLOW_ORIGIN`/`SERVE_URL` in `fly.toml` to match, then re-run
`bash deploy/publish.sh`.

## Every night

`scripts/nightly.sh` rebuilds the forecast and runs `publish.sh`. The container
pulls `forecast.parquet` from the published site once an hour, so nothing needs
inbound access to it and a restart picks up whatever is current.

## Notes

The image builds the OSRM graph at build time from a Geofabrik extract, so the
first `fly deploy` takes 15-20 minutes and later ones are fast unless the
`PBF_URL` or `BBOX` build args change.

Memory is 2 GB because OSRM memory-maps the graph. 1 GB is not enough.

`/api/health` returns 503 once the forecast table is more than 36 hours old, so
a stalled pipeline shows up as an unhealthy machine rather than as a site
quietly serving last week's numbers.
