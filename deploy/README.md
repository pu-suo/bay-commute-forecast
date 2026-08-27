# Deploying

Two halves. Static files go to GitHub Pages; the routing API runs in one
container because it needs OSRM and the forecast table.

## One-time

```bash
gh repo create bay-commute-forecast --public --source=. --remote=origin --push

# Pages: Settings > Pages > Deploy from a branch > gh-pages > / (root)
bash deploy/publish.sh          # creates the branch on first run

# API
cd deploy
fly launch --no-deploy --name bay-commute-forecast
fly secrets set GOOGLE_MAPS_API_KEY=...          # optional; Photon otherwise
fly deploy
```

Then point the two halves at each other:

- `fly.toml`: set `ALLOW_ORIGIN` to your Pages origin and `SERVE_URL` to
  `https://<user>.github.io/bay-commute-forecast/data`, and `fly deploy` again.
- `site/config.json`: set `api_base` to `https://bay-commute-forecast.fly.dev`,
  then re-run `deploy/publish.sh`.

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
