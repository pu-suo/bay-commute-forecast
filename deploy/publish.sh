#!/bin/bash
# Publish the built site to the gh-pages branch.
#
# The site is 9 MB of regenerated JSON a night. Committing that to a tracked
# branch would add about 3 GB of history a year, so gh-pages is rebuilt as a
# single orphan commit each time and force-pushed. Nothing on it is history
# worth keeping; main is the history.
#
# The API container pulls forecast.parquet from the same published tree, which
# is why the serving tables are copied in alongside the JSON.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${TRAFFIC_DATA:-$HOME/traffic-data}"
BRANCH="${PAGES_BRANCH:-gh-pages}"
cd "$REPO"

git rev-parse --git-dir >/dev/null
git remote get-url origin >/dev/null || {
    echo "no origin remote; run: gh repo create --source=. --remote=origin" >&2
    exit 1
}

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp site/index.html site/accuracy.html site/app.js site/style.css "$STAGE/"
cp site/config.json "$STAGE/"
mkdir -p "$STAGE/data"
cp site/data/*.json "$STAGE/data/"
cp "$DATA/serve/forecast.parquet" "$DATA/serve/freeflow.parquet" "$STAGE/data/"
# Pages runs the tree through Jekyll otherwise, which drops files it does not
# recognise and directories beginning with an underscore.
touch "$STAGE/.nojekyll"

SIZE=$(du -sh "$STAGE" | cut -f1)
ORIGIN="$(git remote get-url origin)"
NAME="$(git config user.name || echo deploy)"
EMAIL="$(git config user.email || echo deploy@localhost)"

cd "$STAGE"
git init -q -b "$BRANCH" .
git add -A
git -c user.name="$NAME" -c user.email="$EMAIL" \
    commit -q -m "site $(date -u +%FT%TZ)"
git remote add origin "$ORIGIN"
git push -q --force origin "$BRANCH"

echo "published $SIZE to $BRANCH"
