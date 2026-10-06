#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
reference_dir="$PWD/vendor/pokemon-showdown"
reference_commit=14546894d86f9589ac11130c510bbe73b6968665
if [[ ! -d "$reference_dir/.git" ]]; then
  git init "$reference_dir"
  git -C "$reference_dir" remote add origin https://github.com/smogon/pokemon-showdown.git
  git -C "$reference_dir" fetch --depth=1 origin "$reference_commit"
  git -C "$reference_dir" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$reference_dir" rev-parse HEAD)" == "$reference_commit" ]]
git -C "$reference_dir" diff --quiet HEAD
cd "$reference_dir"
npm ci --omit=dev --omit=optional --ignore-scripts --no-audit --no-fund
# esbuild's exact version comes from the upstream lock; fetch only its platform binary.
node node_modules/esbuild/install.js
node build
