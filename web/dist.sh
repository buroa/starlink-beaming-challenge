#!/usr/bin/env bash
# Build both WASM modules and assemble the static site published to GitHub Pages.
#
#   web/dist/                 the deployable site (serve it with any static host)
#     index.html              the visualizer  ( / )
#     viz-pkg/  pkg/          viz render module + threaded solver module
#     test_cases/             scenarios (fetched by the viz and the solver)
#     viz-solver-worker.js    the viz's off-thread solve worker
#     coi-serviceworker.js    cross-origin isolation (wasm threads) shim
#     solver/                 the solver harness  ( /solver )
#       index.html  solver-worker.js  pkg/  coi-serviceworker.js
#
# Each page uses only relative paths, so the site works at any base URL (project
# Pages live under /<repo>/). The threaded `pkg/` is duplicated into /solver so
# both pages can reference it as ./pkg.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> building wasm modules"
./web/build.sh --viz        # → web/viz-pkg   (eframe/wgpu render module)
./web/build.sh --threaded   # → web/pkg       (threaded solver: solve_scenario + trace_scenario)

DIST=web/dist
echo ">> assembling $DIST"
rm -rf "$DIST"
mkdir -p "$DIST/solver"

# --- viz at the site root ---
cp web/index.html             "$DIST/index.html"
cp web/viz-solver-worker.js   "$DIST/"
cp web/coi-serviceworker.js   "$DIST/"
cp -R web/viz-pkg             "$DIST/viz-pkg"
cp -R web/pkg                 "$DIST/pkg"
cp -R test_cases              "$DIST/test_cases"

# --- solver at /solver (fetches ../test_cases → the shared root copy) ---
cp web/solver.html            "$DIST/solver/index.html"
cp web/solver-worker.js       "$DIST/solver/"
cp web/coi-serviceworker.js   "$DIST/solver/"
cp -R web/pkg                 "$DIST/solver/pkg"

# Tell GitHub Pages to serve everything verbatim (no Jekyll mangling).
touch "$DIST/.nojekyll"

echo ">> done → $DIST"
echo "   serve locally:  (cd $DIST && python3 -m http.server 8000)  then open http://localhost:8000/"
