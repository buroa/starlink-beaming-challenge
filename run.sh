#!/usr/bin/env bash
# Build the solver and run every test case through the official evaluate.py.
#
#   ./run.sh
#
# For the interactive 3D visualizer (Beamer):  cargo run --release --bin beamer
set -euo pipefail
cd "$(dirname "$0")"

echo "Building (release)…"
cargo build --release --quiet --bin beam-planner
BIN=./target/release/beam-planner

for f in test_cases/*.txt; do
    name=$(basename "$f")
    "$BIN" "$f" > "/tmp/${name}.out" 2>/dev/null
    echo " > $name"
    grep '^#' "/tmp/${name}.out" | sed 's/^/   /'
    python3 ./evaluate.py "$f" "/tmp/${name}.out" 2>&1 | grep -E "covered|passed all checks" | sed 's/^/   /'
    rm -f "/tmp/${name}.out"
done
