#!/usr/bin/env bash
# Build the beam-planner solver as a browser WASM module + wasm-bindgen JS glue.
#
#   ./web/build.sh             serial   — stable, runs on any static host
#   ./web/build.sh --threaded  parallel — rayon on Web Workers (nightly +
#                                          build-std; the page must be served
#                                          cross-origin isolated, see serve.py)
#
# Resolves the rustup toolchains explicitly so it works even when another cargo
# (e.g. Homebrew's) shadows rustup on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

WASM_BINDGEN="${WASM_BINDGEN:-$HOME/.cargo/bin/wasm-bindgen}"
WASM="target/wasm32-unknown-unknown/release/beam_planner.wasm"
OUT="web/pkg"

tc_bin() { dirname "$(rustup which --toolchain "$1" cargo)"; }

THREADED=0
if [[ "${1:-}" == "--threaded" ]]; then
  THREADED=1
  echo ">> threaded build (nightly + build-std + shared memory)"
  BIN="$(tc_bin nightly)"
  # This rustc/wasm-ld does not auto-emit the threading setup from +atomics, so
  # request it explicitly: an imported, shared, bounded memory (what wasm-bindgen's
  # thread transform expects) plus the __heap_base / TLS globals it injects into.
  # Single line — RUSTFLAGS is whitespace-split, so no backslash continuations.
  THREAD_FLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=2147483648 -C link-arg=--export=__heap_base -C link-arg=--export-if-defined=__wasm_init_tls -C link-arg=--export-if-defined=__tls_size -C link-arg=--export-if-defined=__tls_align -C link-arg=--export-if-defined=__tls_base"
  PATH="$BIN:$PATH" RUSTFLAGS="$THREAD_FLAGS" \
    cargo build --release --lib --no-default-features --features parallel \
    --target wasm32-unknown-unknown -Z build-std=std,panic_abort
else
  echo ">> serial build (stable)"
  BIN="$(tc_bin stable)"
  PATH="$BIN:$PATH" \
    cargo build --release --lib --no-default-features \
    --target wasm32-unknown-unknown
fi

echo ">> wasm-bindgen --target web → $OUT"
rm -rf "$OUT"   # avoid stale cross-build artifacts (e.g. serial leaving threaded worker snippets)
"$WASM_BINDGEN" "$WASM" --target web --out-dir "$OUT"

if [[ "$THREADED" == "1" ]]; then
  # wasm-bindgen-rayon's worker helper imports the main module as a bare directory
  # (`import('../../..')`), which only resolves through a bundler. On a plain static
  # server the browser fetches the directory, gets text/html, and every worker fails
  # to load — so the pool never starts. Rewrite it to the explicit module file.
  WH="$(find "$OUT/snippets" -name workerHelpers.js)"
  tmp="$(mktemp)"
  sed "s#import('../../..')#import('../../../beam_planner.js')#" "$WH" > "$tmp" && mv "$tmp" "$WH"
  echo ">> patched worker import → ../../../beam_planner.js  ($WH)"
fi

echo ">> done."
echo "   serve:  python3 web/serve.py"
echo "   open:   http://localhost:8000/web/"
