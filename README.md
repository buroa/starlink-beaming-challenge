# Starlink Beam Planning

A fast, fully parallel, deterministic solver for the SpaceX Starlink Beam
Planning tech test — plus a GPU globe to watch it work.

<p align="center">
  <img src="docs/beamer.webp" width="600"
       alt="Beamer rendering the 100,000-user scenario: beams paint onto a transparent, slowly turning globe over a live nebula">
  <br>
  <em>Beamer solving the 100,000-user case (<code>11</code>) — the beam network painting itself onto a transparent globe over a living nebula.</em>
</p>

Given Starlink satellites, users, and non-Starlink "interferer" satellites in
ECEF coordinates, it assigns beams (≤ 32 per satellite, one of 4 colors each) to
serve **as many users as possible** without breaking any constraint. It's a
from-scratch Rust rewrite of a slow, single-threaded Python greedy, and it ships
a **provable near-optimality certificate** with every solution. (See the
[problem statement PDF](docs/satellites-StarlinkBeamPlanningTechTest-091020-1241-938.pdf).)

## The problem

Every served user needs one beam from one satellite, subject to three hard
constraints:

- **Visibility** — the serving satellite must be within 45° of the user's local vertical.
- **Interference** — from the user's view, the satellite must sit ≥ 20° from every non-Starlink satellite.
- **Coloring** — two beams of the same color on one satellite must be ≥ 10° apart.

A satellite carries at most 32 beams across 4 colors. Maximize served users.

## Quickstart

Needs a recent Rust toolchain (and `python3` only for the official validator).

```sh
cargo build --release

# Solve one scenario; the solution (+ a certificate header) goes to stdout.
./target/release/beam-planner test_cases/09_ten_thousand_users.txt

# Maximum-coverage mode: spend seconds of intensive search to recover the last
# few users on the hardest component (default is the ~sub-second solve).
./target/release/beam-planner test_cases/11_one_hundred_thousand_users.txt --max

# Solve and validate every test case through the official evaluate.py.
./run.sh
```

The workspace builds two binaries: **`beam-planner`** (the solver) and
**`beamer`** (the visualizer — see below).

## Results

`achieved` = users served; `bound` = the tighter, coloring-aware ceiling no
valid solution can exceed; `A/bound` = the fraction of that ceiling reached.

| Case | Achieved | Bound | A/bound | Old Python |
|---|---|---|---|---|
| 00_example | 100.00% | 100% | **100%** | 100% |
| 01_simplest_possible | 100.00% | 100% | **100%** | 100% |
| 02_two_users | 100.00% | 100% | **100%** | 100% |
| 03_five_users | 80.00% | 80% (4/5) | **100%** | 80% |
| 04_one_interferer | 0.00% | 0% | **100%** | 0% |
| 05_equatorial_plane | 100.00% | 100% | **100%** | 100% |
| 06_partially_fullfillable | 76.80% | 76.80% | **100%** | 76.8% |
| 07_eighteen_planes | **99.16%** | 100% | 99.2% | 98.88% |
| 08_eighteen_planes_northern | 79.12% | 79.12% | **100%** | 79.12% |
| 09_ten_thousand_users | **93.88%** | 94.06% | 99.8% | 92.95% |
| 10_ten_thousand_users_geo_belt | **84.45%** | 84.81% | 99.6% | 83.77% |
| 11_one_hundred_thousand_users | **29.45%** | 29.79% | 98.9% | 29.40% |

We **match or beat** the old solver on every case, and are **provably optimal**
(achieved = the coloring-aware bound) on **00–06 and 08** — the clique cuts
*certify* case 03's `4/5` as exactly optimal rather than asserting it.

On **07/09/10/11** the coloring bound collapses onto the matching bound: the
binding constraint is per-satellite 32-beam capacity, and the residual gap is a
*global* coloring interaction that per-satellite clique cuts can't tighten. We
can't cheaply certify past ~98.9–99.8% there, but the achieved values are at the
practical optimum — exhaustive ruin-and-recreate on all cores converges to the
same place and recovers only a handful more users for vastly more compute.

## How it works

At 550 km the constellation is sparse: each user sees only ~2–4 satellites, so
the feasibility graph **splits into independent connected components** that share
no satellites. Every component is solved fully in parallel.

1. **Spatial index.** A uniform 3D grid over satellites; a fixed-radius ball
   query yields each user's candidate satellites, then exact visibility +
   interference filters give the feasibility graph (parallel over users).
2. **Upper bounds.** Two ceilings, each an exact Dinic max-flow per component: a
   capacitated **matching** bound (sat cap 32, *ignoring* color), and a tighter
   **coloring-aware** bound — partition each satellite's <10° conflict graph into
   cliques, note that a clique of `k` mutually conflicting users needs `k` colors
   so at most 4 can be served, and cap the satellite at `Σ min(4, |Cᵢ|)`. The
   flow under these caps is still sound and ≤ the matching bound (it proves case
   03's optimum is **4**, not 5).
3. **Ensemble construction.** Four **coloring-integral** greedy variants run in
   parallel per component — a user is admitted to a satellite only if a valid
   color exists, so coloring is never a fragile post-step — keeping the best. A
   fifth **flow-seeded** build runs only when the greedy ensemble falls short of
   the matching bound. Per-satellite 4-coloring is solved *exactly*: DSATUR +
   bounded backtracking, with color-symmetry breaking and a clique cutoff.
4. **Bounded repair.** Short augmenting displacement chains recover stragglers,
   with atomic rollback and a strict budget so it can never blow up.
5. **Large-neighborhood search.** Many **independent** ruin-and-recreate searches
   run in parallel — each tears down the satellites around a still-unserved
   terminal and rebuilds that cluster — keeping the best. A transactional undo
   makes each round O(touched) instead of O(component), and work-stealing keeps
   every core on the hard component. Iteration-bounded, so it stays deterministic.

The result is **valid by construction** (every assignment clears the coloring
oracle) and **bit-for-bit deterministic**: no RNG, a fixed-seed search, explicit
tie-breaks, and no dependence on thread scheduling. The only run-to-run wobble is
a handful of users sitting exactly on the 10°/45° thresholds, where `f64`
rounding can flip under a different compile.

## Performance

The full 100,000-user / 1,440-satellite solve — construction, repair, and the
parallel polish — finishes in **~0.55 s** on all cores, far under the 15 min /
1 GB limits; every smaller case is sub-second (the 10k cases land at ~0.1–0.25 s).
The exact 4-coloring oracle dominates the hard component, so it's the most tuned
hot path: stack-allocated search state, incremental neighbour-color counts,
color-symmetry breaking, and a K5 clique cutoff cut its work by ~16× — and the
whole 100k case by ~30× — versus the first correct version, with no loss of
coverage. The polish is a single speed↔quality knob (`LNS_MAX_ROUNDS`).

The opt-in **`Maximum`** algorithm (CLI `--max`, or the visualizer's algorithm
picker) chases the last few users on any component that still has a gap: a much
larger LNS budget, *both* construction colorings (kept best per component, since
recoloring during construction helps some components and hurts others), and a
coloring-complete repair pass. It recovers **+1 user each on cases 09 and 10 and
+6 on the 100k case** (29,446 → 29,452), taking ~10 s on the 100k case. That an
exhaustive search recovers only ~8 users total — against a residual gap-to-bound
of ~400 — is the evidence that that gap is a **global coloring coupling** the
(loose) matching/clique bound can't see, not unserved-but-servable users. The
default solve is already at the practical optimum; `Maximum` just proves it the
expensive way.

## WebAssembly

The solver core compiles to a browser-ready `wasm-bindgen` module — the same
production code, exposed as one entry point:

```js
import init, { solve_scenario, initThreadPool } from './pkg/beam_planner.js';
await init();
await initThreadPool(navigator.hardwareConcurrency); // threaded build only
const solution = solve_scenario(scenarioText, /* intense = */ false);
// For the threaded build, run these inside a Web Worker — see the note below.
```

Two Cargo features gate this. `viz` (default) pulls in the visualizer's
native-only stack (eframe/wgpu/ureq/…); `--no-default-features` drops it to the
headless solver + library. `parallel` (default) is rayon — a work-stealing
OS-thread pool natively, or rayon-on-Web-Workers in the browser via
`wasm-bindgen-rayon`. The solver is deterministic, so the serial and parallel
builds produce **bit-identical** solutions (verified by diffing the CLI output of
both builds on cases 09 and 11).

A ready-to-run harness lives in [`web/`](web/): the **visualizer** at `/` and a
one-button **solver** at `/solver`. An [`xtask`](xtask/) — the Rust replacement
for build scripts — builds both WASM modules and assembles the deployable site;
wasm-bindgen runs as a library inside it, so there's nothing external to install:

```sh
cargo xtask dist                           # build both modules → web/dist/
(cd web/dist && python3 -m http.server)    # → http://localhost:8000/  (and /solver)
```

(`cargo xtask viz` and `cargo xtask solver` build the individual modules.)

The threaded builds need [cross-origin isolation](https://web.dev/articles/coop-coep)
(for `SharedArrayBuffer`), which static hosts like GitHub Pages can't grant via
headers — so [`web/coi-serviceworker.js`](web/coi-serviceworker.js) supplies it
with a service worker (the harnesses fall back to a serial solve if it's
unavailable). A GitHub Actions workflow ([`.github/workflows/pages.yml`](.github/workflows/pages.yml))
builds both modules and publishes the site to **GitHub Pages** on every push to
`main` — one-time setup: repo *Settings → Pages → Source → "GitHub Actions"*.

The visualizer page wants **both** `--viz` (the render module) and `--threaded`
(the parallel solve module its Worker prefers); with only `--viz` it still works,
solving serially in the Worker.

The serial build is a one-liner on stable; the threaded build is fussier, because
this toolchain's `wasm-ld` does *not* derive the threading setup from `+atomics`
alone. The memory and TLS globals have to be requested explicitly — an
**imported, shared, bounded** memory (what wasm-bindgen's thread transform
expects) plus the `__heap_base`/TLS exports it injects into:

```sh
# Serial:
cargo build --release --lib --no-default-features --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/release/beam_planner.wasm --target web --out-dir web/pkg

# Threaded:
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals \
  -C link-arg=--shared-memory -C link-arg=--import-memory -C link-arg=--max-memory=2147483648 \
  -C link-arg=--export=__heap_base \
  -C link-arg=--export-if-defined=__wasm_init_tls \
  -C link-arg=--export-if-defined=__tls_size \
  -C link-arg=--export-if-defined=__tls_align \
  -C link-arg=--export-if-defined=__tls_base' \
  cargo +nightly build --release --lib --no-default-features --features parallel \
  --target wasm32-unknown-unknown -Z build-std=std,panic_abort
wasm-bindgen target/wasm32-unknown-unknown/release/beam_planner.wasm --target web --out-dir web/pkg
```

`std::time::Instant` (used for the solver's repair deadlines) panics on
`wasm32-unknown-unknown`, so the library uses [`web-time`](https://crates.io/crates/web-time),
an API-identical drop-in that is a zero-cost std re-export off-wasm. The threaded
build's `SharedArrayBuffer` requires the page to be **cross-origin isolated**
(`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy:
require-corp`). Hosts that can set headers can do so directly; `coi-serviceworker.js`
injects them otherwise (and re-serves the cross-origin basemap tiles — which send
`Access-Control-Allow-Origin: *` — as `Cross-Origin-Resource-Policy: cross-origin`
so they survive the policy). The serial build needs none of this.

**Run the parallel solve in a Worker, not on the main thread.** rayon blocks the
calling thread until its pool finishes, and the browser main thread is forbidden
from blocking (`Atomics.wait` throws there) — so a threaded solve invoked on the
main thread *deadlocks*. The harness runs it in
[`web/solver-worker.js`](web/solver-worker.js), which also keeps the UI responsive
during the slower serial solve. (One more static-host wrinkle: wasm-bindgen-rayon's
worker bootstrap imports the main module as a bare directory, which only resolves
through a bundler — `xtask` rewrites it to the explicit file so it works when
served as plain files.)

Threading only wins where there's real parallelism to exploit. Measured in-browser
on 16 hardware threads (`solve_scenario` wall time, all three matching the CLI's
coverage exactly):

| Case | Serial | Threaded (16 workers) |
|---|---|---|
| `03` · 5 users | 4 ms | 15 ms |
| `09` · 10k users | 2.5 s | 4.3 s |
| `11` · 100k users | 31.7 s | **12.0 s** |

The 100k case splits into thousands of independent components, so it parallelizes
~2.6×; the small cases are dominated by per-op atomic overhead and run faster
serial. Reach for the threaded build when the 100k-class solve time matters;
otherwise serial is simpler and needs no special headers.

**The visualizer runs in the browser too.** `cargo xtask viz` compiles the whole
eframe/egui/wgpu app to wasm, and [`web/index.html`](web/index.html) mounts it on a
`<canvas>` via eframe's `WebRunner` — the globe, live nebula, the full HUD, and any
of the 12 scenarios fetched on demand. The solve runs **in a Web Worker**
([`web/viz-solver-worker.js`](web/viz-solver-worker.js)) so the canvas keeps animating
throughout — and it's **parallel**: the worker prefers the threaded solver module
(`cargo xtask solver`), bringing up the same 16-way `wasm-bindgen-rayon` pool
the solver harness uses (10k solves in ~0.4 s vs ~2.5 s serial), and falls back to
the serial module if the threaded build or cross-origin isolation isn't available.
It returns `(Scenario, Feasibility, Trace)` postcard-serialized; the render thread
rebuilds its state. So even the 100k case never freezes. It renders through **WebGL2**: eframe 0.29 ships wgpu 22,
whose WebGPU path requests a device limit (`maxInterStageShaderComponents`) that
current browsers removed, so the entry forces `Backends::GL` and the build enables
wgpu's `webgl` feature. The viz `[[bin]]` moved into the library (`src/viz/`) so
wasm-bindgen can emit it; native and headless modes are
`#[cfg(not(target_arch = "wasm32"))]`-gated. **Live basemap tiles** stream too: the
native streamer's 8-thread + `ureq` model is ported to per-tile async `fetch` on
wasm (keeping the quadtree refinement, LRU cache, and `image` decode — only the
execution model changes), and the web build defaults the basemap on. The
visualizer now runs at **full parity in the browser** — render, all 12 scenarios,
the parallel solve, and live tiles.

## Visualizer — Beamer

`beamer` is a GPU-rendered, interactive 3D globe built straight into the app
(wgpu + egui) — no browser, no external tooling.

```sh
cargo run --release --bin beamer
```

It opens fullscreen, framed on the United States, on the 100k-user case (`11`),
and plays the assignment immediately. The **same production solver** drives it,
so its coverage matches the CLI certificate exactly; `Optimized · ensemble +
repair` is the default algorithm, with the greedy and flow-seeded constructions
selectable for comparison (they report *lower* coverage, as expected) and a
`Maximum · intensive search` mode that trades seconds for the last few users
(the CLI equivalent is `--max`).

Rendering is 4× MSAA with a starfield backdrop and beams as RGB ribbons
(A red, B green, C blue, D yellow). The earth is **transparent by default** —
pick a **basemap** (Dark / Light / Satellite) to stream a live, level-of-detail
globe: nothing is pre-baked, so zooming in pulls higher-detail tiles on
background threads. The **Fresnel atmosphere halo** is a separate toggle, so its
blue glow can ride over the transparent earth with no basemap at all. Scroll all
the way through the surface to the core and watch the network from the inside out.

The black/white/glass HUD is one left-hand control column plus three fixed
readouts; `H` hides all of it:

- **Left column** — scenario and algorithm pickers, color-band toggles, scene
  **Layers** (beams, full/partial satellites, uncovered terminals, and
  **interferers**), and the **Basemap** selector with its independent
  **Atmosphere halo** toggle.
- **Coverage** (top-right) — live served / total and percent-of-optimum.
- **Transport** (bottom-center) — restart, play/pause, scrubber, speed presets.
- **Unserved Terminals** (bottom-right) — counts grouped by *why* each terminal
  failed (no satellite in view, blocked by an interferer, all satellites full, or
  no free color), with a list you can **click to fly the camera to**.

Toggle **Interferers** to plot the non-Starlink satellites; hover one to light up
its **20° field of interference** as a footprint ring on the globe directly
beneath it. Hover any satellite, terminal, or interferer for a tooltip (id, beams
in use, band, or why it went unserved) — and a served terminal's tooltip takes a
faint tint of its assigned band color.

**Satellite focus.** Click any satellite to drop into a cinematic study of just
that one: the camera flies in, everything else falls away, and the satellite gets
a pulsing lock-on **reticle** with its beam fan to the users it serves (colored by
band) and the nearest interferer's 20° field. A focus card replaces the global
readouts with the satellite's identity, a **beams X / 32** gauge, the per-band
**A/B/C/D** breakdown, interferer proximity, and a scoped **replay** scrubber that
renders out just this satellite's assignment. Change the algorithm while focused
to watch the change for that one satellite in isolation. Click another satellite
to switch, or `Esc` / click empty space to exit.

**Controls**

| Input | Action |
|---|---|
| Drag | Orbit |
| Scroll | Zoom (all the way to the core) |
| Click a satellite | Focus it (Esc / click away to exit) |
| `H` | Hide / show the HUD |
| `F11` | Toggle fullscreen |
| `Esc` | Leave focus, then fullscreen |

Two headless modes render without a window: `beamer --shot <scenario> <out.png>
[fraction]` writes a single frame to a PNG, and `beamer --frames <scenario>
<dir> <n> [orbit°]` solves once and writes an `n`-frame playback sweep (with an
optional camera orbit) as numbered PNGs. The looping demo at the top of this page
is a full-color animated **WebP** (much smaller than a 256-color GIF), made with
`ffmpeg` (downscale) + `img2webp` (encode):

```sh
beamer --frames test_cases/11_one_hundred_thousand_users.txt /tmp/frames 60 16
# downscale the frames to 600px wide
ffmpeg -i /tmp/frames/frame_%05d.png -vf "scale=600:-1:flags=lanczos" /tmp/f/frame_%05d.png
# encode an animated WebP, holding the last (full-coverage) frame for a beat
img2webp -loop 0 -lossy -q 84 -m 6 \
  -d 50 /tmp/f/frame_*.png -d 1500 /tmp/f/frame_00060.png -o docs/beamer.webp
```

Basemaps © OpenStreetMap contributors © CARTO; satellite imagery © Esri.
