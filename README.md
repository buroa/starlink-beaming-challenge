# Beamer

**A fast, fully parallel, deterministic solver for the [SpaceX Starlink Beam
Planning tech test](docs/satellites-StarlinkBeamPlanningTechTest-091020-1241-938.pdf)
— with a GPU globe to watch it work.**

<p align="center">
  <img src="docs/beamer.webp" width="600"
       alt="Beamer solving the 100,000-user scenario: beams paint onto a transparent, slowly turning globe over a live nebula">
  <br>
  <em>The 100,000-user case (<code>11</code>) — the beam network painting itself onto a transparent globe over a living nebula. <a href="https://buroa.github.io/beamer/">▶ Try it live.</a></em>
</p>

Given Starlink satellites, users, and non-Starlink "interferer" satellites in
ECEF coordinates, Beamer assigns beams (≤ 32 per satellite, one of 4 colors) to
serve **as many users as possible** without breaking a constraint. It's a
from-scratch Rust rewrite of a slow, single-threaded Python greedy, and it ships
a **provable near-optimality certificate** with every solution — an upper bound,
not a vibe.

The workspace is two crates: **`beamer`**, the solver (library + CLI), and
**`beamer-viz`**, the visualizer (a native GUI and a browser app that embeds the
solver).

## The problem

Every served user needs one beam from one satellite, subject to three hard
constraints:

- **Visibility** — the satellite must be within 45° of the user's local vertical.
- **Interference** — from the user's view, the satellite must sit ≥ 20° from every non-Starlink satellite.
- **Coloring** — two same-color beams on one satellite must be ≥ 10° apart.

Each satellite carries ≤ 32 beams across 4 colors. Maximize served users.

## Quickstart

Needs a recent Rust toolchain (`python3` only for the official validator).

```sh
cargo build --release

# Solve a scenario → validator-format solution (+ certificate header) on stdout.
./target/release/beamer test_cases/09_ten_thousand_users.txt

# Check it with the official validator: it takes the scenario as an argument and
# reads the solution from stdin (the `#` certificate header is ignored), so just pipe.
./target/release/beamer test_cases/09_ten_thousand_users.txt | python3 evaluate.py test_cases/09_ten_thousand_users.txt

# --max: trade seconds of intensive search for the last few users on the hardest case.
./target/release/beamer test_cases/11_one_hundred_thousand_users.txt --max
```

Need a bigger scenario? The bundled `gen` tool builds one at any scale — a Walker
constellation at 550 km, users on the WGS84 ellipsoid, and an optional
geostationary interferer belt (deterministic per `--seed`):

```sh
# 1,000,000 users + 5,000 satellites + a 36-satellite interferer belt → a file
cargo run --release --bin gen -- --users 1000000 --sats 5000 --interferers 36 -o test_cases/big.txt
```

## Results

`achieved` = users served · `bound` = the tighter, coloring-aware ceiling no
valid solution can beat · `A/bound` = the fraction of that ceiling reached.

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

We match or beat the old Python on every case and are **provably optimal**
(achieved = bound) on 00–06 and 08 — the clique cuts *certify* case 03's `4/5` as
exactly optimal rather than asserting it. On 07/09/10/11 the binding constraint is
per-satellite 32-beam capacity and the residual gap is a *global* coloring
coupling the (loose) bound can't see: exhaustive search recovers only a handful
more users for vastly more compute, so the defaults already sit at the practical
optimum.

## How it works

At 550 km the constellation is sparse — each user sees only ~2–4 satellites — so
the feasibility graph **splits into independent connected components** that share
no satellites and are solved fully in parallel.

1. **Spatial index** — a uniform 3D grid over satellites + a fixed-radius ball
   query gives each user's candidate satellites; exact visibility/interference
   filters then build the feasibility graph.
2. **Upper bounds** — two exact Dinic max-flows per component: a capacitated
   **matching** bound (cap 32, ignoring color) and a tighter **coloring-aware**
   bound (partition each satellite's <10° conflict graph into cliques; a
   `k`-clique needs `k` colors, so cap the satellite at `Σ min(4, |Cᵢ|)`). This
   proves case 03's optimum is **4**, not 5.
3. **Ensemble construction** — four **coloring-integral** greedy variants run per
   component (a user is admitted to a satellite only if a valid color exists, so
   coloring is never a fragile post-step), plus a flow-seeded build when they fall
   short. Per-satellite 4-coloring is solved *exactly*: DSATUR + bounded
   backtracking, with color-symmetry breaking and a clique cutoff.
4. **Bounded repair** — short augmenting displacement chains recover stragglers,
   with atomic rollback and a strict budget so it can never blow up.
5. **Large-neighborhood search** — many independent ruin-and-recreate searches run
   in parallel (each tears down and rebuilds the cluster around an unserved
   terminal), keeping the best; a transactional undo makes each round O(touched)
   and work-stealing keeps every core on the hard component.

Every assignment clears the coloring oracle, so the result is **valid by
construction** and **bit-for-bit deterministic**: no RNG, a fixed-seed search,
explicit tie-breaks, and no dependence on thread scheduling.

## Performance

The full 100,000-user / 1,440-satellite solve — construction, repair, and the
parallel polish — finishes in **~0.2 s** on all cores, far under the 15 min /
1 GB limits; every smaller case is sub-second. The exact 4-coloring oracle
dominates the hard component and is the most-tuned hot path (stack-allocated
search state, incremental neighbor-color counts, a K5 clique cutoff, and a search
budget sized to *find* a coloring rather than exhaustively *disprove* one). The
opt-in **`--max`** mode chases the last few users with a much larger search
budget — it recovers ~6 users on the 100k case (29,446 → 29,452) in ~10 s,
evidence that the remaining gap is that global coloring coupling and not servable
users left behind.

It also **scales far past the test set.** Dense, globe-spanning constellations
collapse the feasibility graph into one giant component, so the cross-component
parallelism falls away — to keep all cores busy the solver overlaps the serial
max-flow bound with the parallel greedy ensemble, realizes the flow seed across
satellites in parallel, and on a *saturated* mega-component (more users than beam
slots) skips the ejection-chain repair and ruin-and-recreate polish, which can
only spin when every satellite is already full. Measured on 16 threads
(`gen`-built scenarios; set `BEAM_PROFILE=1` for a per-phase breakdown):

| Users / Satellites | Time | Served / bound |
|---|---|---|
| 1,000,000 / 5,000 | **1.1 s** | 100% |
| 1,000,000 / 10,000 | **2.0 s** | 100% |
| 2,000,000 / 8,000 | **3.3 s** | 100% |
| 5,000,000 / 20,000 | **19 s** | 100% |
| 8,000,000 / 30,000 | **45 s** | 100% |
| 10,000,000 / 40,000 | **73 s** | 100% |

Feasibility, the local graph build, the flow-seed realization, and the
ruin-and-recreate search are all parallel; a sqrt-free visibility pre-filter and
a tightened coloring-search budget cut the constant factors. The one phase that
stays sequential is a single giant component's greedy fill — order-dependent by
nature, so parallelizing it would change the result, which we don't trade for
speed.

## Beamer — the visualizer

A GPU-rendered, interactive 3D globe (wgpu + egui), native or in the browser.

```sh
cargo run --release --bin beamer-viz
```

It opens on the 100k case and plays the assignment live, driven by the **same
production solver** — so coverage matches the CLI certificate exactly. Beams are
RGB ribbons (A red · B green · C blue · D yellow) painting onto a **transparent
earth** over a procedural nebula; pick a **basemap** (Dark / Light / Satellite) to
stream a live level-of-detail globe (nothing pre-baked — zooming pulls
higher-detail tiles on background threads), with an independent **Fresnel
atmosphere** toggle. Scroll through the surface to the core and watch the network
from the inside out.

- **Pick a scenario, or bring your own** — the scenario list ends in **"Add your
  own…"**: upload, paste, or drag-and-drop a validator-format scenario, solve it
  on the globe, and **download** the solution. Switch algorithms (greedy /
  flow-seeded for comparison, or `Maximum`, the `--max` equivalent).
- **Inspect** — color-band and scene-layer toggles (beams, full/partial
  satellites, uncovered terminals, interferers); hover anything for a tooltip;
  toggle **interferers** to light up each one's 20° field of interference as a
  footprint ring on the globe beneath it.
- **Why unserved** — a bottom-right card groups failures by cause (no satellite in
  view / blocked by an interferer / all full / no free color) with a list you can
  **click to fly the camera to**.
- **Focus a satellite** — click one to drop into a cinematic study of just that
  one: a pulsing lock-on reticle, its beam fan to the users it serves, the nearest
  interferer's field, and a scoped **replay** scrubber. `Esc` or click away to exit.

| Input | Action |
|---|---|
| Drag | Orbit |
| Scroll | Zoom (all the way to the core) |
| Click a satellite | Focus it |
| `H` | Hide / show the HUD |
| `F11` | Toggle fullscreen |
| `Esc` | Dismiss the dialog → leave focus → exit fullscreen |

## Run it in the browser

`beamer-viz` compiles to WebAssembly and **embeds the solver**, so the visualizer
*is* the web front end — **[live on GitHub Pages](https://buroa.github.io/beamer/)**.
The browser build is full parity: render, all 12 scenarios, paste-your-own +
download, live basemap tiles, and the **parallel solve**. The solve runs in a Web
Worker; on a [cross-origin-isolated](https://web.dev/articles/coop-coep) page it
brings up a [`wasm-bindgen-rayon`](https://github.com/RReverser/wasm-bindgen-rayon)
thread pool — and because the solver is deterministic, serial and threaded builds
produce bit-identical solutions. Measured in-browser on 16 hardware threads:

| Case | Serial | Threaded (16 workers) |
|---|---|---|
| `09` · 10k users | 2.5 s | **0.38 s** |
| `11` · 100k users | 31.7 s | **12.0 s** |

```sh
cd beamer-viz && npm install && npm run serve
```

`npm run build` compiles the wasm **twice** (a single-thread `pkg/` and a
multi-thread `pkg-parallel/`); webpack bundles a loader that
[feature-detects threads](https://github.com/GoogleChromeLabs/wasm-feature-detect)
and picks the right one, and a [GitHub Actions
workflow](.github/workflows/pages.yml) publishes `dist/` to Pages on every push to
`main`. A few wasm realities, handled: rendering is WebGL2 (`Backends::GL`) for the
widest browser support; [`coi-serviceworker.js`](coi-serviceworker.js) supplies the
cross-origin-isolation headers Pages can't set; and `std::time::Instant` (which
panics on wasm) is swapped for the drop-in [`web-time`](https://crates.io/crates/web-time).

## The hero image

The looping demo above is a full-color animated WebP (much smaller than a
256-color GIF). Two headless modes render without a window — `--shot <scenario>
<out.png> [fraction]` (one frame) and `--frames <scenario> <dir> <n> [orbit°]` (a
playback sweep with an optional camera orbit) — then `ffmpeg` + `img2webp` encode it:

```sh
beamer-viz --frames test_cases/11_one_hundred_thousand_users.txt /tmp/frames 60 16
ffmpeg -i /tmp/frames/frame_%05d.png -vf "scale=600:-1:flags=lanczos" /tmp/f/frame_%05d.png
img2webp -loop 0 -lossy -q 84 -m 6 -d 50 /tmp/f/frame_*.png -d 1500 /tmp/f/frame_00059.png -o docs/beamer.webp
```

---

Basemaps © OpenStreetMap contributors © CARTO; satellite imagery © Esri.
