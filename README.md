# Beamer

**A fast, parallel, deterministic solver for the [SpaceX Starlink beam-planning
tech test](docs/satellites-StarlinkBeamPlanningTechTest-091020-1241-938.pdf) — and
a GPU globe to watch it work.**

<p align="center">
  <img src="docs/beamer.webp" width="600"
       alt="Beamer solving the 100,000-user scenario: beams paint onto a transparent, slowly turning globe over a live nebula">
  <br>
  <em>The 100,000-user case (<code>11</code>) painting itself onto a transparent globe over a living nebula. <a href="https://buroa.github.io/beamer/">▶ Try it live.</a></em>
</p>

Given Starlink satellites, users, and non-Starlink "interferer" satellites in
earth-centered coordinates, Beamer assigns beams — at most 32 per satellite, each
one of four colors — to serve **as many users as possible**. Every solution ships
with a **provable near-optimality certificate**: an upper bound no valid
assignment can beat. Not a vibe — a number.

The workspace is two crates: **`beamer`**, the solver (library, CLI, and a
scenario generator), and **`beamer-viz`**, the visualizer (a native window and a
browser app that embeds the solver).

## The problem

Each served user gets one beam from one satellite, under three hard constraints:

- **Visibility** — the satellite is within 45° of the user's local vertical.
- **Interference** — from the user's view, the satellite is ≥ 20° from every non-Starlink satellite.
- **Coloring** — two same-color beams on one satellite are ≥ 10° apart (4 colors, ≤ 32 beams per satellite).

Maximize served users. The constraints are *coupled* — placing one beam changes
which colors its neighbors can take — so first-fit greedy alone leaves users
stranded.

## Quickstart

A recent Rust toolchain is all you need (`python3` only for the reference validator).

```sh
cargo build --release

# Solve → validator-format beams + a `#` certificate header, on stdout.
./target/release/beamer test_cases/09_ten_thousand_users.txt

# Score it with the official validator (it reads the solution from stdin):
./target/release/beamer test_cases/09_ten_thousand_users.txt | python3 evaluate.py test_cases/09_ten_thousand_users.txt

# `--max`: spend seconds of deeper search for the last few users on the hard cases.
./target/release/beamer test_cases/11_one_hundred_thousand_users.txt --max
```

Want a bigger problem than the test set? `gen` builds one at any scale — a Walker
constellation at 550 km, users on the WGS84 ellipsoid, an optional geostationary
interferer belt — deterministic per `--seed`:

```sh
# 1,000,000 users · 5,000 satellites · a 36-satellite interferer belt.
cargo run --release --bin gen -- --users 1000000 --sats 5000 --interferers 36 -o big.txt
./target/release/beamer big.txt
```

## Results

`achieved` = users served · `bound` = the coloring-aware ceiling no valid
assignment can beat · `A/bound` = the fraction of that ceiling reached.

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

We match or beat the original single-threaded Python greedy on every case, and are
**provably optimal** (achieved = bound) on eight of twelve. The clique cuts don't
*assert* case 03's `4/5` — they *certify* it: five mutually-visible users on one
satellite form a 5-clique, which needs five colors, so four is the true maximum.
The residual gaps on 07/09/10/11 come from a *global* coloring coupling the bound
can't see; `--max` confirms it recovers only a handful of users for far more
compute, so the defaults already sit at the practical optimum.

## How it works

At 550 km the constellation is sparse — a user sees only ~2–4 satellites — so the
feasibility graph **splits into connected components** that share no satellites and
solve fully in parallel.

1. **Feasibility** — a uniform 3-D grid over the satellites answers each user's
   fixed-radius candidate query in O(1); exact visibility + interference filters
   build the bipartite graph. Parallel over users, with a sqrt-free pre-filter for
   the common rejects.
2. **Bounds** — two exact [Dinic](https://en.wikipedia.org/wiki/Dinic%27s_algorithm)
   max-flows per component: a capacitated **matching** ceiling (cap 32, ignoring
   color), and a tighter **coloring-aware** one — partition each satellite's
   <10°-conflict graph into cliques; a `k`-clique needs `k` colors, so cap it at
   `Σ min(4, |Cᵢ|)`.
3. **Construction** — an ensemble of **coloring-integral** greedy variants: a user
   is admitted to a satellite only when a valid 4-coloring still exists, so coloring
   is never a fragile after-step. That oracle is *exact* — DSATUR with bounded
   backtracking, color-symmetry breaking, and a K5 cutoff — and runs in
   microseconds. A flow-seeded build competes when greedy falls short.
4. **Polish** — bounded augmenting displacement chains recover stragglers, then
   parallel **ruin-and-recreate** search tears down and rebuilds the cluster around
   each unserved user, keeping the best. A transactional undo makes every round
   O(touched), never O(component).

Because every placement clears the coloring oracle, the result is **valid by
construction**. And it's **bit-for-bit deterministic** — no RNG, fixed-seed search,
explicit tie-breaks, no thread-order dependence — so the serial and 16-thread
builds produce identical solutions. The 100,000-user case lands in **~0.2 s** on
all cores; every smaller case is sub-second, far inside the 15-minute / 1-GB budget.

## At scale

A dense, globe-spanning constellation links everything into one giant component, so
the cross-component parallelism evaporates. The solver keeps the cores fed anyway:
the serial max-flow bound overlaps the parallel greedy ensemble, the flow seed
realizes across satellites in parallel, the ensemble widens to fill idle cores, and
a *saturated* mega-component — more users than beam slots — skips the displacement
repair and ruin-and-recreate, which can only spin when every satellite is already
full. Measured on 16 threads (`gen` scenarios; set `BEAM_PROFILE=1` for a per-phase
breakdown):

| Users / Satellites | Time | Served / bound |
|---|---|---|
| 1,000,000 / 5,000 | **1.1 s** | 100% |
| 1,000,000 / 10,000 | **2.0 s** | 100% |
| 2,000,000 / 8,000 | **3.3 s** | 100% |
| 5,000,000 / 20,000 | **19 s** | 100% |
| 8,000,000 / 30,000 | **45 s** | 100% |
| 10,000,000 / 40,000 | **73 s** | 100% |

All optimal. The one phase that stays sequential is a single component's greedy
fill — order-dependent by nature, so parallelizing it would change the answer,
which we don't trade for speed.

## The visualizer

```sh
cargo run --release --bin beamer-viz
```

A GPU-rendered interactive globe (wgpu + egui), driven by the **same production
solver** — so what you see matches the CLI certificate exactly. It opens on the
100k case and paints the assignment live: beams are RGB ribbons (**A** red · **B**
green · **C** blue · **D** yellow) on a *transparent* earth over a procedural
nebula. Pick a **basemap** (Dark / Light / Satellite) to stream a live
level-of-detail globe — nothing pre-baked; zooming pulls sharper tiles on
background threads — and scroll all the way through the surface to watch the
network from the inside out.

- **Bring your own** — the scenario list ends in *"Add your own…"*: upload, paste,
  or drag a validator-format scenario, solve it on the globe, and **download** the
  result. Switch algorithms (greedy / flow-seeded / `Maximum`).
- **Inspect** — toggle color bands and scene layers (beams, full/partial
  satellites, uncovered users, interferers); hover anything for a tooltip; light up
  each interferer's 20° field as a footprint ring on the globe beneath it.
- **Why unserved** — a card groups failures by cause (none in view / blocked by an
  interferer / all full / no free color), each entry clickable to fly the camera there.
- **Focus a satellite** — click one for a cinematic study of just that satellite: a
  lock-on reticle, its beam fan to the users it serves, the nearest interferer's
  field, and a scoped replay scrubber.

| Input | Action |
|---|---|
| Drag | Orbit |
| Scroll | Zoom (all the way to the core) |
| Click a satellite | Focus it |
| `H` / `F11` | HUD / fullscreen |
| `Esc` | Dismiss dialog → leave focus → exit fullscreen |

### In the browser

`beamer-viz` compiles to WebAssembly and **embeds the solver**, so the visualizer
*is* the web front end — **[live on GitHub Pages](https://buroa.github.io/beamer/)**,
at full parity. The solve runs in a Web Worker; on a
[cross-origin-isolated](https://web.dev/articles/coop-coep) page it brings up a
[`wasm-bindgen-rayon`](https://github.com/RReverser/wasm-bindgen-rayon) thread pool —
and because the solver is deterministic, the single- and multi-threaded builds
return identical solutions, the second just faster.

```sh
cd beamer-viz && npm install && npm run serve
```

`npm run build` compiles the wasm twice (single-thread `pkg/`, multi-thread
`pkg-parallel/`); webpack
[feature-detects threads](https://github.com/GoogleChromeLabs/wasm-feature-detect)
and loads the right one, and a [GitHub Action](.github/workflows/pages.yml) deploys
`dist/` to Pages on every push to `main`. The wasm realities, handled: WebGL2
(`Backends::GL`) for the widest reach, [`coi-serviceworker.js`](coi-serviceworker.js)
for the isolation headers Pages won't set, and
[`web-time`](https://crates.io/crates/web-time) in place of the
`std::time::Instant` that panics in wasm.

## Reproducing the demo

The looping hero is an animated WebP. Two headless render modes need no window —
`--shot <scenario> <out.png> [fraction]` for one frame, `--frames <scenario> <dir>
<n> [orbit°]` for a sweep — then `ffmpeg` + `img2webp` encode it:

```sh
beamer-viz --frames test_cases/11_one_hundred_thousand_users.txt /tmp/frames 60 16
ffmpeg -i /tmp/frames/frame_%05d.png -vf "scale=600:-1:flags=lanczos" /tmp/f/frame_%05d.png
img2webp -loop 0 -lossy -q 84 -m 6 -d 50 /tmp/f/frame_*.png -d 1500 /tmp/f/frame_00059.png -o docs/beamer.webp
```

---

Basemaps © OpenStreetMap contributors © CARTO · satellite imagery © Esri.
