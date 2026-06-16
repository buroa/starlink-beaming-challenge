# Starlink Beam Planning

A solver for the SpaceX Starlink Beam Planning tech test (see the included PDF).
Given Starlink satellites, users, and non‑Starlink "interferer" satellites in
ECEF coordinates, it assigns beams (≤ 32 per satellite, one of 4 colors each) to
serve **the most users possible** without violating any constraint:

- **Visibility** — the serving satellite must be within 45° of the user's vertical.
- **Interferer** — from the user's view, the satellite must be ≥ 20° from every non‑Starlink satellite.
- **Coloring** — two beams of the same color on one satellite must be ≥ 10° apart.

This is a from‑scratch Rust rewrite (the original was a slow single‑threaded
Python greedy). It is fast, fully parallel, deterministic, and reports a
**provable near‑optimality certificate** with every solution.

## Build & run

Requires a recent Rust toolchain (and `python3` only for the official validator).

```sh
cargo build --release
./target/release/beam-planner test_cases/09_ten_thousand_users.txt   # solution → stdout
./run.sh                                                             # solve + validate every case
```

### Visualizer — Beamer (`beamer`)

A cinematic, GPU-rendered, interactive 3D visualizer — built into the app
(wgpu + egui), no browser or external tooling. Run it from the repo root:

```sh
cargo run --release --bin beamer
```

It opens **fullscreen**, framed on the United States, on the 100k-user case
(`11`), and starts playing the assignment immediately. The same **production
solver** drives it, so the coverage it reports is identical to the CLI
certificate — `Optimized · ensemble + repair` is the default algorithm (the
greedy/flow-seeded constructions remain selectable for comparison and show
*lower* coverage, as expected).

Rendering is **4× MSAA** with a starfield backdrop and beams as clean RGB
ribbons (A red, B green, C blue, D yellow). The **earth is transparent by
default** — pick a **basemap** (Dark / Light / Satellite) from the Map panel to
stream a live, level-of-detail globe on demand (nothing is pre-baked; zoom in
and higher-detail tiles load on background threads, with a Fresnel atmosphere
halo). You can **scroll all the way through the surface to the planet's core**
and view the whole beam network from the inside looking out.

The black/white/glass UI is a slim **top-left toolbar** (title + chips that open
the floating panels + **Hide**, also toggled with `H`), a compact **coverage**
readout (top-right), and a redesigned **transport** bar (bottom-center: restart,
play/pause, scrubber, and speed presets). The toolbar chips open four **movable**
glass cards — **Scene** (scenario + algorithm + rerun), **Bands** (RGB band +
layer toggles), **Map** (basemap selector), and **Unserved Terminals** (counts
grouped by *why* each terminal failed — no satellite in view, blocked by an
interferer, all satellites full, or no free color — with a list you can **click
to fly the camera to**). Drag any card to reposition it.

Hover any satellite or terminal for a tooltip (id, beams in use, band, or why it
couldn't be served). Drag to orbit, scroll to zoom (all the way to the core),
drag any panel to move it. **`H`** hides/shows the HUD, **`F11`** toggles
fullscreen, **`Esc`** leaves fullscreen.

Basemaps © OpenStreetMap contributors © CARTO; satellite imagery © Esri.
(`beamer --shot <scenario> <out.png> [fraction]` renders a single 3-D frame
headlessly to a PNG.)

## Algorithm

The 550 km constellation makes the feasibility graph extremely sparse (each user
sees only ~2–4 satellites), and it **splits into independent connected
components** that share no satellites. Each component is solved fully in
parallel:

1. **Spatial index** — a uniform 3D grid over satellites; a fixed‑radius ball
   query yields each user's candidate satellites, then exact visibility +
   interferer filters give the feasibility graph (parallel over users).
2. **Upper bounds** — two ceilings, both exact Dinic max‑flows per component:
   - a maximum **capacitated matching** (sat cap 32, *ignoring* color), and
   - a tighter **coloring‑aware** bound: per satellite, the feasible users' <10°
     conflict graph is partitioned into cliques, and a clique of `k` mutually
     conflicting users needs `k` colors, so at most 4 can be served → cap each
     satellite at `Σ min(4, |Cᵢ|)` (≤ 32). A flow under these caps is still a
     sound ceiling and is ≤ the matching bound (it correctly proves case 03's
     optimum is **4**, not 5).
3. **Ensemble construction** — four **coloring‑integral greedy** variants run in
   parallel per component, keeping the best (a user is admitted to a satellite
   only if a valid color exists — coloring is never a fragile post‑step). A fifth
   **flow‑seeded** build (realizes the optimal matching with a near‑maximum
   4‑colorable subset per satellite) runs only when the greedy ensemble fell
   short of the matching bound — it can't beat a bound the greedy build already
   reached, so the capacity‑saturated mega‑components skip its costly repair
   entirely. Per‑satellite 4‑coloring is solved exactly (DSATUR + bounded
   backtracking, with color‑symmetry breaking and a clique cutoff).
4. **Bounded augmenting repair** — recovers stragglers via short displacement
   chains, with atomic rollback; strictly budgeted so it can never blow up.
5. **Parallel large‑neighborhood search** — the best construction is then
   polished by many **independent** ruin‑and‑recreate searches launched in
   parallel (each tears down the satellites around a still‑unserved terminal and
   rebuilds that cluster), keeping the best. A **transactional undo** makes each
   round O(touched) instead of O(component), and work‑stealing keeps every core
   busy on the hard component. Iteration‑bounded, so it stays deterministic.

The solution is **valid by construction** (every assignment passes the coloring
oracle) and **deterministic** — no RNG, a fixed‑seed search, explicit
tie‑breaks, and no dependence on thread scheduling, so a given build is
byte‑identical run to run (the only cross‑build wobble is a handful of users
sitting exactly on the 10°/45° thresholds, where `f64` rounding can flip under a
different compile). The printed certificate header states how close to optimal
it is.

## Results

`achieved` = users covered; `bound` = the tighter (coloring‑aware) ceiling — no
valid solution can exceed it; `A/bound` = fraction of that ceiling reached.

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

We **match or beat** the old solver on every case and are now **provably
optimal** (achieved = the coloring‑aware bound) on **00–06 and 08** — the
clique cuts certify case 03's `4/5` as exactly optimal rather than asserting it.

On **07/09/10/11** the coloring bound coincides with the matching bound: there
the binding constraint is per‑satellite 32‑beam capacity, and the residual gap
is a *global* coloring interaction that per‑satellite clique cuts can't tighten.
We can't cheaply certify those past ~98.9–99.8%, but the achieved values are at
the practical optimum: exhaustive parallel ruin‑and‑recreate search (all cores)
converges there and recovers only a handful more users with vastly more compute.

### Performance

The full solve for 100,000 users / 1,440 satellites — construction, repair, and
the parallel large‑neighborhood polish — finishes in **~0.55 s** on all cores,
well under the 15 min / 1 GB limits. Every smaller case is sub‑second (the 10k
cases land at ~0.1–0.25 s). The exact 4‑coloring oracle dominates the hard
component, so it is the most tuned hot path: stack‑allocated search state,
incremental neighbour‑color counts, color‑symmetry breaking, and a K5 clique
cutoff together cut its work by ~16× and the whole 100k case by ~30× over the
first correct version, with no loss of coverage. The polish is still a single
speed↔quality knob (`LNS_MAX_ROUNDS`).

See the **Visualizer** section above to explore any scenario interactively.
