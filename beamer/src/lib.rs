//! Starlink beam-planning core: parsing, geometry, the spatial feasibility
//! graph, the exact max-flow upper bound, coloring-integral assignment, and an
//! instrumented [`trace`] solver that records the assignment step-by-step for
//! the visualizer.
//!
//! This crate is the solver: the `beam_planner` library (used by the native CLI
//! in `bin/`, and by the `viz` crate), plus the browser solver app's wasm-bindgen
//! loader (`app`, behind the `app` feature). The visualizer lives in `../viz`.

pub mod assign;
pub mod coloring;
pub mod components;
pub mod feasibility;
pub mod geom;
pub mod index;
pub mod io;
pub mod matching;
pub mod trace;

/// rayon-or-serial parallelism shim (see [`par`]); keeps the solver's hot-path
/// call sites identical whether or not the `parallel` feature is enabled.
pub(crate) mod par;

// The browser solver app's wasm-bindgen entry points (`solve_scenario` +
// `initThreadPool`). Behind the `app` feature so they aren't pulled into crates
// that depend on this one only for the core library (e.g. `viz`).
#[cfg(all(target_arch = "wasm32", feature = "app"))]
mod app;
