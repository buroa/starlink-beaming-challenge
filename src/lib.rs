//! Starlink beam-planning core: parsing, geometry, the spatial feasibility
//! graph, the exact max-flow upper bound, coloring-integral assignment, and an
//! instrumented [`trace`] solver that records the assignment step-by-step for
//! the visualizer.

pub mod assign;
pub mod bound;
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

/// The GPU visualizer ("Beamer"), behind the `viz` feature. Lives in the library
/// (not a `[[bin]]`) so it can be emitted as a wasm-bindgen `cdylib` for the
/// browser; the native `beamer` binary is a thin shim over [`viz::run`].
#[cfg(feature = "viz")]
pub mod viz;

/// Browser entry points (`wasm-bindgen` solver export + Web Worker thread-pool
/// init). Compiled only for `wasm32`. Public so the re-exported `initThreadPool`
/// binding is reachable and reliably linked into the generated wasm.
#[cfg(target_arch = "wasm32")]
pub mod wasm;
