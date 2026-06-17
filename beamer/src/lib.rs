//! Starlink beam-planning core: parsing, geometry, the spatial feasibility
//! graph, the exact max-flow upper bound, coloring-integral assignment, and an
//! instrumented [`trace`] solver that records the assignment step-by-step for
//! the visualizer.
//!
//! This crate is the solver core: the beam-planning library, used by the native
//! `beamer` CLI in `bin/` and by the `beamer-viz` crate (which embeds it as the
//! single browser front end). The visualizer lives in `../beamer-viz`.

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
