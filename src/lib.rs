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
