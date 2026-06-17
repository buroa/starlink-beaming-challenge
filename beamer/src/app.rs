//! Browser entry for the solver app: `solve_scenario` + the rayon Worker pool
//! init. The native CLI lives in `bin/main.rs`; this module is the wasm-bindgen
//! loader (gated to wasm32 + the `app` feature by `lib.rs`).

use crate::assign::{self, REPAIR_BUDGET};
use crate::{feasibility, io};
use wasm_bindgen::prelude::*;
use web_time::Instant;

/// Solve a scenario and return the validator-format solution (near-optimality
/// certificate header + beam allocation). `intense` selects maximum-coverage
/// (the CLI's `--max`). Parse/encode errors surface as a JavaScript exception.
#[wasm_bindgen]
pub fn solve_scenario(text: &str, intense: bool) -> Result<String, JsError> {
    let scn = io::Scenario::parse(text).map_err(|e| JsError::new(&e))?;
    let feas = feasibility::build(&scn);
    let sol = assign::solve(&scn, &feas, Instant::now() + REPAIR_BUDGET, intense);
    let cert = sol.certificate(&scn, &feas);
    let mut buf = Vec::new();
    io::write_solution(&mut buf, &scn, &sol.per_sat, &cert).map_err(|e| JsError::new(&e.to_string()))?;
    String::from_utf8(buf).map_err(|e| JsError::new(&e.to_string()))
}

/// `wasm-bindgen-rayon`'s `initThreadPool`; call once after `init()` with the
/// worker count. Only present in the multi-thread build (the `parallel` feature).
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;
