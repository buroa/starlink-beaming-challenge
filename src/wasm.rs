//! Browser entry points, compiled only for `wasm32`.
//!
//! Exposes the production solver as a `wasm-bindgen` export and — when built
//! with the `parallel` feature — the `initThreadPool` binding that
//! `wasm-bindgen-rayon` needs to bring up the Web Worker pool.
//!
//! Build (threaded; needs nightly + `build-std` + cross-origin isolation):
//! ```sh
//! RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
//!   cargo +nightly build --release --lib --no-default-features --features parallel \
//!   --target wasm32-unknown-unknown -Z build-std=std,panic_abort
//! ```
//! Build (serial; stable, any static host):
//! ```sh
//! cargo build --release --lib --no-default-features --target wasm32-unknown-unknown
//! ```
//! Then run `wasm-bindgen` on the resulting `.wasm`. From JS:
//! ```js
//! import init, { solve_scenario, initThreadPool } from './pkg/beam_planner.js';
//! await init();
//! await initThreadPool(navigator.hardwareConcurrency); // threaded build only
//! const solution = solve_scenario(scenarioText, false);
//! ```

use crate::{assign, feasibility, io};
use wasm_bindgen::prelude::*;
use web_time::{Duration, Instant};

/// Repair/LNS wall-clock ceiling. A one-shot browser solve isn't racing the
/// 15-minute grader limit, but the solver still wants a deadline to bound its
/// repair and large-neighborhood-search phases — mirror the CLI's budget.
const REPAIR_BUDGET: Duration = Duration::from_secs(120);

/// Solve a scenario and return the validator-format solution (near-optimality
/// certificate header + beam allocation) as a string. `intense` selects the
/// maximum-coverage mode (the CLI's `--max`). Parse/encode failures surface as a
/// JavaScript exception.
#[wasm_bindgen]
pub fn solve_scenario(text: &str, intense: bool) -> Result<String, JsError> {
    let scn = io::Scenario::parse(text).map_err(|e| JsError::new(&e))?;
    let feas = feasibility::build(&scn);
    let sol = assign::solve(&scn, &feas, Instant::now() + REPAIR_BUDGET, intense);
    let cert = io::Certificate {
        total_users: scn.users.len(),
        feasible_users: feas.feasible_users,
        upper_bound: sol.upper_bound,
        colored_bound: sol.colored_bound,
        achieved: sol.achieved,
    };
    let mut buf = Vec::new();
    io::write_solution(&mut buf, &scn, &sol.per_sat, &cert)
        .map_err(|e| JsError::new(&e.to_string()))?;
    String::from_utf8(buf).map_err(|e| JsError::new(&e.to_string()))
}

/// Re-export so `wasm-bindgen` emits the `initThreadPool` JS binding. Call it
/// once, after `init()`, with the desired worker count (typically
/// `navigator.hardwareConcurrency`). Only present in threaded builds.
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;
