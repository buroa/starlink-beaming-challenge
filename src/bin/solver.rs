//! Starlink beam-planning solver CLI: reads a scenario file and writes the
//! validator-format beam allocation (plus a near-optimality certificate header)
//! to stdout.
//!
//!   beam-planner <scenario.txt> [--max]
//!
//! `--max` selects the maximum-coverage algorithm: a much larger LNS budget on
//! residual-gap components — slower (seconds), recovers the last few users.

use beam_planner::{assign, feasibility, io};
use std::process::exit;
use web_time::{Duration, Instant};

/// Wall-clock ceiling for the (optional) repair phase — far under the 15-minute
/// limit; the greedy solution is always complete and valid before repair runs.
const REPAIR_BUDGET: Duration = Duration::from_secs(120);

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let intense = args.iter().any(|a| a == "--max");
    let path = args.iter().find(|a| !a.starts_with("--")).cloned().unwrap_or_else(|| {
        eprintln!("usage: beam-planner <scenario.txt> [--max]");
        exit(2);
    });

    let start = Instant::now();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read scenario '{path}': {e}");
        exit(1);
    });
    let scn = io::Scenario::parse(&text).unwrap_or_else(|e| {
        eprintln!("parse error: {e}");
        exit(1);
    });

    let feas = feasibility::build(&scn);
    let sol = assign::solve(&scn, &feas, start + REPAIR_BUDGET, intense);

    let cert = io::Certificate {
        total_users: scn.users.len(),
        feasible_users: feas.feasible_users,
        upper_bound: sol.upper_bound,
        colored_bound: sol.colored_bound,
        achieved: sol.achieved,
    };
    io::write_solution(std::io::stdout().lock(), &scn, &sol.per_sat, &cert)
        .expect("write solution");
}
