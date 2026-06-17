//! Build the sparse bipartite feasibility graph (which satellites each user may
//! legally be served by), applying the visibility and interferer filters. Both
//! filters depend only on the (user, sat) geometry, so this is computed once and
//! is embarrassingly parallel over users.

use crate::geom::{interferes, visible, visible_prefilter, Vec3};
use crate::index::Grid;
use crate::io::Scenario;
use crate::par::*;

#[cfg_attr(feature = "wire", derive(serde::Serialize, serde::Deserialize))]
pub struct Feasibility {
    /// `sats[u]` = sorted satellite indices user `u` may be served by.
    pub sats: Vec<Vec<u32>>,
    /// Number of users with at least one feasible satellite.
    pub feasible_users: usize,
}

pub fn build(scn: &Scenario) -> Feasibility {
    let grid = Grid::build(&scn.sats, &scn.users);
    // Precompute each user's local vertical (zenith = unit of its position) once.
    let zenith: Vec<Vec3> = scn.users.iter().map(|p| p.unit()).collect();

    let sats: Vec<Vec<u32>> = scn
        .users
        .par_iter()
        .enumerate()
        .map(|(ui, &up)| {
            let zu = zenith[ui];
            // Interferer directions are relative to this user.
            let intf: Vec<Vec3> = scn.interferers.iter().map(|ip| (*ip - up).unit()).collect();
            let mut out: Vec<u32> = Vec::new();
            grid.for_candidates(up, |si| {
                let delta = scn.sats[si as usize] - up;
                // Sqrt-free reject for the (majority) candidates that are nearby
                // but not overhead, before paying for the unit direction.
                if !visible_prefilter(zu, delta) {
                    return;
                }
                let dir_su = delta.unit();
                if !visible(zu, dir_su) {
                    return;
                }
                if intf.iter().any(|&d| interferes(dir_su, d)) {
                    return;
                }
                out.push(si);
            });
            // Deterministic order (grid traversal order is fixed, but sort to be
            // independent of cell layout).
            out.sort_unstable();
            out
        })
        .collect();

    let feasible_users = sats.iter().filter(|s| !s.is_empty()).count();
    Feasibility {
        sats,
        feasible_users,
    }
}
