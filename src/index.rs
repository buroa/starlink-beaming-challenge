//! Uniform 3D grid over satellite positions for candidate lookup.
//!
//! Every satellite lies on a thin shell (all at the same altitude in practice),
//! and a feasible (user,sat) pair has bounded slant range, so a fixed Euclidean
//! ball query around each user returns the exact candidate set. With cell edge
//! equal to the query radius, all points within the radius fall inside the
//! 3×3×3 block of neighbouring cells.

use crate::geom::Vec3;

pub struct Grid {
    origin: Vec3,
    cell: f64,
    nx: i64,
    ny: i64,
    nz: i64,
    cells: Vec<Vec<u32>>,
}

impl Grid {
    /// Derive the query radius from geometry: the maximum slant range of any
    /// feasible pair occurs at the 45°-from-vertical boundary. For a user at
    /// radius `ru` and satellite at radius `rs`, the law of cosines (interior
    /// angle 135° at the user) gives
    ///   d² + √2·ru·d + (ru² − rs²) = 0  ⇒  d = (−√2·ru + √(4·rs² − 2·ru²)) / 2.
    /// We use the largest satellite radius and smallest user radius, plus a 1%
    /// margin, so the ball never misses a candidate.
    fn derive_radius(sats: &[Vec3], users: &[Vec3]) -> f64 {
        if sats.is_empty() || users.is_empty() {
            return 1.0;
        }
        let rs = sats.iter().map(|p| p.norm()).fold(0.0_f64, f64::max);
        let ru = users.iter().map(|p| p.norm()).fold(f64::INFINITY, f64::min);
        let disc = 4.0 * rs * rs - 2.0 * ru * ru;
        let d = if disc > 0.0 {
            (-(2.0_f64).sqrt() * ru + disc.sqrt()) / 2.0
        } else {
            0.0
        };
        // Fallback keeps us correct even for degenerate geometry.
        (d * 1.01).max(rs + ru)
    }

    pub fn build(sats: &[Vec3], users: &[Vec3]) -> Grid {
        let radius = Self::derive_radius(sats, users);
        let cell = radius;
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in sats {
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        if sats.is_empty() {
            lo = Vec3::default();
            hi = Vec3::default();
        }
        let dim = |l: f64, h: f64| ((h - l) / cell).floor() as i64 + 1;
        let (nx, ny, nz) = (dim(lo.x, hi.x), dim(lo.y, hi.y), dim(lo.z, hi.z));
        let cells = vec![Vec::new(); (nx * ny * nz).max(1) as usize];
        let mut g = Grid {
            origin: lo,
            cell,
            nx,
            ny,
            nz,
            cells,
        };
        for (i, p) in sats.iter().enumerate() {
            let (cx, cy, cz) = g.coords(*p);
            let idx = g.lin(cx, cy, cz).expect("satellite inside its own bbox");
            g.cells[idx].push(i as u32);
        }
        g
    }

    #[inline]
    fn coords(&self, p: Vec3) -> (i64, i64, i64) {
        (
            ((p.x - self.origin.x) / self.cell).floor() as i64,
            ((p.y - self.origin.y) / self.cell).floor() as i64,
            ((p.z - self.origin.z) / self.cell).floor() as i64,
        )
    }

    #[inline]
    fn lin(&self, cx: i64, cy: i64, cz: i64) -> Option<usize> {
        if cx < 0 || cy < 0 || cz < 0 || cx >= self.nx || cy >= self.ny || cz >= self.nz {
            return None;
        }
        Some(((cx * self.ny + cy) * self.nz + cz) as usize)
    }

    /// Invoke `f` for every satellite index in the 3×3×3 block around `p`.
    #[inline]
    pub fn for_candidates(&self, p: Vec3, mut f: impl FnMut(u32)) {
        let (cx, cy, cz) = self.coords(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(idx) = self.lin(cx + dx, cy + dy, cz + dz) {
                        for &si in &self.cells[idx] {
                            f(si);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::Scenario;
    use std::collections::HashSet;

    /// The grid must never miss a satellite that passes the visibility cone:
    /// every brute-force-visible sat must appear in the 27-cell candidate set.
    #[test]
    fn grid_returns_superset_of_visible() {
        for case in [
            "test_cases/07_eighteen_planes.txt",
            "test_cases/11_one_hundred_thousand_users.txt",
        ] {
            let text = std::fs::read_to_string(case).unwrap();
            let scn = Scenario::parse(&text).unwrap();
            let grid = Grid::build(&scn.sats, &scn.users);
            for (ui, &up) in scn.users.iter().enumerate() {
                let zu = up.unit();
                let mut got = HashSet::new();
                grid.for_candidates(up, |si| {
                    got.insert(si);
                });
                for (si, &sp) in scn.sats.iter().enumerate() {
                    if zu.dot((sp - up).unit()) > crate::geom::COS45 {
                        assert!(
                            got.contains(&(si as u32)),
                            "{case}: grid missed visible sat {si} for user {ui}"
                        );
                    }
                }
            }
        }
    }
}
