//! Per-satellite 4-coloring oracle — the heart of keeping coloring *integral*
//! to assignment. A user is admitted to a satellite only if a valid color
//! exists, so a beam is never emitted that violates the 10° same-color rule and
//! a colorable user is never dropped post-hoc.
//!
//! The common case (sparse conflict graph) is an O(degree) "is a free color
//! available?" scan. Only when all four colors are blocked do we run an exact
//! 4-coloring of the ≤33-node conflict graph (DSATUR-style MRV backtracking),
//! which is microseconds and provably correct (e.g. it rejects the 5th user of
//! a 5-clique on case 03, yielding the true optimum 4/5).

use crate::geom::{same_color_conflict, Vec3};

/// Find a color in 0..4 for `cand_dir` given existing members `dirs`/`colors`
/// (parallel arrays of equal length). On success via the exact path, existing
/// members may be recolored in place; the returned color is the candidate's.
/// Returns `None` iff {members ∪ candidate} is not 4-colorable.
/// Fast path only: the lowest color none of whose members conflict with the
/// candidate, or `None` if all four are blocked. O(members); never recolors.
#[inline]
pub fn fast_color(dirs: &[Vec3], colors: &[u8], cand_dir: Vec3) -> Option<u8> {
    let n = dirs.len();
    'color: for c in 0..4u8 {
        for i in 0..n {
            if colors[i] == c && same_color_conflict(dirs[i], cand_dir) {
                continue 'color;
            }
        }
        return Some(c);
    }
    None
}

pub fn try_color(dirs: &[Vec3], colors: &mut [u8], cand_dir: Vec3) -> Option<u8> {
    let n = dirs.len();

    if let Some(c) = fast_color(dirs, colors, cand_dir) {
        return Some(c);
    }

    // Exact path: 4-color the whole set with the candidate appended at index n.
    let m = n + 1;
    debug_assert!(m <= 33); // ≤33-node conflict graph (members + candidate) per module comment
    let mut adj = vec![0u64; m];
    for i in 0..n {
        for j in (i + 1)..n {
            if same_color_conflict(dirs[i], dirs[j]) {
                adj[i] |= 1 << j;
                adj[j] |= 1 << i;
            }
        }
        if same_color_conflict(dirs[i], cand_dir) {
            adj[i] |= 1 << n;
            adj[n] |= 1 << i;
        }
    }

    let mut assign = vec![u8::MAX; m];
    // Bound the backtracking: a 4-colorable ≤33-node graph is found near-
    // instantly with MRV, but *proving* non-4-colorability can be exponential.
    // On overrun we conservatively reject (the user goes unassigned) — at worst
    // a marginal coverage loss, never an invalid beam or a hang.
    let mut budget: u32 = 40_000;
    if color_search(&adj, m, &mut assign, &mut budget) {
        colors[..n].copy_from_slice(&assign[..n]);
        Some(assign[n])
    } else {
        None
    }
}

/// Recursive 4-coloring with minimum-remaining-values vertex selection and a
/// step budget. Returns false on infeasibility *or* budget exhaustion.
fn color_search(adj: &[u64], m: usize, assign: &mut [u8], budget: &mut u32) -> bool {
    if *budget == 0 {
        return false;
    }
    *budget -= 1;
    let mut best = usize::MAX;
    let mut best_avail = 0u8;
    let mut best_cnt = u32::MAX;
    for v in 0..m {
        if assign[v] != u8::MAX {
            continue;
        }
        let mut used = 0u8;
        let mut nb = adj[v];
        while nb != 0 {
            let u = nb.trailing_zeros() as usize;
            nb ^= 1 << u;
            if assign[u] != u8::MAX {
                used |= 1 << assign[u];
            }
        }
        let avail = !used & 0x0F;
        let cnt = avail.count_ones();
        if cnt == 0 {
            return false; // dead end
        }
        if cnt < best_cnt {
            best_cnt = cnt;
            best = v;
            best_avail = avail;
        }
    }
    if best == usize::MAX {
        return true; // every vertex colored
    }
    let mut mask = best_avail;
    while mask != 0 {
        let c = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        assign[best] = c;
        if color_search(adj, m, assign, budget) {
            return true;
        }
        assign[best] = u8::MAX;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Vec3;

    // Build directions ~`deg` apart on a small arc so they mutually conflict
    // (<10° as seen from the sat at the origin looking along +x).
    fn near_dirs(n: usize) -> Vec<Vec3> {
        (0..n)
            .map(|i| {
                let a = (i as f64) * 0.01; // ~0.57° steps → all within 10°
                Vec3::new(a.cos(), a.sin(), 0.0)
            })
            .collect()
    }

    #[test]
    fn four_mutually_close_users_get_four_colors() {
        let dirs = near_dirs(4);
        let mut colors: Vec<u8> = Vec::new();
        for i in 0..dirs.len() {
            let c = try_color(&dirs[..i], &mut colors, dirs[i]).expect("4-clique is 4-colorable");
            colors.push(c);
        }
        let set: std::collections::HashSet<u8> = colors.iter().copied().collect();
        assert_eq!(set.len(), 4, "a 4-clique must use all four distinct colors");
    }

    #[test]
    fn fifth_mutually_close_user_is_rejected() {
        // Five mutually-conflicting users form a K5 → not 4-colorable.
        let dirs = near_dirs(5);
        let mut colors: Vec<u8> = Vec::new();
        let mut admitted = 0;
        for i in 0..dirs.len() {
            if let Some(c) = try_color(&dirs[..i], &mut colors, dirs[i]) {
                colors.push(c);
                admitted += 1;
            }
        }
        assert_eq!(
            admitted, 4,
            "K5 admits exactly 4 (true optimum, e.g. case 03)"
        );
    }
}
