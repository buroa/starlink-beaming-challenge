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

/// Conflict-graph capacity for the exact path: members (< 32, since a full
/// satellite short-circuits before `try_color`) plus the candidate. Sizing the
/// working arrays to this lets the exact coloring run with zero heap allocation.
const MAX_M: usize = 33;

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
    debug_assert!(m <= MAX_M); // members + candidate
                               // Stack working set — no per-call heap allocation (this runs millions of
                               // times on dense satellites): adjacency bitsets, the partial coloring, and
                               // per-vertex neighbour-colour counts the search maintains incrementally.
    let mut adj = [0u64; MAX_M];
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

    // Cheap sufficient rejection: a clique of ≥5 mutually-conflicting users needs
    // ≥5 colours, so the set is not 4-colourable. `fast_color` just failed on the
    // candidate, so it sits in a dense neighbourhood — greedily grow a maximal
    // clique from it (lowest-index first); if it reaches 5 we reject without the
    // (budget-bounded, expensive) full search. Sound, and identical to the
    // search's verdict — a K5 is never 4-colourable; a missed clique merely falls
    // through to the search exactly as before.
    let mut cl = adj[n];
    let mut csz = 1u32;
    while cl != 0 {
        csz += 1;
        if csz >= 5 {
            return None;
        }
        cl &= adj[cl.trailing_zeros() as usize];
    }

    let mut assign = [u8::MAX; MAX_M];
    let mut nbcnt = [[0u8; 4]; MAX_M];
    // Bound the backtracking: a 4-colorable graph is found near-instantly with
    // MRV, but *proving* non-4-colorability can be exponential. On overrun we
    // conservatively reject (the user goes unassigned) — at worst a marginal
    // coverage loss, never an invalid beam or a hang.
    let mut budget: u32 = 40_000;
    let ok = color_search(&adj, m, &mut assign, &mut nbcnt, &mut budget, 0);
    if ok {
        colors[..n].copy_from_slice(&assign[..n]);
        Some(assign[n])
    } else {
        None
    }
}

/// Recursive 4-coloring with minimum-remaining-values vertex selection and a
/// step budget. Returns false on infeasibility *or* budget exhaustion.
///
/// `nbcnt[v][c]` tracks how many of `v`'s already-coloured neighbours use colour
/// `c`, maintained incrementally on assign/backtrack. This makes the per-node
/// MRV scan O(m·4) instead of rescanning every vertex's neighbour list — but the
/// vertex/colour selection order is byte-for-byte the same as the rescan
/// version (skip coloured, dead-end on zero options, fewest-options-first with
/// lowest-index tie-break, colours tried ascending), so the coloring found and
/// the accept/reject verdict are identical.
///
/// `used` is the bitmask of colours already placed on the current path. A proper
/// colouring is invariant under permuting colour *labels*, so when extending the
/// search we never need to try more than one *new* colour: the existing colours
/// `used` plus the single lowest unused one. `allowed = used | (used + 1)` is
/// exactly that prefix mask. This breaks the 4! colour symmetry — the dominant
/// cost when *proving* a dense cluster is not 4-colourable (~⅔ of all search
/// steps were symmetric re-explorations). It does **not** change the colouring
/// found or the accept/reject verdict: the lowest available colour at any vertex
/// is always ≤ (#distinct colours used) and so always inside `allowed`, so the
/// leftmost (lowest-colour-first) solution path is never pruned — only the
/// redundant higher-new-colour branches to its right, which a correct search
/// would reject anyway.
fn color_search(
    adj: &[u64],
    m: usize,
    assign: &mut [u8],
    nbcnt: &mut [[u8; 4]],
    budget: &mut u32,
    used: u8,
) -> bool {
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
        let k = &nbcnt[v];
        let avail = (k[0] == 0) as u8
            | (((k[1] == 0) as u8) << 1)
            | (((k[2] == 0) as u8) << 2)
            | (((k[3] == 0) as u8) << 3);
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
    // Try only existing colours plus the single lowest new one (symmetry break).
    let mut mask = best_avail & (used | used.wrapping_add(1));
    while mask != 0 {
        let c = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        assign[best] = c;
        let mut nb = adj[best];
        while nb != 0 {
            let u = nb.trailing_zeros() as usize;
            nb &= nb - 1;
            nbcnt[u][c as usize] += 1;
        }
        if color_search(adj, m, assign, nbcnt, budget, used | (1 << c)) {
            return true;
        }
        let mut nb = adj[best];
        while nb != 0 {
            let u = nb.trailing_zeros() as usize;
            nb &= nb - 1;
            nbcnt[u][c as usize] -= 1;
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
