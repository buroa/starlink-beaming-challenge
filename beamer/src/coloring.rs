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

/// Backtracking-node budget for the exact 4-coloring search. Finding a valid
/// 4-coloring of a ≤33-node conflict graph with MRV + colour-symmetry breaking
/// takes only a handful of nodes; spending more buys nothing but *proofs* of
/// non-4-colorability, which the K5 cutoff already catches cheaply and which —
/// on overrun — we conservatively reject anyway. 256 sits comfortably above the
/// point where any colorable instance is found (measured: coverage is unchanged
/// on every test case down to ~96) while keeping dense satellites cheap, where
/// an oversized budget was the dominant solver cost. On overrun we reject (the
/// user goes unassigned) — at worst a marginal coverage loss, never an invalid
/// beam or a hang.
const COLOR_SEARCH_BUDGET: u32 = 256;

/// Fast path: the lowest color none of whose members conflict with the candidate,
/// or `None` if all four are blocked. O(members); never recolors.
///
/// This stays first-fit on purpose: it is the coverage-defining hot path (its
/// accept/reject feeds construction, repair, and LNS), so it must be cheap and
/// stable. The resulting first-fit skew (color 0 dominates) is corrected purely
/// cosmetically, after the assignment is final, by [`rebalance`].
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
    let mut budget = COLOR_SEARCH_BUDGET;
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

/// Cosmetic, coverage-neutral rebalancing of one satellite's final beams.
///
/// `dirs`/`colors` are a satellite's served beams and their (first-fit, color-0
/// heavy) coloring. This returns a *new* proper 4-coloring of the same beams that
/// levels the four color classes as far as the <10° conflicts allow — DSATUR order
/// (most-saturated beam first) with a least-loaded color choice. It only relabels
/// an already-fixed set, so coverage is untouched; if the heuristic ever gets
/// stuck (a 4-colorable graph can defeat a greedy order) it returns the original
/// coloring unchanged, so the result is always valid.
///
/// `phase` (e.g. the satellite's id) rotates the tie-break among equally-loaded
/// colors, so each satellite's first beam doesn't always land on color 0 —
/// without it, the swarm of low-load satellites re-skews the global mix toward
/// red. It only affects ties, so coverage and per-satellite balance are unchanged.
pub fn rebalance(dirs: &[Vec3], colors: &[u8], phase: usize) -> Vec<u8> {
    let n = dirs.len();
    if n <= 1 {
        // Even a lone beam rotates, so single-beam satellites span all four bands.
        return vec![(phase % 4) as u8; n];
    }
    // Conflict adjacency (same-satellite beams within 10°) + degrees. n ≤ 32, so a
    // 64-bit neighbour bitset per beam is ample.
    let mut adj = vec![0u64; n];
    let mut deg = vec![0u32; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if same_color_conflict(dirs[i], dirs[j]) {
                adj[i] |= 1 << j;
                adj[j] |= 1 << i;
                deg[i] += 1;
                deg[j] += 1;
            }
        }
    }

    let mut out = vec![u8::MAX; n];
    let mut class = [0u32; 4]; // current size of each color class
    for _ in 0..n {
        // DSATUR: color the uncolored beam whose neighbours already use the most
        // distinct colors (tie → highest degree → lowest index).
        let mut best = usize::MAX;
        let (mut best_sat, mut best_deg, mut best_used) = (0u32, 0u32, 0u8);
        for v in 0..n {
            if out[v] != u8::MAX {
                continue;
            }
            let mut used = 0u8;
            let mut nb = adj[v];
            while nb != 0 {
                let u = nb.trailing_zeros() as usize;
                nb &= nb - 1;
                if out[u] != u8::MAX {
                    used |= 1 << out[u];
                }
            }
            let sat = used.count_ones();
            if best == usize::MAX || sat > best_sat || (sat == best_sat && deg[v] > best_deg) {
                best = v;
                best_sat = sat;
                best_deg = deg[v];
                best_used = used;
            }
        }
        match (0..4u8)
            .filter(|&c| best_used & (1 << c) == 0)
            .min_by_key(|&c| (class[c as usize], (c as usize + phase) % 4))
        {
            Some(c) => {
                out[best] = c;
                class[c as usize] += 1;
            }
            // Defeated greedily — keep the known-valid input coloring for this sat.
            None => return colors.to_vec(),
        }
    }
    out
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
