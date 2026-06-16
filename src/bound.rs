//! Lagrangian-decomposition upper bound (currently unused but sound).
//!
//! **Status**: This module provides a theoretically elegant bound that is never
//! empirically tighter than coloring_bound (see assign.rs lines 760-765).
//! It is kept as a documented reference for future improvements.
//!
//! The matching bound ignores coloring; the per-satellite clique bound ignores
//! the *global* single-coverage coupling. Neither tightens the hard cases,
//! because the gap is the **inconsistency** between "a maximum matching" and "a
//! per-satellite 4-colorable assignment": both are individually satisfiable, but
//! not always by the *same* assignment.
//!
//! Lagrangian decomposition (a.k.a. variable splitting) prices exactly that
//! inconsistency. Split the assignment into two copies — `x` constrained to be a
//! capacitated matching (single coverage, ≤32/sat), `y` constrained to be
//! per-satellite 4-colorable — and relax the linking constraint `x = y` with
//! multipliers `μ` (one per feasible user→sat edge):
//!
//! ```text
//!   LD(μ) = max_{x∈Matching} Σ x_e (1 + μ_e)   +   max_{y∈Colorable} Σ y_e (−μ_e)
//!         = f1(μ)                               +   f2(μ)
//! ```
//!
//! `LD(μ) ≥ optimum` for every `μ`, and `LD(0) = matching bound`. A subgradient
//! descent on `μ` (driven by the disagreement `x_e − y_e`) drives the bound
//! *below* the matching bound, toward the true optimum. `f1` is an exact
//! max-weight assignment (min-cost flow); `f2` decomposes per satellite and is
//! upper-bounded by the weighted clique cut (kept ≥ the true value so `LD` stays
//! a sound bound).

use crate::components::Component;
use crate::geom::{same_color_conflict, Vec3};
use crate::io::Scenario;
use std::time::Instant;

/// Components larger than this are left to the matching bound — a min-cost flow
/// per subgradient iteration is too costly at, e.g., case 11's ~17k-user
/// component. (The moderate cases' components fit comfortably.)
const LD_MAX_USERS: usize = 4096;
const LD_ITERS: u32 = 400;

/// Min-cost max-flow specialized to **max-weight assignment**: push flow only
/// along profitable (negative-cost) augmenting paths, so unprofitable users stay
/// unassigned. Dijkstra with Johnson potentials; the graph is rebuilt cheaply
/// each subgradient iteration (only edge costs change).
struct Mcmf {
    n: usize,
    to: Vec<u32>,
    cap: Vec<i32>,
    cost: Vec<f64>,
    head: Vec<Vec<u32>>,
}

impl Mcmf {
    fn new(n: usize) -> Self {
        Mcmf {
            n,
            to: Vec::new(),
            cap: Vec::new(),
            cost: Vec::new(),
            head: vec![Vec::new(); n],
        }
    }
    fn add(&mut self, u: usize, v: usize, cap: i32, cost: f64) -> usize {
        let e = self.to.len();
        self.to.push(v as u32);
        self.cap.push(cap);
        self.cost.push(cost);
        self.head[u].push(e as u32);
        self.to.push(u as u32);
        self.cap.push(0);
        self.cost.push(-cost);
        self.head[v].push((e + 1) as u32);
        e
    }

    /// Run to the min-cost flow (any value) — i.e. augment while the cheapest
    /// src→sink path has negative cost. Returns the total assigned weight
    /// (= −total cost). Flow is left in the residual caps for recovery.
    fn solve(&mut self, src: usize, sink: usize) -> f64 {
        const INF: f64 = f64::INFINITY;
        // Johnson potentials via Bellman-Ford (costs may be negative initially).
        let mut h = vec![INF; self.n];
        h[src] = 0.0;
        // Bellman-Ford (graph is a small DAG-ish bipartite layer; |V| passes).
        for _ in 0..self.n {
            let mut changed = false;
            for u in 0..self.n {
                if h[u] == INF {
                    continue;
                }
                for &e in &self.head[u] {
                    let e = e as usize;
                    if self.cap[e] > 0 {
                        let v = self.to[e] as usize;
                        let nd = h[u] + self.cost[e];
                        if nd < h[v] - 1e-12 {
                            h[v] = nd;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        let mut total = 0.0;
        loop {
            // Dijkstra on reduced costs.
            let mut dist = vec![INF; self.n];
            let mut pe = vec![u32::MAX; self.n]; // edge used to reach node
            dist[src] = 0.0;
            // Binary heap of (dist, node); use a simple Vec-based heap.
            let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(OrdF, usize)>> =
                std::collections::BinaryHeap::new();
            heap.push(std::cmp::Reverse((OrdF(0.0), src)));
            while let Some(std::cmp::Reverse((OrdF(d), u))) = heap.pop() {
                if d > dist[u] + 1e-12 {
                    continue;
                }
                for &e in &self.head[u] {
                    let e = e as usize;
                    if self.cap[e] <= 0 {
                        continue;
                    }
                    let v = self.to[e] as usize;
                    if h[u] == INF || h[v] == INF {
                        continue;
                    }
                    let rc = self.cost[e] + h[u] - h[v]; // reduced cost ≥ 0
                    let nd = d + rc;
                    if nd < dist[v] - 1e-12 {
                        dist[v] = nd;
                        pe[v] = e as u32;
                        heap.push(std::cmp::Reverse((OrdF(nd), v)));
                    }
                }
            }
            if dist[sink] == INF {
                break; // sink unreachable
            }
            // Update potentials.
            for v in 0..self.n {
                if dist[v] < INF {
                    h[v] += dist[v];
                }
            }
            // Actual cost of the augmenting path = h[sink] − h[src] (h[src]=0).
            let path_cost = h[sink];
            if path_cost >= -1e-9 {
                break; // no profitable augmentation remains
            }
            // Augment one unit along the path (all relevant caps are 1 or 32).
            let mut v = sink;
            while v != src {
                let e = pe[v] as usize;
                self.cap[e] -= 1;
                self.cap[e ^ 1] += 1;
                v = self.to[e ^ 1] as usize;
            }
            total += -path_cost;
        }
        total
    }
}

/// Total-order wrapper for f64 in the heap (values are finite here).
#[derive(Clone, Copy, PartialEq)]
struct OrdF(f64);
impl Eq for OrdF {}
impl PartialOrd for OrdF {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&o.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// One satellite's coloring structure for `f2`: its incident edges and a fixed
/// clique partition of the <10° conflict graph (cliques don't depend on `μ`).
struct SatColoring {
    /// (edge id, local user) incident to this satellite.
    edges: Vec<(u32, u32)>,
    /// Clique partition as lists of indices into `edges`.
    cliques: Vec<Vec<u32>>,
}

/// Lagrangian-decomposition upper bound for one component. Returns a value
/// `≤ matching_ub` that is still a sound ceiling on the component optimum.
/// Falls back to `matching_ub` for components too large to flow per iteration.
#[allow(dead_code)]
pub fn ld_bound(
    scn: &Scenario,
    c: &Component,
    local_feas: &[Vec<u32>],
    ns: usize,
    matching_ub: usize,
    deadline: Instant,
) -> usize {
    let nusers = local_feas.len();
    if nusers == 0 || nusers > LD_MAX_USERS {
        return matching_ub;
    }

    // Edge indexing: edge ids are assigned per user in feasibility order.
    let mut edge_off = vec![0u32; nusers + 1];
    for u in 0..nusers {
        edge_off[u + 1] = edge_off[u] + local_feas[u].len() as u32;
    }
    let n_edges = edge_off[nusers] as usize;
    if n_edges == 0 {
        return matching_ub;
    }
    // edge -> (user, sat)
    let mut edge_user = vec![0u32; n_edges];
    let mut edge_sat = vec![0u32; n_edges];
    for u in 0..nusers {
        for (k, &s) in local_feas[u].iter().enumerate() {
            let e = edge_off[u] as usize + k;
            edge_user[e] = u as u32;
            edge_sat[e] = s;
        }
    }

    // Per-satellite coloring structure (clique partition over its users).
    let mut sat_color: Vec<SatColoring> = (0..ns)
        .map(|_| SatColoring {
            edges: Vec::new(),
            cliques: Vec::new(),
        })
        .collect();
    for e in 0..n_edges {
        sat_color[edge_sat[e] as usize]
            .edges
            .push((e as u32, edge_user[e]));
    }
    for (s, sc) in sat_color.iter_mut().enumerate().take(ns) {
        let m = sc.edges.len();
        if m == 0 {
            continue;
        }
        // Directions from sat s to each incident user.
        let satpos = scn.sats[c.sats[s] as usize];
        let dirs: Vec<Vec3> = sc
            .edges
            .iter()
            .map(|&(_, lu)| (scn.users[c.users[lu as usize] as usize] - satpos).unit())
            .collect();
        // Greedy clique partition (indices into sc.edges).
        let mut covered = vec![false; m];
        let mut clique: Vec<u32> = Vec::new();
        for i in 0..m {
            if covered[i] {
                continue;
            }
            covered[i] = true;
            clique.clear();
            clique.push(i as u32);
            for j in (i + 1)..m {
                if !covered[j]
                    && clique
                        .iter()
                        .all(|&k| same_color_conflict(dirs[k as usize], dirs[j]))
                {
                    clique.push(j as u32);
                    covered[j] = true;
                }
            }
            sc.cliques.push(clique.clone());
        }
    }

    // Subgradient descent on μ.
    let mut mu = vec![0f64; n_edges];
    let mut best = matching_ub;
    let mut x_sel = vec![false; n_edges]; // f1 selection
    let mut y_sel = vec![false; n_edges]; // f2 selection
    let src = nusers + ns;
    let sink = nusers + ns + 1;
    let node_n = nusers + ns + 2;

    for iter in 0..LD_ITERS {
        if iter % 16 == 0 && Instant::now() >= deadline {
            break;
        }
        // ---- f1: max-weight assignment with weight w_e = 1 + μ_e ----
        let mut g = Mcmf::new(node_n);
        let mut eid = vec![usize::MAX; n_edges];
        for u in 0..nusers {
            g.add(src, u, 1, 0.0);
        }
        for s in 0..ns {
            g.add(nusers + s, sink, 32, 0.0);
        }
        for e in 0..n_edges {
            let w = 1.0 + mu[e];
            // cost = −weight (min-cost flow maximizes weight via negative costs)
            eid[e] = g.add(edge_user[e] as usize, nusers + edge_sat[e] as usize, 1, -w);
        }
        let f1 = g.solve(src, sink);
        // Recover x: a forward user→sat edge with zero residual carried flow.
        x_sel.iter_mut().for_each(|b| *b = false);
        for e in 0..n_edges {
            if g.cap[eid[e]] == 0 {
                x_sel[e] = true;
            }
        }

        // ---- f2: per-sat max-weight 4-colorable subset, weight −μ_e ----
        // Upper-bounded by the weighted clique cut: ≤4 users per clique, ≤32 total.
        y_sel.iter_mut().for_each(|b| *b = false);
        let mut f2 = 0.0;
        for sc in sat_color.iter().take(ns) {
            if sc.edges.is_empty() {
                continue;
            }
            // Per clique, take up to 4 highest positive weights; then cap the
            // satellite at its 32 highest of those.
            let mut picked: Vec<(f64, u32)> = Vec::new(); // (weight, edge id)
            for clq in &sc.cliques {
                let mut wv: Vec<(f64, u32)> = clq
                    .iter()
                    .map(|&idx| {
                        let (e, _) = sc.edges[idx as usize];
                        (-mu[e as usize], e)
                    })
                    .filter(|&(w, _)| w > 0.0)
                    .collect();
                wv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                for &item in wv.iter().take(4) {
                    picked.push(item);
                }
            }
            if picked.len() > 32 {
                picked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                picked.truncate(32);
            }
            for (w, e) in picked {
                f2 += w;
                y_sel[e as usize] = true;
            }
        }

        // ---- bound + subgradient step ----
        // LD(μ) ≥ the integer optimum, so floor(LD) is a valid integer ceiling.
        let ld = f1 + f2;
        let bound_here = (ld + 1e-9).floor() as usize;
        if bound_here < best {
            best = bound_here;
        }
        // Subgradient: disagreement x_e − y_e. Step on a diminishing schedule.
        let step = 1.0 / (1.0 + 0.05 * iter as f64);
        let mut any = false;
        for e in 0..n_edges {
            let gsub = (x_sel[e] as i32) - (y_sel[e] as i32);
            if gsub != 0 {
                mu[e] -= step * gsub as f64;
                any = true;
            }
        }
        if !any {
            break; // x == y everywhere: bound equals optimum for this μ
        }
    }

    best
}
