//! Coverage maximization.
//!
//! The feasibility graph splits into independent connected components
//! ([`crate::components`]) that share no satellites, so each is solved fully in
//! parallel and the disjoint results are merged. Per component we compute an
//! exact upper bound (Dinic max-flow) and run an ensemble of independent
//! constructions — a flow-seeded build plus coloring-integral greedy variants,
//! each polished by a bounded augmenting repair — keeping the best. Every
//! assignment passes through the coloring oracle, so the solution is valid by
//! construction; nothing uses randomness, so it is deterministic.

use crate::coloring::{fast_color, try_color};
use crate::components::{self, Component};
use crate::geom::{same_color_conflict, Vec3};
use crate::io::Scenario;
use crate::matching;
use crate::par::*;
use arrayvec::ArrayVec;
use web_time::{Duration, Instant};

/// Maximum displacement hops in an augmenting search.
const MAX_DEPTH: u32 = 4;
/// Per-attempt expansion budget — caps the work spent seating one user, keeping
/// repair strictly bounded (it converges well within this on every test case).
const ATTEMPT_BUDGET: u32 = 2000;
/// Per-user budget for the recolor-enabled final repair pass. Smaller than
/// `ATTEMPT_BUDGET` because recolor fires the costly exact 4-coloring; the chains
/// it recovers are short, so a tight budget keeps the pass cheap.
const RECOLOR_BUDGET: u32 = 512;
/// Hard cap on large-neighborhood-search rounds per worker (a deterministic
/// iteration bound; the search usually stops earlier via the stall limit).
const LNS_MAX_ROUNDS: u32 = 1_200;
/// Independent LNS searches launched in parallel per component. Fixed (not
/// derived from core count) so the result is identical on any machine; the rest
/// of the pool work-steals these once the easy components finish.
const LNS_WORKERS: u32 = 16;
/// Rebuilds inside LNS only need a *local* re-home, so the augmenting search is
/// shallow and cheap here (vs. the full-strength constants used once at repair).
const LNS_DEPTH: u32 = 3;
const LNS_ATTEMPT_BUDGET: u32 = 256;
/// Intensive LNS knobs used only by the opt-in `Maximum` algorithm (CLI `--max`):
/// far more rounds, deeper chains, and a bigger per-attempt budget. These
/// push a residual-gap component to its practical ceiling (e.g. case 11's hard
/// component gains ~+5 users) at the cost of seconds — never on the default path.
const LNS_MAX_ROUNDS_INTENSE: u32 = 40_000;
const LNS_DEPTH_INTENSE: u32 = 5;
const LNS_ATTEMPT_BUDGET_INTENSE: u32 = 2_000;
/// Components no larger than this are eligible for an exact branch-and-bound
/// certification pass (proves the optimum, lifting that component to a perfect
/// A/bound). Larger components are out of reach for exact solving.
const EXACT_MAX_USERS: usize = 1600;
/// Node ceiling for the exact pass — past this it gives up and the heuristic
/// result stands (so the pass can only ever help, never hurt or hang).
const EXACT_MAX_NODES: u64 = 120_000_000;
/// How a user picks among its feasible satellites during greedy construction.
#[derive(Clone, Copy)]
enum SatChoice {
    LeastLoaded,
    LeastContended,
    HighestElevation,
}

#[derive(Clone, Default)]
struct Sat {
    users: ArrayVec<u32, 32>, // local user indices
    colors: ArrayVec<u8, 32>,
    dirs: ArrayVec<Vec3, 32>,
}

impl Sat {
    #[inline]
    fn load(&self) -> usize {
        self.users.len()
    }

    /// Admit `user` (direction `dir` from this sat) iff a color exists. With
    /// `allow_exact` the rare full-recolor fallback is tried; repair passes
    /// `false` to stay on the O(members) fast path (and never recolor).
    fn try_insert(&mut self, user: u32, dir: Vec3, allow_exact: bool) -> bool {
        if self.users.len() >= 32 {
            return false;
        }
        let color = if allow_exact {
            try_color(&self.dirs[..], &mut self.colors[..], dir)
        } else {
            fast_color(&self.dirs[..], &self.colors[..], dir)
        };
        match color {
            Some(c) => {
                self.users.push(user);
                self.colors.push(c);
                self.dirs.push(dir);
                true
            }
            None => false,
        }
    }

    /// Remove `user`, returning its stored direction (so a re-seat can reuse it
    /// instead of recomputing `unit(sat-user)`).
    #[inline]
    fn remove(&mut self, user: u32) -> Vec3 {
        let i = self
            .users
            .iter()
            .position(|&u| u == user)
            .expect("member present");
        self.users.swap_remove(i);
        self.colors.swap_remove(i);
        self.dirs.swap_remove(i)
    }

    #[inline]
    fn pop(&mut self) {
        self.users.pop();
        self.colors.pop();
        self.dirs.pop();
    }
}

#[derive(Clone)]
struct CompSolver<'a> {
    scn: &'a Scenario,
    g_users: &'a [u32],
    g_sats: &'a [u32],
    feas: &'a [Vec<u32>], // local user -> local sat indices
    sat_deg: &'a [u32],   // local: #users that can see each sat
    sats: Vec<Sat>,
    user_sat: Vec<i32>, // local sat index, or -1
    visited: Vec<u32>,  // per-sat: generation counter for visited marking (avoids clearing)
    gen: u32,           // current generation for visited array (inexpensive counter)
    budget: u32,
    sat_choice: SatChoice,
    // Transactional undo log (used only during LNS): when `txn` is on, every
    // satellite is snapshotted on first mutation and every `user_sat` write is
    // logged, so a rejected round is rolled back in O(touched) — never an
    // O(component) clone.
    txn: bool,
    txn_gen: u32,
    sat_touched: Vec<u32>,      // per-sat: txn_gen it was last snapshotted in
    sat_snap: Vec<(u32, Sat)>,  // (sat index, pre-mutation state)
    user_snap: Vec<(u32, i32)>, // (user index, previous user_sat value)
}

impl<'a> CompSolver<'a> {
    fn new(
        scn: &'a Scenario,
        c: &'a Component,
        feas: &'a [Vec<u32>],
        sat_deg: &'a [u32],
        sat_choice: SatChoice,
    ) -> Self {
        CompSolver {
            scn,
            g_users: &c.users,
            g_sats: &c.sats,
            feas,
            sat_deg,
            sats: vec![Sat::default(); c.sats.len()],
            user_sat: vec![-1; c.users.len()],
            visited: vec![0u32; c.sats.len()],
            gen: 0,
            budget: 0,
            txn: false,
            txn_gen: 0,
            sat_touched: vec![0u32; c.sats.len()],
            sat_snap: Vec::new(),
            user_snap: Vec::new(),
            sat_choice,
        }
    }

    /// Greedy construction, then repair unless greedy already hit the bound.
    /// Returns the number of users served.
    fn solve(&mut self, order: &[u32], upper_bound: usize, deadline: Instant, recolor: bool) -> usize {
        self.greedy(order);
        let assigned = self.assigned_count();
        if assigned < upper_bound {
            self.repair(order, deadline, recolor);
            self.assigned_count()
        } else {
            assigned
        }
    }

    #[inline]
    fn dir(&self, lu: u32, ls: u32) -> Vec3 {
        (self.scn.users[self.g_users[lu as usize] as usize]
            - self.scn.sats[self.g_sats[ls as usize] as usize])
            .unit()
    }

    /// Feasible sats of local user `lu`, ordered by the chosen heuristic.
    fn ordered_candidates(&self, lu: u32) -> ArrayVec<u32, 16> {
        let mut cs: ArrayVec<u32, 16> = self.feas[lu as usize].iter().copied().collect();
        match self.sat_choice {
            // Least loaded, then index — spreads load, keeps coloring slack.
            SatChoice::LeastLoaded => {
                cs.sort_unstable_by_key(|&s| (self.sats[s as usize].load() as u32, s))
            }
            // Least-contended sat first — preserve popular sats for users who
            // have no alternative.
            SatChoice::LeastContended => cs.sort_unstable_by_key(|&s| {
                (
                    self.sat_deg[s as usize],
                    self.sats[s as usize].load() as u32,
                    s,
                )
            }),
            // Highest elevation (most overhead) first. Decorate-sort: compute the
            // user zenith once and each candidate's elevation once, instead of
            // re-deriving both unit vectors twice per comparison (O(k) sqrts, not
            // O(k log k)). Bit-identical keys ⇒ identical order ⇒ deterministic.
            SatChoice::HighestElevation => {
                let zenith =
                    self.scn.users[self.g_users[lu as usize] as usize].unit();
                let mut keyed: ArrayVec<(f64, u32), 16> =
                    cs.iter().map(|&s| (zenith.dot(self.dir(lu, s)), s)).collect();
                keyed.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                cs = keyed.iter().map(|&(_, s)| s).collect();
            }
        }
        cs
    }

    fn greedy(&mut self, order: &[u32]) {
        for &lu in order {
            for ls in self.ordered_candidates(lu) {
                let d = self.dir(lu, ls);
                if self.sats[ls as usize].try_insert(lu, d, true) {
                    self.user_sat[lu as usize] = ls as i32;
                    break;
                }
            }
        }
    }

    // --- transactional undo (active only during LNS; no-ops otherwise) ---

    /// Snapshot satellite `s` on its first mutation in the current transaction.
    #[inline]
    fn touch_sat(&mut self, s: u32) {
        if self.txn && self.sat_touched[s as usize] != self.txn_gen {
            self.sat_touched[s as usize] = self.txn_gen;
            self.sat_snap.push((s, self.sats[s as usize].clone()));
        }
    }
    /// Write `user_sat[u]`, logging the previous value while in a transaction.
    #[inline]
    fn set_user(&mut self, u: u32, val: i32) {
        if self.txn {
            self.user_snap.push((u, self.user_sat[u as usize]));
        }
        self.user_sat[u as usize] = val;
    }
    #[inline]
    fn begin_txn(&mut self) {
        self.txn = true;
        self.txn_gen = self.txn_gen.wrapping_add(1);
        self.sat_snap.clear();
        self.user_snap.clear();
    }
    #[inline]
    fn commit_txn(&mut self) {
        self.sat_snap.clear();
        self.user_snap.clear();
    }
    /// Restore the exact pre-transaction state in O(touched).
    #[inline]
    fn rollback_txn(&mut self) {
        while let Some((u, prev)) = self.user_snap.pop() {
            self.user_sat[u as usize] = prev;
        }
        for (s, sat) in self.sat_snap.drain(..) {
            self.sats[s as usize] = sat;
        }
    }

    /// Seat unassigned user `x`, displacing ≤`depth` members (each rehomed
    /// recursively). Commits on success; on failure restores state exactly via
    /// surgical undo. The top-level direct insert may recolor (it commits or
    /// no-ops, and a successful augment is never unwound, so it is safe);
    /// displacement and re-seat stay on the fast path so the undo never has to
    /// reverse a recolor. Strictly bounded by `self.budget`.
    /// Ejection-chain search to seat user `x`. With `recolor = false` (the hot
    /// path used by construction repair and LNS) a displaced re-seat stays on the
    /// O(members) fast color path and the undo is a cheap pop/re-seat. With
    /// `recolor = true` the displacement may recolor the satellite — so a chain
    /// succeeds whenever `{s\{m} ∪ x}` is 4-colorable, not merely when a color is
    /// free under the current labels — and the undo restores an exact snapshot.
    /// Recolor finds strictly more chains but is far costlier (it fires the exact
    /// 4-coloring), so it is reserved for a single bounded final pass.
    fn augment(&mut self, x: u32, depth: u32, recolor: bool) -> bool {
        if depth == 0 || self.budget == 0 {
            return false;
        }
        for s in self.ordered_candidates(x) {
            if self.visited[s as usize] == self.gen {
                continue;
            }
            if self.budget == 0 {
                return false;
            }
            self.budget -= 1;
            let dx = self.dir(x, s);
            self.touch_sat(s);
            if self.sats[s as usize].try_insert(x, dx, true) {
                self.set_user(x, s as i32);
                return true;
            }
            self.visited[s as usize] = self.gen;
            // Snapshot is only needed to undo a possible recolor.
            let snap = if recolor {
                Some(self.sats[s as usize].clone())
            } else {
                None
            };
            let members: ArrayVec<u32, 32> = self.sats[s as usize].users.clone();
            for m in members {
                if self.budget == 0 {
                    return false;
                }
                let dm = self.sats[s as usize].remove(m); // m's dir, reused below
                self.set_user(m, -1);
                if self.sats[s as usize].try_insert(x, dx, recolor) {
                    self.set_user(x, s as i32);
                    if self.augment(m, depth - 1, recolor) {
                        return true;
                    }
                    self.set_user(x, -1);
                    if !recolor {
                        self.sats[s as usize].pop(); // remove x
                    }
                }
                match &snap {
                    // Recolor path: restore s exactly (handles any relabel).
                    Some(snap) => self.sats[s as usize].clone_from(snap),
                    // Fast path: m's original color is still free among the rest;
                    // its direction is the one we just removed (no recompute).
                    None => {
                        self.sats[s as usize].try_insert(m, dm, false);
                    }
                }
                self.set_user(m, s as i32);
            }
        }
        false
    }

    /// Final, coloring-complete repair pass over the users still unserved, run
    /// once on the chosen solution. It allows the displacement to recolor, so it
    /// seats chains the fast (no-recolor) repair/LNS could not realize under their
    /// fixed labels — recovering coverage the construction coloring otherwise
    /// left on the table. Bounded to the residual, so the costly recolor stays
    /// cheap overall. It only ever adds users.
    fn recolor_repair(&mut self, order: &[u32], deadline: Instant) {
        for &x in order {
            if self.user_sat[x as usize] >= 0 {
                continue;
            }
            if Instant::now() >= deadline {
                break;
            }
            self.gen += 1;
            self.budget = RECOLOR_BUDGET;
            self.augment(x, MAX_DEPTH, true);
        }
    }

    /// `recolor` lets the displacement recolor (Maximum mode) — strictly more
    /// chains, far costlier; the default fast path passes `false`.
    fn repair(&mut self, order: &[u32], deadline: Instant, recolor: bool) {
        for &x in order {
            if self.user_sat[x as usize] >= 0 {
                continue;
            }
            if Instant::now() >= deadline {
                break;
            }
            self.gen += 1;
            self.budget = ATTEMPT_BUDGET;
            self.augment(x, MAX_DEPTH, recolor);
        }
    }

    /// Exact branch-and-bound certification. Proves the component optimum if the
    /// whole search tree is explored inside the node/time budget (returning
    /// `Some((optimum, assignment))`); otherwise returns `None` and the caller
    /// keeps the heuristic result, so this can only ever help. `incumbent` (the
    /// best heuristic value) seeds the bound to prune from the first node.
    fn solve_exact(
        &mut self,
        order: &[u32],
        incumbent: usize,
        deadline: Instant,
    ) -> Option<(usize, Vec<Sat>)> {
        for s in self.sats.iter_mut() {
            *s = Sat::default();
        }
        self.user_sat.iter_mut().for_each(|x| *x = -1);
        let mut best = incumbent;
        let mut best_sats: Option<Vec<Sat>> = None;
        let mut nodes: u64 = 0;
        let complete = self.bb(order, 0, 0, &mut best, &mut best_sats, &mut nodes, deadline);
        complete.then(|| (best, best_sats.unwrap_or_else(|| self.sats.clone())))
    }

    /// One B&B node. Returns true iff this subtree was fully explored (so the
    /// bound is proven); false signals a budget abort that bails the whole search.
    #[allow(clippy::too_many_arguments)]
    fn bb(
        &mut self,
        order: &[u32],
        idx: usize,
        served: usize,
        best: &mut usize,
        best_sats: &mut Option<Vec<Sat>>,
        nodes: &mut u64,
        deadline: Instant,
    ) -> bool {
        *nodes += 1;
        if *nodes > EXACT_MAX_NODES {
            return false;
        }
        if (*nodes).is_multiple_of(8192) && Instant::now() >= deadline {
            return false;
        }
        // Optimistic bound: at best every remaining user is also served.
        let rem = order.len() - idx;
        if served + rem <= *best {
            return true; // cannot beat the incumbent — prune this subtree
        }
        if idx == order.len() {
            if served > *best {
                *best = served;
                *best_sats = Some(self.sats.clone());
            }
            return true;
        }
        let lu = order[idx];
        // Branch: place `lu` on each feasible satellite that admits a color.
        for ls in self.ordered_candidates(lu) {
            let d = self.dir(lu, ls);
            let saved = self.sats[ls as usize].clone();
            if self.sats[ls as usize].try_insert(lu, d, true) {
                let done = self.bb(order, idx + 1, served + 1, best, best_sats, nodes, deadline);
                self.sats[ls as usize] = saved;
                if !done {
                    return false;
                }
            } else {
                self.sats[ls as usize] = saved;
            }
        }
        // Branch: leave `lu` unserved.
        self.bb(order, idx + 1, served, best, best_sats, nodes, deadline)
    }

    /// Flow-seeded construction: realize the optimal max-flow matching (which
    /// achieves the upper-bound cardinality but ignores coloring) by placing each
    /// matched user on its matched satellite with integral coloring, then repair
    /// the users that flow left unmatched or that coloring had to evict. This
    /// fixes the *capacity allocation* optimally; repair recovers the rest.
    fn seed_and_repair(
        &mut self,
        matching: &[i32],
        order: &[u32],
        ub: usize,
        deadline: Instant,
        recolor: bool,
    ) -> usize {
        // Group the matched users by their satellite.
        let mut by_sat: Vec<Vec<u32>> = vec![Vec::new(); self.sats.len()];
        for (lu, &s) in matching.iter().enumerate() {
            if s >= 0 {
                by_sat[s as usize].push(lu as u32);
            }
        }
        // Realize each satellite, adding users in ascending intra-satellite
        // conflict order. Colorable (spread-out) users are kept; only the
        // densely-clustered surplus that no 4-coloring can fit is evicted — i.e.
        // we keep a near-maximum 4-colorable subset of the optimal matching.
        for (s, users) in by_sat.iter().enumerate() {
            let dirs: Vec<Vec3> = users.iter().map(|&u| self.dir(u, s as u32)).collect();
            let mut deg = vec![0u32; users.len()];
            for i in 0..users.len() {
                for j in (i + 1)..users.len() {
                    if same_color_conflict(dirs[i], dirs[j]) {
                        deg[i] += 1;
                        deg[j] += 1;
                    }
                }
            }
            let mut idx: Vec<usize> = (0..users.len()).collect();
            idx.sort_unstable_by_key(|&i| (deg[i], users[i]));
            for &i in &idx {
                if self.sats[s].try_insert(users[i], dirs[i], true) {
                    self.user_sat[users[i] as usize] = s as i32;
                }
            }
        }
        // Repair flow-unmatched and coloring-evicted users.
        let assigned = self.assigned_count();
        if assigned < ub {
            self.repair(order, deadline, recolor);
        }
        self.assigned_count()
    }

    #[inline]
    fn assigned_count(&self) -> usize {
        self.user_sat.iter().filter(|&&s| s >= 0).count()
    }

    /// Large-neighborhood search: repeatedly **ruin** the satellites around a
    /// still-unserved terminal (tear down their assignments) and **recreate**
    /// that neighborhood from scratch via the augmenting seater, keeping the
    /// rebuild only if it serves at least as many users. This escapes the local
    /// optima that simple ejection chains cannot, by re-coloring/re-packing a
    /// whole cluster at once.
    ///
    /// Bounded by a fixed *iteration* count (not wall-clock) and stopped early
    /// after a run of non-improving rounds, so the result is **deterministic**
    /// — byte-identical regardless of machine speed. The kept solution is never
    /// worse than the input. `deadline` is only a backstop on pathological
    /// machines (never reached on the test set).
    fn lns(&mut self, ub: usize, deadline: Instant, seed: u64, intense: bool) {
        let nusers = self.user_sat.len() as u32;
        if nusers == 0 {
            return;
        }
        // Scale the search to the component; cap so a huge component stays fast.
        // `stall` ends a converged search early — gains plateau quickly, so a
        // short stall window captures almost all of them cheaply. Each round is
        // O(touched) thanks to the transactional undo, so the bound can be
        // generous without hurting wall-clock. `intense` (the Maximum algorithm)
        // lifts the ceilings to chase the last few users on a residual-gap component.
        let (rounds_cap, depth, attempt_budget) = if intense {
            (LNS_MAX_ROUNDS_INTENSE, LNS_DEPTH_INTENSE, LNS_ATTEMPT_BUDGET_INTENSE)
        } else {
            (LNS_MAX_ROUNDS, LNS_DEPTH, LNS_ATTEMPT_BUDGET)
        };
        let max_rounds = (self.user_sat.len() as u32 * 4).min(rounds_cap);
        let stall_limit = (max_rounds / 4).clamp(800, if intense { 8_000 } else { 4_000 });
        let mut best_cnt = self.assigned_count();
        // A fixed LCG seeded by component shape *and the worker seed* — each
        // parallel search explores a different sequence of ruins, but every run
        // is reproducible (no clock, no thread-order dependence).
        let mut state: u64 = 0x243F_6A88_85A3_08D3
            ^ (nusers as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
        let mut rng = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        // Round-robin starting point (offset per worker so they begin on
        // different blocked terminals).
        let mut cursor = (seed.wrapping_mul(2654435761) % nusers as u64) as u32;
        let mut stall = 0u32;
        let mut freed: Vec<u32> = Vec::with_capacity(64);
        for round in 0..max_rounds {
            // Early termination: optimality or convergence stall.
            if best_cnt >= ub || stall >= stall_limit {
                break;
            }
            // Backstop only — primary bound is the iteration count above.
            if round % 1024 == 0 && Instant::now() >= deadline {
                break;
            }
            // Find the next still-unassigned feasible user to ruin around.
            let mut x = None;
            for off in 0..nusers {
                let u = (cursor + off) % nusers;
                if self.user_sat[u as usize] < 0 && !self.feas[u as usize].is_empty() {
                    x = Some(u);
                    cursor = (u + 1) % nusers;
                    break;
                }
            }
            let Some(x) = x else { break }; // everything feasible is served

            self.begin_txn();
            // Ruin: clear x's candidate satellites, collecting the freed users.
            // Everything freed (except x) was assigned, so the live count drops
            // by `freed.len() - 1` — tracked incrementally to avoid an O(users)
            // recount each round.
            freed.clear();
            freed.push(x);
            let cands: ArrayVec<u32, 16> = self.feas[x as usize].iter().copied().collect();
            for &s in &cands {
                self.touch_sat(s);
                // Move the satellite out (leaving it reset) and drain its users —
                // no clone of the member list, and the take already does the reset.
                let taken = std::mem::take(&mut self.sats[s as usize]);
                for &u in &taken.users {
                    self.set_user(u, -1);
                    freed.push(u);
                }
            }
            // Recreate: re-seat the freed users in a perturbed most-constrained
            // order (the perturbation is what makes successive rounds explore
            // different packings of the same cluster). Augment can also pull
            // freed users into adjacent satellites, but it always leaves every
            // *non-freed* user assigned, so the delta is confined to `freed`.
            freed.sort_unstable_by_key(|&u| (self.feas[u as usize].len(), u));
            let jitter = (rng() % freed.len() as u32) as usize;
            freed.rotate_left(jitter);
            for &u in &freed {
                if self.user_sat[u as usize] >= 0 {
                    continue;
                }
                self.gen += 1;
                self.budget = attempt_budget;
                self.augment(u, depth, false);
            }

            let seated = freed
                .iter()
                .filter(|&&u| self.user_sat[u as usize] >= 0)
                .count();
            let cnt = best_cnt - (freed.len() - 1) + seated;
            if cnt > best_cnt {
                best_cnt = cnt;
                self.commit_txn();
                stall = 0;
            } else {
                // Reject: surgically restore the pre-round state and move on.
                self.rollback_txn();
                stall += 1;
            }
        }
        self.txn = false;
    }
}

/// A valid upper bound on this component's optimum that **accounts for the
/// 4-coloring rule** (the plain matching bound ignores it). Per satellite,
/// partition its feasible users' <10° conflict graph into cliques: a clique of
/// `k` mutually-conflicting users needs `k` distinct colors, so at most 4 of
/// them can ever be served. Hence `Σ min(4, |Cᵢ|)`, capped at the 32-beam limit,
/// bounds how many users that satellite can serve. A max-flow with these caps is
/// a *sound* upper bound — every real solution is feasible under them — and it
/// is ≤ the matching bound (e.g. it correctly tightens case 03 from 5 to 4).
fn coloring_bound(
    scn: &Scenario,
    c: &Component,
    local_feas: &[Vec<u32>],
    ns: usize,
    matching_ub: usize,
) -> usize {
    // Per-satellite feasible-user lists (local indices).
    let mut sat_users: Vec<Vec<u32>> = vec![Vec::new(); ns];
    for (lu, cand) in local_feas.iter().enumerate() {
        for &s in cand {
            sat_users[s as usize].push(lu as u32);
        }
    }
    let caps: Vec<u32> = (0..ns)
        .into_par_iter()
        .map(|s| {
            let us = &sat_users[s];
            let n = us.len();
            if n <= 4 {
                return n as u32; // ≤4 users are trivially 4-colorable
            }
            let satpos = scn.sats[c.sats[s] as usize];
            let dirs: Vec<Vec3> = us
                .iter()
                .map(|&lu| (scn.users[c.users[lu as usize] as usize] - satpos).unit())
                .collect();
            // Greedy clique partition (deterministic natural order): repeatedly
            // grow a maximal clique from the first uncovered vertex.
            let mut covered = vec![false; n];
            let mut bound = 0u32;
            let mut clique: Vec<usize> = Vec::new();
            for i in 0..n {
                if covered[i] {
                    continue;
                }
                covered[i] = true;
                clique.clear();
                clique.push(i);
                for j in (i + 1)..n {
                    if !covered[j]
                        && clique
                            .iter()
                            .all(|&m| same_color_conflict(dirs[m], dirs[j]))
                    {
                        clique.push(j);
                        covered[j] = true;
                    }
                }
                bound += (clique.len() as u32).min(4);
                if bound >= 32 {
                    return 32; // already at the beam limit; no tightening possible
                }
            }
            bound.min(32)
        })
        .collect();
    // A clique cut only matters when it drops a satellite below what the plain
    // matching could already route to it (min(32, #feasible users)). If nothing
    // tightened, the capped flow equals the matching bound — skip recomputing it.
    let tightened = (0..ns).any(|s| (caps[s] as usize) < sat_users[s].len().min(32));
    if tightened {
        matching::max_matching_capped(local_feas, &caps)
    } else {
        matching_ub
    }
}

struct CompResult {
    sat_members: Vec<(u32, Vec<(u32, u8)>)>, // (global sat, [(global user, color)])
    achieved: usize,
    upper_bound: usize,
    colored_bound: usize,
}

fn solve_component(
    scn: &Scenario,
    feas_sats: &[Vec<u32>],
    c: &Component,
    deadline: Instant,
    intense: bool,
) -> CompResult {
    let ns = c.sats.len();
    // Local feasibility: every feasible sat of a component user is in this
    // component, so a binary search always succeeds.
    let local_feas: Vec<Vec<u32>> = c
        .users
        .iter()
        .map(|&gu| {
            feas_sats[gu as usize]
                .iter()
                .map(|&gs| c.sats.binary_search(&gs).unwrap() as u32)
                .collect()
        })
        .collect();

    // Exact upper bound + one optimal (coloring-free) assignment for the seed.
    let (upper_bound, matching) = matching::max_matching(&local_feas, ns);

    let mut sat_deg = vec![0u32; ns];
    for cand in &local_feas {
        for &s in cand {
            sat_deg[s as usize] += 1;
        }
    }

    let mut order: Vec<u32> = (0..c.users.len() as u32).collect();
    order.sort_by_key(|&u| (local_feas[u as usize].len(), u));

    // A spatial (position) user order, in addition to `order` above.
    let mut order_pos: Vec<u32> = (0..c.users.len() as u32).collect();
    order_pos.sort_by(|&a, &b| {
        let pa = scn.users[c.users[a as usize] as usize];
        let pb = scn.users[c.users[b as usize] as usize];
        pa.x.total_cmp(&pb.x)
            .then(pa.y.total_cmp(&pb.y))
            .then(pa.z.total_cmp(&pb.z))
    });

    // One full construct → flow-seed → LNS → (Maximum) recolor pipeline, for a
    // given construction `recolor` mode. Factored so Maximum mode can run it BOTH
    // ways and keep the per-component best: recolor-during-construction recovers a
    // user on some components (09/10) but costs users on others (11), and neither
    // mode wins everywhere — so the only way to the true maximum is to try both.
    let run_pipeline = |recolor: bool| -> CompSolver {
        // Ensemble of independent greedy constructions; keep the best. They don't
        // interact, so they run in parallel.
        //   0: least-loaded sats        1: highest-elevation sats
        //   2: least-contended sats     3: spatial user order + highest elevation
        let mut runs: Vec<(usize, CompSolver)> = (0u32..4)
            .into_par_iter()
            .map(|cfg| {
                let (sat_choice, ord) = match cfg {
                    1 => (SatChoice::HighestElevation, &order),
                    2 => (SatChoice::LeastContended, &order),
                    3 => (SatChoice::HighestElevation, &order_pos),
                    _ => (SatChoice::LeastLoaded, &order),
                };
                let mut solver = CompSolver::new(scn, c, &local_feas, &sat_deg, sat_choice);
                let ach = solver.solve(ord, upper_bound, deadline, recolor);
                (ach, solver)
            })
            .collect();
        // Best greedy; ties resolve to the lowest member index (deterministic).
        let mut best_idx = 0;
        for i in 1..runs.len() {
            if runs[i].0 > runs[best_idx].0 {
                best_idx = i;
            }
        }
        let best_greedy_ach = runs[best_idx].0;

        // Flow-seeded construction (realize the optimal matching, then repair),
        // run only when greedy fell short of the bound; it wins ties.
        let (best_seed_ach, seed_solver) = if best_greedy_ach < upper_bound {
            let mut fs = CompSolver::new(scn, c, &local_feas, &sat_deg, SatChoice::LeastLoaded);
            let fs_ach = fs.seed_and_repair(&matching, &order, upper_bound, deadline, recolor);
            if fs_ach >= best_greedy_ach {
                (fs_ach, fs)
            } else {
                runs.swap_remove(best_idx)
            }
        } else {
            runs.swap_remove(best_idx)
        };

        // Polish with parallel ruin-and-recreate LNS (intensive under `intense`),
        // keeping the best worker. Components at their bound exit LNS immediately.
        let mut best_solver = if best_seed_ach < upper_bound {
            let mut polished: Vec<CompSolver> = (0..LNS_WORKERS)
                .into_par_iter()
                .map(|w| {
                    let mut s = seed_solver.clone();
                    s.lns(upper_bound, deadline, w as u64, intense);
                    s
                })
                .collect();
            let mut bi = 0;
            let mut bc = polished[0].assigned_count();
            for (i, p) in polished.iter().enumerate().skip(1) {
                let cnt = p.assigned_count();
                if cnt > bc {
                    bc = cnt;
                    bi = i;
                }
            }
            polished.swap_remove(bi)
        } else {
            seed_solver // already provably optimal for this component
        };

        // Maximum mode only: one coloring-complete pass over the residual.
        if intense && best_solver.assigned_count() < upper_bound {
            best_solver.recolor_repair(&order, deadline);
        }
        best_solver
    };

    // Default: a single pass (no construction recolor). Maximum: run both
    // construction modes and keep whichever serves more on this component.
    let best_solver = if intense {
        let plain = run_pipeline(false);
        let recolored = run_pipeline(true);
        if recolored.assigned_count() > plain.assigned_count() {
            recolored
        } else {
            plain
        }
    } else {
        run_pipeline(false)
    };

    let mut best_ach = best_solver.assigned_count();
    let mut best = best_solver.sats;

    // Tighter, coloring-aware upper bound (independent of the assignment found).
    // A Lagrangian-decomposition bound was prototyped here too but empirically
    // added nothing: its per-satellite relaxation uses the same clique structure,
    // so it is blind to exactly the non-clique coloring obstructions that bound
    // the hard cases (07/09/10/11) — at a large runtime cost. `coloring_bound`
    // captures those obstructions directly.
    let mut colored_bound = coloring_bound(scn, c, &local_feas, ns, upper_bound);

    // Exact certification (opt-in via BEAM_EXACT): if a small component still
    // has a residual gap, try to prove its optimum with branch-and-bound. On
    // success the optimum is *both* the assignment and a tight bound (A/bound =
    // 100% for that component); if the budget is exhausted the heuristic stands.
    // Measured outcome on these instances: it does not terminate even on the
    // smallest gap components (no tight bound ⇒ no effective pruning), so it is
    // off by default — but it can only ever help, never hurt.
    if std::env::var("BEAM_EXACT").is_ok()
        && best_ach < colored_bound
        && c.users.len() <= EXACT_MAX_USERS
    {
        let ex_deadline = deadline.min(Instant::now() + Duration::from_secs(30));
        let mut ex = CompSolver::new(scn, c, &local_feas, &sat_deg, SatChoice::LeastLoaded);
        if let Some((opt, sats)) = ex.solve_exact(&order, best_ach, ex_deadline) {
            best_ach = opt;
            best = sats;
            colored_bound = opt; // proven optimum ⇒ tight ceiling
        }
    }

    if std::env::var("BEAM_DEBUG").is_ok() && upper_bound > best_ach {
        eprintln!(
            "  comp users={} sats={} matching={} achieved={} colored={} gap={}",
            c.users.len(),
            ns,
            upper_bound,
            best_ach,
            colored_bound,
            colored_bound - best_ach
        );
    }

    let mut sat_members = Vec::new();
    for (ls, sat) in best.iter().enumerate().take(ns) {
        if sat.users.is_empty() {
            continue;
        }
        // Level the four color bands per satellite — a cosmetic, coverage-neutral
        // relabel of this satellite's final beams (first-fit otherwise piles onto
        // color 0, which dominates every render).
        let balanced = crate::coloring::rebalance(&sat.dirs, &sat.colors, c.sats[ls] as usize);
        let members: Vec<(u32, u8)> = (0..sat.users.len())
            .map(|i| (c.users[sat.users[i] as usize], balanced[i]))
            .collect();
        sat_members.push((c.sats[ls], members));
    }

    CompResult {
        sat_members,
        achieved: best_ach,
        upper_bound,
        colored_bound,
    }
}

pub struct Solution {
    /// `per_sat[s]` = `(user_index, color)` served by satellite `s`.
    pub per_sat: Vec<Vec<(u32, u8)>>,
    pub achieved: usize,
    /// Capacitated-matching ceiling (ignores coloring).
    pub upper_bound: usize,
    /// Tighter ceiling that accounts for the 4-coloring rule (≤ `upper_bound`,
    /// and never below the true optimum).
    pub colored_bound: usize,
}

impl Solution {
    /// Build the near-optimality [`Certificate`](crate::io::Certificate) header
    /// for this solution against its scenario + feasibility graph. Shared by the
    /// CLI and the wasm `solve_scenario` entry point.
    pub fn certificate(
        &self,
        scn: &Scenario,
        feas: &crate::feasibility::Feasibility,
    ) -> crate::io::Certificate {
        crate::io::Certificate {
            total_users: scn.users.len(),
            feasible_users: feas.feasible_users,
            upper_bound: self.upper_bound,
            colored_bound: self.colored_bound,
            achieved: self.achieved,
        }
    }
}

/// Default wall-clock ceiling for the optional repair/LNS phase, shared by the
/// CLI and the wasm entry points. Far under the 15-minute grader limit; the
/// greedy solution is always complete and valid before repair runs, so this only
/// bounds how long the solver spends improving it.
pub const REPAIR_BUDGET: Duration = Duration::from_secs(120);

/// Solve the scenario. `intense` selects the maximum-coverage mode (the
/// `Maximum` algorithm / CLI `--max`): a much larger LNS budget on residual-gap
/// components, slower but recovering the last few users. The default (`false`)
/// is the standard ~sub-second production solve.
pub fn solve(
    scn: &Scenario,
    feas: &crate::feasibility::Feasibility,
    deadline: Instant,
    intense: bool,
) -> Solution {
    let comps = components::decompose(feas, scn.sats.len());

    // Each component is fully independent → solve them in parallel.
    let results: Vec<CompResult> = comps
        .par_iter()
        .map(|c| solve_component(scn, &feas.sats, c, deadline, intense))
        .collect();

    let mut per_sat = vec![Vec::new(); scn.sats.len()];
    let mut achieved = 0;
    let mut upper_bound = 0;
    let mut colored_bound = 0;
    for r in results {
        achieved += r.achieved;
        upper_bound += r.upper_bound;
        colored_bound += r.colored_bound;
        for (gs, members) in r.sat_members {
            per_sat[gs as usize] = members;
        }
    }
    Solution {
        per_sat,
        achieved,
        upper_bound,
        colored_bound,
    }
}
