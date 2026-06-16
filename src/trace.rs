//! Instrumented, single-construction solvers that record the assignment
//! step-by-step (for real-time animation) and classify why each unserved user
//! could not be assigned (for the inspector). These are intentionally simple and
//! global (no component/parallel machinery) so the produced order tells a clear
//! visual story; the production solver in [`crate::assign`] is what optimizes
//! coverage.

use crate::coloring::fast_color;
use crate::feasibility::Feasibility;
use crate::geom::{visible, Vec3};
use crate::index::Grid;
use crate::io::Scenario;
use crate::{assign, matching};
use std::time::{Duration, Instant};

/// A construction the user can watch / compare.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// The production solver ([`crate::assign`]): per-component ensemble +
    /// augmenting repair + parallel large-neighborhood search. This is what the
    /// CLI emits, so its coverage matches the reported certificate exactly.
    Optimized,
    /// Same pipeline as [`Optimized`] but with a much larger LNS budget on any
    /// component that still has a residual gap — the maximum-coverage mode. It is
    /// markedly slower (seconds, not sub-second) and recovers the last few users a
    /// normal run leaves behind (e.g. +5 on the 100k case), so its coverage can
    /// exceed the default certificate's.
    Maximum,
    GreedyLeastLoaded,
    GreedyHighestElevation,
    GreedyLeastContended,
    /// Spatial (west→east) user order with highest-elevation satellite choice —
    /// the 5th ensemble construction, now watchable on its own.
    GreedySpatial,
    FlowSeeded,
    /// A single greedy build, then the **ruin-and-recreate LNS** run live: watch
    /// it tear down clusters around blocked terminals and recover the stragglers.
    LnsOnly,
    /// The maximum capacitated matching itself — the **upper bound**, drawn as
    /// beams *ignoring* the 4-color rule (so it serves more than any valid
    /// solution can). Selecting it visualizes the ceiling.
    Matching,
}

impl Algorithm {
    pub const ALL: [Algorithm; 9] = [
        Algorithm::Optimized,
        Algorithm::Maximum,
        Algorithm::GreedyLeastLoaded,
        Algorithm::GreedyHighestElevation,
        Algorithm::GreedyLeastContended,
        Algorithm::GreedySpatial,
        Algorithm::FlowSeeded,
        Algorithm::LnsOnly,
        Algorithm::Matching,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Optimized => "Optimized · ensemble + repair",
            Algorithm::Maximum => "Maximum · intensive search",
            Algorithm::GreedyLeastLoaded => "Greedy · least-loaded",
            Algorithm::GreedyHighestElevation => "Greedy · highest elevation",
            Algorithm::GreedyLeastContended => "Greedy · least-contended",
            Algorithm::GreedySpatial => "Greedy · spatial sweep",
            Algorithm::FlowSeeded => "Flow-seeded · optimal matching",
            Algorithm::LnsOnly => "LNS · ruin & recreate",
            Algorithm::Matching => "Matching · upper bound",
        }
    }
}

/// One satellite→user beam, recorded in the order it was formed.
#[derive(Clone, Copy)]
pub struct Event {
    pub user: u32,
    pub sat: u32,
    pub color: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    NoSatInView,
    AllInterfered,
    AllFull,
    ColorBlocked,
}

impl Reason {
    pub fn label(self) -> &'static str {
        match self {
            Reason::NoSatInView => "No satellite in view",
            Reason::AllInterfered => "Blocked by interferer",
            Reason::AllFull => "All visible satellites full",
            Reason::ColorBlocked => "No free color",
        }
    }
    pub fn detail(self) -> &'static str {
        match self {
            Reason::NoSatInView => "No Starlink satellite is within 45° of this user's vertical.",
            Reason::AllInterfered => "Every satellite in view is within 20° of a non-Starlink satellite (from the user's perspective).",
            Reason::AllFull => "Every satellite this user can see already serves 32 beams.",
            Reason::ColorBlocked => "Visible satellites have room, but every color would sit within 10° of a same-color beam.",
        }
    }
}

pub struct Unassigned {
    pub user: u32,
    pub reason: Reason,
    /// Satellites within 45° of vertical (ignoring interferers).
    pub in_view: u32,
}

pub struct Trace {
    pub events: Vec<Event>,
    pub unassigned: Vec<Unassigned>,
    pub upper_bound: usize,
    /// Tighter, coloring-aware ceiling (equals `upper_bound` unless clique cuts
    /// bind, e.g. case 03 where it drops 5→4).
    pub colored_bound: usize,
    pub feasible_users: usize,
}

/// Transient state for a single-pass trace construction (not the production solver).
/// Tracks per-satellite beams and user assignments during a greedy or LNS build,
/// recording events for visualization.
struct State {
    sat_users: Vec<Vec<u32>>,
    sat_colors: Vec<Vec<u8>>,
    sat_dirs: Vec<Vec<Vec3>>,
    user_sat: Vec<i32>,
    events: Vec<Event>,
}

impl State {
    fn new(n_users: usize, n_sats: usize) -> Self {
        State {
            sat_users: vec![Vec::new(); n_sats],
            sat_colors: vec![Vec::new(); n_sats],
            sat_dirs: vec![Vec::new(); n_sats],
            user_sat: vec![-1; n_users],
            events: Vec::new(),
        }
    }
    #[inline]
    fn load(&self, s: usize) -> usize {
        self.sat_users[s].len()
    }
    fn try_place(&mut self, scn: &Scenario, u: usize, s: usize) -> bool {
        if self.place(scn, u, s) {
            let i = self.sat_users[s].len() - 1;
            self.events.push(Event {
                user: u as u32,
                sat: s as u32,
                color: self.sat_colors[s][i],
            });
            true
        } else {
            false
        }
    }
    /// Place `u` on `s` if a color exists, *without* recording an event (used by
    /// the live LNS, whose event stream is rebuilt from the final state).
    fn place(&mut self, scn: &Scenario, u: usize, s: usize) -> bool {
        if self.sat_users[s].len() >= 32 {
            return false;
        }
        let d = (scn.users[u] - scn.sats[s]).unit();
        if let Some(c) = fast_color(&self.sat_dirs[s], &self.sat_colors[s], d) {
            self.sat_users[s].push(u as u32);
            self.sat_colors[s].push(c);
            self.sat_dirs[s].push(d);
            self.user_sat[u] = s as i32;
            true
        } else {
            false
        }
    }
}

pub fn run(scn: &Scenario, feas: &Feasibility, algo: Algorithm) -> Trace {
    match algo {
        Algorithm::Optimized => run_optimized(scn, feas, false),
        Algorithm::Maximum => run_optimized(scn, feas, true),
        Algorithm::Matching => run_matching(scn, feas),
        Algorithm::LnsOnly => run_lns_only(scn, feas),
        _ => run_greedy(scn, feas, algo),
    }
}

/// Run the full production solver and present its result as a trace. The reveal
/// order is the solver's own most-constrained-first commit order, so the
/// animation tells a coherent story; the final coverage equals the CLI's.
fn run_optimized(scn: &Scenario, feas: &Feasibility, intense: bool) -> Trace {
    let n_users = scn.users.len();
    let n_sats = scn.sats.len();
    // Match the CLI's repair budget so the visualizer's coverage is guaranteed
    // identical to the certificate (repair converges in a few seconds anyway).
    let deadline = Instant::now() + Duration::from_secs(300);
    let sol = assign::solve(scn, feas, deadline, intense);

    let mut user_sat = vec![-1i32; n_users];
    let mut user_color = vec![0u8; n_users];
    let mut sat_load = vec![0usize; n_sats];
    for (s, members) in sol.per_sat.iter().enumerate() {
        sat_load[s] = members.len();
        for &(u, c) in members {
            user_sat[u as usize] = s as i32;
            user_color[u as usize] = c;
        }
    }

    let mut order: Vec<u32> = (0..n_users as u32)
        .filter(|&u| user_sat[u as usize] >= 0)
        .collect();
    order.sort_by_key(|&u| (feas.sats[u as usize].len(), u));
    let events: Vec<Event> = order
        .iter()
        .map(|&u| Event {
            user: u,
            sat: user_sat[u as usize] as u32,
            color: user_color[u as usize],
        })
        .collect();

    let unassigned = classify(scn, feas, &user_sat, &sat_load);
    Trace {
        events,
        unassigned,
        upper_bound: sol.upper_bound,
        colored_bound: sol.colored_bound,
        feasible_users: feas.feasible_users,
    }
}

fn run_greedy(scn: &Scenario, feas: &Feasibility, algo: Algorithm) -> Trace {
    let n_users = scn.users.len();
    let n_sats = scn.sats.len();
    let (upper_bound, matching) = matching::max_matching(&feas.sats, n_sats);

    // Per-satellite global demand (for the least-contended heuristic).
    let mut sat_deg = vec![0u32; n_sats];
    for cand in &feas.sats {
        for &s in cand {
            sat_deg[s as usize] += 1;
        }
    }

    let mut st = State::new(n_users, n_sats);
    if algo == Algorithm::FlowSeeded {
        flow_seed(&mut st, scn, &matching);
    }
    // Greedy sweep (the whole story for the greedy algorithms; the gap-filler
    // for the flow-seed). Most-constrained-first order, except the spatial
    // sweep which orders users west→east by position.
    let mut order: Vec<u32> = (0..n_users as u32)
        .filter(|&u| !feas.sats[u as usize].is_empty())
        .collect();
    if matches!(algo, Algorithm::GreedySpatial) {
        order.sort_by(|&a, &b| {
            let pa = scn.users[a as usize];
            let pb = scn.users[b as usize];
            pa.x.total_cmp(&pb.x)
                .then(pa.y.total_cmp(&pb.y))
                .then(pa.z.total_cmp(&pb.z))
        });
    } else {
        order.sort_by_key(|&u| (feas.sats[u as usize].len(), u));
    }
    for &u in &order {
        if st.user_sat[u as usize] >= 0 {
            continue;
        }
        let mut cands: Vec<u32> = feas.sats[u as usize].clone();
        sort_candidates(&mut cands, scn, &st, &sat_deg, u as usize, algo);
        for &s in &cands {
            if st.try_place(scn, u as usize, s as usize) {
                break;
            }
        }
    }

    let sat_load: Vec<usize> = st.sat_users.iter().map(|v| v.len()).collect();
    let unassigned = classify(scn, feas, &st.user_sat, &sat_load);
    // The greedy traces don't compute the (more expensive) coloring bound.
    Trace {
        events: st.events,
        unassigned,
        upper_bound,
        colored_bound: upper_bound,
        feasible_users: feas.feasible_users,
    }
}

/// The maximum capacitated matching, drawn as beams **ignoring the 4-color
/// rule** — i.e. the upper bound. Colors are a best-effort fast pick; a
/// satellite forced to reuse a color is exactly the violation that makes this
/// count unreachable by any valid solution.
fn run_matching(scn: &Scenario, feas: &Feasibility) -> Trace {
    let n_users = scn.users.len();
    let n_sats = scn.sats.len();
    let (upper_bound, matching) = matching::max_matching(&feas.sats, n_sats);

    let mut st = State::new(n_users, n_sats);
    let mut order: Vec<u32> = (0..n_users as u32)
        .filter(|&u| matching[u as usize] >= 0)
        .collect();
    order.sort_by_key(|&u| (feas.sats[u as usize].len(), u));
    for &u in &order {
        let s = matching[u as usize] as usize;
        let d = (scn.users[u as usize] - scn.sats[s]).unit();
        let c = fast_color(&st.sat_dirs[s], &st.sat_colors[s], d).unwrap_or(0);
        st.sat_users[s].push(u);
        st.sat_colors[s].push(c);
        st.sat_dirs[s].push(d);
        st.user_sat[u as usize] = s as i32;
        st.events.push(Event {
            user: u,
            sat: s as u32,
            color: c,
        });
    }

    let sat_load: Vec<usize> = st.sat_users.iter().map(|v| v.len()).collect();
    let unassigned = classify(scn, feas, &st.user_sat, &sat_load);
    Trace {
        events: st.events,
        unassigned,
        upper_bound,
        colored_bound: upper_bound,
        feasible_users: feas.feasible_users,
    }
}

/// A single least-loaded greedy build, then the **ruin-and-recreate LNS** run
/// live: it clears the satellites around a still-blocked terminal, repacks that
/// cluster's colors, and keeps the rebuild only if it seats the blocked user.
/// The event stream replays the greedy first, then the users the search claws
/// back, so the recoveries pop in one by one at the end.
fn run_lns_only(scn: &Scenario, feas: &Feasibility) -> Trace {
    let n_users = scn.users.len();
    let n_sats = scn.sats.len();
    let (upper_bound, _) = matching::max_matching(&feas.sats, n_sats);

    let mut st = State::new(n_users, n_sats);
    let mut order: Vec<u32> = (0..n_users as u32)
        .filter(|&u| !feas.sats[u as usize].is_empty())
        .collect();
    order.sort_by_key(|&u| (feas.sats[u as usize].len(), u));
    for &u in &order {
        if st.user_sat[u as usize] >= 0 {
            continue;
        }
        let mut cands: Vec<u32> = feas.sats[u as usize].clone();
        cands.sort_by_key(|&s| (st.load(s as usize) as u32, s));
        for &s in &cands {
            if st.try_place(scn, u as usize, s as usize) {
                break;
            }
        }
    }
    let greedy_order: Vec<u32> = st.events.iter().map(|e| e.user).collect();

    lns(&mut st, scn, feas, upper_bound);

    // Rebuild the event stream from the final state: greedy users (final colors)
    // in their original order, then the LNS-recovered users appended.
    // Helper to find a user's color on a satellite; always succeeds within the state.
    let find_color = |s: usize, u: u32| -> u8 {
        st.sat_users[s]
            .iter()
            .position(|&x| x == u)
            .map(|i| st.sat_colors[s][i])
            .unwrap_or(0)
    };
    let mut events: Vec<Event> = Vec::with_capacity(greedy_order.len());
    let mut emitted = vec![false; n_users];
    for &u in &greedy_order {
        let s = st.user_sat[u as usize];
        if s >= 0 {
            events.push(Event {
                user: u,
                sat: s as u32,
                color: find_color(s as usize, u),
            });
            emitted[u as usize] = true;
        }
    }
    for (u, &s) in st.user_sat.iter().enumerate().take(n_users) {
        if s >= 0 && !emitted[u] {
            events.push(Event {
                user: u as u32,
                sat: s as u32,
                color: find_color(s as usize, u as u32),
            });
        }
    }

    let sat_load: Vec<usize> = st.sat_users.iter().map(|v| v.len()).collect();
    let unassigned = classify(scn, feas, &st.user_sat, &sat_load);
    Trace {
        events,
        unassigned,
        upper_bound,
        colored_bound: upper_bound,
        feasible_users: feas.feasible_users,
    }
}

/// Surgical ruin-and-recreate on the global `State`: repeatedly clear a blocked
/// terminal's candidate satellites, repack the freed users (plus the blocked
/// one) among just those satellites, and keep the result only if it serves more.
/// Restores exactly on rejection. Iteration-bounded; never records events.
fn lns(st: &mut State, scn: &Scenario, feas: &Feasibility, ub: usize) {
    let n_users = st.user_sat.len();
    let mut cur = st.user_sat.iter().filter(|&&s| s >= 0).count();
    let max_rounds = (n_users * 2).min(60_000);
    let mut cursor = 0usize;
    let mut freed: Vec<u32> = Vec::new();
    for _ in 0..max_rounds {
        if cur >= ub {
            break;
        }
        // Next still-unassigned feasible user.
        let mut x = None;
        for off in 0..n_users {
            let u = (cursor + off) % n_users;
            if st.user_sat[u] < 0 && !feas.sats[u].is_empty() {
                x = Some(u);
                cursor = (u + 1) % n_users;
                break;
            }
        }
        let Some(x) = x else { break };

        // Ruin: snapshot and clear x's candidate satellites.
        let cands = &feas.sats[x];
        let saved: Vec<(Vec<u32>, Vec<u8>, Vec<Vec3>)> = cands
            .iter()
            .map(|&s| {
                let s = s as usize;
                (
                    st.sat_users[s].clone(),
                    st.sat_colors[s].clone(),
                    st.sat_dirs[s].clone(),
                )
            })
            .collect();
        freed.clear();
        for &s in cands {
            let s = s as usize;
            for &u in &st.sat_users[s] {
                st.user_sat[u as usize] = -1;
                freed.push(u);
            }
            st.sat_users[s].clear();
            st.sat_colors[s].clear();
            st.sat_dirs[s].clear();
        }
        let removed = freed.len(); // all cleared users had been assigned
        freed.push(x as u32);
        // Recreate: re-place freed users (most-constrained first) onto the
        // cleared satellites only, so the repack is fully reversible.
        freed.sort_by_key(|&u| (feas.sats[u as usize].len(), u));
        for &u in &freed {
            if st.user_sat[u as usize] >= 0 {
                continue;
            }
            for &s in cands {
                if st.place(scn, u as usize, s as usize) {
                    break;
                }
            }
        }
        let reseated = freed
            .iter()
            .filter(|&&u| st.user_sat[u as usize] >= 0)
            .count();
        if reseated > removed {
            cur += reseated - removed; // a straggler was recovered — keep
        } else {
            // Reject: restore the cleared satellites exactly.
            for (i, &s) in cands.iter().enumerate() {
                let s = s as usize;
                for &u in &st.sat_users[s] {
                    st.user_sat[u as usize] = -1;
                }
                let (u, c, d) = &saved[i];
                st.sat_users[s] = u.clone();
                st.sat_colors[s] = c.clone();
                st.sat_dirs[s] = d.clone();
                for &u in &st.sat_users[s] {
                    st.user_sat[u as usize] = s as i32;
                }
            }
        }
    }
}

fn sort_candidates(
    cands: &mut [u32],
    scn: &Scenario,
    st: &State,
    sat_deg: &[u32],
    u: usize,
    algo: Algorithm,
) {
    match algo {
        // Both elevation-based variants pick the most-overhead satellite first.
        Algorithm::GreedyHighestElevation | Algorithm::GreedySpatial => {
            let zen = scn.users[u].unit();
            cands.sort_by(|&a, &b| {
                let ea = zen.dot((scn.sats[a as usize] - scn.users[u]).unit());
                let eb = zen.dot((scn.sats[b as usize] - scn.users[u]).unit());
                eb.total_cmp(&ea).then(a.cmp(&b))
            });
        }
        Algorithm::GreedyLeastContended => {
            cands.sort_by_key(|&s| (sat_deg[s as usize], st.load(s as usize) as u32, s));
        }
        // Least-loaded for both the plain greedy and the flow-seed gap-filler.
        _ => cands.sort_by_key(|&s| (st.load(s as usize) as u32, s)),
    }
}

/// Realize the optimal matching: place each matched user on its matched
/// satellite, adding per satellite in ascending intra-conflict order so a
/// near-maximum 4-colorable subset is kept.
fn flow_seed(st: &mut State, scn: &Scenario, matching: &[i32]) {
    let n_sats = st.sat_users.len();
    let mut by_sat: Vec<Vec<u32>> = vec![Vec::new(); n_sats];
    for (u, &s) in matching.iter().enumerate() {
        if s >= 0 {
            by_sat[s as usize].push(u as u32);
        }
    }
    for (s, users) in by_sat.iter().enumerate() {
        let dirs: Vec<Vec3> = users
            .iter()
            .map(|&u| (scn.users[u as usize] - scn.sats[s]).unit())
            .collect();
        let mut deg = vec![0u32; users.len()];
        for i in 0..users.len() {
            for j in (i + 1)..users.len() {
                if crate::geom::same_color_conflict(dirs[i], dirs[j]) {
                    deg[i] += 1;
                    deg[j] += 1;
                }
            }
        }
        let mut idx: Vec<usize> = (0..users.len()).collect();
        idx.sort_unstable_by_key(|&i| (deg[i], users[i]));
        for &i in &idx {
            st.try_place(scn, users[i] as usize, s);
        }
    }
}

/// Classify every unserved user given the final assignment (`user_sat[u] >= 0`
/// iff served) and per-satellite loads. Visibility-only counts come from a fresh
/// grid query (cheap — only users with no feasible satellite need it). Shared by
/// both the greedy traces and the optimized solver.
fn classify(
    scn: &Scenario,
    feas: &Feasibility,
    user_sat: &[i32],
    sat_load: &[usize],
) -> Vec<Unassigned> {
    let grid = Grid::build(&scn.sats, &scn.users);
    let mut out = Vec::new();
    for (u, &sat) in user_sat.iter().enumerate() {
        if sat >= 0 {
            continue;
        }
        let cand = &feas.sats[u];
        if cand.is_empty() {
            // Distinguish "nothing overhead" from "all overhead is interfered".
            let up = scn.users[u];
            let zen = up.unit();
            let mut in_view = 0u32;
            grid.for_candidates(up, |si| {
                if visible(zen, (scn.sats[si as usize] - up).unit()) {
                    in_view += 1;
                }
            });
            let reason = if in_view > 0 {
                Reason::AllInterfered
            } else {
                Reason::NoSatInView
            };
            out.push(Unassigned {
                user: u as u32,
                reason,
                in_view,
            });
        } else {
            let has_room = cand.iter().any(|&s| sat_load[s as usize] < 32);
            let reason = if has_room {
                Reason::ColorBlocked
            } else {
                Reason::AllFull
            };
            out.push(Unassigned {
                user: u as u32,
                reason,
                in_view: cand.len() as u32,
            });
        }
    }
    out
}
