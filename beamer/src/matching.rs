//! Exact maximum capacitated b-matching via Dinic's max-flow.
//!
//! This is the near-optimality *certificate*: the maximum number of users that
//! could be served if the coloring constraint did not exist (source→user cap 1,
//! user→feasible-sat cap 1, sat→sink cap 32). It upper-bounds any valid
//! solution. It is reported only — it never gates the assignment, since a
//! match-then-color strategy is known to collapse on dense inputs.

use std::collections::VecDeque;

struct Dinic {
    to: Vec<u32>,
    cap: Vec<i32>,
    head: Vec<Vec<u32>>, // edge indices out of each node
    level: Vec<i32>,
    iter: Vec<usize>,
}

impl Dinic {
    fn new(n: usize) -> Self {
        Dinic {
            to: Vec::new(),
            cap: Vec::new(),
            head: vec![Vec::new(); n],
            level: vec![0; n],
            iter: vec![0; n],
        }
    }
    fn add(&mut self, u: usize, v: usize, c: i32) {
        let e = self.to.len() as u32;
        self.to.push(v as u32);
        self.cap.push(c);
        self.head[u].push(e);
        self.to.push(u as u32);
        self.cap.push(0);
        self.head[v].push(e + 1);
    }
    fn bfs(&mut self, s: usize, t: usize) -> bool {
        self.level.iter_mut().for_each(|x| *x = -1);
        let mut q = VecDeque::new();
        self.level[s] = 0;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            for &e in &self.head[u] {
                let v = self.to[e as usize] as usize;
                if self.cap[e as usize] > 0 && self.level[v] < 0 {
                    self.level[v] = self.level[u] + 1;
                    q.push_back(v);
                }
            }
        }
        self.level[t] >= 0
    }
    fn dfs(&mut self, u: usize, t: usize, f: i32) -> i32 {
        if u == t {
            return f;
        }
        while self.iter[u] < self.head[u].len() {
            let e = self.head[u][self.iter[u]] as usize;
            let v = self.to[e] as usize;
            if self.cap[e] > 0 && self.level[v] == self.level[u] + 1 {
                let d = self.dfs(v, t, f.min(self.cap[e]));
                if d > 0 {
                    self.cap[e] -= d;
                    self.cap[e ^ 1] += d;
                    return d;
                }
            }
            self.iter[u] += 1;
        }
        0
    }
    fn max_flow(&mut self, s: usize, t: usize) -> i32 {
        let mut flow = 0;
        while self.bfs(s, t) {
            self.iter.iter_mut().for_each(|x| *x = 0);
            loop {
                let f = self.dfs(s, t, i32::MAX);
                if f == 0 {
                    break;
                }
                flow += f;
            }
        }
        flow
    }
}

/// Maximum users servable ignoring coloring on one component (the exact upper
/// bound when summed across components), together with one optimal assignment:
/// `matching[u]` is the local satellite user `u` is matched to, or `-1`.
pub fn max_matching(local_feas: &[Vec<u32>], num_sats: usize) -> (usize, Vec<i32>) {
    let num_users = local_feas.len();
    // node ids: 0=src, 1..=U users, U+1..=U+S sats, U+S+1 = sink
    let src = 0;
    let sink = num_users + num_sats + 1;
    let mut d = Dinic::new(sink + 1);
    for (u, cand) in local_feas.iter().enumerate() {
        if cand.is_empty() {
            continue;
        }
        d.add(src, 1 + u, 1);
        for &s in cand {
            d.add(1 + u, 1 + num_users + s as usize, 1);
        }
    }
    for s in 0..num_sats {
        d.add(1 + num_users + s, sink, 32);
    }
    let flow = d.max_flow(src, sink) as usize;

    // Recover the matching: a user→sat forward edge that carried a unit of flow
    // now has zero residual capacity (sat nodes are the only out-neighbors of a
    // user node besides the source back-edge, which points outside the sat range).
    let mut matching = vec![-1i32; num_users];
    for (u, entry) in matching.iter_mut().enumerate() {
        for &e in &d.head[1 + u] {
            let v = d.to[e as usize] as usize;
            if (1 + num_users..=num_users + num_sats).contains(&v) && d.cap[e as usize] == 0 {
                *entry = (v - 1 - num_users) as i32;
                break;
            }
        }
    }
    (flow, matching)
}

/// Maximum users servable on one component under an explicit **per-satellite
/// capacity** `caps[s]` (instead of a flat 32). Used for the coloring-aware
/// upper bound, where `caps[s]` is a valid ceiling on how many users satellite
/// `s` can 4-color. Returns the flow value only.
pub fn max_matching_capped(local_feas: &[Vec<u32>], caps: &[u32]) -> usize {
    let num_users = local_feas.len();
    let num_sats = caps.len();
    let src = 0;
    let sink = num_users + num_sats + 1;
    let mut d = Dinic::new(sink + 1);
    for (u, cand) in local_feas.iter().enumerate() {
        if cand.is_empty() {
            continue;
        }
        d.add(src, 1 + u, 1);
        for &s in cand {
            d.add(1 + u, 1 + num_users + s as usize, 1);
        }
    }
    for (s, &cap) in caps.iter().enumerate() {
        if cap > 0 {
            d.add(1 + num_users + s, sink, cap as i32);
        }
    }
    d.max_flow(src, sink) as usize
}
