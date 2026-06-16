//! Connected-component decomposition of the feasibility graph.
//!
//! A user only reaches satellites within its narrow visibility cone, and a
//! satellite only reaches nearby users; distinct connected components therefore
//! share no satellites and no users. Each component is a fully independent
//! subproblem — the assignment, coloring, repair, and upper bound of one
//! component cannot affect another — so they can be solved in parallel, and the
//! merged result is identical to solving the whole graph at once.

use crate::feasibility::Feasibility;

/// One independent subproblem: global user and satellite indices. Both lists are
/// ascending (so a global sat id maps to a local index by binary search).
pub struct Component {
    pub users: Vec<u32>,
    pub sats: Vec<u32>,
}

/// Union-find with path halving and union by rank.
struct Dsu {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (ra, rb) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else {
            (ra, rb)
        };
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
    }
}

/// Partition the feasibility graph into independent components. Users with no
/// feasible satellite are dropped (they can never be served). Components are
/// ordered by their smallest member user index; member lists are ascending.
pub fn decompose(feas: &Feasibility, num_sats: usize) -> Vec<Component> {
    let num_users = feas.sats.len();
    let mut dsu = Dsu::new(num_users + num_sats);
    for (u, cand) in feas.sats.iter().enumerate() {
        for &s in cand {
            dsu.union(u as u32, num_users as u32 + s);
        }
    }

    // Map each component root to a dense index, discovered in user order so the
    // resulting component list is deterministic.
    let mut root_to_comp = vec![u32::MAX; num_users + num_sats];
    let mut comps: Vec<Component> = Vec::new();
    for u in 0..num_users as u32 {
        if feas.sats[u as usize].is_empty() {
            continue;
        }
        let r = dsu.find(u) as usize;
        let ci = root_to_comp[r];
        let ci = if ci == u32::MAX {
            root_to_comp[r] = comps.len() as u32;
            comps.push(Component {
                users: Vec::new(),
                sats: Vec::new(),
            });
            comps.len() - 1
        } else {
            ci as usize
        };
        comps[ci].users.push(u);
    }
    for s in 0..num_sats as u32 {
        let r = dsu.find(num_users as u32 + s) as usize;
        let ci = root_to_comp[r];
        if ci != u32::MAX {
            comps[ci as usize].sats.push(s);
        }
    }
    comps
}
