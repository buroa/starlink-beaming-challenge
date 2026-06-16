//! Scenario parsing and solution output.

use crate::geom::Vec3;
use std::io::{self, BufWriter, Write};

/// A parsed scenario. Satellite and user ids are arbitrary strings in the input
/// (preserved verbatim for output) but interned to dense indices for the solver.
pub struct Scenario {
    pub sat_ids: Vec<String>,
    pub user_ids: Vec<String>,
    pub sats: Vec<Vec3>,
    pub users: Vec<Vec3>,
    pub interferers: Vec<Vec3>,
}

impl Scenario {
    pub fn parse(text: &str) -> Result<Scenario, String> {
        let mut s = Scenario {
            sat_ids: Vec::new(),
            user_ids: Vec::new(),
            sats: Vec::new(),
            users: Vec::new(),
            interferers: Vec::new(),
        };
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut t = line.split_whitespace();
            let kind = t.next().unwrap_or("");
            let id = t.next();
            let coords: Vec<&str> = t.collect();
            let pos = |c: &[&str]| -> Result<Vec3, String> {
                if c.len() != 3 {
                    return Err(format!("expected 3 coordinates, got line: {line}"));
                }
                let p = |v: &str| {
                    v.parse::<f64>()
                        .map_err(|_| format!("bad number in: {line}"))
                };
                Ok(Vec3::new(p(c[0])?, p(c[1])?, p(c[2])?))
            };
            match (kind, id) {
                ("sat", Some(id)) => {
                    s.sat_ids.push(id.to_string());
                    s.sats.push(pos(&coords)?);
                }
                ("user", Some(id)) => {
                    s.user_ids.push(id.to_string());
                    s.users.push(pos(&coords)?);
                }
                ("interferer", Some(_)) => s.interferers.push(pos(&coords)?),
                _ => return Err(format!("unrecognized line: {line}")),
            }
        }
        Ok(s)
    }
}

/// Counts reported in the `#` certificate header.
pub struct Certificate {
    pub total_users: usize,
    pub feasible_users: usize,
    pub upper_bound: usize,
    /// Tighter ceiling accounting for the 4-coloring rule.
    pub colored_bound: usize,
    pub achieved: usize,
}

const COLORS: [&str; 4] = ["A", "B", "C", "D"];

/// Write the certificate comment header and the beam allocation to `out`.
/// `per_sat[s]` is the list of `(user_index, color 0..3)` served by satellite
/// `s`; beam ids are assigned `1..=N` per satellite.
pub fn write_solution<W: Write>(
    out: W,
    scn: &Scenario,
    per_sat: &[Vec<(u32, u8)>],
    cert: &Certificate,
) -> io::Result<()> {
    let mut w = BufWriter::new(out);
    let pct = |n: usize, d: usize| {
        if d == 0 {
            100.0
        } else {
            n as f64 / d as f64 * 100.0
        }
    };
    writeln!(
        w,
        "# beam-planner: saturation-greedy + integral 4-coloring + augmenting repair"
    )?;
    writeln!(
        w,
        "# users_total={} feasible={}",
        cert.total_users, cert.feasible_users
    )?;
    writeln!(
        w,
        "# upper_bound (capacitated matching, ignoring coloring) = {} ({:.2}%)",
        cert.upper_bound,
        pct(cert.upper_bound, cert.total_users)
    )?;
    writeln!(
        w,
        "# upper_bound (+ 4-coloring clique cuts) = {} ({:.2}%)",
        cert.colored_bound,
        pct(cert.colored_bound, cert.total_users)
    )?;
    writeln!(
        w,
        "# achieved = {} ({:.2}%)   near-optimality achieved/bound = {:.2}%",
        cert.achieved,
        pct(cert.achieved, cert.total_users),
        pct(cert.achieved, cert.colored_bound)
    )?;
    for (si, members) in per_sat.iter().enumerate() {
        for (beam, &(ui, color)) in members.iter().enumerate() {
            writeln!(
                w,
                "sat {} beam {} user {} color {}",
                scn.sat_ids[si],
                beam + 1,
                scn.user_ids[ui as usize],
                COLORS[color as usize],
            )?;
        }
    }
    w.flush()
}
