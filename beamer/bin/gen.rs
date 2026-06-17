//! Scenario generator for the Starlink beam-planning solver.
//!
//! Emits a scenario in validator format — `sat` / `user` / `interferer` lines in
//! ECEF kilometres — at any scale, matching the conventions of the bundled
//! `test_cases`: a Walker constellation of satellites at 550 km, users on the
//! WGS84 ellipsoid within the served latitude band, and an optional geostationary
//! interferer belt. Deterministic for a given `--seed`.
//!
//! ```sh
//! # 1,000,000 users and 5,000 satellites → a file
//! cargo run --release --bin gen -- --users 1000000 --sats 5000 -o test_cases/big.txt
//!
//! # …or to stdout, piped straight into the solver
//! cargo run --release --bin gen -- -u 50000 -s 720 -i 36 | ./target/release/beamer /dev/stdin
//! ```
//!
//! Flags (all optional): `--users/-u N`, `--sats/-s N`, `--interferers/-i N`,
//! `--seed N`, `--inclination DEG` (0 = a single equatorial ring),
//! `--max-lat DEG` (user latitude band; default = inclination + 4°),
//! `--altitude KM` (default 550), `--out/-o FILE` (default stdout).

use std::io::{self, BufWriter, Write};

const R_EARTH: f64 = 6371.0; // km — spherical, as the bundled scenarios place satellites
const WGS84_A: f64 = 6378.137; // km — WGS84 semi-major axis (users sit on the ellipsoid)
const WGS84_F: f64 = 1.0 / 298.257_223_563; // WGS84 flattening
const GEO_ALT: f64 = 35_786.0; // km — geostationary altitude (interferer belt)
const PER_PLANE: usize = 20; // satellites per orbital plane (the test_cases convention)
const TAU: f64 = std::f64::consts::TAU;

/// SplitMix64 — a tiny, fast, seedable PRNG, so the same flags reproduce the same
/// scenario byte-for-byte (no `rand` dependency).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }
}

fn die(msg: &str) -> ! {
    eprintln!("gen: {msg}");
    eprintln!("usage: gen [--users N] [--sats N] [--interferers N] [--seed N] \
               [--inclination DEG] [--max-lat DEG] [--altitude KM] [--out FILE]");
    std::process::exit(2);
}

fn main() -> io::Result<()> {
    let mut users = 1_000usize;
    let mut sats = 360usize;
    let mut interferers = 0usize;
    let mut seed = 1u64;
    let mut inclination = 53.0f64;
    let mut max_lat = f64::NAN; // default derived from inclination below
    let mut altitude = 550.0f64;
    let mut out: Option<String> = None;

    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let next = |argv: &[String], i: &mut usize, flag: &str| -> String {
        *i += 1;
        argv.get(*i).cloned().unwrap_or_else(|| die(&format!("missing value for `{flag}`")))
    };
    while i < argv.len() {
        let flag = argv[i].clone();
        let mut num = || next(&argv, &mut i, &flag);
        match flag.as_str() {
            "--users" | "-u" => users = num().parse().unwrap_or_else(|_| die("--users wants an integer")),
            "--sats" | "-s" => sats = num().parse().unwrap_or_else(|_| die("--sats wants an integer")),
            "--interferers" | "-i" => interferers = num().parse().unwrap_or_else(|_| die("--interferers wants an integer")),
            "--seed" => seed = num().parse().unwrap_or_else(|_| die("--seed wants an integer")),
            "--inclination" => inclination = num().parse().unwrap_or_else(|_| die("--inclination wants a number")),
            "--max-lat" => max_lat = num().parse().unwrap_or_else(|_| die("--max-lat wants a number")),
            "--altitude" => altitude = num().parse().unwrap_or_else(|_| die("--altitude wants a number")),
            "--out" | "-o" => out = Some(num()),
            "--help" | "-h" => die("(help)"),
            other => die(&format!("unknown argument `{other}`")),
        }
        i += 1;
    }
    if max_lat.is_nan() {
        max_lat = (inclination.abs() + 4.0).min(89.0);
    }

    let writer: Box<dyn Write> = match &out {
        Some(path) => Box::new(std::fs::File::create(path)?),
        None => Box::new(io::stdout().lock()),
    };
    let mut w = BufWriter::new(writer);
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D).wrapping_add(1));

    // Header comments are ignored by the parser and validator; record the recipe.
    writeln!(
        w,
        "# generated: users={users} sats={sats} interferers={interferers} seed={seed} \
         inclination={inclination} max_lat={max_lat} altitude={altitude}"
    )?;

    write_satellites(&mut w, sats, R_EARTH + altitude, inclination.to_radians())?;
    write_users(&mut w, &mut rng, users, max_lat)?;
    write_interferers(&mut w, interferers, R_EARTH + GEO_ALT)?;

    w.flush()?;
    eprintln!("gen: wrote {sats} sats, {users} users, {interferers} interferers");
    Ok(())
}

/// A Walker constellation: `~PER_PLANE` satellites per orbital plane, each plane's
/// RAAN evenly spread and its satellites staggered (Walker phasing). An inclination
/// near 0 collapses to a single equatorial ring (`05_equatorial_plane`).
fn write_satellites(w: &mut impl Write, sats: usize, r: f64, inc: f64) -> io::Result<()> {
    if sats == 0 {
        return Ok(());
    }
    let planes = if inc.abs() < 1e-9 {
        1
    } else {
        ((sats as f64 / PER_PLANE as f64).round() as usize).clamp(1, sats)
    };
    let (si, ci) = inc.sin_cos();
    let mut id = 0usize;
    for p in 0..planes {
        let (sr, cr) = (TAU * p as f64 / planes as f64).sin_cos(); // RAAN
        let count = sats / planes + usize::from(p < sats % planes);
        let phase = TAU * p as f64 / sats as f64; // stagger planes against each other
        for k in 0..count {
            let (su, cu) = (TAU * k as f64 / count as f64 + phase).sin_cos();
            // In-plane point, tilted by the inclination (about x), then rotated to
            // the plane's ascending node (about z).
            let (xo, yo) = (r * cu, r * su);
            let (x1, y1, z) = (xo, yo * ci, yo * si);
            let (x, y) = (x1 * cr - y1 * sr, x1 * sr + y1 * cr);
            id += 1;
            writeln!(w, "sat {id} {x} {y} {z}")?;
        }
    }
    Ok(())
}

/// Users on the WGS84 ellipsoid surface: longitude uniform, latitude uniform
/// within ±`max_lat` (the band the constellation can actually serve).
fn write_users(w: &mut impl Write, rng: &mut Rng, users: usize, max_lat: f64) -> io::Result<()> {
    let e2 = WGS84_F * (2.0 - WGS84_F); // first eccentricity squared
    for id in 1..=users {
        let (slat, clat) = rng.range(-max_lat, max_lat).to_radians().sin_cos();
        let (slon, clon) = rng.range(-180.0, 180.0).to_radians().sin_cos();
        let n = WGS84_A / (1.0 - e2 * slat * slat).sqrt(); // prime-vertical radius
        let (x, y, z) = (n * clat * clon, n * clat * slon, n * (1.0 - e2) * slat);
        writeln!(w, "user {id} {x} {y} {z}")?;
    }
    Ok(())
}

/// Non-Starlink interferers as a geostationary belt: equatorial, evenly spaced in
/// longitude (`10_..._geo_belt` / the 100k case use 36).
fn write_interferers(w: &mut impl Write, n: usize, r: f64) -> io::Result<()> {
    for k in 0..n {
        let (s, c) = (TAU * k as f64 / n as f64).sin_cos();
        writeln!(w, "interferer {} {} {} 0", k + 1, r * c, r * s)?;
    }
    Ok(())
}
