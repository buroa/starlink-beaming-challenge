//! Beamer — the interactive GPU visualizer for the Starlink beam planner.
//!
//! Watch the solver assign satellite beams to user terminals in real time over a
//! 3D globe: switch scenarios and algorithms, toggle color bands and full/empty
//! satellites, control the playback speed, and inspect exactly why any terminal
//! could not be served.

mod camera;
mod scene;
mod tiles;

use beamer::trace::{self, Algorithm, Reason, Trace};
use beamer::{feasibility, io};
use camera::OrbitCamera;
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use glam::Vec3;
use scene::{BeamInstance, PointInstance, Scene};
use std::sync::Arc;

const BANDS: [&str; 4] = ["A", "B", "C", "D"];
/// Phosphor near-white — primary data text, gauge fills, reticle cores.
const WHITE: egui::Color32 = egui::Color32::from_rgb(232, 238, 244);
const DIM: egui::Color32 = egui::Color32::from_rgb(150, 160, 176);
/// Text-shadow under-draw for frameless readouts (legibility with no backdrop).
const SHADOW: egui::Color32 = egui::Color32::from_rgba_premultiplied(0, 0, 0, 170);
/// White at a given alpha — the monochrome "chrome" used for brackets, rules,
/// hairlines, and widget fills. (egui stores premultiplied: white@a == (a,a,a,a).)
const fn alpha(a: u8) -> egui::Color32 {
    egui::Color32::from_rgba_premultiplied(a, a, a, a)
}
/// Clear, distinct RGB band colors: A red, B green, C blue, D yellow.
const BAND_RGB: [[f32; 3]; 4] = [
    [1.00, 0.23, 0.23],
    [0.30, 1.00, 0.36],
    [0.30, 0.55, 1.00],
    [1.00, 0.85, 0.20],
];
fn band_rgb(c: u8) -> [f32; 3] {
    BAND_RGB.get(c as usize).copied().unwrap_or([1.0, 1.0, 1.0])
}
fn band_color32(c: u8) -> egui::Color32 {
    let r = band_rgb(c);
    egui::Color32::from_rgb(
        (r[0] * 255.0) as u8,
        (r[1] * 255.0) as u8,
        (r[2] * 255.0) as u8,
    )
}

/// Fixed sun direction (ECEF) — places a gentle day/night terminator.
fn sun_dir() -> Vec3 {
    Vec3::new(1.0, 0.18, 0.30).normalize()
}

/// Smooth camera fly-to animation.
#[derive(Clone, Copy)]
struct CamAnim {
    from: (f32, f32, f32), // yaw, pitch, distance
    to: (f32, f32, f32),
    t: f32, // 0..1
}

struct Loaded {
    scn: io::Scenario,
    feas: feasibility::Feasibility,
    user_pos: Vec<Vec3>,
    sat_pos: Vec<Vec3>,
    /// Non-Starlink interferer satellites (ECEF, f32 for the GPU).
    interferer_pos: Vec<Vec3>,
    /// Cached unit directions (normalized positions) for the per-frame hover
    /// horizon cull, so `pick()` never `normalize()`s every entity every frame.
    user_dir: Vec<Vec3>,
    sat_dir: Vec<Vec3>,
    interferer_dir: Vec<Vec3>,
    trace: Trace,
    /// Per-satellite event indices into `trace.events`, in assignment order.
    /// Lets focus mode extract one satellite's ≤32 beams without rescanning all
    /// ~100k events every frame.
    sat_events: Vec<Vec<u32>>,
    /// Event index per user (`-1` if never served). Each user gets at most one
    /// beam, so the hover tooltip resolves served/pending in O(1).
    user_event: Vec<i32>,
    /// Reason counts, in the order of `Reason` variants used below.
    reason_counts: [usize; 4],
}

struct ViewOpts {
    bands: [bool; 4],
    show_full: bool,
    show_empty: bool,
    show_uncovered: bool,
    show_beams: bool,
    show_interferers: bool,
}

/// Reusable per-frame scratch for [`compose`]/[`compose_focus`], so the live
/// render loop reuses (clears, not reallocates) the point/beam Vecs and the
/// user-color/sat-load index buffers each frame instead of malloc-ing ~7 MB.
#[derive(Default)]
struct ComposeBuffers {
    points: Vec<PointInstance>,
    beams: Vec<BeamInstance>,
    user_color: Vec<i8>,
    sat_load: Vec<u32>,
}

/// What the cursor is over, resolved once per frame and reused for both the
/// tooltip and (for an interferer) its field-of-interference overlay.
struct Hover {
    title: String,
    lines: Vec<String>,
    /// Assigned band (0..3) of a served terminal — tints the tooltip.
    band: Option<u8>,
    /// Index of a hovered interferer — draws its 20° field.
    interferer: Option<usize>,
}

/// The scene entity under the cursor, resolved by [`App::pick`].
#[derive(Clone, Copy, PartialEq)]
enum Picked {
    Sat(usize),
    User(usize),
    Interferer(usize),
}

/// An interferer counts as "around" a focused satellite when their geocentric
/// directions are within this angle — i.e. they share the same patch of sky.
const FOCUS_INTERFERER_COS: f32 = 0.906_307_8; // cos(25°)

/// Build the GPU point + beam lists for the first `revealed_f` beams. Shared by
/// the live UI and the headless screenshot. Clears and refills the caller's
/// reusable [`ComposeBuffers`], so the hot frame-loop never reallocates (up to
/// 100k users × 1440 satellites in the worst case).
fn compose(
    l: &Loaded,
    revealed_f: f64,
    o: &ViewOpts,
    selected: Option<usize>,
    hover_interferer: Option<usize>,
    buf: &mut ComposeBuffers,
) {
    let revealed = (revealed_f as usize).min(l.trace.events.len());
    let events = &l.trace.events[..revealed];

    buf.points.clear();
    buf.beams.clear();
    buf.user_color.clear();
    buf.user_color.resize(l.user_pos.len(), -1);
    buf.sat_load.clear();
    buf.sat_load.resize(l.sat_pos.len(), 0);
    // Aggregate which users are covered and satellite loads from events.
    for e in events {
        buf.user_color[e.user as usize] = e.color as i8;
        buf.sat_load[e.sat as usize] += 1;
    }

    // Terminals.
    let uncovered_color = [0.45, 0.47, 0.55, 0.6];
    for (i, p) in l.user_pos.iter().enumerate() {
        let pos = (*p * 1.002).to_array();
        let c = buf.user_color[i];
        if c >= 0 {
            if !o.bands[c as usize] {
                continue;
            }
            let rgb = BAND_RGB[c as usize];
            buf.points.push(PointInstance {
                pos,
                size: 5.5,
                color: [rgb[0], rgb[1], rgb[2], 1.0],
            });
        } else if o.show_uncovered {
            buf.points.push(PointInstance {
                pos,
                size: 4.0,
                color: uncovered_color,
            });
        }
    }

    // Satellites.
    for (i, p) in l.sat_pos.iter().enumerate() {
        let load = buf.sat_load[i];
        let full = load >= 32;
        if (full && !o.show_full) || (!full && !o.show_empty) {
            continue;
        }
        let color = if load == 0 {
            [0.40, 0.46, 0.58, 0.5]
        } else if full {
            [1.0, 0.92, 0.78, 1.0]
        } else {
            [0.70, 0.92, 1.0, 0.95]
        };
        let size = if load == 0 { 6.5 } else { 10.5 };
        buf.points.push(PointInstance {
            pos: p.to_array(),
            size,
            color,
        });
    }

    // Beams.
    if o.show_beams {
        for e in events {
            if !o.bands[e.color as usize] {
                continue;
            }
            let rgb = band_rgb(e.color);
            buf.beams.push(BeamInstance {
                a: l.sat_pos[e.sat as usize].to_array(),
                b: (l.user_pos[e.user as usize] * 1.002).to_array(),
                color: [rgb[0], rgb[1], rgb[2], 0.62],
            });
        }
    }

    // Inspector highlight: the selected unserved terminal + faint rays to the
    // satellites it can see (visible but unusable).
    if let Some(sel) = selected {
        if let Some(info) = l.trace.unassigned.get(sel) {
            let up = l.user_pos[info.user as usize];
            let upos = (up * 1.004).to_array();
            buf.points.push(PointInstance {
                pos: upos,
                size: 26.0,
                color: [1.0, 0.30, 0.38, 0.85],
            });
            buf.points.push(PointInstance {
                pos: upos,
                size: 13.0,
                color: [1.0, 1.0, 1.0, 1.0],
            });
            for &s in &l.feas.sats[info.user as usize] {
                buf.beams.push(BeamInstance {
                    a: l.sat_pos[s as usize].to_array(),
                    b: upos,
                    color: [1.0, 0.45, 0.5, 0.45],
                });
            }
        }
    }

    // Interferers (non-Starlink satellites) as distinct markers, plus the 20°
    // field of interference for whichever one the cursor is over.
    if o.show_interferers {
        for ip in &l.interferer_pos {
            buf.points.push(PointInstance {
                pos: ip.to_array(),
                size: 9.0,
                color: INTERFERER_RGBA,
            });
        }
        if let Some(hi) = hover_interferer {
            if let Some(&ip) = l.interferer_pos.get(hi) {
                push_interference_field(&mut buf.points, &mut buf.beams, ip);
            }
        }
    }
}

/// Magenta, distinct from the four band colors and the satellite-status palette.
const INTERFERER_RGBA: [f32; 4] = [0.85, 0.35, 1.0, 0.95];
/// The same magenta as readable UI text (for interferer labels/warnings).
const INTERFERER_UI: egui::Color32 = egui::Color32::from_rgb(217, 140, 255);
/// The interferer magenta at a given alpha (the GPU's four-channel array).
fn interferer_rgba(a: f32) -> [f32; 4] {
    [INTERFERER_RGBA[0], INTERFERER_RGBA[1], INTERFERER_RGBA[2], a]
}

/// Draw one interferer's 20° field of interference as a "bullseye" footprint on
/// the globe directly beneath it — a bold ring at 20° geocentric radius around
/// the sub-interferer point, filled with sparse radial spokes — plus an axis
/// line running out to the interferer marker itself. (The 20° rule is
/// observer-relative; this footprint is the clean, legible proxy for the region
/// the interferer sterilizes, drawn on the surface where it reads clearly rather
/// than buried inside the globe.)
fn push_interference_field(points: &mut Vec<PointInstance>, beams: &mut Vec<BeamInstance>, ip: Vec3) {
    const R: f32 = 6371.0 * 1.004; // just above the Earth's surface
    let axis = ip.normalize_or_zero();
    if axis == Vec3::ZERO {
        return;
    }
    // Orthonormal basis spanning the plane perpendicular to the axis.
    let seed = if axis.z.abs() < 0.95 { Vec3::Z } else { Vec3::X };
    let u = axis.cross(seed).normalize();
    let v = axis.cross(u);
    let (sin20, cos20) = 20f32.to_radians().sin_cos();

    let sub = axis * R; // sub-interferer point on the surface
    // Axis line out to the interferer, and a marker at the footprint centre.
    beams.push(BeamInstance {
        a: sub.to_array(),
        b: ip.to_array(),
        color: interferer_rgba(0.5),
    });
    points.push(PointInstance {
        pos: sub.to_array(),
        size: 11.0,
        color: interferer_rgba(0.95),
    });

    const N: usize = 64;
    let mut prev: Option<[f32; 3]> = None;
    for k in 0..=N {
        let t = k as f32 / N as f32 * std::f32::consts::TAU;
        let dir = cos20 * axis + sin20 * (t.cos() * u + t.sin() * v);
        let rim = (dir * R).to_array();
        // Sparse spokes from the centre fill the footprint disc.
        if k % 4 == 0 {
            beams.push(BeamInstance {
                a: sub.to_array(),
                b: rim,
                color: interferer_rgba(0.18),
            });
        }
        // The bold exclusion ring.
        if let Some(p) = prev {
            beams.push(BeamInstance {
                a: p,
                b: rim,
                color: interferer_rgba(0.7),
            });
        }
        prev = Some(rim);
    }
}

/// Local-frame scale factor for the flat focus view (km → local units).
const FOCUS_K: f32 = 2.0;

/// Framing for the flat focus schematic: the ENU basis at the sub-satellite
/// point, the displayed satellite height (capped so the cone aspect always
/// reads), and a `fit` radius bounding the served footprint so the camera and
/// grid frame *every* satellite well, whatever its real footprint size.
struct FocusLayout {
    east: Vec3,
    north: Vec3,
    sub: Vec3,
    sat_h: f32,
    fit: f32,
}

fn focus_layout(l: &Loaded, s: usize) -> FocusLayout {
    let sat = l.sat_pos[s];
    let up = sat.normalize_or_zero();
    let east = if up == Vec3::ZERO {
        Vec3::X
    } else if up.z.abs() < 0.95 {
        Vec3::Z.cross(up).normalize()
    } else {
        Vec3::X.cross(up).normalize()
    };
    let north = if up == Vec3::ZERO { Vec3::Y } else { up.cross(east) };
    let sub = up * 6371.0;
    // Footprint radius of the users this satellite serves (local units).
    let mut rmax = 0.0f32;
    for &ei in &l.sat_events[s] {
        let e = &l.trace.events[ei as usize];
        let d = l.user_pos[e.user as usize] - sub;
        let r = (d.dot(east).powi(2) + d.dot(north).powi(2)).sqrt() * FOCUS_K;
        rmax = rmax.max(r);
    }
    // Cap the drawn satellite height to the footprint so tiny clusters don't make
    // a tall empty spike and wide ones don't flatten — always a readable cone.
    let real_h = (sat.length() - 6371.0) * FOCUS_K;
    let sat_h = real_h.clamp(rmax * 0.9, rmax * 2.2).max(750.0);
    // The cone is centered vertically about the origin (ground at -sat_h/2, sat
    // at +sat_h/2), so the fit only needs half the height plus margin.
    let fit = (rmax * 1.3).max(sat_h * 0.8).max(900.0);
    FocusLayout { east, north, sub, sat_h, fit }
}

/// Satellite focus mode, rendered as a **flat schematic seen from above**: the
/// satellite hovers over a ground grid and beams straight down to the users it
/// serves, each projected into the satellite's local East/North/Up frame. The
/// globe + atmosphere are suppressed (see `Scene::set_focus_mode`) so this plane
/// stands alone over the nebula. Bands the user has toggled off are hidden.
fn compose_focus_flat(
    l: &Loaded,
    s: usize,
    focus_bands: [bool; 4],
    reveal: f64,
    t: f32,
    buf: &mut ComposeBuffers,
) {
    buf.points.clear();
    buf.beams.clear();
    if s >= l.sat_pos.len() {
        return;
    }
    let lay = focus_layout(l, s);
    let (east, north, sub) = (lay.east, lay.north, lay.sub);
    // Center the cone vertically about the origin so the camera (which targets
    // the origin) frames satellite-above-ground symmetrically: ground at -h/2,
    // satellite at +h/2.
    let z0 = -lay.sat_h * 0.5;
    let sat_local = Vec3::new(0.0, 0.0, z0 + lay.sat_h);
    let ext = lay.fit * 1.5;
    let foot = ext * 0.30; // footprint-disc radius under the satellite
    let ground = |e_: f32, n_: f32| Vec3::new(e_, n_, z0);

    // Ground grid + range rings (behind the data).
    push_ground_grid(&mut buf.beams, ext, z0, t);
    // Sparse radial spokes under the satellite — reads as "beaming straight down".
    for k in 0..28 {
        let th = k as f32 / 28.0 * std::f32::consts::TAU;
        buf.beams.push(BeamInstance {
            a: ground(0.0, 0.0).to_array(),
            b: ground(th.cos() * foot, th.sin() * foot).to_array(),
            color: [1.0, 1.0, 1.0, 0.07],
        });
    }

    // Beams down to served users, in assignment order up to `reveal`. Bands the
    // user toggled off are hidden but still counted, so the replay scrubber and
    // the gauge stay consistent with the full assignment.
    for (idx, &ei) in l.sat_events[s].iter().enumerate() {
        if (idx as f64) >= reveal {
            break;
        }
        let e = &l.trace.events[ei as usize];
        if !focus_bands[e.color as usize] {
            continue;
        }
        let d = l.user_pos[e.user as usize] - sub;
        let g = ground(d.dot(east) * FOCUS_K, d.dot(north) * FOCUS_K);
        let rgb = band_rgb(e.color);
        buf.beams.push(BeamInstance {
            a: sat_local.to_array(),
            b: g.to_array(),
            color: [rgb[0], rgb[1], rgb[2], 0.85],
        });
        buf.points.push(PointInstance {
            pos: g.to_array(),
            size: 7.5,
            color: [rgb[0], rgb[1], rgb[2], 1.0],
        });
    }

    // The satellite: a bright phosphor core wrapped in a flat lock-on reticle
    // lying parallel to the ground (so it reads from directly above).
    push_flat_reticle(&mut buf.beams, sat_local, t);
    buf.points.push(PointInstance {
        pos: sat_local.to_array(),
        size: 22.0,
        color: [0.91, 0.93, 0.96, 1.0],
    });

    // Nearest interferer sharing this patch of sky: a magenta marker hanging in
    // the sky toward its azimuth, dropping a thin axis to a small exclusion ring
    // on the ground — symbolic, but it shows where the 25° field comes from.
    let up = sub.normalize_or_zero();
    let nearest = l
        .interferer_pos
        .iter()
        .map(|ip| {
            let n = ip.normalize_or_zero();
            (n.dot(up), n)
        })
        .filter(|(d, _)| *d > FOCUS_INTERFERER_COS)
        .max_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, dir_i)) = nearest {
        let mut horiz = Vec3::new(dir_i.dot(east), dir_i.dot(north), 0.0);
        horiz = if horiz.length() > 1e-4 { horiz.normalize() } else { Vec3::X };
        let foot_g = ground(horiz.x * ext * 0.55, horiz.y * ext * 0.55);
        let marker = foot_g + Vec3::new(0.0, 0.0, lay.sat_h * 1.4);
        buf.beams.push(BeamInstance { a: marker.to_array(), b: foot_g.to_array(), color: interferer_rgba(0.45) });
        buf.points.push(PointInstance { pos: marker.to_array(), size: 11.0, color: interferer_rgba(0.95) });
        push_ring_xy(&mut buf.beams, foot_g, foot * 0.7, 0.5);
    }
}

/// A flat ground grid (minor/major lines + center crosshair) sized to `ext`,
/// plus the 5°/10°/20° range rings that fit, drawn as additive beam ribbons on
/// the z = 0 plane. One ring "pings" — its alpha breathes — so the satellite
/// visibly sweeps the ground.
fn push_ground_grid(beams: &mut Vec<BeamInstance>, ext: f32, z: f32, t: f32) {
    const N: i32 = 12;
    let step = ext / N as f32;
    let mut line = |a: Vec3, b: Vec3, w: f32| {
        beams.push(BeamInstance { a: a.to_array(), b: b.to_array(), color: [1.0, 1.0, 1.0, w] });
    };
    for i in -N..=N {
        let x = i as f32 * step;
        let w = if i == 0 { 0.7 } else if i % 5 == 0 { 0.42 } else { 0.22 };
        line(Vec3::new(x, -ext, z), Vec3::new(x, ext, z), w);
        line(Vec3::new(-ext, x, z), Vec3::new(ext, x, z), w);
    }
    for (k, &deg) in [5.0f32, 10.0, 20.0].iter().enumerate() {
        let r = 6371.0 * deg.to_radians().tan() * FOCUS_K;
        if r > ext * 1.15 {
            continue;
        }
        let a = if k == 1 { 0.40 + 0.20 * (t * 1.6).sin() } else { 0.28 };
        push_ring_xy(beams, Vec3::new(0.0, 0.0, z), r, a.max(0.10));
    }
}

/// A 64-segment ring of additive ribbons in a plane parallel to the ground,
/// centered at `center` (radius `r`, alpha `a`).
fn push_ring_xy(beams: &mut Vec<BeamInstance>, center: Vec3, r: f32, a: f32) {
    const N: usize = 64;
    let mut prev: Option<[f32; 3]> = None;
    for k in 0..=N {
        let th = k as f32 / N as f32 * std::f32::consts::TAU;
        let p = (center + Vec3::new(r * th.cos(), r * th.sin(), 0.0)).to_array();
        if let Some(pp) = prev {
            beams.push(BeamInstance { a: pp, b: p, color: [1.0, 1.0, 1.0, a] });
        }
        prev = Some(p);
    }
}

/// A flat lock-on reticle around the satellite, lying parallel to the ground:
/// a steady inner ring, a pulsing outer ring, and four diagonal corner ticks.
fn push_flat_reticle(beams: &mut Vec<BeamInstance>, c: Vec3, t: f32) {
    let pulse = 0.5 + 0.5 * (t * 2.2).sin();
    push_ring_xy(beams, c, 80.0, 0.7);
    push_ring_xy(beams, c, 150.0 + 55.0 * pulse, 0.25 + 0.3 * pulse);
    for k in 0..4 {
        let th = k as f32 * std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_4;
        let dir = Vec3::new(th.cos(), th.sin(), 0.0);
        beams.push(BeamInstance {
            a: (c + dir * 92.0).to_array(),
            b: (c + dir * 128.0).to_array(),
            color: [1.0, 1.0, 1.0, 0.65],
        });
    }
}

struct App {
    scene: Scene,
    device: wgpu::Device,
    renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    texture_id: egui::TextureId,

    scenarios: Vec<(String, String)>, // (label, path)
    current: usize,
    /// "Add your own…" mode: solve `custom_text` (an uploaded / dropped scenario)
    /// instead of a bundled one. Lets the viz double as the solver front end.
    custom: bool,
    custom_text: String,
    /// Collapsed (minimized to their header) states for the two glass panels, so a
    /// small / mobile screen can reclaim the space. Toggled by tapping the header.
    left_collapsed: bool,
    unserved_collapsed: bool,
    algo: Algorithm,
    loaded: Option<Loaded>,
    /// In-flight background solve (the production solver is too heavy to run on
    /// the UI thread — it would freeze the window for seconds).
    loading: Option<std::sync::mpsc::Receiver<Result<Loaded, String>>>,
    /// Text from an in-flight file-picker upload (wasm), delivered to the UI thread
    /// like `loading`. Drag-and-drop is handled inline in `update()` instead.
    upload_rx: Option<std::sync::mpsc::Receiver<String>>,
    error: Option<String>,

    camera: OrbitCamera,

    revealed: f64,
    playing: bool,
    /// Playback rate as a multiple of "whole timeline in ~12 s" — adapts to
    /// scenarios of wildly different size.
    speed_mult: f64,

    show_full: bool,
    show_empty: bool,
    show_uncovered: bool,
    show_beams: bool,
    show_interferers: bool,
    bands: [bool; 4],

    tile_source: tiles::TileSource,
    /// Fresnel atmosphere halo, toggleable independently of the basemap.
    show_atmosphere: bool,
    /// Window fullscreen state (opens fullscreen; F11 toggles, Esc exits).
    fullscreen: bool,
    /// Master HUD visibility (toggle with `H`).
    show_ui: bool,

    selected: Option<usize>, // index into trace.unassigned

    /// Satellite focus mode: isolate one satellite, its users, and any nearby
    /// interferer. `None` = normal whole-constellation view.
    focused_sat: Option<usize>,
    /// Scoped playback for the focused satellite's own beams (count revealed);
    /// `f64::INFINITY` shows the full assignment.
    focus_reveal: f64,
    focus_playing: bool,
    /// Per-band visibility for the focus view only — the left column's global
    /// `bands` stays untouched, so leaving focus restores the global view exactly.
    focus_bands: [bool; 4],
    /// Camera state saved on entering focus, restored on exit so the globe view
    /// returns exactly where it was (focus reuses `camera` for the flat scene).
    saved_cam: Option<(f32, f32, f32)>,

    time: f32,
    anim: Option<CamAnim>,

    /// Per-frame scratch reused across frames (no ~7 MB/frame reallocation).
    bufs: ComposeBuffers,
    /// Key of the inputs `compose` last consumed; lets a frame skip recompose +
    /// GPU upload when nothing the scene depends on changed (e.g. orbiting a
    /// paused timeline). `None` forces a rebuild.
    last_compose_key: Option<(usize, u16, i64, i64)>,
    /// Cached hit-test: (pointer, view-proj, show_interferers) → picked entity,
    /// so a still cursor over a still camera doesn't re-scan every entity.
    last_pick: Option<(egui::Pos2, glam::Mat4, bool, Option<Picked>)>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let rs = cc.wgpu_render_state.as_ref().expect("wgpu render state");
        let scene = Scene::new(rs.device.clone(), rs.queue.clone());
        let renderer = rs.renderer.clone();
        let texture_id = renderer.write().register_native_texture(
            &rs.device,
            scene.color_view(),
            wgpu::FilterMode::Linear,
        );

        // Native: scan ./test_cases for *.txt — the second tuple field is a PATH,
        // read lazily at solve time (the 100k file is 6.7 MB).
        #[cfg(not(target_arch = "wasm32"))]
        let scenarios: Vec<(String, String)> = {
            let mut s: Vec<(String, String)> = std::fs::read_dir(test_cases_dir())
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false))
                        .map(|p| {
                            let label = p
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or("?")
                                .to_string();
                            (label, p.to_string_lossy().into_owned())
                        })
                        .collect()
                })
                .unwrap_or_default();
            s.sort();
            s
        };
        // WASM has no filesystem: the second tuple field is a URL (relative to
        // the page), fetched on demand by `load()`.
        #[cfg(target_arch = "wasm32")]
        let scenarios: Vec<(String, String)> = [
            "00_example", "01_simplest_possible", "02_two_users", "03_five_users",
            "04_one_interferer", "05_equatorial_plane", "06_partially_fullfillable",
            "07_eighteen_planes", "08_eighteen_planes_northern", "09_ten_thousand_users",
            "10_ten_thousand_users_geo_belt", "11_one_hundred_thousand_users",
        ]
        .iter()
        .map(|n| (n.to_string(), format!("test_cases/{n}.txt")))
        .collect();

        let mut app = App {
            scene,
            device: rs.device.clone(),
            renderer,
            texture_id,
            scenarios,
            current: 0,
            custom: false,
            custom_text: String::new(),
            left_collapsed: false,
            unserved_collapsed: false,
            algo: Algorithm::Optimized,
            loaded: None,
            loading: None,
            upload_rx: None,
            error: None,
            camera: OrbitCamera::default(),
            revealed: 0.0,
            playing: true,
            speed_mult: 1.0,
            show_full: true,
            show_empty: true,
            show_uncovered: true,
            show_beams: true,
            show_interferers: false,
            bands: [true; 4],
            tile_source: tiles::TileSource::Off,
            show_atmosphere: true,
            fullscreen: true,
            show_ui: true,
            selected: None,
            focused_sat: None,
            focus_reveal: f64::INFINITY,
            focus_bands: [true; 4],
            saved_cam: None,
            focus_playing: false,
            time: 0.0,
            anim: None,
            bufs: ComposeBuffers::default(),
            last_compose_key: None,
            last_pick: None,
        };
        style_egui(&cc.egui_ctx);
        // Open on the headline 100k case with the basemap off, so the beam network
        // paints onto the transparent globe over the nebula — the prettiest first
        // look. A basemap (Dark / Light / Satellite) is one click away in GLOBE.
        if let Some(i) = app.scenarios.iter().position(|(l, _)| l.starts_with("11")) {
            app.current = i;
        }
        app.load();
        app
    }

    /// Fly the camera so `pos` (an ECEF point) faces the viewer, zoomed in.
    fn focus_on(&mut self, pos: Vec3) {
        let d = pos.normalize();
        let to = (d.y.atan2(d.x), d.z.asin().clamp(-1.4, 1.4), 9200.0);
        self.anim = Some(CamAnim {
            from: (self.camera.yaw, self.camera.pitch, self.camera.distance),
            to,
            t: 0.0,
        });
    }

    /// Kick off a background solve of the current scenario + algorithm. The
    /// result is picked up in `update()`; a "Solving…" overlay shows meanwhile,
    /// so the window never freezes (the 100k case takes a few seconds).
    fn load(&mut self) {
        self.error = None;
        self.loaded = None;
        self.selected = None;
        self.revealed = 0.0;
        self.playing = true;
        // Invalidate per-frame caches keyed by the old scenario.
        self.last_compose_key = None;
        self.last_pick = None;
        let algo = self.algo;

        // "Add your own…": solve the pasted text directly (no fetch / no file).
        if self.custom {
            let text = self.custom_text.clone();
            if text.trim().is_empty() {
                self.loading = None;
                return;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let (tx, rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let _ = tx.send(build_loaded(&text, algo));
                });
                self.loading = Some(rx);
            }
            #[cfg(target_arch = "wasm32")]
            {
                let (tx, rx) = std::sync::mpsc::channel();
                self.loading = Some(rx);
                wasm_bindgen_futures::spawn_local(async move {
                    let _ = tx.send(solve_text_via_worker(&text, algo).await);
                });
            }
            return;
        }

        let Some((_, src)) = self.scenarios.get(self.current).cloned() else {
            self.error = Some("No scenarios found in ./test_cases".into());
            self.loading = None;
            return;
        };
        // Native: solve on a background thread (`src` is a path), polled in
        // update() so the 100k case never freezes the window.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(load_scenario(&src, algo));
            });
            self.loading = Some(rx);
        }
        // WASM: `src` is a URL. Fetch it, then solve in a Web Worker (the solve
        // would otherwise block the render thread — ~32 s on the 100k case). The
        // worker returns the result serialized; we rebuild `Loaded` and deliver
        // it through the same mpsc channel the native path uses, so update()'s
        // poll is unchanged and the "Solving…" overlay shows throughout.
        #[cfg(target_arch = "wasm32")]
        {
            let (tx, rx) = std::sync::mpsc::channel();
            self.loading = Some(rx);
            wasm_bindgen_futures::spawn_local(async move {
                let _ = tx.send(solve_via_worker(&src, algo).await);
            });
        }
    }

    /// Export the current solution in validator format. On wasm this triggers a
    /// browser download; natively it writes a `<scenario>.solution.txt` in the cwd.
    fn download_solution(&mut self) {
        let Some(l) = &self.loaded else { return };
        let text = match solution_text(l) {
            Ok(t) => t,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };
        let name = if self.custom {
            "custom".to_string()
        } else {
            self.scenarios
                .get(self.current)
                .map(|s| s.0.clone())
                .unwrap_or_else(|| "scenario".into())
        };
        let filename = format!("{name}.solution.txt");
        #[cfg(target_arch = "wasm32")]
        download_text_js(&filename, &text);
        #[cfg(not(target_arch = "wasm32"))]
        match std::fs::write(&filename, &text) {
            Ok(()) => eprintln!("wrote {filename}"),
            Err(e) => self.error = Some(format!("write {filename}: {e}")),
        }
    }

    /// Solve an uploaded / dropped scenario: switch to "Add your own…" mode, store
    /// the text, and kick off the background solve (which clears the prior render).
    fn ingest_scenario(&mut self, text: String) {
        if text.trim().is_empty() {
            return;
        }
        self.custom = true;
        self.custom_text = text;
        self.exit_focus();
        self.load();
    }

    /// Leave "Add your own…" (via Esc or a click off the dialog) and restore the
    /// previously-selected bundled scenario. The pasted draft is kept so reopening
    /// the dialog doesn't lose it.
    fn exit_custom(&mut self) {
        self.custom = false;
        self.error = None;
        self.load();
    }

    /// Open the browser file picker; the chosen file's text arrives via `upload_rx`,
    /// polled in `update()`. (Native uses drag-and-drop instead.)
    #[cfg(target_arch = "wasm32")]
    fn open_file_picker(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.upload_rx = Some(rx);
        wasm_bindgen_futures::spawn_local(async move {
            if let Ok(v) = pick_scenario_js().await {
                if let Some(s) = v.as_string() {
                    let _ = tx.send(s);
                }
            }
        });
    }

    fn total_events(&self) -> usize {
        self.loaded
            .as_ref()
            .map(|l| l.trace.events.len())
            .unwrap_or(0)
    }

    /// Display label for the current scenario selection: the bundled scenario's
    /// name, or the "Add your own…" sentinel in paste mode.
    fn current_label(&self) -> &str {
        if self.custom {
            "Add your own…"
        } else {
            self.scenarios.get(self.current).map(|s| s.0.as_str()).unwrap_or("")
        }
    }

    /// Hit-test the scene entity under the cursor (front hemisphere only, when
    /// outside the globe). Shared by the hover tooltip and click-to-focus.
    fn pick(&self, ptr: egui::Pos2, rect: egui::Rect, vp: glam::Mat4) -> Option<Picked> {
        let l = self.loaded.as_ref()?;
        let eye = self.camera.eye();
        // Horizon-cull only from outside the globe; inside (core view) every
        // entity is potentially visible.
        let outside = eye.length() > 6371.0;
        let horizon = 6371.0 / eye.length();
        let eye_n = eye.normalize();
        let project = |p: Vec3| -> Option<egui::Pos2> {
            let c = vp * p.extend(1.0);
            if c.w <= 0.0 {
                return None;
            }
            Some(egui::pos2(
                rect.min.x + (c.x / c.w * 0.5 + 0.5) * rect.width(),
                rect.min.y + (0.5 - c.y / c.w * 0.5) * rect.height(),
            ))
        };
        let mut best: Option<(f32, Picked)> = None;
        for (i, sp) in l.sat_pos.iter().enumerate() {
            if outside && l.sat_dir[i].dot(eye_n) < horizon - 0.2 {
                continue;
            }
            if let Some(s) = project(*sp) {
                let dd = s.distance(ptr);
                if dd < 13.0 && best.as_ref().is_none_or(|(b, _)| dd < *b) {
                    best = Some((dd, Picked::Sat(i)));
                }
            }
        }
        const USER_THRESHOLD: f32 = 9.0;
        for (i, up) in l.user_pos.iter().enumerate() {
            if outside && l.user_dir[i].dot(eye_n) < horizon {
                continue;
            }
            if let Some(s) = project(*up) {
                let dd = s.distance(ptr);
                if dd < USER_THRESHOLD && best.as_ref().is_none_or(|(b, _)| dd < *b) {
                    best = Some((dd, Picked::User(i)));
                }
            }
        }
        // Interferers (only when shown). Cull the far hemisphere like satellites,
        // so an interferer hidden behind an opaque globe isn't pickable through it.
        if self.show_interferers {
            for (i, ip) in l.interferer_pos.iter().enumerate() {
                if outside && l.interferer_dir[i].dot(eye_n) < horizon - 0.2 {
                    continue;
                }
                if let Some(s) = project(*ip) {
                    let dd = s.distance(ptr);
                    if dd < 13.0 && best.as_ref().is_none_or(|(b, _)| dd < *b) {
                        best = Some((dd, Picked::Interferer(i)));
                    }
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// Cached hit-test: reuse the last frame's pick when the pointer and camera
    /// haven't moved, so a still cursor doesn't re-scan every entity each frame.
    fn pick_cached(&mut self, ptr: egui::Pos2, rect: egui::Rect, vp: glam::Mat4) -> Option<Picked> {
        let si = self.show_interferers;
        if let Some((p, m, s, cached)) = self.last_pick {
            if p == ptr && m == vp && s == si {
                return cached;
            }
        }
        let picked = self.pick(ptr, rect, vp);
        self.last_pick = Some((ptr, vp, si, picked));
        picked
    }

    /// Tooltip (title, detail lines, band tint, interferer index) for an
    /// already-resolved [`Picked`] entity. Depends on `revealed` (the beam count
    /// updates as playback proceeds), so it is re-formatted each frame even when
    /// the pick itself is served from cache.
    fn hover_entity(&self, picked: Picked) -> Option<Hover> {
        let l = self.loaded.as_ref()?;
        let revealed = (self.revealed as usize).min(l.trace.events.len());
        match picked {
            Picked::Sat(i) => {
                // `sat_events[i]` is ascending event order, so the beams revealed
                // so far is simply how many of its indices precede `revealed`.
                let load = l.sat_events[i].partition_point(|&ei| (ei as usize) < revealed);
                Some(Hover {
                    title: format!("Satellite {}", l.scn.sat_ids[i]),
                    lines: vec![format!("{load} / 32 beams in use"), "click to focus →".into()],
                    band: None,
                    interferer: None,
                })
            }
            Picked::User(i) => {
                // Each user has at most one beam (`user_event[i]`), so served /
                // pending resolves in O(1) — no scan of the revealed events.
                let title = format!("Terminal {}", l.scn.user_ids[i]);
                let ev = l.user_event[i];
                if ev >= 0 && (ev as usize) < revealed {
                    let e = &l.trace.events[ev as usize];
                    Some(Hover {
                        title,
                        lines: vec![format!(
                            "served · band {} · sat {}",
                            BANDS[e.color as usize], l.scn.sat_ids[e.sat as usize]
                        )],
                        band: Some(e.color),
                        interferer: None,
                    })
                } else {
                    // Served-but-not-yet-revealed needs no lookup; only a genuine
                    // miss (`ev < 0`) consults the unassigned list for its reason.
                    let line = if ev >= 0 {
                        "not yet assigned".to_string()
                    } else {
                        l.trace
                            .unassigned
                            .iter()
                            .find(|u| u.user as usize == i)
                            .map(|u| u.reason.label().to_string())
                            .unwrap_or_else(|| "not yet assigned".into())
                    };
                    Some(Hover {
                        title,
                        lines: vec![line],
                        band: None,
                        interferer: None,
                    })
                }
            }
            Picked::Interferer(i) => Some(Hover {
                title: format!("Interferer {}", l.scn.interferer_ids[i]),
                lines: vec!["non-Starlink · 20° exclusion field".into()],
                band: None,
                interferer: Some(i),
            }),
        }
    }

    /// Enter satellite focus mode: isolate satellite `s` as a flat schematic.
    /// No fly-in (per brief) — the camera snaps to a near-nadir framing of the
    /// ground plane. The globe-view camera is saved so exiting restores it.
    fn enter_focus(&mut self, s: usize) {
        if self.saved_cam.is_none() {
            self.saved_cam = Some((self.camera.yaw, self.camera.pitch, self.camera.distance));
        }
        self.focused_sat = Some(s);
        self.focus_reveal = f64::INFINITY; // show the full assignment first
        self.focus_playing = false;
        self.focus_bands = [true; 4];
        self.selected = None;
        self.anim = None;
        // Reuse the orbit camera (it targets the origin) for the local flat
        // scene: near straight-down, keeping the current azimuth, with distance
        // fit to this satellite's footprint (45° vfov → half-extent = d·tan22.5°).
        self.camera.pitch = 1.22; // high-oblique "from above" — ground reads as a plane
        let fit = self
            .loaded
            .as_ref()
            .map(|l| focus_layout(l, s).fit)
            .unwrap_or(2400.0);
        self.camera.distance = (fit / 0.4142 * 1.12).clamp(1800.0, 20000.0);
    }

    /// Leave satellite focus mode, restoring the saved globe-view camera.
    fn exit_focus(&mut self) {
        self.focused_sat = None;
        self.anim = None;
        if let Some((y, p, d)) = self.saved_cam.take() {
            self.camera.yaw = y;
            self.camera.pitch = p;
            self.camera.distance = d;
        }
    }

    /// Beams assigned to the focused satellite (assignment order).
    fn focus_beam_count(&self, s: usize) -> usize {
        self.loaded
            .as_ref()
            .and_then(|l| l.sat_events.get(s))
            .map(|ev| ev.len())
            .unwrap_or(0)
    }

    /// Refill `self.bufs` with the points/beams for the current frame. In focus
    /// mode the whole-constellation view is replaced by an isolated study.
    fn build_world(&mut self, hover_interferer: Option<usize>) {
        let focused = self.focused_sat;
        let revealed = self.revealed;
        let selected = self.selected;
        let focus_reveal = self.focus_reveal;
        let focus_bands = self.focus_bands;
        let time = self.time;
        let opts = focused.is_none().then(|| self.view_opts());
        let Some(l) = self.loaded.as_ref() else {
            self.bufs.points.clear();
            self.bufs.beams.clear();
            return;
        };
        if let Some(s) = focused {
            compose_focus_flat(l, s, focus_bands, focus_reveal, time, &mut self.bufs);
        } else if let Some(opts) = opts {
            compose(l, revealed, &opts, selected, hover_interferer, &mut self.bufs);
        }
    }

    fn view_opts(&self) -> ViewOpts {
        ViewOpts {
            bands: self.bands,
            show_full: self.show_full,
            show_empty: self.show_empty,
            show_uncovered: self.show_uncovered,
            show_beams: self.show_beams,
            show_interferers: self.show_interferers,
        }
    }

    /// Pack the layer/band toggles into bits for the recompose dirty-check key.
    fn view_opts_bits(&self) -> u16 {
        let mut b = 0u16;
        for (i, &on) in self.bands.iter().enumerate() {
            if on {
                b |= 1 << i;
            }
        }
        let flags = [
            self.show_full,
            self.show_empty,
            self.show_uncovered,
            self.show_beams,
            self.show_interferers,
        ];
        for (i, &on) in flags.iter().enumerate() {
            if on {
                b |= 1 << (4 + i);
            }
        }
        b
    }

    // Top-left: every selector + toggle in one always-visible glass column.
    fn left_panel(&mut self, ui: &mut egui::Ui) {
        // Wide enough that the longest scenario name and the layer toggles (two
        // rows) fit without overflowing — the panel is a fixed, tidy width and
        // combos truncate rather than stretch the backdrop.
        const W: f32 = 290.0;
        ui.set_width(W);
        // Header doubles as a minimize control: tap the title or the chevron to
        // collapse the panel to just this row (reclaims space on small screens).
        ui.horizontal(|ui| {
            let title = ui.add(
                egui::Label::new(egui::RichText::new("BEAMER").color(WHITE).strong().size(15.0))
                    .sense(egui::Sense::click()),
            );
            if title.clicked() {
                self.left_collapsed = !self.left_collapsed;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let chevron = if self.left_collapsed { "▸" } else { "▾" };
                if ui
                    .add(
                        egui::Label::new(egui::RichText::new(chevron).color(DIM).size(13.0))
                            .sense(egui::Sense::click()),
                    )
                    .on_hover_text(if self.left_collapsed { "Expand" } else { "Minimize" })
                    .clicked()
                {
                    self.left_collapsed = !self.left_collapsed;
                }
                // Blinking LIVE telltale (globe view only — focus mode carries its
                // own ● LOCK telltale).
                if self.playing && self.loaded.is_some() && self.focused_sat.is_none() {
                    let p = (0.5 + 0.5 * (self.time * 3.0).sin()) * 255.0;
                    let c = egui::Color32::from_rgba_unmultiplied(120, 230, 160, p as u8);
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("● LIVE").color(c).size(9.5).strong());
                }
            });
        });
        if self.left_collapsed {
            return;
        }
        ui.add_space(11.0);

        // ── SCENARIO (+ a Download-solution button to its right) ─────────────
        section(ui, "SCENARIO");
        let mut changed_scn = false;
        // The Matching view is the capacitated-matching upper bound (4-coloring
        // ignored) — not a valid solution — so it isn't downloadable.
        let can_download = self.loaded.is_some() && self.algo != Algorithm::Matching;
        ui.horizontal(|ui| {
            let combo_w = if can_download { W - 36.0 } else { W };
            let current_label = self.current_label();
            let combo = egui::ComboBox::from_id_salt("scn")
                .selected_text(current_label)
                .width(combo_w)
                .truncate()
                .show_ui(ui, |ui| {
                    for (i, (label, _)) in self.scenarios.iter().enumerate() {
                        if ui.selectable_label(!self.custom && self.current == i, label).clicked() {
                            self.current = i;
                            self.custom = false;
                            changed_scn = true;
                        }
                    }
                    ui.separator();
                    // Add-your-own: clear the previous solve on select (load() drops
                    // `loaded`, and the empty-text branch returns without solving, so
                    // the globe stays empty until a file is uploaded / dropped).
                    if ui.selectable_label(self.custom, "Add your own…").clicked() && !self.custom {
                        self.custom = true;
                        changed_scn = true;
                    }
                });
            // Match the combo's exact height so the button aligns flush with it.
            if can_download
                && ui
                    .add_sized([28.0, combo.response.rect.height()], egui::Button::new("↓"))
                    .on_hover_text("Download the validator-format solution (certificate + beam allocation)")
                    .clicked()
            {
                self.download_solution();
            }
        });

        // (Upload / paste / drag-and-drop for "Add your own…" is the center dialog.)

        // ── ALGORITHM ───────────────────────────────────────────────────────
        ui.add_space(10.0);
        section(ui, "ALGORITHM");
        let mut changed_algo = false;
        egui::ComboBox::from_id_salt("algo")
            .selected_text(self.algo.name())
            .width(W)
            .truncate()
            .show_ui(ui, |ui| {
                for a in Algorithm::ALL {
                    if ui.selectable_label(self.algo == a, a.name()).clicked() {
                        self.algo = a;
                        changed_algo = true;
                    }
                }
            });

        // ── DISPLAY (color bands + scene layers) ─────────────────────────────
        ui.add_space(10.0);
        section(ui, "DISPLAY");
        // When a satellite is focused, the A/B/C/D chips show *its* per-band beam
        // counts ("A 8") and toggle that satellite's bands; otherwise they're the
        // global band filter ("A").
        let focus_counts: Option<[u32; 4]> = self.focused_sat.and_then(|s| {
            self.loaded.as_ref().map(|l| {
                let mut bc = [0u32; 4];
                for &ei in &l.sat_events[s] {
                    bc[l.trace.events[ei as usize].color as usize] += 1;
                }
                bc
            })
        });
        ui.horizontal(|ui| {
            let cw = (W - 3.0 * ui.spacing().item_spacing.x) / 4.0;
            for c in 0..4u8 {
                let (on, label) = match focus_counts {
                    Some(bc) => (self.focus_bands[c as usize], format!("{} {}", BANDS[c as usize], bc[c as usize])),
                    None => (self.bands[c as usize], BANDS[c as usize].to_string()),
                };
                if band_chip(ui, c, on, &label, [cw, 26.0]).clicked() {
                    if focus_counts.is_some() {
                        self.focus_bands[c as usize] = !self.focus_bands[c as usize];
                    } else {
                        self.bands[c as usize] = !self.bands[c as usize];
                    }
                }
            }
        });
        ui.add_space(5.0);
        // Scene layers as a filled grid (no ragged white space): three then two,
        // each cell stretched to fill its row.
        let sp = ui.spacing().item_spacing.x;
        let (w3, w2) = ((W - 2.0 * sp) / 3.0, (W - sp) / 2.0);
        ui.horizontal(|ui| {
            layer_toggle(ui, w3, &mut self.show_beams, "Beams");
            layer_toggle(ui, w3, &mut self.show_full, "Full");
            layer_toggle(ui, w3, &mut self.show_empty, "Partial");
        });
        ui.horizontal(|ui| {
            layer_toggle(ui, w2, &mut self.show_uncovered, "Uncovered");
            layer_toggle(ui, w2, &mut self.show_interferers, "Interferers")
                .on_hover_text("Non-Starlink satellites — hover one to see its 20° field of interference");
        });

        // ── GLOBE (basemap + atmosphere-halo toggle) ─────────────────────────
        ui.add_space(10.0);
        section(ui, "GLOBE");
        let mut pick = None;
        ui.horizontal(|ui| {
            let combo = egui::ComboBox::from_id_salt("basemap")
                .selected_text(self.tile_source.label())
                .width(W - 36.0)
                .truncate()
                .show_ui(ui, |ui| {
                    for src in tiles::TileSource::ALL {
                        if ui.selectable_label(self.tile_source == src, src.label()).clicked() {
                            pick = Some(src);
                        }
                    }
                });
            // Atmosphere-halo toggle, flush to the right of the basemap dropdown.
            // "○" reads as the halo ring; highlighted when the glow is on.
            if ui
                .add_sized(
                    [28.0, combo.response.rect.height()],
                    egui::Button::selectable(self.show_atmosphere, "○"),
                )
                .on_hover_text("Atmosphere halo — Fresnel glow, independent of the basemap")
                .clicked()
            {
                self.show_atmosphere = !self.show_atmosphere;
            }
        });
        if let Some(src) = pick {
            self.tile_source = src;
            self.scene.set_tile_source(src);
        }

        if changed_scn {
            // A different scenario means different satellites — drop the focus.
            self.exit_focus();
            self.load();
        } else if changed_algo {
            // Same satellites: keep focus (no exit_focus) so the re-solve
            // re-renders for the currently focused one.
            self.load();
        }
    }

    // Center dialog for "Add your own…": upload a file, paste text, or drop a file.
    // Shown while in custom mode with nothing solved yet; once a solve lands the
    // normal HUD returns. Painted with the same glass + brackets as every panel.
    fn custom_dialog(&mut self, ui: &mut egui::Ui) {
        const W: f32 = 330.0;
        ui.set_width(W);
        shadow_text(ui, "ADD YOUR OWN", 17.0, WHITE);
        ui.label(
            egui::RichText::new("Upload, paste, or drop a scenario — validator format.")
                .color(DIM)
                .size(10.0),
        );
        ui.add_space(11.0);

        #[cfg(target_arch = "wasm32")]
        {
            if ui
                .add_sized([W, 30.0], egui::Button::new("⬆  Upload a scenario file"))
                .clicked()
            {
                self.open_file_picker();
            }
            ui.add_space(10.0);
        }

        section(ui, "OR PASTE");
        ui.add(
            egui::TextEdit::multiline(&mut self.custom_text)
                .hint_text("# satellites / users / interferers in ECEF…")
                .desired_rows(6)
                .desired_width(W)
                .font(egui::TextStyle::Monospace),
        );
        ui.add_space(6.0);
        let can_solve = !self.custom_text.trim().is_empty();
        if ui
            .add_enabled(can_solve, egui::Button::new("Solve").min_size(egui::vec2(W, 28.0)))
            .clicked()
        {
            self.load();
        }
        // A parse / validation failure (e.g. a comment-only paste) lands here
        // rather than rendering an empty scene.
        if let Some(e) = &self.error {
            ui.add_space(7.0);
            ui.label(
                egui::RichText::new(format!("⚠ {e}"))
                    .color(egui::Color32::from_rgb(255, 120, 95))
                    .size(10.5),
            );
        }
        ui.add_space(9.0);
        ui.label(
            egui::RichText::new("…or drag & drop a scenario file anywhere")
                .color(DIM)
                .size(10.0),
        );
    }

    // Top-right: the coverage headline. (The loading state is drawn as a
    // full-screen acquisition overlay, so nothing is shown here while solving.)
    fn coverage_module(&self, ui: &mut egui::Ui) {
        if self.loading.is_some() {
            return;
        }
        // Pin the width: a right-anchored Area otherwise gives the content ~0
        // wrap width, collapsing the headline into one glyph per line.
        ui.set_width(196.0);
        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
            if let Some(l) = &self.loaded {
                let covered = (self.revealed as usize).min(l.trace.events.len());
                let total = l.user_pos.len().max(1);
                // The Matching view is the *upper bound*, drawn ignoring the
                // 4-color rule — not a valid solution. Label it as such so its
                // count never reads as a real (let alone optimal) result.
                let is_bound = self.algo == Algorithm::Matching;
                ui.label(
                    egui::RichText::new(format!("{:.2}%", covered as f64 / total as f64 * 100.0))
                        .size(30.0)
                        .strong()
                        .color(if is_bound {
                            egui::Color32::from_rgb(255, 205, 95)
                        } else {
                            WHITE
                        }),
                );
                if is_bound {
                    ui.label(
                        egui::RichText::new("UPPER BOUND")
                            .color(egui::Color32::from_rgb(255, 205, 95))
                            .size(9.0),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(format!("{covered} / {total}"))
                            .color(WHITE)
                            .size(12.0),
                    );
                    ui.label(
                        egui::RichText::new("coloring ignored · not a valid solution")
                            .color(DIM)
                            .size(10.0),
                    );
                } else {
                    ui.label(
                        egui::RichText::new("TERMINALS COVERED")
                            .color(DIM)
                            .size(9.0),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        egui::RichText::new(format!("{covered} / {total}"))
                            .color(WHITE)
                            .size(12.0),
                    );
                    if l.trace.colored_bound > 0 {
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.1}% of optimum · bound {}",
                                l.trace.events.len() as f64 / l.trace.colored_bound as f64 * 100.0,
                                l.trace.colored_bound
                            ))
                            .color(DIM)
                            .size(10.0),
                        );
                    }
                }
            }
            // Surface a load/solve/download error even while a previous solution is
            // still shown — download failures set `error` with `loaded` non-None.
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            }
        });
    }

    // Bottom-center: transport — rerun, restart, play/pause, scrubber, speed.
    fn transport_module(&mut self, ui: &mut egui::Ui, compact: bool) {
        const SPEEDS: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 8.0];
        let rate_label = |m: f64| if m < 1.0 { format!("{m}×") } else { format!("{}×", m as i64) };
        let total = self.total_events();
        let done = (self.revealed as usize).min(total);

        // The bar auto-sizes to its content (no fixed width → no empty space). Only
        // the scrubber scales with the viewport; it's bounded so the bar still clears
        // the bottom-right unserved card. The count/speed cluster sits flush after it.
        let screen_w = ui.ctx().content_rect().width();
        let scrubber_w = if compact {
            (screen_w - 470.0).clamp(80.0, 200.0)
        } else {
            (screen_w - 1080.0).clamp(150.0, 240.0)
        };

        ui.horizontal(|ui| {
            // Left transport controls (RERUN icon-only when compact).
            let (rerun, rerun_w) = if compact { ("⟲", 30.0) } else { ("⟲ RERUN", 84.0) };
            if ui
                .add_sized([rerun_w, 28.0], egui::Button::new(rerun))
                .on_hover_text("Re-solve the scenario with the selected algorithm")
                .clicked()
            {
                self.load();
            }
            if ui
                .add_sized([if compact { 30.0 } else { 34.0 }, 28.0], egui::Button::new("⏮"))
                .on_hover_text("Restart")
                .clicked()
            {
                self.revealed = 0.0;
                self.playing = true;
            }
            let lbl = if self.playing { "[ ⏸ ]" } else { "[ ▶ ]" };
            if ui
                .add_sized(
                    [if compact { 40.0 } else { 46.0 }, 28.0],
                    egui::Button::new(egui::RichText::new(lbl).size(14.0)),
                )
                .clicked()
            {
                if done >= total {
                    self.revealed = 0.0;
                }
                self.playing = !self.playing;
            }

            // Set the rail width directly (not via add_sized, which would leave the
            // rail un-stretched inside a padded box → visible whitespace).
            let mut rev = self.revealed.min(total as f64);
            ui.spacing_mut().slider_width = scrubber_w;
            let resp = ui.add(
                egui::Slider::new(&mut rev, 0.0..=(total.max(1) as f64)).show_value(false),
            );
            if resp.dragged() || resp.changed() {
                self.revealed = rev;
                self.playing = false;
            }

            if compact {
                // One cycling speed chip flush after the scrubber — saves width, and
                // the top-right readout already carries the covered count.
                if ui
                    .add_sized([40.0, 24.0], egui::Button::new(rate_label(self.speed_mult)))
                    .on_hover_text("Playback speed (tap to cycle)")
                    .clicked()
                {
                    let i = SPEEDS.iter().position(|&m| (m - self.speed_mult).abs() < 1e-6).unwrap_or(1);
                    self.speed_mult = SPEEDS[(i + 1) % SPEEDS.len()];
                }
            } else {
                // The count sits flush against the scrubber's right edge ("up to the
                // numbers"); RATE presets follow.
                ui.label(egui::RichText::new(format!("{done} / {total}")).color(WHITE).size(11.0));
                ui.label(egui::RichText::new("RATE").color(DIM).size(10.0));
                for &m in &SPEEDS {
                    if ui.selectable_label((self.speed_mult - m).abs() < 1e-6, rate_label(m)).clicked() {
                        self.speed_mult = m;
                    }
                }
            }
        });
    }

    // Unserved card: why terminals failed + jump-to list. Returns a clicked index.
    fn unserved_module(&mut self, ui: &mut egui::Ui) -> Option<usize> {
        const W: f32 = 250.0;
        let n_unserved = self.loaded.as_ref().map(|l| l.trace.unassigned.len())?;
        ui.set_width(W);
        let mut clicked = None;
        // Header spans the full width: title left, fault count + minimize chevron
        // right. Tapping the title or the chevron collapses the card.
        let mut toggle = false;
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::Label::new(egui::RichText::new("UNSERVED // FAULTS").color(DIM).size(10.0).strong())
                        .sense(egui::Sense::click()),
                )
                .on_hover_text(if self.unserved_collapsed { "Expand" } else { "Minimize" })
                .clicked()
            {
                toggle = true;
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let chevron = if self.unserved_collapsed { "▸" } else { "▾" };
                if ui
                    .add(egui::Label::new(egui::RichText::new(chevron).color(DIM).size(10.0)).sense(egui::Sense::click()))
                    .clicked()
                {
                    toggle = true;
                }
                ui.add_space(4.0);
                ui.label(egui::RichText::new(format!("×{n_unserved}")).color(DIM).size(10.0).strong());
            });
        });
        if toggle {
            self.unserved_collapsed = !self.unserved_collapsed;
        }
        if self.unserved_collapsed {
            return None;
        }

        // Body: right-aligned so it hugs the bottom-right corner.
        let l = self.loaded.as_ref().unwrap();
        let t = &l.trace;
        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
            ui.add_space(3.0);
            for (idx, &count) in l.reason_counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let r = REASONS[idx];
                ui.label(
                    egui::RichText::new(format!("◦ {}  ×{}", r.label(), count))
                        .color(reason_color(r))
                        .size(11.0),
                )
                .on_hover_text(r.detail());
            }
            if !t.unassigned.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("CLICK TO FLY TO TERMINAL")
                        .color(DIM)
                        .size(9.0),
                );
                ui.add_space(2.0);
                // Virtualized: lay out only the on-screen rows (the list can hold
                // thousands of faults on oversubscribed scenarios, and this runs every
                // frame). auto_shrink y=true keeps the bottom-anchored panel measuring
                // correctly. Rows are uniform single labels, so a fixed height is exact.
                let n = t.unassigned.len().min(4000);
                let line_h = ui
                    .painter()
                    .layout_no_wrap("0".to_owned(), egui::FontId::monospace(11.0), WHITE)
                    .size()
                    .y;
                let row_h = (line_h + 2.0 * ui.spacing().button_padding.y).max(ui.spacing().interact_size.y);
                egui::ScrollArea::vertical()
                    .max_height(150.0)
                    .auto_shrink([false, true])
                    .show_rows(ui, row_h, n, |ui, range| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
                            for i in range {
                                let info = &t.unassigned[i];
                                let sel = self.selected == Some(i);
                                let txt = egui::RichText::new(format!(
                                    "terminal {} · {}",
                                    l.scn.user_ids[info.user as usize],
                                    info.reason.label()
                                ))
                                .color(reason_color(info.reason))
                                .size(11.0);
                                // Build the tooltip only on actual hover, not per row/frame.
                                let (reason, in_view) = (info.reason, info.in_view);
                                let resp = ui.selectable_label(sel, txt).on_hover_ui(|ui| {
                                    ui.label(format!(
                                        "{}\nsatellites in view: {}",
                                        reason.detail(),
                                        in_view
                                    ));
                                });
                                if resp.clicked() {
                                    clicked = Some(i);
                                }
                            }
                        });
                    });
            }
        });
        clicked
    }

    /// Focus-mode study readout (top-right): **frameless** — identity, a beam
    /// gauge, interactive per-band toggles, interferer proximity, and a scoped
    /// "replay" of just this satellite's beams. Legibility comes from text
    /// shadows + corner brackets, not a backing card. Returns `true` on close.
    fn focus_panel(&mut self, ui: &mut egui::Ui) -> bool {
        const W: f32 = 254.0;
        let Some(s) = self.focused_sat else { return false };
        let (sat_id, total, near_n) = {
            let Some(l) = self.loaded.as_ref() else { return false };
            if s >= l.sat_pos.len() {
                return false;
            }
            // Per-band counts now live in the left panel's A/B/C/D chips; here we
            // only need the total (one event per served beam).
            let total = l.sat_events[s].len() as u32;
            let sat_dir = l.sat_pos[s].normalize_or_zero();
            let near_n = l
                .interferer_pos
                .iter()
                .filter(|ip| ip.normalize_or_zero().dot(sat_dir) > FOCUS_INTERFERER_COS)
                .count();
            (l.scn.sat_ids[s].clone(), total, near_n)
        };

        // Transparent padded frame: no fill/stroke (frameless), but the margin
        // gives the corner brackets room to read as a containing instrument.
        let inner = egui::Frame::NONE
            .inner_margin(egui::Margin::symmetric(13, 11))
            .show(ui, |ui| {
                ui.set_width(W);
                let mut close = false;

                ui.horizontal(|ui| {
                    let p = (0.5 + 0.5 * (self.time * 3.0).sin()) * 255.0;
                    let lock = egui::Color32::from_rgba_unmultiplied(232, 238, 244, p as u8);
                    ui.label(egui::RichText::new("● LOCK").color(lock).size(9.5).strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(egui::RichText::new("✕").size(13.0)).frame(false))
                            .on_hover_text("Exit focus (Esc)")
                            .clicked()
                        {
                            close = true;
                        }
                    });
                });
                shadow_text(ui, &format!("SAT {sat_id}"), 17.0, WHITE);
                ui.label(egui::RichText::new("TRACK // FOCUS").color(DIM).size(9.0));
                ui.add_space(9.0);

                // BEAMS gauge fills the width up to the "n / 32" count (no trailing
                // whitespace). The per-band breakdown lives in the left A/B/C/D chips.
                section(ui, "BEAMS");
                let sp = ui.spacing().item_spacing.x;
                ui.horizontal(|ui| {
                    gauge(ui, W - 46.0 - sp, total as f32 / 32.0);
                    ui.label(egui::RichText::new(format!("{total:02} / 32")).color(WHITE).size(10.0));
                });
                ui.add_space(9.0);

                if near_n == 0 {
                    ui.label(egui::RichText::new("NO INTERFERENCE").color(DIM).size(10.0));
                } else {
                    ui.label(
                        egui::RichText::new(format!("⚠ {near_n} IN RANGE // 25° FIELD"))
                            .color(INTERFERER_UI)
                            .size(10.5),
                    );
                }
                ui.add_space(9.0);

                // REPLAY: play button + a scrubber that fills up to the count.
                section(ui, "REPLAY");
                ui.horizontal(|ui| {
                    let lbl = if self.focus_playing { "[ ⏸ ]" } else { "[ ▶ ]" };
                    if ui.add_sized([38.0, 22.0], egui::Button::new(lbl)).clicked() {
                        if self.focus_playing {
                            self.focus_playing = false;
                        } else {
                            if !self.focus_reveal.is_finite() || self.focus_reveal as u32 >= total {
                                self.focus_reveal = 0.0;
                            }
                            self.focus_playing = true;
                        }
                    }
                    let mut shown = if self.focus_reveal.is_finite() {
                        self.focus_reveal.min(total as f64)
                    } else {
                        total as f64
                    };
                    ui.spacing_mut().slider_width = (W - 38.0 - 46.0 - 2.0 * sp).max(60.0);
                    let resp = ui.add(
                        egui::Slider::new(&mut shown, 0.0..=(total.max(1) as f64)).show_value(false),
                    );
                    if resp.dragged() || resp.changed() {
                        self.focus_reveal = shown;
                        self.focus_playing = false;
                    }
                    ui.label(
                        egui::RichText::new(format!("{:02}/{:02}", shown as u32, total))
                            .color(WHITE)
                            .size(10.0),
                    );
                });
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("CLICK SAT // ESC EXIT")
                        .color(DIM)
                        .size(9.0),
                );
                close
            });
        brackets(ui.painter(), inner.response.rect, 13.0, alpha(50));
        inner.inner
    }
}

/// A thin horizontal bar gauge (track + phosphor fill) drawn with the painter.
fn gauge(ui: &mut egui::Ui, width: f32, frac: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 8.0), egui::Sense::hover());
    let y = rect.center().y;
    let p = ui.painter();
    p.rect_filled(
        egui::Rect::from_min_max(egui::pos2(rect.left(), y - 1.5), egui::pos2(rect.right(), y + 1.5)),
        egui::CornerRadius::ZERO,
        alpha(22),
    );
    let fw = (width * frac.clamp(0.0, 1.0)).max(0.0);
    p.rect_filled(
        egui::Rect::from_min_max(egui::pos2(rect.left(), y - 1.5), egui::pos2(rect.left() + fw, y + 1.5)),
        egui::CornerRadius::ZERO,
        alpha(190),
    );
}

/// The full-screen "TARGET ACQUISITION" loading sequence, drawn entirely with
/// the painter (no widgets) over the still-drifting nebula: a snap-rotating
/// bracket reticle, a radar sweep, growing range rings, and cycling telemetry.
fn loading_overlay(painter: &egui::Painter, rect: egui::Rect, t: f32, name: &str, algo: &str) {
    let c = rect.center();
    let mono = |s: f32| egui::FontId::new(s, egui::FontFamily::Monospace);

    // Snap-rotating square reticle (4 corner brackets), 15° mechanical steps.
    let ang = (t * 6.0).floor() * (std::f32::consts::PI / 12.0);
    let (sa, ca) = ang.sin_cos();
    let rot = |x: f32, y: f32| c + egui::vec2(x * ca - y * sa, x * sa + y * ca);
    let r = 120.0;
    let tick = 26.0;
    let st = egui::Stroke::new(1.5_f32, alpha(75));
    for (sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        let (cx, cy) = (sx * r, sy * r);
        painter.line_segment([rot(cx, cy), rot(cx - sx * tick, cy)], st);
        painter.line_segment([rot(cx, cy), rot(cx, cy - sy * tick)], st);
    }

    // Steady inner ring + two outward "ping" rings.
    painter.circle_stroke(c, 52.0, egui::Stroke::new(1.0_f32, alpha(45)));
    for k in 0..2 {
        let f = ((t / 1.5) + k as f32 * 0.5).fract();
        painter.circle_stroke(c, 52.0 + f * 92.0, egui::Stroke::new(1.0_f32, alpha(((1.0 - f) * 80.0) as u8)));
    }

    // Sweeping radar line with a short fading trail.
    for k in 0..7 {
        let a = t * 2.0 - k as f32 * 0.06;
        let (s, co) = a.sin_cos();
        painter.line_segment(
            [c, c + egui::vec2(co, s) * 118.0],
            egui::Stroke::new(1.0_f32, alpha((90.0 * (1.0 - k as f32 / 7.0)) as u8)),
        );
    }

    // Telemetry text below the reticle.
    let dots = ".".repeat(((t * 2.0) as i64 % 4) as usize);
    painter.text(c + egui::vec2(0.0, 92.0), egui::Align2::CENTER_CENTER, format!("SOLVING{dots}"), mono(20.0), WHITE);
    painter.text(
        c + egui::vec2(0.0, 116.0),
        egui::Align2::CENTER_CENTER,
        format!("◄ {name} ► // {algo}"),
        mono(10.0),
        DIM,
    );
    let hex = (t * 9000.0) as u32 & 0xFFFF;
    painter.text(
        c + egui::vec2(0.0, 134.0),
        egui::Align2::CENTER_CENTER,
        format!("BEAMS 0x{hex:04X}  //  STAND BY"),
        mono(9.0),
        alpha(95),
    );
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // eframe 0.34 drives the app through `ui` with a frameless, full-window
        // root `Ui`. Take an owned `Context` handle so the rest of the method
        // reads exactly as the old `update(ctx, …)` did.
        let ctx = ui.ctx().clone();
        let ctx = &ctx;
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        self.time += dt;
        // Drive a continuous frame clock so the nebula, starfield, and overlays
        // animate live — not only when an input event happens to wake the UI.
        // This is the *sole* repaint driver; per-branch repaints below would be
        // redundant (request_repaint is idempotent within a frame).
        ctx.request_repaint();

        // Responsive HUD scale for small / mobile viewports: shrink the whole UI so
        // the controls don't dominate a narrow screen. `content_rect().width()` is in
        // points (physical px ÷ pixels_per_point), and pixels_per_point = native ×
        // zoom, so `width × zoom` is the device-independent CSS width — invariant to
        // the zoom we set here, so it converges in one frame. The 3D scene fills its
        // rect at full physical resolution regardless, so only the HUD scales — the
        // globe stays crisp. `compact` then relays the bottom row so it can't overlap.
        let css_w = ctx.content_rect().width() * ctx.zoom_factor();
        let target_zoom = (css_w / 940.0).clamp(0.62, 1.0);
        if (ctx.zoom_factor() - target_zoom).abs() > 0.01 {
            ctx.set_zoom_factor(target_zoom);
        }
        // Layout regime, in points (= physical px ÷ pixels_per_point):
        //  • `compact`  — condense the transport and pin it bottom-left, so it and
        //    the bottom-right unserved card both fit a phone/tablet width.
        //  • `short`    — a landscape-phone-height viewport, where the tall left
        //    column is capped + scrollable so it can't overflow into the transport.
        let sr = ctx.content_rect();
        let compact = sr.width() < 1240.0;
        let short = sr.height() < 520.0;

        // Escape, in order: dismiss the "Add your own…" dialog (even while its text
        // box has focus, so it always gets you out), then leave satellite focus,
        // then exit fullscreen — never trap the user.
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.custom && self.loaded.is_none() && self.loading.is_none() {
                self.exit_custom();
            } else if !ctx.egui_wants_keyboard_input() {
                if self.focused_sat.is_some() {
                    self.exit_focus();
                } else if self.fullscreen {
                    self.fullscreen = false;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                }
            }
        }
        // H (hide HUD) / F11 (fullscreen) — suppressed while a widget wants keyboard
        // input (e.g. the paste box has focus), so typing a scenario can't fire them.
        // egui's `key_pressed` reports keys regardless of focus, so the guard is required.
        if !ctx.egui_wants_keyboard_input() {
            if ctx.input(|i| i.key_pressed(egui::Key::H)) {
                self.show_ui = !self.show_ui;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
                self.fullscreen = !self.fullscreen;
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
            }
        }

        // Pick up a finished background solve.
        if let Some(rx) = self.loading.as_ref() {
            match rx.try_recv() {
                Ok(Ok(l)) => {
                    // Keep a valid focus across an algorithm rerun; drop a stale one.
                    if self.focused_sat.is_some_and(|s| s >= l.sat_pos.len()) {
                        self.exit_focus();
                    }
                    self.focus_reveal = f64::INFINITY;
                    self.focus_playing = false;
                    self.loaded = Some(l);
                    self.revealed = 0.0;
                    self.playing = true;
                    self.loading = None;
                }
                Ok(Err(e)) => {
                    self.error = Some(e);
                    self.loading = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.error = Some("solver thread terminated unexpectedly".into());
                    self.loading = None;
                }
            }
        }

        // An uploaded (file-picker) or drag-and-dropped scenario → solve it.
        if let Some(rx) = self.upload_rx.as_ref() {
            if let Ok(text) = rx.try_recv() {
                self.upload_rx = None;
                self.ingest_scenario(text);
            }
        }
        let dropped = ctx.input(|i| i.raw.dropped_files.first().cloned());
        if let Some(file) = dropped {
            // Web delivers the bytes inline; native delivers a path to read.
            let text = file.bytes.as_ref().and_then(|b| String::from_utf8(b.to_vec()).ok());
            #[cfg(not(target_arch = "wasm32"))]
            let text = text.or_else(|| file.path.as_ref().and_then(|p| std::fs::read_to_string(p).ok()));
            if let Some(text) = text {
                self.ingest_scenario(text);
            }
        }

        // Camera fly-to animation.
        if let Some(mut anim) = self.anim {
            anim.t = (anim.t + dt / 0.9).min(1.0);
            let ease_t = ease(anim.t);
            let (from_yaw, from_pitch, from_dist) = anim.from;
            let (to_yaw, to_pitch, to_dist) = anim.to;
            self.camera.yaw = lerp_angle(from_yaw, to_yaw, ease_t);
            self.camera.pitch = from_pitch + (to_pitch - from_pitch) * ease_t;
            self.camera.distance = from_dist + (to_dist - from_dist) * ease_t;
            self.anim = if anim.t >= 1.0 { None } else { Some(anim) };
        }

        // Assignment animation. Speed is "whole timeline in ~12 s" × the chosen
        // multiplier, so cases of very different size all play at a sane pace.
        let total = self.total_events();
        if self.playing && total > 0 {
            let bps = (total as f64 / 12.0 * self.speed_mult).max(1.0);
            self.revealed = (self.revealed + bps * dt as f64).min(total as f64);
            if self.revealed as usize >= total {
                self.playing = false;
            }
        }

        // Scoped replay of the focused satellite's own beams ("render out the
        // change" for just that one).
        if self.focus_playing {
            if let Some(s) = self.focused_sat {
                let n = self.focus_beam_count(s);
                if !self.focus_reveal.is_finite() {
                    self.focus_reveal = n as f64;
                }
                let rate = (n as f64 / 2.5).max(1.0); // whole satellite in ~2.5 s
                self.focus_reveal = (self.focus_reveal + rate * dt as f64).min(n as f64);
                if self.focus_reveal as usize >= n {
                    self.focus_playing = false;
                }
            } else {
                self.focus_playing = false;
            }
        }

        // "Add your own…" with nothing solved yet shows the center dialog; a click on
        // the bare scene (egui routes dialog/panel clicks to those areas, so this only
        // fires off them) dismisses it — see the click handler below.
        let awaiting_custom = self.custom && self.loaded.is_none() && self.loading.is_none();
        let mut dismiss_custom = false;

        // Full-window 3D scene.
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| {
                let size = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                // Camera input: two-finger pinch zooms (touch); otherwise a
                // one-finger / mouse drag orbits and the wheel zooms. Orbit is
                // suppressed during a pinch so the globe doesn't lurch as the
                // touch centroid drifts.
                if let Some(mt) = ui.input(|i| i.multi_touch()) {
                    if mt.zoom_delta != 1.0 {
                        self.camera.zoom_by(mt.zoom_delta);
                        self.anim = None;
                    }
                } else {
                    if response.dragged() {
                        let d = response.drag_delta();
                        self.camera.orbit(d.x, d.y);
                        self.anim = None;
                    }
                    if response.hovered() {
                        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                        if scroll != 0.0 {
                            self.camera.zoom(scroll);
                        }
                    }
                }
                let ppp = ctx.pixels_per_point();
                let pw = (rect.width() * ppp).round().max(1.0) as u32;
                let ph = (rect.height() * ppp).round().max(1.0) as u32;
                if self.scene.resize(pw, ph) {
                    self.renderer.write().update_egui_texture_from_wgpu_texture(
                        &self.device,
                        self.scene.color_view(),
                        wgpu::FilterMode::Linear,
                        self.texture_id,
                    );
                }
                let vp = self.camera.view_proj(pw as f32 / ph as f32);
                let focused = self.focused_sat.is_some();
                // In focus mode the globe is hidden, so don't stream tiles. (The
                // unconditional repaint at the top of update() already drives any
                // tiles still arriving to upload on the next frame.)
                if !focused {
                    self.scene.update(vp, self.camera.eye(), ph as f32);
                }
                self.scene
                    .set_camera(vp, self.camera.eye(), sun_dir(), self.time);
                self.scene.set_atmosphere(self.show_atmosphere);
                self.scene.set_focus_mode(focused);
                self.scene
                    .set_load(if self.loading.is_some() { self.time } else { 0.0 });
                // A bare-scene click: dismiss the "Add your own…" dialog if it's up,
                // else focus a clicked satellite / leave focus on empty space.
                if response.clicked() {
                    if awaiting_custom {
                        dismiss_custom = true;
                    } else if self.focused_sat.is_some() {
                        // Focus mode renders a local-space schematic, so an ECEF pick
                        // is meaningless here — any scene click is a deliberate exit.
                        self.exit_focus();
                    } else if let Some(p) = response.interact_pointer_pos() {
                        match self.pick(p, rect, vp) {
                            Some(Picked::Sat(s)) => self.enter_focus(s),
                            _ => self.exit_focus(),
                        }
                    }
                }
                // Resolve the hovered entity once: it drives both the tooltip and
                // (for an interferer) its field-of-interference overlay. Suppressed
                // in focus mode, where the focus panel carries the information. The
                // pick is cached on (pointer, camera); only the tooltip text is
                // re-formatted each frame.
                let hover = if self.show_ui && self.focused_sat.is_none() {
                    if let Some(ptr) = response.hover_pos() {
                        let pk = self.pick_cached(ptr, rect, vp);
                        pk.and_then(|p| self.hover_entity(p)).map(|h| (ptr, h))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let hover_interferer = hover.as_ref().and_then(|(_, h)| h.interferer);

                // Recompose + re-upload only when something the scene depends on
                // changed. Focus mode pulses its reticle, so it always rebuilds;
                // otherwise orbiting/zooming a paused scene re-renders the existing
                // GPU buffers without rebuilding ~100k instances every frame.
                let recompose = if self.focused_sat.is_some() {
                    self.last_compose_key = None;
                    true
                } else if let Some(l) = &self.loaded {
                    let rev = (self.revealed as usize).min(l.trace.events.len());
                    let bits = self.view_opts_bits();
                    let key = (
                        rev,
                        bits,
                        self.selected.map_or(-1i64, |x| x as i64),
                        hover_interferer.map_or(-1i64, |x| x as i64),
                    );
                    let changed = self.last_compose_key != Some(key);
                    self.last_compose_key = Some(key);
                    changed
                } else {
                    false
                };
                if recompose {
                    self.build_world(hover_interferer);
                    self.scene.set_points(&self.bufs.points);
                    self.scene.set_beams(&self.bufs.beams);
                }
                self.scene.render();
                ui.painter().image(
                    self.texture_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // A live progress scanline: only in the globe view while the
                // assignment is playing, its vertical position tracks coverage
                // (revealed / total) — sweeps down as the solve fills in, then
                // disappears. Suppressed in focus mode (its own replay applies).
                if self.show_ui && self.playing && self.focused_sat.is_none() {
                    if let Some(l) = &self.loaded {
                        let total = l.trace.events.len().max(1);
                        let frac = (self.revealed / total as f64).clamp(0.0, 1.0) as f32;
                        let y = rect.top() + frac * rect.height();
                        ui.painter().rect_filled(
                            egui::Rect::from_min_max(
                                egui::pos2(rect.left(), y),
                                egui::pos2(rect.right(), y + 1.5),
                            ),
                            egui::CornerRadius::ZERO,
                            alpha(20),
                        );
                    }
                }

                // Full-screen "TARGET ACQUISITION" sequence while a solve runs.
                if self.loading.is_some() {
                    // In "Add your own…" mode there is no bundled scenario name —
                    // don't mislabel it with the last-selected one.
                    let name = if self.custom {
                        "custom scenario"
                    } else {
                        self.scenarios.get(self.current).map(|s| s.0.as_str()).unwrap_or("")
                    };
                    loading_overlay(ui.painter(), rect, self.time, name, self.algo.name());
                }

                // Hover tooltip, tinted by the hovered terminal's band.
                if let Some((ptr, h)) = hover {
                    egui::Area::new(egui::Id::new("hovertip"))
                        .order(egui::Order::Tooltip)
                        .fixed_pos(ptr + egui::vec2(16.0, 14.0))
                        .show(ctx, |ui| {
                            let r = tooltip_glass(h.band)
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(h.title)
                                            .color(WHITE)
                                            .strong()
                                            .size(12.0),
                                    );
                                    for l in h.lines {
                                        ui.label(egui::RichText::new(l).color(DIM).size(11.0));
                                    }
                                })
                                .response
                                .rect;
                            brackets(ui.painter(), r, 9.0, alpha(55));
                        });
                }
            });

        // Frameless HUD drawn directly on the scene (no panel "bubbles"):
        // all selectors top-left, coverage/solve top-right, transport
        // bottom-center, unserved inspector bottom-right. `H` hides it all.
        let mut clicked = None;
        let mut close_focus = false;
        // `awaiting_custom` (computed above) → a center dialog stands in for the
        // coverage / transport / unserved modules (there's no solution to show).
        if self.show_ui {
            // On a short (landscape-phone) screen the column is capped + scrollable
            // so it can't overflow into the transport; tall screens show it in full.
            let cap_h = (sr.height() - 110.0).max(150.0);
            egui::Area::new(egui::Id::new("left"))
                .anchor(egui::Align2::LEFT_TOP, [18.0, 16.0])
                .show(ctx, |ui| {
                    let r = glass()
                        .show(ui, |ui| {
                            if short {
                                egui::ScrollArea::vertical()
                                    .max_height(cap_h)
                                    .auto_shrink([false, true])
                                    .show(ui, |ui| self.left_panel(ui));
                            } else {
                                self.left_panel(ui);
                            }
                        })
                        .response
                        .rect;
                    brackets(ui.painter(), r, 13.0, alpha(45));
                });
            if self.focused_sat.is_some() {
                // Focus mode: one frameless study readout stands in for the global
                // coverage / transport / unserved modules.
                egui::Area::new(egui::Id::new("focus"))
                    .anchor(egui::Align2::RIGHT_TOP, [-22.0, 18.0])
                    .show(ctx, |ui| close_focus = self.focus_panel(ui));
            } else if awaiting_custom {
                egui::Area::new(egui::Id::new("customdlg"))
                    .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                    .show(ctx, |ui| {
                        let r = glass().show(ui, |ui| self.custom_dialog(ui)).response.rect;
                        brackets(ui.painter(), r, 13.0, alpha(45));
                    });
            } else {
                egui::Area::new(egui::Id::new("cov"))
                    .anchor(egui::Align2::RIGHT_TOP, [-20.0, 18.0])
                    .show(ctx, |ui| self.coverage_module(ui));
                // Centered transport normally; pinned bottom-left on narrow screens
                // so it can't overlap the bottom-right unserved inspector.
                let (t_anchor, t_offset) = if compact {
                    (egui::Align2::LEFT_BOTTOM, [18.0, -16.0])
                } else {
                    (egui::Align2::CENTER_BOTTOM, [0.0, -16.0])
                };
                egui::Area::new(egui::Id::new("transport"))
                    .anchor(t_anchor, t_offset)
                    .show(ctx, |ui| {
                        let r = glass()
                            .show(ui, |ui| self.transport_module(ui, compact))
                            .response
                            .rect;
                        brackets(ui.painter(), r, 13.0, alpha(45));
                    });
                egui::Area::new(egui::Id::new("uns"))
                    .anchor(egui::Align2::RIGHT_BOTTOM, [-18.0, -16.0])
                    .show(ctx, |ui| {
                        let resp = glass().show(ui, |ui| self.unserved_module(ui));
                        clicked = resp.inner;
                        brackets(ui.painter(), resp.response.rect, 13.0, alpha(45));
                    });
            }
        }

        if dismiss_custom {
            self.exit_custom();
        }
        if close_focus {
            self.exit_focus();
        }
        if let Some(i) = clicked {
            if self.selected == Some(i) {
                self.selected = None;
            } else {
                self.selected = Some(i);
                let pos = self
                    .loaded
                    .as_ref()
                    .map(|l| l.user_pos[l.trace.unassigned[i].user as usize]);
                if let Some(p) = pos {
                    self.focus_on(p);
                }
            }
        }
    }
}

fn ease(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}
fn lerp_angle(from: f32, to: f32, s: f32) -> f32 {
    let mut d = to - from;
    while d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    }
    while d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    from + d * s
}
/// An uppercase section heading with a hairline rule running out to the right
/// edge — the "SCENARIO ───────" instrument look.
fn section(ui: &mut egui::Ui, text: &str) {
    let resp = ui.label(egui::RichText::new(text).color(DIM).size(9.5).strong());
    let y = resp.rect.center().y;
    let x0 = resp.rect.right() + 8.0;
    let x1 = ui.max_rect().right();
    if x1 > x0 + 4.0 {
        ui.painter().line_segment(
            [egui::pos2(x0, y), egui::pos2(x1, y)],
            egui::Stroke::new(1.0_f32, alpha(22)),
        );
    }
    ui.add_space(3.0);
}

/// The container primitive: four L-shaped corner brackets around `rect`. Used in
/// place of full borders so panels read as a targeting overlay, not a bubble.
fn brackets(painter: &egui::Painter, rect: egui::Rect, len: f32, color: egui::Color32) {
    let s = egui::Stroke::new(1.0_f32, color);
    let seg = |a: egui::Pos2, b: egui::Pos2| painter.line_segment([a, b], s);
    let v = egui::vec2;
    let (tl, tr, bl, br) = (
        rect.left_top(),
        rect.right_top(),
        rect.left_bottom(),
        rect.right_bottom(),
    );
    seg(tl, tl + v(len, 0.0));
    seg(tl, tl + v(0.0, len));
    seg(tr, tr + v(-len, 0.0));
    seg(tr, tr + v(0.0, len));
    seg(bl, bl + v(len, 0.0));
    seg(bl, bl + v(0.0, -len));
    seg(br, br + v(-len, 0.0));
    seg(br, br + v(0.0, -len));
}

/// A squared band "targeting light": filled in the band color when ON, a
/// monochrome chip with a dormant band-colored bottom rule when OFF.
fn band_chip(ui: &mut egui::Ui, c: u8, on: bool, label: &str, size: [f32; 2]) -> egui::Response {
    let col = band_color32(c);
    let (txt_col, fill) = if on {
        (egui::Color32::BLACK, col)
    } else {
        (col, alpha(16))
    };
    let txt = egui::RichText::new(label).color(txt_col).strong();
    let resp = ui.add_sized(
        size,
        egui::Button::new(txt).fill(fill).corner_radius(egui::CornerRadius::ZERO),
    );
    if !on {
        let r = resp.rect;
        ui.painter()
            .line_segment([r.left_bottom(), r.right_bottom()], egui::Stroke::new(1.0_f32, col));
    }
    resp
}

/// A fixed-width scene-layer toggle that flips `*on` when clicked. Returns the
/// response so callers can attach a hover tooltip.
fn layer_toggle(ui: &mut egui::Ui, w: f32, on: &mut bool, label: &str) -> egui::Response {
    let resp = ui.add_sized([w, 24.0], egui::Button::selectable(*on, label));
    if resp.clicked() {
        *on = !*on;
    }
    resp
}

/// Draw a monospace text run with a 1px drop shadow, so frameless readouts stay
/// legible over bright scene content with no backing panel. Returns the rect.
fn shadow_text(ui: &mut egui::Ui, text: &str, size: f32, color: egui::Color32) -> egui::Rect {
    let font = egui::FontId::new(size, egui::FontFamily::Monospace);
    // Shape the run once, then draw it twice (cheap Arc clones): a shadow copy
    // recolored via the override, then the real text.
    let galley = ui.painter().layout_no_wrap(text.to_owned(), font, color);
    let (rect, _) = ui.allocate_exact_size(galley.size(), egui::Sense::hover());
    let p = ui.painter();
    p.galley_with_override_text_color(rect.min + egui::vec2(1.0, 1.0), galley.clone(), SHADOW);
    p.galley(rect.min, galley, color);
    rect
}

fn reason_color(r: Reason) -> egui::Color32 {
    match r {
        Reason::NoSatInView => egui::Color32::from_rgb(150, 160, 185),
        Reason::AllInterfered => egui::Color32::from_rgb(255, 120, 95),
        Reason::AllFull => egui::Color32::from_rgb(255, 205, 95),
        Reason::ColorBlocked => egui::Color32::from_rgb(185, 135, 255),
    }
}

/// Squared dark glass for a dense readout panel — fill only, no border (corner
/// brackets supply the containment). Composited by egui *over* the scene image
/// with normal alpha, so the additive scene can never wash it out.
fn glass() -> egui::Frame {
    egui::Frame::NONE
        .fill(egui::Color32::from_rgba_unmultiplied(6, 9, 13, 168))
        .corner_radius(egui::CornerRadius::ZERO)
        .inner_margin(egui::Margin::symmetric(14, 11))
}

/// Dark backing for the transient hover tooltip, faintly tinted toward the
/// hovered terminal's band color (a nudged fill + a band-colored hairline).
fn tooltip_glass(band: Option<u8>) -> egui::Frame {
    let (fill, stroke) = match band {
        Some(c) => {
            let rgb = band_rgb(c);
            let mix = |base: f32, ch: f32| (base + ch * 55.0).min(255.0) as u8;
            (
                egui::Color32::from_rgba_unmultiplied(mix(6.0, rgb[0]), mix(9.0, rgb[1]), mix(13.0, rgb[2]), 220),
                egui::Color32::from_rgba_unmultiplied(
                    (rgb[0] * 255.0) as u8,
                    (rgb[1] * 255.0) as u8,
                    (rgb[2] * 255.0) as u8,
                    130,
                ),
            )
        }
        None => (
            egui::Color32::from_rgba_unmultiplied(6, 9, 13, 220),
            alpha(40),
        ),
    };
    egui::Frame::NONE
        .fill(fill)
        .stroke(egui::Stroke::new(1.0_f32, stroke))
        .corner_radius(egui::CornerRadius::ZERO)
        .inner_margin(egui::Margin::same(9))
}

fn style_egui(ctx: &egui::Context) {
    let mut style = (*ctx.global_style()).clone();
    let mut v = egui::Visuals::dark();
    v.panel_fill = egui::Color32::TRANSPARENT;
    v.override_text_color = Some(WHITE);
    // Dropdown menus / popups read as squared dark glass.
    v.window_fill = egui::Color32::from_rgba_unmultiplied(8, 11, 16, 232);
    v.window_stroke = egui::Stroke::new(1.0_f32, alpha(40));
    v.window_corner_radius = egui::CornerRadius::ZERO;
    v.selection.bg_fill = alpha(30);
    v.selection.stroke = egui::Stroke::new(1.0_f32, WHITE);
    // Frameless controls: hairline border on a barely-there fill, all squared,
    // so each widget reads as an etched instrument directly on the scene.
    let strokes = [28u8, 60, 80, 60]; // inactive, hovered, active, open
    let fills = [14u8, 28, 40, 28];
    for (i, w) in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ]
    .into_iter()
    .enumerate()
    {
        w.corner_radius = egui::CornerRadius::ZERO;
        w.bg_stroke = egui::Stroke::new(1.0_f32, alpha(strokes[i]));
        w.weak_bg_fill = alpha(fills[i]);
        w.bg_fill = alpha(fills[i]);
    }
    style.visuals = v;
    // Monospace everywhere (data-instrument feel); symbols/emoji still fall back.
    use egui::{FontFamily, FontId, TextStyle};
    style.text_styles = [
        (TextStyle::Small, FontId::new(9.0, FontFamily::Monospace)),
        (TextStyle::Body, FontId::new(12.0, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(12.0, FontFamily::Monospace)),
        (TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace)),
        (TextStyle::Heading, FontId::new(17.0, FontFamily::Monospace)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 7.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    ctx.set_global_style(style);
}

/// Locate the `test_cases` directory: explicit env var, then the working
/// directory, then walking up from the executable's location.
#[cfg(not(target_arch = "wasm32"))]
fn test_cases_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("BEAM_TEST_CASES") {
        return p.into();
    }
    let cwd = std::path::PathBuf::from("test_cases");
    if cwd.is_dir() {
        return cwd;
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        while let Some(d) = dir {
            let cand = d.join("test_cases");
            if cand.is_dir() {
                return cand;
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
    }
    "test_cases".into()
}

/// Single source of truth for the unserved-reason ordering, shared by the
/// `reason_counts: [usize; 4]` array and the bottom-right fault readout.
const REASONS: [Reason; 4] = [
    Reason::NoSatInView,
    Reason::AllInterfered,
    Reason::AllFull,
    Reason::ColorBlocked,
];
fn reason_idx(r: Reason) -> usize {
    REASONS.iter().position(|&x| x == r).unwrap_or(0)
}

/// Read a scenario file and solve it (native; `path` is a filesystem path).
#[cfg(not(target_arch = "wasm32"))]
fn load_scenario(path: &str, algo: Algorithm) -> Result<Loaded, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    build_loaded(&text, algo)
}

/// Fetch a scenario's text over HTTP (wasm; `url` is relative to the page).
#[cfg(target_arch = "wasm32")]
async fn fetch_scenario(url: &str) -> Result<String, String> {
    let resp = ehttp::fetch_async(ehttp::Request::get(url))
        .await
        .map_err(|e| format!("fetch {url}: {e}"))?;
    if !resp.ok {
        return Err(format!("fetch {url}: HTTP {}", resp.status));
    }
    resp.text()
        .map(str::to_owned)
        .ok_or_else(|| format!("{url}: response was not valid UTF-8"))
}

/// Format a solved scenario as the validator-format solution: the near-optimality
/// certificate header plus the per-satellite beam allocation — identical to what
/// the CLI writes. Reconstructs `per_sat` from the trace's events (each event is a
/// final beam), so it works for any algorithm.
fn solution_text(l: &Loaded) -> Result<String, String> {
    let (scn, tr) = (&l.scn, &l.trace);
    let mut per_sat = vec![Vec::new(); scn.sats.len()];
    for e in &tr.events {
        per_sat[e.sat as usize].push((e.user, e.color));
    }
    let cert = io::Certificate {
        total_users: scn.users.len(),
        feasible_users: tr.feasible_users,
        upper_bound: tr.upper_bound,
        colored_bound: tr.colored_bound,
        achieved: tr.events.len(),
    };
    let mut buf = Vec::new();
    io::write_solution(&mut buf, scn, &per_sat, &cert).map_err(|e| e.to_string())?;
    String::from_utf8(buf).map_err(|e| e.to_string())
}

/// Reject a scenario the visualizer can't meaningfully show — a comment-only or
/// empty paste parses cleanly into zero satellites/users, which would otherwise
/// render as a confusing blank globe. The solver/CLI handle these fine; here we
/// surface a clear error so "Add your own…" gives feedback instead of a void.
fn validate_scenario(scn: &io::Scenario) -> Result<(), String> {
    if scn.sats.is_empty() || scn.users.is_empty() {
        return Err("scenario needs at least one `sat` and one `user` line".into());
    }
    Ok(())
}

/// Parse + solve scenario text into a [`Loaded`] (platform-independent: no I/O).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))] // native-only path on wasm
fn build_loaded(text: &str, algo: Algorithm) -> Result<Loaded, String> {
    let scn = io::Scenario::parse(text).map_err(|e| format!("parse: {e}"))?;
    validate_scenario(&scn)?;
    let feas = feasibility::build(&scn);
    let trace = trace::run(&scn, &feas, algo);
    Ok(loaded_from_parts(scn, feas, trace))
}

// ---- WASM: solve in a Web Worker, off the render thread ---------------------
// `trace_scenario` (the Worker entry) is exported at the bottom of this file; the
// render thread imports `solveText` / `downloadText` from index.js.

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    /// JS glue (index.js): posts `{text, algo}` to the solve Worker and resolves
    /// with the postcard bytes as a `Uint8Array`.
    #[wasm_bindgen(js_name = solveText, catch)]
    async fn solve_text_js(text: &str, algo: u8) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;

    /// JS glue (index.js): save `text` to the user's machine as `filename`.
    #[wasm_bindgen(js_name = downloadText)]
    fn download_text_js(filename: &str, text: &str);

    /// JS glue (index.js): open the file picker and resolve with the chosen file's
    /// text (empty string if the user cancels).
    #[wasm_bindgen(js_name = pickScenario, catch)]
    async fn pick_scenario_js() -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsValue>;
}

/// A clean, human-readable message from a rejected-promise [`JsValue`] (an
/// `Error`), without the raw `{:?}` debug form and its stack trace.
#[cfg(target_arch = "wasm32")]
fn js_error_message(e: &wasm_bindgen::JsValue) -> String {
    use wasm_bindgen::JsCast;
    e.dyn_ref::<js_sys::Error>()
        .map(|err| String::from(err.message()))
        .or_else(|| e.as_string())
        .unwrap_or_else(|| "solve failed".to_string())
}

/// Fetch a scenario URL, solve it in the Worker, and rebuild [`Loaded`].
#[cfg(target_arch = "wasm32")]
async fn solve_via_worker(url: &str, algo: Algorithm) -> Result<Loaded, String> {
    let text = fetch_scenario(url).await?;
    solve_text_via_worker(&text, algo).await
}

/// Solve scenario `text` in the Worker and rebuild [`Loaded`]. Shared by the
/// bundled-scenario path (after fetch) and "Add your own…" (pasted text).
#[cfg(target_arch = "wasm32")]
async fn solve_text_via_worker(text: &str, algo: Algorithm) -> Result<Loaded, String> {
    let algo = Algorithm::ALL.iter().position(|&a| a == algo).unwrap_or(0) as u8;
    let bytes = solve_text_js(text, algo).await.map_err(|e| js_error_message(&e))?;
    let bytes = js_sys::Uint8Array::new(&bytes).to_vec();
    let (scn, feas, trace): (io::Scenario, feasibility::Feasibility, Trace) =
        postcard::from_bytes(&bytes).map_err(|e| format!("decode solve result: {e}"))?;
    Ok(loaded_from_parts(scn, feas, trace))
}

/// Assemble a [`Loaded`] from an already-solved scenario: the GPU-side f32
/// position/direction caches, the per-satellite event index, and the
/// unserved-reason counts. Cheap (just maps the data), so it runs on the render
/// thread even on wasm — where the expensive parse + solve happen in a worker.
fn loaded_from_parts(scn: io::Scenario, feas: feasibility::Feasibility, trace: Trace) -> Loaded {
    let user_pos = to_f32(&scn.users);
    let sat_pos = to_f32(&scn.sats);
    let interferer_pos = to_f32(&scn.interferers);
    // Cache unit directions once (positions never move) for the hover cull.
    let user_dir = unit_dirs(&user_pos);
    let sat_dir = unit_dirs(&sat_pos);
    let interferer_dir = unit_dirs(&interferer_pos);
    let mut reason_counts = [0usize; 4];
    for u in &trace.unassigned {
        reason_counts[reason_idx(u.reason)] += 1;
    }
    // Per-satellite event indices and the single event per user, both derived in
    // one pass over the already-ordered events.
    let mut sat_events = vec![Vec::new(); sat_pos.len()];
    let mut user_event = vec![-1i32; user_pos.len()];
    for (i, e) in trace.events.iter().enumerate() {
        sat_events[e.sat as usize].push(i as u32);
        user_event[e.user as usize] = i as i32;
    }
    Loaded {
        scn,
        feas,
        user_pos,
        sat_pos,
        interferer_pos,
        user_dir,
        sat_dir,
        interferer_dir,
        trace,
        sat_events,
        user_event,
        reason_counts,
    }
}

/// Convert solver-space f64 positions to the GPU's f32 [`Vec3`].
fn to_f32(pts: &[beamer::geom::Vec3]) -> Vec<Vec3> {
    pts.iter()
        .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect()
}
/// Unit directions for a position list (positions are never at the origin).
fn unit_dirs(pos: &[Vec3]) -> Vec<Vec3> {
    pos.iter().map(|p| p.normalize()).collect()
}

/// Set up a headless wgpu device + queue (no surface). Shared by the single-frame
/// `--shot` and the `--frames` sequence renderer.
#[cfg(not(target_arch = "wasm32"))]
fn init_gpu() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("no GPU adapter: {e}"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|e| format!("device: {e}"))?;
    Ok((device, queue))
}

/// All scene layers on — the default look for headless stills and sequences.
/// Interferers stay off here to keep the hero renders clean (their field is a
/// hover-only overlay anyway).
#[cfg(not(target_arch = "wasm32"))]
fn full_view() -> ViewOpts {
    ViewOpts {
        bands: [true; 4],
        show_full: true,
        show_empty: true,
        show_uncovered: true,
        show_beams: true,
        show_interferers: false,
    }
}

/// Copy the scene's offscreen color texture back to the CPU and save it as a PNG.
#[cfg(not(target_arch = "wasm32"))]
fn save_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &Scene,
    w: u32,
    h: u32,
    out: &str,
) -> Result<(), String> {
    let bpp = 4u32;
    let unpadded = w * bpp;
    let padded = unpadded.div_ceil(256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: scene.color_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(enc.finish()));

    buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("device poll: {e:?}"))?;
    let data = buf.slice(..).get_mapped_range();
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for row in 0..h {
        let s = (row * padded) as usize;
        pixels.extend_from_slice(&data[s..s + unpadded as usize]);
    }
    image::save_buffer(out, &pixels, w, h, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("save {out}: {e}"))?;
    Ok(())
}

/// Headless: render one frame of a scenario to a PNG (no window). Used to verify
/// and preview the renderer. `--shot <scenario.txt> <out.png> [fraction 0..1]`.
#[cfg(not(target_arch = "wasm32"))]
fn screenshot(scenario: &str, out: &str, fraction: f64) -> Result<(), String> {
    let (w, h) = (1600u32, 1000u32);
    let (device, queue) = init_gpu()?;
    let l = load_scenario(scenario, Algorithm::Optimized)?;
    let revealed = (l.trace.events.len() as f64) * fraction.clamp(0.0, 1.0);
    let mut buf = ComposeBuffers::default();
    compose(&l, revealed, &full_view(), None, None, &mut buf);

    let mut scene = Scene::new(device.clone(), queue.clone());
    // Enable the atmosphere halo for the still (tiles won't stream in headlessly).
    scene.set_tile_source(tiles::TileSource::Dark);
    scene.set_atmosphere(true);
    scene.resize(w, h);
    let cam = OrbitCamera::default();
    scene.set_camera(cam.view_proj(w as f32 / h as f32), cam.eye(), sun_dir(), 0.0);
    scene.set_points(&buf.points);
    scene.set_beams(&buf.beams);
    scene.render();

    save_frame(&device, &queue, &scene, w, h, out)?;
    eprintln!(
        "wrote {out} ({w}x{h}, {} beams of {})",
        revealed as usize,
        l.trace.events.len()
    );
    Ok(())
}

/// Headless: render a whole playback sequence to numbered PNGs (`frame_00000.png`,
/// …) under `dir`, ready to be encoded into a GIF/MP4 by an external tool such as
/// ffmpeg. The scenario is solved once and the GPU device reused across frames, so
/// this is far cheaper than calling `--shot` per frame.
///
/// `--frames <scenario.txt> <dir> <n_frames> [orbit_degrees]`: the playback
/// fraction sweeps 0→1 across the frames while the camera orbits by `orbit_degrees`
/// total (default 0), so the beam network paints itself onto a slowly turning globe.
#[cfg(not(target_arch = "wasm32"))]
fn frames(scenario: &str, dir: &str, n: usize, orbit_deg: f32) -> Result<(), String> {
    if n == 0 {
        return Err("n_frames must be > 0".into());
    }
    let (w, h) = (1600u32, 1000u32);
    let (device, queue) = init_gpu()?;
    let l = load_scenario(scenario, Algorithm::Optimized)?;
    let opts = full_view();

    let mut scene = Scene::new(device.clone(), queue.clone());
    scene.set_tile_source(tiles::TileSource::Dark);
    scene.set_atmosphere(true);
    scene.resize(w, h);

    std::fs::create_dir_all(dir).map_err(|e| format!("create {dir}: {e}"))?;
    let total = l.trace.events.len() as f64;
    let aspect = w as f32 / h as f32;
    let yaw0 = OrbitCamera::default().yaw;
    let mut buf = ComposeBuffers::default();

    for i in 0..n {
        let f = if n == 1 { 1.0 } else { i as f64 / (n - 1) as f64 };
        let cam = OrbitCamera {
            yaw: yaw0 - orbit_deg.to_radians() * f as f32,
            ..OrbitCamera::default()
        };
        scene.set_camera(cam.view_proj(aspect), cam.eye(), sun_dir(), i as f32 * 0.05);
        compose(&l, total * f, &opts, None, None, &mut buf);
        scene.set_points(&buf.points);
        scene.set_beams(&buf.beams);
        scene.render();
        save_frame(&device, &queue, &scene, w, h, &format!("{dir}/frame_{i:05}.png"))?;
    }
    eprintln!("wrote {n} frames to {dir}/ ({w}x{h}, {} beams)", total as usize);
    Ok(())
}

/// Native entry point: parse CLI args (`--shot` / `--frames` headless modes) and
/// otherwise launch the interactive window. Called by the thin `beamer` binary.
#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> eframe::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s == "--shot").unwrap_or(false) {
        let scenario = args
            .get(2)
            .expect("--shot <scenario.txt> <out.png> [fraction]");
        let out = args
            .get(3)
            .expect("--shot <scenario.txt> <out.png> [fraction]");
        let fraction = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1.0);
        if let Err(e) = screenshot(scenario, out, fraction) {
            eprintln!("screenshot failed: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }

    if args.get(1).map(|s| s == "--frames").unwrap_or(false) {
        let scenario = args
            .get(2)
            .expect("--frames <scenario.txt> <dir> <n_frames> [orbit_degrees]");
        let dir = args
            .get(3)
            .expect("--frames <scenario.txt> <dir> <n_frames> [orbit_degrees]");
        let n: usize = args
            .get(4)
            .and_then(|s| s.parse().ok())
            .expect("--frames <scenario.txt> <dir> <n_frames> [orbit_degrees]");
        let orbit_deg: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        if let Err(e) = frames(scenario, dir, n, orbit_deg) {
            eprintln!("frames failed: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_fullscreen(true)
            .with_inner_size([1600.0, 1000.0])
            .with_title("Beamer"),
        ..Default::default()
    };
    eframe::run_native("Beamer", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}

/// Browser entry point: mount the visualizer onto a `<canvas>`. Called from JS
/// once the wasm module is initialized. Async because eframe brings the GPU
/// surface up asynchronously. The `App::new` closure is identical to native.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), wasm_bindgen::JsValue> {
    // Force the GL (WebGL2) backend for the widest browser support. egui-wgpu
    // 0.34 moved backend selection into `WgpuSetup`; override the instance
    // descriptor's backends on the default "create new" setup.
    let mut wgpu_setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    wgpu_setup.instance_descriptor.backends = wgpu::Backends::GL;
    let web_options = eframe::WebOptions {
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(wgpu_setup),
            ..Default::default()
        },
        ..Default::default()
    };
    eframe::WebRunner::new()
        .start(canvas, web_options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    // The viz solve Worker serializes (Scenario, Feasibility, Trace) with postcard
    // and the render thread deserializes it. Exercise that round-trip natively so
    // the cross-Worker handoff can't silently drift.
    #[test]
    fn solve_roundtrip_postcard() {
        let text = include_str!("../../test_cases/03_five_users.txt");
        let scn = io::Scenario::parse(text).unwrap();
        let feas = feasibility::build(&scn);
        let trace = trace::run(&scn, &feas, Algorithm::Optimized);
        let (n_users, n_sats, n_events, n_ids) =
            (scn.users.len(), scn.sats.len(), trace.events.len(), scn.user_ids.len());

        let bytes = postcard::to_allocvec(&(&scn, &feas, &trace)).unwrap();
        let (scn2, feas2, trace2): (io::Scenario, feasibility::Feasibility, Trace) =
            postcard::from_bytes(&bytes).unwrap();

        assert_eq!(scn2.users.len(), n_users);
        assert_eq!(scn2.user_ids.len(), n_ids);
        assert_eq!(feas2.feasible_users, feas.feasible_users);
        assert_eq!(trace2.events.len(), n_events);

        // The deserialized parts must rebuild a Loaded with consistent caches.
        let l = loaded_from_parts(scn2, feas2, trace2);
        assert_eq!(l.sat_pos.len(), n_sats);
        assert_eq!(l.sat_events.len(), n_sats);

        // The per-user event cache the hover tooltip resolves against: one entry
        // per user, and every event reachable by its user index (i.e. each user
        // has at most one beam — the invariant the O(1) lookup depends on).
        assert_eq!(l.user_event.len(), n_users);
        for (i, e) in l.trace.events.iter().enumerate() {
            assert_eq!(l.user_event[e.user as usize], i as i32);
        }
    }

    // A comment-only / empty paste parses cleanly but has no geometry — the viz
    // must reject it (instead of rendering a blank globe), while a real scenario
    // passes. Guards the "Add your own…" invalid-input path.
    #[test]
    fn rejects_degenerate_scenario() {
        let empty = io::Scenario::parse("# bla\n\n  # just comments\n").unwrap();
        assert!(validate_scenario(&empty).is_err());

        let real = io::Scenario::parse(include_str!("../../test_cases/03_five_users.txt")).unwrap();
        assert!(validate_scenario(&real).is_ok());
    }

    // The "Download" button formats `solution_text` from a solved Loaded. Check it
    // produces the validator format — certificate header + one beam line per event.
    #[test]
    fn solution_text_is_well_formed() {
        let text = include_str!("../../test_cases/03_five_users.txt");
        let l = build_loaded(text, Algorithm::Optimized).unwrap();
        let out = solution_text(&l).unwrap();

        assert!(out.starts_with("# beamer:"), "missing certificate header");
        assert!(
            out.contains(&format!("# achieved = {} ", l.trace.events.len())),
            "achieved count must match the trace"
        );
        let beam_lines: Vec<&str> = out.lines().filter(|ln| ln.starts_with("sat ")).collect();
        assert_eq!(beam_lines.len(), l.trace.events.len(), "one beam line per event");

        // Each beam line must keep the validator's exact token shape.
        for ln in beam_lines {
            let t: Vec<&str> = ln.split_whitespace().collect();
            assert!(
                matches!(
                    t.as_slice(),
                    ["sat", _, "beam", _, "user", _, "color", c] if matches!(*c, "A" | "B" | "C" | "D")
                ),
                "malformed beam line: {ln}"
            );
        }
    }
}

// ---- WASM: the visualizer app's wasm-bindgen bindings -----------------------

/// Solve a scenario for the visualizer and return `(Scenario, Feasibility, Trace)`
/// postcard-serialized; the render thread deserializes and rebuilds its state.
/// `algo` indexes [`beamer::trace::Algorithm::ALL`]. In the multi-thread
/// build this runs the rayon Worker pool — call `initThreadPool` first.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn trace_scenario(text: &str, algo: u8) -> Result<Vec<u8>, wasm_bindgen::JsError> {
    let algo = Algorithm::ALL.get(algo as usize).copied().unwrap_or(Algorithm::Optimized);
    let scn = io::Scenario::parse(text).map_err(|e| wasm_bindgen::JsError::new(&e))?;
    validate_scenario(&scn).map_err(|e| wasm_bindgen::JsError::new(&e))?;
    let feas = feasibility::build(&scn);
    let tr = trace::run(&scn, &feas, algo);
    postcard::to_allocvec(&(scn, feas, tr)).map_err(|e| wasm_bindgen::JsError::new(&e.to_string()))
}

/// `wasm-bindgen-rayon`'s `initThreadPool`; call once after `init()` with the
/// worker count. Only present in the multi-thread build (the `parallel` feature).
#[cfg(all(target_arch = "wasm32", feature = "parallel"))]
pub use wasm_bindgen_rayon::init_thread_pool;
