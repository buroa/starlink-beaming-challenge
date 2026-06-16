//! Beamer — the interactive GPU visualizer for the Starlink beam planner.
//!
//! Watch the solver assign satellite beams to user terminals in real time over a
//! 3D globe: switch scenarios and algorithms, toggle color bands and full/empty
//! satellites, control the playback speed, and inspect exactly why any terminal
//! could not be served.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod camera;
mod scene;
mod tiles;

use beam_planner::trace::{self, Algorithm, Reason, Trace};
use beam_planner::{feasibility, io};
use camera::OrbitCamera;
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use glam::Vec3;
use scene::{BeamInstance, PointInstance, Scene};
use std::sync::Arc;

const BANDS: [&str; 4] = ["A", "B", "C", "D"];
const WHITE: egui::Color32 = egui::Color32::from_rgb(238, 242, 248);
const DIM: egui::Color32 = egui::Color32::from_rgb(150, 160, 176);
/// Clear, distinct RGB band colors.
fn band_rgb(c: u8) -> [f32; 3] {
    match c {
        0 => [1.00, 0.23, 0.23], // A — red
        1 => [0.30, 1.00, 0.36], // B — green
        2 => [0.30, 0.55, 1.00], // C — blue
        3 => [1.00, 0.85, 0.20], // D — yellow
        _ => [1.0, 1.0, 1.0],
    }
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
    trace: Trace,
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
/// the live UI and the headless screenshot.
///
/// Allocates point and beam vectors with capacity hints to minimize reallocations
/// in the hot frame-loop (up to 100k users × 1440 satellites in worst case).
fn compose(
    l: &Loaded,
    revealed_f: f64,
    o: &ViewOpts,
    selected: Option<usize>,
    hover_interferer: Option<usize>,
) -> (Vec<PointInstance>, Vec<BeamInstance>) {
    let revealed = (revealed_f as usize).min(l.trace.events.len());
    let events = &l.trace.events[..revealed];

    let mut user_color: Vec<i8> = vec![-1; l.user_pos.len()];
    let mut sat_load: Vec<u32> = vec![0; l.sat_pos.len()];
    // Aggregate which users are covered and satellite loads from events.
    for e in events {
        user_color[e.user as usize] = e.color as i8;
        sat_load[e.sat as usize] += 1;
    }

    let mut points = Vec::with_capacity(l.user_pos.len() + l.sat_pos.len());

    // Terminals.
    let uncovered_color = [0.45, 0.47, 0.55, 0.6];
    let band_colors = [band_rgb(0), band_rgb(1), band_rgb(2), band_rgb(3)];
    for (i, p) in l.user_pos.iter().enumerate() {
        let pos = (*p * 1.002).to_array();
        let c = user_color[i];
        if c >= 0 {
            if !o.bands[c as usize] {
                continue;
            }
            let rgb = band_colors[c as usize];
            points.push(PointInstance {
                pos,
                size: 5.5,
                color: [rgb[0], rgb[1], rgb[2], 1.0],
            });
        } else if o.show_uncovered {
            points.push(PointInstance {
                pos,
                size: 4.0,
                color: uncovered_color,
            });
        }
    }

    // Satellites.
    for (i, p) in l.sat_pos.iter().enumerate() {
        let load = sat_load[i];
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
        points.push(PointInstance {
            pos: p.to_array(),
            size,
            color,
        });
    }

    // Beams.
    let beam_capacity = if o.show_beams { events.len() } else { 0 };
    let mut beams = Vec::with_capacity(beam_capacity);
    if o.show_beams {
        for e in events {
            if !o.bands[e.color as usize] {
                continue;
            }
            let rgb = band_rgb(e.color);
            beams.push(BeamInstance {
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
            points.push(PointInstance {
                pos: upos,
                size: 26.0,
                color: [1.0, 0.30, 0.38, 0.85],
            });
            points.push(PointInstance {
                pos: upos,
                size: 13.0,
                color: [1.0, 1.0, 1.0, 1.0],
            });
            for &s in &l.feas.sats[info.user as usize] {
                beams.push(BeamInstance {
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
            points.push(PointInstance {
                pos: ip.to_array(),
                size: 9.0,
                color: INTERFERER_RGBA,
            });
        }
        if let Some(hi) = hover_interferer {
            if let Some(&ip) = l.interferer_pos.get(hi) {
                push_interference_field(&mut points, &mut beams, ip);
            }
        }
    }

    (points, beams)
}

/// Magenta, distinct from the four band colors and the satellite-status palette.
const INTERFERER_RGBA: [f32; 4] = [0.85, 0.35, 1.0, 0.95];
/// The same magenta as readable UI text (for interferer labels/warnings).
const INTERFERER_UI: egui::Color32 = egui::Color32::from_rgb(217, 140, 255);

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
    let purple = |a: f32| [INTERFERER_RGBA[0], INTERFERER_RGBA[1], INTERFERER_RGBA[2], a];

    let sub = axis * R; // sub-interferer point on the surface
    // Axis line out to the interferer, and a marker at the footprint centre.
    beams.push(BeamInstance {
        a: sub.to_array(),
        b: ip.to_array(),
        color: purple(0.5),
    });
    points.push(PointInstance {
        pos: sub.to_array(),
        size: 11.0,
        color: purple(0.95),
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
                color: purple(0.18),
            });
        }
        // The bold exclusion ring.
        if let Some(p) = prev {
            beams.push(BeamInstance {
                a: p,
                b: rim,
                color: purple(0.7),
            });
        }
        prev = Some(rim);
    }
}

/// Satellite focus mode: an isolated study of one satellite. Draws only that
/// satellite (a bright core inside a pulsing camera-facing reticle), the users
/// it serves up to `reveal` beams (colored by band, with their beams), and any
/// interferer sharing its patch of sky (marker + 20° field). Everything else in
/// the constellation is omitted — the globe + atmosphere remain as backdrop.
fn compose_focus(
    l: &Loaded,
    s: usize,
    reveal: f64,
    t: f32,
    eye: Vec3,
) -> (Vec<PointInstance>, Vec<BeamInstance>) {
    let mut points = Vec::with_capacity(64);
    let mut beams = Vec::with_capacity(64);
    let satpos = l.sat_pos[s];

    // This satellite's beams in assignment order, up to the scoped reveal.
    let mut shown = 0usize;
    for e in &l.trace.events {
        if e.sat as usize != s {
            continue;
        }
        if (shown as f64) >= reveal {
            break;
        }
        shown += 1;
        let rgb = band_rgb(e.color);
        let upos = (l.user_pos[e.user as usize] * 1.002).to_array();
        beams.push(BeamInstance {
            a: satpos.to_array(),
            b: upos,
            color: [rgb[0], rgb[1], rgb[2], 0.85],
        });
        points.push(PointInstance {
            pos: upos,
            size: 7.5,
            color: [rgb[0], rgb[1], rgb[2], 1.0],
        });
    }

    // The focused satellite: a bright core wrapped in a pulsing reticle.
    points.push(PointInstance {
        pos: satpos.to_array(),
        size: 20.0,
        color: [0.80, 0.95, 1.0, 1.0],
    });
    push_reticle(&mut points, &mut beams, satpos, eye, t);

    // The nearest interferer sharing this satellite's patch of sky: marker + 20°
    // field. (Drawing only the closest keeps the footprint legible — a belt of
    // GEO interferers would otherwise stack into overlapping rings.)
    let sat_dir = satpos.normalize_or_zero();
    let nearest = l
        .interferer_pos
        .iter()
        .map(|ip| (ip.normalize_or_zero().dot(sat_dir), *ip))
        .filter(|(d, _)| *d > FOCUS_INTERFERER_COS)
        .max_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, ip)) = nearest {
        points.push(PointInstance {
            pos: ip.to_array(),
            size: 9.0,
            color: INTERFERER_RGBA,
        });
        push_interference_field(&mut points, &mut beams, ip);
    }

    (points, beams)
}

/// A camera-facing "lock-on" reticle around a focused satellite: a pulsing outer
/// ring, a steady inner ring, and four crosshair ticks. Pure eye-candy (cyan),
/// animated by `t`.
fn push_reticle(points: &mut Vec<PointInstance>, beams: &mut Vec<BeamInstance>, center: Vec3, eye: Vec3, t: f32) {
    let normal = (eye - center).normalize_or_zero();
    if normal == Vec3::ZERO {
        return;
    }
    let seed = if normal.z.abs() < 0.95 { Vec3::Z } else { Vec3::X };
    let u = normal.cross(seed).normalize();
    let v = normal.cross(u);
    let cyan = |a: f32| [0.45, 0.9, 1.0, a];
    let pulse = 0.5 + 0.5 * (t * 2.2).sin(); // 0..1
    let ring = |beams: &mut Vec<BeamInstance>, r: f32, a: f32| {
        const N: usize = 48;
        let mut prev: Option<[f32; 3]> = None;
        for k in 0..=N {
            let th = k as f32 / N as f32 * std::f32::consts::TAU;
            let p = (center + r * (th.cos() * u + th.sin() * v)).to_array();
            if let Some(pp) = prev {
                beams.push(BeamInstance { a: pp, b: p, color: cyan(a) });
            }
            prev = Some(p);
        }
    };
    // Pulsing outer ring + steady inner ring.
    ring(beams, 230.0 + 90.0 * pulse, 0.45 + 0.45 * pulse);
    ring(beams, 150.0, 0.8);
    // Four crosshair ticks just outside the inner ring, each capped with a node.
    for k in 0..4 {
        let th = k as f32 * std::f32::consts::FRAC_PI_2;
        let dir = th.cos() * u + th.sin() * v;
        beams.push(BeamInstance {
            a: (center + 165.0 * dir).to_array(),
            b: (center + 205.0 * dir).to_array(),
            color: cyan(0.7),
        });
        points.push(PointInstance {
            pos: (center + 205.0 * dir).to_array(),
            size: 4.5,
            color: cyan(0.9),
        });
    }
}

struct App {
    scene: Scene,
    device: Arc<wgpu::Device>,
    renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    texture_id: egui::TextureId,

    scenarios: Vec<(String, String)>, // (label, path)
    current: usize,
    algo: Algorithm,
    loaded: Option<Loaded>,
    /// In-flight background solve (the production solver is too heavy to run on
    /// the UI thread — it would freeze the window for seconds).
    loading: Option<std::sync::mpsc::Receiver<Result<Loaded, String>>>,
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

    time: f32,
    anim: Option<CamAnim>,
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

        let mut scenarios: Vec<(String, String)> = std::fs::read_dir(test_cases_dir())
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
        scenarios.sort();

        let mut app = App {
            scene,
            device: rs.device.clone(),
            renderer,
            texture_id,
            scenarios,
            current: 0,
            algo: Algorithm::Optimized,
            loaded: None,
            loading: None,
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
            focus_playing: false,
            time: 0.0,
            anim: None,
        };
        style_egui(&cc.egui_ctx);
        // Open on the headline 100k-user scenario and start playing immediately.
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
        let Some((_, path)) = self.scenarios.get(self.current).cloned() else {
            self.error = Some("No scenarios found in ./test_cases".into());
            self.loading = None;
            return;
        };
        let algo = self.algo;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(load_scenario(&path, algo));
        });
        self.loading = Some(rx);
    }

    /// Re-solve (e.g. after an algorithm change). Re-parsing is cheap; the solve
    /// is the cost, so it goes through the same background path as `load`.
    fn rerun(&mut self) {
        self.load();
    }

    fn total_events(&self) -> usize {
        self.loaded
            .as_ref()
            .map(|l| l.trace.events.len())
            .unwrap_or(0)
    }

    /// Hit-test satellites/terminals under the cursor (front hemisphere only),
    /// returning a tooltip title + detail lines.
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
            if outside && sp.normalize().dot(eye_n) < horizon - 0.2 {
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
            if outside && up.normalize().dot(eye_n) < horizon {
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
                if outside && ip.normalize().dot(eye_n) < horizon - 0.2 {
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

    /// Tooltip (title, detail lines, band tint, interferer index) for the entity
    /// under the cursor.
    fn hover_entity(&self, ptr: egui::Pos2, rect: egui::Rect, vp: glam::Mat4) -> Option<Hover> {
        let l = self.loaded.as_ref()?;
        let revealed = (self.revealed as usize).min(l.trace.events.len());
        match self.pick(ptr, rect, vp)? {
            Picked::Sat(i) => {
                let load = l.trace.events[..revealed]
                    .iter()
                    .filter(|e| e.sat as usize == i)
                    .count();
                Some(Hover {
                    title: format!("Satellite {}", l.scn.sat_ids[i]),
                    lines: vec![format!("{load} / 32 beams in use"), "click to focus →".into()],
                    band: None,
                    interferer: None,
                })
            }
            Picked::User(i) => {
                if let Some(e) = l.trace.events[..revealed]
                    .iter()
                    .find(|e| e.user as usize == i)
                {
                    Some(Hover {
                        title: format!("Terminal {}", l.scn.user_ids[i]),
                        lines: vec![format!(
                            "served · band {} · sat {}",
                            BANDS[e.color as usize], l.scn.sat_ids[e.sat as usize]
                        )],
                        band: Some(e.color),
                        interferer: None,
                    })
                } else {
                    let line = l
                        .trace
                        .unassigned
                        .iter()
                        .find(|u| u.user as usize == i)
                        .map(|u| u.reason.label().to_string())
                        .unwrap_or_else(|| "not yet assigned".into());
                    Some(Hover {
                        title: format!("Terminal {}", l.scn.user_ids[i]),
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

    /// Enter satellite focus mode: isolate satellite `s` and fly to it.
    fn enter_focus(&mut self, s: usize) {
        self.focused_sat = Some(s);
        self.focus_reveal = f64::INFINITY; // show the full assignment first
        self.focus_playing = false;
        self.selected = None;
        if let Some(l) = self.loaded.as_ref() {
            let p = l.sat_pos[s];
            self.focus_on(p);
        }
    }

    /// Beams assigned to the focused satellite, newest last (assignment order).
    fn focus_beam_count(&self, s: usize) -> usize {
        self.loaded
            .as_ref()
            .map(|l| l.trace.events.iter().filter(|e| e.sat as usize == s).count())
            .unwrap_or(0)
    }

    /// Build the points + beams to display for the current frame. In satellite
    /// focus mode the whole-constellation view is replaced by an isolated study
    /// of one satellite.
    fn build_world(
        &self,
        hover_interferer: Option<usize>,
    ) -> (Vec<PointInstance>, Vec<BeamInstance>) {
        let Some(l) = &self.loaded else {
            return (Vec::new(), Vec::new());
        };
        if let Some(s) = self.focused_sat {
            compose_focus(l, s, self.focus_reveal, self.time, self.camera.eye())
        } else {
            compose(l, self.revealed, &self.view_opts(), self.selected, hover_interferer)
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

    // Top-left: every selector + toggle in one frameless, always-visible column.
    fn left_panel(&mut self, ui: &mut egui::Ui) {
        const W: f32 = 214.0;
        ui.set_width(W);
        ui.label(
            egui::RichText::new("BEAMER")
                .color(WHITE)
                .strong()
                .size(15.0),
        );
        ui.label(
            egui::RichText::new("starlink beam planner")
                .color(DIM)
                .size(10.0),
        );
        ui.add_space(13.0);

        // Scenario.
        section(ui, "SCENARIO");
        let mut changed_scn = false;
        let current_label = self
            .scenarios
            .get(self.current)
            .map(|s| s.0.as_str())
            .unwrap_or("");
        egui::ComboBox::from_id_salt("scn")
            .selected_text(current_label)
            .width(W)
            .show_ui(ui, |ui| {
                for (i, (label, _)) in self.scenarios.iter().enumerate() {
                    if ui.selectable_label(self.current == i, label).clicked() {
                        self.current = i;
                        changed_scn = true;
                    }
                }
            });

        // Algorithm.
        ui.add_space(11.0);
        section(ui, "ALGORITHM");
        let mut changed_algo = false;
        egui::ComboBox::from_id_salt("algo")
            .selected_text(self.algo.name())
            .width(W)
            .show_ui(ui, |ui| {
                for a in Algorithm::ALL {
                    if ui.selectable_label(self.algo == a, a.name()).clicked() {
                        self.algo = a;
                        changed_algo = true;
                    }
                }
            });

        // Color bands.
        ui.add_space(11.0);
        section(ui, "BANDS");
        ui.horizontal(|ui| {
            for (c, _) in BANDS.iter().enumerate() {
                let col = band_color32(c as u8);
                let on = self.bands[c];
                let txt = egui::RichText::new(BANDS[c])
                    .color(if on { egui::Color32::BLACK } else { col })
                    .strong();
                let fill = if on {
                    col
                } else {
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 16)
                };
                if ui
                    .add_sized([47.0, 26.0], egui::Button::new(txt).fill(fill))
                    .clicked()
                {
                    self.bands[c] = !self.bands[c];
                }
            }
        });

        // Layers.
        ui.add_space(11.0);
        section(ui, "LAYERS");
        ui.horizontal_wrapped(|ui| {
            ui.toggle_value(&mut self.show_beams, "Beams");
            ui.toggle_value(&mut self.show_full, "Full");
            ui.toggle_value(&mut self.show_empty, "Partial");
            ui.toggle_value(&mut self.show_uncovered, "Uncovered");
            ui.toggle_value(&mut self.show_interferers, "Interferers")
                .on_hover_text("Non-Starlink satellites — hover one to see its 20° field of interference");
        });

        // Basemap.
        ui.add_space(11.0);
        section(ui, "BASEMAP");
        let mut pick = None;
        egui::ComboBox::from_id_salt("basemap")
            .selected_text(self.tile_source.label())
            .width(W)
            .show_ui(ui, |ui| {
                for src in tiles::TileSource::ALL {
                    if ui
                        .selectable_label(self.tile_source == src, src.label())
                        .clicked()
                    {
                        pick = Some(src);
                    }
                }
            });
        if let Some(src) = pick {
            self.tile_source = src;
            self.scene.set_tile_source(src);
        }
        ui.add_space(7.0);
        ui.toggle_value(&mut self.show_atmosphere, "Atmosphere halo")
            .on_hover_text("Fresnel atmosphere glow — independent of the basemap");

        if changed_scn {
            // A different scenario means different satellites — drop the focus.
            self.focused_sat = None;
            self.load();
        } else if changed_algo {
            // Same satellites: keep focus so the change re-renders for this one.
            self.rerun();
        }
    }

    // Top-right: the coverage headline + live solve indicator.
    fn coverage_module(&self, ui: &mut egui::Ui) {
        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
            if self.loading.is_some() {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new().size(15.0));
                    ui.add_space(6.0);
                    let name = self
                        .scenarios
                        .get(self.current)
                        .map(|s| s.0.as_str())
                        .unwrap_or("");
                    ui.label(
                        egui::RichText::new(format!("Solving  {name}  …"))
                            .color(WHITE)
                            .size(13.0),
                    );
                });
                return;
            }
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
            } else if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::LIGHT_RED, e);
            }
        });
    }

    // Bottom-center: transport — rerun, restart, play/pause, scrubber, speed.
    fn transport_module(&mut self, ui: &mut egui::Ui) {
        let total = self.total_events();
        let done = (self.revealed as usize).min(total);
        ui.horizontal(|ui| {
            if ui
                .add_sized([78.0, 28.0], egui::Button::new("⟲  Rerun"))
                .on_hover_text("Re-solve the scenario with the selected algorithm")
                .clicked()
            {
                self.rerun();
            }
            ui.separator();
            if ui
                .add_sized([30.0, 28.0], egui::Button::new("⏮"))
                .on_hover_text("Restart")
                .clicked()
            {
                self.revealed = 0.0;
                self.playing = true;
            }
            let lbl = if self.playing { "⏸" } else { "▶" };
            if ui
                .add_sized(
                    [34.0, 28.0],
                    egui::Button::new(egui::RichText::new(lbl).size(15.0)),
                )
                .clicked()
            {
                if done >= total {
                    self.revealed = 0.0;
                }
                self.playing = !self.playing;
            }
            let mut rev = self.revealed.min(total as f64);
            let resp = ui.add_sized(
                [320.0, 18.0],
                egui::Slider::new(&mut rev, 0.0..=(total.max(1) as f64)).show_value(false),
            );
            if resp.dragged() || resp.changed() {
                self.revealed = rev;
                self.playing = false;
            }
            ui.label(
                egui::RichText::new(format!("{done} / {total}"))
                    .color(WHITE)
                    .size(11.0),
            );
            ui.separator();
            ui.label(egui::RichText::new("speed").color(DIM).size(10.0));
            for &m in &[0.5f64, 1.0, 2.0, 4.0, 8.0] {
                let on = (self.speed_mult - m).abs() < 1e-6;
                let label = if m < 1.0 {
                    format!("{m}×")
                } else {
                    format!("{}×", m as i64)
                };
                if ui.selectable_label(on, label).clicked() {
                    self.speed_mult = m;
                }
            }
        });
    }

    // Unserved card: why terminals failed + jump-to list. Returns a clicked index.
    fn unserved_module(&mut self, ui: &mut egui::Ui) -> Option<usize> {
        const W: f32 = 250.0;
        ui.set_width(W);
        let mut clicked = None;
        let Some(l) = &self.loaded else { return None };
        let t = &l.trace;
        // Everything right-aligned so the block hugs the bottom-right corner.
        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
            ui.label(
                egui::RichText::new("UNSERVED TERMINALS")
                    .color(DIM)
                    .size(10.0),
            );
            ui.add_space(3.0);
            for (idx, &count) in l.reason_counts.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let r = reason_from_idx(idx);
                ui.label(
                    egui::RichText::new(format!("●  {}  ×{}", r.label(), count))
                        .color(reason_color(r))
                        .size(11.0),
                )
                .on_hover_text(r.detail());
            }
            if !t.unassigned.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("click a terminal to fly there")
                        .color(DIM)
                        .size(9.0),
                );
                ui.add_space(2.0);
                // auto_shrink y=true keeps the list bounded (≤ max_height) so the
                // bottom-anchored panel measures correctly and never overflows.
                egui::ScrollArea::vertical()
                    .max_height(150.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.with_layout(egui::Layout::top_down(egui::Align::RIGHT), |ui| {
                            for (i, info) in t.unassigned.iter().enumerate().take(4000) {
                                let sel = self.selected == Some(i);
                                let txt = egui::RichText::new(format!(
                                    "terminal {} · {}",
                                    l.scn.user_ids[info.user as usize],
                                    info.reason.label()
                                ))
                                .color(reason_color(info.reason))
                                .size(11.0);
                                let resp = ui.selectable_label(sel, txt).on_hover_text(format!(
                                    "{}\nsatellites in view: {}",
                                    info.reason.detail(),
                                    info.in_view
                                ));
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

    /// Focus-mode study card (top-right): identity, a beam gauge, the per-band
    /// breakdown, interferer proximity, and a scoped "replay" of just this
    /// satellite's beams. Returns `true` when the user closes it.
    fn focus_panel(&mut self, ui: &mut egui::Ui) -> bool {
        const W: f32 = 278.0;
        let Some(s) = self.focused_sat else { return false };
        // Pull the immutable data first, then release the borrow before touching
        // the scoped-replay state below.
        let (sat_id, band_counts, total, near) = {
            let Some(l) = self.loaded.as_ref() else { return false };
            if s >= l.sat_pos.len() {
                return false;
            }
            let mut band_counts = [0u32; 4];
            for e in &l.trace.events {
                if e.sat as usize == s {
                    band_counts[e.color as usize] += 1;
                }
            }
            let total: u32 = band_counts.iter().sum();
            let sat_dir = l.sat_pos[s].normalize_or_zero();
            let near: Vec<String> = l
                .interferer_pos
                .iter()
                .enumerate()
                .filter(|(_, ip)| ip.normalize_or_zero().dot(sat_dir) > FOCUS_INTERFERER_COS)
                .map(|(i, _)| l.scn.interferer_ids[i].clone())
                .collect();
            (l.scn.sat_ids[s].clone(), band_counts, total, near)
        };

        let accent = egui::Color32::from_rgb(120, 220, 255);
        let mut close = false;
        ui.set_width(W);
        egui::Frame::none()
            .fill(egui::Color32::from_rgba_unmultiplied(8, 12, 18, 226))
            .stroke(egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_unmultiplied(120, 220, 255, 96),
            ))
            .rounding(egui::Rounding::same(10.0))
            .inner_margin(egui::Margin::same(13.0))
            .show(ui, |ui| {
                ui.set_width(W);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("◈ FOCUS").color(accent).size(10.0).strong());
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
                ui.label(
                    egui::RichText::new(format!("Satellite {sat_id}"))
                        .color(WHITE)
                        .size(17.0)
                        .strong(),
                );
                ui.add_space(9.0);

                section(ui, "BEAMS");
                ui.add(
                    egui::ProgressBar::new(total as f32 / 32.0)
                        .desired_height(10.0)
                        .fill(accent)
                        .text(
                            egui::RichText::new(format!("{total} / 32"))
                                .size(10.0)
                                .color(egui::Color32::BLACK),
                        ),
                );
                ui.add_space(9.0);

                section(ui, "BANDS");
                ui.horizontal(|ui| {
                    for c in 0..4u8 {
                        let col = band_color32(c);
                        let n = band_counts[c as usize];
                        let on = n > 0;
                        let txt = egui::RichText::new(format!("{}·{}", BANDS[c as usize], n))
                            .color(if on { egui::Color32::BLACK } else { col })
                            .strong()
                            .size(11.0);
                        let fill = if on {
                            col
                        } else {
                            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 12)
                        };
                        let _ = ui.add_sized([54.0, 24.0], egui::Button::new(txt).fill(fill));
                    }
                });
                ui.add_space(9.0);

                if near.is_empty() {
                    ui.label(
                        egui::RichText::new("no interferers in range")
                            .color(DIM)
                            .size(10.0),
                    );
                } else {
                    let n = near.len();
                    ui.label(
                        egui::RichText::new(format!(
                            "⚠ {n} interferer{} in range",
                            if n == 1 { "" } else { "s" }
                        ))
                        .color(INTERFERER_UI)
                        .size(11.0),
                    );
                    ui.label(
                        egui::RichText::new("nearest 20° field shown on the globe")
                            .color(DIM)
                            .size(9.0),
                    );
                }
                ui.add_space(9.0);

                section(ui, "REPLAY");
                ui.horizontal(|ui| {
                    let lbl = if self.focus_playing { "⏸" } else { "▶" };
                    if ui.add_sized([30.0, 22.0], egui::Button::new(lbl)).clicked() {
                        if self.focus_playing {
                            self.focus_playing = false;
                        } else {
                            // Restart from the top if at (or past) the end.
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
                    let resp = ui.add(
                        egui::Slider::new(&mut shown, 0.0..=(total.max(1) as f64)).show_value(false),
                    );
                    if resp.dragged() || resp.changed() {
                        self.focus_reveal = shown;
                        self.focus_playing = false;
                    }
                    ui.label(
                        egui::RichText::new(format!("{} / {}", shown as u32, total))
                            .color(WHITE)
                            .size(10.0),
                    );
                });
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new("click another satellite · Esc to exit")
                        .color(DIM)
                        .size(9.0),
                );
            });
        close
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        self.time += dt;

        // Toggle the whole HUD for a clean cinematic view.
        if ctx.input(|i| i.key_pressed(egui::Key::H)) {
            self.show_ui = !self.show_ui;
        }
        // Fullscreen controls: F11 toggles, Esc exits — never trap the user.
        if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
            self.fullscreen = !self.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            // Esc leaves satellite focus first, then fullscreen — never trap the user.
            if self.focused_sat.is_some() {
                self.focused_sat = None;
            } else if self.fullscreen {
                self.fullscreen = false;
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            }
        }

        // Pick up a finished background solve.
        if let Some(rx) = self.loading.as_ref() {
            match rx.try_recv() {
                Ok(Ok(l)) => {
                    // Keep a valid focus across an algorithm rerun; drop a stale one.
                    if self.focused_sat.is_some_and(|s| s >= l.sat_pos.len()) {
                        self.focused_sat = None;
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
                Err(std::sync::mpsc::TryRecvError::Empty) => ctx.request_repaint(),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.error = Some("solver thread terminated unexpectedly".into());
                    self.loading = None;
                }
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
            ctx.request_repaint();
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
            ctx.request_repaint();
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
        // Keep repainting while focused so the reticle keeps pulsing.
        if self.focused_sat.is_some() {
            ctx.request_repaint();
        }

        // Full-window 3D scene.
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let size = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
                if response.dragged() {
                    let d = response.drag_delta();
                    self.camera.orbit(d.x, d.y);
                    self.anim = None;
                }
                if response.hovered() {
                    let scroll = ui.input(|i| i.raw_scroll_delta.y);
                    if scroll != 0.0 {
                        self.camera.zoom(scroll);
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
                let streaming = self.scene.update(vp, self.camera.eye(), ph as f32);
                self.scene
                    .set_camera(vp, self.camera.eye(), sun_dir(), self.time);
                self.scene.set_atmosphere(self.show_atmosphere);
                // Keep repainting while tiles are still streaming in.
                if streaming {
                    ctx.request_repaint();
                }
                // Click a satellite to focus it; click empty space to leave focus.
                if response.clicked() {
                    if let Some(p) = response.interact_pointer_pos() {
                        match self.pick(p, rect, vp) {
                            Some(Picked::Sat(s)) => self.enter_focus(s),
                            _ => self.focused_sat = None,
                        }
                    }
                }
                // Resolve the hovered entity once: it drives both the tooltip and
                // (for an interferer) its field-of-interference overlay. Suppressed
                // in focus mode, where the focus panel carries the information.
                let hover = if self.show_ui && self.focused_sat.is_none() {
                    response
                        .hover_pos()
                        .and_then(|ptr| self.hover_entity(ptr, rect, vp).map(|h| (ptr, h)))
                } else {
                    None
                };
                let hover_interferer = hover.as_ref().and_then(|(_, h)| h.interferer);
                let (points, beams) = self.build_world(hover_interferer);
                self.scene.set_points(&points);
                self.scene.set_beams(&beams);
                self.scene.render();
                ui.painter().image(
                    self.texture_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // Hover tooltip, tinted by the hovered terminal's band.
                if let Some((ptr, h)) = hover {
                    egui::Area::new(egui::Id::new("hovertip"))
                        .order(egui::Order::Tooltip)
                        .fixed_pos(ptr + egui::vec2(16.0, 14.0))
                        .show(ctx, |ui| {
                            glass(h.band).inner_margin(9.0).show(ui, |ui| {
                                ui.label(
                                    egui::RichText::new(h.title)
                                        .color(WHITE)
                                        .strong()
                                        .size(12.0),
                                );
                                for l in h.lines {
                                    ui.label(egui::RichText::new(l).color(DIM).size(11.0));
                                }
                            });
                        });
                }
            });

        // Frameless HUD drawn directly on the scene (no panel "bubbles"):
        // all selectors top-left, coverage/solve top-right, transport
        // bottom-center, unserved inspector bottom-right. `H` hides it all.
        let mut clicked = None;
        let mut close_focus = false;
        if self.show_ui {
            egui::Area::new(egui::Id::new("left"))
                .anchor(egui::Align2::LEFT_TOP, [18.0, 16.0])
                .show(ctx, |ui| self.left_panel(ui));
            if self.focused_sat.is_some() {
                // Focus mode: one dedicated study panel stands in for the global
                // coverage / transport / unserved readouts.
                egui::Area::new(egui::Id::new("focus"))
                    .anchor(egui::Align2::RIGHT_TOP, [-18.0, 16.0])
                    .show(ctx, |ui| close_focus = self.focus_panel(ui));
            } else {
                egui::Area::new(egui::Id::new("cov"))
                    .anchor(egui::Align2::RIGHT_TOP, [-18.0, 16.0])
                    .show(ctx, |ui| self.coverage_module(ui));
                egui::Area::new(egui::Id::new("transport"))
                    .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -16.0])
                    .show(ctx, |ui| self.transport_module(ui));
                egui::Area::new(egui::Id::new("uns"))
                    .anchor(egui::Align2::RIGHT_BOTTOM, [-18.0, -16.0])
                    .show(ctx, |ui| clicked = self.unserved_module(ui));
            }
        }

        if close_focus {
            self.focused_sat = None;
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
/// A small dim, uppercase section heading used down the left column.
fn section(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).color(DIM).size(9.5).strong());
    ui.add_space(3.0);
}
fn reason_color(r: Reason) -> egui::Color32 {
    match r {
        Reason::NoSatInView => egui::Color32::from_rgb(150, 160, 185),
        Reason::AllInterfered => egui::Color32::from_rgb(255, 120, 95),
        Reason::AllFull => egui::Color32::from_rgb(255, 205, 95),
        Reason::ColorBlocked => egui::Color32::from_rgb(185, 135, 255),
    }
}
/// Subtle dark backing for the transient hover tooltip (the only floating
/// element that needs contrast over arbitrary scene content). When the hovered
/// item carries a band assignment, the glass takes a faint hint of that band
/// color: a nudged fill plus a clearer band-colored hairline border.
fn glass(band: Option<u8>) -> egui::Frame {
    let (fill, stroke) = match band {
        Some(c) => {
            let rgb = band_rgb(c);
            let mix = |base: f32, ch: f32| (base + ch * 60.0).min(255.0) as u8;
            (
                egui::Color32::from_rgba_unmultiplied(
                    mix(6.0, rgb[0]),
                    mix(8.0, rgb[1]),
                    mix(12.0, rgb[2]),
                    214,
                ),
                egui::Color32::from_rgba_unmultiplied(
                    (rgb[0] * 255.0) as u8,
                    (rgb[1] * 255.0) as u8,
                    (rgb[2] * 255.0) as u8,
                    120,
                ),
            )
        }
        None => (
            egui::Color32::from_rgba_unmultiplied(6, 8, 12, 210),
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 26),
        ),
    };
    egui::Frame::none()
        .fill(fill)
        .stroke(egui::Stroke::new(1.0, stroke))
        .rounding(egui::Rounding::same(9.0))
        .inner_margin(egui::Margin::same(9.0))
}

fn style_egui(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let mut v = egui::Visuals::dark();
    let alpha = |a| egui::Color32::from_rgba_unmultiplied(255, 255, 255, a);
    v.panel_fill = egui::Color32::TRANSPARENT;
    v.override_text_color = Some(WHITE);
    // Dropdown menus / popups read as dark glass.
    v.window_fill = egui::Color32::from_rgba_unmultiplied(10, 12, 16, 240);
    v.window_stroke = egui::Stroke::new(1.0, alpha(26));
    v.window_rounding = egui::Rounding::same(10.0);
    v.selection.bg_fill = alpha(46);
    v.selection.stroke = egui::Stroke::new(1.0, WHITE);
    // Frameless controls: subtle fill + hairline border so each widget reads
    // crisply directly on the scene (no surrounding panel).
    let r = egui::Rounding::same(8.0);
    let strokes = [28u8, 60, 80, 60]; // inactive, hovered, active, open
    let fills = [22u8, 42, 64, 42];
    for (i, w) in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ]
    .into_iter()
    .enumerate()
    {
        w.rounding = r;
        w.bg_stroke = egui::Stroke::new(1.0, alpha(strokes[i]));
        w.weak_bg_fill = alpha(fills[i]);
        w.bg_fill = alpha(fills[i]);
    }
    style.visuals = v;
    style.spacing.item_spacing = egui::vec2(8.0, 7.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    ctx.set_style(style);
}

/// Locate the `test_cases` directory: explicit env var, then the working
/// directory, then walking up from the executable's location.
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

fn reason_idx(r: Reason) -> usize {
    match r {
        Reason::NoSatInView => 0,
        Reason::AllInterfered => 1,
        Reason::AllFull => 2,
        Reason::ColorBlocked => 3,
    }
}
fn reason_from_idx(i: usize) -> Reason {
    match i {
        0 => Reason::NoSatInView,
        1 => Reason::AllInterfered,
        2 => Reason::AllFull,
        _ => Reason::ColorBlocked,
    }
}

fn load_scenario(path: &str, algo: Algorithm) -> Result<Loaded, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let scn = io::Scenario::parse(&text).map_err(|e| format!("parse: {e}"))?;
    let feas = feasibility::build(&scn);
    let trace = trace::run(&scn, &feas, algo);
    let user_pos = scn
        .users
        .iter()
        .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let sat_pos = scn
        .sats
        .iter()
        .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let interferer_pos = scn
        .interferers
        .iter()
        .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let mut reason_counts = [0usize; 4];
    for u in &trace.unassigned {
        reason_counts[reason_idx(u.reason)] += 1;
    }
    Ok(Loaded {
        scn,
        feas,
        user_pos,
        sat_pos,
        interferer_pos,
        trace,
        reason_counts,
    })
}

/// Set up a headless wgpu device + queue (no surface). Shared by the single-frame
/// `--shot` and the `--frames` sequence renderer.
fn init_gpu() -> Result<(Arc<wgpu::Device>, Arc<wgpu::Queue>), String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("no GPU adapter")?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
            .map_err(|e| format!("device: {e}"))?;
    Ok((Arc::new(device), Arc::new(queue)))
}

/// All scene layers on — the default look for headless stills and sequences.
/// Interferers stay off here to keep the hero renders clean (their field is a
/// hover-only overlay anyway).
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
        wgpu::ImageCopyTexture {
            texture: scene.color_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buf,
            layout: wgpu::ImageDataLayout {
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
    device.poll(wgpu::Maintain::Wait);
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
fn screenshot(scenario: &str, out: &str, fraction: f64) -> Result<(), String> {
    let (w, h) = (1600u32, 1000u32);
    let (device, queue) = init_gpu()?;
    let l = load_scenario(scenario, Algorithm::Optimized)?;
    let revealed = (l.trace.events.len() as f64) * fraction.clamp(0.0, 1.0);
    let (points, beams) = compose(&l, revealed, &full_view(), None, None);

    let mut scene = Scene::new(device.clone(), queue.clone());
    // Enable the atmosphere halo for the still (tiles won't stream in headlessly).
    scene.set_tile_source(tiles::TileSource::Dark);
    scene.set_atmosphere(true);
    scene.resize(w, h);
    let cam = OrbitCamera::default();
    scene.set_camera(cam.view_proj(w as f32 / h as f32), cam.eye(), sun_dir(), 0.0);
    scene.set_points(&points);
    scene.set_beams(&beams);
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

    for i in 0..n {
        let f = if n == 1 { 1.0 } else { i as f64 / (n - 1) as f64 };
        let cam = OrbitCamera {
            yaw: yaw0 - orbit_deg.to_radians() * f as f32,
            ..OrbitCamera::default()
        };
        scene.set_camera(cam.view_proj(aspect), cam.eye(), sun_dir(), i as f32 * 0.05);
        let (points, beams) = compose(&l, total * f, &opts, None, None);
        scene.set_points(&points);
        scene.set_beams(&beams);
        scene.render();
        save_frame(&device, &queue, &scene, w, h, &format!("{dir}/frame_{i:05}.png"))?;
    }
    eprintln!("wrote {n} frames to {dir}/ ({w}x{h}, {} beams)", total as usize);
    Ok(())
}

fn main() -> eframe::Result<()> {
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
