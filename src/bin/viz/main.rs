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
}

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

    (points, beams)
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
    bands: [bool; 4],

    tile_source: tiles::TileSource,
    /// Window fullscreen state (opens fullscreen; F11 toggles, Esc exits).
    fullscreen: bool,
    /// Master HUD visibility (toggle with `H`).
    show_ui: bool,

    selected: Option<usize>, // index into trace.unassigned
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
            bands: [true; 4],
            tile_source: tiles::TileSource::Off,
            fullscreen: true,
            show_ui: true,
            selected: None,
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
    fn hover_entity(
        &self,
        ptr: egui::Pos2,
        rect: egui::Rect,
        vp: glam::Mat4,
    ) -> Option<(String, Vec<String>)> {
        enum H {
            Sat(usize),
            User(usize),
        }
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
        let mut best: Option<(f32, H)> = None;
        for (i, sp) in l.sat_pos.iter().enumerate() {
            if outside && sp.normalize().dot(eye_n) < horizon - 0.2 {
                continue;
            }
            if let Some(s) = project(*sp) {
                let dd = s.distance(ptr);
                if dd < 13.0 && best.as_ref().is_none_or(|(b, _)| dd < *b) {
                    best = Some((dd, H::Sat(i)));
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
                    best = Some((dd, H::User(i)));
                }
            }
        }
        let revealed = (self.revealed as usize).min(l.trace.events.len());
        match best?.1 {
            H::Sat(i) => {
                let load = l.trace.events[..revealed]
                    .iter()
                    .filter(|e| e.sat as usize == i)
                    .count();
                Some((
                    format!("Satellite {}", l.scn.sat_ids[i]),
                    vec![format!("{load} / 32 beams in use")],
                ))
            }
            H::User(i) => {
                if let Some(e) = l.trace.events[..revealed]
                    .iter()
                    .find(|e| e.user as usize == i)
                {
                    Some((
                        format!("Terminal {}", l.scn.user_ids[i]),
                        vec![format!(
                            "served · band {} · sat {}",
                            BANDS[e.color as usize], l.scn.sat_ids[e.sat as usize]
                        )],
                    ))
                } else {
                    let line = l
                        .trace
                        .unassigned
                        .iter()
                        .find(|u| u.user as usize == i)
                        .map(|u| u.reason.label().to_string())
                        .unwrap_or_else(|| "not yet assigned".into());
                    Some((format!("Terminal {}", l.scn.user_ids[i]), vec![line]))
                }
            }
        }
    }

    /// Build the points + beams to display for the current frame.
    fn build_world(&self) -> (Vec<PointInstance>, Vec<BeamInstance>) {
        let Some(l) = &self.loaded else {
            return (Vec::new(), Vec::new());
        };
        compose(l, self.revealed, &self.view_opts(), self.selected)
    }

    fn view_opts(&self) -> ViewOpts {
        ViewOpts {
            bands: self.bands,
            show_full: self.show_full,
            show_empty: self.show_empty,
            show_uncovered: self.show_uncovered,
            show_beams: self.show_beams,
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

        if changed_scn {
            self.load();
        } else if changed_algo {
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
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && self.fullscreen {
            self.fullscreen = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        }

        // Pick up a finished background solve.
        if let Some(rx) = self.loading.as_ref() {
            match rx.try_recv() {
                Ok(Ok(l)) => {
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

        // Full-window 3D scene.
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let size = ui.available_size();
                let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
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
                // Keep repainting while tiles are still streaming in.
                if streaming {
                    ctx.request_repaint();
                }
                let (points, beams) = self.build_world();
                self.scene.set_points(&points);
                self.scene.set_beams(&beams);
                self.scene.render();
                ui.painter().image(
                    self.texture_id,
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );

                // Hover tooltip for satellites / terminals under the cursor.
                if self.show_ui {
                    if let Some(ptr) = response.hover_pos() {
                        if let Some((title, lines)) = self.hover_entity(ptr, rect, vp) {
                            egui::Area::new(egui::Id::new("hovertip"))
                                .order(egui::Order::Tooltip)
                                .fixed_pos(ptr + egui::vec2(16.0, 14.0))
                                .show(ctx, |ui| {
                                    glass().inner_margin(9.0).show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new(title)
                                                .color(WHITE)
                                                .strong()
                                                .size(12.0),
                                        );
                                        for l in lines {
                                            ui.label(egui::RichText::new(l).color(DIM).size(11.0));
                                        }
                                    });
                                });
                        }
                    }
                }
            });

        // Frameless HUD drawn directly on the scene (no panel "bubbles"):
        // all selectors top-left, coverage/solve top-right, transport
        // bottom-center, unserved inspector bottom-right. `H` hides it all.
        let mut clicked = None;
        if self.show_ui {
            egui::Area::new(egui::Id::new("left"))
                .anchor(egui::Align2::LEFT_TOP, [18.0, 16.0])
                .show(ctx, |ui| self.left_panel(ui));
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
/// element that needs contrast over arbitrary scene content).
fn glass() -> egui::Frame {
    egui::Frame::none()
        .fill(egui::Color32::from_rgba_unmultiplied(6, 8, 12, 210))
        .stroke(egui::Stroke::new(
            1.0,
            egui::Color32::from_rgba_unmultiplied(255, 255, 255, 26),
        ))
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
    let mut reason_counts = [0usize; 4];
    for u in &trace.unassigned {
        reason_counts[reason_idx(u.reason)] += 1;
    }
    Ok(Loaded {
        scn,
        feas,
        user_pos,
        sat_pos,
        trace,
        reason_counts,
    })
}

/// Headless: render one frame of a scenario to a PNG (no window). Used to verify
/// and preview the renderer. `--shot <scenario.txt> <out.png> [fraction 0..1]`.
fn screenshot(scenario: &str, out: &str, fraction: f64) -> Result<(), String> {
    let (w, h) = (1600u32, 1000u32);
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
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let l = load_scenario(scenario, Algorithm::Optimized)?;
    let revealed = (l.trace.events.len() as f64) * fraction.clamp(0.0, 1.0);
    let opts = ViewOpts {
        bands: [true; 4],
        show_full: true,
        show_empty: true,
        show_uncovered: true,
        show_beams: true,
    };
    let (points, beams) = compose(&l, revealed, &opts, None);

    let mut scene = Scene::new(device.clone(), queue.clone());
    // Enable the atmosphere halo for the still (tiles won't stream in headlessly).
    scene.set_tile_source(tiles::TileSource::Dark);
    scene.resize(w, h);
    let cam = OrbitCamera::default();
    scene.set_camera(
        cam.view_proj(w as f32 / h as f32),
        cam.eye(),
        sun_dir(),
        0.0,
    );
    scene.set_points(&points);
    scene.set_beams(&beams);
    scene.render();

    // Read the offscreen color texture back and save it.
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
    eprintln!(
        "wrote {out} ({}x{}, {} beams of {})",
        w,
        h,
        revealed as usize,
        l.trace.events.len()
    );
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
