//! Live, level-of-detail globe streamed from a Web-Mercator tile server. Tiles
//! are fetched on demand on background threads as the camera moves/zooms —
//! nothing is pre-downloaded. A quadtree is refined each frame against
//! screen-space tile size; a parent tile keeps covering its area until all four
//! children have loaded, so the tiling is always seamless (no overlaps, no gaps
//! over loaded regions). The basemap can be switched at runtime, or turned off
//! entirely for a transparent earth.

use eframe::egui_wgpu::wgpu;
use glam::{Mat4, Vec3};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
// Native fetch backend only: the shared job queue (VecDeque) and the
// Mutex/Condvar that the worker threads park on. Wasm fetches per-tile.
#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Condvar, Mutex};
use wgpu::util::DeviceExt;

const R: f32 = 6371.0; // Earth radius (km)
const MAX_Z: u8 = 16; // max Web-Mercator zoom level
const GRID: u32 = 16; // patch grid subdivisions per side (256 triangles/tile)
const REFINE_PX: f32 = 340.0; // LOD threshold: refine when projected tile > 340px
const CACHE_CAP: usize = 1024; // max tiles in memory (~100MB at 256x256 RGBA8)
#[cfg(not(target_arch = "wasm32"))]
const WORKERS: usize = 8; // background fetch threads (native only; wasm fetches per-tile)

/// A selectable basemap (or none). Off renders nothing — a transparent earth.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TileSource {
    Off,
    Dark,
    Light,
    Satellite,
}

/// How to fetch + shade a given source.
#[derive(Clone, Copy)]
struct SourceSpec {
    /// Base URL; the worker appends `/{z}/{x}/{y}` (or `/{z}/{y}/{x}`) + suffix.
    url: &'static str,
    /// Retina `@2x.png` tiles (CARTO) vs plain.
    retina: bool,
    /// Esri orders tiles `z/y/x` with no extension; XYZ servers use `z/x/y`.
    yx: bool,
    /// Shading style passed to the tile shader (0 dark, 1 light, 2 imagery).
    style: f32,
}

impl TileSource {
    pub const ALL: [TileSource; 4] = [
        TileSource::Off,
        TileSource::Dark,
        TileSource::Light,
        TileSource::Satellite,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TileSource::Off => "Off",
            TileSource::Dark => "Dark",
            TileSource::Light => "Light",
            TileSource::Satellite => "Satellite",
        }
    }

    /// Shading style for the tile shader; 0 when off (unused).
    pub fn style(self) -> f32 {
        self.spec().map(|s| s.style).unwrap_or(0.0)
    }

    // © OpenStreetMap contributors © CARTO; imagery © Esri.
    fn spec(self) -> Option<SourceSpec> {
        match self {
            TileSource::Off => None,
            TileSource::Dark => Some(SourceSpec {
                url: "https://a.basemaps.cartocdn.com/dark_nolabels",
                retina: true,
                yx: false,
                style: 0.0,
            }),
            TileSource::Light => Some(SourceSpec {
                url: "https://a.basemaps.cartocdn.com/rastertiles/voyager",
                retina: true,
                yx: false,
                style: 1.0,
            }),
            TileSource::Satellite => Some(SourceSpec {
                url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile",
                retina: false,
                yx: true,
                style: 2.0,
            }),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TileId {
    z: u8,
    x: u32,
    y: u32,
}

impl TileId {
    fn children(self) -> [TileId; 4] {
        let (z, x, y) = (self.z + 1, self.x * 2, self.y * 2);
        [
            TileId { z, x, y },
            TileId { z, x: x + 1, y },
            TileId { z, x, y: y + 1 },
            TileId {
                z,
                x: x + 1,
                y: y + 1,
            },
        ]
    }
    fn parent(self) -> Option<TileId> {
        (self.z > 0).then(|| TileId {
            z: self.z - 1,
            x: self.x / 2,
            y: self.y / 2,
        })
    }
    /// Normalized Web-Mercator origin + span ([0,1] space, y from north).
    fn merc(self) -> (f32, f32, f32) {
        let n = (1u32 << self.z) as f32;
        (self.x as f32 / n, self.y as f32 / n, 1.0 / n)
    }
    fn center_pos(self) -> Vec3 {
        let (ox, oy, sp) = self.merc();
        merc_to_pos(ox + sp * 0.5, oy + sp * 0.5)
    }
}

fn merc_to_pos(mx: f32, my: f32) -> Vec3 {
    let lon = mx * std::f32::consts::TAU - std::f32::consts::PI;
    let lat = (std::f32::consts::PI - my * std::f32::consts::TAU)
        .sinh()
        .atan();
    let cl = lat.cos();
    Vec3::new(cl * lon.cos(), cl * lon.sin(), lat.sin()) * R
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TileUniform {
    origin: [f32; 2],
    span: f32,
    _pad: f32,
}

struct Tile {
    bind_group: wgpu::BindGroup,
    last_used: u64,
}

/// A fetch request: which tile, from which source, in which generation (so
/// stale results from a previous source are discarded). Native only — wasm
/// passes these fields straight to `spawn_tile_fetch` instead of queueing.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct Job {
    id: TileId,
    spec: SourceSpec,
    gen: u64,
}

struct Loaded {
    id: TileId,
    gen: u64,
    rgba: Vec<u8>,
    w: u32,
    h: u32,
}

/// A worker outcome: a decoded tile, or a fetch/decode failure (so the main
/// thread can free it from `pending` and schedule a bounded retry).
enum TileMsg {
    Loaded(Loaded),
    Failed { id: TileId, gen: u64 },
}

/// Give up re-requesting a tile after this many failures (avoids hammering a
/// genuinely-missing tile; the region just stays at a coarser zoom).
const MAX_TILE_RETRIES: u8 = 4;

pub struct TileGlobe {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    tile_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    patch_vbuf: wgpu::Buffer,
    patch_ibuf: wgpu::Buffer,
    patch_indices: u32,

    source: TileSource,
    gen: u64,
    cache: HashMap<TileId, Tile>,
    pending: HashSet<TileId>,
    /// Tiles whose fetch failed: id -> (retry-not-before frame, attempts).
    failed: HashMap<TileId, (u64, u8)>,
    /// Native fetch backend: worker threads pull `Job`s off this shared queue.
    /// Wasm has no threads, so it has no queue — it fetches per-tile instead.
    #[cfg(not(target_arch = "wasm32"))]
    queue_shared: Arc<(Mutex<VecDeque<Job>>, Condvar)>,
    results: Receiver<TileMsg>,
    /// Wasm fetch backend: cloned into each async `fetch` task to post results
    /// back (no worker pool to hold the sender, as the native build has).
    #[cfg(target_arch = "wasm32")]
    res_tx: Sender<TileMsg>,
    frame: u64,
    render_list: Vec<TileId>,
}

impl TileGlobe {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        cam_bgl: &wgpu::BindGroupLayout,
        samples: u32,
    ) -> Self {
        let tile_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tile-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile-layout"),
            bind_group_layouts: &[cam_bgl, &tile_bgl],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile-shader"),
            source: wgpu::ShaderSource::Wgsl(TILE_WGSL.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tile-pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }],
            },
            // No culling: from inside the globe (core view) the surface faces
            // away from the camera, so back faces must still draw. Depth keeps
            // it correct from the outside.
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview: None,
            cache: None,
        });

        // Shared patch grid in [0,1]² parameter space.
        let mut verts: Vec<[f32; 2]> = Vec::new();
        for j in 0..=GRID {
            for i in 0..=GRID {
                verts.push([i as f32 / GRID as f32, j as f32 / GRID as f32]);
            }
        }
        let mut idx: Vec<u32> = Vec::new();
        let row = GRID + 1;
        for j in 0..GRID {
            for i in 0..GRID {
                let a = j * row + i;
                let b = a + row;
                idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        let patch_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch-v"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let patch_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch-i"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let patch_indices = idx.len() as u32;

        // Fetch backend. Native: a shared job queue drained by a pool of worker
        // threads (each holding a `res_tx` clone). Wasm: no threads — `res_tx` is
        // kept and cloned into a `fetch` task per missing tile (see `update`).
        let (res_tx, results) = std::sync::mpsc::channel::<TileMsg>();
        #[cfg(not(target_arch = "wasm32"))]
        let queue_shared = Arc::new((Mutex::new(VecDeque::<Job>::new()), Condvar::new()));
        #[cfg(not(target_arch = "wasm32"))]
        for _ in 0..WORKERS {
            let qs = queue_shared.clone();
            let tx = res_tx.clone();
            std::thread::spawn(move || worker(qs, tx));
        }

        TileGlobe {
            device,
            queue,
            pipeline,
            tile_bgl,
            sampler,
            patch_vbuf,
            patch_ibuf,
            patch_indices,
            source: TileSource::Off,
            gen: 0,
            cache: HashMap::new(),
            pending: HashSet::new(),
            failed: HashMap::new(),
            #[cfg(not(target_arch = "wasm32"))]
            queue_shared,
            results,
            #[cfg(target_arch = "wasm32")]
            res_tx,
            frame: 0,
            render_list: Vec::new(),
        }
    }

    /// Switch basemap (or turn it off). Drops the current tile set and any
    /// in-flight requests; results from the old source are ignored via `gen`.
    pub fn set_source(&mut self, source: TileSource) {
        if source == self.source {
            return;
        }
        self.source = source;
        self.gen += 1;
        self.cache.clear();
        self.pending.clear();
        self.failed.clear();
        self.render_list.clear();
        // Drop any queued-but-unstarted native fetches from the old source.
        // (Wasm has no queue; its in-flight fetches are discarded by `gen`.)
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (lock, _) = &*self.queue_shared;
            lock.lock().unwrap().clear();
        }
    }

    /// Pump completed downloads, then recompute the visible tile set for this
    /// camera and request any missing tiles. Returns true while tiles are still
    /// streaming in (so the caller keeps repainting).
    pub fn update(&mut self, view_proj: Mat4, eye: Vec3, viewport_h: f32) -> bool {
        self.frame += 1;

        // Basemap off: nothing to draw, nothing streaming.
        let Some(spec) = self.source.spec() else {
            // Drain any late arrivals so the channel doesn't grow unbounded.
            while self.results.try_recv().is_ok() {}
            self.render_list.clear();
            return false;
        };

        // Upload any tiles that finished downloading (bounded per frame).
        for _ in 0..32 {
            let Ok(msg) = self.results.try_recv() else {
                break;
            };
            let t = match msg {
                TileMsg::Loaded(t) => t,
                TileMsg::Failed { id, gen } => {
                    // Free the slot and schedule a bounded, backed-off retry.
                    if gen == self.gen {
                        self.pending.remove(&id);
                        let e = self.failed.entry(id).or_insert((0, 0));
                        e.1 = e.1.saturating_add(1);
                        e.0 = self.frame + 30 * e.1 as u64;
                    }
                    continue;
                }
            };
            if t.gen != self.gen {
                continue; // from a previous basemap — discard
            }
            self.pending.remove(&t.id);
            self.failed.remove(&t.id);
            // Stored as plain UNORM (not sRGB): the offscreen scene target is
            // Rgba8Unorm and every other element writes display-space values, so
            // tiles must pass through unmodified to stay color-consistent.
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("tile"),
                size: wgpu::Extent3d {
                    width: t.w,
                    height: t.h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &t.rgba,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * t.w),
                    rows_per_image: Some(t.h),
                },
                wgpu::Extent3d {
                    width: t.w,
                    height: t.h,
                    depth_or_array_layers: 1,
                },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            let (ox, oy, sp) = t.id.merc();
            let unif = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("tile-u"),
                    contents: bytemuck::bytes_of(&TileUniform {
                        origin: [ox, oy],
                        span: sp,
                        _pad: 0.0,
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile-bind"),
                layout: &self.tile_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: unif.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.cache.insert(
                t.id,
                Tile {
                    bind_group,
                    last_used: self.frame,
                },
            );
        }

        // Refine the quadtree.
        let mut want_request: Vec<TileId> = Vec::new();
        let mut list: Vec<TileId> = Vec::new();
        let dist = eye.length();
        // Roots: the 4 z=1 tiles.
        for x in 0..2 {
            for y in 0..2 {
                self.refine(
                    TileId { z: 1, x, y },
                    view_proj,
                    eye,
                    dist,
                    viewport_h,
                    &mut list,
                    &mut want_request,
                );
            }
        }
        for id in &list {
            if let Some(t) = self.cache.get_mut(id) {
                t.last_used = self.frame;
            }
        }
        self.render_list = list;

        // Request missing tiles (nearest-first), de-duplicated.
        want_request.sort_by(|a, b| {
            let da = a.center_pos().distance_squared(eye);
            let db = b.center_pos().distance_squared(eye);
            da.partial_cmp(&db).unwrap()
        });
        let mut to_fetch: Vec<TileId> = Vec::new();
        for id in want_request {
            if self.cache.contains_key(&id) || self.pending.contains(&id) {
                continue;
            }
            // Respect retry backoff; give up after MAX_TILE_RETRIES.
            if let Some(&(retry_after, attempts)) = self.failed.get(&id) {
                if attempts >= MAX_TILE_RETRIES || self.frame < retry_after {
                    continue;
                }
                self.failed.remove(&id);
            }
            self.pending.insert(id);
            to_fetch.push(id);
        }

        // Native: hand the jobs to the worker thread pool. Wasm: kick off an
        // async `fetch` per tile (the browser caps concurrency per origin); both
        // deliver decoded tiles into `results`, drained at the top of update().
        #[cfg(not(target_arch = "wasm32"))]
        if !to_fetch.is_empty() {
            let (lock, cv) = &*self.queue_shared;
            let mut q = lock.lock().unwrap();
            for id in to_fetch {
                q.push_back(Job { id, spec, gen: self.gen });
            }
            cv.notify_all();
        }
        #[cfg(target_arch = "wasm32")]
        for id in to_fetch {
            spawn_tile_fetch(id, spec, self.gen, self.res_tx.clone());
        }

        self.evict();
        !self.pending.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    fn refine(
        &self,
        id: TileId,
        vp: Mat4,
        eye: Vec3,
        dist: f32,
        vh: f32,
        out: &mut Vec<TileId>,
        req: &mut Vec<TileId>,
    ) {
        if !visible(id, vp, eye, dist) {
            return;
        }
        let want_split = id.z < MAX_Z && screen_px(id, eye, vh) > REFINE_PX;
        if want_split {
            let kids = id.children();
            if kids.iter().all(|k| self.cache.contains_key(k)) {
                for k in kids {
                    self.refine(k, vp, eye, dist, vh, out, req);
                }
                return;
            }
            for k in kids {
                if !self.cache.contains_key(&k) {
                    req.push(k);
                }
            }
        }
        // Render the highest loaded tile on this path (climb if needed).
        let mut cur = id;
        loop {
            if self.cache.contains_key(&cur) {
                out.push(cur);
                if cur != id {
                    req.push(id);
                }
                return;
            }
            match cur.parent() {
                Some(p) => cur = p,
                None => {
                    req.push(id);
                    return;
                }
            }
        }
    }

    fn evict(&mut self) {
        if self.cache.len() <= CACHE_CAP {
            return;
        }
        let mut by_age: Vec<(TileId, u64)> =
            self.cache.iter().map(|(k, v)| (*k, v.last_used)).collect();
        by_age.sort_by_key(|(_, a)| *a);
        let remove = self.cache.len() - CACHE_CAP;
        for (id, _) in by_age.into_iter().take(remove) {
            self.cache.remove(&id);
        }
    }

    /// Draw the current tile set. Camera bind group (group 0) must already be set.
    pub fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.patch_vbuf.slice(..));
        pass.set_index_buffer(self.patch_ibuf.slice(..), wgpu::IndexFormat::Uint32);
        for id in &self.render_list {
            if let Some(t) = self.cache.get(id) {
                pass.set_bind_group(1, &t.bind_group, &[]);
                pass.draw_indexed(0..self.patch_indices, 0, 0..1);
            }
        }
    }
}

fn screen_px(id: TileId, eye: Vec3, vh: f32) -> f32 {
    let c = id.center_pos();
    let lat = (c.z / R).clamp(-1.0, 1.0).asin();
    let ground = R * std::f32::consts::TAU / (1u32 << id.z) as f32 * lat.cos().max(0.12);
    let d = (c - eye).length().max(1.0);
    ground / d * vh * 1.207
}

fn visible(id: TileId, vp: Mat4, eye: Vec3, dist: f32) -> bool {
    let c = id.center_pos();
    // Horizon cull: is the tile center on the near hemisphere (with margin)?
    // Skipped when the camera is inside the globe (core view), where every tile
    // faces away and the whole interior dome is visible.
    let cos_to = c.normalize().dot(eye.normalize());
    if dist > R {
        let ang_radius = std::f32::consts::PI / (1u32 << id.z) as f32;
        if cos_to < (R / dist) - ang_radius.sin() - 0.04 {
            return false;
        }
    }
    // Frustum cull: project center; keep if near the screen.
    let clip = vp * c.extend(1.0);
    if clip.w <= 0.0 {
        return cos_to > 0.7; // behind near plane but possibly large/close
    }
    let ndc = clip.truncate() / clip.w;
    let margin = (screen_px(id, eye, 2.0)).max(0.6) + 0.4;
    ndc.x.abs() < 1.0 + margin && ndc.y.abs() < 1.0 + margin
}

/// Tile URL for a source + tile id. CARTO: `/z/x/y@2x.png`; Esri imagery:
/// `/z/y/x` (no extension, returns jpeg).
fn tile_url(spec: &SourceSpec, id: TileId) -> String {
    if spec.yx {
        format!("{}/{}/{}/{}", spec.url, id.z, id.y, id.x)
    } else if spec.retina {
        format!("{}/{}/{}/{}@2x.png", spec.url, id.z, id.x, id.y)
    } else {
        format!("{}/{}/{}/{}.png", spec.url, id.z, id.x, id.y)
    }
}

/// Decode encoded tile bytes (PNG/JPEG) to raw RGBA8 + dimensions. Shared by the
/// native worker threads and the wasm `fetch` path.
fn decode_rgba(bytes: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
    let rgba = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = rgba.dimensions();
    Some((rgba.into_raw(), w, h))
}

/// Wasm tile fetch: `fetch` the tile, decode it, and post the result into
/// `results` (the same channel the native worker threads use). The browser caps
/// concurrent requests per origin, so kicking one off per missing tile is fine.
#[cfg(target_arch = "wasm32")]
fn spawn_tile_fetch(id: TileId, spec: SourceSpec, gen: u64, tx: Sender<TileMsg>) {
    let url = tile_url(&spec, id);
    wasm_bindgen_futures::spawn_local(async move {
        let msg = match fetch_decode(&url).await {
            Some((rgba, w, h)) => TileMsg::Loaded(Loaded { id, gen, rgba, w, h }),
            None => TileMsg::Failed { id, gen },
        };
        let _ = tx.send(msg);
    });
}

#[cfg(target_arch = "wasm32")]
async fn fetch_decode(url: &str) -> Option<(Vec<u8>, u32, u32)> {
    let resp = ehttp::fetch_async(ehttp::Request::get(url)).await.ok()?;
    if !resp.ok {
        return None;
    }
    decode_rgba(&resp.bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn worker(qs: Arc<(Mutex<VecDeque<Job>>, Condvar)>, tx: Sender<TileMsg>) {
    let (lock, cv) = &*qs;
    loop {
        let job = {
            let mut q = lock.lock().expect("tile worker: lock poisoned");
            loop {
                if let Some(job) = q.pop_front() {
                    break job;
                }
                q = cv.wait(q).expect("tile worker: condition variable failed");
            }
        };
        let Job { id, spec, gen } = job;
        let url = tile_url(&spec, id);
        let fail = |tx: &Sender<TileMsg>| {
            let _ = tx.send(TileMsg::Failed { id, gen });
        };
        let Ok(resp) = ureq::get(&url).set("User-Agent", "beam-viz/1.0").call() else {
            fail(&tx);
            continue;
        };
        let mut buf = Vec::new();
        if std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf).is_err() {
            fail(&tx);
            continue;
        }
        match decode_rgba(&buf) {
            Some((rgba, w, h)) => {
                let _ = tx.send(TileMsg::Loaded(Loaded { id, gen, rgba, w, h }));
            }
            None => fail(&tx),
        }
    }
}

const TILE_WGSL: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct TileU { origin: vec2<f32>, span: f32, pad: f32 };
@group(1) @binding(0) var<uniform> tile: TileU;
@group(1) @binding(1) var tex: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

const R: f32 = 6371.0;
const PI: f32 = 3.14159265;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) n: vec3<f32>, @location(2) world: vec3<f32> };
@vertex fn vs(@location(0) p: vec2<f32>) -> VOut {
    let mx = tile.origin.x + p.x * tile.span;
    let my = tile.origin.y + p.y * tile.span;
    let lon = mx * 2.0 * PI - PI;
    let lat = atan(sinh(PI - my * 2.0 * PI));
    let cl = cos(lat);
    let pos = vec3<f32>(cl * cos(lon), cl * sin(lon), sin(lat)) * R;
    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(pos, 1.0);
    o.uv = p;
    o.n = pos / R;
    o.world = pos;
    return o;
}
@fragment fn fs(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.n);
    let view = normalize(cam.cam_pos.xyz - in.world);
    let sun = normalize(cam.sun_dir.xyz);
    let base = textureSample(tex, samp, in.uv).rgb;
    let lambert = max(dot(n, sun), 0.0);
    let style = cam.params.w; // 0 dark, 1 light, 2 satellite imagery
    var col: vec3<f32>;
    var rim_tint: vec3<f32>;
    if (style < 0.5) {
        // Dark basemap — sleek, gentle directional shading.
        col = base * (0.95 + 0.40 * lambert);
        rim_tint = vec3<f32>(0.16, 0.34, 0.78);
    } else if (style < 1.5) {
        // Light (Voyager) basemap.
        col = base * (0.82 + 0.42 * lambert);
        rim_tint = vec3<f32>(0.30, 0.46, 0.86);
    } else {
        // Satellite imagery — fuller day/night terminator.
        col = base * (0.55 + 0.85 * lambert);
        rim_tint = vec3<f32>(0.34, 0.52, 0.92);
    }
    // Cool limb glow.
    let rim = pow(1.0 - max(dot(n, view), 0.0), 4.0);
    col += rim_tint * rim * 0.45;
    return vec4<f32>(col, 1.0);
}
"#;
