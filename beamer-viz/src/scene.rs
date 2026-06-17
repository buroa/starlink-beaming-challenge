//! Cinematic GPU rendering: a tile-sourced satellite Earth (Esri World Imagery)
//! under near-uniform daylight with a Fresnel atmosphere halo, a procedural
//! galaxy + starfield backdrop, instanced billboard points (satellites and
//! terminals), and instanced "beam" quads that zap into existence and carry
//! flowing data packets. Drawn to an offscreen texture the egui layer
//! composites.

use super::tiles::{TileGlobe, TileSource};
use eframe::egui_wgpu::wgpu;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    params: [f32; 4],  // viewport.x, viewport.y, time, map_style
    params2: [f32; 4], // load_pulse, _, _, _
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobeVertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointInstance {
    pub pos: [f32; 3],
    pub size: f32,
    pub color: [f32; 4],
}

/// One beam ribbon: satellite `a` → terminal `b`, colored by band.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BeamInstance {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
}

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const MSAA: u32 = 4;

pub struct Scene {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,

    camera_buf: wgpu::Buffer,
    cam_bind: wgpu::BindGroup,
    tile_globe: TileGlobe,

    bg_pipeline: wgpu::RenderPipeline,
    atmo_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    beam_pipeline: wgpu::RenderPipeline,

    globe_vbuf: wgpu::Buffer,
    globe_ibuf: wgpu::Buffer,
    globe_indices: u32,
    quad_vbuf: wgpu::Buffer,

    point_buf: wgpu::Buffer,
    point_cap: u64,
    point_count: u32,
    beam_buf: wgpu::Buffer,
    beam_cap: u64,
    beam_count: u32,

    /// Tile-shading style (passed to shaders as camera `params.w`).
    map_style: f32,
    /// Draw the Fresnel atmosphere halo? Controlled independently of the basemap
    /// via [`Scene::set_atmosphere`], so the hue can glow over a transparent earth.
    draw_atmo: bool,
    /// Solve/loading animation pulse (>0 while a background solve is in flight),
    /// fed to the background shader for the screen-centered wavefront.
    load_t: f32,
    /// Satellite-focus mode: suppress the globe + atmosphere so the flat ground
    /// plane stands alone over the nebula, regardless of basemap/halo settings.
    focus_mode: bool,

    size: (u32, u32),
    color_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

impl Scene {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cam_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cam-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let cam_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam-bind"),
            layout: &cam_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        // Live-streamed satellite globe (its own pipeline + tile cache).
        let tile_globe = TileGlobe::new(device.clone(), queue.clone(), &cam_layout, MSAA);

        let layout0 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout0"),
            bind_group_layouts: &[&cam_layout],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaders"),
            source: wgpu::ShaderSource::Wgsl(SHADERS.into()),
        });

        let bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bg"),
            layout: Some(&layout0),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "bg_vs",
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Always)),
            multisample: wgpu::MultisampleState {
                count: MSAA,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "bg_fs",
                compilation_options: Default::default(),
                targets: &[Some(COLOR_FORMAT.into())],
            }),
            multiview: None,
            cache: None,
        });

        let globe_buffers = &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlobeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        }];
        let atmo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("atmo"),
            layout: Some(&layout0),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "atmo_vs",
                compilation_options: Default::default(),
                buffers: globe_buffers,
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(depth_state(false, wgpu::CompareFunction::Always)),
            multisample: wgpu::MultisampleState {
                count: MSAA,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "atmo_fs",
                compilation_options: Default::default(),
                targets: &[Some(additive_target())],
            }),
            multiview: None,
            cache: None,
        });

        // Unit quad in [-1,1]² reused by points and beams.
        const QUAD: &[[f32; 2]] = &[
            [-1., -1.],
            [1., -1.],
            [1., 1.],
            [-1., -1.],
            [1., 1.],
            [-1., 1.],
        ];
        let quad_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad"),
            contents: bytemuck::cast_slice(QUAD),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("points"),
            layout: Some(&layout0),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "point_vs",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<PointInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32, 3 => Float32x4],
                    },
                ],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_state(false, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "point_fs",
                compilation_options: Default::default(),
                targets: &[Some(additive_target())],
            }),
            multiview: None,
            cache: None,
        });

        let beam_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("beams"),
            layout: Some(&layout0),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "beam_vs",
                compilation_options: Default::default(),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 8,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<BeamInstance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x3, 2 => Float32x3, 3 => Float32x4],
                    },
                ],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(depth_state(false, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState { count: MSAA, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "beam_fs",
                compilation_options: Default::default(),
                targets: &[Some(additive_target())],
            }),
            multiview: None,
            cache: None,
        });

        let (gv, gi) = globe_mesh(96, 192);
        let globe_indices = gi.len() as u32;
        let globe_vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globe-v"),
            contents: bytemuck::cast_slice(&gv),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let globe_ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globe-i"),
            contents: bytemuck::cast_slice(&gi),
            usage: wgpu::BufferUsages::INDEX,
        });

        let point_cap = 1024;
        let point_buf = make_dyn(
            &device,
            point_cap * std::mem::size_of::<PointInstance>() as u64,
        );
        let beam_cap = 1024;
        let beam_buf = make_dyn(
            &device,
            beam_cap * std::mem::size_of::<BeamInstance>() as u64,
        );

        let size = (16, 16);
        let (color_tex, color_view, msaa_view, depth_view) = make_targets(&device, size);

        Scene {
            device,
            queue,
            camera_buf,
            cam_bind,
            tile_globe,
            bg_pipeline,
            atmo_pipeline,
            point_pipeline,
            beam_pipeline,
            globe_vbuf,
            globe_ibuf,
            globe_indices,
            quad_vbuf,
            point_buf,
            point_cap,
            point_count: 0,
            beam_buf,
            beam_cap,
            beam_count: 0,
            map_style: 0.0,
            draw_atmo: false,
            load_t: 0.0,
            focus_mode: false,
            size,
            color_tex,
            color_view,
            msaa_view,
            depth_view,
        }
    }

    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }
    /// Used by the native headless frame capture (`save_frame`); wasm renders
    /// straight to the canvas and never reads the offscreen color texture back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn color_texture(&self) -> &wgpu::Texture {
        &self.color_tex
    }

    pub fn resize(&mut self, w: u32, h: u32) -> bool {
        let w = w.max(1);
        let h = h.max(1);
        if (w, h) == self.size {
            return false;
        }
        self.size = (w, h);
        let (ct, cv, mv, dv) = make_targets(&self.device, (w, h));
        self.color_tex = ct;
        self.color_view = cv;
        self.msaa_view = mv;
        self.depth_view = dv;
        true
    }

    /// Per-frame tile streaming update; returns true while tiles are loading.
    pub fn update(&mut self, view_proj: Mat4, eye: Vec3, viewport_h: f32) -> bool {
        self.tile_globe.update(view_proj, eye, viewport_h)
    }

    /// Switch the basemap (or turn it off → transparent earth). The atmosphere
    /// halo is controlled separately by [`Scene::set_atmosphere`].
    pub fn set_tile_source(&mut self, source: TileSource) {
        self.tile_globe.set_source(source);
        self.map_style = source.style();
    }

    /// Show or hide the Fresnel atmosphere halo, independent of the basemap.
    pub fn set_atmosphere(&mut self, on: bool) {
        self.draw_atmo = on;
    }

    /// Drive the background's loading wavefront. Pass the elapsed time while a
    /// background solve is in flight, or 0.0 to switch the effect off.
    pub fn set_load(&mut self, t: f32) {
        self.load_t = t;
    }

    /// Enter/leave satellite-focus mode: hides the globe + atmosphere so the
    /// flat ground plane reads cleanly over the nebula backdrop.
    pub fn set_focus_mode(&mut self, on: bool) {
        self.focus_mode = on;
    }

    pub fn set_camera(&self, view_proj: Mat4, cam_pos: Vec3, sun: Vec3, time: f32) {
        let inv = view_proj.inverse();
        let u = CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: inv.to_cols_array_2d(),
            cam_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            params: [self.size.0 as f32, self.size.1 as f32, time, self.map_style],
            params2: [self.load_t, 0.0, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn set_points(&mut self, points: &[PointInstance]) {
        self.point_count = points.len() as u32;
        if points.is_empty() {
            return;
        }
        let needed = points.len() as u64;
        if needed > self.point_cap {
            self.point_cap = needed.next_power_of_two();
            self.point_buf = make_dyn(
                &self.device,
                self.point_cap * std::mem::size_of::<PointInstance>() as u64,
            );
        }
        self.queue
            .write_buffer(&self.point_buf, 0, bytemuck::cast_slice(points));
    }

    pub fn set_beams(&mut self, beams: &[BeamInstance]) {
        self.beam_count = beams.len() as u32;
        if beams.is_empty() {
            return;
        }
        let needed = beams.len() as u64;
        if needed > self.beam_cap {
            self.beam_cap = needed.next_power_of_two();
            self.beam_buf = make_dyn(
                &self.device,
                self.beam_cap * std::mem::size_of::<BeamInstance>() as u64,
            );
        }
        self.queue
            .write_buffer(&self.beam_buf, 0, bytemuck::cast_slice(beams));
    }

    pub fn render(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scene"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa_view,
                    resolve_target: Some(&self.color_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.cam_bind, &[]);

            pass.set_pipeline(&self.bg_pipeline);
            pass.draw(0..3, 0..1);

            // Live satellite globe (group 0 camera already bound). Suppressed in
            // satellite-focus mode, where the flat ground plane replaces it.
            if !self.focus_mode {
                self.tile_globe.render(&mut pass);
            }

            // Fresnel atmosphere halo — toggled independently of the basemap
            // (see set_atmosphere), so the glow can ride over a transparent earth.
            if self.draw_atmo && !self.focus_mode {
                pass.set_pipeline(&self.atmo_pipeline);
                pass.set_vertex_buffer(0, self.globe_vbuf.slice(..));
                pass.set_index_buffer(self.globe_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.globe_indices, 0, 0..1);
            }

            if self.beam_count > 0 {
                pass.set_pipeline(&self.beam_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.set_vertex_buffer(1, self.beam_buf.slice(..));
                pass.draw(0..6, 0..self.beam_count);
            }
            if self.point_count > 0 {
                pass.set_pipeline(&self.point_pipeline);
                pass.set_vertex_buffer(0, self.quad_vbuf.slice(..));
                pass.set_vertex_buffer(1, self.point_buf.slice(..));
                pass.draw(0..6, 0..self.point_count);
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }
}

/// Create a dynamic GPU buffer for vertex/instance data (resizable on realloc).
fn make_dyn(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dyn"),
        size,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: write,
        depth_compare: compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn additive_target() -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format: COLOR_FORMAT,
        blend: Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        }),
        write_mask: wgpu::ColorWrites::ALL,
    }
}

fn make_targets(
    device: &wgpu::Device,
    (w, h): (u32, u32),
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::TextureView,
    wgpu::TextureView,
) {
    let ext = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: 1,
    };
    // 1-sample resolve target (what egui samples).
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-color"),
        size: ext,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    // MSAA color + depth (4 samples).
    let msaa = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-msaa"),
        size: ext,
        mip_level_count: 1,
        sample_count: MSAA,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth"),
        size: ext,
        mip_level_count: 1,
        sample_count: MSAA,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let cv = color.create_view(&wgpu::TextureViewDescriptor::default());
    let mv = msaa.create_view(&wgpu::TextureViewDescriptor::default());
    let dv = depth.create_view(&wgpu::TextureViewDescriptor::default());
    (color, cv, mv, dv)
}

fn globe_mesh(stacks: u32, slices: u32) -> (Vec<GlobeVertex>, Vec<u32>) {
    const R: f32 = 6371.0;
    let mut verts = Vec::new();
    for i in 0..=stacks {
        let phi = (i as f32 / stacks as f32) * std::f32::consts::PI;
        let (sp, cp) = phi.sin_cos();
        for j in 0..=slices {
            let theta = (j as f32 / slices as f32) * std::f32::consts::TAU;
            let (st, ct) = theta.sin_cos();
            let n = [sp * ct, sp * st, cp]; // z = north pole (ECEF)
            verts.push(GlobeVertex {
                pos: [n[0] * R, n[1] * R, n[2] * R],
                normal: n,
            });
        }
    }
    let mut idx = Vec::new();
    let row = slices + 1;
    for i in 0..stacks {
        for j in 0..slices {
            let a = i * row + j;
            let b = a + row;
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (verts, idx)
}

const SHADERS: &str = include_str!("shaders.wgsl");
