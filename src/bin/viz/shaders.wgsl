// Scene shaders: galaxy backdrop, tile-sourced satellite Earth, atmosphere halo,
// glowing billboard points, and zap-in beams with flowing data packets.

const PI: f32 = 3.14159265;

struct Camera {
    view_proj: mat4x4<f32>,       // view * projection matrix
    inv_view_proj: mat4x4<f32>,   // inverse view-proj; unprojects NDC to world for background rays
    cam_pos: vec4<f32>,           // camera position (ECEF, w unused)
    sun_dir: vec4<f32>,           // normalized sun direction (ECEF, w unused)
    params: vec4<f32>,            // (viewport.x, viewport.y, time, map_style)
};
@group(0) @binding(0) var<uniform> cam: Camera;

// ----------------------------------------------------------------- noise
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash31(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 5; i = i + 1) {
        v += a * vnoise(q);
        q = q * 2.02;
        a *= 0.5;
    }
    return v;
}
const STAR_TWEEN_BASE: f32 = 0.7;
const STAR_TWEEN_AMP: f32 = 0.3;
const STAR_JITTER_SCALE: f32 = 0.7;
const STAR_SMOOTH_OUTER: f32 = 0.5;
const STAR_FREQ: f32 = 120.0;

fn star_layer(dir: vec3<f32>, scale: f32, thresh: f32) -> f32 {
    let p = dir * scale;
    let id = floor(p);
    let h = hash31(id);
    if (h < thresh) {
        let jitter = vec3<f32>(hash31(id + 1.0), hash31(id + 2.0), hash31(id + 3.0)) - 0.5;
        let center = id + 0.5 + jitter * STAR_JITTER_SCALE;
        let d = length(p - center);
        let tw = STAR_TWEEN_BASE + STAR_TWEEN_AMP * sin(cam.params.z * 2.0 + h * STAR_FREQ);
        return smoothstep(STAR_SMOOTH_OUTER, 0.0, d) * tw;
    }
    return 0.0;
}
// Black space with a subtle cool gradient and a sparse, calm starfield —
// matching Google Earth's clean backdrop (no nebula).
fn galaxy(dir: vec3<f32>) -> vec3<f32> {
    var col = vec3<f32>(0.004, 0.006, 0.013);
    col += vec3<f32>(0.0, 0.004, 0.012) * smoothstep(-0.4, 0.6, dir.z);
    var s = 0.0;
    s += star_layer(dir, 120.0, 0.010) * 1.1;
    s += star_layer(dir, 240.0, 0.008) * 0.7;
    s += star_layer(dir, 480.0, 0.006) * 0.4;
    col += vec3<f32>(0.80, 0.86, 1.0) * s;
    return col;
}

// ----------------------------------------------------------------- background
struct BgOut { @builtin(position) clip: vec4<f32>, @location(0) ndc: vec2<f32> };
@vertex fn bg_vs(@builtin(vertex_index) vid: u32) -> BgOut {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var o: BgOut;
    o.clip = vec4<f32>(p[vid], 1.0, 1.0);
    o.ndc = p[vid];
    return o;
}
@fragment fn bg_fs(in: BgOut) -> @location(0) vec4<f32> {
    let far = cam.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let world = far.xyz / far.w;
    let dir = normalize(world - cam.cam_pos.xyz);
    return vec4<f32>(galaxy(dir), 1.0);
}

// ----------------------------------------------------------------- atmosphere
struct AOut { @builtin(position) clip: vec4<f32>, @location(0) n: vec3<f32>, @location(1) world: vec3<f32> };
@vertex fn atmo_vs(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>) -> AOut {
    var o: AOut;
    let p = pos * 1.035;
    o.clip = cam.view_proj * vec4<f32>(p, 1.0);
    o.n = nrm;
    o.world = p;
    return o;
}
@fragment fn atmo_fs(in: AOut) -> @location(0) vec4<f32> {
    let n = normalize(in.n);
    let view = normalize(cam.cam_pos.xyz - in.world);
    let sun = normalize(cam.sun_dir.xyz);
    // Thin, pale-blue Google-Earth halo (no purple).
    let fres = pow(1.0 - max(dot(n, view), 0.0), 3.2);
    let col = vec3<f32>(0.40, 0.62, 1.0) * fres * 1.25;
    return vec4<f32>(col, fres);
}

// ----------------------------------------------------------------- points
struct POut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) color: vec4<f32> };
@vertex fn point_vs(@location(0) corner: vec2<f32>, @location(1) ipos: vec3<f32>,
                    @location(2) isize: f32, @location(3) icolor: vec4<f32>) -> POut {
    var o: POut;
    let center = cam.view_proj * vec4<f32>(ipos, 1.0);
    let off = corner * (isize / cam.params.xy) * center.w;
    o.clip = vec4<f32>(center.xy + off, center.z, center.w);
    o.uv = corner;
    o.color = icolor;
    return o;
}
@fragment fn point_fs(in: POut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    if (r > 1.0) { discard; }
    let core = smoothstep(1.0, 0.0, r);
    let glow = pow(core, 2.0);
    return vec4<f32>(in.color.rgb * (0.45 + glow), in.color.a * glow);
}

// ----------------------------------------------------------------- beams
struct BmOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) s: f32,
    @location(1) color: vec4<f32>,
};
@vertex fn beam_vs(@location(0) corner: vec2<f32>, @location(1) a: vec3<f32>,
                   @location(2) b: vec3<f32>, @location(3) color: vec4<f32>) -> BmOut {
    let u = corner.x * 0.5 + 0.5; // 0 at satellite (a), 1 at terminal (b)
    let s = corner.y;             // -1..1 across the ribbon
    let p = mix(a, b, u);
    let viewdir = normalize(cam.cam_pos.xyz - p);
    var perp = cross(normalize(b - a), viewdir);
    perp = perp / max(length(perp), 1e-4);
    let halfw = clamp(length(cam.cam_pos.xyz) * 0.0009, 3.0, 30.0);
    var o: BmOut;
    o.clip = cam.view_proj * vec4<f32>(p + perp * s * halfw, 1.0);
    o.s = s;
    o.color = color;
    return o;
}
@fragment fn beam_fs(in: BmOut) -> @location(0) vec4<f32> {
    let across = exp(-in.s * in.s * 3.2);
    let a = across * in.color.a;
    return vec4<f32>(in.color.rgb * a, a); // additive
}
