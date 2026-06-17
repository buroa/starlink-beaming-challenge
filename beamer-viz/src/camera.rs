//! Orbit camera around the earth's center. Positions are in kilometres
//! (earth radius 6371, satellites at 6921).

use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        // Framed on the United States (lon ≈ -98°, lat ≈ 39°): the eye sits over
        // that point so it faces the viewer.
        OrbitCamera {
            yaw: -98f32.to_radians(),
            pitch: 39f32.to_radians(),
            distance: 16500.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        self.distance * Vec3::new(cp * cy, cp * sy, sp)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), Vec3::ZERO, Vec3::Z);
        // A small near plane lets the camera skim the surface and continue all
        // the way down to the core for the inside-out view.
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect.max(0.01), 12.0, 200_000.0);
        proj * view
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-1.45, 1.45);
    }

    /// Scroll to zoom. The clamp reaches from far out (90,000 km) down to near
    /// the planet's core (40 km), passing smoothly through the surface; with a
    /// transparent earth the whole beam network is then visible from within.
    /// Multiplicative steps slow the approach automatically as it gets close.
    /// Sensitivity factor 0.0015 can be tuned to adjust scroll responsiveness.
    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.0015)).clamp(40.0, 90_000.0);
    }

    /// Pinch-zoom by a multiplicative factor (egui's two-finger `zoom_delta`):
    /// `> 1` zooms in (fingers apart), `< 1` zooms out, sharing [`zoom`]'s clamp.
    pub fn zoom_by(&mut self, factor: f32) {
        if factor > 0.0 {
            self.distance = (self.distance / factor).clamp(40.0, 90_000.0);
        }
    }
}
