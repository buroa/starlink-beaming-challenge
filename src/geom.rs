//! 3D vector math and the three constraint tests.
//!
//! All angle constraints are evaluated as comparisons against the *cosine* of
//! the threshold (monotone-decreasing in angle), so we never call `acos` and
//! never risk `NaN`. A tiny safety epsilon is applied on the conservative side
//! of every boundary so a beam can never sit microscopically on the wrong side
//! of the validator's strict comparison (a near-boundary false-accept is a hard
//! validation failure; a false-reject merely costs at most a marginal user).

/// Vector / point in earth-centered, earth-fixed coordinates (kilometres).
#[cfg_attr(feature = "viz", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    #[inline]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline]
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    /// Unit vector. Caller guarantees a non-zero magnitude (positions are never
    /// at the origin in any valid scenario).
    #[inline]
    pub fn unit(self) -> Vec3 {
        let n = self.norm();
        Vec3::new(self.x / n, self.y / n, self.z / n)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

// cos of the constraint thresholds (exact f64 of math.cos(radians(deg))).
pub const COS45: f64 = std::f64::consts::FRAC_1_SQRT_2; // cos 45° = 1/√2
pub const COS20: f64 = 0.939_692_620_785_908_4; //          cos 20°
pub const COS10: f64 = 0.984_807_753_012_208; //            cos 10°

/// Conservative margin in cosine-space. ~1e-9 in cosine is < 1e-6 degrees near
/// these thresholds — far below any meaningful coverage loss, far above f64
/// rounding noise.
const EPS: f64 = 1e-9;

/// Visibility: the satellite must be strictly within 45° of the user's local
/// vertical. `zenith` is `unit(user_pos)` (earth centre is the origin);
/// `dir_su` is `unit(sat_pos - user_pos)`. Validator fails when the angle from
/// zenith is >= 45°, i.e. when `dot <= cos45`; we additionally require margin.
#[inline]
pub fn visible(zenith: Vec3, dir_su: Vec3) -> bool {
    zenith.dot(dir_su) > COS45 + EPS
}

/// Interferer separation as seen from the user: the angle between the serving
/// satellite and the interferer must be >= 20°. Returns `true` when this pair
/// is too close (i.e. the (user,sat) assignment would be invalid).
/// `dir_su = unit(sat - user)`, `dir_iu = unit(interferer - user)`.
#[inline]
pub fn interferes(dir_su: Vec3, dir_iu: Vec3) -> bool {
    dir_su.dot(dir_iu) > COS20 - EPS
}

/// Same-color conflict: two users served by one satellite on the same color
/// must be >= 10° apart as seen from the satellite. `dir_a`/`dir_b` are the
/// unit directions from the satellite to each user. Returns `true` when they
/// are too close to share a color.
#[inline]
pub fn same_color_conflict(dir_a: Vec3, dir_b: Vec3) -> bool {
    dir_a.dot(dir_b) > COS10 - EPS
}
