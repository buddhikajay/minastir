//! Camera calibration results (computed by the vision crate from a background image).
//! Plain data so the inventory can store it without depending on the ML runtime.

use serde::{Deserialize, Serialize};

/// One lens of a camera. Single-lens cameras have exactly one, covering u ∈ [0, 1].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensCalibration {
    /// Horizontal span of the frame this lens covers (normalised 0..1).
    pub u0: f64,
    pub u1: f64,
    /// Field of view, degrees. With barrel distortion (`k1`), measured at the centre's scale:
    /// the edges really see further out than this.
    pub hfov: f64,
    pub vfov: f64,
    /// Downward tilt of the optical axis, degrees.
    pub pitch: f64,
    /// Sideways lean, degrees (positive: image rotated clockwise).
    pub roll: f64,
    /// Heading of this lens relative to the camera's heading (anticlockwise from above), degrees.
    pub yaw_offset: f64,
    /// Height of the lens above the fitted ground plane, metres.
    pub height: f64,
    /// Ground points that fit the plane, and how well (RMS distance, metres).
    pub ground_inliers: usize,
    pub ground_rms: f64,
    /// "Up" (gravity, from walls + ground) in this lens's OpenCV camera frame.
    pub up: [f64; 3],
    /// Vertical-surface samples used, and how much the visible ground slopes relative to "up".
    pub wall_points: usize,
    pub ground_slope: f64,
    /// Part of a stitched cylindrical panorama: azimuth is linear in u (even angle per pixel
    /// across the frame) instead of a flat-image (pinhole) projection.
    #[serde(default)]
    pub cylindrical: bool,
    /// Barrel distortion: a pixel `r` from the centre (as a fraction of the half-diagonal) is
    /// where a flat lens would show it at `r · (1 + k1·r²)`. 0 = a flat lens.
    #[serde(default)]
    pub k1: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub lenses: Vec<LensCalibration>,
    /// Downward tilt and sideways lean of the camera body, degrees. For a dual-lens camera the
    /// lenses are turned ±yaw_offset *within the tilted body*, which is why each lens on its own
    /// appears rolled — combining them recovers one body orientation.
    pub pitch: f64,
    pub roll: f64,
    /// Mounting height (average over lenses), metres — in this camera's own depth scale.
    pub height: f64,
    /// Correction to this camera's depth scale found by aligning it with other cameras
    /// (single-image depth is only accurate to ~5–10%). Multiply its 3D points by this.
    #[serde(default = "one")]
    pub scale: f64,
    /// Set when heading/position/scale were solved from points matched with other cameras.
    #[serde(default)]
    pub alignment: Option<Alignment>,
    /// How much the lens's field of view was widened (tan of the half-angle multiplied by this)
    /// after comparing it with overlapping cameras. Single-image depth reads wide, distorted
    /// security lenses as too narrow.
    #[serde(default = "one")]
    pub fov_correction: f64,
    /// Unix seconds.
    pub computed_at: f64,
}


fn one() -> f64 {
    1.0
}

/// How a camera was placed relative to the others.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alignment {
    /// Camera everything else is placed relative to (kept where the user put it).
    pub anchor: bool,
    /// Camera this one was solved against, and how many matched points agreed.
    pub reference: Option<String>,
    pub inliers: usize,
    /// RMS disagreement of those points after solving, metres.
    pub rms: f64,
    /// The user's placement before alignment: (yaw, x, z).
    pub rough: [f64; 3],
}
