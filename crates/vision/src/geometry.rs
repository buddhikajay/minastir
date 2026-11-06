//! Image point → ground point, using a camera's hand-placed (later: calibrated) pose.
//!
//! World frame matches the UI: Y up, a camera with yaw θ looks along (sin θ, 0, cos θ), pitch is
//! downward tilt. Camera-local frame: +Z forward, +Y up — so image-right is local −X.

use minastir_core::onvif::Pose;

/// Horizontal field of view (degrees) guessed from the image shape. Kept identical to the UI's
/// `hfovOf` so what you see on the plan is what the projection uses.
pub fn hfov_deg(aspect: f64) -> f64 {
    if aspect >= 2.4 {
        180.0
    } else {
        (2.0 * (aspect * 27f64.to_radians().tan()).atan()).to_degrees().min(120.0)
    }
}

/// Stitched dual-lens panoramas are closer to a cylindrical projection than a pinhole.
pub fn is_panoramic(aspect: f64) -> bool {
    aspect >= 2.4
}

/// Unit-ish ray direction in world space for normalised image point (u, v) ∈ [0,1]².
pub fn ray(pose: &Pose, aspect: f64, u: f64, v: f64) -> [f64; 3] {
    let hfov = hfov_deg(aspect).to_radians();
    let px = (u - 0.5) * aspect; // image units: height = 1
    let py = v - 0.5;
    let (x, y, z) = if is_panoramic(aspect) {
        let f = aspect / hfov; // image units per radian of azimuth
        let az = px / f;
        (-az.sin(), -py / f, az.cos())
    } else {
        let f = (aspect / 2.0) / (hfov / 2.0).tan();
        (-px / f, -py / f, 1.0)
    };
    // Tilt down (rotation about X), then heading (rotation about Y).
    let (sp, cp) = pose.pitch.to_radians().sin_cos();
    let (y, z) = (y * cp - z * sp, y * sp + z * cp);
    let (sy, cy) = pose.yaw.to_radians().sin_cos();
    [x * cy + z * sy, y, -x * sy + z * cy]
}

/// Where the ray through (u, v) meets the ground (y = 0), if it does within `max_range` metres.
pub fn ground_point(pose: &Pose, aspect: f64, u: f64, v: f64, max_range: f64) -> Option<(f64, f64)> {
    let d = ray(pose, aspect, u, v);
    if d[1] > -1e-3 {
        return None; // at or above the horizon
    }
    let t = pose.height / -d[1];
    let (dx, dz) = (d[0] * t, d[2] * t);
    if dx.hypot(dz) > max_range {
        return None;
    }
    Some((pose.x + dx, pose.z + dz))
}

/// A camera as calibrated: per-lens field of view and heading within the body, body tilt and
/// lean, plus (optionally) the background's per-pixel 3D map for exact ground lookups.
pub struct CameraModel {
    pub pose: Pose,
    lenses: Vec<Lens>,
    /// Rotation from the body's local frame (x left, y up, z forward) to the world.
    rot: [[f64; 3]; 3],
    maps: Option<std::sync::Arc<Vec<crate::scenemap::LensMap>>>,
    /// Depth-scale correction from cross-camera alignment.
    scale: f64,
}

struct Lens {
    u0: f64,
    u1: f64,
    tan_h: f64,
    tan_v: f64,
    hfov: f64,
    cylindrical: bool,
    yaw_offset: f64,
    /// Barrel distortion (see `LensCalibration::k1`).
    k1: f64,
}

impl Lens {
    /// Ray through normalised lens point (lu, v), in the lens's OpenCV frame (unnormalised).
    fn dir(&self, lu: f64, v: f64) -> [f64; 3] {
        let y = (v - 0.5) * 2.0 * self.tan_v;
        if self.cylindrical {
            let (s, c) = ((lu - 0.5) * self.hfov).sin_cos();
            [s, y, c]
        } else {
            let x = (lu - 0.5) * 2.0 * self.tan_h;
            // Distance from the centre as a share of the half-diagonal.
            let s = crate::distortion::scale(self.k1, (x * x + y * y) / (self.tan_h * self.tan_h + self.tan_v * self.tan_v));
            [x * s, y * s, 1.0]
        }
    }

    /// Normalised lens point (lu, v) of a direction in the lens's OpenCV frame (flat lenses), the
    /// inverse of `dir`; None behind the lens.
    fn flat_point(&self, v: [f64; 3]) -> Option<(f64, f64)> {
        if v[2] <= 1e-6 {
            return None;
        }
        let (mut x, mut y) = (v[0] / v[2], v[1] / v[2]);
        if self.k1 != 0.0 {
            let diag = self.tan_h.hypot(self.tan_v);
            let ru = x.hypot(y) / diag;
            if ru > 1e-12 {
                let k = crate::distortion::distorted_radius(self.k1, ru) / ru;
                (x, y) = (x * k, y * k);
            }
        }
        Some((0.5 + x / (2.0 * self.tan_h), 0.5 + y / (2.0 * self.tan_v)))
    }
}

impl CameraModel {
    /// `pose.pitch` / `pose.height` are taken as current (calibration writes its measurements
    /// there, and the user may still adjust them); roll and lenses come from the calibration.
    pub fn new(
        pose: Pose,
        calib: &minastir_core::calib::CameraCalibration,
        maps: Option<std::sync::Arc<Vec<crate::scenemap::LensMap>>>,
    ) -> Self {
        let lenses = calib
            .lenses
            .iter()
            .map(|l| Lens {
                u0: l.u0,
                u1: l.u1,
                tan_h: (l.hfov.to_radians() / 2.0).tan(),
                tan_v: (l.vfov.to_radians() / 2.0).tan(),
                hfov: l.hfov.to_radians(),
                cylindrical: l.cylindrical,
                yaw_offset: l.yaw_offset,
                k1: if l.cylindrical { 0.0 } else { l.k1 },
            })
            .collect();
        let (p, r) = (pose.pitch.to_radians(), calib.roll.to_radians());
        // "Up" in the body's OpenCV frame, then in its local frame (x left, y up, z forward).
        let up_cv = [p.cos() * r.sin(), -p.cos() * r.cos(), -p.sin()];
        let up = [-up_cv[0], -up_cv[1], up_cv[2]];
        let level = rotation_between(up, [0.0, 1.0, 0.0]);
        // Turn so the body's forward direction points along the user's heading.
        let f = mat_vec(&level, [0.0, 0.0, 1.0]);
        let h0 = f[0].atan2(f[2]);
        let yaw = pose.yaw.to_radians() - h0;
        let (sy, cy) = yaw.sin_cos();
        let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        Self { pose, lenses, rot: mat_mul(&ry, &level), maps, scale: calib.scale }
    }

    fn lens_for(&self, u: f64) -> Option<(usize, f64)> {
        let i = self.lenses.iter().position(|l| u >= l.u0 && u <= l.u1)?;
        let l = &self.lenses[i];
        Some((i, (u - l.u0) / (l.u1 - l.u0)))
    }

    /// A vector in lens `i`'s OpenCV frame → world (relative to the camera).
    fn to_world(&self, i: usize, v: [f64; 3]) -> [f64; 3] {
        let (s, c) = (-self.lenses[i].yaw_offset).to_radians().sin_cos();
        let b = [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]; // lens → body (OpenCV)
        mat_vec(&self.rot, [-b[0], -b[1], b[2]])
    }

    /// Ground position (world x, z) under normalised image point (u, v).
    pub fn ground_point(&self, u: f64, v: f64, max_range: f64) -> Option<(f64, f64)> {
        self.ground_point3(u, v, max_range).map(|p| (p[0], p[2]))
    }

    /// Like `ground_point`, with the ground's height too: world (x, height above this camera's
    /// ground plane, z). Slopes, kerbs and driveways put it above or below zero.
    pub fn ground_point3(&self, u: f64, v: f64, max_range: f64) -> Option<[f64; 3]> {
        let (i, lu) = self.lens_for(u)?;
        let at = |w: [f64; 3]| [self.pose.x + w[0], self.pose.height + w[1], self.pose.z + w[2]];
        if let Some(map) = self.maps.as_ref().and_then(|m| m.get(i)) {
            let k = self.scale;
            if let Some(p) = map.ground_at(lu, v) {
                let w = self.to_world(i, [p[0] as f64 * k, p[1] as f64 * k, p[2] as f64 * k]);
                if w[0].hypot(w[2]) <= max_range {
                    return Some(at(w));
                }
                return None;
            }
            // Something stood on the ground here in the background (a letterbox, a toy, a car):
            // follow this pixel's true line of sight (from the map, distortion included) down to
            // the height of the nearest ground the map does show. Falling back to the flat lens
            // model instead would jump many metres on a lens it doesn't describe well.
            if let (Some(p), Some(g)) = (map.ray_at(lu, v), map.ground_near(lu, v, 40)) {
                let d = self.to_world(i, [p[0] as f64, p[1] as f64, p[2] as f64]);
                let gw = self.to_world(i, [g[0] as f64 * k, g[1] as f64 * k, g[2] as f64 * k]);
                if d[1] < -1e-3 && gw[1] < 0.0 {
                    let t = gw[1] / d[1];
                    let (dx, dz) = (d[0] * t, d[2] * t);
                    return (dx.hypot(dz) <= max_range).then(|| at([dx, gw[1], dz]));
                }
            }
        }
        let l = &self.lenses[i];
        let d = self.to_world(i, l.dir(lu, v));
        if d[1] > -1e-3 {
            return None;
        }
        let t = self.pose.height / -d[1];
        let (dx, dz) = (d[0] * t, d[2] * t);
        (dx.hypot(dz) <= max_range).then(|| at([dx, -self.pose.height, dz]))
    }

    /// The world-frame direction of the line of sight through image point (u, v): from the
    /// scene map where it shows something there (lens distortion included), else the lens model.
    pub fn sight(&self, u: f64, v: f64) -> Option<[f64; 3]> {
        let (i, lu) = self.lens_for(u)?;
        if let Some(p) = self.maps.as_ref().and_then(|m| m.get(i)).and_then(|m| m.ray_at(lu, v)) {
            return Some(self.to_world(i, [p[0] as f64, p[1] as f64, p[2] as f64]));
        }
        Some(self.to_world(i, self.lenses[i].dir(lu, v)))
    }
}

impl CameraModel {
    /// Where a world point (x, z, height above this camera's ground) appears in the image, as
    /// normalised (u, v) — the inverse of the ray lookups. None if it's behind the camera or
    /// outside every lens.
    pub fn project(&self, x: f64, z: f64, h: f64) -> Option<(f64, f64)> {
        self.project_lens(x, z, h, false)
    }

    /// Like `project`, but points a little outside the frame still get coordinates (beyond 0..1),
    /// from the nearest lens — for drawing objects cut off by the image edge.
    pub fn project_beyond(&self, x: f64, z: f64, h: f64) -> Option<(f64, f64)> {
        self.project_lens(x, z, h, true)
    }

    fn project_lens(&self, x: f64, z: f64, h: f64, beyond: bool) -> Option<(f64, f64)> {
        let w = [x - self.pose.x, h - self.pose.height, z - self.pose.z];
        // World → body local (x left, y up, z forward) → body OpenCV.
        let r = &self.rot;
        let l = [r[0][0] * w[0] + r[1][0] * w[1] + r[2][0] * w[2], r[0][1] * w[0] + r[1][1] * w[1] + r[2][1] * w[2], r[0][2] * w[0] + r[1][2] * w[1] + r[2][2] * w[2]];
        let b = [-l[0], -l[1], l[2]];
        let mut nearest: Option<(f64, (f64, f64))> = None;
        for lens in &self.lenses {
            let (s, c) = (-lens.yaw_offset).to_radians().sin_cos();
            let v = [b[0] * c - b[2] * s, b[1], b[0] * s + b[2] * c]; // body → lens
            let (lu, lv) = if lens.cylindrical {
                let flat = v[0].hypot(v[2]);
                if flat < 1e-9 {
                    continue;
                }
                (0.5 + v[0].atan2(v[2]) / lens.hfov, 0.5 + v[1] / flat / (2.0 * lens.tan_v))
            } else {
                let Some(p) = lens.flat_point(v) else { continue };
                p
            };
            if (0.0..=1.0).contains(&lu) && (0.0..=1.0).contains(&lv) {
                return Some((lens.u0 + lu * (lens.u1 - lens.u0), lv));
            }
            if beyond {
                let out = (-lu).max(lu - 1.0).max(0.0) + (-lv).max(lv - 1.0).max(0.0);
                if out < 0.5 && nearest.is_none_or(|(o, _)| out < o) {
                    nearest = Some((out, (lens.u0 + lu * (lens.u1 - lens.u0), lv)));
                }
            }
        }
        nearest.map(|(_, p)| p)
    }

    /// 3D point (world x, z, height above this camera's ground) at normalised image (u, v),
    /// from the scene map — for matched keypoints.
    pub fn point_at(&self, u: f64, v: f64) -> Option<[f64; 3]> {
        let (i, lu) = self.lens_for(u)?;
        let p = self.maps.as_ref()?.get(i)?.point_at(lu, v)?;
        let k = self.scale;
        let w = self.to_world(i, [p[0] as f64 * k, p[1] as f64 * k, p[2] as f64 * k]);
        Some([self.pose.x + w[0], self.pose.z + w[2], self.pose.height + w[1]])
    }

    /// Where the ray through (u, v) is `h` metres above the ground (e.g. a car's roofline).
    pub fn point_at_height(&self, u: f64, v: f64, h: f64, max_range: f64) -> Option<(f64, f64)> {
        let (i, lu) = self.lens_for(u)?;
        let l = &self.lenses[i];
        let d = self.to_world(i, l.dir(lu, v));
        let drop = self.pose.height - h;
        if drop < 0.1 || d[1] > -1e-3 {
            return None;
        }
        let t = drop / -d[1];
        let (dx, dz) = (d[0] * t, d[2] * t);
        (dx.hypot(dz) <= max_range).then(|| (self.pose.x + dx, self.pose.z + dz))
    }

    /// What the background showed inside a detection box (normalised x0, y0, x1, y1): fractions of
    /// (vehicle, fixed objects — walls, fences, bins, plants, poles…, bins alone). None without a
    /// scene map.
    pub fn box_labels(&self, b: [f64; 4]) -> Option<(f64, f64, f64)> {
        use crate::segment::Label;
        let maps = self.maps.as_ref()?;
        let (mut veh, mut fixed, mut bin, mut n) = (0usize, 0usize, 0usize, 0usize);
        for iy in 0..12 {
            for ix in 0..12 {
                let u = b[0] + (b[2] - b[0]) * (ix as f64 + 0.5) / 12.0;
                let v = b[1] + (b[3] - b[1]) * (iy as f64 + 0.5) / 12.0;
                let Some((i, lu)) = self.lens_for(u) else { continue };
                let Some(m) = maps.get(i) else { continue };
                let (x, y) = (((lu * m.width as f64) as usize).min(m.width - 1), ((v * m.height as f64) as usize).min(m.height - 1));
                n += 1;
                let l = m.labels[y * m.width + x];
                if l == Label::Bin {
                    bin += 1;
                }
                match l {
                    Label::Vehicle => veh += 1,
                    Label::Other | Label::Building | Label::Fence | Label::Plant | Label::Pole | Label::Tree | Label::Bin => fixed += 1,
                    _ => {}
                }
            }
        }
        (n > 0).then(|| (veh as f64 / n as f64, fixed as f64 / n as f64, bin as f64 / n as f64))
    }

    /// The static scene this camera sees, from its calibration map: (world x, world z, height
    /// above the camera's ground, label) for every `step`-th pixel within `max_range` metres.
    pub fn scene_points(&self, step: usize, max_range: f64) -> Vec<(f64, f64, f64, crate::segment::Label)> {
        use crate::segment::Label;
        self.labelled_points(step, max_range, |l| !matches!(l, Label::Sky | Label::Other | Label::Vehicle))
    }

    /// Like `scene_points`, keeping only labels `keep` accepts.
    pub fn labelled_points(
        &self,
        step: usize,
        max_range: f64,
        keep: impl Fn(crate::segment::Label) -> bool,
    ) -> Vec<(f64, f64, f64, crate::segment::Label)> {
        let Some(maps) = &self.maps else { return Vec::new() };
        let mut out = Vec::new();
        for (i, m) in maps.iter().enumerate() {
            for y in (0..m.height).step_by(step) {
                for x in (0..m.width).step_by(step) {
                    let k = y * m.width + x;
                    let p = m.points[k];
                    let label = m.labels[k];
                    if p[0].is_nan() || !keep(label) || m.on_depth_edge(x, y) {
                        continue;
                    }
                    let k = self.scale;
                    let w = self.to_world(i, [p[0] as f64 * k, p[1] as f64 * k, p[2] as f64 * k]);
                    if w[0].hypot(w[2]) > max_range {
                        continue;
                    }
                    out.push((self.pose.x + w[0], self.pose.z + w[2], self.pose.height + w[1], label));
                }
            }
        }
        out
    }
}

fn mat_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2])
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            o[r][c] = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    o
}

/// Smallest rotation taking unit vector `a` onto unit vector `b` (Rodrigues).
fn rotation_between(a: [f64; 3], b: [f64; 3]) -> [[f64; 3]; 3] {
    let v = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let c = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let i = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if s2 < 1e-18 {
        return i;
    }
    let k = [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]];
    let k2 = mat_mul(&k, &k);
    let f = (1.0 - c) / s2;
    let mut r = i;
    for x in 0..3 {
        for y in 0..3 {
            r[x][y] += k[x][y] + k2[x][y] * f;
        }
    }
    r
}

#[cfg(test)]
mod tests {
    #[test]
    fn calibrated_model_matches_simple_model_without_roll() {
        use minastir_core::calib::{CameraCalibration, LensCalibration};
        let lens = LensCalibration {
            u0: 0.0, u1: 1.0, hfov: 80.0, vfov: 50.0, pitch: 30.0, roll: 0.0, yaw_offset: 0.0, height: 3.0,
            ground_inliers: 0, ground_rms: 0.0, up: [0.0, -1.0, 0.0], wall_points: 0, ground_slope: 0.0, cylindrical: false, k1: 0.0,
        };
        let calib = CameraCalibration { lenses: vec![lens], pitch: 30.0, roll: 0.0, height: 3.0, computed_at: 0.0, scale: 1.0, alignment: None, fov_correction: 1.0 };
        for yaw in [0.0, 90.0, 225.0] {
            let pose = Pose { x: 1.0, z: -2.0, height: 3.0, yaw, pitch: 30.0 };
            let m = CameraModel::new(pose, &calib, None);
            // Centre pixel: 3 m high, 30° down → 3/tan30 ahead along the heading.
            let (x, z) = m.ground_point(0.5, 0.5, 50.0).unwrap();
            let d = 3.0 / 30f64.to_radians().tan();
            let (ex, ez) = (1.0 + d * yaw.to_radians().sin(), -2.0 + d * yaw.to_radians().cos());
            assert!((x - ex).abs() < 1e-6 && (z - ez).abs() < 1e-6, "yaw {yaw}: {x},{z} vs {ex},{ez}");
            // Image-right is the camera's right (clockwise from the heading, seen from above).
            let (rx, rz) = m.ground_point(0.8, 0.5, 50.0).unwrap();
            let right = [-(yaw.to_radians().cos()), yaw.to_radians().sin()]; // heading rotated −90°
            assert!((rx - x) * right[0] + (rz - z) * right[1] > 0.0, "yaw {yaw}: right side");
        }
    }

    #[test]
    fn projection_inverts_the_ground_lookup() {
        use minastir_core::calib::{CameraCalibration, LensCalibration};
        let lens = |u0: f64, u1: f64, yaw_offset: f64, cylindrical: bool, k1: f64| LensCalibration {
            u0, u1, hfov: 95.0, vfov: 60.0, pitch: 25.0, roll: 3.0, yaw_offset, height: 3.0,
            ground_inliers: 0, ground_rms: 0.0, up: [0.0, -1.0, 0.0], wall_points: 0, ground_slope: 0.0, cylindrical, k1,
        };
        for lenses in [
            vec![lens(0.0, 1.0, 0.0, false, 0.0)],
            vec![lens(0.0, 1.0, 0.0, false, 0.4)],
            vec![lens(0.0, 0.5, 45.0, true, 0.0), lens(0.5, 1.0, -45.0, true, 0.0)],
        ] {
            let calib = CameraCalibration { lenses, pitch: 25.0, roll: 3.0, height: 3.0, computed_at: 0.0, scale: 1.0, alignment: None, fov_correction: 1.0 };
            let m = CameraModel::new(Pose { x: 2.0, z: -1.0, height: 3.0, yaw: 130.0, pitch: 25.0 }, &calib, None);
            let mut checked = 0;
            for (u, v) in [(0.1, 0.7), (0.45, 0.6), (0.55, 0.9), (0.9, 0.65)] {
                let Some((x, z)) = m.ground_point(u, v, 80.0) else { continue };
                checked += 1;
                let (pu, pv) = m.project(x, z, 0.0).expect("visible");
                assert!((pu - u).abs() < 1e-6 && (pv - v).abs() < 1e-6, "({u},{v}) → ({pu},{pv})");
                // And a point above the ground along the same pixel's height ray.
                if let Some((hx, hz)) = m.point_at_height(u, v, 1.5, 80.0) {
                    let (qu, qv) = m.project(hx, hz, 1.5).unwrap();
                    assert!((qu - u).abs() < 1e-6 && (qv - v).abs() < 1e-6);
                }
            }
            assert!(checked >= 3, "only {checked} points reached the ground");
        }
    }

    use super::*;

    fn pose(yaw: f64, pitch: f64) -> Pose {
        Pose { x: 0.0, z: 0.0, height: 3.0, yaw, pitch }
    }

    #[test]
    fn centre_pixel_lands_straight_ahead() {
        // 45° down from 3 m → 3 m ahead along +Z.
        let (x, z) = ground_point(&pose(0.0, 45.0), 16.0 / 9.0, 0.5, 0.5, 50.0).unwrap();
        assert!(x.abs() < 1e-9 && (z - 3.0).abs() < 1e-9, "{x} {z}");
        // Facing +X (yaw 90).
        let (x, z) = ground_point(&pose(90.0, 45.0), 16.0 / 9.0, 0.5, 0.5, 50.0).unwrap();
        assert!((x - 3.0).abs() < 1e-9 && z.abs() < 1e-9, "{x} {z}");
    }

    #[test]
    fn image_right_is_camera_right() {
        // Facing +Z, the camera's right-hand side is −X.
        let (x, _) = ground_point(&pose(0.0, 45.0), 16.0 / 9.0, 0.8, 0.5, 50.0).unwrap();
        assert!(x < 0.0);
        let (x, _) = ground_point(&pose(0.0, 45.0), 3.0, 0.8, 0.5, 50.0).unwrap();
        assert!(x < 0.0, "panoramic too");
    }

    #[test]
    fn lower_in_image_is_closer() {
        let near = ground_point(&pose(0.0, 30.0), 16.0 / 9.0, 0.5, 0.9, 50.0).unwrap().1;
        let far = ground_point(&pose(0.0, 30.0), 16.0 / 9.0, 0.5, 0.6, 50.0).unwrap().1;
        assert!(near < far);
    }

    #[test]
    fn sky_has_no_ground_point() {
        assert!(ground_point(&pose(0.0, 10.0), 16.0 / 9.0, 0.5, 0.1, 50.0).is_none());
    }
}
