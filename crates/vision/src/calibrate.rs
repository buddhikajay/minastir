//! Single-camera calibration from one people-free background image.
//!
//! For each lens (dual-lens panoramas are split into their two halves):
//!   1. MoGe-2 → metric 3D point per pixel + field of view
//!   2. SegFormer → which pixels are ground (road, footpath, grass, earth…)
//!   3. RANSAC plane through the ground points → the camera's height above it, and its tilt
//!      and roll relative to it
//!
//! Heading (yaw) and position on the plan can't be seen from one camera alone; those stay as the
//! user placed them until cameras are aligned with each other.

use crate::{
    moge::{Moge, PointMap},
    segment::{Label, LabelMap, Segmenter},
};
use anyhow::{bail, Result};
pub use minastir_core::calib::{CameraCalibration, LensCalibration};
use serde::{Deserialize, Serialize};

/// A labelled 3D point in a lens's camera frame (OpenCV axes), for building the scene later.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScenePoint {
    pub p: [f32; 3],
    pub label: Label,
}

pub struct CalibrationResult {
    pub calibration: CameraCalibration,
    /// Per lens: (point map, labels) — for overlays and scene building.
    pub lenses: Vec<(PointMap, LabelMap)>,
}

impl CalibrationResult {
    /// Scene maps to save: points plus labels resampled onto the point grid.
    pub fn lens_maps(&self) -> Vec<crate::scenemap::LensMap> {
        self.lenses
            .iter()
            .map(|(pm, lm)| {
                let labels = (0..pm.width * pm.height)
                    .map(|i| {
                        let (x, y) = (i % pm.width, i / pm.width);
                        lm.at((x as f64 + 0.5) / pm.width as f64, (y as f64 + 0.5) / pm.height as f64)
                    })
                    .collect();
                crate::scenemap::LensMap { width: pm.width, height: pm.height, points: pm.points.clone(), labels }
            })
            .collect()
    }
}

/// Dual-lens panoramas (e.g. Reolink Duo, ~2.67:1) are two lenses stitched side by side into
/// one cylindrical image (~180°); we handle each half as its own lens.
pub fn lens_count(aspect: f64) -> usize {
    if aspect >= 2.4 {
        2
    } else {
        1
    }
}

/// Short side (pixels) of the fine labelling pass: ~3× the whole-image pass, enough to put
/// kerb and footpath edges where they really are.
const SEG_FINE: usize = 1280;

pub struct Calibrator {
    pub moge: Moge,
    pub segmenter: Segmenter,
    /// MoGe resolution control (1200–3600; more = finer, slower).
    pub num_tokens: i64,
}

/// One lens's raw model output, before the geometry is fitted.
pub struct LensRaw {
    pub points: PointMap,
    pub labels: LabelMap,
    pub u0: f64,
    pub u1: f64,
    /// Half of a cylindrical panorama (points already on the panorama's pixel grid).
    pub cylindrical: bool,
    /// Barrel distortion measured from the frame's kerbs (points already undistorted and put back
    /// on the frame's pixel grid). 0 = a flat lens.
    pub k1: f64,
}

impl Calibrator {
    /// `rgb`: full background frame, width × height RGB24.
    pub fn calibrate(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<CalibrationResult> {
        let raw = self.analyse(rgb, width, height)?;
        assemble(&raw, &vec![1.0; raw.len()])
    }

    /// Run the models (MoGe + SegFormer) on each lens. Slow; the geometry fit (`assemble`) is cheap
    /// and can be repeated, e.g. for different field-of-view corrections.
    pub fn analyse(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<LensRaw>> {
        let n = lens_count(width as f64 / height as f64);
        if n == 2 {
            return self.analyse_panorama(rgb, width, height);
        }
        let mut out = Vec::new();
        for i in 0..n {
            let (x0, x1) = (width * i / n, width * (i + 1) / n);
            let crop = crop(rgb, width, x0, x1, height);
            let cw = x1 - x0;
            // MoGe at ~900 px wide (it resizes to its token grid internally anyway).
            let mw = 896usize;
            let mh = ((mw as f64 * height as f64 / cw as f64).round() as usize).max(14);
            let labels = self.segmenter.segment_multiscale(&crop, cw, height, SEG_FINE)?;
            // Wide security lenses bend straight kerbs into arcs; depth models assume they don't.
            let k1 = crate::distortion::from_kerbs(&labels).map_or(0.0, |f| f.k1);
            let points = if k1.abs() > 0.02 {
                self.infer_undistorted(&crop, cw, height, k1, mw, mh)?
            } else {
                self.moge.infer(&resize(&crop, cw, height, mw, mh), mw, mh, self.num_tokens)?
            };
            out.push(LensRaw { points, labels, u0: x0 as f64 / width as f64, u1: x1 as f64 / width as f64, cylindrical: false, k1 });
        }
        Ok(out)
    }

    /// A stitched dual-lens panorama is a cylindrical projection: an even angle per pixel across
    /// the whole frame (~180°), so straight kerbs bow into arcs. Depth models expect a flat
    /// (pinhole) image, and read each bowed half as a narrow ~75° lens. So:
    ///   1. measure the cylinder's focal length (pixels per radian) from the vertical field of view,
    ///      which the projection leaves intact;
    ///   2. unwrap each half into a flat image of its true width and run depth on that, pinned
    ///      to the known width;
    ///   3. resample the points back onto the panorama's pixel grid, so everything downstream
    ///      keeps looking up by panorama pixel.
    fn analyse_panorama(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<LensRaw>> {
        let half = width / 2;
        let (mw, mh) = (896usize, ((896.0 * height as f64 / half as f64).round() as usize).max(14));
        let mut crops = Vec::new();
        let mut fs = Vec::new();
        for i in 0..2 {
            let c = crop(rgb, width, half * i, half * (i + 1), height);
            let pm = self.moge.infer(&resize(&c, half, height, mw, mh), mw, mh, self.num_tokens)?;
            fs.push((height as f64 / 2.0) / (pm.vfov.to_radians() / 2.0).tan());
            crops.push(c);
        }
        // Pixels per radian; a sane total width for these cameras is 150–200°.
        let f = ((fs[0] + fs[1]) / 2.0).clamp(width as f64 / 200f64.to_radians(), width as f64 / 150f64.to_radians());
        let span = half as f64 / f; // radians per half
        let tv = (height as f64 / 2.0) / f; // tan(vfov / 2)
        let (cx, cy) = (half as f64 / 2.0, height as f64 / 2.0);

        let mut out = Vec::new();
        for (i, c) in crops.iter().enumerate() {
            // 2. Flat view of this half.
            let pw = 896usize;
            let fp = (pw as f64 / 2.0) / (span / 2.0).tan();
            let ph = ((2.0 * fp * tv).round() as usize).max(14);
            let mut flat = vec![0u8; pw * ph * 3];
            for y in 0..ph {
                for x in 0..pw {
                    let (xp, yp) = (x as f64 + 0.5 - pw as f64 / 2.0, y as f64 + 0.5 - ph as f64 / 2.0);
                    let a = (xp / fp).atan();
                    let tel = yp * a.cos() / fp;
                    let px = bilinear(c, half, height, cx + f * a - 0.5, cy + f * tel - 0.5);
                    flat[(y * pw + x) * 3..(y * pw + x) * 3 + 3].copy_from_slice(&px);
                }
            }
            let pm = self.moge.infer(&flat, pw, ph, self.num_tokens)?;
            let pm = widen(&pm, (span / 2.0).tan() / (pm.hfov.to_radians() / 2.0).tan());
            // 3. Back onto the panorama grid (nearest: never blend across a depth edge).
            let mut points = vec![[f32::NAN; 3]; mw * mh];
            let mut normals = vec![[0f32, -1.0, 0.0]; mw * mh];
            for y in 0..mh {
                for x in 0..mw {
                    let a = ((x as f64 + 0.5) / mw as f64 * half as f64 - cx) / f;
                    let tel = ((y as f64 + 0.5) / mh as f64 * height as f64 - cy) / f;
                    let xp = (fp * a.tan() + pw as f64 / 2.0).floor();
                    let yp = (fp * tel / a.cos() + ph as f64 / 2.0).floor();
                    if xp < 0.0 || yp < 0.0 || xp >= pw as f64 || yp >= ph as f64 {
                        continue;
                    }
                    let k = yp as usize * pw + xp as usize;
                    points[y * mw + x] = pm.points[k];
                    normals[y * mw + x] = pm.normals[k];
                }
            }
            let points = PointMap {
                width: mw,
                height: mh,
                points,
                normals,
                hfov: span.to_degrees(),
                vfov: (2.0 * tv.atan()).to_degrees(),
            };
            let labels = self.segmenter.segment_multiscale(c, half, height, SEG_FINE)?;
            out.push(LensRaw { points, labels, u0: i as f64 / 2.0, u1: (i + 1) as f64 / 2.0, cylindrical: true, k1: 0.0 });
        }
        Ok(out)
    }

    /// Depth for a barrel-distorted frame (`w` × `h`): undistort it into the flat image a pinhole
    /// lens would have seen, run depth on that, and put the points back on an `mw` × `mh` grid
    /// over the original frame, so everything downstream keeps looking up by frame pixel. The
    /// returned field of view is the frame's at its centre's scale (what `LensCalibration`
    /// stores alongside `k1`).
    fn infer_undistorted(&mut self, rgb: &[u8], w: usize, h: usize, k1: f64, mw: usize, mh: usize) -> Result<PointMap> {
        use crate::distortion::{distorted_radius, scale};
        let (cx, cy, half) = (w as f64 / 2.0, h as f64 / 2.0, (w as f64).hypot(h as f64) / 2.0);
        // The flat image covers the whole undistorted frame (its corners reach furthest out);
        // slivers at the edge midpoints that the lens never saw stay black.
        let corner = scale(k1, 1.0);
        let (xmax, ymax) = (cx / half * corner, cy / half * corner);
        let (pw, ph) = (mw, mh);
        let mut flat = vec![0u8; pw * ph * 3];
        for y in 0..ph {
            for x in 0..pw {
                let ux = ((x as f64 + 0.5) / pw as f64 * 2.0 - 1.0) * xmax;
                let uy = ((y as f64 + 0.5) / ph as f64 * 2.0 - 1.0) * ymax;
                let ru = ux.hypot(uy);
                let k = if ru > 1e-12 { distorted_radius(k1, ru) / ru } else { 1.0 };
                let (sx, sy) = (cx + ux * k * half, cy + uy * k * half);
                if sx < 0.0 || sy < 0.0 || sx > w as f64 || sy > h as f64 {
                    continue;
                }
                let px = bilinear(rgb, w, h, sx - 0.5, sy - 0.5);
                flat[(y * pw + x) * 3..(y * pw + x) * 3 + 3].copy_from_slice(&px);
            }
        }
        let pm = self.moge.infer(&flat, pw, ph, self.num_tokens)?;
        // Back onto the frame's grid (nearest: never blend across a depth edge).
        let mut points = vec![[f32::NAN; 3]; mw * mh];
        let mut normals = vec![[0f32, -1.0, 0.0]; mw * mh];
        for y in 0..mh {
            for x in 0..mw {
                let (dx, dy) = (((x as f64 + 0.5) / mw as f64 * w as f64 - cx) / half, ((y as f64 + 0.5) / mh as f64 * h as f64 - cy) / half);
                let s = scale(k1, dx * dx + dy * dy);
                let fx = ((dx * s / xmax + 1.0) / 2.0 * pw as f64).floor();
                let fy = ((dy * s / ymax + 1.0) / 2.0 * ph as f64).floor();
                if fx < 0.0 || fy < 0.0 || fx >= pw as f64 || fy >= ph as f64 {
                    continue;
                }
                let k = fy as usize * pw + fx as usize;
                points[y * mw + x] = pm.points[k];
                normals[y * mw + x] = pm.normals[k];
            }
        }
        // tan(half-angle) per half-diagonal unit of the flat image, from depth's own estimate.
        let per = (pm.hfov.to_radians() / 2.0).tan() / xmax;
        let fov = |extent: f64| (2.0 * (extent / half * per).atan()).to_degrees();
        Ok(PointMap { width: mw, height: mh, points, normals, hfov: fov(cx), vfov: fov(cy) })
    }
}

/// Bilinear RGB sample at pixel coordinates (clamped to the image).
fn bilinear(rgb: &[u8], w: usize, h: usize, x: f64, y: f64) -> [u8; 3] {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (x - x0 as f64, y - y0 as f64);
    let p = |xx: usize, yy: usize, k: usize| rgb[(yy * w + xx) * 3 + k] as f64;
    [0, 1, 2].map(|k| {
        let top = p(x0, y0, k) * (1.0 - tx) + p(x1, y0, k) * tx;
        let bot = p(x0, y1, k) * (1.0 - tx) + p(x1, y1, k) * tx;
        (top * (1.0 - ty) + bot * ty).round() as u8
    })
}

/// Widen (k > 1) or narrow a lens's field of view: tan(fov/2) scales by k, depth stays. Points
/// spread sideways by k, normals tilt accordingly. This is what MoGe itself does when given a
/// known field of view (depth re-projected through the new intrinsics).
pub fn widen(pm: &PointMap, k: f64) -> PointMap {
    if (k - 1.0).abs() < 1e-9 {
        return PointMap { width: pm.width, height: pm.height, points: pm.points.clone(), normals: pm.normals.clone(), hfov: pm.hfov, vfov: pm.vfov };
    }
    let kf = k as f32;
    let points = pm.points.iter().map(|p| [p[0] * kf, p[1] * kf, p[2]]).collect();
    let normals = pm
        .normals
        .iter()
        .map(|n| {
            let m = [n[0] / kf, n[1] / kf, n[2]];
            let l = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt().max(1e-9);
            [m[0] / l, m[1] / l, m[2] / l]
        })
        .collect();
    let f = |deg: f64| (2.0 * ((deg.to_radians() / 2.0).tan() * k).atan()).to_degrees();
    PointMap { width: pm.width, height: pm.height, points, normals, hfov: f(pm.hfov), vfov: f(pm.vfov) }
}

/// Fit the camera's geometry from analysed lenses, with a field-of-view correction per lens.
pub fn assemble(raw: &[LensRaw], fov_scale: &[f64]) -> Result<CalibrationResult> {
    let n = raw.len();
    let mut lenses = Vec::new();
    let mut maps = Vec::new();
    for (i, r) in raw.iter().enumerate() {
        let points = widen(&r.points, fov_scale[i]);
        let mut lens = fit_lens(&points, &r.labels)?;
        lens.u0 = r.u0;
        lens.u1 = r.u1;
        lens.cylindrical = r.cylindrical;
        lens.k1 = r.k1;
        lenses.push(lens);
        maps.push((points, r.labels.clone()));
    }
    // Lenses of a stitched panorama meet at the seam: the left lens looks hfov/2 to the left of
    // the seam (anticlockwise = positive), the right lens hfov/2 to the right.
    if n == 2 {
        lenses[0].yaw_offset = lenses[0].hfov / 2.0;
        lenses[1].yaw_offset = -lenses[1].hfov / 2.0;
    }
    // Body "up": undo each lens's turn within the body, then average (weighted by evidence).
    let mut up = [0.0f64; 3];
    for l in &lenses {
        let b = rot_y(l.up, -l.yaw_offset);
        for k in 0..3 {
            up[k] += b[k] * l.ground_inliers as f64;
        }
    }
    let len = norm(&up).max(1e-12);
    let up = [up[0] / len, up[1] / len, up[2] / len];
    let (pitch, roll) = pitch_roll(up);
    let total: usize = lenses.iter().map(|l| l.ground_inliers).sum();
    let height_m = if total > 0 {
        lenses.iter().map(|l| l.height * l.ground_inliers as f64).sum::<f64>() / total as f64
    } else {
        lenses.iter().map(|l| l.height).sum::<f64>() / lenses.len() as f64
    };
    let computed_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
    Ok(CalibrationResult { calibration: CameraCalibration { lenses, pitch, roll, height: height_m, computed_at, scale: 1.0, alignment: None, fov_correction: fov_scale.iter().sum::<f64>() / fov_scale.len().max(1) as f64 }, lenses: maps })
}

/// Find "up" and the mounting height for one lens.
///
/// Ground alone is unreliable: front lawns and driveways slope, so a plane through them leans.
/// Walls, fences and poles are vertical, so their surface normals are horizontal: "up" is the
/// direction perpendicular to all of them. We minimise
///   Σ_walls (u·n)² + λ·‖u × n_ground‖²
/// (smallest eigenvector), which uses walls where there are any and falls back to the ground.
fn fit_lens(points: &PointMap, labels: &LabelMap) -> Result<LensCalibration> {
    let (w, h) = (points.width, points.height);
    let mut ground: Vec<[f64; 3]> = Vec::new();
    let mut vertical: Vec<[f64; 3]> = Vec::new(); // normals of wall-like surfaces
    for y in (0..h).step_by(2) {
        for x in (0..w).step_by(2) {
            let i = y * w + x;
            let p = points.points[i];
            if p[0].is_nan() {
                continue;
            }
            let d = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            if d > 35.0 {
                continue; // far geometry is noisy
            }
            let label = labels.at((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
            if label.is_ground() {
                ground.push([p[0] as f64, p[1] as f64, p[2] as f64]);
            } else if matches!(label, Label::Building | Label::Fence | Label::Pole) {
                let n = points.normals[i];
                vertical.push([n[0] as f64, n[1] as f64, n[2] as f64]);
            }
        }
    }
    if ground.len() < 200 {
        bail!("not enough visible ground to calibrate ({} points)", ground.len());
    }
    let (normal, d, inliers, rms) = ransac_plane(&ground);
    // Orient the ground normal "up": the camera (origin) is above the ground.
    let n_ground = if d > 0.0 { normal } else { [-normal[0], -normal[1], -normal[2]] };

    // Keep only genuinely vertical surfaces (normal within ~20° of horizontal w.r.t. the ground).
    let walls: Vec<[f64; 3]> = vertical.into_iter().filter(|n| dot(*n, n_ground).abs() < 0.35).collect();
    let mut m = [[0.0f64; 3]; 3];
    if walls.len() > 300 {
        for n in &walls {
            for i in 0..3 {
                for j in 0..3 {
                    m[i][j] += n[i] * n[j] / walls.len() as f64;
                }
            }
        }
    }
    let lambda = if walls.len() > 300 { 0.15 } else { 1.0 };
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] += lambda * ((i == j) as u8 as f64 - n_ground[i] * n_ground[j]);
        }
    }
    let mut up = smallest_eigenvector(m);
    if dot(up, n_ground) < 0.0 {
        up = [-up[0], -up[1], -up[2]];
    }
    // Height: how far below the camera (along "up") the ground is near the camera's foot —
    // median over ground points within ~8 m horizontally.
    let mut below: Vec<f64> = ground
        .iter()
        .filter_map(|p| {
            let vert = -dot(*p, up);
            let horiz = (dot(*p, *p) - vert * vert).max(0.0).sqrt();
            (horiz < 8.0).then_some(vert)
        })
        .collect();
    below.sort_by(f64::total_cmp);
    let height = if below.len() > 50 { below[below.len() / 2] } else { d.abs() };

    let (pitch, roll) = pitch_roll(up);
    Ok(LensCalibration {
        up,
        u0: 0.0,
        u1: 1.0,
        hfov: points.hfov,
        vfov: points.vfov,
        pitch,
        roll,
        yaw_offset: 0.0,
        height,
        ground_inliers: inliers,
        ground_rms: rms,
        wall_points: walls.len(),
        ground_slope: dot(up, n_ground).clamp(-1.0, 1.0).acos().to_degrees(),
        cylindrical: false, k1: 0.0,
    })
}

/// Robust plane fit: RANSAC with a distance-scaled threshold (depth error grows with range),
/// then least-squares refinement on the inliers. Returns (unit normal, d, inliers, rms).
fn ransac_plane(pts: &[[f64; 3]]) -> ([f64; 3], f64, usize, f64) {
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut next = |n: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % n as u64) as usize
    };
    let thresh = |p: &[f64; 3]| 0.05 + 0.02 * norm(p);
    let mut best: ([f64; 3], f64, usize) = ([0.0, -1.0, 0.0], 0.0, 0);
    for _ in 0..600 {
        let (a, b, c) = (pts[next(pts.len())], pts[next(pts.len())], pts[next(pts.len())]);
        let n = cross(sub(b, a), sub(c, a));
        let len = norm(&n);
        if len < 1e-9 {
            continue;
        }
        let n = [n[0] / len, n[1] / len, n[2] / len];
        let d = -dot(n, a);
        let count = pts.iter().filter(|p| (dot(n, **p) + d).abs() < thresh(p)).count();
        if count > best.2 {
            best = (n, d, count);
        }
    }
    let inl: Vec<[f64; 3]> = pts.iter().copied().filter(|p| (dot(best.0, *p) + best.1).abs() < thresh(p)).collect();
    let (n, d) = least_squares_plane(&inl).unwrap_or((best.0, best.1));
    let rms = (inl.iter().map(|p| (dot(n, *p) + d).powi(2)).sum::<f64>() / inl.len().max(1) as f64).sqrt();
    (n, d, inl.len(), rms)
}

/// Plane through the centroid with normal = smallest-eigenvalue eigenvector of the covariance.
fn least_squares_plane(pts: &[[f64; 3]]) -> Option<([f64; 3], f64)> {
    if pts.len() < 3 {
        return None;
    }
    let n = pts.len() as f64;
    let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
    let mut m = [[0.0f64; 3]; 3];
    for p in pts {
        let q = sub(*p, c);
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += q[i] * q[j];
            }
        }
    }
    let normal = smallest_eigenvector(m);
    Some((normal, -dot(normal, c)))
}

/// Jacobi eigen-decomposition of a symmetric 3×3 matrix; returns the eigenvector of the smallest
/// eigenvalue.
fn smallest_eigenvector(mut a: [[f64; 3]; 3]) -> [f64; 3] {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let (mut p, mut q, mut max) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > max {
                    (p, q, max) = (i, j, a[i][j].abs());
                }
            }
        }
        if max < 1e-12 {
            break;
        }
        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
        let t = if theta == 0.0 { 1.0 } else { t };
        let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
        let (app, aqq, apq) = (a[p][p], a[q][q], a[p][q]);
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..3 {
            if r != p && r != q {
                let (arp, arq) = (a[r][p], a[r][q]);
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        for row in v.iter_mut() {
            let (vp, vq) = (row[p], row[q]);
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }
    let k = (0..3).min_by(|&i, &j| a[i][i].total_cmp(&a[j][j])).unwrap();
    let e = [v[0][k], v[1][k], v[2][k]];
    let len = norm(&e);
    [e[0] / len, e[1] / len, e[2] / len]
}

/// Tilt and lean from an "up" vector in an OpenCV camera frame (x right, y down, z forward).
fn pitch_roll(up: [f64; 3]) -> (f64, f64) {
    ((-up[2]).clamp(-1.0, 1.0).asin().to_degrees(), up[0].atan2(-up[1]).to_degrees())
}

/// Rotate about the camera's vertical axis by `deg` (positive turns the view to the left).
fn rot_y(v: [f64; 3], deg: f64) -> [f64; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: &[f64; 3]) -> f64 {
    dot(*a, *a).sqrt()
}

fn crop(rgb: &[u8], width: usize, x0: usize, x1: usize, height: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity((x1 - x0) * height * 3);
    for y in 0..height {
        out.extend_from_slice(&rgb[(y * width + x0) * 3..(y * width + x1) * 3]);
    }
    out
}

/// Box-filtered resize (good enough for model input; avoids an image-crate dependency).
pub fn resize(rgb: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Vec<u8> {
    let mut out = vec![0u8; nw * nh * 3];
    for y in 0..nh {
        let (sy0, sy1) = (y * h / nh, ((y + 1) * h / nh).max(y * h / nh + 1).min(h));
        for x in 0..nw {
            let (sx0, sx1) = (x * w / nw, ((x + 1) * w / nw).max(x * w / nw + 1).min(w));
            let mut acc = [0u32; 3];
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let i = (sy * w + sx) * 3;
                    acc[0] += rgb[i] as u32;
                    acc[1] += rgb[i + 1] as u32;
                    acc[2] += rgb[i + 2] as u32;
                }
            }
            let n = ((sy1 - sy0) * (sx1 - sx0)) as u32;
            let o = (y * nw + x) * 3;
            out[o] = (acc[0] / n) as u8;
            out[o + 1] = (acc[1] / n) as u8;
            out[o + 2] = (acc[2] / n) as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_fit_recovers_tilted_ground() {
        // Camera 3 m above flat ground, tilted 20° down: ground points in the camera frame.
        let (t, hgt) = (20f64.to_radians(), 3.0);
        let mut pts = Vec::new();
        for i in 0..40 {
            for j in 0..40 {
                // World: x right, y up, z forward; ground at y = -hgt.
                let (wx, wz) = (-8.0 + i as f64 * 0.4, 2.0 + j as f64 * 0.5);
                let wy = -hgt;
                // Rotate world into camera (pitch down by t), then flip y to OpenCV (down).
                let cy = wy * t.cos() + wz * t.sin();
                let cz = -wy * t.sin() + wz * t.cos();
                pts.push([wx, -cy, cz]);
            }
        }
        // A few outliers (a wall).
        for k in 0..50 {
            pts.push([2.0, -(k as f64) * 0.05, 6.0]);
        }
        let (n, d, _, _) = ransac_plane(&pts);
        let (n, d) = if d > 0.0 { (n, d) } else { ([-n[0], -n[1], -n[2]], -d) };
        assert!((d - hgt).abs() < 1e-6, "height {d}");
        let pitch = (-n[2]).asin().to_degrees();
        assert!((pitch - 20.0).abs() < 1e-6, "pitch {pitch}");
        assert!(n[0].atan2(-n[1]).to_degrees().abs() < 1e-6, "roll");
    }
}

// ---------- file-level entry point (used by the server) ----------

/// Calibrate from a background image file; writes per-lens scene maps (`scene_lens{i}.bin`) and a
/// visual check (`calibration.jpg`: labels, computed horizon, true verticals) into `out_dir`.
pub fn calibrate_file(cal: &mut Calibrator, background: &std::path::Path, out_dir: &std::path::Path) -> Result<CameraCalibration> {
    let img = image::open(background)?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let r = cal.calibrate(img.as_raw(), w, h)?;
    write_outputs(&img, &r, out_dir)?;
    Ok(r.calibration)
}

/// Save per-lens scene maps (`scene_lens{i}.bin`) and the visual check (`calibration.jpg`).
pub fn write_outputs(img: &image::RgbImage, r: &CalibrationResult, out_dir: &std::path::Path) -> Result<()> {
    let (w, h) = (img.width(), img.height());
    for (i, m) in r.lens_maps().iter().enumerate() {
        m.save(&out_dir.join(format!("scene_lens{i}.bin")))?;
    }
    let ov = render_overlay(img, r);
    image::imageops::resize(&ov, 1600, 1600 * h / w, image::imageops::FilterType::Triangle).save(out_dir.join("calibration.jpg"))?;
    Ok(())
}

pub fn label_color(l: Label) -> [u8; 3] {

    match l {
        Label::Road => [120, 120, 130], Label::Sidewalk => [200, 180, 140], Label::Grass => [90, 200, 90],
        Label::Ground => [170, 130, 80], Label::Building => [220, 80, 80], Label::Tree => [20, 120, 40],
        Label::Plant => [140, 220, 60], Label::Fence => [230, 160, 40], Label::Vehicle => [60, 120, 230],
        Label::Sky => [150, 210, 250], Label::Water => [40, 80, 200], Label::Pole => [250, 250, 0], Label::Bin => [200, 60, 200], Label::Other => [0, 0, 0],
    }
}

/// The background with labels blended in, the computed horizon (yellow) and true verticals (cyan)
/// rising 3 m from ground points. If calibration is right, verticals run parallel to walls/poles.
pub fn render_overlay(img: &image::RgbImage, r: &CalibrationResult) -> image::RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
                // Overlay: image half-blended with label colours, lenses side by side.
                let mut ov = img.clone();
                let n = r.lenses.len();
                for (y, x, px) in ov.enumerate_pixels_mut().map(|(x, y, p)| (y, x, p)) {
                    let li = (x as usize * n / w).min(n - 1);
                    let (x0, x1) = (w * li / n, w * (li + 1) / n);
                    let u = (x as usize - x0) as f64 / (x1 - x0) as f64;
                    let c = label_color(r.lenses[li].1.at(u, y as f64 / h as f64));
                    for k in 0..3 { px[k] = ((px[k] as u16 + c[k] as u16) / 2) as u8; }
                }
                // Calibration check: horizon (yellow) and true verticals (cyan) rising 3 m from
                // ground points. If calibration is right, verticals parallel walls and poles.
                for (li, lens) in r.calibration.lenses.iter().enumerate() {
                    let (pm, lm) = &r.lenses[li];
                    let (x0, x1) = (w * li / n, w * (li + 1) / n);
                    let lw = (x1 - x0) as f64;
                    let (th, tv) = ((lens.hfov.to_radians() / 2.0).tan(), (lens.vfov.to_radians() / 2.0).tan());
                    let u = lens.up;
                    let proj = |p: [f64; 3]| -> Option<(f64, f64)> {
                        if p[2] <= 0.1 { return None; }
                        Some((x0 as f64 + (0.5 + p[0] / p[2] / (2.0 * th)) * lw, (0.5 + p[1] / p[2] / (2.0 * tv)) * h as f64))
                    };
                    let mut line = |a: (f64, f64), b: (f64, f64), c: [u8; 3]| {
                        let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) as usize).max(1);
                        for k in 0..=steps {
                            let t = k as f64 / steps as f64;
                            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                            for dx in -2i64..=2 { for dy in -2i64..=2 {
                                let (px, py) = (x as i64 + dx, y as i64 + dy);
                                if px >= x0 as i64 && px < x1 as i64 && py >= 0 && py < h as i64 { ov.put_pixel(px as u32, py as u32, image::Rgb(c)); }
                            }}
                        }
                    };
                    // Horizon: directions d with d·up = 0.
                    let mut prev = None;
                    for k in 0..=40 {
                        let xn = k as f64 / 40.0;
                        let dx = (xn - 0.5) * 2.0 * th;
                        // u0*dx + u1*dy + u2 = 0  → dy
                        if u[1].abs() < 1e-6 { break; }
                        let dy = -(u[0] * dx + u[2]) / u[1];
                        let pt = (x0 as f64 + xn * lw, (0.5 + dy / (2.0 * tv)) * h as f64);
                        if let Some(pp) = prev { line(pp, pt, [255, 230, 0]); }
                        prev = Some(pt);
                    }
                    for gy in 1..6 { for gx in 1..8 {
                        let (un, vn) = (gx as f64 / 8.0, 0.45 + gy as f64 * 0.09);
                        if !lm.at(un, vn).is_ground() { continue; }
                        let (pxl, pyl) = (((un * pm.width as f64) as usize).min(pm.width - 1), ((vn * pm.height as f64) as usize).min(pm.height - 1));
                        let q = pm.points[pyl * pm.width + pxl];
                        if q[0].is_nan() { continue; }
                        let base = [q[0] as f64, q[1] as f64, q[2] as f64];
                        let top = [base[0] + 3.0 * u[0], base[1] + 3.0 * u[1], base[2] + 3.0 * u[2]];
                        if let (Some(a), Some(b)) = (proj(base), proj(top)) { line(a, b, [0, 240, 255]); }
                    }}
                }
    ov
}
