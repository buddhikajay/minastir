//! MoGe-2 (Microsoft, MIT): monocular geometry. From one image it predicts a 3D point per pixel,
//! a validity mask and a metric scale; we recover the focal length (field of view) and the depth
//! shift the same way the reference implementation does, then scale to metres.
//!
//! Output points are in the OpenCV camera frame: x right, y down, z forward.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;

pub struct Moge {
    session: Session,
}

/// A metric point map for one image.
pub struct PointMap {
    pub width: usize,
    pub height: usize,
    /// width × height points (metres, camera frame); NaN where the model has no estimate (sky…).
    pub points: Vec<[f32; 3]>,
    /// Unit surface normals per pixel (camera frame).
    pub normals: Vec<[f32; 3]>,
    /// Horizontal / vertical field of view, degrees.
    pub hfov: f64,
    pub vfov: f64,
}

impl Moge {
    pub fn load(model: &Path, device: Device) -> Result<Self> {
        Ok(Self { session: onnx::session(model, device, &[])? })
    }

    /// `rgb`: width × height RGB24. `num_tokens` trades detail for speed (MoGe uses 1200–3600).
    pub fn infer(&mut self, rgb: &[u8], width: usize, height: usize, num_tokens: i64) -> Result<PointMap> {
        let plane = width * height;
        let mut input = vec![0f32; 3 * plane];
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                input[c * plane + i] = px[c] as f32 / 255.0; // MoGe normalises internally
            }
        }
        let image = Tensor::from_array(([1usize, 3, height, width], input)).map_err(|e| anyhow!("{e}"))?;
        let tokens = Tensor::from_array((Vec::<usize>::new(), vec![num_tokens])).map_err(|e| anyhow!("{e}"))?;
        let out = self
            .session
            .run(ort::inputs!["image" => image, "num_tokens" => tokens])
            .map_err(|e| anyhow!("{e}"))?;
        let (pshape, points) = out["points"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (_, normal) = out["normal"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let normals: Vec<[f32; 3]> = normal.chunks_exact(3).map(|n| [n[0], n[1], n[2]]).collect();
        let (_, mask) = out["mask"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (_, scale) = out["metric_scale"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (h, w) = (pshape[1] as usize, pshape[2] as usize);
        anyhow::ensure!(h == height && w == width, "unexpected point map size {w}x{h}");
        let raw: Vec<[f32; 3]> = points.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let valid: Vec<bool> = mask.iter().map(|&m| m > 0.5).collect();

        let aspect = width as f64 / height as f64;
        let (focal, shift) = recover_focal_shift(&raw, &valid, width, height, aspect);
        let scale = scale[0];
        let points = raw
            .iter()
            .zip(&valid)
            .map(|(p, &ok)| {
                if ok {
                    [p[0] * scale, p[1] * scale, (p[2] + shift as f32) * scale]
                } else {
                    [f32::NAN; 3]
                }
            })
            .collect();
        // `focal` is relative to the image's half-diagonal; convert to per-axis focal lengths
        // normalised by width / height, exactly as MoGe does.
        let diag = (1.0 + aspect * aspect).sqrt();
        let fx = focal / 2.0 * diag / aspect;
        let fy = focal / 2.0 * diag;
        Ok(PointMap {
            width,
            height,
            points,
            normals,
            hfov: (2.0 * (0.5 / fx).atan()).to_degrees(),
            vfov: (2.0 * (0.5 / fy).atan()).to_degrees(),
        })
    }
}

/// Find focal f and depth shift s so that the predicted points, perspective-projected, land on
/// their own pixels: minimise Σ‖f·(x, y)/(z + s) − uv‖². For a fixed s the best f is closed-form,
/// so this is a 1-D search over s (coarse scan, then golden-section refinement). uv uses MoGe's
/// "view plane" coordinates, where the half-diagonal is 1.
fn recover_focal_shift(points: &[[f32; 3]], valid: &[bool], w: usize, h: usize, aspect: f64) -> (f64, f64) {
    let span_x = aspect / (1.0 + aspect * aspect).sqrt();
    let span_y = 1.0 / (1.0 + aspect * aspect).sqrt();
    // Subsample to ~64×64 like the reference.
    let step = (w.max(h) / 64).max(1);
    let mut samples: Vec<(f64, f64, f64, f64, f64)> = Vec::new(); // x, y, z, u, v
    for py in (0..h).step_by(step) {
        for px in (0..w).step_by(step) {
            let i = py * w + px;
            if !valid[i] {
                continue;
            }
            let p = points[i];
            let u = span_x * (2.0 * (px as f64 + 0.5) / w as f64 - 1.0);
            let v = span_y * (2.0 * (py as f64 + 0.5) / h as f64 - 1.0);
            samples.push((p[0] as f64, p[1] as f64, p[2] as f64, u, v));
        }
    }
    if samples.is_empty() {
        return (1.0, 0.0);
    }
    let min_z = samples.iter().map(|s| s.2).fold(f64::INFINITY, f64::min);

    let cost = |s: f64| -> (f64, f64) {
        let (mut num, mut den) = (0.0, 0.0);
        for &(x, y, z, u, v) in &samples {
            let zz = z + s;
            if zz <= 1e-6 {
                return (f64::INFINITY, 1.0);
            }
            let (a, b) = (x / zz, y / zz);
            num += a * u + b * v;
            den += a * a + b * b;
        }
        let f = if den > 0.0 { num / den } else { 1.0 };
        let mut err = 0.0;
        for &(x, y, z, u, v) in &samples {
            let zz = z + s;
            let (du, dv) = (f * x / zz - u, f * y / zz - v);
            err += du * du + dv * dv;
        }
        (err, f)
    };

    // Shift must keep every depth positive.
    let lo = -min_z + 1e-3;
    let hi = lo + 20.0 * samples.iter().map(|s| s.2.abs()).fold(0.0, f64::max).max(1.0);
    let n = 200;
    let mut best = (f64::INFINITY, lo);
    for i in 0..=n {
        let s = lo + (hi - lo) * (i as f64 / n as f64).powi(2); // denser near lo
        let (e, _) = cost(s);
        if e < best.0 {
            best = (e, s);
        }
    }
    let width_step = (hi - lo) / n as f64 * 2.0;
    let (mut a, mut b) = ((best.1 - width_step).max(lo), best.1 + width_step);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..60 {
        let c = b - g * (b - a);
        let d = a + g * (b - a);
        if cost(c).0 < cost(d).0 {
            b = d;
        } else {
            a = c;
        }
    }
    let s = (a + b) / 2.0;
    (cost(s).1, s)
}
