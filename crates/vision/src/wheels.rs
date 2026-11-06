//! Which way a parked vehicle points, from its wheels.
//!
//! An outline barely says which way a car points once it's 20 m away and seen from a few metres
//! up: the side-on blob of a car along the kerb matches a car turned 60° almost as well. Wheels
//! do say: two wheels on the same side touch the ground on the car's axis, so the line through
//! their contact points, projected onto the ground through the camera's lens and scene map, is
//! the car's direction (up to which end is the front).
//!
//! Wheels are found by Grounding DINO (tiny; Apache-2.0, IDEA Research), an open-vocabulary
//! detector asked for "wheel", run on a crop around the vehicle's box. Input: the box widened 20%
//! and padded to a square, resampled to 800×800 RGB, normalised with ImageNet mean / std; plus
//! the prompt's token ids. Output: 900 candidate boxes (centre, size; normalised to the crop) and
//! per-token logits.

use crate::{
    geometry::CameraModel,
    onnx::{self, Device},
};
use anyhow::{anyhow, Context, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;

/// Folder under the models directory, and the files it needs.
pub const MODEL_DIR: &str = "grounding_dino_tiny";
pub const MODEL_FILES: [&str; 2] = ["model.onnx", "tokenizer.json"];
pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/grounding-dino-tiny-ONNX/resolve/main/";

const SIDE: usize = 800;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const PROMPT: &str = "wheel .";

/// Detector confidence for a wheel (Grounding DINO's sigmoid score; clear wheels score 0.4–0.7).
pub const MIN_SCORE: f32 = 0.3;

/// A detected wheel: score and box (normalised frame coordinates).
#[derive(Clone, Copy, Debug)]
pub struct Wheel {
    pub score: f32,
    pub bbox: [f32; 4],
}

impl Wheel {
    /// Where the tyre touches the ground: the bottom centre of its box.
    pub fn contact(&self) -> (f64, f64) {
        (((self.bbox[0] + self.bbox[2]) / 2.0) as f64, self.bbox[3] as f64)
    }
}

pub struct Wheels {
    session: Session,
    ids: Vec<i64>,
}

impl Wheels {
    pub fn present(models: &Path) -> bool {
        MODEL_FILES.iter().all(|f| models.join(MODEL_DIR).join(f).exists())
    }

    pub fn load(models: &Path, device: Device) -> Result<Self> {
        let dir = models.join(MODEL_DIR);
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        let ids: Vec<i64> = tok.encode(PROMPT, true).map_err(|e| anyhow!("{e}"))?.get_ids().iter().map(|&i| i as i64).collect();
        let session = onnx::session(&dir.join("model.onnx"), device, &[("sequence_length", ids.len() as i64)])?;
        Ok(Self { session, ids })
    }

    /// Wheels inside box `b` (normalised x0, y0, x1, y1) of an RGB24 frame, most confident first,
    /// overlapping duplicates removed.
    pub fn detect(&mut self, frame: (&[u8], usize, usize), b: [f32; 4]) -> Result<Vec<Wheel>> {
        let (px, fw, fh) = frame;
        let (cx, cy) = ((b[0] + b[2]) as f64 / 2.0 * fw as f64, (b[1] + b[3]) as f64 / 2.0 * fh as f64);
        let side = ((b[2] - b[0]) as f64 * fw as f64).max((b[3] - b[1]) as f64 * fh as f64) * 1.2;
        let (x0, y0) = (cx - side / 2.0, cy - side / 2.0);
        let plane = SIDE * SIDE;
        let mut input = vec![0f32; 3 * plane];
        for y in 0..SIDE {
            for x in 0..SIDE {
                let sx = x0 + (x as f64 + 0.5) * side / SIDE as f64 - 0.5;
                let sy = y0 + (y as f64 + 0.5) * side / SIDE as f64 - 0.5;
                // Outside the frame: black (padding), as the model was trained with.
                let rgb = if sx < 0.0 || sy < 0.0 || sx > (fw - 1) as f64 || sy > (fh - 1) as f64 {
                    [0f32; 3]
                } else {
                    let (ix, iy) = (sx as usize, sy as usize);
                    let (ax, ay) = ((sx - ix as f64) as f32, (sy - iy as f64) as f32);
                    let (jx, jy) = ((ix + 1).min(fw - 1), (iy + 1).min(fh - 1));
                    let at = |xx: usize, yy: usize, c: usize| px[(yy * fw + xx) * 3 + c] as f32;
                    [0, 1, 2].map(|c| {
                        let top = at(ix, iy, c) * (1.0 - ax) + at(jx, iy, c) * ax;
                        let bot = at(ix, jy, c) * (1.0 - ax) + at(jx, jy, c) * ax;
                        top * (1.0 - ay) + bot * ay
                    })
                };
                for c in 0..3 {
                    input[c * plane + y * SIDE + x] = (rgb[c] / 255.0 - MEAN[c]) / STD[c];
                }
            }
        }
        let n = self.ids.len();
        let t = |v: Vec<i64>| Tensor::from_array(([1usize, n], v)).map_err(|e| anyhow!("{e}"));
        let outputs = self
            .session
            .run(ort::inputs![
                "pixel_values" => Tensor::from_array(([1usize, 3, SIDE, SIDE], input)).map_err(|e| anyhow!("{e}"))?,
                "input_ids" => t(self.ids.clone())?,
                "token_type_ids" => t(vec![0; n])?,
                "attention_mask" => t(vec![1; n])?,
                "pixel_mask" => Tensor::from_array(([1usize, SIDE, SIDE], vec![1i64; plane])).map_err(|e| anyhow!("{e}"))?,
            ])
            .map_err(|e| anyhow!("{e}"))?;
        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (_, boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (queries, width) = (lshape[1] as usize, lshape[2] as usize);
        let mut found: Vec<Wheel> = Vec::new();
        for q in 0..queries {
            // The prompt's own tokens (not [CLS] / [SEP]).
            let score = (1..n - 1).map(|k| logits[q * width + k]).fold(f32::MIN, f32::max);
            let score = 1.0 / (1.0 + (-score).exp());
            if score < MIN_SCORE {
                continue;
            }
            let bx = &boxes[q * 4..q * 4 + 4];
            let to_frame = |u: f32, v: f32| (((x0 + u as f64 * side) / fw as f64) as f32, ((y0 + v as f64 * side) / fh as f64) as f32);
            let (a0, a1) = to_frame(bx[0] - bx[2] / 2.0, bx[1] - bx[3] / 2.0);
            let (c0, c1) = to_frame(bx[0] + bx[2] / 2.0, bx[1] + bx[3] / 2.0);
            found.push(Wheel { score, bbox: [a0, a1, c0, c1] });
        }
        found.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Wheel> = Vec::new();
        for w in found {
            // The same wheel found twice (tyre, rim): keep the most confident.
            if kept.iter().all(|k| overlap(&k.bbox, &w.bbox) < 0.3) {
                kept.push(w);
            }
        }
        Ok(kept)
    }
}

/// Intersection over the smaller box's area.
fn overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let small = ((a[2] - a[0]) * (a[3] - a[1])).min((b[2] - b[0]) * (b[3] - b[1]));
    if small > 0.0 {
        w * h / small
    } else {
        0.0
    }
}

/// A vehicle's axis from one view's wheels.
#[derive(Clone, Copy, Debug)]
pub struct Axis {
    /// Degrees, camera-yaw convention, modulo 180 (which end is the front isn't known).
    pub heading: f64,
    /// How far apart the two contact points are in the image (normalised width units): longer
    /// is better conditioned.
    pub spread: f64,
    /// Their distance apart on the ground, metres.
    pub apart: f64,
}

/// Wheelbase as a share of a vehicle's length: same-side wheels of cars, utes and vans sit
/// 55–70% of the length apart. Two contact points much closer than this are the two wheels of
/// one axle (a car seen nose-on) or one wheel twice; much further, not the same vehicle.
const WHEELBASE: (f64, f64) = (0.45, 0.9);

/// The axis through the best-separated pair of wheels inside `vehicle` (normalised box) whose
/// ground distance is a plausible wheelbase for a vehicle `length` metres long.
pub fn axis(model: &CameraModel, wheels: &[Wheel], vehicle: [f32; 4], length: f64, aspect: f64) -> Option<Axis> {
    let inside: Vec<(f64, f64, [f64; 3])> = wheels
        .iter()
        .filter(|w| {
            let (u, v) = (((w.bbox[0] + w.bbox[2]) / 2.0), ((w.bbox[1] + w.bbox[3]) / 2.0));
            u >= vehicle[0] && u <= vehicle[2] && v >= vehicle[1] && v <= vehicle[3]
        })
        .filter_map(|w| {
            let (u, v) = w.contact();
            Some((u, v, model.ground_point3(u, v, 80.0)?))
        })
        .collect();
    let mut best: Option<Axis> = None;
    for i in 0..inside.len() {
        for j in i + 1..inside.len() {
            let (p, q) = (inside[i], inside[j]);
            let (dx, dz) = (q.2[0] - p.2[0], q.2[2] - p.2[2]);
            let apart = dx.hypot(dz);
            if apart < WHEELBASE.0 * length || apart > WHEELBASE.1 * length {
                continue;
            }
            let spread = ((q.0 - p.0) * aspect).hypot(q.1 - p.1);
            if best.is_none_or(|b| spread > b.spread) {
                best = Some(Axis { heading: dx.atan2(dz).to_degrees().rem_euclid(180.0), spread, apart });
            }
        }
    }
    best
}

/// Several views' axes as one: the mean of the doubled angles (axes are 180°-periodic), each
/// weighted by how well its wheels are separated in the image.
pub fn combine(axes: &[Axis]) -> Option<f64> {
    let (mut s, mut c) = (0.0, 0.0);
    for a in axes {
        let t = (2.0 * a.heading).to_radians();
        s += a.spread * t.sin();
        c += a.spread * t.cos();
    }
    (s != 0.0 || c != 0.0).then(|| (s.atan2(c).to_degrees() / 2.0).rem_euclid(180.0))
}

/// Downloads the model files that aren't there yet.
pub fn fetch(models: &Path) -> Result<()> {
    for f in MODEL_FILES {
        let dest = models.join(MODEL_DIR).join(f);
        if !dest.exists() {
            let remote = if f.ends_with(".onnx") { format!("onnx/{f}") } else { f.to_string() };
            crate::live::download(&format!("{MODEL_URL}{remote}"), &dest).with_context(|| format!("downloading {f}"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_average_across_the_180_degree_wrap() {
        let a = |h: f64| Axis { heading: h, spread: 1.0, apart: 3.0 };
        let m = combine(&[a(178.0), a(4.0)]).unwrap();
        assert!((m - 1.0).abs() < 1e-9 || (m - 181.0).abs() < 1e-9, "{m}");
    }
}
