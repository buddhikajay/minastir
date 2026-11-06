//! Semantic segmentation: SegFormer-B5 trained on ADE20K (150 classes), via ONNX Runtime.
//! We collapse ADE's classes into the handful a home/street scene needs.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const INPUT: usize = 640;

/// Scene classes Minastir cares about.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Label {
    Other = 0,
    Road,
    Sidewalk,
    Grass,
    /// Bare earth, gravel, field, generic floor/ground.
    Ground,
    Building,
    Tree,
    Plant,
    Fence,
    Vehicle,
    Sky,
    Water,
    Pole,
    /// Rubbish / wheelie bin (ADE20K "ashcan").
    Bin,
}

impl Label {
    pub fn from_u8(b: u8) -> Self {
        use Label::*;
        [Other, Road, Sidewalk, Grass, Ground, Building, Tree, Plant, Fence, Vehicle, Sky, Water, Pole, Bin]
            .get(b as usize)
            .copied()
            .unwrap_or(Other)
    }

    pub fn is_ground(self) -> bool {
        matches!(self, Label::Road | Label::Sidewalk | Label::Grass | Label::Ground)
    }
}

/// ADE20K index → our label (indices from the model's id2label).
fn from_ade(i: usize) -> Label {
    match i {
        6 | 52 => Label::Road,                    // road, path
        11 | 53 => Label::Sidewalk,               // sidewalk, stairs (outdoor steps)
        9 => Label::Grass,                        // grass
        13 | 29 | 3 | 46 | 91 | 94 => Label::Ground, // earth, field, floor, sand, dirt track, land
        1 | 25 | 48 | 0 | 79 | 84 | 61 => Label::Building, // building, house, skyscraper, wall, hut, tower, bridge
        4 | 72 => Label::Tree,                    // tree, palm
        17 | 66 => Label::Plant,                  // plant, flower
        32 | 38 => Label::Fence,                  // fence, railing
        20 | 80 | 83 | 102 | 127 | 116 => Label::Vehicle, // car, bus, truck, van, bicycle, minibike
        2 => Label::Sky,
        21 | 26 | 60 | 109 | 128 => Label::Water, // water, sea, river, pool, lake
        93 | 87 => Label::Pole,                   // pole, streetlight
        138 => Label::Bin,                        // ashcan, trash can, garbage can, wastebin
        _ => Label::Other,
    }
}

pub struct Segmenter {
    session: Session,
}

/// Labels on a coarse grid (the model predicts at 1/4 input resolution).
#[derive(Clone)]
pub struct LabelMap {
    pub width: usize,
    pub height: usize,
    pub labels: Vec<Label>,
}

impl LabelMap {
    /// Label at normalised image coordinates.
    pub fn at(&self, u: f64, v: f64) -> Label {
        let x = ((u * self.width as f64) as usize).min(self.width - 1);
        let y = ((v * self.height as f64) as usize).min(self.height - 1);
        self.labels[y * self.width + x]
    }
}

impl Segmenter {
    pub fn load(model: &Path, device: Device) -> Result<Self> {
        let dims = [("batch_size", 1), ("num_channels", 3), ("height", INPUT as i64), ("width", INPUT as i64)];
        Ok(Self { session: onnx::session(model, device, &dims)? })
    }

    /// `rgb`: 640×640 RGB24 (the image squashed to square, as SegFormer's preprocessor does).
    pub fn segment(&mut self, rgb: &[u8]) -> Result<LabelMap> {
        let (probs, h, w) = self.probs(rgb)?;
        Ok(argmax(&probs, w, h))
    }

    /// Per-pixel probabilities of our labels (summed over the ADE classes that map to each), on
    /// the model's output grid (1/4 of the input): `NL` planes of h × w.
    fn probs(&mut self, rgb: &[u8]) -> Result<(Vec<f32>, usize, usize)> {
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let plane = INPUT * INPUT;
        anyhow::ensure!(rgb.len() == plane * 3, "segmenter input must be {INPUT}x{INPUT} RGB24");
        let mut input = vec![0f32; 3 * plane];
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                input[c * plane + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        let t = Tensor::from_array(([1usize, 3, INPUT, INPUT], input)).map_err(|e| anyhow!("{e}"))?;
        let out = self.session.run(ort::inputs![t]).map_err(|e| anyhow!("{e}"))?;
        let (shape, logits) = out["logits"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (classes, h, w) = (shape[1] as usize, shape[2] as usize, shape[3] as usize);
        let map: Vec<usize> = (0..classes).map(|c| from_ade(c) as usize).collect();
        let mut probs = vec![0f32; NL * h * w];
        let mut e = vec![0f32; classes];
        for i in 0..h * w {
            let mx = (0..classes).map(|c| logits[c * h * w + i]).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for c in 0..classes {
                e[c] = (logits[c * h * w + i] - mx).exp();
                sum += e[c];
            }
            for c in 0..classes {
                probs[map[c] * h * w + i] += e[c] / sum;
            }
        }
        Ok((probs, h, w))
    }

    /// Labels for a whole lens image at fine resolution. Two passes, blended per pixel:
    /// the whole image squashed to the model's input (full context: road vs footpath is a matter
    /// of where things sit in the scene), and overlapping tiles of the image at `fine_short` pixels
    /// on its short side (detail: kerb and path edges). Output grid: 1/4 of the fine image.
    pub fn segment_multiscale(&mut self, rgb: &[u8], width: usize, height: usize, fine_short: usize) -> Result<LabelMap> {
        let (gp, gh, gw) = self.probs(&crate::calibrate::resize(rgb, width, height, INPUT, INPUT))?;
        let s = fine_short as f64 / width.min(height) as f64;
        let (fw, fh) = (((width as f64 * s) as usize).max(INPUT), ((height as f64 * s) as usize).max(INPUT));
        let fine = crate::calibrate::resize(rgb, width, height, fw, fh);
        let (ow, oh) = (fw / 4, fh / 4);
        let mut acc = vec![0f32; NL * ow * oh];
        let mut wsum = vec![0f32; ow * oh];
        let starts = |len: usize| -> Vec<usize> {
            let n = (len - INPUT).div_ceil(INPUT * 3 / 4) + 1;
            (0..n).map(|i| if n == 1 { 0 } else { i * (len - INPUT) / (n - 1) }).collect()
        };
        let mut tile = vec![0u8; INPUT * INPUT * 3];
        for &ty in &starts(fh) {
            for &tx in &starts(fw) {
                for y in 0..INPUT {
                    let src = ((ty + y) * fw + tx) * 3;
                    tile[y * INPUT * 3..(y + 1) * INPUT * 3].copy_from_slice(&fine[src..src + INPUT * 3]);
                }
                let (tp, th, tw) = self.probs(&tile)?;
                for y in 0..th {
                    let oy = ty / 4 + y;
                    if oy >= oh {
                        continue;
                    }
                    // Trust tile centres more than edges (less context there).
                    let wy = 1.0 - 0.85 * ((y as f32 + 0.5) / th as f32 * 2.0 - 1.0).abs();
                    for x in 0..tw {
                        let ox = tx / 4 + x;
                        if ox >= ow {
                            continue;
                        }
                        let wx = 1.0 - 0.85 * ((x as f32 + 0.5) / tw as f32 * 2.0 - 1.0).abs();
                        let wgt = wx * wy;
                        let o = oy * ow + ox;
                        wsum[o] += wgt;
                        for l in 0..NL {
                            acc[l * ow * oh + o] += wgt * tp[l * th * tw + y * tw + x];
                        }
                    }
                }
            }
        }
        // Blend with the whole-image pass (bilinear from its coarse grid).
        const W_GLOBAL: f32 = 0.4;
        for oy in 0..oh {
            let gy = ((oy as f64 + 0.5) / oh as f64 * gh as f64 - 0.5).clamp(0.0, (gh - 1) as f64);
            let (y0, ty) = (gy.floor() as usize, (gy - gy.floor()) as f32);
            let y1 = (y0 + 1).min(gh - 1);
            for ox in 0..ow {
                let gx = ((ox as f64 + 0.5) / ow as f64 * gw as f64 - 0.5).clamp(0.0, (gw - 1) as f64);
                let (x0, tx) = (gx.floor() as usize, (gx - gx.floor()) as f32);
                let x1 = (x0 + 1).min(gw - 1);
                let o = oy * ow + ox;
                let ws = wsum[o].max(1e-6);
                for l in 0..NL {
                    let g = |y: usize, x: usize| gp[l * gh * gw + y * gw + x];
                    let glob = (g(y0, x0) * (1.0 - tx) + g(y0, x1) * tx) * (1.0 - ty) + (g(y1, x0) * (1.0 - tx) + g(y1, x1) * tx) * ty;
                    let i = l * ow * oh + o;
                    acc[i] = W_GLOBAL * glob + (1.0 - W_GLOBAL) * acc[i] / ws;
                }
            }
        }
        Ok(argmax(&acc, ow, oh))
    }
}

/// Number of our labels.
const NL: usize = 14;

fn argmax(probs: &[f32], w: usize, h: usize) -> LabelMap {
    let n = w * h;
    let labels = (0..n)
        .map(|i| {
            let best = (0..NL).max_by(|&a, &b| probs[a * n + i].total_cmp(&probs[b * n + i])).unwrap();
            Label::from_u8(best as u8)
        })
        .collect();
    LabelMap { width: w, height: h, labels }
}
