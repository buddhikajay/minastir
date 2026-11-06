//! Object detector: D-FINE (Apache-2.0, COCO classes) via ONNX Runtime, on the Apple GPU / Neural
//! Engine through CoreML when available, CPU otherwise.
//!
//! Input: 640×640 RGB (callers resize — ffmpeg does it during decode), scaled to 0..1, NCHW.
//! Output: 300 queries × 80 class logits (sigmoid) + boxes as normalised (cx, cy, w, h). The model
//! is end-to-end: no NMS needed.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use serde::Serialize;
use std::path::Path;

pub const INPUT: usize = 640;

/// COCO classes we care about for a home/store scene, with the label we show.
pub fn class_name(coco: usize) -> Option<&'static str> {
    Some(match coco {
        0 => "person",
        1 => "bicycle",
        2 => "car",
        3 => "motorcycle",
        5 => "bus",
        7 => "truck",
        15 => "cat",
        16 => "dog",
        _ => return None,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub class: &'static str,
    pub score: f32,
    /// Normalised to the frame (0..1): left, top, right, bottom.
    pub bbox: [f32; 4],
}

pub struct Detector {
    session: Session,
    pub accelerated: bool,
}

impl Detector {
    pub fn load(model: &Path, use_gpu: bool) -> Result<Self> {
        let device = if use_gpu { Device::Gpu } else { Device::Cpu };
        let dims = [("batch_size", 1), ("height", INPUT as i64), ("width", INPUT as i64)];
        Ok(Self { session: onnx::session(model, device, &dims)?, accelerated: use_gpu })
    }

    pub fn describe(&self) -> String {
        let io = |o: &[ort::value::Outlet]| o.iter().map(|x| format!("{} {:?}", x.name(), x.dtype())).collect::<Vec<_>>().join(", ");
        format!("inputs: {} | outputs: {}", io(self.session.inputs()), io(self.session.outputs()))
    }

    /// Detect in a batch of 640×640 RGB24 frames.
    pub fn detect(&mut self, frames: &[&[u8]], min_score: f32) -> Result<Vec<Vec<Detection>>> {
        let n = frames.len();
        let plane = INPUT * INPUT;
        let mut input = vec![0f32; n * 3 * plane];
        for (b, rgb) in frames.iter().enumerate() {
            anyhow::ensure!(rgb.len() == plane * 3, "frame must be {INPUT}x{INPUT} RGB24");
            let base = b * 3 * plane;
            for (i, px) in rgb.chunks_exact(3).enumerate() {
                input[base + i] = px[0] as f32 / 255.0;
                input[base + plane + i] = px[1] as f32 / 255.0;
                input[base + 2 * plane + i] = px[2] as f32 / 255.0;
            }
        }
        let tensor = Tensor::from_array(([n, 3, INPUT, INPUT], input)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(|e| anyhow!("{e}"))?;
        let (lshape, logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (_, boxes) = outputs["pred_boxes"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (queries, classes) = (lshape[1] as usize, lshape[2] as usize);

        // Inverse sigmoid of the threshold, so we only exp() the few candidates that pass.
        let min_logit = (min_score / (1.0 - min_score)).ln();
        let mut out = Vec::with_capacity(n);
        for b in 0..n {
            let mut dets = Vec::new();
            for q in 0..queries {
                let row = &logits[(b * queries + q) * classes..][..classes];
                let Some((c, &l)) = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) else { continue };
                if l < min_logit {
                    continue;
                }
                let Some(class) = class_name(c) else { continue };
                let bx = &boxes[(b * queries + q) * 4..][..4];
                let (cx, cy, w, h) = (bx[0], bx[1], bx[2], bx[3]);
                dets.push(Detection {
                    class,
                    score: 1.0 / (1.0 + (-l).exp()),
                    bbox: [(cx - w / 2.0).max(0.0), (cy - h / 2.0).max(0.0), (cx + w / 2.0).min(1.0), (cy + h / 2.0).min(1.0)],
                });
            }
            out.push(suppress_duplicates(dets));
        }
        Ok(out)
    }
}

/// Coarse class families: the model may call one vehicle both "car" and "truck".
pub fn family(class: &str) -> u8 {
    match class {
        "person" => 0,
        "car" | "truck" | "bus" => 1,
        "bicycle" | "motorcycle" => 2,
        _ => 3,
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// D-FINE needs no NMS in principle, but it still emits near-identical boxes for one object at
/// lower scores (often with a different vehicle label). Keep the best box per object.
fn suppress_duplicates(mut dets: Vec<Detection>) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::with_capacity(dets.len());
    for d in dets {
        if !kept.iter().any(|k| family(k.class) == family(d.class) && iou(&k.bbox, &d.bbox) > 0.55) {
            kept.push(d);
        }
    }
    kept
}
