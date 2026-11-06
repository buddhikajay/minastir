//! Point matching between two camera images: ALIKED keypoints + LightGlue (Apache-2.0), as the
//! end-to-end ONNX pipeline from fabio-sim/LightGlue-ONNX. Both images are letterboxed onto one
//! canvas (the pipeline batches them), and matches come back in each image's own normalised
//! coordinates.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;

pub const MODEL_URL: &str =
    "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v3.0/raco_aliked_lightglue_pipeline_k2048.onnx";
pub const MODEL_FILE: &str = "aliked_lightglue_k2048.onnx";

/// Canvas both images are fitted into (multiples of 32, as the model requires). Distant
/// structure (the far kerb, houses across the road) needs the resolution.
const CW: usize = 1536;
const CH: usize = 1152;

pub struct Matcher {
    session: Session,
}

/// One correspondence: normalised (u, v) in image A and in image B, plus confidence.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub a: (f64, f64),
    pub b: (f64, f64),
    pub score: f32,
}

/// An RGB24 image view.
pub struct Img<'a> {
    pub rgb: &'a [u8],
    pub width: usize,
    pub height: usize,
}

/// Where an image sits on the canvas: scale and offset.
struct Fit {
    s: f64,
    ox: f64,
    oy: f64,
    w: usize,
    h: usize,
}

impl Matcher {
    pub fn load(model: &Path, device: Device) -> Result<Self> {
        Ok(Self { session: onnx::session(model, device, &[])? })
    }

    pub fn match_pair(&mut self, a: &Img, b: &Img, min_score: f32) -> Result<Vec<Match>> {
        let plane = CW * CH;
        let mut input = vec![0f32; 2 * 3 * plane];
        let fits = [place(a, &mut input[..3 * plane]), place(b, &mut input[3 * plane..])];
        let t = Tensor::from_array(([2usize, 3, CH, CW], input)).map_err(|e| anyhow!("{e}"))?;
        let out = self.session.run(ort::inputs![t]).map_err(|e| anyhow!("{e}"))?;
        let (kshape, kps) = out["keypoints"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (_, matches) = out["matches"].try_extract_tensor::<i64>().map_err(|e| anyhow!("{e}"))?;
        let (_, scores) = out["mscores"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let k = kshape[1] as usize;
        let kp = |img: usize, i: usize| -> (f64, f64) {
            let o = (img * k + i) * 2;
            (kps[o] as f64, kps[o + 1] as f64)
        };
        let mut res = Vec::new();
        for (m, &score) in matches.chunks_exact(3).zip(scores.iter()) {
            if score < min_score {
                continue;
            }
            let (pa, pb) = (kp(0, m[1] as usize), kp(1, m[2] as usize));
            let (Some(ua), Some(ub)) = (unfit(&fits[0], pa), unfit(&fits[1], pb)) else { continue };
            res.push(Match { a: ua, b: ub, score });
        }
        Ok(res)
    }
}

/// Letterbox `img` onto the canvas (CHW floats 0..1) and remember where it went.
fn place(img: &Img, out: &mut [f32]) -> Fit {
    let s = (CW as f64 / img.width as f64).min(CH as f64 / img.height as f64);
    let (w, h) = ((img.width as f64 * s) as usize, (img.height as f64 * s) as usize);
    let (ox, oy) = ((CW - w) / 2, (CH - h) / 2);
    let small = crate::calibrate::resize(img.rgb, img.width, img.height, w, h);
    let plane = CW * CH;
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let o = (y + oy) * CW + x + ox;
            for c in 0..3 {
                out[c * plane + o] = small[i + c] as f32 / 255.0;
            }
        }
    }
    Fit { s, ox: ox as f64, oy: oy as f64, w, h }
}

/// Canvas pixel → normalised image coordinates (None if in the padding).
fn unfit(f: &Fit, (x, y): (f64, f64)) -> Option<(f64, f64)> {
    let (u, v) = ((x - f.ox) / f.w as f64, (y - f.oy) / f.h as f64);
    let _ = f.s;
    ((0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)).then_some((u, v))
}
