//! Body keypoints for a person box: RTMPose-m (Apache-2.0, OpenMMLab; trained on seven public
//! body datasets, 17 COCO keypoints) via ONNX Runtime.
//!
//! Used to find where a person really stands: their ankles when visible, and otherwise — feet
//! hidden behind a letterbox, a car or another child — their head together with their height,
//! learned while their feet were in view. The bottom of the detection box is only the bottom of
//! what's visible.
//!
//! Input: the box widened by 25% and padded to 3:4, resampled to 192×256 RGB, normalised with
//! ImageNet mean / std, NCHW. Output: per keypoint, a 1D distribution over x (384 bins) and y
//! (512 bins) at half-pixel resolution ("SimCC"); the peak of each is the keypoint and its height
//! the confidence.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;

pub const MODEL_FILE: &str = "rtmpose_m_body7.onnx";
/// Distributed inside a zip (`20230831/rtmpose_onnx/<name>/end2end.onnx`).
pub const MODEL_URL: &str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip";

const W: usize = 192;
const H: usize = 256;
const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const STD: [f32; 3] = [58.395, 57.12, 57.375];

/// COCO keypoint indices.
pub mod kp {
    pub const NOSE: usize = 0;
    pub const L_EYE: usize = 1;
    pub const R_EYE: usize = 2;
    pub const L_SHOULDER: usize = 5;
    pub const R_SHOULDER: usize = 6;
    pub const L_HIP: usize = 11;
    pub const R_HIP: usize = 12;
    pub const L_KNEE: usize = 13;
    pub const R_KNEE: usize = 14;
    pub const L_ANKLE: usize = 15;
    pub const R_ANKLE: usize = 16;
}

/// 17 keypoints: normalised frame coordinates (u, v) and confidence.
#[derive(Clone, Copy, Debug)]
pub struct Keypoints(pub [[f32; 3]; 17]);

impl Keypoints {
    /// Where the feet touch the ground, in normalised frame coordinates, if the model sees them:
    /// the lower of the confidently seen ankles (with the other's x when both are seen).
    pub fn feet(&self, min_score: f32) -> Option<(f32, f32)> {
        let feet = self.ankles(min_score)?;
        // Legs must come out a plausible length against the torso: with the lower body hidden
        // (behind a car's bonnet) the model still guesses "ankles", bunched up under the hips.
        let mid = |a: usize, b: usize| {
            let (p, q) = (self.0[a], self.0[b]);
            (p[2] >= min_score && q[2] >= min_score).then(|| (p[1] + q[1]) / 2.0)
        };
        if let (Some(sh), Some(hip)) = (mid(kp::L_SHOULDER, kp::R_SHOULDER), mid(kp::L_HIP, kp::R_HIP)) {
            let torso = hip - sh;
            if torso > 0.0 && feet.1 - hip < LEG_TO_TORSO * torso {
                return None;
            }
        }
        Some(feet)
    }

    fn ankles(&self, min_score: f32) -> Option<(f32, f32)> {
        let seen: Vec<[f32; 3]> = [kp::L_ANKLE, kp::R_ANKLE].iter().map(|&i| self.0[i]).filter(|p| p[2] >= min_score).collect();
        match seen.as_slice() {
            [a] => Some((a[0], a[1])),
            [a, b] => Some(((a[0] + b[0]) / 2.0, a[1].max(b[1]))),
            _ => None,
        }
    }
}

/// Shortest believable leg (hip to ankle, in the image) as a fraction of the torso (shoulder to
/// hip). Adults' legs are ~1.5 torsos, small children's ~1.1; well under that, the "ankles" are a
/// guess at hidden feet.
const LEG_TO_TORSO: f32 = 0.8;

/// Keypoint confidence for a seen ankle (RTMPose's SimCC peak height; ~0.5+ when clearly visible).
pub const SEEN: f32 = 0.35;

pub struct Pose {
    session: Session,
}

impl Pose {
    pub fn load(models: &Path, device: Device) -> Result<Self> {
        let path = models.join(MODEL_FILE);
        Ok(Self { session: onnx::session(&path, device, &[("batch", 1)])? })
    }

    pub fn describe(&self) -> String {
        let io = |o: &[ort::value::Outlet]| o.iter().map(|x| format!("{} {:?}", x.name(), x.dtype())).collect::<Vec<_>>().join(", ");
        format!("inputs: {} | outputs: {}", io(self.session.inputs()), io(self.session.outputs()))
    }

    /// Keypoints of the person in box `b` (normalised x0, y0, x1, y1) of an RGB24 frame.
    pub fn estimate(&mut self, frame: (&[u8], usize, usize), b: [f32; 4]) -> Result<Keypoints> {
        let (px, fw, fh) = frame;
        // Crop region in pixels: centre, widened 25%, fixed to the model's 3:4.
        let (cx, cy) = ((b[0] + b[2]) as f64 / 2.0 * fw as f64, (b[1] + b[3]) as f64 / 2.0 * fh as f64);
        let (mut sw, mut sh) = ((b[2] - b[0]) as f64 * fw as f64 * 1.25, (b[3] - b[1]) as f64 * fh as f64 * 1.25);
        let aspect = W as f64 / H as f64;
        if sw > sh * aspect {
            sh = sw / aspect;
        } else {
            sw = sh * aspect;
        }
        let (x0, y0) = (cx - sw / 2.0, cy - sh / 2.0);
        let plane = W * H;
        let mut input = vec![0f32; 3 * plane];
        for y in 0..H {
            for x in 0..W {
                // Bilinear sample at the pixel centre; outside the frame stays black (zero before
                // normalising, as in training).
                let sx = x0 + (x as f64 + 0.5) * sw / W as f64 - 0.5;
                let sy = y0 + (y as f64 + 0.5) * sh / H as f64 - 0.5;
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
                    input[c * plane + y * W + x] = (rgb[c] - MEAN[c]) / STD[c];
                }
            }
        }
        let tensor = Tensor::from_array(([1usize, 3, H, W], input)).map_err(|e| anyhow!("{e}"))?;
        let outputs = self.session.run(ort::inputs![tensor]).map_err(|e| anyhow!("{e}"))?;
        let (sx_shape, simcc_x) = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (sy_shape, simcc_y) = outputs[1].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (nx, ny) = (sx_shape[2] as usize, sy_shape[2] as usize);
        let mut pts = [[0f32; 3]; 17];
        for (k, p) in pts.iter_mut().enumerate() {
            let peak = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
            let (ix, vx) = peak(&simcc_x[k * nx..(k + 1) * nx]);
            let (iy, vy) = peak(&simcc_y[k * ny..(k + 1) * ny]);
            // Bins are half pixels of the 192×256 input.
            let (ux, uy) = (ix as f64 / 2.0, iy as f64 / 2.0);
            let fx = x0 + ux * sw / W as f64;
            let fy = y0 + uy * sh / H as f64;
            *p = [(fx / fw as f64) as f32, (fy / fh as f64) as f32, vx.min(vy)];
        }
        Ok(Keypoints(pts))
    }
}
