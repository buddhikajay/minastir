//! Appearance fingerprints for detected objects: a learned embedding of each object's image crop.
//! Two crops of the same car — from different cameras, or the same camera seconds apart — land
//! close together; different cars land apart. Used to tell which sightings are the same object,
//! instead of distance rules.
//!
//! Two models are supported:
//! - DINOv3 ViT-B/16 (Meta, 2025; DINOv3 License) — the strongest general visual features;
//!   preferred;
//! - DINOv2-small (Meta, Apache-2.0, ONNX from onnx-community) — general-purpose fallback.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use ort::{session::Session, value::Tensor};
use std::path::Path;

pub const MODEL_FILE: &str = "dinov2_small.onnx";
pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx";
/// DINOv3 ViT-B/16 (Meta, Aug 2025; ONNX from onnx-community, weights in model.onnx_data).
pub const DINO3_MODEL_FILE: &str = "dinov3_vitb16/model.onnx";
pub const DINO3_MODEL_URL: &str = "https://huggingface.co/onnx-community/dinov3-vitb16-pretrain-lvd1689m-ONNX/resolve/main/onnx/";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// DINOv2: 224×224, ImageNet-normalised in, tokens [N, T, 384] out.
    Dino,
    /// DINOv3: 256×256, ImageNet-normalised in, tokens [N, 1 + 4 registers + patches, 768] out.
    Dino3,
}

pub struct Embedder {
    session: Session,
    kind: Kind,
    /// The GPU needs fixed input shapes, so it runs one crop at a time.
    one_at_a_time: bool,
}

/// An RGB crop to fingerprint.
pub struct Crop<'a> {
    pub rgb: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl Embedder {
    /// The best fingerprint model in `models`: DINOv3, else DINOv2.
    pub fn load_best(models: &Path, device: Device) -> Result<Self> {
        let dino3 = models.join(DINO3_MODEL_FILE);
        if dino3.exists() {
            return Self::load_kind(&dino3, Kind::Dino3, device);
        }
        Self::load_kind(&models.join(MODEL_FILE), Kind::Dino, device)
    }

    pub fn load(model: &Path, device: Device) -> Result<Self> {
        Self::load_kind(model, Kind::Dino, device)
    }

    pub fn load_kind(model: &Path, kind: Kind, device: Device) -> Result<Self> {
        let side = kind.side() as i64;
        let dims: Vec<(&str, i64)> = match kind {
            Kind::Dino => vec![("num_channels", 3), ("height", side), ("width", side)],
            Kind::Dino3 if device == Device::Gpu => vec![("s99", 1), ("s100", side), ("s4", side)],
            Kind::Dino3 => vec![("s100", side), ("s4", side)],
        };
        let one_at_a_time = device == Device::Gpu && kind == Kind::Dino3;
        Ok(Self { session: onnx::session(model, device, &dims)?, kind, one_at_a_time })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// One unit-length fingerprint per crop.
    pub fn embed(&mut self, crops: &[Crop]) -> Result<Vec<Vec<f32>>> {
        if self.one_at_a_time && crops.len() > 1 {
            let mut out = Vec::with_capacity(crops.len());
            for c in crops.chunks(1) {
                out.extend(self.embed(c)?);
            }
            return Ok(out);
        }
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let side = self.kind.side();
        let plane = side * side;
        let mut input = vec![0f32; crops.len() * 3 * plane];
        for (k, c) in crops.iter().enumerate() {
            letterbox(c, self.kind, &mut input[k * 3 * plane..(k + 1) * 3 * plane]);
        }
        let t = Tensor::from_array(([crops.len(), 3, side, side], input)).map_err(|e| anyhow!("{e}"))?;
        // The image's summary token joined with the mean of its patch tokens (each
        // normalised), so both overall look and local detail count.
        let out = self.session.run(ort::inputs!["pixel_values" => t]).map_err(|e| anyhow!("{e}"))?;
        let (shape, h) = out["last_hidden_state"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (n, tokens, dim) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
        let mut res = Vec::with_capacity(n);
        for k in 0..n {
            let base = k * tokens * dim;
            let cls: Vec<f32> = h[base..base + dim].to_vec();
            let mut mean = vec![0f32; dim];
            for t in self.kind.first_patch()..tokens {
                for d in 0..dim {
                    mean[d] += h[base + t * dim + d];
                }
            }
            let mut v = normalise(cls);
            v.extend(normalise(mean));
            res.push(normalise(v));
        }
        Ok(res)
    }
}

impl Kind {
    fn side(self) -> usize {
        match self {
            Kind::Dino => 224,  // multiple of the 14-px patch
            Kind::Dino3 => 256, // multiple of the 16-px patch
        }
    }

    /// Index of the first image-patch token (after the summary and register tokens).
    fn first_patch(self) -> usize {
        match self {
            Kind::Dino3 => 5,
            _ => 1,
        }
    }
}

/// Cosine similarity of two fingerprints (both unit length): 1 = identical look.
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= n);
    v
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Fit the crop into the model's square input keeping its shape (padding with the mean colour),
/// CHW, ImageNet-normalised.
fn letterbox(c: &Crop, kind: Kind, out: &mut [f32]) {
    let side = kind.side();
    let plane = side * side;
    for ch in 0..3 {
        // 0 = the mean colour after normalisation.
        out[ch * plane..(ch + 1) * plane].iter_mut().for_each(|v| *v = 0.0);
    }
    let s = (side as f64 / c.width.max(1) as f64).min(side as f64 / c.height.max(1) as f64);
    let (w, h) = (((c.width as f64 * s).round() as usize).clamp(1, side), ((c.height as f64 * s).round() as usize).clamp(1, side));
    let (ox, oy) = ((side - w) / 2, (side - h) / 2);
    for y in 0..h {
        for x in 0..w {
            let sx = (((x as f64 + 0.5) / s) as usize).min(c.width - 1);
            let sy = (((y as f64 + 0.5) / s) as usize).min(c.height - 1);
            let i = (sy * c.width + sx) * 3;
            let o = (y + oy) * side + x + ox;
            for ch in 0..3 {
                let v = c.rgb[i + ch] as f32;
                out[ch * plane + o] = (v / 255.0 - MEAN[ch]) / STD[ch];
            }
        }
    }
}
