//! The scene describer: a vision-language model (Qwen3-VL-2B-Instruct, Apache-2.0; 4-bit ONNX
//! from onnx-community) that answers questions about an object's crop — what it is, which way it
//! faces. Answers are read as probabilities over the allowed choices (the model's own confidence),
//! not as free text, so they can be weighed as evidence like everything else.
//!
//! Pipeline (as in transformers' Qwen3VLForConditionalGeneration):
//!   image → 16-px patches (two identical frames, grouped 2×2) → vision encoder → one feature per
//!   2×2 group → spliced into the prompt's token embeddings at the image placeholder → decoder,
//!   with 3-D rotary positions (time, row, column) for the image tokens.

use crate::onnx::{self, Device};
use anyhow::{anyhow, Context, Result};
use half::f16;
use ort::{
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor},
};
use std::{borrow::Cow, path::Path};
use tokenizers::Tokenizer;

pub const MODEL_DIR: &str = "qwen3vl_2b";
pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX/resolve/main/";
/// Files to fetch (relative to the model dir and the URL).
pub const FILES: &[&str] = &[
    "tokenizer.json",
    "onnx/vision_encoder_q4f16.onnx",
    "onnx/vision_encoder_q4f16.onnx_data",
    "onnx/embed_tokens_q4f16.onnx",
    "onnx/embed_tokens_q4f16.onnx_data",
    "onnx/decoder_model_merged_q4f16.onnx",
    "onnx/decoder_model_merged_q4f16.onnx_data",
];

const PATCH: usize = 16;
const MERGE: usize = 2;
const TEMPORAL: usize = 2;
const IMAGE_PAD: u32 = 151655;
const IM_END: u32 = 151645;
const LAYERS: usize = 28;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
/// Longest image side fed to the model (pixels): small crops are enlarged to this.
const SIDE: f64 = 448.0;

pub struct Describer {
    vision: Session,
    embed: Session,
    decoder: Session,
    tok: Tokenizer,
}

/// The state after reading the prompt: what comes next, and the cache to continue from.
struct Prefill {
    logits: Vec<f32>,
    cache: Vec<(String, DynValue)>,
    next_pos: i64,
    len: usize,
}

impl Describer {
    pub fn load(models: &Path, device: Device) -> Result<Self> {
        let dir = models.join(MODEL_DIR);
        let s = |f: &str| onnx::session(&dir.join("onnx").join(f), device, &[("batch_size", 1)]);
        Ok(Self {
            vision: s("vision_encoder_q4f16.onnx")?,
            embed: s("embed_tokens_q4f16.onnx")?,
            decoder: s("decoder_model_merged_q4f16.onnx")?,
            tok: Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?,
        })
    }

    /// Probabilities (summing to 1) that the answer to `question` about the RGB image is each of
    /// `choices` — from the model's likelihood of each choice's first token.
    pub fn choose(&mut self, rgb: &[u8], width: usize, height: usize, question: &str, choices: &[&str]) -> Result<Vec<f32>> {
        let p = self.prefill(rgb, width, height, question)?;
        let firsts: Vec<u32> = choices.iter().map(|c| self.first_token(c)).collect::<Result<_>>()?;
        let logit: Vec<f32> = firsts.iter().map(|&t| p.logits[t as usize]).collect();
        let m = logit.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = logit.iter().map(|l| (l - m).exp()).collect();
        let sum: f32 = e.iter().sum();
        Ok(e.iter().map(|v| v / sum).collect())
    }

    /// Free-text answer (greedy), for diagnostics.
    pub fn answer(&mut self, rgb: &[u8], width: usize, height: usize, question: &str, max_tokens: usize) -> Result<String> {
        let mut p = self.prefill(rgb, width, height, question)?;
        let mut out = Vec::new();
        for _ in 0..max_tokens {
            let next = argmax(&p.logits) as u32;
            if next == IM_END {
                break;
            }
            out.push(next);
            let emb = self.embed_ids(&[next])?;
            let (logits, cache) = self.decode(emb, 1, p.len + 1, vec![p.next_pos; 3], std::mem::take(&mut p.cache))?;
            p.logits = logits;
            p.cache = cache;
            p.next_pos += 1;
            p.len += 1;
        }
        self.tok.decode(&out, true).map_err(|e| anyhow!("{e}"))
    }

    fn first_token(&self, s: &str) -> Result<u32> {
        let enc = self.tok.encode(s, false).map_err(|e| anyhow!("{e}"))?;
        enc.get_ids().first().copied().context("empty choice")
    }

    fn ids(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(s, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec())
    }

    fn embed_ids(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = Tensor::from_array(([1usize, ids.len()], ids.iter().map(|&i| i as i64).collect::<Vec<_>>())).map_err(|e| anyhow!("{e}"))?;
        let out = self.embed.run(ort::inputs!["input_ids" => t]).map_err(|e| anyhow!("{e}"))?;
        let (_, e) = out["inputs_embeds"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        Ok(e.to_vec())
    }

    fn prefill(&mut self, rgb: &[u8], width: usize, height: usize, question: &str) -> Result<Prefill> {
        let (patches, gh, gw) = patchify(rgb, width, height);
        let n_patches = gh * gw;
        let pv = Tensor::from_array(([n_patches, 3 * TEMPORAL * PATCH * PATCH], patches)).map_err(|e| anyhow!("{e}"))?;
        let thw = Tensor::from_array(([1usize, 3], vec![1i64, gh as i64, gw as i64])).map_err(|e| anyhow!("{e}"))?;
        let out = self.vision.run(ort::inputs!["pixel_values" => pv, "image_grid_thw" => thw]).map_err(|e| anyhow!("{e}"))?;
        let (_, feats) = out["image_features"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let feats = feats.to_vec();
        drop(out);
        let (rows, cols) = (gh / MERGE, gw / MERGE);
        let n_img = rows * cols;

        let prefix = self.ids("<|im_start|>user\n<|vision_start|>")?;
        let suffix = self.ids(&format!("<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"))?;
        let mut ids = prefix.clone();
        ids.extend(std::iter::repeat_n(IMAGE_PAD, n_img));
        ids.extend(&suffix);
        let len = ids.len();

        let mut emb = self.embed_ids(&ids)?;
        let dim = emb.len() / len;
        anyhow::ensure!(feats.len() == n_img * dim, "image features {} ≠ {}×{}", feats.len(), n_img, dim);
        let p0 = prefix.len();
        emb[p0 * dim..(p0 + n_img) * dim].copy_from_slice(&feats);

        // Rotary positions: text counts up on all three axes; image tokens share the time index
        // and take their row/column; text after the image resumes past the image's extent.
        let mut pos = vec![0i64; 3 * len];
        for i in 0..p0 {
            for a in 0..3 {
                pos[a * len + i] = i as i64;
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                let i = p0 + r * cols + c;
                pos[i] = p0 as i64;
                pos[len + i] = (p0 + r) as i64;
                pos[2 * len + i] = (p0 + c) as i64;
            }
        }
        let resume = (p0 + rows.max(cols)) as i64;
        for k in 0..suffix.len() {
            let i = p0 + n_img + k;
            for a in 0..3 {
                pos[a * len + i] = resume + k as i64;
            }
        }
        let next_pos = resume + suffix.len() as i64;

        let mut cache = Vec::with_capacity(2 * LAYERS);
        for l in 0..LAYERS {
            for kv in ["key", "value"] {
                let t = Tensor::<f16>::from_array(([1usize, KV_HEADS, 0, HEAD_DIM], Vec::<f16>::new())).map_err(|e| anyhow!("{e}"))?;
                cache.push((format!("present.{l}.{kv}"), t.into_dyn()));
            }
        }
        let (logits, cache) = self.decode(emb, len, len, pos, cache)?;
        Ok(Prefill { logits, cache, next_pos, len })
    }

    /// Run the decoder on `n` new embeddings; returns the last position's logits and the cache.
    fn decode(&mut self, emb: Vec<f32>, n: usize, total: usize, pos: Vec<i64>, cache: Vec<(String, DynValue)>) -> Result<(Vec<f32>, Vec<(String, DynValue)>)> {
        let dim = emb.len() / n;
        let mut inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> = vec![
            ("inputs_embeds".into(), Tensor::from_array(([1usize, n, dim], emb)).map_err(|e| anyhow!("{e}"))?.into()),
            ("attention_mask".into(), Tensor::from_array(([1usize, total], vec![1i64; total])).map_err(|e| anyhow!("{e}"))?.into()),
            ("position_ids".into(), Tensor::from_array(([3usize, 1, n], pos)).map_err(|e| anyhow!("{e}"))?.into()),
        ];
        for (name, v) in cache {
            inputs.push((name.replacen("present", "past_key_values", 1).into(), v.into()));
        }
        let mut out = self.decoder.run(inputs).map_err(|e| anyhow!("{e}"))?;
        let (shape, l) = out["logits"].try_extract_tensor::<f16>().map_err(|e| anyhow!("{e}"))?;
        let vocab = shape[2] as usize;
        let last = l[l.len() - vocab..].iter().map(|v| v.to_f32()).collect();
        let mut cache = Vec::with_capacity(2 * LAYERS);
        for l in 0..LAYERS {
            for kv in ["key", "value"] {
                let name = format!("present.{l}.{kv}");
                let v = out.remove(name.as_str()).context("missing cache output")?;
                cache.push((name, v));
            }
        }
        Ok((last, cache))
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

/// Resize so the longest side is about `SIDE` (sides multiples of 32), normalise to [-1, 1], and
/// cut into patches in the model's order: 2×2 groups of 16-px patches, each patch flattened as
/// (channel, frame, y, x) with the single image used for both frames. Returns (patches, rows,
/// columns) of 16-px patches.
fn patchify(rgb: &[u8], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let unit = (PATCH * MERGE) as f64;
    let s = SIDE / width.max(height).max(1) as f64;
    let w = (((width as f64 * s) / unit).round().max(1.0) * unit) as usize;
    let h = (((height as f64 * s) / unit).round().max(1.0) * unit) as usize;
    let img = image::RgbImage::from_raw(width as u32, height as u32, rgb.to_vec()).expect("rgb size");
    let img = image::imageops::resize(&img, w as u32, h as u32, image::imageops::FilterType::CatmullRom);
    let (gh, gw) = (h / PATCH, w / PATCH);
    let per = 3 * TEMPORAL * PATCH * PATCH;
    let mut out = Vec::with_capacity(gh * gw * per);
    for bh in 0..gh / MERGE {
        for bw in 0..gw / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    let (r, c) = (bh * MERGE + mh, bw * MERGE + mw);
                    for ch in 0..3 {
                        for _ in 0..TEMPORAL {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    let v = img.get_pixel((c * PATCH + px) as u32, (r * PATCH + py) as u32)[ch];
                                    out.push(v as f32 / 127.5 - 1.0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (out, gh, gw)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_follow_the_model_order() {
        // 64×32 image → resized to 448×224 → 28×14 patches.
        let (w, h) = (64usize, 32usize);
        let rgb = vec![128u8; w * h * 3];
        let (p, gh, gw) = patchify(&rgb, w, h);
        assert_eq!((gh, gw), (14, 28));
        assert_eq!(p.len(), gh * gw * 1536);
    }
}
