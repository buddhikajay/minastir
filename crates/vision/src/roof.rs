//! Outline a roof in an aerial photo from one click.
//!
//! Region growing from the clicked pixel on a lightly smoothed copy: pixels join when their colour
//! (chroma) is close to the roof's and their brightness is within a wide band (sunlit and shaded
//! roof faces differ), and the step from their neighbour isn't a sharp edge. Holes (solar panels,
//! skylights) are filled and thin leaks opened away; if the region is too big to be a house the
//! tolerances tighten and we try again. The shape is then squared up: turned to its own dominant
//! direction, rasterised at ~0.5 m, and traced as a right-angled outline (a clean rectangle when
//! it is nearly one).

use crate::onnx::{self, Device};
use anyhow::{anyhow, Result};
use image::RgbImage;
use ort::{session::Session, value::Tensor};
use std::path::Path;

/// Segment Anything (Meta, Apache-2.0), ViT-B, as encoder + decoder ONNX (AnyLabeling export:
/// the encoder takes a raw RGB image up to 1024 × 682 and normalises / pads internally).
pub struct Sam {
    encoder: Session,
    decoder: Session,
    /// Last encoded window (photo id, x0, y0, k) and its embedding — refinements reuse it.
    cache: Option<((u64, i64, i64, i64), Vec<f32>)>,
}

/// An object's outline: a mask over a window of the image (pixels; `x0`, `y0` = window corner).
#[derive(Debug, Clone)]
pub struct BoxMask {
    pub x0: f64,
    pub y0: f64,
    pub width: usize,
    pub height: usize,
    pub cells: Vec<bool>,
}

/// A prompt point in photo pixels: include (true) or exclude (false).
#[derive(Debug, Clone, Copy)]
pub struct Prompt {
    pub u: f64,
    pub v: f64,
    pub include: bool,
}

pub const SAM_ENCODER: &str = "sam_vit_b.encoder.onnx";
pub const SAM_DECODER: &str = "sam_vit_b.decoder.onnx";
const SAM_W: usize = 1024;
const SAM_H: usize = 682;
/// Width (metres) of the photo window around a click: 72 × 48 m holds even a long house with
/// room around it, at ~7 cm per model pixel.
const WINDOW_M: f64 = 72.0;

impl Sam {
    pub fn load(models: &Path) -> Result<Self> {
        Ok(Self {
            encoder: onnx::session(&models.join(SAM_ENCODER), Device::Cpu, &[])?,
            decoder: onnx::session(&models.join(SAM_DECODER), Device::Cpu, &[])?,
            cache: None,
        })
    }

    /// One click: `outline_prompts` with a single include point.
    pub fn outline(&mut self, img: &RgbImage, seed: (f64, f64), scale: f64) -> Result<Option<Vec<[f64; 2]>>> {
        self.outline_prompts(img, 0, &[Prompt { u: seed.0, v: seed.1, include: true }], scale)
    }

    /// Roof outline (image pixels) from prompt points; the first point anchors the window
    /// (72 × 48 m, enlarged to the model input). `photo` identifies the image for caching.
    pub fn outline_prompts(&mut self, img: &RgbImage, photo: u64, prompts: &[Prompt], scale: f64) -> Result<Option<Vec<[f64; 2]>>> {
        let Some(first) = prompts.first() else { return Ok(None) };
        let seed = (first.u, first.v);
        let (iw, ih) = (img.width() as f64, img.height() as f64);
        if seed.0 < 0.0 || seed.1 < 0.0 || seed.0 >= iw || seed.1 >= ih {
            return Ok(None);
        }
        // Window in photo pixels, aspect of the model input.
        let win_w = (WINDOW_M / scale).min(iw);
        let win_h = (win_w * SAM_H as f64 / SAM_W as f64).min(ih);
        let x0 = (seed.0 - win_w / 2.0).clamp(0.0, (iw - win_w).max(0.0));
        let y0 = (seed.1 - win_h / 2.0).clamp(0.0, (ih - win_h).max(0.0));
        let k = (SAM_W as f64 / win_w).min(SAM_H as f64 / win_h); // window px → model px
        let key = (photo, x0.round() as i64, y0.round() as i64, (k * 1000.0).round() as i64);
        let emb = match &self.cache {
            Some((c, e)) if *c == key => e.clone(),
            _ => {
                let e = self.encode(img, x0, y0, win_w, win_h, k)?;
                self.cache = Some((key, e.clone()));
                e
            }
        };
        self.decode(img, emb, prompts, scale, (x0, y0, win_w, win_h, k))
    }

    /// The outline of the object in a box (image pixels x0, y0, x1, y1) as a mask over a window
    /// around it. `photo` identifies the image for caching.
    pub fn box_mask(&mut self, img: &RgbImage, photo: u64, b: [f64; 4]) -> Result<BoxMask> {
        let (iw, ih) = (img.width() as f64, img.height() as f64);
        let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
        // Window: the box with a margin, shaped like the model input.
        let mut win_w = (bw * 1.4).max(bh * 1.4 * SAM_W as f64 / SAM_H as f64);
        let mut win_h = win_w * SAM_H as f64 / SAM_W as f64;
        win_w = win_w.min(iw);
        win_h = win_h.min(ih);
        let cx = (b[0] + b[2]) / 2.0;
        let cy = (b[1] + b[3]) / 2.0;
        let x0 = (cx - win_w / 2.0).clamp(0.0, (iw - win_w).max(0.0));
        let y0 = (cy - win_h / 2.0).clamp(0.0, (ih - win_h).max(0.0));
        let k = (SAM_W as f64 / win_w).min(SAM_H as f64 / win_h);
        let key = (photo, x0.round() as i64, y0.round() as i64, (k * 1000.0).round() as i64);
        let emb = match &self.cache {
            Some((c, e)) if *c == key => e.clone(),
            _ => {
                let e = self.encode(img, x0, y0, win_w, win_h, k)?;
                self.cache = Some((key, e.clone()));
                e
            }
        };
        let to = |u: f64, v: f64| [((u - x0) * k) as f32, ((v - y0) * k) as f32];
        let (p, q) = (to(b[0], b[1]), to(b[2], b[3]));
        let inputs = ort::inputs![
            "image_embeddings" => Tensor::from_array(([1usize, 256, 64, 64], emb)).map_err(|e| anyhow!("{e}"))?,
            "point_coords" => Tensor::from_array(([1usize, 2, 2], vec![p[0], p[1], q[0], q[1]])).map_err(|e| anyhow!("{e}"))?,
            "point_labels" => Tensor::from_array(([1usize, 2], vec![2f32, 3.0])).map_err(|e| anyhow!("{e}"))?,
            "mask_input" => Tensor::from_array(([1usize, 1, 256, 256], vec![0f32; 256 * 256])).map_err(|e| anyhow!("{e}"))?,
            "has_mask_input" => Tensor::from_array(([1usize], vec![0f32])).map_err(|e| anyhow!("{e}"))?,
            "orig_im_size" => Tensor::from_array(([2usize], vec![SAM_H as f32, SAM_W as f32])).map_err(|e| anyhow!("{e}"))?
        ];
        let out = self.decoder.run(inputs).map_err(|e| anyhow!("{e}"))?;
        let (shape, masks) = out["masks"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (mh, mw) = (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize);
        let (ww, wh) = (win_w.ceil() as usize, win_h.ceil() as usize);
        let mut cells = vec![false; ww * wh];
        for y in 0..wh {
            for x in 0..ww {
                let (mx, my) = (((x as f64 + 0.5) * k) as usize, ((y as f64 + 0.5) * k) as usize);
                if mx < mw && my < mh {
                    cells[y * ww + x] = masks[my * mw + mx] > 0.0;
                }
            }
        }
        Ok(BoxMask { x0, y0, width: ww, height: wh, cells })
    }

    fn encode(&mut self, img: &RgbImage, x0: f64, y0: f64, win_w: f64, win_h: f64, k: f64) -> Result<Vec<f32>> {
        let mut input = vec![0f32; SAM_H * SAM_W * 3];
        for y in 0..SAM_H {
            for x in 0..SAM_W {
                let (u, v) = (x0 + (x as f64 + 0.5) / k - 0.5, y0 + (y as f64 + 0.5) / k - 0.5);
                if u < x0 - 0.5 || v < y0 - 0.5 || u > x0 + win_w || v > y0 + win_h {
                    continue; // padding
                }
                let p = sample(img, u, v);
                let o = (y * SAM_W + x) * 3;
                input[o..o + 3].copy_from_slice(&p);
            }
        }
        let t = Tensor::from_array(([SAM_H, SAM_W, 3usize], input)).map_err(|e| anyhow!("{e}"))?;
        let out = self.encoder.run(ort::inputs!["input_image" => t]).map_err(|e| anyhow!("{e}"))?;
        let (_, emb) = out["image_embeddings"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        Ok(emb.to_vec())
    }

    fn decode(
        &mut self,
        img: &RgbImage,
        emb: Vec<f32>,
        prompts: &[Prompt],
        scale: f64,
        win: (f64, f64, f64, f64, f64),
    ) -> Result<Option<Vec<[f64; 2]>>> {
        let (x0, y0, _, _, k) = win;
        let seed = (prompts[0].u, prompts[0].v);
        let to = |u: f64, v: f64| [((u - x0) * k) as f32, ((v - y0) * k) as f32];
        let user: Vec<(f32, f32, f32)> = prompts
            .iter()
            .map(|p| {
                let q = to(p.u, p.v);
                (q[0], q[1], if p.include { 1.0 } else { 0.0 })
            })
            .collect();
        // Candidate prompts. Once the user has refined with more points, use exactly theirs.
        // For a lone click — ambiguous between one roof face and the whole house, and between the
        // roof and whatever is next to it — try a few and keep the most roof-like answer.
        let mut cands: Vec<Vec<(f32, f32, f32)>> = vec![user.clone()];
        if prompts.len() == 1 {
            // Click + rough box from colour growing (brackets the roof; may overshoot).
            if let Some(rough) = outline(img, seed, scale) {
                let (mut bx0, mut by0, mut bx1, mut by1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                for p in &rough {
                    (bx0, by0, bx1, by1) = (bx0.min(p[0]), by0.min(p[1]), bx1.max(p[0]), by1.max(p[1]));
                }
                let lim = 20.0 / scale;
                let (a, b) = (to(bx0.max(seed.0 - lim), by0.max(seed.1 - lim)), to(bx1.min(seed.0 + lim), by1.min(seed.1 + lim)));
                cands.push(vec![user[0], (a[0], a[1], 2.0), (b[0], b[1], 3.0)]);
            }
            // Click + nearby points of the same roof colour (spreads the prompt over the roof).
            let near = same_colour_points(img, seed, scale);
            if !near.is_empty() {
                let mut c = user.clone();
                c.extend(near.iter().map(|&(u, v)| {
                    let q = to(u, v);
                    (q[0], q[1], 1.0)
                }));
                cands.push(c);
            }
        }
        let mut best: Option<(f64, f64, Vec<(usize, usize)>)> = None; // (uniformity, area, region)
        let mut found = Vec::new();
        for c in &cands {
            let Some(raw) = self.run(&emb, c, seed, scale, win)? else { continue };
            let uni = uniformity(img, &raw, (x0, y0));
            // Trim spill-over (a strip of path or lawn along an edge) by colour, then keep the
            // piece holding the click.
            let Some(region) = trim_to_roof(img, &raw, seed, scale, win) else { continue };
            let area = region.len() as f64 * scale * scale;
            if !(MIN_AREA..=MAX_AREA).contains(&area) || uni < 0.8 {
                continue;
            }
            if std::env::var("MINASTIR_DEBUG_SAM").is_ok() {
                eprintln!("  candidate with {} prompts: {area:.0} m², uniformity {uni:.3}", c.len());
            }
            found.push((uni, area, region));
        }
        // Debug: force one candidate (index into those found).
        if let Some(i) = std::env::var("MINASTIR_SAM_PICK").ok().and_then(|v| v.parse::<usize>().ok()) {
            if i < found.len() {
                let region = found.swap_remove(i).2;
                let poly = square_up(&region, scale);
                return Ok(Some(poly.into_iter().map(|[u, v]| [u + x0, v + y0]).collect()));
            }
        }
        // After trimming, every candidate is roof-coloured; the largest is the whole house
        // rather than one face of it.
        for f in found {
            if best.as_ref().is_none_or(|b| f.1 > b.1) {
                best = Some(f);
            }
        }
        let Some((_, _, region)) = best else { return Ok(None) };
        let poly = square_up(&region, scale);
        Ok(Some(poly.into_iter().map(|[u, v]| [u + x0, v + y0]).collect()))
    }

    /// Run the decoder for one prompt set; the cleaned-up region (window pixels) holding the seed.
    fn run(
        &mut self,
        emb: &[f32],
        prompt: &[(f32, f32, f32)],
        seed: (f64, f64),
        scale: f64,
        (x0, y0, win_w, win_h, k): (f64, f64, f64, f64, f64),
    ) -> Result<Option<Vec<(usize, usize)>>> {
        let mut coords: Vec<f32> = prompt.iter().flat_map(|p| [p.0, p.1]).collect();
        let mut labels: Vec<f32> = prompt.iter().map(|p| p.2).collect();
        if !labels.contains(&2.0) {
            coords.extend([0.0, 0.0]); // padding point the export expects without a box
            labels.push(-1.0);
        }
        let n = labels.len();
        let inputs = ort::inputs![
            "image_embeddings" => Tensor::from_array(([1usize, 256, 64, 64], emb.to_vec())).map_err(|e| anyhow!("{e}"))?,
            "point_coords" => Tensor::from_array(([1usize, n, 2], coords)).map_err(|e| anyhow!("{e}"))?,
            "point_labels" => Tensor::from_array(([1usize, n], labels)).map_err(|e| anyhow!("{e}"))?,
            "mask_input" => Tensor::from_array(([1usize, 1, 256, 256], vec![0f32; 256 * 256])).map_err(|e| anyhow!("{e}"))?,
            "has_mask_input" => Tensor::from_array(([1usize], vec![0f32])).map_err(|e| anyhow!("{e}"))?,
            "orig_im_size" => Tensor::from_array(([2usize], vec![SAM_H as f32, SAM_W as f32])).map_err(|e| anyhow!("{e}"))?
        ];
        let out = self.decoder.run(inputs).map_err(|e| anyhow!("{e}"))?;
        let (shape, masks) = out["masks"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        let (mh, mw) = (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize);
        // Back to photo pixels at the photo's own resolution (within the window).
        let (ww, wh) = (win_w.ceil() as usize, win_h.ceil() as usize);
        let mut mask = vec![false; ww * wh];
        for y in 0..wh {
            for x in 0..ww {
                let (mx, my) = (((x as f64 + 0.5) * k) as usize, ((y as f64 + 0.5) * k) as usize);
                if mx < mw && my < mh {
                    mask[y * ww + x] = masks[my * mw + mx] > 0.0;
                }
            }
        }
        fill_holes(&mut mask, ww, wh);
        // Cut away strips narrower than ~1.6 m (side paths, fence lines).
        let r = ((0.8 / scale).round() as usize).clamp(1, 10);
        let mask = open(&mask, ww, wh, r);
        let seed_w = (((seed.0 - x0).max(0.0) as usize).min(ww - 1), ((seed.1 - y0).max(0.0) as usize).min(wh - 1));
        Ok(component(&mask, ww, wh, seed_w))
    }
}

/// Up to ten points 3–9 m from the click whose colour matches the clicked roof: chroma tight
/// (keeps lawn, brown paths and other roofing out), brightness loose (sunlit / shaded faces).
fn same_colour_points(img: &RgbImage, seed: (f64, f64), scale: f64) -> Vec<(f64, f64)> {
    let s = lab(sample(img, seed.0, seed.1));
    let mut out = Vec::new();
    for (ring, r) in [3.0, 6.0, 9.0].into_iter().enumerate() {
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::FRAC_PI_4 + ring as f64 * 0.39;
            let (u, v) = (seed.0 + a.cos() * r / scale, seed.1 + a.sin() * r / scale);
            if u < 0.0 || v < 0.0 || u >= img.width() as f64 || v >= img.height() as f64 {
                continue;
            }
            // Average a small patch so tile texture doesn't decide.
            let mut acc = [0f32; 3];
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let p = sample(img, u + dx as f64, v + dy as f64);
                    for c in 0..3 {
                        acc[c] += p[c] / 25.0;
                    }
                }
            }
            let q = lab(acc);
            if (q[1] - s[1]).hypot(q[2] - s[2]) < 9.0 && (q[0] - s[0]).abs() < 55.0 {
                out.push((u, v));
            }
        }
    }
    out.truncate(10);
    out
}

fn roof_like(q: [f32; 3], m: [f32; 3]) -> bool {
    ((q[1] - m[1]).hypot(q[2] - m[2]) < 14.0 && (q[0] - m[0]).abs() < 60.0) || (q[0] < 75.0 && q[2] < 6.0 && q[1].abs() < 18.0)
}

/// Keep only the region's pixels that match its own typical colour, cut narrow bits, and return
/// the piece containing the seed (window pixels).
fn trim_to_roof(
    img: &RgbImage,
    region: &[(usize, usize)],
    seed: (f64, f64),
    scale: f64,
    (x0, y0, win_w, win_h, _): (f64, f64, f64, f64, f64),
) -> Option<Vec<(usize, usize)>> {
    let (ww, wh) = (win_w.ceil() as usize, win_h.ceil() as usize);
    let col: Vec<[f32; 3]> = region.iter().map(|&(x, y)| lab(sample(img, x as f64 + x0, y as f64 + y0))).collect();
    let med = |k: usize| {
        let mut v: Vec<f32> = col.iter().map(|p| p[k]).collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let m = [med(0), med(1), med(2)];
    let mut mask = vec![false; ww * wh];
    for (&(x, y), q) in region.iter().zip(&col) {
        if roof_like(*q, m) {
            mask[y * ww + x] = true;
        }
    }
    let rc = ((0.4 / scale).round() as usize).clamp(1, 6);
    let mut mask = close(&mask, ww, wh, rc);
    fill_holes(&mut mask, ww, wh);
    let r = ((0.8 / scale).round() as usize).clamp(1, 10);
    let mask = open(&mask, ww, wh, r);
    let seed_w = (((seed.0 - x0).max(0.0) as usize).min(ww - 1), ((seed.1 - y0).max(0.0) as usize).min(wh - 1));
    component(&mask, ww, wh, seed_w)
}

/// Share of a region's pixels whose colour matches the region's own typical colour (chroma
/// close; brightness within a wide band for sunlit / shaded faces; solar panels count as roof).
/// Roofs are uniform; outlines that spill onto paths, lawn or a neighbour score lower.
fn uniformity(img: &RgbImage, region: &[(usize, usize)], (x0, y0): (f64, f64)) -> f64 {
    let px: Vec<[f32; 3]> = region
        .iter()
        .step_by((region.len() / 4000).max(1))
        .map(|&(x, y)| lab(sample(img, x as f64 + x0, y as f64 + y0)))
        .collect();
    let med = |k: usize| {
        let mut v: Vec<f32> = px.iter().map(|p| p[k]).collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    };
    let m = [med(0), med(1), med(2)];
    let ok = px.iter().filter(|q| roof_like(**q, m)).count();
    ok as f64 / px.len().max(1) as f64
}

/// Bilinear RGB sample (0–255 floats).
fn sample(img: &RgbImage, u: f64, v: f64) -> [f32; 3] {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (x0, y0) = (u.floor() as i64, v.floor() as i64);
    let (tx, ty) = ((u - x0 as f64) as f32, (v - y0 as f64) as f32);
    let px = |x: i64, y: i64| img.get_pixel(x.clamp(0, w - 1) as u32, y.clamp(0, h - 1) as u32).0.map(|c| c as f32);
    let (a, b, c, d) = (px(x0, y0), px(x0 + 1, y0), px(x0, y0 + 1), px(x0 + 1, y0 + 1));
    [0, 1, 2].map(|i| (a[i] * (1.0 - tx) + b[i] * tx) * (1.0 - ty) + (c[i] * (1.0 - tx) + d[i] * tx) * ty)
}

/// Plausible single-house roof area, m².
const MAX_AREA: f64 = 600.0;
const MIN_AREA: f64 = 25.0;

/// Roof outline (image pixel coordinates, (u, v)) around `seed`; `scale` is metres per pixel.
pub fn outline(img: &RgbImage, seed: (f64, f64), scale: f64) -> Option<Vec<[f64; 2]>> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let (sx, sy) = (seed.0.round() as i64, seed.1.round() as i64);
    if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
        return None;
    }
    let smooth = blur(img);
    let px_area = scale * scale;
    // Seed colour: median of a small window.
    let mut win: Vec<[f32; 3]> = Vec::new();
    for dy in -3..=3 {
        for dx in -3..=3 {
            let (x, y) = (sx + dx, sy + dy);
            if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 {
                win.push(smooth[y as usize * w + x as usize]);
            }
        }
    }
    let seed_c = median3(&mut win);
    // Try progressively tighter tolerances until the region is house-sized.
    for &(chroma, bright, step) in &[(22.0f32, 70.0f32, 26.0f32), (16.0, 55.0, 20.0), (11.0, 40.0, 15.0), (8.0, 30.0, 11.0)] {
        let reach = (25.0 / scale) as i64;
        let mut mask = grow(&smooth, w, h, (sx as usize, sy as usize), seed_c, chroma, bright, step, (MAX_AREA * 1.5 / px_area) as usize, reach);
        // Close small gaps (~0.6 m), fill holes (panels, skylights), then cut away anything
        // narrower than ~2 m — side paths and fence lines, never part of a house.
        let rc = ((0.6 / scale).round() as usize).clamp(1, 8);
        mask = close(&mask, w, h, rc);
        fill_holes(&mut mask, w, h);
        let ro = ((1.0 / scale).round() as usize).clamp(1, 12);
        mask = open(&mask, w, h, ro);
        let keep = component(&mask, w, h, (sx as usize, sy as usize));
        let Some(region) = keep else { continue };
        let area = region.len() as f64 * px_area;
        if area > MAX_AREA {
            continue;
        }
        if area < MIN_AREA {
            return None;
        }
        return Some(square_up(&region, scale));
    }
    None
}

fn blur(img: &RgbImage) -> Vec<[f32; 3]> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let src: Vec<[f32; 3]> = img.pixels().map(|p| p.0.map(|c| c as f32)).collect();
    // 3×3 box, twice.
    let pass = |s: &[[f32; 3]]| -> Vec<[f32; 3]> {
        let mut o = vec![[0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0f32; 3];
                let mut n = 0.0;
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                        if xx >= 0 && yy >= 0 && xx < w as i64 && yy < h as i64 {
                            let p = s[yy as usize * w + xx as usize];
                            for k in 0..3 {
                                acc[k] += p[k];
                            }
                            n += 1.0;
                        }
                    }
                }
                o[y * w + x] = acc.map(|a| a / n);
            }
        }
        o
    };
    pass(&pass(&src))
}

fn median3(v: &mut [[f32; 3]]) -> [f32; 3] {
    let mut out = [0f32; 3];
    for k in 0..3 {
        let mut c: Vec<f32> = v.iter().map(|p| p[k]).collect();
        c.sort_by(f32::total_cmp);
        out[k] = c[c.len() / 2];
    }
    out
}

/// Brightness and two chroma axes.
fn lab(p: [f32; 3]) -> [f32; 3] {
    [(p[0] + p[1] + p[2]) / 3.0, p[0] - p[1], (p[0] + p[1]) / 2.0 - p[2]]
}

#[allow(clippy::too_many_arguments)]
fn grow(
    img: &[[f32; 3]],
    w: usize,
    h: usize,
    seed: (usize, usize),
    seed_c: [f32; 3],
    chroma: f32,
    bright: f32,
    step: f32,
    limit: usize,
    reach: i64,
) -> Vec<bool> {
    let s = lab(seed_c);
    // Roof-coloured, or a solar panel (dark, neutral to bluish) lying on the roof.
    let panel = |q: [f32; 3]| q[0] < 75.0 && q[2] < 6.0 && q[1].abs() < 18.0;
    let ok = |p: [f32; 3]| {
        let q = lab(p);
        ((q[1] - s[1]).hypot(q[2] - s[2]) < chroma && (q[0] - s[0]).abs() < bright) || panel(q)
    };
    let mut mask = vec![false; w * h];
    let mut stack = vec![seed];
    mask[seed.1 * w + seed.0] = true;
    let mut count = 1;
    while let Some((x, y)) = stack.pop() {
        let here = lab(img[y * w + x]);
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (xx, yy) = (x as i64 + dx, y as i64 + dy);
            if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                continue;
            }
            if (xx - seed.0 as i64).abs() > reach || (yy - seed.1 as i64).abs() > reach {
                continue;
            }
            let i = yy as usize * w + xx as usize;
            if mask[i] {
                continue;
            }
            let p = img[i];
            let q = lab(p);
            let edge_ok = (q[0] - here[0]).abs() < step || panel(q) || panel(here);
            if ok(p) && edge_ok {
                mask[i] = true;
                count += 1;
                if count > limit {
                    return mask; // too big; caller tightens
                }
                stack.push((xx as usize, yy as usize));
            }
        }
    }
    mask
}

fn dilate(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut a = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if m[y * w + x] {
                for xx in x.saturating_sub(r)..=(x + r).min(w - 1) {
                    a[y * w + xx] = true;
                }
            }
        }
    }
    let mut b = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            if a[y * w + x] {
                for yy in y.saturating_sub(r)..=(y + r).min(h - 1) {
                    b[yy * w + x] = true;
                }
            }
        }
    }
    b
}

fn erode(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let inv: Vec<bool> = m.iter().map(|v| !v).collect();
    dilate(&inv, w, h, r).into_iter().map(|v| !v).collect()
}

fn close(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    erode(&dilate(m, w, h, r), w, h, r)
}

fn open(m: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    dilate(&erode(m, w, h, r), w, h, r)
}

/// Fill everything not reachable from the image border through empty pixels.
fn fill_holes(m: &mut [bool], w: usize, h: usize) {
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for x in 0..w {
        stack.extend([x, (h - 1) * w + x]);
    }
    for y in 0..h {
        stack.extend([y * w, y * w + w - 1]);
    }
    while let Some(i) = stack.pop() {
        if outside[i] || m[i] {
            continue;
        }
        outside[i] = true;
        let (x, y) = (i % w, i / w);
        if x > 0 {
            stack.push(i - 1);
        }
        if x + 1 < w {
            stack.push(i + 1);
        }
        if y > 0 {
            stack.push(i - w);
        }
        if y + 1 < h {
            stack.push(i + w);
        }
    }
    for i in 0..w * h {
        if !outside[i] {
            m[i] = true;
        }
    }
}

/// Pixels of the connected region containing (or nearest to) the seed.
fn component(m: &[bool], w: usize, h: usize, seed: (usize, usize)) -> Option<Vec<(usize, usize)>> {
    let mut start = None;
    'find: for r in 0..12i64 {
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (seed.0 as i64 + dx, seed.1 as i64 + dy);
                if x >= 0 && y >= 0 && x < w as i64 && y < h as i64 && m[y as usize * w + x as usize] {
                    start = Some(y as usize * w + x as usize);
                    break 'find;
                }
            }
        }
    }
    let start = start?;
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = vec![start];
    seen[start] = true;
    while let Some(i) = stack.pop() {
        out.push((i % w, i / w));
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize| {
            if m[j] && !seen[j] {
                seen[j] = true;
                stack.push(j);
            }
        };
        if x > 0 {
            push(i - 1);
        }
        if x + 1 < w {
            push(i + 1);
        }
        if y > 0 {
            push(i - w);
        }
        if y + 1 < h {
            push(i + w);
        }
    }
    Some(out)
}

/// Right-angled outline of a pixel region, in image pixels.
fn square_up(region: &[(usize, usize)], scale: f64) -> Vec<[f64; 2]> {
    // Dominant direction: angle (mod 90°) where the region's boundary-aligned bounding box is
    // smallest — the minimum-area rectangle over a sweep.
    let pts: Vec<[f64; 2]> = region.iter().map(|&(x, y)| [x as f64 + 0.5, y as f64 + 0.5]).collect();
    let area_px = pts.len() as f64;
    let bbox = |th: f64| {
        let (s, c) = th.sin_cos();
        let (mut a0, mut a1, mut b0, mut b1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for p in &pts {
            let (a, b) = (p[0] * c + p[1] * s, -p[0] * s + p[1] * c);
            (a0, a1, b0, b1) = (a0.min(a), a1.max(a), b0.min(b), b1.max(b));
        }
        (a0, a1, b0, b1)
    };
    let mut best = (0.0, f64::MAX);
    for k in 0..180 {
        let th = (k as f64 * 0.5).to_radians();
        let (a0, a1, b0, b1) = bbox(th);
        let ar = (a1 - a0) * (b1 - b0);
        if ar < best.1 {
            best = (th, ar);
        }
    }
    let th = best.0;
    let (s, c) = th.sin_cos();
    let back = |a: f64, b: f64| [a * c - b * s, a * s + b * c];
    let (a0, a1, b0, b1) = bbox(th);
    // Nearly a rectangle: return it.
    if area_px / best.1 > 0.92 {
        return vec![back(a0, b0), back(a1, b0), back(a1, b1), back(a0, b1)];
    }
    // Otherwise a right-angled outline on a grid of ~0.5 m in the rotated frame.
    let cell = (0.5 / scale).max(1.0);
    let (nw, nh) = (((a1 - a0) / cell).ceil() as usize + 1, ((b1 - b0) / cell).ceil() as usize + 1);
    let mut cnt = vec![0u32; nw * nh];
    for p in &pts {
        let (a, b) = (p[0] * c + p[1] * s, -p[0] * s + p[1] * c);
        let (i, j) = (((a - a0) / cell) as usize, ((b - b0) / cell) as usize);
        cnt[j.min(nh - 1) * nw + i.min(nw - 1)] += 1;
    }
    let full = cell * cell;
    let mut grid: Vec<bool> = cnt.iter().map(|&n| n as f64 > 0.5 * full).collect();
    // Remove 1-cell notches: close then open.
    grid = open(&close(&grid, nw, nh, 1), nw, nh, 1);
    let Some(lp) = trace(&grid, nw, nh) else {
        return vec![back(a0, b0), back(a1, b0), back(a1, b1), back(a0, b1)];
    };
    let lp = drop_short_edges(lp, (1.5 / (cell * scale)).ceil() as i64);
    lp.into_iter().map(|(i, j)| back(a0 + i as f64 * cell, b0 + j as f64 * cell)).collect()
}

/// Remove edges shorter than `min` cells from a right-angled loop: the short edge's two
/// neighbours are parallel; slide the shorter neighbour onto the longer one's line.
fn drop_short_edges(mut lp: Vec<(i64, i64)>, min: i64) -> Vec<(i64, i64)> {
    let len = |a: (i64, i64), b: (i64, i64)| (a.0 - b.0).abs() + (a.1 - b.1).abs();
    loop {
        let n = lp.len();
        if n <= 4 {
            return lp;
        }
        let Some(k) = (0..n).filter(|&k| len(lp[k], lp[(k + 1) % n]) < min).min_by_key(|&k| len(lp[k], lp[(k + 1) % n])) else {
            return lp;
        };
        let (ip, i0, i1, i2) = ((k + n - 1) % n, k, (k + 1) % n, (k + 2) % n);
        let d = (lp[i1].0 - lp[i0].0, lp[i1].1 - lp[i0].1); // the short edge
        if len(lp[ip], lp[i0]) >= len(lp[i1], lp[i2]) {
            // Keep the previous edge's line: move the next edge back by −d.
            lp[i2] = (lp[i2].0 - d.0, lp[i2].1 - d.1);
            lp[i1] = lp[i0];
        } else {
            // Keep the next edge's line: move the previous edge forward by +d.
            lp[ip] = (lp[ip].0 + d.0, lp[ip].1 + d.1);
            lp[i0] = lp[i1];
        }
        // Remove duplicates and collinear points.
        lp.dedup();
        if lp.len() > 1 && lp.first() == lp.last() {
            lp.pop();
        }
        let m = lp.len();
        lp = (0..m)
            .filter(|&q| {
                let (a, b, c) = (lp[(q + m - 1) % m], lp[q], lp[(q + 1) % m]);
                (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0) != 0
            })
            .map(|q| lp[q])
            .collect();
    }
}

/// Outer boundary of the filled cells as a loop of grid corners, collinear points removed.
fn trace(g: &[bool], w: usize, h: usize) -> Option<Vec<(i64, i64)>> {
    let at = |i: i64, j: i64| i >= 0 && j >= 0 && (i as usize) < w && (j as usize) < h && g[j as usize * w + i as usize];
    // Directed boundary edges with the filled cell on the right (clockwise in image coordinates).
    let mut next: std::collections::HashMap<(i64, i64), Vec<(i64, i64)>> = Default::default();
    let mut edges = 0;
    for j in 0..h as i64 {
        for i in 0..w as i64 {
            if !at(i, j) {
                continue;
            }
            if !at(i, j - 1) {
                next.entry((i, j)).or_default().push((i + 1, j));
                edges += 1;
            }
            if !at(i + 1, j) {
                next.entry((i + 1, j)).or_default().push((i + 1, j + 1));
                edges += 1;
            }
            if !at(i, j + 1) {
                next.entry((i + 1, j + 1)).or_default().push((i, j + 1));
                edges += 1;
            }
            if !at(i - 1, j) {
                next.entry((i, j + 1)).or_default().push((i, j));
                edges += 1;
            }
        }
    }
    // Start at the top-left-most corner (on the outer boundary).
    let start = *next.keys().min_by_key(|k| (k.1, k.0))?;
    let mut lp = vec![start];
    let mut cur = start;
    for _ in 0..edges {
        let n = next.get_mut(&cur)?.pop()?;
        if n == start {
            break;
        }
        lp.push(n);
        cur = n;
    }
    // Drop collinear corners.
    let n = lp.len();
    let simp: Vec<(i64, i64)> = (0..n)
        .filter(|&k| {
            let (p, q, r) = (lp[(k + n - 1) % n], lp[k], lp[(k + 1) % n]);
            (q.0 - p.0) * (r.1 - q.1) - (q.1 - p.1) * (r.0 - q.0) != 0
        })
        .map(|k| lp[k])
        .collect();
    (simp.len() >= 4).then_some(simp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlines_a_rectangular_roof_with_two_faces_and_panels() {
        // Grass background; a 120×80 px grey roof whose two halves differ in brightness (sun),
        // with dark solar panels on it; 0.1 m/px.
        let img = RgbImage::from_fn(300, 240, |x, y| {
            let inside = (90..210).contains(&x) && (80..160).contains(&y);
            if !inside {
                return image::Rgb([90, 150, 70]);
            }
            if (110..140).contains(&x) && (90..110).contains(&y) {
                return image::Rgb([30, 35, 50]); // panels
            }
            if y < 120 { image::Rgb([150, 150, 152]) } else { image::Rgb([110, 110, 113]) }
        });
        let poly = outline(&img, (180.0, 130.0), 0.1).expect("roof");
        assert_eq!(poly.len(), 4, "{poly:?}");
        let xs: Vec<f64> = poly.iter().map(|p| p[0]).collect();
        let ys: Vec<f64> = poly.iter().map(|p| p[1]).collect();
        let (x0, x1) = (xs.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max));
        let (y0, y1) = (ys.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MIN, f64::max));
        assert!((x0 - 90.0).abs() < 4.0 && (x1 - 210.0).abs() < 4.0, "{poly:?}");
        assert!((y0 - 80.0).abs() < 4.0 && (y1 - 160.0).abs() < 4.0, "{poly:?}");
    }

    #[test]
    fn labels_go_long_paths_stay() {
        // Grey roof; "letters" = short white 2-px strokes with a dark outline; a long white path.
        let img = RgbImage::from_fn(400, 300, |x, y| {
            let letter = (0..6).any(|k| {
                let lx = 100 + k * 14;
                (lx..lx + 2).contains(&x) && (100..112).contains(&y)
            });
            let outline = (0..6).any(|k| {
                let lx = 100 + k * 14;
                (lx - 2..lx + 4).contains(&x) && (98..114).contains(&y)
            });
            if (250..252).contains(&x) {
                image::Rgb([240, 240, 240]) // long path
            } else if letter {
                image::Rgb([250, 250, 250])
            } else if outline {
                image::Rgb([40, 40, 40])
            } else {
                image::Rgb([120, 122, 125])
            }
        });
        let out = remove_labels(&img);
        let grey = |p: &image::Rgb<u8>| (p.0[0] as i32 - 120).abs() < 25;
        assert!(grey(out.get_pixel(100, 105)) && grey(out.get_pixel(99, 99)), "{:?}", out.get_pixel(100, 105));
        assert!(out.get_pixel(250, 150).0[0] > 220, "path kept");
    }

    #[test]
    fn l_shaped_roof_keeps_its_corner() {
        let img = RgbImage::from_fn(300, 300, |x, y| {
            let inside = ((60..240).contains(&x) && (60..140).contains(&y)) || ((60..140).contains(&x) && (60..240).contains(&y));
            if inside { image::Rgb([200, 120, 90]) } else { image::Rgb([90, 150, 70]) }
        });
        let poly = outline(&img, (100.0, 100.0), 0.1).expect("roof");
        assert_eq!(poly.len(), 6, "{poly:?}");
    }
}

/// Remove map labels (street names, place names, house numbers: thin white text with a dark
/// halo, drawn over satellite screenshots) by painting them over from their surroundings. The
/// text looks like hard edges to segmentation and splits roofs in two.
pub fn remove_labels(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bright: Vec<bool> = img
        .pixels()
        .map(|p| {
            let [r, g, b] = p.0.map(|c| c as f32);
            let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
            mx > 215.0 && (mx - mn) / mx.max(1.0) < 0.15
        })
        .collect();
    // Thin bright strokes: bright pixels that vanish under an opening (wide things — white
    // roofs, cars — survive it). Stroke width scales with how the screenshot was taken.
    let r = ((w.max(h) as f64 / 900.0).round() as usize).clamp(2, 5);
    let wide = open(&bright, w, h, r);
    let thin: Vec<bool> = bright.iter().zip(&wide).map(|(b, o)| *b && !*o).collect();
    // Letters are short, separate strokes; paths, fences and gutters are long thin lines that
    // separate roofs — keep those. Drop thin pieces longer than a character.
    let max_len = 10 * r;
    let mut keep = vec![false; w * h];
    let mut seen = vec![false; w * h];
    for start in 0..w * h {
        if !thin[start] || seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut comp = Vec::new();
        seen[start] = true;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
        while let Some(i) = stack.pop() {
            comp.push(i);
            let (x, y) = (i % w, i / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                if xx >= 0 && yy >= 0 && xx < w as i64 && yy < h as i64 {
                    let j = yy as usize * w + xx as usize;
                    if thin[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        if (x1 - x0).max(y1 - y0) <= max_len {
            for i in comp {
                keep[i] = true;
            }
        }
    }
    let thin = keep;
    // Take in the dark halo around the letters (but never paint over wide white areas).
    // Labels have a dark outline a few pixels wide around the white strokes.
    let dark: Vec<bool> = img.pixels().map(|p| (p.0[0] as u32 + p.0[1] as u32 + p.0[2] as u32) < 3 * 95).collect();
    let near = dilate(&thin, w, h, r);
    let halo = dilate(&thin, w, h, 2 * r + 2);
    let mask: Vec<bool> = (0..w * h).map(|i| near[i] || (halo[i] && dark[i])).collect();
    let mask = dilate(&mask, w, h, 1);
    let mask: Vec<bool> = mask.iter().zip(&wide).map(|(m, o)| *m && !*o).collect();
    // Paint over: peel the mask from its boundary inwards, averaging known neighbours.
    let mut px: Vec<[f32; 3]> = img.pixels().map(|p| p.0.map(|c| c as f32)).collect();
    let mut known: Vec<bool> = mask.iter().map(|m| !m).collect();
    let mut frontier: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    for _ in 0..64 {
        if frontier.is_empty() {
            break;
        }
        let mut fill = Vec::new();
        let mut rest = Vec::new();
        for &i in &frontier {
            let (x, y) = (i % w, i / w);
            let mut acc = [0f32; 3];
            let mut n = 0.0;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                    if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if known[j] {
                        for c in 0..3 {
                            acc[c] += px[j][c];
                        }
                        n += 1.0;
                    }
                }
            }
            if n > 0.0 {
                fill.push((i, acc.map(|a| a / n)));
            } else {
                rest.push(i);
            }
        }
        for (i, c) in fill {
            px[i] = c;
            known[i] = true;
        }
        frontier = rest;
    }
    RgbImage::from_fn(w as u32, h as u32, |x, y| image::Rgb(px[y as usize * w + x as usize].map(|c| c.round().clamp(0.0, 255.0) as u8)))
}
