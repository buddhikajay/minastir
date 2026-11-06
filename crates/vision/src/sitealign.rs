//! Automatically place a map image (a satellite screenshot) on the cameras' map.
//!
//! Both show the same layout from above: strips of lawn alternating with paved road, footpath and
//! driveway. From the cameras we build a "grass vs paved" grid in world coordinates (+1 grass,
//! −1 paved, 0 unknown), with the clean street bands painted over the raw ground maps. From the
//! photo, the same signal from colour: green → grass, grey and unsaturated → paved. Then we search
//! every rotation and a wide range of scales; for each, the best position comes from one FFT
//! cross-correlation over the whole photo. The best candidates are refined on a sharper copy.

use crate::{
    groundmap::{GroundMap, FINE},
    segment::Label,
    street::Street,
};
use minastir_core::onvif::Pose;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

/// Template cell size, metres.
const CELL: f64 = 0.5;
/// Only ground this close to a camera goes into the template (further out, depth is rough).
const RELIABLE: f64 = 25.0;

/// The cameras' map as a grass (+) / paved (−) grid.
pub struct Template {
    x0: f64,
    z0: f64,
    w: usize,
    h: usize,
    t: Vec<f32>,
}

impl Template {
    fn at(&self, x: f64, z: f64) -> f32 {
        let (i, j) = ((x - self.x0) / CELL, (z - self.z0) / CELL);
        if i < 0.0 || j < 0.0 || i >= self.w as f64 || j >= self.h as f64 {
            return 0.0;
        }
        self.t[j as usize * self.w + i as usize]
    }
}

pub fn template(maps: &[(&GroundMap, Pose)], street: Option<&Street>) -> Option<Template> {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for (g, p) in maps {
        lo = [lo[0].min(p.x - g.half), lo[1].min(p.z - g.half)];
        hi = [hi[0].max(p.x + g.half), hi[1].max(p.z + g.half)];
    }
    if maps.is_empty() {
        return None;
    }
    let (w, h) = (((hi[0] - lo[0]) / CELL).ceil() as usize + 1, ((hi[1] - lo[1]) / CELL).ceil() as usize + 1);
    // Votes per cell: grass, paved.
    let mut v = vec![[0f32; 2]; w * h];
    for (g, p) in maps {
        let (s, c) = p.yaw.to_radians().sin_cos();
        let step = (CELL / FINE / 2.0).round().max(1.0) as usize;
        for r in (0..g.size).step_by(step) {
            for cc in (0..g.size).step_by(step) {
                let l = Label::from_u8(g.ground[r * g.size + cc]);
                let k = match l {
                    Label::Grass => 0,
                    Label::Road | Label::Sidewalk => 1,
                    _ => continue,
                };
                let lx = -g.half + (cc as f64 + 0.5) * FINE;
                let lz = -g.half + (r as f64 + 0.5) * FINE;
                let d = lx.hypot(lz);
                if d > RELIABLE {
                    continue;
                }
                let wt = 1.0 / (1.0 + (d / 10.0).powi(2));
                let (wx, wz) = (p.x + lx * c + lz * s, p.z - lx * s + lz * c);
                let (i, j) = (((wx - lo[0]) / CELL) as usize, ((wz - lo[1]) / CELL) as usize);
                if i < w && j < h {
                    v[j * w + i][k] += wt as f32;
                }
            }
        }
    }
    let mut totals: Vec<f32> = v.iter().map(|x| x[0] + x[1]).filter(|&t| t > 0.0).collect();
    if totals.len() < 200 {
        return None;
    }
    totals.sort_by(f32::total_cmp);
    let typical = totals[totals.len() / 2];
    let mut t: Vec<f32> = v
        .iter()
        .map(|x| {
            let tot = x[0] + x[1];
            if tot <= 0.0 {
                0.0
            } else {
                (x[0] - x[1]) / tot * (tot / typical).min(1.0)
            }
        })
        .collect();
    // The fitted street is cleaner than the raw votes: paint its strips at full confidence.
    if let Some(st) = street {
        let (sn, cs) = st.heading.to_radians().sin_cos();
        let near_cam = |x: f64, z: f64| maps.iter().any(|(_, p)| (p.x - x).hypot(p.z - z) <= RELIABLE);
        let mut paint = |s0: f64, s1: f64, t0: f64, t1: f64, val: f32| {
            let mut a = s0;
            while a < s1 {
                let mut b = t0;
                while b < t1 {
                    let (x, z) = (st.origin[0] + a * sn + b * cs, st.origin[1] + a * cs - b * sn);
                    let (i, j) = (((x - lo[0]) / CELL) as isize, ((z - lo[1]) / CELL) as isize);
                    if i >= 0 && j >= 0 && (i as usize) < w && (j as usize) < h && near_cam(x, z) {
                        t[j as usize * w + i as usize] = val;
                    }
                    b += CELL * 0.5;
                }
                a += CELL * 0.5;
            }
        };
        for b in &st.bands {
            let val = match b.label {
                Label::Grass => 1.0,
                Label::Road | Label::Sidewalk => -1.0,
                _ => continue,
            };
            paint(b.s0, b.s1, b.t0, b.t1, val);
        }
        for d in &st.driveways {
            paint(d.s0, d.s1, d.t0, d.t1, -1.0);
        }
    }
    // Balance grass against paving (zero mean over cells with data), so a uniform patch of
    // the photo — a park, a car park — scores nothing; only the pattern counts.
    let (sum, n) = t.iter().filter(|v| **v != 0.0).fold((0f64, 0usize), |(s, n), v| (s + *v as f64, n + 1));
    let mean = (sum / n.max(1) as f64) as f32;
    for v in t.iter_mut().filter(|v| **v != 0.0) {
        *v -= mean;
        if *v == 0.0 {
            *v = 1e-6; // keep "has data"
        }
    }
    // Crop to where there is data (each camera's map is a 80 m square, mostly empty).
    let (mut i0, mut i1, mut j0, mut j1) = (w, 0, h, 0);
    for j in 0..h {
        for i in 0..w {
            if t[j * w + i] != 0.0 {
                (i0, i1, j0, j1) = (i0.min(i), i1.max(i), j0.min(j), j1.max(j));
            }
        }
    }
    if i1 < i0 || j1 < j0 {
        return None;
    }
    let (cw, ch) = (i1 - i0 + 1, j1 - j0 + 1);
    let mut ct = vec![0f32; cw * ch];
    for j in 0..ch {
        ct[j * cw..(j + 1) * cw].copy_from_slice(&t[(j + j0) * w + i0..(j + j0) * w + i0 + cw]);
    }
    Some(Template { x0: lo[0] + i0 as f64 * CELL, z0: lo[1] + j0 as f64 * CELL, w: cw, h: ch, t: ct })
}

/// Agreement of a given placement (same measure as the search), for comparisons.
pub fn evaluate(img: &image::RgbImage, tpl: &Template, x: f64, z: f64, scale: f64, rotation: f64) -> f64 {
    let f = (img.width() as f64 / 640.0).max(1.0);
    let (sig, w, h) = photo_signal(img, f);
    let (sn, cs) = rotation.to_radians().sin_cos();
    let (mut acc, mut mass) = (0f64, 0f64);
    for j in 0..tpl.h {
        for i in 0..tpl.w {
            let t = tpl.t[j * tpl.w + i] as f64;
            if t == 0.0 {
                continue;
            }
            mass += t.abs();
            let (wx, wz) = (tpl.x0 + (i as f64 + 0.5) * CELL, tpl.z0 + (j as f64 + 0.5) * CELL);
            let (dx, dz) = ((wx - x) / scale, (wz - z) / scale);
            let (u, v) = ((dx * cs + dz * sn) / f, (-dx * sn + dz * cs) / f);
            if u >= 0.0 && v >= 0.0 && (u as usize) < w && (v as usize) < h {
                acc += t * sig[v as usize * w + u as usize] as f64;
            }
        }
    }
    acc / mass.max(1e-9)
}

/// Refines a placement close to where it already is: the cameras' map changed size or shifted a
/// little (e.g. after recalibrating), and a street of near-identical houses has many good fits a
/// few doors along, so a fresh search can land on the wrong house. Scales about `about` (world
/// x, z — the cameras), within ±35% scale, ±5° rotation and ±6 m.
pub fn refine(img: &image::RgbImage, tpl: &Template, start: (f64, f64, f64, f64), about: (f64, f64)) -> Placement {
    let f = (img.width() as f64 / 640.0).max(1.0);
    let (sig, w, h) = photo_signal(img, f);
    let cells: Vec<(f64, f64, f64)> = (0..tpl.h)
        .flat_map(|j| (0..tpl.w).map(move |i| (i, j)))
        .filter_map(|(i, j)| {
            let t = tpl.t[j * tpl.w + i] as f64;
            (t != 0.0).then(|| (tpl.x0 + (i as f64 + 0.5) * CELL, tpl.z0 + (j as f64 + 0.5) * CELL, t))
        })
        .collect();
    let mass: f64 = cells.iter().map(|c| c.2.abs()).sum::<f64>().max(1e-9);
    // Placement as (photo pixel under `about`, scale, rotation, offset of that pixel in metres).
    let (x0, z0, s0, r0) = start;
    let (sn, cs) = r0.to_radians().sin_cos();
    let (dx, dz) = ((about.0 - x0) / s0, (about.1 - z0) / s0);
    let anchor = (dx * cs + dz * sn, -dx * sn + dz * cs);
    let place = |k: f64, rot: f64, ox: f64, oz: f64| -> (f64, f64, f64, f64) {
        let scale = s0 * k;
        let (sn, cs) = rot.to_radians().sin_cos();
        let (u, v) = anchor;
        (about.0 + ox - scale * (u * cs - v * sn), about.1 + oz - scale * (u * sn + v * cs), scale, rot)
    };
    let score = |p: (f64, f64, f64, f64)| -> f64 {
        let (sn, cs) = p.3.to_radians().sin_cos();
        let mut acc = 0.0;
        for &(wx, wz, t) in &cells {
            let (dx, dz) = ((wx - p.0) / p.2, (wz - p.1) / p.2);
            let (u, v) = ((dx * cs + dz * sn) / f, (-dx * sn + dz * cs) / f);
            if u >= 0.0 && v >= 0.0 && (u as usize) < w && (v as usize) < h {
                acc += t * sig[v as usize * w + u as usize] as f64;
            }
        }
        acc / mass
    };
    let mut best = (1.0, r0, 0.0, 0.0);
    let mut best_s = score(place(best.0, best.1, best.2, best.3));
    let initial = best_s;
    // Coarse grid over scale and rotation with a position sweep, then finer passes around the best.
    for (ks, rs, os, n) in [(0.05, 1.0, 1.0, 6i32), (0.01, 0.25, 0.25, 4), (0.0025, 0.1, 0.1, 3)] {
        let c = best;
        for a in -n..=n {
            let k = c.0 + a as f64 * ks;
            if !(0.65..=1.35).contains(&k) {
                continue;
            }
            for b in -n..=n {
                let rot = c.1 + b as f64 * rs;
                if (rot - r0).abs() > 5.0 {
                    continue;
                }
                for i in -n..=n {
                    for j in -n..=n {
                        let (ox, oz) = (c.2 + i as f64 * os, c.3 + j as f64 * os);
                        if ox.hypot(oz) > 6.0 {
                            continue;
                        }
                        let s = score(place(k, rot, ox, oz));
                        if s > best_s {
                            (best, best_s) = ((k, rot, ox, oz), s);
                        }
                    }
                }
            }
        }
    }
    let p = place(best.0, best.1, best.2, best.3);
    Placement { x: p.0, z: p.1, scale: p.2, rotation: p.3.rem_euclid(360.0), score: best_s, confidence: best_s / initial.max(1e-9) }
}

/// Where the image goes: pixel (u, v) → world (x, z) + scale · R(rotation) · (u, v).
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub x: f64,
    pub z: f64,
    pub scale: f64,
    pub rotation: f64,
    /// Agreement per template pixel (≈ −1..1).
    pub score: f64,
    /// Best score over the best clearly different alternative (> 1 is good, ≥ 1.3 confident).
    pub confidence: f64,
}

/// The photo's grass (+) / paved (−) signal, downsampled by `f` (working px = f original px),
/// zero-mean.
fn photo_signal(img: &image::RgbImage, f: f64) -> (Vec<f32>, usize, usize) {
    let (w0, h0) = (img.width() as usize, img.height() as usize);
    let (w, h) = (((w0 as f64) / f).floor() as usize, ((h0 as f64) / f).floor() as usize);
    let mut acc = vec![0f32; w * h];
    let mut cnt = vec![0f32; w * h];
    for (x, y, p) in img.enumerate_pixels() {
        let (i, j) = (((x as f64) / f) as usize, ((y as f64) / f) as usize);
        if i >= w || j >= h {
            continue;
        }
        let [r, g, b] = p.0.map(|c| c as f32);
        let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
        let sat = (mx - mn) / mx.max(1.0);
        let val = mx / 255.0;
        let green = if g >= r && g >= b { ((g - r.max(b)) / 30.0).clamp(0.0, 1.0) } else { 0.0 };
        let grey = if sat < 0.15 && (0.3..0.92).contains(&val) { 1.0 - sat / 0.15 } else { 0.0 };
        acc[j * w + i] += green - grey;
        cnt[j * w + i] += 1.0;
    }
    for k in 0..w * h {
        acc[k] /= cnt[k].max(1.0);
    }
    let mean = acc.iter().sum::<f32>() / acc.len().max(1) as f32;
    for a in &mut acc {
        *a -= mean;
    }
    (acc, w, h)
}

struct Fft2 {
    n1: usize,
    n2: usize,
    f1: Arc<dyn Fft<f32>>,
    f2: Arc<dyn Fft<f32>>,
    i1: Arc<dyn Fft<f32>>,
    i2: Arc<dyn Fft<f32>>,
}

impl Fft2 {
    fn new(n1: usize, n2: usize) -> Self {
        let mut p = FftPlanner::new();
        Self { n1, n2, f1: p.plan_fft_forward(n1), f2: p.plan_fft_forward(n2), i1: p.plan_fft_inverse(n1), i2: p.plan_fft_inverse(n2) }
    }

    /// In-place 2D transform of an n2 × n1 (rows × cols) grid.
    fn run(&self, data: &mut [Complex32], inverse: bool) {
        let (a, b) = if inverse { (&self.i1, &self.i2) } else { (&self.f1, &self.f2) };
        for row in data.chunks_exact_mut(self.n1) {
            a.process(row);
        }
        let mut col = vec![Complex32::new(0.0, 0.0); self.n2];
        for x in 0..self.n1 {
            for y in 0..self.n2 {
                col[y] = data[y * self.n1 + x];
            }
            b.process(&mut col);
            for y in 0..self.n2 {
                data[y * self.n1 + x] = col[y];
            }
        }
    }
}

/// One search level: the photo at working factor `f`; candidates (scale m/px, rotation °).
/// Returns (score, scale, rotation, x, z) per candidate that fits.
fn search(img: &image::RgbImage, tpl: &Template, f: f64, cands: &[(f64, f64)]) -> Vec<(f64, f64, f64, f64, f64)> {
    let (sig, w, h) = photo_signal(img, f);
    // Group candidates by FFT size: padding only as large as each candidate's template needs.
    // The cameras' (reliable) area must lie inside the photo.
    let mut groups: std::collections::BTreeMap<(usize, usize), Vec<(f64, f64)>> = Default::default();
    for &(scale, rot) in cands {
        let (bw, bh) = footprint(tpl, scale * f, rot);
        if bw > w || bh > h || bw < 6 || bh < 6 {
            continue;
        }
        groups.entry(((w + bw).next_power_of_two(), (h + bh).next_power_of_two())).or_default().push((scale, rot));
    }
    let out = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    for ((n1, n2), group) in groups {
        let fft = Fft2::new(n1, n2);
        let mut sf = vec![Complex32::new(0.0, 0.0); n1 * n2];
        for y in 0..h {
            for x in 0..w {
                sf[y * n1 + x].re = sig[y * w + x];
            }
        }
        fft.run(&mut sf, false);
        let mut s2f = vec![Complex32::new(0.0, 0.0); n1 * n2];
        for y in 0..h {
            for x in 0..w {
                s2f[y * n1 + x].re = sig[y * w + x] * sig[y * w + x];
            }
        }
        fft.run(&mut s2f, false);
        let mean_e = sig.iter().map(|v| v * v).sum::<f32>() / sig.len().max(1) as f32;
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut buf = vec![Complex32::new(0.0, 0.0); n1 * n2];
                    let mut buf2 = vec![Complex32::new(0.0, 0.0); n1 * n2];
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= group.len() {
                            break;
                        }
                        let (scale, rot) = group[k];
                        if let Some(r) = one(tpl, &sf, &s2f, mean_e, &fft, &mut buf, &mut buf2, w, h, f, scale, rot) {
                            out.lock().unwrap().push(r);
                        }
                    }
                });
            }
        });
    }
    out.into_inner().unwrap()
}

/// Size (working px) of the template's bounding box at `px` metres per pixel, turned by `rot`.
fn footprint(tpl: &Template, px: f64, rot: f64) -> (usize, usize) {
    let (sn, cs) = rot.to_radians().sin_cos();
    let (tw, th) = (tpl.w as f64 * CELL, tpl.h as f64 * CELL);
    let bw = (tw * cs.abs() + th * sn.abs()) / px;
    let bh = (tw * sn.abs() + th * cs.abs()) / px;
    (bw.ceil() as usize + 1, bh.ceil() as usize + 1)
}

/// Best position for one (scale, rotation): render the template into working pixels, correlate.
#[allow(clippy::too_many_arguments)]
fn one(
    tpl: &Template,
    sf: &[Complex32],
    s2f: &[Complex32],
    mean_e: f32,
    fft: &Fft2,
    buf: &mut [Complex32],
    mask: &mut [Complex32],
    w: usize,
    h: usize,
    f: f64,
    scale: f64,
    rot: f64,
) -> Option<(f64, f64, f64, f64, f64)> {
    let (n1, n2) = (fft.n1, fft.n2);
    let px = scale * f; // metres per working pixel
    let (sn, cs) = rot.to_radians().sin_cos();
    // World → working px (relative to the template origin A): q = R(−θ)(w − A) / px.
    let a = [tpl.x0, tpl.z0];
    let (tw, th) = (tpl.w as f64 * CELL, tpl.h as f64 * CELL);
    let corners = [[0.0, 0.0], [tw, 0.0], [tw, th], [0.0, th]].map(|d: [f64; 2]| [(d[0] * cs + d[1] * sn) / px, (-d[0] * sn + d[1] * cs) / px]);
    let qmin = [corners.iter().map(|c| c[0]).fold(f64::MAX, f64::min).floor(), corners.iter().map(|c| c[1]).fold(f64::MAX, f64::min).floor()];
    let qmax = [corners.iter().map(|c| c[0]).fold(f64::MIN, f64::max).ceil(), corners.iter().map(|c| c[1]).fold(f64::MIN, f64::max).ceil()];
    let (bw, bh) = ((qmax[0] - qmin[0]) as usize, (qmax[1] - qmin[1]) as usize);
    if bw + w > n1 || bh + h > n2 {
        return None;
    }
    buf.iter_mut().for_each(|c| *c = Complex32::new(0.0, 0.0));
    mask.iter_mut().for_each(|c| *c = Complex32::new(0.0, 0.0));
    let mut mass = 0f64;
    let mut t2 = 0f64;
    let mut count = 0f64;
    for j in 0..bh {
        for i in 0..bw {
            let q = [qmin[0] + i as f64 + 0.5, qmin[1] + j as f64 + 0.5];
            let wx = a[0] + (q[0] * cs - q[1] * sn) * px;
            let wz = a[1] + (q[0] * sn + q[1] * cs) * px;
            let t = tpl.at(wx, wz);
            if t != 0.0 {
                buf[j * n1 + i].re = t;
                mask[j * n1 + i].re = 1.0;
                mass += t.abs() as f64;
                t2 += (t * t) as f64;
                count += 1.0;
            }
        }
    }
    if mass < 150.0 {
        return None; // too few pixels to judge
    }
    fft.run(buf, false);
    for (b, s) in buf.iter_mut().zip(sf) {
        *b = s * b.conj();
    }
    fft.run(buf, true);
    // Photo contrast under the template at each offset (for normalised correlation).
    fft.run(mask, false);
    for (m, s) in mask.iter_mut().zip(s2f) {
        *m = s * m.conj();
    }
    fft.run(mask, true);
    // Offsets keeping the template inside the photo; normalised cross-correlation.
    let norm = (n1 * n2) as f32;
    let floor = 0.25 * mean_e as f64 * count;
    let mut best = (f64::MIN, 0i64, 0i64);
    for ty in 0..=(h - bh) as i64 {
        for tx in 0..=(w - bw) as i64 {
            let k = ty as usize * n1 + tx as usize;
            let c = (buf[k].re / norm) as f64;
            let e = ((mask[k].re / norm) as f64).max(floor);
            let v = c / (e * t2).sqrt().max(1e-9);
            if v > best.0 {
                best = (v, tx, ty);
            }
        }
    }
    let score = best.0;
    // Template origin A sits at working pixel t − qmin → original pixel u_A; then the photo's
    // own origin: (x, z) = A − scale · R(θ) · u_A.
    let ua = [(best.1 as f64 - qmin[0]) * f, (best.2 as f64 - qmin[1]) * f];
    let x = a[0] - scale * (ua[0] * cs - ua[1] * sn);
    let z = a[1] - scale * (ua[0] * sn + ua[1] * cs);
    Some((score, scale, rot, x, z))
}

/// Search all rotations and plausible scales, then refine the best candidates.
pub fn align(img: &image::RgbImage, tpl: &Template) -> Option<Placement> {
    // Coarse: photo ~256 px wide, 2° steps, scales 0.02–0.5 m per original pixel.
    let fc = (img.width() as f64 / 256.0).max(1.0);
    let mut scales = Vec::new();
    let mut s = 0.02;
    while s <= 0.5 {
        scales.push(s);
        s *= 1.07;
    }
    let coarse: Vec<(f64, f64)> = scales.iter().flat_map(|&s| (0..180).map(move |k| (s, k as f64 * 2.0))).collect();
    let mut found = search(img, tpl, fc, &coarse);
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Distinct leaders: different by > 8° or > 12% scale or > 6 m.
    // Compare by where the middle of the cameras' map lands on the photo (in metres), not by the
    // photo's corner, which swings far for tiny turns.
    let centre = [tpl.x0 + tpl.w as f64 * CELL / 2.0, tpl.z0 + tpl.h as f64 * CELL / 2.0];
    let on_photo = |c: &(f64, f64, f64, f64, f64)| {
        let (sn, cs) = c.2.to_radians().sin_cos();
        let (dx, dz) = (centre[0] - c.3, centre[1] - c.4);
        [(dx * cs + dz * sn) / c.1, (-dx * sn + dz * cs) / c.1] // photo pixels
    };
    let differ = |a: &(f64, f64, f64, f64, f64), b: &(f64, f64, f64, f64, f64)| {
        let dr = ((a.2 - b.2 + 540.0) % 360.0 - 180.0).abs();
        let (pa, pb) = (on_photo(a), on_photo(b));
        let shift = (pa[0] - pb[0]).hypot(pa[1] - pb[1]) * a.1; // metres on the ground
        dr > 8.0 || (a.1 / b.1).ln().abs() > 0.12 || shift > 6.0
    };
    let mut leaders: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for c in &found {
        if leaders.iter().all(|l| differ(l, c)) {
            leaders.push(*c);
            if leaders.len() == 4 {
                break;
            }
        }
    }
    if leaders.is_empty() {
        return None;
    }
    // Refine each leader on a sharper photo.
    let ff = (img.width() as f64 / 640.0).max(1.0);
    let mut refined: Vec<(f64, f64, f64, f64, f64)> = Vec::new();
    for l in &leaders {
        let cands: Vec<(f64, f64)> = (-5..=5)
            .flat_map(|i| (-5..=5).map(move |k| (l.1 * 1.02f64.powi(i), l.2 + k as f64 * 0.5)))
            .collect();
        let mut r = search(img, tpl, ff, &cands);
        r.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some(b) = r.first() {
            refined.push(*b);
        }
    }
    refined.sort_by(|a, b| b.0.total_cmp(&a.0));
    if std::env::var("MINASTIR_DEBUG_ALIGN").is_ok() {
        for r in &refined {
            let p = on_photo(r);
            eprintln!("  candidate score {:.4} scale {:.4} rot {:.1} centre on photo ({:.0}, {:.0}) px", r.0, r.1, r.2, p[0], p[1]);
        }
    }
    let best = refined[0];
    let second = refined.iter().skip(1).find(|c| differ(c, &best)).map_or(0.0, |c| c.0);
    Some(Placement {
        x: best.3,
        z: best.4,
        scale: best.1,
        rotation: (best.2 + 360.0) % 360.0,
        score: best.0,
        confidence: if second > 0.0 { best.0 / second } else { 9.9 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic "photo" made from a known layout, scaled and rotated; align must recover it.
    #[test]
    fn recovers_known_placement() {
        // Layout in world metres: grass everywhere, a 9 m road along x at z ∈ [10, 19], a 2 m
        // footpath at z ∈ [6, 8], and three 6 m driveways crossing from the road to z = 0.
        let paved = |x: f64, z: f64| {
            (10.0..19.0).contains(&z) || (6.0..8.0).contains(&z) || ((0.0..10.0).contains(&z) && [5.0, 22.0, 41.0].iter().any(|&d| (d..d + 6.0).contains(&x)))
        };
        // Template over a 50 × 30 m patch.
        let (w, h) = ((50.0 / CELL) as usize, (30.0 / CELL) as usize);
        let t = (0..w * h)
            .map(|k| {
                let (x, z) = ((k % w) as f64 * CELL + 0.25, (k / w) as f64 * CELL + 0.25);
                if paved(x, z) { -1.0 } else { 1.0 }
            })
            .collect();
        let tpl = Template { x0: 0.0, z0: 0.0, w, h, t };
        // Photo: 0.1 m/px, rotated 30°, covering a bigger area.
        let (scale, rot) = (0.1f64, 30f64);
        let (ox, oz) = (3.5, -37.8); // template centre (25, 15) lands near the photo centre
        let (sn, cs) = rot.to_radians().sin_cos();
        let img = image::RgbImage::from_fn(900, 700, |u, v| {
            let (u, v) = (u as f64, v as f64);
            let (x, z) = (ox + scale * (u * cs - v * sn), oz + scale * (u * sn + v * cs));
            if paved(x, z) { image::Rgb([128, 128, 128]) } else { image::Rgb([90, 150, 70]) }
        });
        let p = align(&img, &tpl).expect("aligned");
        let d = ((p.rotation - rot + 540.0) % 360.0 - 180.0).abs();
        assert!(d < 1.5, "{p:?}");
        assert!((p.scale / scale - 1.0).abs() < 0.04, "{p:?}");
        // Check a photo pixel lands where it should.
        let (u, v) = (450.0, 350.0);
        let (sn2, cs2) = p.rotation.to_radians().sin_cos();
        let got = (p.x + p.scale * (u * cs2 - v * sn2), p.z + p.scale * (u * sn2 + v * cs2));
        let want = (ox + scale * (u * cs - v * sn), oz + scale * (u * sn + v * cs));
        assert!((got.0 - want.0).hypot(got.1 - want.1) < 1.5, "{p:?} {got:?} {want:?}");
    }
}
