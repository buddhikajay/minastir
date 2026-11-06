//! A clean top-down map of what one camera sees: ground types (road, footpath, lawn, bare ground)
//! on a grid with cars and people taken out, plus the vertical structure around them (walls,
//! fences, trees, poles) as columns. Built from the calibration scene map, in the camera's levelled
//! frame at the zero pose (camera at the origin, heading 0), so views can place it with the
//! camera's current pose.
//!
//! Steps: vote each labelled 3D point into a 0.2 m cell; fill the ground hidden under and behind
//! parked cars from the surrounding ground (no camera sees under a car, and driveways and roads
//! are flat); close the gaps that perspective leaves between far rows of pixels; smooth the edges
//! with a majority filter; and upsample to 0.1 m with smooth class borders.

use crate::{geometry::CameraModel, segment::Label};
use std::collections::HashMap;

/// Coarse cell size (metres) for voting and filling.
pub const CELL: f64 = 0.2;
/// Output cell size: the coarse grid upsampled with smooth borders.
pub const FINE: f64 = 0.1;
/// Structure column size.
pub const COLUMN: f64 = 0.4;

const GROUND: [Label; 4] = [Label::Road, Label::Sidewalk, Label::Grass, Label::Ground];

pub struct GroundMap {
    /// Map covers x, z ∈ [-half, half] around the camera.
    pub half: f64,
    /// Cells per side at `FINE` resolution. Cell (c, r) is centred at
    /// x = -half + (c + ½)·FINE, z = -half + (r + ½)·FINE; stored row-major by r.
    pub size: usize,
    /// `Label` code per fine cell; 0 = unknown / not ground.
    pub ground: Vec<u8>,
    pub structures: Vec<Structure>,
    /// The structures as simple shapes (walls, trees, poles).
    pub shapes: crate::shapes::Shapes,
    /// How many coarse cells were filled in under / behind cars.
    pub filled_under_vehicles: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Structure {
    pub x: f32,
    pub z: f32,
    /// Top of the column, metres above the camera's ground.
    pub top: f32,
    pub label: Label,
}

fn ground_index(l: Label) -> Option<usize> {
    GROUND.iter().position(|&g| g == l)
}

fn is_structure(l: Label) -> bool {
    matches!(l, Label::Building | Label::Fence | Label::Tree | Label::Plant | Label::Pole | Label::Bin)
}

pub fn build(model: &CameraModel, max_range: f64) -> GroundMap {
    let n = (2.0 * max_range / CELL).ceil() as usize;
    let half = n as f64 * CELL / 2.0;
    let cell_of = |x: f64, z: f64| -> Option<usize> {
        let (c, r) = ((x + half) / CELL, (z + half) / CELL);
        (c >= 0.0 && r >= 0.0 && (c as usize) < n && (r as usize) < n).then(|| r as usize * n + c as usize)
    };
    let cam_h = model.pose.height;

    let pts = model.labelled_points(1, max_range, |l| l != Label::Sky);
    let mut votes = vec![[0u32; 4]; n * n];
    let mut shadow = vec![false; n * n];
    // Cells right under a car's body (its surface points projected straight down).
    let mut under = vec![false; n * n];
    let mut cols: HashMap<(i32, i32), (HashMap<u8, u32>, Vec<f32>)> = HashMap::new();
    for &(x, z, h, l) in &pts {
        if let Some(g) = ground_index(l) {
            if h.abs() < 1.5 {
                if let Some(i) = cell_of(x, z) {
                    votes[i][g] += 1;
                }
            }
        } else if l == Label::Vehicle {
            // The car's own footprint, plus the ground it hides from this camera: the ray over
            // the car's surface meets the ground at range d·H/(H − h).
            if let Some(i) = cell_of(x, z) {
                under[i] = true;
            }
            let d = x.hypot(z);
            let h = h.clamp(0.0, 2.6);
            let reach = if h < cam_h - 0.3 { (d * cam_h / (cam_h - h)).min(d + 8.0) } else { d + 8.0 };
            let (ux, uz) = (x / d.max(1e-6), z / d.max(1e-6));
            let mut t = d;
            while t <= reach {
                if let Some(i) = cell_of(ux * t, uz * t) {
                    shadow[i] = true;
                }
                t += CELL * 0.7;
            }
        } else if is_structure(l) && (0.15..12.0).contains(&h) {
            let key = ((x / COLUMN).floor() as i32, (z / COLUMN).floor() as i32);
            let e = cols.entry(key).or_default();
            *e.0.entry(l as u8).or_default() += 1;
            e.1.push(h as f32);
        }
    }

    // Classify: majority vote. 0 = unknown, else GROUND index + 1.
    let mut cls: Vec<u8> = votes
        .iter()
        .map(|v| {
            let (best, cnt) = v.iter().enumerate().max_by_key(|(_, c)| **c).unwrap();
            if *cnt == 0 { 0 } else { best as u8 + 1 }
        })
        .collect();

    // Fill zone: ground hidden by cars, widened a little (bumpers, wheel shadows).
    let zone = dilate(&shadow, n, 4);
    let under = dilate(&under, n, 2);
    let mut filled_under_vehicles = 0;
    for iter in 0..40 {
        let snap = cls.clone();
        let mut changed = 0;
        for r in 1..n - 1 {
            for c in 1..n - 1 {
                let i = r * n + c;
                if snap[i] != 0 {
                    continue;
                }
                let in_zone = zone[i];
                // Outside car shadows only close real gaps (cells mostly surrounded by ground),
                // so the map's outer edge doesn't creep outward.
                let need = if in_zone { 3 } else if iter < 4 { 5 } else { continue };
                let mut count = [0u8; 5];
                for dr in [-1i64, 0, 1] {
                    for dc in [-1i64, 0, 1] {
                        count[snap[(r as i64 + dr) as usize * n + (c as i64 + dc) as usize] as usize] += 1;
                    }
                }
                let known: u8 = count[1..].iter().sum();
                if known >= need {
                    let best = if under[i] {
                        // Cars stand on paved ground: road if that's what surrounds them,
                        // otherwise paving (driveway, parking pad).
                        if count[1] > count[2] { 1 } else { 2 }
                    } else {
                        (1..5).max_by_key(|&k| count[k]).unwrap()
                    };
                    cls[i] = best as u8;
                    changed += 1;
                    if in_zone {
                        filled_under_vehicles += 1;
                    }
                }
            }
        }
        if changed == 0 {
            break;
        }
    }

    // Smooth edges: majority over a 5×5 window among known cells (two passes).
    for _ in 0..2 {
        let snap = cls.clone();
        for r in 2..n - 2 {
            for c in 2..n - 2 {
                let i = r * n + c;
                if snap[i] == 0 {
                    continue;
                }
                let mut count = [0u8; 5];
                for dr in -2i64..=2 {
                    for dc in -2i64..=2 {
                        count[snap[(r as i64 + dr) as usize * n + (c as i64 + dc) as usize] as usize] += 1;
                    }
                }
                let best = (1..5).max_by_key(|&k| count[k]).unwrap();
                if count[best] > count[snap[i] as usize] {
                    cls[i] = best as u8;
                }
            }
        }
    }
    // Drop specks: known cells with almost no known neighbours.
    let snap = cls.clone();
    for r in 1..n - 1 {
        for c in 1..n - 1 {
            let i = r * n + c;
            if snap[i] == 0 {
                continue;
            }
            let mut known = 0;
            for dr in [-1i64, 0, 1] {
                for dc in [-1i64, 0, 1] {
                    known += (snap[(r as i64 + dr) as usize * n + (c as i64 + dc) as usize] != 0) as u8;
                }
            }
            if known <= 2 {
                cls[i] = 0;
            }
        }
    }

    // Upsample with smooth borders: bilinear blend of each class's indicator, then argmax.
    let k = (CELL / FINE).round() as usize;
    let size = n * k;
    let mut ground = vec![0u8; size * size];
    for fr in 0..size {
        let gr = (fr as f64 + 0.5) / k as f64 - 0.5;
        let r0 = gr.floor().clamp(0.0, (n - 1) as f64) as usize;
        let r1 = (r0 + 1).min(n - 1);
        let tr = (gr - r0 as f64).clamp(0.0, 1.0);
        for fc in 0..size {
            let gc = (fc as f64 + 0.5) / k as f64 - 0.5;
            let c0 = gc.floor().clamp(0.0, (n - 1) as f64) as usize;
            let c1 = (c0 + 1).min(n - 1);
            let tc = (gc - c0 as f64).clamp(0.0, 1.0);
            let mut w = [0f64; 5];
            w[cls[r0 * n + c0] as usize] += (1.0 - tr) * (1.0 - tc);
            w[cls[r0 * n + c1] as usize] += (1.0 - tr) * tc;
            w[cls[r1 * n + c0] as usize] += tr * (1.0 - tc);
            w[cls[r1 * n + c1] as usize] += tr * tc;
            let best = (0..5).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap();
            ground[fr * size + fc] = if best == 0 { 0 } else { GROUND[best - 1] as u8 };
        }
    }

    // Structure columns: enough points, majority label, robust top (90th percentile).
    let mut structures: Vec<Structure> = cols
        .into_iter()
        .filter_map(|((cx, cz), (labels, mut hs))| {
            if hs.len() < 4 {
                return None;
            }
            hs.sort_by(f32::total_cmp);
            let top = hs[(hs.len() * 9 / 10).min(hs.len() - 1)];
            let (&l, _) = labels.iter().max_by_key(|(_, c)| **c)?;
            Some(Structure {
                x: ((cx as f64 + 0.5) * COLUMN) as f32,
                z: ((cz as f64 + 0.5) * COLUMN) as f32,
                top: top.min(9.0),
                label: Label::from_u8(l),
            })
        })
        .collect();
    structures.sort_by(|a, b| a.z.total_cmp(&b.z).then(a.x.total_cmp(&b.x)));

    let shapes = crate::shapes::extract(&structures, COLUMN);
    GroundMap { half, size, ground, structures, shapes, filled_under_vehicles }
}

/// Square dilation by `r` cells (separable max filter).
fn dilate(m: &[bool], n: usize, r: usize) -> Vec<bool> {
    let mut a = vec![false; n * n];
    for y in 0..n {
        for x in 0..n {
            if m[y * n + x] {
                for xx in x.saturating_sub(r)..=(x + r).min(n - 1) {
                    a[y * n + xx] = true;
                }
            }
        }
    }
    let mut b = vec![false; n * n];
    for y in 0..n {
        for x in 0..n {
            if a[y * n + x] {
                for yy in y.saturating_sub(r)..=(y + r).min(n - 1) {
                    b[yy * n + x] = true;
                }
            }
        }
    }
    b
}

impl GroundMap {
    /// Ground labels as run-length pairs (label, count, label, count, …), row-major.
    pub fn rle(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let mut it = self.ground.iter();
        let Some(&first) = it.next() else { return out };
        let (mut cur, mut len) = (first, 1u32);
        for &v in it {
            if v == cur {
                len += 1;
            } else {
                out.extend([cur as u32, len]);
                cur = v;
                len = 1;
            }
        }
        out.extend([cur as u32, len]);
        out
    }

    /// Top-down debug picture (north = +z up), one pixel per fine cell.
    pub fn image(&self) -> image::RgbImage {
        let col = |l: u8| -> [u8; 3] {
            match Label::from_u8(l) {
                Label::Road => [150, 155, 163],
                Label::Sidewalk => [214, 200, 176],
                Label::Grass => [150, 200, 138],
                Label::Ground => [200, 178, 140],
                _ => [245, 246, 247],
            }
        };
        let s = self.size;
        let mut img = image::RgbImage::from_fn(s as u32, s as u32, |x, y| {
            image::Rgb(col(self.ground[(s - 1 - y as usize) * s + x as usize]))
        });
        let px = |v: f32| ((v as f64 + self.half) / FINE) as i64;
        let c = (COLUMN / FINE) as i64;
        for st in &self.structures {
            let rgb = match st.label {
                Label::Building => [200, 110, 100],
                Label::Fence => [222, 160, 80],
                Label::Tree | Label::Plant => [70, 130, 85],
                _ => [90, 90, 90],
            };
            let (x0, y0) = (px(st.x) - c / 2, s as i64 - 1 - px(st.z) - c / 2);
            for dy in 0..c {
                for dx in 0..c {
                    let (x, y) = (x0 + dx, y0 + dy);
                    if x >= 0 && y >= 0 && x < s as i64 && y < s as i64 {
                        img.put_pixel(x as u32, y as u32, image::Rgb(rgb));
                    }
                }
            }
        }
        // Camera at the centre.
        let m = s as i64 / 2;
        for d in -3..=3 {
            for (x, y) in [(m + d, m), (m, m + d)] {
                img.put_pixel(x as u32, y as u32, image::Rgb([20, 20, 20]));
            }
        }
        img
    }
}
