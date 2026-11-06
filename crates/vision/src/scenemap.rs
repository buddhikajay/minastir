//! Per-lens "scene map" saved at calibration: a metric 3D point and a semantic label for every
//! pixel (at MoGe resolution) of the people-free background. The live engine looks up the ground
//! under a detection's feet here, so slopes, kerbs and driveways come out right.

use crate::segment::Label;
use anyhow::{bail, Context, Result};
use std::{
    io::{Read, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"MNSTRSM1";

pub struct LensMap {
    pub width: usize,
    pub height: usize,
    /// OpenCV camera frame of this lens, metres; NaN = unknown (sky…).
    pub points: Vec<[f32; 3]>,
    pub labels: Vec<Label>,
}

impl LensMap {
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&(self.width as u32).to_le_bytes())?;
        f.write_all(&(self.height as u32).to_le_bytes())?;
        for p in &self.points {
            for c in p {
                f.write_all(&c.to_le_bytes())?;
            }
        }
        f.write_all(&self.labels.iter().map(|l| *l as u8).collect::<Vec<_>>())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut buf = Vec::new();
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?.read_to_end(&mut buf)?;
        if buf.len() < 16 || &buf[..8] != MAGIC {
            bail!("{} is not a scene map", path.display());
        }
        let width = u32::from_le_bytes(buf[8..12].try_into()?) as usize;
        let height = u32::from_le_bytes(buf[12..16].try_into()?) as usize;
        let n = width * height;
        if buf.len() != 16 + n * 12 + n {
            bail!("{} is truncated", path.display());
        }
        let f = |i: usize| f32::from_le_bytes(buf[i..i + 4].try_into().unwrap());
        let points = (0..n).map(|k| {
            let o = 16 + k * 12;
            [f(o), f(o + 4), f(o + 8)]
        }).collect();
        let labels = buf[16 + n * 12..].iter().map(|&b| Label::from_u8(b)).collect();
        Ok(Self { width, height, points, labels })
    }

    /// Any 3D point (camera frame) at normalised lens coordinates — for matched keypoints. Rejects
    /// spots on a depth edge (keypoints love edges, where one pixel over is metres away).
    pub fn point_at(&self, u: f64, v: f64) -> Option<[f32; 3]> {
        let cx = (u * self.width as f64) as i64;
        let cy = (v * self.height as f64) as i64;
        let mut pts = Vec::with_capacity(9);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
                    return None;
                }
                let i = y as usize * self.width + x as usize;
                let p = self.points[i];
                if p[0].is_nan() || self.labels[i] == Label::Sky {
                    return None;
                }
                pts.push(p);
            }
        }
        let depth = |p: &[f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let mut ds: Vec<f32> = pts.iter().map(depth).collect();
        ds.sort_by(f32::total_cmp);
        let med = ds[4];
        if ds[7] - ds[1] > 0.12 * med + 0.15 {
            return None;
        }
        pts.into_iter().min_by(|a, b| (depth(a) - med).abs().total_cmp(&(depth(b) - med).abs()))
    }

    /// Whether pixel (x, y) straddles a depth jump. Depth models blend foreground and background
    /// across object edges, leaving "flying" points strung out between them.
    pub fn on_depth_edge(&self, x: usize, y: usize) -> bool {
        let depth = |p: [f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let d = depth(self.points[y * self.width + x]);
        let tol = 0.06 * d + 0.1;
        for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (xx, yy) = (x as i64 + dx, y as i64 + dy);
            if xx < 0 || yy < 0 || xx >= self.width as i64 || yy >= self.height as i64 {
                continue;
            }
            let q = self.points[yy as usize * self.width + xx as usize];
            if q[0].is_nan() || (depth(q) - d).abs() > tol {
                return true;
            }
        }
        false
    }

    /// The background's 3D point at (u, v), whatever it is (camera frame) — it lies on that
    /// pixel's true line of sight, lens distortion included. None for sky / unknown.
    pub fn ray_at(&self, u: f64, v: f64) -> Option<[f32; 3]> {
        let (x, y) = (((u * self.width as f64) as usize).min(self.width - 1), ((v * self.height as f64) as usize).min(self.height - 1));
        let p = self.points[y * self.width + x];
        (!p[0].is_nan()).then_some(p)
    }

    /// The nearest ground the background shows around (u, v): median of ground points in the
    /// smallest window (up to `max_r` pixels out) that holds enough of them. For spots where the
    /// background had something standing on the ground (a letterbox, a parked car).
    pub fn ground_near(&self, u: f64, v: f64, max_r: i64) -> Option<[f32; 3]> {
        let cx = (u * self.width as f64) as i64;
        let cy = (v * self.height as f64) as i64;
        let mut r = 4;
        while r <= max_r {
            let mut pts = Vec::new();
            for y in (cy - r).max(0)..=(cy + r).min(self.height as i64 - 1) {
                for x in (cx - r).max(0)..=(cx + r).min(self.width as i64 - 1) {
                    let i = y as usize * self.width + x as usize;
                    if !self.points[i][0].is_nan() && self.labels[i].is_ground() {
                        pts.push(self.points[i]);
                    }
                }
            }
            if pts.len() >= 12 {
                let med = |k: usize| {
                    let mut v: Vec<f32> = pts.iter().map(|p| p[k]).collect();
                    v.sort_by(f32::total_cmp);
                    v[v.len() / 2]
                };
                return Some([med(0), med(1), med(2)]);
            }
            r *= 2;
        }
        None
    }

    /// Ground point (camera frame) under normalised lens coordinates (u, v): the median of valid
    /// ground-labelled points in a small window. None when that spot isn't ground in the
    /// background (e.g. a parked car or a wall) — callers then fall back to the plane model.
    pub fn ground_at(&self, u: f64, v: f64) -> Option<[f32; 3]> {
        let cx = (u * self.width as f64) as i64;
        let cy = (v * self.height as f64) as i64;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        for dy in -2..=2 {
            for dx in -2..=2 {
                let (x, y) = (cx + dx, cy + dy);
                if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
                    continue;
                }
                let i = y as usize * self.width + x as usize;
                let p = self.points[i];
                if p[0].is_nan() || !self.labels[i].is_ground() {
                    continue;
                }
                xs.push(p[0]);
                ys.push(p[1]);
                zs.push(p[2]);
            }
        }
        if xs.len() < 5 {
            return None;
        }
        let med = |v: &mut Vec<f32>| {
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        Some([med(&mut xs), med(&mut ys), med(&mut zs)])
    }
}
