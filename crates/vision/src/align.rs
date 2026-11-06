//! Cross-camera alignment: refine each camera's heading and position on the plan so that what
//! overlapping cameras see lines up — road on road, walls on walls, the parked ute on the parked ute.
//!
//! Each calibrated camera contributes a top-down, labelled point set (from its scene map). One
//! camera stays where the user put it (the anchor); every other camera is searched over heading
//! and position around the user's rough placement, coarse then fine, maximising label agreement
//! with the cameras already placed. Cameras that barely overlap anything keep their rough pose.

use crate::segment::Label;
use minastir_core::onvif::Pose;

/// A camera's top-down points in its own levelled frame (pose x = z = yaw = 0).
pub struct AlignCam {
    pub id: String,
    pub rough: Pose,
    pub local: Vec<(f32, f32, Label)>,
}

#[derive(Debug, Clone)]
pub struct Aligned {
    pub id: String,
    pub yaw: f64,
    pub x: f64,
    pub z: f64,
    /// Fraction of this camera's points that agree with the others (0..1); None = not aligned
    /// (anchor, or too little overlap).
    pub agreement: Option<f64>,
    pub anchor: bool,
}

const CELL: f64 = 0.25;
const CLASSES: usize = 8;

fn class(l: Label) -> Option<usize> {
    Some(match l {
        Label::Road => 0,
        Label::Sidewalk => 1,
        Label::Grass => 2,
        Label::Ground => 3,
        Label::Building => 4,
        Label::Fence => 5,
        Label::Tree | Label::Plant => 6,
        Label::Vehicle => 7,
        _ => return None,
    })
}

/// Label histogram per cell, for the cameras placed so far.
struct Grid {
    x0: f64,
    z0: f64,
    w: usize,
    h: usize,
    counts: Vec<[u16; CLASSES]>,
}

impl Grid {
    fn new(half: f64) -> Self {
        let n = (2.0 * half / CELL) as usize;
        Self { x0: -half, z0: -half, w: n, h: n, counts: vec![[0; CLASSES]; n * n] }
    }

    fn cell(&self, x: f64, z: f64) -> Option<usize> {
        let (cx, cz) = ((x - self.x0) / CELL, (z - self.z0) / CELL);
        if cx < 0.0 || cz < 0.0 {
            return None;
        }
        let (cx, cz) = (cx as usize, cz as usize);
        (cx < self.w && cz < self.h).then(|| cz * self.w + cx)
    }

    fn add(&mut self, cam: &AlignCam, yaw: f64, x: f64, z: f64) {
        let (s, c) = yaw.to_radians().sin_cos();
        for &(lx, lz, l) in &cam.local {
            let Some(k) = class(l) else { continue };
            let (lx, lz) = (lx as f64, lz as f64);
            if let Some(i) = self.cell(x + lx * c + lz * s, z - lx * s + lz * c) {
                self.counts[i][k] = self.counts[i][k].saturating_add(1);
            }
        }
    }

    /// (agreement score, fraction of points landing on observed cells).
    fn score(&self, pts: &[(f32, f32, usize)], yaw: f64, x: f64, z: f64) -> (f64, f64) {
        let (s, c) = yaw.to_radians().sin_cos();
        let (mut score, mut seen) = (0.0, 0usize);
        for &(lx, lz, k) in pts {
            let (lx, lz) = (lx as f64, lz as f64);
            let Some(i) = self.cell(x + lx * c + lz * s, z - lx * s + lz * c) else { continue };
            let h = &self.counts[i];
            let total: u32 = h.iter().map(|&v| v as u32).sum();
            if total == 0 {
                continue;
            }
            seen += 1;
            let agree = h[k] as f64 / total as f64;
            // Reward matching labels, penalise contradictions (e.g. lawn where others see a wall).
            score += agree - 0.35 * (1.0 - agree);
        }
        (score, seen as f64 / pts.len().max(1) as f64)
    }
}

fn subsample(cam: &AlignCam, max: usize) -> Vec<(f32, f32, usize)> {
    let pts: Vec<(f32, f32, usize)> = cam.local.iter().filter_map(|&(x, z, l)| class(l).map(|k| (x, z, k))).collect();
    let step = (pts.len() / max).max(1);
    pts.into_iter().step_by(step).collect()
}

/// Best (yaw, x, z) for `pts` against `grid` in a window around `start`.
fn search(grid: &Grid, pts: &[(f32, f32, usize)], start: (f64, f64, f64), span: (f64, f64), step: (f64, f64)) -> (f64, f64, f64, f64, f64) {
    let (yaw0, x0, z0) = start;
    let ny = (span.0 / step.0).round() as i64;
    let nt = (span.1 / step.1).round() as i64;
    let mut best = (f64::NEG_INFINITY, yaw0, x0, z0, 0.0);
    for iy in -ny..=ny {
        let yaw = yaw0 + iy as f64 * step.0;
        for ix in -nt..=nt {
            for iz in -nt..=nt {
                let (x, z) = (x0 + ix as f64 * step.1, z0 + iz as f64 * step.1);
                let (s, seen) = grid.score(pts, yaw, x, z);
                // A gentle pull toward the user's placement breaks ties on featureless ground.
                let prior = 0.002 * pts.len() as f64 * (((yaw - yaw0) / 20.0).powi(2) + ((x - x0).powi(2) + (z - z0).powi(2)) / 25.0);
                let total = s - prior;
                if total > best.0 {
                    best = (total, yaw, x, z, seen);
                }
            }
        }
    }
    best
}

pub fn align(cams: &[AlignCam]) -> Vec<Aligned> {
    let n = cams.len();
    let mut poses: Vec<(f64, f64, f64)> = cams.iter().map(|c| (c.rough.yaw, c.rough.x, c.rough.z)).collect();
    let mut result: Vec<Aligned> =
        cams.iter().map(|c| Aligned { id: c.id.clone(), yaw: c.rough.yaw, x: c.rough.x, z: c.rough.z, agreement: None, anchor: false }).collect();
    if n < 2 {
        return result;
    }
    let pts: Vec<Vec<(f32, f32, usize)>> = cams.iter().map(|c| subsample(c, 12_000)).collect();

    // Overlap at the rough poses decides the anchor (most overlap) and the order.
    let overlap = |i: usize, poses: &[(f64, f64, f64)]| -> f64 {
        let mut g = Grid::new(70.0);
        for j in 0..n {
            if j != i {
                g.add(&cams[j], poses[j].0, poses[j].1, poses[j].2);
            }
        }
        g.score(&pts[i], poses[i].0, poses[i].1, poses[i].2).1
    };
    let mut ov: Vec<(usize, f64)> = (0..n).map(|i| (i, overlap(i, &poses))).collect();
    ov.sort_by(|a, b| b.1.total_cmp(&a.1));
    let anchor = ov[0].0;
    result[anchor].anchor = true;
    let order: Vec<usize> = ov.iter().skip(1).map(|&(i, _)| i).collect();

    // Two rounds: first place each camera against those already placed, then refine each against
    // all the others.
    let mut placed = vec![anchor];
    for round in 0..2 {
        for &i in &order {
            let mut g = Grid::new(70.0);
            let refs: Vec<usize> = if round == 0 { placed.clone() } else { (0..n).filter(|&j| j != i).collect() };
            for &j in &refs {
                g.add(&cams[j], poses[j].0, poses[j].1, poses[j].2);
            }
            let rough = (cams[i].rough.yaw, cams[i].rough.x, cams[i].rough.z);
            let start = if round == 0 { rough } else { poses[i] };
            let (span, step) = if round == 0 { ((20.0, 5.0), (2.0, 0.5)) } else { ((4.0, 1.0), (1.0, 0.25)) };
            let coarse = search(&g, &pts[i], start, span, step);
            let fine = search(&g, &pts[i], (coarse.1, coarse.2, coarse.3), (2.0, 0.5), (0.25, 0.1));
            // Too little in common with the others → keep the user's placement.
            if fine.4 < 0.08 {
                if round == 0 {
                    placed.push(i);
                }
                continue;
            }
            poses[i] = (fine.1, fine.2, fine.3);
            let (s, seen) = g.score(&pts[i], fine.1, fine.2, fine.3);
            result[i].agreement = Some(((s / (pts[i].len() as f64 * seen).max(1.0)) + 0.35) / 1.35);
            if round == 0 {
                placed.push(i);
            }
        }
    }
    for (i, r) in result.iter_mut().enumerate() {
        r.yaw = poses[i].0.rem_euclid(360.0);
        r.x = poses[i].1;
        r.z = poses[i].2;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cameras seeing the same L-shaped scene; the second is placed 8° and 1.5 m off.
    #[test]
    fn recovers_a_misplaced_camera() {
        let mut world = Vec::new();
        for i in 0..200 {
            let t = i as f32 * 0.1;
            world.push((t - 10.0, 8.0, Label::Building)); // a wall along x
            world.push((5.0, t - 5.0, Label::Fence)); // a fence along z
            for j in 0..10 {
                world.push((t - 10.0, 2.0 + j as f32 * 0.5, Label::Road)); // a road band
                world.push((-4.0 + j as f32 * 0.3, t - 5.0, Label::Grass));
            }
        }
        // Express world points in a camera's local frame given its true pose.
        let local = |yaw: f64, x: f64, z: f64| -> Vec<(f32, f32, Label)> {
            let (s, c) = yaw.to_radians().sin_cos();
            world
                .iter()
                .map(|&(wx, wz, l)| {
                    let (dx, dz) = (wx as f64 - x, wz as f64 - z);
                    // inverse of: w = R(yaw)·l + t
                    ((dx * c - dz * s) as f32, (dx * s + dz * c) as f32, l)
                })
                .collect()
        };
        let a = AlignCam { id: "a".into(), rough: Pose { x: 0.0, z: 0.0, height: 3.0, yaw: 0.0, pitch: 30.0 }, local: local(0.0, 0.0, 0.0) };
        let truth = (30.0, 2.0, -1.0);
        let b = AlignCam {
            id: "b".into(),
            rough: Pose { x: truth.1 + 1.5, z: truth.2 - 0.8, height: 3.0, yaw: truth.0 - 8.0, pitch: 30.0 },
            local: local(truth.0, truth.1, truth.2),
        };
        let r = align(&[a, b]);
        let rb = r.iter().find(|x| x.id == "b").unwrap();
        let ra = r.iter().find(|x| x.id == "a").unwrap();
        // Either may be the anchor; check the relative pose is right.
        let (ay, by) = (ra.yaw, rb.yaw);
        let dyaw = ((by - ay - truth.0 + 540.0) % 360.0) - 180.0;
        assert!(dyaw.abs() < 1.0, "relative yaw off by {dyaw}");
        if ra.anchor {
            assert!((rb.x - truth.1).abs() < 0.3 && (rb.z - truth.2).abs() < 0.3, "{rb:?}");
        }
    }
}

// ---------- alignment from matched points ----------

/// p_A = s·R(θ)·p_B + t in the horizontal plane (R turns headings anticlockwise by θ, as yaw
/// does); `ty` is the median vertical offset of B's ground relative to A's.
#[derive(Debug, Clone, Copy)]
pub struct Sim {
    pub theta: f64,
    pub s: f64,
    pub tx: f64,
    pub tz: f64,
    pub ty: f64,
}

impl Sim {
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let (sn, cs) = self.theta.to_radians().sin_cos();
        [self.tx + self.s * (p[0] * cs + p[1] * sn), self.tz + self.s * (-p[0] * sn + p[1] * cs), self.ty + self.s * p[2]]
    }
}

/// Robustly fit B→A from 3D point pairs given as [x, z, height]. Points live in each camera's
/// levelled local frame (camera at the origin, heading 0). Returns (sim, inlier indices, rms m).
pub fn solve_similarity(pairs: &[([f64; 3], [f64; 3])]) -> Option<(Sim, Vec<usize>, f64)> {
    if pairs.len() < 6 {
        return None;
    }
    // Complex numbers w = z + i·x make "rotate heading by θ, scale by s" a multiplication.
    let c = |p: &[f64; 3]| (p[1], p[0]);
    let tol = |a: &[f64; 3]| 0.35 + 0.05 * a[0].hypot(a[1]);
    let inliers_of = |k: (f64, f64), t: (f64, f64)| -> Vec<usize> {
        pairs
            .iter()
            .enumerate()
            .filter(|(_, (a, b))| {
                let (bz, bx) = c(b);
                let (pz, px) = (k.0 * bz - k.1 * bx + t.0, k.0 * bx + k.1 * bz + t.1);
                let (az, ax) = c(a);
                (pz - az).hypot(px - ax) < tol(a)
            })
            .map(|(i, _)| i)
            .collect()
    };
    let fit = |idx: &[usize]| -> ((f64, f64), (f64, f64)) {
        let n = idx.len() as f64;
        let (mut ma, mut mb) = ((0.0, 0.0), (0.0, 0.0));
        for &i in idx {
            let (a, b) = (c(&pairs[i].0), c(&pairs[i].1));
            ma = (ma.0 + a.0 / n, ma.1 + a.1 / n);
            mb = (mb.0 + b.0 / n, mb.1 + b.1 / n);
        }
        let (mut num, mut den) = ((0.0, 0.0), 0.0);
        for &i in idx {
            let (a, b) = (c(&pairs[i].0), c(&pairs[i].1));
            let (a, b) = ((a.0 - ma.0, a.1 - ma.1), (b.0 - mb.0, b.1 - mb.1));
            // a · conj(b)
            num = (num.0 + a.0 * b.0 + a.1 * b.1, num.1 + a.1 * b.0 - a.0 * b.1);
            den += b.0 * b.0 + b.1 * b.1;
        }
        let k = (num.0 / den, num.1 / den);
        let t = (ma.0 - (k.0 * mb.0 - k.1 * mb.1), ma.1 - (k.0 * mb.1 + k.1 * mb.0));
        (k, t)
    };

    let mut rng = 0x2545f4914f6cdd1du64;
    let mut next = |n: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % n as u64) as usize
    };
    let mut best: Vec<usize> = Vec::new();
    for _ in 0..3000 {
        let (i, j) = (next(pairs.len()), next(pairs.len()));
        if i == j {
            continue;
        }
        let (a1, a2, b1, b2) = (c(&pairs[i].0), c(&pairs[j].0), c(&pairs[i].1), c(&pairs[j].1));
        let (da, db) = ((a2.0 - a1.0, a2.1 - a1.1), (b2.0 - b1.0, b2.1 - b1.1));
        let nb = db.0 * db.0 + db.1 * db.1;
        if nb < 1.0 {
            continue; // points too close together to fix rotation
        }
        let k = ((da.0 * db.0 + da.1 * db.1) / nb, (da.1 * db.0 - da.0 * db.1) / nb);
        let s = k.0.hypot(k.1);
        if !(0.7..=1.4).contains(&s) {
            continue; // single-image depth is never this far off
        }
        let t = (a1.0 - (k.0 * b1.0 - k.1 * b1.1), a1.1 - (k.0 * b1.1 + k.1 * b1.0));
        let inl = inliers_of(k, t);
        if inl.len() > best.len() {
            best = inl;
        }
    }
    if best.len() < 6 {
        return None;
    }
    // Refine: least squares on inliers, re-select, repeat.
    let (mut k, mut t) = fit(&best);
    for _ in 0..3 {
        let inl = inliers_of(k, t);
        if inl.len() < 6 {
            break;
        }
        best = inl;
        (k, t) = fit(&best);
    }
    let s = k.0.hypot(k.1);
    let theta = k.1.atan2(k.0).to_degrees();
    let mut dy: Vec<f64> = best.iter().map(|&i| pairs[i].0[2] - s * pairs[i].1[2]).collect();
    dy.sort_by(f64::total_cmp);
    let sim = Sim { theta, s, tz: t.0, tx: t.1, ty: dy[dy.len() / 2] };
    let rms = (best
        .iter()
        .map(|&i| {
            let p = sim.apply(pairs[i].1);
            (p[0] - pairs[i].0[0]).powi(2) + (p[1] - pairs[i].0[1]).powi(2)
        })
        .sum::<f64>()
        / best.len() as f64)
        .sqrt();
    Some((sim, best, rms))
}

/// A camera's world placement: world = s·R(yaw)·local + (x, z).
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub yaw: f64,
    pub x: f64,
    pub z: f64,
    pub s: f64,
}

impl Placement {
    /// Place a child camera given the edge sim (parent-local ← child-local).
    pub fn child(&self, e: &Sim) -> Placement {
        let (sn, cs) = self.yaw.to_radians().sin_cos();
        let (tx, tz) = (self.s * (e.tx * cs + e.tz * sn), self.s * (-e.tx * sn + e.tz * cs));
        Placement { yaw: (self.yaw + e.theta).rem_euclid(360.0), x: self.x + tx, z: self.z + tz, s: self.s * e.s }
    }
}

#[cfg(test)]
mod sim_tests {
    use super::*;

    #[test]
    fn similarity_round_trip_with_outliers() {
        let truth = Sim { theta: 23.0, s: 1.12, tx: 4.0, tz: -7.0, ty: 0.3 };
        let mut pairs = Vec::new();
        for i in 0..60 {
            let b = [(i % 7) as f64 * 1.3 - 4.0, 3.0 + (i / 7) as f64 * 1.1, (i % 3) as f64 * 0.5];
            pairs.push((truth.apply(b), b));
        }
        for i in 0..25 {
            pairs.push(([i as f64, -(i as f64), 0.0], [(i * 3 % 11) as f64, (i * 5 % 13) as f64, 0.0]));
        }
        let (s, inl, rms) = solve_similarity(&pairs).unwrap();
        assert!((s.theta - 23.0).abs() < 0.01 && (s.s - 1.12).abs() < 1e-3, "{s:?}");
        assert!((s.tx - 4.0).abs() < 1e-3 && (s.tz + 7.0).abs() < 1e-3 && (s.ty - 0.3).abs() < 1e-6, "{s:?}");
        assert!(inl.len() >= 60 && rms < 1e-6);
        // Composition: a child placed via the edge lands its points where the parent sees them.
        let parent = Placement { yaw: 40.0, x: 1.0, z: 2.0, s: 0.9 };
        let child = parent.child(&s);
        let b = [2.0, 5.0, 0.0];
        let world = |p: &Placement, l: [f64; 3]| {
            let (sn, cs) = p.yaw.to_radians().sin_cos();
            (p.x + p.s * (l[0] * cs + l[1] * sn), p.z + p.s * (-l[0] * sn + l[1] * cs))
        };
        let via_parent = world(&parent, s.apply(b));
        let direct = world(&child, b);
        assert!((via_parent.0 - direct.0).abs() < 1e-9 && (via_parent.1 - direct.1).abs() < 1e-9);
    }
}
