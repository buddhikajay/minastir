//! Simple shapes from one camera's structure columns: straight walls and fences (buildings get a
//! block behind their visible wall), round trees and shrubs, thin poles.
//!
//! Works in the camera's levelled frame (camera at the origin), where every column was seen along
//! a ray from the origin. That matters: single-image depth smears objects *along* the ray, so a
//! tree's width is read across the ray (its angular span × range) and its centre sits one radius
//! behind the surface the camera saw, not at the centroid of the smear.

use crate::{groundmap::Structure, segment::Label};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Wall {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub top: f32,
    pub label: Label,
    /// Depth of the building block behind the wall (0 for fences); it extends to the side of
    /// the wall away from the camera.
    pub depth: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Tree {
    pub x: f32,
    pub z: f32,
    pub radius: f32,
    pub top: f32,
    /// Low plant without a trunk.
    pub shrub: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Pole {
    pub x: f32,
    pub z: f32,
    pub top: f32,
}

/// A wheelie bin standing on the ground.
#[derive(Debug, Clone, Serialize)]
pub struct Bin {
    pub x: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Shapes {
    pub walls: Vec<Wall>,
    pub trees: Vec<Tree>,
    pub poles: Vec<Pole>,
    pub bins: Vec<Bin>,
}

/// Assumed depth of a house behind the wall a camera sees.
const BLOCK_DEPTH: f32 = 6.0;

pub fn extract(cols: &[Structure], column: f64) -> Shapes {
    let mut out = Shapes::default();
    let walls: Vec<&Structure> = cols.iter().filter(|c| matches!(c.label, Label::Building | Label::Fence)).collect();
    let veg: Vec<&Structure> = cols.iter().filter(|c| matches!(c.label, Label::Tree | Label::Plant)).collect();
    let poles: Vec<&Structure> = cols.iter().filter(|c| c.label == Label::Pole).collect();
    let bins: Vec<&Structure> = cols.iter().filter(|c| c.label == Label::Bin).collect();
    for group in components(&bins, column) {
        out.bins.extend(bins_from(&group));
    }

    out.walls = fit_walls(&walls, column);
    for group in components(&veg, column) {
        out.trees.extend(trees_from(&group));
    }
    for group in components(&poles, column) {
        let n = group.len() as f32;
        let (x, z) = (group.iter().map(|c| c.x).sum::<f32>() / n, group.iter().map(|c| c.z).sum::<f32>() / n);
        let top = group.iter().map(|c| c.top).fold(0.0, f32::max);
        if top > 1.5 {
            out.poles.push(Pole { x, z, top: top.min(9.0) });
        }
    }
    out
}

/// Groups of columns touching each other (8-neighbourhood on the column grid).
fn components<'a>(cols: &[&'a Structure], column: f64) -> Vec<Vec<&'a Structure>> {
    let key = |c: &Structure| (((c.x as f64) / column).floor() as i64, ((c.z as f64) / column).floor() as i64);
    let index: std::collections::HashMap<(i64, i64), usize> = cols.iter().enumerate().map(|(i, c)| (key(c), i)).collect();
    let mut seen = vec![false; cols.len()];
    let mut out = Vec::new();
    for start in 0..cols.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut group = Vec::new();
        while let Some(i) = stack.pop() {
            group.push(cols[i]);
            let (kx, kz) = key(cols[i]);
            for dx in -1..=1 {
                for dz in -1..=1 {
                    if let Some(&j) = index.get(&(kx + dx, kz + dz)) {
                        if !seen[j] {
                            seen[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        out.push(group);
    }
    out
}

/// Bins in a group of bin columns: a row of bins side by side is split by width (~0.7 m each),
/// placed just behind the surface the camera saw. Groups whose height isn't bin-like are dropped.
fn bins_from(group: &[&Structure]) -> Vec<Bin> {
    let mut tops: Vec<f32> = group.iter().map(|c| c.top).collect();
    tops.sort_by(f32::total_cmp);
    let top = tops[tops.len() / 2];
    if !(0.5..=1.7).contains(&top) {
        return Vec::new();
    }
    let ang: Vec<f32> = group.iter().map(|c| c.x.atan2(c.z)).collect();
    let mut rng: Vec<f32> = group.iter().map(|c| c.x.hypot(c.z)).collect();
    let (a0, a1) = (ang.iter().cloned().fold(f32::MAX, f32::min), ang.iter().cloned().fold(f32::MIN, f32::max));
    rng.sort_by(f32::total_cmp);
    let near = rng[rng.len() / 10];
    let width = (a1 - a0) * rng[rng.len() / 2] + 0.4;
    let n = ((width / 0.72).round() as usize).clamp(1, 6);
    let r = near + 0.35;
    (0..n)
        .map(|i| {
            let a = a0 + (a1 - a0) * (i as f32 + 0.5) / n as f32;
            Bin { x: r * a.sin(), z: r * a.cos() }
        })
        .collect()
}

/// One tree per group, or several when the group is too wide for one canopy (a row of trees).
fn trees_from(group: &[&Structure]) -> Vec<Tree> {
    if group.len() < 3 {
        return Vec::new();
    }
    // Angle across the camera's view and range along it.
    let ang: Vec<f32> = group.iter().map(|c| c.x.atan2(c.z)).collect();
    let rng: Vec<f32> = group.iter().map(|c| c.x.hypot(c.z)).collect();
    let (a0, a1) = (ang.iter().cloned().fold(f32::MAX, f32::min), ang.iter().cloned().fold(f32::MIN, f32::max));
    let mut sorted = rng.clone();
    sorted.sort_by(f32::total_cmp);
    let near = sorted[sorted.len() / 10];
    let width = (a1 - a0) * sorted[sorted.len() / 2] + 0.4;
    let parts = (width / 7.0).ceil().max(1.0) as usize;
    let mut out = Vec::new();
    for p in 0..parts {
        let (lo, hi) = (a0 + (a1 - a0) * p as f32 / parts as f32, a0 + (a1 - a0) * (p + 1) as f32 / parts as f32);
        let idx: Vec<usize> = (0..group.len()).filter(|&i| ang[i] >= lo && (ang[i] < hi || (p + 1 == parts && ang[i] <= hi))).collect();
        if idx.len() < 2 {
            continue;
        }
        let mut tops: Vec<f32> = idx.iter().map(|&i| group[i].top).collect();
        tops.sort_by(f32::total_cmp);
        let top = tops[(tops.len() * 9 / 10).min(tops.len() - 1)].min(14.0);
        let plants = idx.iter().filter(|&&i| group[i].label == Label::Plant).count();
        let shrub = top < 2.2 || plants * 2 > idx.len();
        let w = width / parts as f32;
        let mut radius = (w / 2.0).clamp(0.4, 3.2);
        if shrub {
            radius = radius.min(1.1).min(top.max(0.4));
        } else {
            radius = radius.min(top * 0.45);
        }
        let mid = (lo + hi) / 2.0;
        let mut nears: Vec<f32> = idx.iter().map(|&i| rng[i]).collect();
        nears.sort_by(f32::total_cmp);
        let r = nears[nears.len() / 10].max(near) + radius;
        out.push(Tree { x: r * mid.sin(), z: r * mid.cos(), radius, top: top.max(radius), shrub });
    }
    out
}

/// Straight wall segments by sequential RANSAC over all wall columns; segments split where the
/// columns along a line have gaps.
fn fit_walls(cols: &[&Structure], column: f64) -> Vec<Wall> {
    let mut left: Vec<&Structure> = cols.to_vec();
    let mut out = Vec::new();
    let mut rng = 0x2545f4914f6cdd1du64;
    let mut next = |n: usize| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng % n as u64) as usize
    };
    // Depth smears a wall into a band along the camera's rays; one line should take it all.
    let tol = (column * 2.2) as f32;
    for _ in 0..40 {
        if left.len() < 4 {
            break;
        }
        // Best line through two columns.
        let mut best: Option<([f32; 2], [f32; 2], usize)> = None;
        for _ in 0..300 {
            let (p, q) = (left[next(left.len())], left[next(left.len())]);
            let d = [q.x - p.x, q.z - p.z];
            let len = d[0].hypot(d[1]);
            if len < 0.8 {
                continue;
            }
            let u = [d[0] / len, d[1] / len];
            let count = left.iter().filter(|c| ((c.x - p.x) * u[1] - (c.z - p.z) * u[0]).abs() < tol).count();
            if best.as_ref().is_none_or(|b| count > b.2) {
                best = Some(([p.x, p.z], u, count));
            }
        }
        let Some((o, u, count)) = best else { break };
        if count < 4 {
            break;
        }
        let (inl, rest): (Vec<&Structure>, Vec<&Structure>) =
            left.iter().partition(|c| ((c.x - o[0]) * u[1] - (c.z - o[1]) * u[0]).abs() < tol);
        left = rest;
        // Refit the line through its inliers (least squares on the perpendicular offset).
        let n = inl.len() as f32;
        let (cx, cz) = (inl.iter().map(|c| c.x).sum::<f32>() / n, inl.iter().map(|c| c.z).sum::<f32>() / n);
        let (mut sxx, mut sxz, mut szz) = (0.0, 0.0, 0.0);
        for c in &inl {
            let (dx, dz) = (c.x - cx, c.z - cz);
            sxx += dx * dx;
            sxz += dx * dz;
            szz += dz * dz;
        }
        let th = 0.5 * (2.0 * sxz).atan2(sxx - szz);
        let u = [th.cos(), th.sin()];
        // Snap to the band's near edge: the surface the camera actually saw.
        let nrm = { let n = [-u[1], u[0]]; if n[0] * cx + n[1] * cz > 0.0 { [-n[0], -n[1]] } else { n } }; // towards camera
        let mut offs: Vec<f32> = inl.iter().map(|c| (c.x - cx) * nrm[0] + (c.z - cz) * nrm[1]).collect();
        offs.sort_by(f32::total_cmp);
        let shift = offs[offs.len() * 3 / 4];
        let (cx, cz) = (cx + nrm[0] * shift, cz + nrm[1] * shift);
        // Split along the line at gaps.
        let mut along: Vec<(f32, &Structure)> = inl.iter().map(|c| ((c.x - cx) * u[0] + (c.z - cz) * u[1], *c)).collect();
        along.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut seg: Vec<(f32, &Structure)> = Vec::new();
        let flush = |seg: &[(f32, &Structure)], out: &mut Vec<Wall>| {
            if seg.len() < 3 {
                return;
            }
            let (t0, t1) = (seg[0].0 - column as f32 / 2.0, seg[seg.len() - 1].0 + column as f32 / 2.0);
            if t1 - t0 < 1.2 {
                return;
            }
            let mut tops: Vec<f32> = seg.iter().map(|s| s.1.top).collect();
            tops.sort_by(f32::total_cmp);
            let fences = seg.iter().filter(|s| s.1.label == Label::Fence).count();
            let label = if fences * 2 > seg.len() { Label::Fence } else { Label::Building };
            let top = if label == Label::Fence { tops[tops.len() / 2].clamp(0.8, 2.4) } else { tops[tops.len() * 3 / 4].clamp(0.8, 9.0) };
            out.push(Wall {
                a: [cx + u[0] * t0, cz + u[1] * t0],
                b: [cx + u[0] * t1, cz + u[1] * t1],
                top,
                label,
                depth: if label == Label::Building { BLOCK_DEPTH } else { 0.0 },
            });
        };
        for item in along {
            if let Some(last) = seg.last() {
                if item.0 - last.0 > 1.5 {
                    flush(&seg, &mut out);
                    seg.clear();
                }
            }
            seg.push(item);
        }
        flush(&seg, &mut out);
    }
    let mut walls = merge_parallel(out);
    // The camera's own house: a wall the camera is mounted on (or right beside) is seen edge-on
    // and can't be placed sensibly. Short or near walls (porches, letterbox boards, fixtures)
    // stay plain walls; only longer walls further off get a house block.
    walls.retain(|w| seg_dist(w.a, w.b) >= 2.0);
    for w in &mut walls {
        let len = (w.b[0] - w.a[0]).hypot(w.b[1] - w.a[1]);
        if w.depth > 0.0 && (len < 3.5 || seg_dist(w.a, w.b) < 6.0) {
            w.depth = 0.0;
        }
        if w.depth > 0.0 {
            w.top = w.top.max(2.4); // a house is at least one storey
        }
    }
    walls
}

/// Distance from the camera (origin) to a segment.
fn seg_dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = (d[0] * d[0] + d[1] * d[1]).max(1e-9);
    let t = (-(a[0] * d[0] + a[1] * d[1]) / l2).clamp(0.0, 1.0);
    (a[0] + t * d[0]).hypot(a[1] + t * d[1])
}

/// Merge wall segments that are nearly parallel, close side by side and overlapping or touching
/// along their length — leftovers of one wall split by depth noise. Keeps the one nearer the
/// camera (the visible surface), stretched over both.
fn merge_parallel(mut walls: Vec<Wall>) -> Vec<Wall> {
    loop {
        let mut merged = false;
        'outer: for i in 0..walls.len() {
            for j in i + 1..walls.len() {
                if walls[i].label != walls[j].label {
                    continue;
                }
                let (a, b) = (&walls[i], &walls[j]);
                let da = [a.b[0] - a.a[0], a.b[1] - a.a[1]];
                let la = da[0].hypot(da[1]);
                let u = [da[0] / la, da[1] / la];
                let db = [b.b[0] - b.a[0], b.b[1] - b.a[1]];
                let lb = db[0].hypot(db[1]);
                if (u[0] * db[0] + u[1] * db[1]).abs() / lb < 0.95 {
                    continue; // > ~18° apart
                }
                let side = |p: [f32; 2]| (p[0] - a.a[0]) * u[1] - (p[1] - a.a[1]) * u[0];
                let along = |p: [f32; 2]| (p[0] - a.a[0]) * u[0] + (p[1] - a.a[1]) * u[1];
                if side(b.a).abs() > 1.6 || side(b.b).abs() > 1.6 {
                    continue;
                }
                let (b0, b1) = (along(b.a).min(along(b.b)), along(b.a).max(along(b.b)));
                if b0 > la + 1.0 || b1 < -1.0 {
                    continue;
                }
                // Keep the nearer line; extend it over both.
                let mid = |w: &Wall| ((w.a[0] + w.b[0]) / 2.0).hypot((w.a[1] + w.b[1]) / 2.0);
                let (keep, other) = if mid(a) <= mid(b) { (i, j) } else { (j, i) };
                let k = walls[keep].clone();
                let o = walls[other].clone();
                let kd = [k.b[0] - k.a[0], k.b[1] - k.a[1]];
                let kl = kd[0].hypot(kd[1]);
                let ku = [kd[0] / kl, kd[1] / kl];
                let t = |p: [f32; 2]| (p[0] - k.a[0]) * ku[0] + (p[1] - k.a[1]) * ku[1];
                let (t0, t1) = (0f32.min(t(o.a)).min(t(o.b)), kl.max(t(o.a)).max(t(o.b)));
                let w = Wall {
                    a: [k.a[0] + ku[0] * t0, k.a[1] + ku[1] * t0],
                    b: [k.a[0] + ku[0] * t1, k.a[1] + ku[1] * t1],
                    top: k.top.max(o.top),
                    label: k.label,
                    depth: k.depth,
                };
                walls[keep] = w;
                walls.remove(other);
                merged = true;
                break 'outer;
            }
        }
        if !merged {
            return walls;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(x: f32, z: f32, top: f32, label: Label) -> Structure {
        Structure { x, z, top, label }
    }

    #[test]
    fn straight_wall_becomes_one_segment() {
        let cols: Vec<Structure> = (0..20).map(|i| col(-3.8 + i as f32 * 0.4, 10.2, 5.0, Label::Building)).collect();
        let s = extract(&cols, 0.4);
        assert_eq!(s.walls.len(), 1, "{:?}", s.walls);
        let w = &s.walls[0];
        assert!((w.a[1] - 10.2).abs() < 0.1 && (w.b[1] - 10.2).abs() < 0.1);
        assert!(((w.b[0] - w.a[0]).abs() - 8.0).abs() < 0.5);
    }

    #[test]
    fn row_of_bins_splits_into_bins() {
        // Three wheelie bins side by side, ~2.1 m wide, 1.05 m tall, 8 m in front of the camera.
        let cols: Vec<Structure> = (0..5).map(|i| col(-0.8 + i as f32 * 0.4, 8.2, 1.05, Label::Bin)).collect();
        let s = extract(&cols, 0.4);
        assert_eq!(s.bins.len(), 3, "{:?}", s.bins);
        assert!(s.bins.iter().all(|b| b.z > 8.2 && b.z < 9.0), "{:?}", s.bins);
        // Something bin-coloured but 3 m tall isn't a bin.
        let tall: Vec<Structure> = (0..3).map(|i| col(i as f32 * 0.4, 6.2, 3.0, Label::Bin)).collect();
        assert!(extract(&tall, 0.4).bins.is_empty());
    }

    #[test]
    fn tree_sits_behind_its_visible_surface() {
        // A canopy's front, smeared along the ray from the camera (depth noise), at ~10 m.
        let mut cols = Vec::new();
        for i in 0..8 {
            for k in 0..4 {
                cols.push(col(-1.4 + i as f32 * 0.4, 10.2 + k as f32 * 0.4, 6.0, Label::Tree));
            }
        }
        let s = extract(&cols, 0.4);
        assert_eq!(s.trees.len(), 1, "{:?}", s.trees);
        let t = &s.trees[0];
        assert!(t.radius > 1.0 && t.radius < 2.5, "{t:?}");
        assert!(t.z > 10.7, "{t:?}");
        assert!(!t.shrub);
    }
}

/// A wall in world coordinates, after merging cameras.
#[derive(Debug, Clone, Serialize)]
pub struct WorldWall {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub top: f32,
    pub label: Label,
    pub depth: f32,
    /// Unit direction (world x, z) the building block extends in.
    pub back: [f32; 2],
    /// Distance from the camera that saw it (for preferring near views); not sent.
    #[serde(skip)]
    range: f32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct WorldShapes {
    pub walls: Vec<WorldWall>,
    pub trees: Vec<Tree>,
    pub poles: Vec<Pole>,
    pub bins: Vec<Bin>,
}

/// Every camera's shapes in world coordinates, merged where cameras saw the same thing.
pub fn merge_world(per_camera: &[(&Shapes, minastir_core::onvif::Pose)]) -> WorldShapes {
    // Trees and poles carry a weight: near views are measured far better.
    let mut trees: Vec<(Tree, f32)> = Vec::new();
    let mut poles: Vec<(Pole, f32)> = Vec::new();
    let mut bins: Vec<(Bin, f32)> = Vec::new();
    let mut walls: Vec<WorldWall> = Vec::new();
    for (s, p) in per_camera {
        let (sn, cs) = (p.yaw as f32).to_radians().sin_cos();
        let to_world = |x: f32, z: f32| [p.x as f32 + x * cs + z * sn, p.z as f32 - x * sn + z * cs];
        let weight = |x: f32, z: f32| 1.0 / (1.0 + (x.hypot(z) / 8.0).powi(2));
        for t in &s.trees {
            let w = to_world(t.x, t.z);
            trees.push((Tree { x: w[0], z: w[1], ..t.clone() }, weight(t.x, t.z)));
        }
        for q in &s.poles {
            let w = to_world(q.x, q.z);
            poles.push((Pole { x: w[0], z: w[1], top: q.top }, weight(q.x, q.z)));
        }
        for b in &s.bins {
            let w = to_world(b.x, b.z);
            bins.push((Bin { x: w[0], z: w[1] }, weight(b.x, b.z)));
        }
        for w in &s.walls {
            // Block side: away from this camera (the origin, locally).
            let (dx, dz) = (w.b[0] - w.a[0], w.b[1] - w.a[1]);
            let len = dx.hypot(dz).max(1e-6);
            let mut n = [-dz / len, dx / len];
            if n[0] * (w.a[0] + w.b[0]) + n[1] * (w.a[1] + w.b[1]) < 0.0 {
                n = [-n[0], -n[1]];
            }
            let back = [n[0] * cs + n[1] * sn, -n[0] * sn + n[1] * cs];
            walls.push(WorldWall {
                a: to_world(w.a[0], w.a[1]),
                b: to_world(w.b[0], w.b[1]),
                top: w.top,
                label: w.label,
                depth: w.depth,
                back,
                range: seg_dist(w.a, w.b),
            });
        }
    }

    // Trees: greedily merge canopies that overlap substantially.
    let mut merged_trees: Vec<(Tree, f32)> = Vec::new();
    trees.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (t, w) in trees {
        let hit = merged_trees.iter_mut().find(|(m, _)| (m.x - t.x).hypot(m.z - t.z) < 0.8 * m.radius.max(t.radius) + 0.5);
        match hit {
            Some((m, mw)) => {
                let k = w / (*mw + w);
                m.x += (t.x - m.x) * k;
                m.z += (t.z - m.z) * k;
                m.radius += (t.radius - m.radius) * k;
                m.top += (t.top - m.top) * k;
                m.shrub = m.shrub && t.shrub;
                *mw += w;
            }
            None => merged_trees.push((t, w)),
        }
    }
    let mut merged_poles: Vec<(Pole, f32)> = Vec::new();
    for (q, w) in poles {
        match merged_poles.iter_mut().find(|(m, _)| (m.x - q.x).hypot(m.z - q.z) < 1.0) {
            Some((m, mw)) => {
                let k = w / (*mw + w);
                m.x += (q.x - m.x) * k;
                m.z += (q.z - m.z) * k;
                m.top = m.top.max(q.top);
                *mw += w;
            }
            None => merged_poles.push((q, w)),
        }
    }

    // Bins: the same bin seen by two cameras lands within ~0.6 m.
    let mut merged_bins: Vec<(Bin, f32)> = Vec::new();
    bins.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (b, w) in bins {
        match merged_bins.iter_mut().find(|(m, _)| (m.x - b.x).hypot(m.z - b.z) < 0.6) {
            Some((m, mw)) => {
                let k = w / (*mw + w);
                m.x += (b.x - m.x) * k;
                m.z += (b.z - m.z) * k;
                *mw += w;
            }
            None => merged_bins.push((b, w)),
        }
    }

    WorldShapes {
        bins: merged_bins.into_iter().map(|(b, _)| b).collect(),
        walls: merge_world_walls(walls),
        trees: merged_trees.into_iter().map(|(t, _)| t).collect(),
        poles: merged_poles.into_iter().map(|(q, _)| q).collect(),
    }
}

/// Merge near-parallel, side-by-side, overlapping walls; the nearer camera's line wins.
fn merge_world_walls(mut walls: Vec<WorldWall>) -> Vec<WorldWall> {
    loop {
        let mut merged = false;
        'outer: for i in 0..walls.len() {
            for j in i + 1..walls.len() {
                let (a, b) = (&walls[i], &walls[j]);
                if a.label != b.label {
                    continue;
                }
                let da = [a.b[0] - a.a[0], a.b[1] - a.a[1]];
                let la = da[0].hypot(da[1]).max(1e-6);
                let u = [da[0] / la, da[1] / la];
                let db = [b.b[0] - b.a[0], b.b[1] - b.a[1]];
                let lb = db[0].hypot(db[1]).max(1e-6);
                if (u[0] * db[0] + u[1] * db[1]).abs() / lb < 0.95 {
                    continue;
                }
                let side = |p: [f32; 2]| (p[0] - a.a[0]) * u[1] - (p[1] - a.a[1]) * u[0];
                let along = |p: [f32; 2]| (p[0] - a.a[0]) * u[0] + (p[1] - a.a[1]) * u[1];
                if side(b.a).abs() > 1.2 || side(b.b).abs() > 1.2 {
                    continue;
                }
                let (b0, b1) = (along(b.a).min(along(b.b)), along(b.a).max(along(b.b)));
                if b0 > la + 1.0 || b1 < -1.0 {
                    continue;
                }
                let (keep, other) = if a.range <= b.range { (i, j) } else { (j, i) };
                let (k, o) = (walls[keep].clone(), walls[other].clone());
                let kd = [k.b[0] - k.a[0], k.b[1] - k.a[1]];
                let kl = kd[0].hypot(kd[1]).max(1e-6);
                let ku = [kd[0] / kl, kd[1] / kl];
                let t = |p: [f32; 2]| (p[0] - k.a[0]) * ku[0] + (p[1] - k.a[1]) * ku[1];
                let (t0, t1) = (0f32.min(t(o.a)).min(t(o.b)), kl.max(t(o.a)).max(t(o.b)));
                walls[keep] = WorldWall {
                    a: [k.a[0] + ku[0] * t0, k.a[1] + ku[1] * t0],
                    b: [k.a[0] + ku[0] * t1, k.a[1] + ku[1] * t1],
                    top: k.top.max(o.top),
                    depth: k.depth.max(o.depth),
                    ..k
                };
                walls.remove(other);
                merged = true;
                break 'outer;
            }
        }
        if !merged {
            return walls;
        }
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use minastir_core::onvif::Pose;

    #[test]
    fn same_tree_from_two_cameras_merges() {
        // A tree at world (0, 10): camera A at the origin looking +z; camera B at (10, 10)
        // looking -x (yaw -90°), seeing it 10 m ahead.
        let a = Shapes { trees: vec![Tree { x: 0.2, z: 10.0, radius: 2.0, top: 6.0, shrub: false }], ..Default::default() };
        let b = Shapes { trees: vec![Tree { x: 0.0, z: 10.3, radius: 2.4, top: 6.5, shrub: false }], ..Default::default() };
        let pa = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: 3.0, pitch: 30.0 };
        let pb = Pose { x: 10.0, z: 10.0, yaw: -90.0, height: 3.0, pitch: 30.0 };
        let w = merge_world(&[(&a, pa), (&b, pb)]);
        assert_eq!(w.trees.len(), 1, "{:?}", w.trees);
        assert!((w.trees[0].x).abs() < 0.6 && (w.trees[0].z - 10.0).abs() < 0.6, "{:?}", w.trees[0]);
    }
}
