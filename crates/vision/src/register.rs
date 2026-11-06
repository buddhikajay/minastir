//! Place overlapping cameras relative to each other from matched image points.
//!
//! For every pair of cameras (every lens pair), ALIKED+LightGlue matches points between their
//! people-free backgrounds; each matched pixel's 3D position comes from that camera's scene map.
//! A robust similarity fit then gives the pair's relative heading, position and depth scale. The
//! camera with the most matches stays where the user put it; the others are chained from it along
//! the strongest links. Cameras with nothing in common keep their hand placement.

use crate::{
    align::{solve_similarity, Placement, Sim},
    geometry::CameraModel,
    matcher::{Img, Matcher},
};
use anyhow::Result;
use minastir_core::{
    calib::{Alignment, CameraCalibration},
    onvif::Pose,
};

pub struct RegCam {
    pub id: String,
    pub label: String,
    pub rough: Pose,
    pub calibration: CameraCalibration,
    /// Model at the zero pose (x = z = yaw = 0, scale 1): points come out in the camera's
    /// levelled local frame.
    pub local: CameraModel,
    pub image: image::RgbImage,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    /// p_a = sim(p_b)
    pub sim: Sim,
    pub matches: usize,
    pub inliers: usize,
    pub rms: f64,
}

#[derive(Debug, Clone)]
pub struct Registered {
    pub id: String,
    pub pose: Pose,
    pub scale: f64,
    pub alignment: Option<Alignment>,
}

/// Burned-in overlays (logo, date, camera name, vendor watermark) sit at fixed screen positions
/// and "match" perfectly between unrelated cameras. Normalised full-frame boxes.
fn in_overlay(u: f64, v: f64) -> bool {
    (u < 0.12 && v < 0.13) || ((0.3..0.7).contains(&u) && v < 0.07) || (u > 0.85 && v > 0.9)
}

/// A raw correspondence between two cameras: full-frame normalised pixels and lens indices.
#[derive(Debug, Clone, Copy)]
pub struct RawMatch {
    pub la: usize,
    pub a: (f64, f64),
    pub lb: usize,
    pub b: (f64, f64),
}

/// Matched pixels for every camera pair (a < b), overlays removed.
pub fn collect_matches(cams: &[RegCam], matcher: &mut Matcher, min_score: f32) -> Result<Vec<((usize, usize), Vec<RawMatch>)>> {
    let mut out = Vec::new();
    for a in 0..cams.len() {
        for b in a + 1..cams.len() {
            let mut list = Vec::new();
            for (ia_l, la) in cams[a].calibration.lenses.iter().enumerate() {
                for (ib_l, lb) in cams[b].calibration.lenses.iter().enumerate() {
                    let crop = |c: &RegCam, u0: f64, u1: f64| {
                        let (w, h) = (c.image.width(), c.image.height());
                        let x0 = (u0 * w as f64) as u32;
                        let x1 = ((u1 * w as f64) as u32).min(w);
                        image::imageops::crop_imm(&c.image, x0, 0, x1 - x0, h).to_image()
                    };
                    let (ia, ib) = (crop(&cams[a], la.u0, la.u1), crop(&cams[b], lb.u0, lb.u1));
                    let ms = matcher.match_pair(
                        &Img { rgb: ia.as_raw(), width: ia.width() as usize, height: ia.height() as usize },
                        &Img { rgb: ib.as_raw(), width: ib.width() as usize, height: ib.height() as usize },
                        min_score,
                    )?;
                    for m in ms {
                        // Same spot in both images ⇒ an overlay, not the scene.
                        if (m.a.0 - m.b.0).abs() < 0.02 && (m.a.1 - m.b.1).abs() < 0.02 {
                            continue;
                        }
                        let ua = la.u0 + m.a.0 * (la.u1 - la.u0);
                        let ub = lb.u0 + m.b.0 * (lb.u1 - lb.u0);
                        if in_overlay(ua, m.a.1) || in_overlay(ub, m.b.1) {
                            continue;
                        }
                        list.push(RawMatch { la: ia_l, a: (ua, m.a.1), lb: ib_l, b: (ub, m.b.1) });
                    }
                }
            }
            out.push(((a, b), list));
        }
    }
    Ok(out)
}

/// 3D point pairs (in each camera's local frame) for one camera pair's matches.
pub fn depth_pairs(cams: &[RegCam], a: usize, b: usize, ms: &[RawMatch]) -> Vec<([f64; 3], [f64; 3])> {
    ms.iter()
        .filter_map(|m| {
            let (pa, pb) = (cams[a].local.point_at(m.a.0, m.a.1)?, cams[b].local.point_at(m.b.0, m.b.1)?);
            // Ignore far points: depth error grows with distance.
            (pa[0].hypot(pa[1]) < 35.0 && pb[0].hypot(pb[1]) < 35.0).then_some((pa, pb))
        })
        .collect()
}

pub fn match_edges(cams: &[RegCam], matcher: &mut Matcher, min_score: f32, log: &mut dyn FnMut(String)) -> Result<Vec<Edge>> {
    let all = collect_matches(cams, matcher, min_score)?;
    Ok(edges_from(cams, &all, log))
}

pub fn edges_from(cams: &[RegCam], all: &[((usize, usize), Vec<RawMatch>)], log: &mut dyn FnMut(String)) -> Vec<Edge> {
    let mut edges = Vec::new();
    for ((a, b), ms) in all {
        let (a, b) = (*a, *b);
        let raw = ms.len();
        let pairs = depth_pairs(cams, a, b, ms);
        let solved = solve_similarity(&pairs);
        match &solved {
            Some((sim, inl, rms)) => log(format!(
                "{} ↔ {}: {raw} matches, {} with depth, {} agree (rms {:.2} m) → turn {:.1}°, scale {:.3}",
                cams[a].label, cams[b].label, pairs.len(), inl.len(), rms, sim.theta, sim.s
            )),
            None => log(format!("{} ↔ {}: {raw} matches, {} with depth — no consistent fit", cams[a].label, cams[b].label, pairs.len())),
        }
        if let Some((sim, inl, rms)) = solved {
            // Only trust a strong, physically plausible fit: enough agreeing points, a good
            // share of them, tight residuals, and a depth-scale correction single-image depth
            // could actually need.
            if inl.len() >= 20 && inl.len() * 3 >= pairs.len() && rms <= 0.45 && (0.85..=1.18).contains(&sim.s) {
                edges.push(Edge { a, b, sim, matches: pairs.len(), inliers: inl.len(), rms });
            }
        }
    }
    edges
}

/// Inverse of a similarity (p_b = inv(p_a)).
pub fn invert(s: &Sim) -> Sim {
    let (sn, cs) = (-s.theta).to_radians().sin_cos();
    let k = 1.0 / s.s;
    let (x, z) = (-s.tx, -s.tz);
    Sim { theta: -s.theta, s: k, tx: k * (x * cs + z * sn), tz: k * (-x * sn + z * cs), ty: -s.ty * k }
}

/// Chain placements from the best-connected camera along the strongest links.
pub fn place(cams: &[RegCam], edges: &[Edge]) -> Vec<Registered> {
    let n = cams.len();
    let mut out: Vec<Registered> = cams
        .iter()
        .map(|c| Registered { id: c.id.clone(), pose: c.rough, scale: 1.0, alignment: None })
        .collect();
    if edges.is_empty() {
        return out;
    }
    let weight = |i: usize| edges.iter().filter(|e| e.a == i || e.b == i).map(|e| e.inliers).sum::<usize>();
    let anchor = (0..n).max_by_key(|&i| weight(i)).unwrap();
    let rough = |c: &RegCam| [c.rough.yaw, c.rough.x, c.rough.z];
    let mut placed: Vec<Option<Placement>> = vec![None; n];
    placed[anchor] = Some(Placement { yaw: cams[anchor].rough.yaw, x: cams[anchor].rough.x, z: cams[anchor].rough.z, s: 1.0 });
    out[anchor].alignment = Some(Alignment { anchor: true, reference: None, inliers: weight(anchor), rms: 0.0, rough: rough(&cams[anchor]) });
    // Prim's algorithm on inlier counts.
    loop {
        let mut best: Option<(usize, usize, Sim, &Edge)> = None; // (parent, child, child→parent sim)
        for e in edges {
            let (from, to, sim) = match (placed[e.a].is_some(), placed[e.b].is_some()) {
                (true, false) => (e.a, e.b, e.sim),
                (false, true) => (e.b, e.a, invert(&e.sim)),
                _ => continue,
            };
            if best.as_ref().is_none_or(|b| e.inliers > b.3.inliers) {
                best = Some((from, to, sim, e));
            }
        }
        let Some((from, to, sim, e)) = best else { break };
        let p = placed[from].unwrap().child(&sim);
        placed[to] = Some(p);
        let c = &cams[to];
        out[to].pose = Pose {
            yaw: (p.yaw * 10.0).round() / 10.0,
            x: (p.x * 100.0).round() / 100.0,
            z: (p.z * 100.0).round() / 100.0,
            height: ((c.calibration.height * p.s) * 100.0).round() / 100.0,
            pitch: c.rough.pitch,
        };
        out[to].scale = p.s;
        out[to].alignment =
            Some(Alignment { anchor: false, reference: Some(cams[from].id.clone()), inliers: e.inliers, rms: e.rms, rough: rough(c) });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_undoes_similarity() {
        let s = Sim { theta: 31.0, s: 1.1, tx: 2.0, tz: -3.0, ty: 0.4 };
        let p = [1.5, 4.0, 0.7];
        let q = invert(&s).apply(s.apply(p));
        assert!((q[0] - p[0]).abs() < 1e-9 && (q[1] - p[1]).abs() < 1e-9 && (q[2] - p[2]).abs() < 1e-9, "{q:?}");
    }
}
