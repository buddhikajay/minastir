//! Barrel distortion of wide security lenses, measured from the scene itself.
//!
//! A wide lens squeezes the edges of its picture towards the centre, so straight kerbs bow into
//! arcs and anything near the edge sits at a steeper angle than a flat (pinhole) lens would put
//! it. Depth models assume a flat lens: on a distorted frame they stretch the far edges, which
//! puts a car parked across the street 50 m away instead of 20.
//!
//! Model: one radial term. A pixel `r` from the centre (as a fraction of the half-diagonal) is
//! where a flat lens would show it at `r · (1 + k1·r²)`.
//!
//! Measurement (plumb line): kerbs are straight in the world, and a flat lens keeps straight
//! lines straight, so the right `k1` is the one that makes the edges between road and
//! footpath / grass straightest. Found by sweeping `k1`, and at each value finding the straight
//! lines among the undistorted kerb pixels (RANSAC); then refined by least squares on the pixels
//! of the lines found. Nothing about a particular street is assumed: curved driveways simply
//! don't form lines and drop out.

use crate::segment::{Label, LabelMap};

/// The undistortion factor at radius² `r2` (half-diagonal units).
pub fn scale(k1: f64, r2: f64) -> f64 {
    1.0 + k1 * r2
}

/// Distorted radius whose undistorted radius is `ru` (both in half-diagonal units): the inverse
/// of `r · (1 + k1·r²)`, by Newton's method.
pub fn distorted_radius(k1: f64, ru: f64) -> f64 {
    if k1 == 0.0 || ru == 0.0 {
        return ru;
    }
    let mut r = ru;
    for _ in 0..30 {
        let f = r + k1 * r * r * r - ru;
        let d = 1.0 + 3.0 * k1 * r * r;
        if d.abs() < 1e-9 {
            break;
        }
        let step = f / d;
        r -= step;
        if step.abs() < 1e-12 {
            break;
        }
    }
    r
}

/// A measured distortion.
#[derive(Clone, Debug)]
pub struct Fit {
    pub k1: f64,
    /// RMS distance of the kerb pixels from their straight lines after undistortion (pixels of
    /// the measuring grid, `GRID` wide).
    pub rms: f64,
    /// Straight kerb lines found, and how many pixels lie on them.
    pub lines: usize,
    pub pixels: usize,
}

/// Width of the grid the labels are sampled on.
const GRID: usize = 896;
/// Distortions considered. Security lenses are barrel (k1 > 0); a little negative room lets a
/// lens with none show up as ~0 rather than pinned at the bound.
const K_RANGE: (f64, f64) = (-0.2, 1.2);
/// A kerb pixel belongs to a line within this distance (grid pixels).
const TOL: f64 = 1.5;

/// Measures `k1` from where road meets footpath, grass or earth in a frame's labels. None when
/// the frame doesn't show enough straight kerb to tell, or straightening doesn't help.
pub fn from_kerbs(labels: &LabelMap) -> Option<Fit> {
    let (gw, gh) = (GRID, ((GRID as f64 * labels.height as f64 / labels.width as f64).round() as usize).max(2));
    let at = |x: usize, y: usize| labels.at((x as f64 + 0.5) / gw as f64, (y as f64 + 0.5) / gh as f64);
    let grid: Vec<Label> = (0..gw * gh).map(|i| at(i % gw, i / gw)).collect();
    let beside = |l: Label| matches!(l, Label::Sidewalk | Label::Grass | Label::Ground);
    // Kerb pixels: road with footpath / grass / earth right next to it. Not the frame's border.
    let mut kerb: Vec<(f64, f64)> = Vec::new();
    for y in 1..gh - 1 {
        for x in 1..gw - 1 {
            if grid[y * gw + x] != Label::Road {
                continue;
            }
            let n = [grid[(y - 1) * gw + x], grid[(y + 1) * gw + x], grid[y * gw + x - 1], grid[y * gw + x + 1]];
            if n.iter().any(|&l| beside(l)) {
                kerb.push((x as f64 + 0.5, y as f64 + 0.5));
            }
        }
    }
    if kerb.len() < 150 {
        return None;
    }
    let (cx, cy, half) = (gw as f64 / 2.0, gh as f64 / 2.0, (gw as f64).hypot(gh as f64) / 2.0);
    // Undistorted, in grid pixels about the centre.
    let undistort = |k1: f64, pts: &[(f64, f64)]| -> Vec<(f64, f64)> {
        pts.iter()
            .map(|&(x, y)| {
                let (dx, dy) = ((x - cx) / half, (y - cy) / half);
                let s = scale(k1, dx * dx + dy * dy);
                (dx * s * half, dy * s * half)
            })
            .collect()
    };
    let min_line = (kerb.len() / 25).max(60);
    let min_span = gw as f64 * 0.2;
    let score_at = |k1: f64| -> (f64, Vec<Vec<usize>>) {
        let pts = undistort(k1, &kerb);
        let lines = straight_lines(&pts, min_line, min_span);
        let score = lines.iter().map(|l| l.1).sum();
        (score, lines.into_iter().map(|l| l.0).collect())
    };
    let steps = ((K_RANGE.1 - K_RANGE.0) / 0.02).round() as usize;
    let (mut best_k, mut best) = (0.0, score_at(0.0));
    let flat = best.0;
    for s in 0..=steps {
        let k = K_RANGE.0 + s as f64 * 0.02;
        let r = score_at(k);
        if r.0 > best.0 {
            (best_k, best) = (k, r);
        }
    }
    // Straightening must clearly help: otherwise the lens is flat enough, or these edges aren't
    // straight kerbs.
    if best.1.is_empty() || best.0 < flat * 1.1 + 20.0 {
        return None;
    }
    // Refine on the pixels of the lines found: least squares over k1 (golden section).
    let groups: Vec<Vec<(f64, f64)>> = best.1.iter().map(|g| g.iter().map(|&i| kerb[i]).collect()).collect();
    let cost = |k1: f64| -> f64 { groups.iter().map(|g| line_residual(&undistort(k1, g))).sum() };
    let (mut a, mut b) = (best_k - 0.04, best_k + 0.04);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..40 {
        let (c, d) = (b - g * (b - a), a + g * (b - a));
        if cost(c) < cost(d) {
            b = d;
        } else {
            a = c;
        }
    }
    let k1 = ((a + b) / 2.0).clamp(K_RANGE.0, K_RANGE.1);
    let pixels: usize = groups.iter().map(Vec::len).sum();
    Some(Fit { k1, rms: (cost(k1) / pixels as f64).sqrt(), lines: groups.len(), pixels })
}

/// Sum of squared perpendicular distances of points from their best-fit line.
fn line_residual(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len() as f64;
    let (mx, my) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 / n, a.1 + p.1 / n));
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (x, y) = (p.0 - mx, p.1 - my);
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    // Smallest eigenvalue of the scatter matrix.
    let tr = sxx + syy;
    let det = sxx * syy - sxy * sxy;
    (tr / 2.0 - ((tr * tr / 4.0 - det).max(0.0)).sqrt()).max(0.0)
}

/// Straight lines among `pts` by sequential RANSAC: each line's point indices and a soft score
/// (points near the line count more). Lines need `min_pts` points spanning `min_span`.
fn straight_lines(pts: &[(f64, f64)], min_pts: usize, min_span: f64) -> Vec<(Vec<usize>, f64)> {
    let mut left: Vec<usize> = (0..pts.len()).collect();
    let mut out = Vec::new();
    // Deterministic sampling, so a calibration gives the same answer twice.
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rand = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    for _ in 0..4 {
        if left.len() < min_pts {
            break;
        }
        let mut best: Option<(f64, (f64, f64, f64))> = None;
        for _ in 0..300 {
            let (p, q) = (pts[left[rand(left.len())]], pts[left[rand(left.len())]]);
            let (dx, dy) = (q.0 - p.0, q.1 - p.1);
            let len = dx.hypot(dy);
            if len < min_span / 4.0 {
                continue;
            }
            // Unit normal (a, b), offset c: a·x + b·y = c.
            let (a, b) = (-dy / len, dx / len);
            let c = a * p.0 + b * p.1;
            let s: f64 = left.iter().map(|&i| soft(a * pts[i].0 + b * pts[i].1 - c)).sum();
            if best.is_none_or(|(bs, _)| s > bs) {
                best = Some((s, (a, b, c)));
            }
        }
        let Some((_, (a, b, c))) = best else { break };
        let (on, off): (Vec<usize>, Vec<usize>) = left.iter().partition(|&&i| (a * pts[i].0 + b * pts[i].1 - c).abs() < TOL);
        // How far the line's pixels stretch along it.
        let along: Vec<f64> = on.iter().map(|&i| b * pts[i].0 - a * pts[i].1).collect();
        let span = along.iter().cloned().fold(f64::MIN, f64::max) - along.iter().cloned().fold(f64::MAX, f64::min);
        if on.len() < min_pts || span < min_span {
            break;
        }
        let score = on.iter().map(|&i| soft(a * pts[i].0 + b * pts[i].1 - c)).sum();
        out.push((on, score));
        left = off;
    }
    out
}

fn soft(d: f64) -> f64 {
    (1.0 - (d / TOL).powi(2)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distorted_radius_inverts_the_model() {
        for k1 in [0.0, 0.2, 0.42, 0.9] {
            for r in [0.0, 0.3, 0.7, 1.0] {
                let ru = r * scale(k1, r * r);
                assert!((distorted_radius(k1, ru) - r).abs() < 1e-9, "k1 {k1} r {r}");
            }
        }
    }

    /// A synthetic frame: two straight kerbs (road between, grass either side), bent by a known
    /// barrel distortion. The plumb-line fit should find that distortion again.
    #[test]
    fn kerbs_reveal_the_distortion() {
        let (w, h) = (1600usize, 900usize);
        let k1 = 0.4;
        let half = (w as f64).hypot(h as f64) / 2.0;
        let mut labels = vec![Label::Grass; w * h];
        for y in 0..h {
            for x in 0..w {
                // Where this distorted pixel would be through a flat lens.
                let (dx, dy) = ((x as f64 + 0.5 - w as f64 / 2.0) / half, (y as f64 + 0.5 - h as f64 / 2.0) / half);
                let s = scale(k1, dx * dx + dy * dy);
                let (ux, uy) = (dx * s * half, dy * s * half);
                // Road: between two slanted straight lines, above the centre.
                let top = -330.0 + 0.12 * ux;
                let bot = -170.0 + 0.10 * ux;
                if uy > top && uy < bot {
                    labels[y * w + x] = Label::Road;
                }
            }
        }
        let fit = from_kerbs(&LabelMap { width: w, height: h, labels }).expect("a fit");
        assert!((fit.k1 - k1).abs() < 0.04, "{fit:?}");
        assert!(fit.lines >= 2, "{fit:?}");
    }
}
