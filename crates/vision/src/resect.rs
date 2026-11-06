//! Lens resection: solve one lens's field of view, radial distortion, orientation and position
//! from pixels whose 3D positions another (already placed) camera has measured.
//!
//! Single-image depth gets the field of view of wide, distorted security lenses badly wrong
//! (a UniFi G4 Bullet's ~87° came out as ~64°), and every 3D point that camera produces inherits
//! the error. Seen from a second camera, the same points pin the lens down: we minimise the
//! reprojection error of the other camera's 3D points in this camera's image (Levenberg–Marquardt
//! with a robust loss, several starts).

/// One correspondence: normalised pixel (u, v) inside this lens's image, and the point's 3D
/// position (x, up, z) in the reference frame.
#[derive(Debug, Clone, Copy)]
pub struct Obs {
    pub u: f64,
    pub v: f64,
    pub p: [f64; 3],
}

/// Pinhole lens + one radial distortion term, posed in the reference frame. Angles in degrees;
/// yaw/pitch follow the app's conventions (yaw 0 looks along +Z, pitch is downward tilt).
#[derive(Debug, Clone, Copy)]
pub struct LensPose {
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
    pub c: [f64; 3],
    /// tan(hfov / 2) of the undistorted pinhole.
    pub tan_h: f64,
    /// Radial distortion on coordinates normalised to the image half-width (negative = barrel).
    pub k1: f64,
}

impl LensPose {
    pub fn hfov(&self) -> f64 {
        (2.0 * self.tan_h.atan()).to_degrees()
    }

    /// Normalised pixel of a reference-frame point. `aspect` = lens image width / height.
    pub fn project(&self, aspect: f64, p: [f64; 3]) -> Option<(f64, f64)> {
        let d = [p[0] - self.c[0], p[1] - self.c[1], p[2] - self.c[2]];
        let (sy, cy) = self.yaw.to_radians().sin_cos();
        let (x1, z1) = (d[0] * cy - d[2] * sy, d[0] * sy + d[2] * cy);
        let (sp, cp) = self.pitch.to_radians().sin_cos();
        let (y2, z2) = (d[1] * cp + z1 * sp, -d[1] * sp + z1 * cp);
        let (sr, cr) = self.roll.to_radians().sin_cos();
        let (x3, y3) = (x1 * cr + y2 * sr, -x1 * sr + y2 * cr);
        if z2 < 0.3 {
            return None;
        }
        // Local frame is x left, y up; the image's are right and down.
        let (nx, ny) = (-x3 / z2 / self.tan_h, -y3 / z2 / self.tan_h);
        let f = 1.0 + self.k1 * (nx * nx + ny * ny);
        Some((0.5 + nx * f / 2.0, 0.5 + ny * f * aspect / 2.0))
    }

    fn to_vec(self) -> [f64; N] {
        [self.yaw, self.pitch, self.roll, self.c[0], self.c[1], self.c[2], self.tan_h.ln(), self.k1]
    }

    fn from_vec(v: &[f64; N]) -> Self {
        LensPose { yaw: v[0], pitch: v[1], roll: v[2], c: [v[3], v[4], v[5]], tan_h: v[6].exp(), k1: v[7] }
    }
}

const N: usize = 8;
/// Robust-loss scale: residuals are in image heights (0.006 ≈ 3 px on a 540-line image).
const HUBER: f64 = 0.006;

pub struct Resection {
    pub pose: LensPose,
    pub inliers: Vec<usize>,
    /// RMS reprojection error of the inliers, in image heights.
    pub rms: f64,
}

/// Solve the lens from correspondences, starting near `init`. Tries a few fields of view and both
/// roll signs, keeps the best robust fit, then refines on its inliers (with distortion when the
/// data supports it).
pub fn resect(obs: &[Obs], init: LensPose, aspect: f64) -> Option<Resection> {
    if obs.len() < 12 {
        return None;
    }
    let all: Vec<usize> = (0..obs.len()).collect();
    let mut best: Option<(f64, LensPose)> = None;
    for mul in [1.0, 1.25, 1.5, 1.8] {
        for roll in [init.roll, -init.roll] {
            for dyaw in [0.0, -15.0, 15.0] {
                let start = LensPose { tan_h: init.tan_h * mul, roll, yaw: init.yaw + dyaw, k1: 0.0, ..init };
                let p = lm(obs, &all, start, aspect, false, 60);
                let c = cost(obs, &all, &p, aspect);
                if best.as_ref().is_none_or(|b| c < b.0) {
                    best = Some((c, p));
                }
            }
        }
    }
    let (_, mut pose) = best?;
    let mut inliers: Vec<usize>;
    for (tol, distort) in [(0.03, false), (0.018, true), (0.012, true)] {
        inliers = (0..obs.len()).filter(|&i| residual(&obs[i], &pose, aspect).is_some_and(|r| r.0.hypot(r.1) < tol)).collect();
        if inliers.len() < 10 {
            return None;
        }
        pose = lm(obs, &inliers, pose, aspect, distort && inliers.len() >= 30, 100);
    }
    inliers = (0..obs.len()).filter(|&i| residual(&obs[i], &pose, aspect).is_some_and(|r| r.0.hypot(r.1) < 0.012)).collect();
    let rms = (inliers.iter().map(|&i| residual(&obs[i], &pose, aspect).map_or(0.0, |r| r.0 * r.0 + r.1 * r.1)).sum::<f64>()
        / inliers.len().max(1) as f64)
        .sqrt();
    Some(Resection { pose, inliers, rms })
}

fn residual(o: &Obs, p: &LensPose, aspect: f64) -> Option<(f64, f64)> {
    let (u, v) = p.project(aspect, o.p)?;
    Some(((u - o.u) * aspect, v - o.v))
}

fn huber(r: f64) -> f64 {
    let a = r.abs();
    if a <= HUBER {
        r * r / 2.0
    } else {
        HUBER * (a - HUBER / 2.0)
    }
}

fn cost(obs: &[Obs], idx: &[usize], p: &LensPose, aspect: f64) -> f64 {
    idx.iter()
        .map(|&i| match residual(&obs[i], p, aspect) {
            Some((a, b)) => huber(a) + huber(b),
            None => huber(1.0) * 2.0,
        })
        .sum()
}

/// Levenberg–Marquardt with Huber weights (IRLS) and a numeric Jacobian.
fn lm(obs: &[Obs], idx: &[usize], start: LensPose, aspect: f64, distort: bool, iters: usize) -> LensPose {
    let free: Vec<usize> = (0..N).filter(|&k| distort || k != 7).collect();
    let steps = [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5];
    let mut x = start.to_vec();
    let mut c = cost(obs, idx, &LensPose::from_vec(&x), aspect);
    let mut lambda = 1e-3;
    for _ in 0..iters {
        let p = LensPose::from_vec(&x);
        let m = free.len();
        let mut jtj = vec![vec![0.0; m]; m];
        let mut jtr = vec![0.0; m];
        for &i in idx {
            let Some(r0) = residual(&obs[i], &p, aspect) else { continue };
            let mut jac = vec![(0.0, 0.0); m];
            let mut ok = true;
            for (j, &k) in free.iter().enumerate() {
                let mut xp = x;
                xp[k] += steps[k];
                let mut xm = x;
                xm[k] -= steps[k];
                match (residual(&obs[i], &LensPose::from_vec(&xp), aspect), residual(&obs[i], &LensPose::from_vec(&xm), aspect)) {
                    (Some(a), Some(b)) => jac[j] = ((a.0 - b.0) / (2.0 * steps[k]), (a.1 - b.1) / (2.0 * steps[k])),
                    _ => ok = false,
                }
            }
            if !ok {
                continue;
            }
            for (r, comp) in [(r0.0, 0usize), (r0.1, 1)] {
                let w = if r.abs() <= HUBER { 1.0 } else { HUBER / r.abs() };
                for a in 0..m {
                    let ja = if comp == 0 { jac[a].0 } else { jac[a].1 };
                    jtr[a] += w * ja * r;
                    for b in 0..m {
                        let jb = if comp == 0 { jac[b].0 } else { jac[b].1 };
                        jtj[a][b] += w * ja * jb;
                    }
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut a = jtj.clone();
            for k in 0..m {
                a[k][k] += lambda * (jtj[k][k] + 1e-9);
            }
            let Some(delta) = solve(a, jtr.iter().map(|v| -v).collect()) else { break };
            let mut xn = x;
            for (j, &k) in free.iter().enumerate() {
                xn[k] += delta[j];
            }
            xn[6] = xn[6].clamp(0.2f64.ln(), 4.0f64.ln()); // 22°–152°
            xn[7] = xn[7].clamp(-0.6, 0.6);
            let cn = cost(obs, idx, &LensPose::from_vec(&xn), aspect);
            if cn < c {
                let rel = (c - cn) / c.max(1e-12);
                x = xn;
                c = cn;
                lambda = (lambda * 0.3).max(1e-7);
                improved = rel > 1e-7;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    LensPose::from_vec(&x)
}

/// Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-15 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_wide_lens_from_noisy_points() {
        let truth = LensPose { yaw: 204.0, pitch: 18.0, roll: -3.0, c: [4.9, 2.4, -5.0], tan_h: (44f64).to_radians().tan(), k1: -0.08 };
        let aspect = 16.0 / 9.0;
        let mut obs = Vec::new();
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        for i in 0..120 {
            // Points on the ground and on a few raised things in front of the camera.
            let (fx, fz) = (rnd() * 16.0, 3.0 + (rnd() + 0.5) * 18.0);
            let (s, c) = 204f64.to_radians().sin_cos();
            let p = [4.9 + fz * s + fx * c, if i % 4 == 0 { 1.0 + rnd() } else { 0.0 }, -5.0 + fz * c - fx * s];
            let Some((u, v)) = truth.project(aspect, p) else { continue };
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let (u, v) = if i % 9 == 0 { (rnd() + 0.5, rnd() + 0.5) } else { (u + rnd() * 0.004, v + rnd() * 0.004) };
            obs.push(Obs { u, v, p });
        }
        let init = LensPose { yaw: 215.0, pitch: 15.0, roll: 2.0, c: [4.0, 2.0, -4.0], tan_h: (32f64).to_radians().tan(), k1: 0.0 };
        let r = resect(&obs, init, aspect).unwrap();
        assert!((r.pose.hfov() - 88.0).abs() < 3.0, "hfov {}", r.pose.hfov());
        assert!((r.pose.yaw - 204.0).abs() < 1.5 && (r.pose.pitch - 18.0).abs() < 1.5, "{:?}", r.pose);
        assert!((r.pose.c[0] - 4.9).abs() < 0.4 && (r.pose.c[2] + 5.0).abs() < 0.4, "{:?}", r.pose);
    }
}
