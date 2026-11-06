//! Learned association: "are these two sightings the same person?", as a calibrated probability.
//!
//! The model is trained by the `assoc_train` example from a harvest log (MINASTIR_HARVEST) and
//! saved as JSON; the image tracker uses it in place of hand-set matching thresholds. The
//! features are defined here, once, so training and tracking compute exactly the same thing.

use crate::imgtrack::{outfit_match, OUTFIT};
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Where the people model lives, under the models directory.
pub const PEOPLE_MODEL_FILE: &str = "assoc_people.json";

pub const FEATURES: [&str; 10] = ["dt", "same_cam", "iou", "centre_dist", "outfit", "ground_dist", "ground_dist_rel", "size_ratio", "stature_diff", "has_stature"];
pub const N: usize = FEATURES.len();

/// One sighting, as the features see it.
#[derive(Clone, Copy, Debug)]
pub struct Sighting<'a> {
    pub t: f64,
    pub cam: usize,
    /// Normalised x0, y0, x1, y1.
    pub bbox: [f32; 4],
    pub outfit: &'a [f32; OUTFIT],
    /// Ground position (metres) and distance from its camera, when placed.
    pub ground: Option<(f64, f64, f64)>,
    /// Height measured in this sighting (metres), when the feet were seen. A single
    /// measurement, never a track's average: averages would make a track's own pairs agree
    /// by construction.
    pub stature: Option<f32>,
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let i = w * h;
    let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
    if u > 0.0 {
        i / u
    } else {
        0.0
    }
}

/// The features of a pair; `aspect` is the image's width / height.
pub fn features(a: &Sighting, b: &Sighting, aspect: f32) -> [f64; N] {
    let same = a.cam == b.cam;
    let h = (a.bbox[3] - a.bbox[1]).max(1e-3) as f64;
    let (ca, cb) = (((a.bbox[0] + a.bbox[2]) / 2.0, (a.bbox[1] + a.bbox[3]) / 2.0), ((b.bbox[0] + b.bbox[2]) / 2.0, (b.bbox[1] + b.bbox[3]) / 2.0));
    let cd = if same { (((ca.0 - cb.0) * aspect).hypot(ca.1 - cb.1) as f64) / h } else { 0.0 };
    // Unplaced sightings: a neutral, middling distance.
    let (gd, gr) = match (a.ground, b.ground) {
        (Some(p), Some(q)) => {
            let d = (p.0 - q.0).hypot(p.1 - q.1);
            (d, d / (0.5 + 0.08 * (p.2 + q.2)))
        }
        _ => (3.0, 3.0),
    };
    let stature = a.stature.zip(b.stature).map(|(p, q)| (p as f64, q as f64));
    let sa = (a.bbox[3] - a.bbox[1]).max(1e-3) as f64;
    let sb = (b.bbox[3] - b.bbox[1]).max(1e-3) as f64;
    [
        (a.t - b.t).abs(),
        same as u8 as f64,
        if same { iou(&a.bbox, &b.bbox) as f64 } else { 0.0 },
        cd.min(10.0),
        outfit_match(a.outfit, b.outfit) as f64,
        gd.min(30.0),
        gr.min(30.0),
        if same { (sa / sb).ln().abs() } else { 0.0 },
        stature.map_or(0.0, |(p, q)| (p - q).abs() / ((p + q) / 2.0)),
        stature.is_some() as u8 as f64,
    ]
}

/// A small network: N features → hidden tanh → 1 sigmoid, with input standardisation.
#[derive(Clone, Debug)]
pub struct Net {
    pub mean: [f64; N],
    pub sd: [f64; N],
    pub w1: Vec<[f64; N]>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: f64,
}

/// Mean log-loss of `net` over `xs` / `ys`, each class weighted to half the total (pairs are
/// mostly "different": unweighted, the loss would mostly reward saying no).
pub fn balanced_loss(net: &Net, xs: &[[f64; N]], ys: &[f64]) -> f64 {
    let (mut l, mut n) = ([0.0; 2], [0usize; 2]);
    for (x, &y) in xs.iter().zip(ys) {
        let p = net.prob(x);
        let k = (y > 0.5) as usize;
        l[k] -= if k == 1 { p.max(1e-6).ln() } else { (1.0 - p).max(1e-6).ln() };
        n[k] += 1;
    }
    (l[0] / n[0].max(1) as f64 + l[1] / n[1].max(1) as f64) / 2.0
}

impl Net {
    /// A fresh, randomly initialised network whose standardisation fits `xs`.
    pub fn new(xs: &[[f64; N]], hidden: usize, seed: u64) -> Self {
        let n = xs.len().max(1) as f64;
        let mut mean = [0.0; N];
        let mut sd = [0.0; N];
        for x in xs {
            for k in 0..N {
                mean[k] += x[k] / n;
            }
        }
        for x in xs {
            for k in 0..N {
                sd[k] += (x[k] - mean[k]).powi(2) / n;
            }
        }
        sd.iter_mut().for_each(|s| *s = s.sqrt().max(1e-6));
        let mut seed = seed;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.6
        };
        Self { mean, sd, w1: (0..hidden).map(|_| [(); N].map(|_| rnd())).collect(), b1: vec![0.0; hidden], w2: (0..hidden).map(|_| rnd()).collect(), b2: 0.0 }
    }

    fn norm(&self, x: &[f64; N]) -> [f64; N] {
        let mut z = [0.0; N];
        for k in 0..N {
            z[k] = (x[k] - self.mean[k]) / self.sd[k];
        }
        z
    }

    fn forward(&self, x: &[f64; N]) -> (Vec<f64>, f64) {
        let z = self.norm(x);
        let h: Vec<f64> = self.w1.iter().zip(&self.b1).map(|(w, b)| (b + (0..N).map(|k| w[k] * z[k]).sum::<f64>()).tanh()).collect();
        let o = self.b2 + h.iter().zip(&self.w2).map(|(a, b)| a * b).sum::<f64>();
        (h, 1.0 / (1.0 + (-o).exp()))
    }

    /// Probability the pair is the same person.
    pub fn prob(&self, x: &[f64; N]) -> f64 {
        self.forward(x).1
    }

    /// One pass of plain SGD on log-loss, in an order shuffled from `seed`.
    pub fn epoch(&mut self, xs: &[[f64; N]], ys: &[f64], lr: f64, seed: &mut u64) {
        let mut order: Vec<usize> = (0..xs.len()).collect();
        for i in (1..order.len()).rev() {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            order.swap(i, (*seed >> 33) as usize % (i + 1));
        }
        for &i in &order {
            let z = self.norm(&xs[i]);
            let (h, p) = self.forward(&xs[i]);
            let g = p - ys[i];
            for j in 0..h.len() {
                let gh = g * self.w2[j] * (1.0 - h[j] * h[j]);
                self.w2[j] -= lr * g * h[j];
                for k in 0..N {
                    self.w1[j][k] -= lr * gh * z[k];
                }
                self.b1[j] -= lr * gh;
            }
            self.b2 -= lr * g;
        }
    }

    /// Trains on `xs` / `ys` and stops where the held-out `val` set stops improving: keeps the
    /// weights of the best epoch, and gives up after `patience` epochs without a better one.
    /// Returns that best validation loss and its epoch.
    pub fn train(&mut self, xs: &[[f64; N]], ys: &[f64], val: (&[[f64; N]], &[f64]), max_epochs: usize, patience: usize, seed: u64) -> (f64, usize) {
        let mut shuffle = seed ^ 0x9e37_79b9_7f4a_7c15;
        let (mut best, mut best_at, mut kept) = (balanced_loss(self, val.0, val.1), 0, self.clone());
        for e in 1..=max_epochs {
            self.epoch(xs, ys, 0.02, &mut shuffle);
            let l = balanced_loss(self, val.0, val.1);
            if l < best {
                (best, best_at, kept) = (l, e, self.clone());
            } else if e - best_at >= patience {
                break;
            }
        }
        *self = kept;
        (best, best_at)
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "mean": self.mean, "sd": self.sd, "w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2 })
    }

    fn from_json(v: &serde_json::Value) -> Result<Self> {
        let arr = |x: &serde_json::Value| -> Result<[f64; N]> {
            let v: Vec<f64> = x.as_array().context("expected an array")?.iter().filter_map(|y| y.as_f64()).collect();
            v.try_into().map_err(|_| anyhow::anyhow!("wrong length"))
        };
        let vec = |x: &serde_json::Value| -> Vec<f64> { x.as_array().into_iter().flatten().filter_map(|y| y.as_f64()).collect() };
        let w1 = v["w1"].as_array().context("w1")?.iter().map(arr).collect::<Result<Vec<_>>>()?;
        let (b1, w2) = (vec(&v["b1"]), vec(&v["w2"]));
        if b1.len() != w1.len() || w2.len() != w1.len() {
            bail!("malformed network");
        }
        Ok(Self { mean: arr(&v["mean"])?, sd: arr(&v["sd"])?, w1, b1, w2, b2: v["b2"].as_f64().context("b2")? })
    }
}

/// The association model: several networks trained from different random starts, averaged.
/// One small network's answer depends noticeably on its starting weights; the average doesn't.
#[derive(Clone, Debug)]
pub struct Model {
    pub nets: Vec<Net>,
}

impl Model {
    pub fn single(net: Net) -> Self {
        Self { nets: vec![net] }
    }

    /// Probability the pair is the same person: the members' mean.
    pub fn prob(&self, x: &[f64; N]) -> f64 {
        self.nets.iter().map(|n| n.prob(x)).sum::<f64>() / self.nets.len().max(1) as f64
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "features": FEATURES, "members": self.nets.iter().map(Net::to_json).collect::<Vec<_>>() })
    }

    /// Reads a model written by `to_json` (or a single network, the older format).
    pub fn load(path: &Path) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)?;
        let names: Vec<&str> = v["features"].as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect();
        if names != FEATURES {
            bail!("{} was trained on other features ({names:?})", path.display());
        }
        let nets = match v["members"].as_array() {
            Some(m) => m.iter().map(Net::from_json).collect::<Result<Vec<_>>>(),
            None => Net::from_json(&v).map(|n| vec![n]),
        }
        .with_context(|| format!("reading {}", path.display()))?;
        if nets.is_empty() {
            bail!("{} has no networks", path.display());
        }
        Ok(Self { nets })
    }
}

/// Minimum-cost one-to-one assignment (Hungarian algorithm) for a rows × cols cost matrix.
/// Returns, per row, the column it got (None when there are more rows than columns).
pub fn assign(cost: &[Vec<f64>]) -> Vec<Option<usize>> {
    let rows = cost.len();
    let cols = cost.first().map_or(0, |r| r.len());
    if rows == 0 || cols == 0 {
        return vec![None; rows];
    }
    // Square it up with a large dummy cost.
    let n = rows.max(cols);
    let big = 1e6;
    let c = |i: usize, j: usize| if i < rows && j < cols { cost[i][j] } else { big };
    let (mut u, mut v, mut p, mut way) = (vec![0.0; n + 1], vec![0.0; n + 1], vec![0usize; n + 1], vec![0usize; n + 1]);
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut minv = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let (mut delta, mut j1) = (f64::INFINITY, 0);
            for j in 1..=n {
                if !used[j] {
                    let cur = c(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = vec![None; rows];
    for j in 1..=n {
        if p[j] >= 1 && p[j] <= rows && j <= cols {
            out[p[j] - 1] = Some(j - 1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_finds_the_global_best_not_the_greedy_one() {
        // Greedy would take (0,0)=1 and then be stuck with (1,1)=10; the best total is 2+2.
        let cost = vec![vec![1.0, 2.0], vec![2.0, 10.0]];
        assert_eq!(assign(&cost), vec![Some(1), Some(0)]);
        // More rows than columns: one row goes without.
        let cost = vec![vec![1.0], vec![0.5], vec![3.0]];
        assert_eq!(assign(&cost), vec![None, Some(0), None]);
    }

    #[test]
    fn model_round_trips_through_json() {
        let xs = vec![[0.0; N], [1.0; N]];
        let m = Model { nets: vec![Net::new(&xs, 4, 1), Net::new(&xs, 4, 2)] };
        let path = std::env::temp_dir().join("minastir-assoc-test.json");
        std::fs::write(&path, m.to_json().to_string()).unwrap();
        let back = Model::load(&path).unwrap();
        assert!((m.prob(&[0.5; N]) - back.prob(&[0.5; N])).abs() < 1e-12);
        assert_eq!(back.nets.len(), 2);
    }

    #[test]
    fn training_keeps_the_best_epoch_on_the_held_out_set() {
        // "Same" when the first feature is small: learnable, so the held-out loss must fall.
        let xs: Vec<[f64; N]> = (0..400).map(|i| { let mut x = [0.0; N]; x[0] = (i % 40) as f64 / 10.0; x }).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (x[0] < 2.0) as u8 as f64).collect();
        let mut net = Net::new(&xs, 4, 3);
        let before = balanced_loss(&net, &xs, &ys);
        let (best, _) = net.train(&xs, &ys, (&xs, &ys), 20, 3, 3);
        assert!(best < before && (balanced_loss(&net, &xs, &ys) - best).abs() < 1e-12);
    }
}
