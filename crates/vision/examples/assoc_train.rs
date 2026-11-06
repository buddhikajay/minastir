//! Learn "are these two sightings the same person?" from a harvest log (MINASTIR_HARVEST), to
//! replace the tracker's hand-set matching thresholds with a learned, calibrated probability.
//! cargo run --release -p minastir-vision --example assoc_train -- <harvest.jsonl>… <model out.json>
//!
//! Several logs can be given (separate replays): each is split by time on its own — the first 70%
//! to learn from, the last 30% to test. Of the first 70%, every seventh 20-second block is held
//! back to decide when to stop training (spread through it, so it has crowds and empty yards
//! alike; one contiguous slice could hold no "different" pairs at all).
//!
//! The model is an average of ASSOC_MEMBERS (default 5) networks trained from different random
//! starts (ASSOC_SEED, default 7, is the first): one network's score moves by a couple of points
//! with its starting weights alone, which would hide whether a new feature helps.
//!
//! Labels come only from evidence that can't (or almost can't) be wrong:
//!   same      — one camera, one image track, up to 3 s apart (short gaps in one image are
//!               reliable); two cameras at the same moment whose tracks move in tight lockstep for
//!               two seconds or more (people don't walk in lockstep with strangers)
//!   different — two tracks that were ever in the same frame of one camera; for a cross-camera
//!               lockstep pair, anyone else in the other camera's frame at that moment
//! Pairs that could go either way are left out: those are what the model is for. Trained on the
//! first 70% of the log by time, tested on the rest, compared with the hand-set rules.
use minastir_vision::{
    assoc::{features, Model, Net, Sighting, N},
    imgtrack::OUTFIT,
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone)]
struct Obs {
    t: f64,
    cam: usize,
    uid: u64,
    bbox: [f32; 4],
    outfit: [f32; OUTFIT],
    x: f64,
    z: f64,
    range: f64,
    stature: Option<f32>,
}

impl Obs {
    fn sighting(&self) -> Sighting<'_> {
        Sighting { t: self.t, cam: self.cam, bbox: self.bbox, outfit: &self.outfit, ground: Some((self.x, self.z, self.range)), stature: self.stature }
    }
}

/// Image width / height of the cameras in the log (the Bullet and Turret are 16:9).
const ASPECT: f32 = 16.0 / 9.0;

fn feats(a: &Obs, b: &Obs) -> [f64; N] {
    let mut f = features(&a.sighting(), &b.sighting(), ASPECT);
    // Leak test: ASSOC_BLANK=<feature,…> zeroes features (e.g. stature_diff, keeping has_stature:
    // if that still helps, the flag itself gives the answer away).
    if let Ok(names) = std::env::var("ASSOC_BLANK") {
        for n in names.split(',') {
            if let Some(i) = minastir_vision::assoc::FEATURES.iter().position(|f| f == &n) {
                f[i] = 0.0;
            }
        }
    }
    f
}

/// The tracker's old hand-set rule for "same", where it applies (same camera only).
fn rule(f: &[f64; N]) -> Option<bool> {
    if f[1] < 0.5 {
        return None;
    }
    let (o, app, dn, dt) = (f[2], f[4], f[3], f[0]);
    Some((o >= 0.2 && app >= 0.5) || (dn <= 1.0 + 0.5 * dt && app >= 0.75))
}

const HIDDEN: usize = 16;

/// Longest gap between the two sightings of a pair, seconds (both classes, so the gap itself
/// says nothing about the answer).
const MAX_GAP: f64 = 3.0;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    anyhow::ensure!(a.len() >= 2, "usage: assoc_train <harvest.jsonl>… <model out.json>");
    let out_path = a.last().unwrap().clone();
    let mut cams: Vec<String> = Vec::new();
    let mut obs: Vec<Obs> = Vec::new();
    // Each log's clock starts at zero: shift each into its own far-apart stretch of time (so no
    // pairs form across logs), and keep its cameras apart from the same camera in another log.
    const LOG_GAP: f64 = 1e6;
    let mut splits: Vec<f64> = Vec::new();
    for (li, path) in a[..a.len() - 1].iter().enumerate() {
        let text = std::fs::read_to_string(path)?;
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
            let t = v["t"].as_f64().unwrap_or(0.0) + li as f64 * LOG_GAP;
            let cam = format!("{li}:{}", v["cam"].as_str().unwrap_or(""));
            if v["look"].is_u64() {
                continue; // fingerprints: not used by this model
            }
            let ci = cams.iter().position(|c| *c == cam).unwrap_or_else(|| {
                cams.push(cam.clone());
                cams.len() - 1
            });
            for b in v["boxes"].as_array().into_iter().flatten() {
                if b["fam"].as_u64() != Some(0) || b["x"].is_null() {
                    continue; // people with a ground placement
                }
                let bx: Vec<f32> = b["box"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
                let of: Vec<f32> = b["outfit"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
                let mut outfit = [0f32; OUTFIT];
                outfit.copy_from_slice(&of[..OUTFIT]);
                obs.push(Obs {
                    t,
                    cam: ci,
                    // Image-track ids are numbered per camera: make them unique across cameras.
                    uid: ((ci as u64) << 40) | b["uid"].as_u64().unwrap_or(0),
                    bbox: [bx[0], bx[1], bx[2], bx[3]],
                    outfit,
                    x: b["x"].as_f64().unwrap(),
                    z: b["z"].as_f64().unwrap(),
                    range: b["range"].as_f64().unwrap_or(10.0),
                stature: b["h"].as_f64().map(|h| h as f32),
                });
            }
        }
        for o in obs.iter().filter(|o| (o.t / LOG_GAP) as usize == li) {
            lo = lo.min(o.t);
            hi = hi.max(o.t);
        }
        splits.push(if lo <= hi { lo + 0.7 * (hi - lo) } else { f64::MAX });
    }
    obs.sort_by(|p, q| p.t.total_cmp(&q.t));
    println!("{} person sightings from {} cameras", obs.len(), cams.len());

    // Image tracks seen together in one frame of one camera: provably different people.
    let mut frames: BTreeMap<(usize, i64), Vec<usize>> = BTreeMap::new();
    for (i, o) in obs.iter().enumerate() {
        frames.entry((o.cam, (o.t * 1000.0) as i64)).or_default().push(i);
    }
    let mut together: HashSet<(u64, u64)> = HashSet::new();
    for idx in frames.values() {
        for &i in idx {
            for &j in idx {
                if obs[i].uid != obs[j].uid {
                    together.insert((obs[i].uid, obs[j].uid));
                }
            }
        }
    }
    // Cross-camera lockstep: image tracks in different cameras whose positions keep a steady
    // offset while moving, sampled every ~0.25 s.
    let mut by_uid: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, o) in obs.iter().enumerate() {
        by_uid.entry(o.uid).or_default().push(i);
    }
    let mut lockstep: HashSet<(u64, u64)> = HashSet::new();
    let uids: Vec<u64> = by_uid.keys().copied().collect();
    for &u in &uids {
        for &w in &uids {
            if u >= w || obs[by_uid[&u][0]].cam == obs[by_uid[&w][0]].cam {
                continue;
            }
            let (pu, pw) = (&by_uid[&u], &by_uid[&w]);
            let mut offs = Vec::new();
            let mut last_t = f64::MIN;
            for &i in pu {
                if obs[i].t - last_t < 0.25 {
                    continue;
                }
                if let Some(&j) = pw.iter().min_by(|&&p, &&q| (obs[p].t - obs[i].t).abs().total_cmp(&(obs[q].t - obs[i].t).abs())) {
                    if (obs[j].t - obs[i].t).abs() < 0.15 {
                        offs.push((obs[j].x - obs[i].x, obs[j].z - obs[i].z, obs[i].x, obs[i].z, obs[i].t));
                        last_t = obs[i].t;
                    }
                }
            }
            if offs.len() < 8 {
                continue;
            }
            let span = offs.last().unwrap().4 - offs[0].4;
            let travelled = offs.iter().map(|o| (o.2 - offs[0].2).hypot(o.3 - offs[0].3)).fold(0.0, f64::max);
            let n = offs.len() as f64;
            let (mx, mz) = (offs.iter().map(|o| o.0).sum::<f64>() / n, offs.iter().map(|o| o.1).sum::<f64>() / n);
            let spread = (offs.iter().map(|o| (o.0 - mx).powi(2) + (o.1 - mz).powi(2)).sum::<f64>() / n).sqrt();
            // Walking the same 2+ m path, offset steady to within a quarter of the distance walked,
            // for 3+ s: strangers don't do that.
            if span >= 3.0 && travelled >= 2.0 && spread < 0.25 * travelled && spread < 0.7 && mx.hypot(mz) < 7.0 {
                lockstep.insert((u, w));
                lockstep.insert((w, u));
            }
        }
    }
    println!("{} image tracks; {} pairs seen together in one frame; {} cross-camera lockstep pairs", uids.len(), together.len() / 2, lockstep.len() / 2);

    // Labelled pairs.
    let mut samples: Vec<([f64; N], f64, f64, &'static str)> = Vec::new(); // (features, label, time, kind)
    let mut seed = 3u64;
    let mut keep = |p: f64| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) < p
    };
    for (i, a) in obs.iter().enumerate() {
        for b in obs[i + 1..].iter().take_while(|b| b.t - a.t <= MAX_GAP) {
            if b.t - a.t < 0.05 && a.cam == b.cam {
                continue;
            }
            if a.cam == b.cam {
                if a.uid == b.uid {
                    if keep(0.15) {
                        samples.push((feats(a, b), 1.0, a.t, "same camera"));
                    }
                } else if a.uid != b.uid && together.contains(&(a.uid, b.uid)) {
                    samples.push((feats(a, b), 0.0, a.t, "same camera"));
                }
            } else if (b.t - a.t).abs() < 0.15 {
                if lockstep.contains(&(a.uid, b.uid)) {
                    samples.push((feats(a, b), 1.0, a.t, "cross camera"));
                } else {
                    // b is someone else if b's camera also saw a's lockstep partner then.
                    let partner_there = obs.iter().any(|c| c.cam == b.cam && (c.t - b.t).abs() < 0.05 && c.uid != b.uid && lockstep.contains(&(a.uid, c.uid)));
                    if partner_there {
                        samples.push((feats(a, b), 0.0, a.t, "cross camera"));
                    }
                }
            }
        }
    }
    let part = |s: &&([f64; N], f64, f64, &str)| {
        if s.2 >= splits[(s.2 / LOG_GAP) as usize] {
            2
        } else if (s.2 / 20.0) as u64 % 7 == 3 {
            1
        } else {
            0
        }
    };
    let train: Vec<_> = samples.iter().filter(|s| part(s) == 0).collect();
    let val: Vec<_> = samples.iter().filter(|s| part(s) == 1).collect();
    let test: Vec<_> = samples.iter().filter(|s| part(s) == 2).collect();
    for kind in ["same camera", "cross camera"] {
        let c = |v: &Vec<&([f64; N], f64, f64, &str)>, y: f64| v.iter().filter(|s| s.3 == kind && s.1 == y).count();
        println!(
            "{kind}: train {} same / {} different, validate {} / {}, test {} / {}",
            c(&train, 1.0), c(&train, 0.0), c(&val, 1.0), c(&val, 0.0), c(&test, 1.0), c(&test, 0.0)
        );
    }
    if train.is_empty() || val.is_empty() || test.is_empty() {
        anyhow::bail!("not enough labelled pairs");
    }
    // Balance the classes in training by weighting through repetition.
    let pos = train.iter().filter(|s| s.1 > 0.5).count().max(1);
    let neg = train.iter().filter(|s| s.1 < 0.5).count().max(1);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for s in &train {
        let reps = if s.1 > 0.5 { (neg as f64 / pos as f64).max(1.0) } else { (pos as f64 / neg as f64).max(1.0) };
        for _ in 0..(reps.round() as usize).clamp(1, 20) {
            xs.push(s.0);
            ys.push(s.1);
        }
    }
    let (vx, vy): (Vec<[f64; N]>, Vec<f64>) = val.iter().map(|s| (s.0, s.1)).unzip();
    let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let (seed, members) = (env("ASSOC_SEED", 7), env("ASSOC_MEMBERS", 5).max(1));
    let mut nets = Vec::new();
    for k in 0..members {
        let mut net = Net::new(&xs, HIDDEN, seed + k);
        let (loss, epoch) = net.train(&xs, &ys, (&vx, &vy), 60, 6, seed + k);
        let one = Model::single(net.clone());
        println!(
            "member {k} (seed {}): best at epoch {epoch}, validation loss {loss:.3}; test same camera {:.1}%",
            seed + k,
            score(&test, "same camera", &one).0
        );
        nets.push(net);
    }
    let m = Model { nets };

    // Test: the model vs the hand-set rule, per kind.
    for kind in ["same camera", "cross camera"] {
        if !test.iter().any(|s| s.3 == kind) {
            continue;
        }
        let (bal, same, diff, ll, rule_txt, cal) = score(&test, kind, &m);
        println!("test, {kind}: model balanced accuracy {bal:.1}% (same {same:.1}%, different {diff:.1}%, log-loss {ll:.3}){rule_txt}");
        println!("  calibration (predicted → actual): {cal}");
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&m.to_json())?)?;
    println!("model written to {out_path}");
    Ok(())
}

/// Balanced accuracy (the mean of how often "same" and "different" are each called right: pairs
/// are mostly "different", so plain accuracy would flatter a model that says no), its two parts,
/// log-loss, the hand-set rule's score, and calibration, over the test pairs of one kind.
fn score(test: &[&([f64; N], f64, f64, &str)], kind: &str, m: &Model) -> (f64, f64, f64, f64, String, String) {
    let ts: Vec<_> = test.iter().filter(|s| s.3 == kind).collect();
    let (mut ll, mut bins) = (0.0, [(0usize, 0usize); 5]);
    let (mut m_tp, mut m_tn, mut r_tp, mut r_tn, mut np, mut nn, mut r_np, mut r_nn) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for s in &ts {
        let p = m.prob(&s.0);
        let y = s.1 > 0.5;
        ll -= if y { p.max(1e-6).ln() } else { (1.0 - p).max(1e-6).ln() };
        let b = ((p * 5.0) as usize).min(4);
        bins[b].0 += 1;
        bins[b].1 += y as usize;
        if y { np += 1; m_tp += (p > 0.5) as usize; } else { nn += 1; m_tn += (p <= 0.5) as usize; }
        if let Some(r) = rule(&s.0) {
            if y { r_np += 1; r_tp += r as usize; } else { r_nn += 1; r_tn += (!r) as usize; }
        }
    }
    let bal = |tp: usize, p: usize, tn: usize, n: usize| 50.0 * (tp as f64 / p.max(1) as f64 + tn as f64 / n.max(1) as f64);
    let rule_txt = if r_np + r_nn > 0 {
        format!("; hand-set rule {:.1}% (same {:.1}%, different {:.1}%)", bal(r_tp, r_np, r_tn, r_nn), 100.0 * r_tp as f64 / r_np.max(1) as f64, 100.0 * r_tn as f64 / r_nn.max(1) as f64)
    } else {
        String::new()
    };
    let cal: Vec<String> = bins.iter().enumerate().filter(|(_, b)| b.0 > 0).map(|(k, b)| format!("{:.0}–{:.0}%: {:.0}% same (n {})", k as f64 * 20.0, (k + 1) as f64 * 20.0, 100.0 * b.1 as f64 / b.0 as f64, b.0)).collect();
    (bal(m_tp, np, m_tn, nn), 100.0 * m_tp as f64 / np.max(1) as f64, 100.0 * m_tn as f64 / nn.max(1) as f64, ll / ts.len().max(1) as f64, rule_txt, cal.join(" | "))
}
