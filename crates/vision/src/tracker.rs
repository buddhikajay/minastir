//! World-space multi-camera tracker.
//!
//! Each camera's detections, once projected to the ground, update one shared set of tracks.
//! Cross-camera matching reasons about *where each camera can be wrong*: a single camera places
//! things well across its line of sight but poorly along it (depth error grows with range), so a
//! sighting is an ellipse stretched along the camera's ray, not a point. Two cameras' sightings of
//! one car can be metres apart yet compatible; two cars side by side can be close yet are proven
//! distinct when one camera sees both. (Tesla fuses cameras inside one network trained on its
//! fleet; without that, this is the geometric equivalent.)
//!
//! Each track also carries an appearance fingerprint (see `appearance`), averaged over many
//! crops. Position and look are combined as evidence: two tracks that look alike may be merged
//! from further apart, two that look different need to be nearly on top of each other, and a car
//! that leaves and comes back gets its old identity. How alike "alike" is isn't fixed: the tracker
//! learns it from the fingerprints themselves — one track's look from two different cameras shows
//! what "same" looks like across viewpoints (consecutive frames of one camera would be far too
//! easy), and two objects in the same frame of one camera (provably different) show "different".

use serde::Serialize;
use std::collections::{BTreeSet, HashMap};

#[derive(Debug, Clone)]
pub struct GroundDetection {
    pub camera: String,
    pub class: &'static str,
    pub score: f32,
    pub x: f64,
    pub z: f64,
    /// Distance from the camera, metres (near views are trusted more).
    pub range: f64,
    /// The camera's ground position (world x, z): the line of sight runs from here.
    pub from: (f64, f64),
    /// The track this camera's box in the previous frame went to, when this box clearly overlaps
    /// it: the same object, however much its ground position jittered.
    pub prev: Option<u64>,
}

/// One camera's latest sighting of a track.
#[derive(Debug, Clone)]
struct Obs {
    camera: String,
    x: f64,
    z: f64,
    t: f64,
    w: f64,
    from: (f64, f64),
}

/// Placement error between cameras (1σ, metres): camera poses agree to about ±0.4 m.
const ALIGN: f64 = 0.45;
/// χ² (2 dof) at 99%: sightings closer than this in Mahalanobis² are compatible.
const COMPATIBLE: f64 = 9.21;

/// Ground-position uncertainty of a sighting as a covariance [xx, xz, zz]: large along the
/// camera's line of sight (depth error grows with range), small across it.
fn sighting_cov(x: f64, z: f64, from: (f64, f64)) -> [f64; 3] {
    let (dx, dz) = (x - from.0, z - from.1);
    let r = dx.hypot(dz).max(0.5);
    let (ux, uz) = (dx / r, dz / r);
    let along = 0.25 + 0.06 * r + 0.004 * r * r;
    let across = 0.15 + 0.012 * r;
    let (a, c) = (along * along, across * across);
    [c + (a - c) * ux * ux, (a - c) * ux * uz, c + (a - c) * uz * uz]
}

/// Mahalanobis² distance between two sightings from different cameras.
fn mahalanobis2(p: (f64, f64, (f64, f64)), q: (f64, f64, (f64, f64))) -> f64 {
    let (sp, sq) = (sighting_cov(p.0, p.1, p.2), sighting_cov(q.0, q.1, q.2));
    let k = 2.0 * ALIGN * ALIGN;
    let (a, b, c) = (sp[0] + sq[0] + k, sp[1] + sq[1], sp[2] + sq[2] + k);
    let det = (a * c - b * b).max(1e-9);
    let (dx, dz) = (p.0 - q.0, p.1 - q.1);
    (c * dx * dx - 2.0 * b * dx * dz + a * dz * dz) / det
}

/// Upper bound on how far apart two sightings of one object can be, whatever the ellipses say
/// (far-off depth is so uncertain that everything would otherwise match).
fn max_apart(class: &str) -> f64 {
    match family(class) {
        0 => 3.0,
        1 => 7.0,
        _ => 4.0,
    }
}

/// What the fingerprints have taught the tracker: how similar two crops of one object are
/// (mean and spread), and two different objects.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AppearanceStats {
    pub same: f64,
    pub same_sd: f64,
    pub different: f64,
    pub different_sd: f64,
    pub same_n: u64,
    pub different_n: u64,
}

impl Default for AppearanceStats {
    /// Starting point before anything is learned (DINOv3, measured on this site's cameras).
    fn default() -> Self {
        Self { same: 0.62, same_sd: 0.08, different: 0.50, different_sd: 0.08, same_n: 0, different_n: 0 }
    }
}

/// Running mean and spread, weighting recent samples more once there are many.
fn learn(mean: &mut f64, sd: &mut f64, n: &mut u64, s: f64) {
    *n += 1;
    let a = 1.0 / (*n as f64 + 20.0).min(500.0);
    let d = s - *mean;
    *mean += a * d;
    *sd = ((1.0 - a) * (*sd * *sd + a * d * d)).sqrt().max(0.02);
}

impl AppearanceStats {
    fn learn_same(&mut self, s: f64) {
        learn(&mut self.same, &mut self.same_sd, &mut self.same_n, s);
    }

    fn learn_different(&mut self, s: f64) {
        learn(&mut self.different, &mut self.different_sd, &mut self.different_n, s);
    }

    /// Likelihood ratio "same object : different objects" for a similarity — how much more
    /// likely it is under the learned "same" distribution than the "different" one, with a
    /// shared spread so it rises steadily with similarity (bounded: one look never decides
    /// alone).
    pub fn odds(&self, s: f64) -> f64 {
        // Until "same" reliably looks more alike than "different", looks are no evidence.
        if self.same <= self.different + 0.01 {
            return 1.0;
        }
        let var = (self.same_sd.powi(2) + self.different_sd.powi(2)) / 2.0;
        let log = (self.same - self.different) * (s - (self.same + self.different) / 2.0) / var;
        log.exp().clamp(0.25, 4.0)
    }
}

/// A track's fingerprint: the running sum of its unit-length crop fingerprints.
#[derive(Debug, Clone)]
struct Look {
    sum: Vec<f32>,
    n: u32,
}

impl Look {
    fn mean(&self) -> Vec<f32> {
        let norm = self.sum.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
        self.sum.iter().map(|v| v / norm).collect()
    }

    fn add(&mut self, f: &[f32]) {
        self.sum.iter_mut().zip(f).for_each(|(a, b)| *a += b);
        self.n += 1;
    }

    fn merge(&mut self, o: &Look) {
        self.sum.iter_mut().zip(&o.sum).for_each(|(a, b)| *a += b);
        self.n += o.n;
    }

    fn similarity(&self, o: &Look) -> f64 {
        crate::appearance::similarity(&self.mean(), &o.mean()) as f64
    }
}

/// Fingerprinted crops needed before a look is trusted.
const LOOK_MIN: u32 = 2;

/// A track that ended, remembered so a returning object gets its identity back.
struct Lost {
    id: u64,
    class: &'static str,
    x: f64,
    z: f64,
    look: Option<Look>,
    hits: u32,
    at: f64,
}

/// How long ended tracks are remembered, seconds.
const REMEMBER: f64 = 900.0;

#[derive(Debug, Clone, Serialize)]
pub struct TrackedObject {
    pub id: u64,
    pub class: &'static str,
    pub x: f64,
    pub z: f64,
    /// Direction of travel (same convention as camera yaw), degrees.
    pub heading: f64,
    /// Whether `heading` was measured (it has moved); parked objects that never moved don't know
    /// which way they face.
    pub heading_known: bool,
    /// m/s
    pub speed: f64,
    pub score: f32,
    /// Cameras that saw it in the last second.
    pub cameras: Vec<String>,
    /// Crops fingerprinted so far (0 = identified by position only).
    pub looks: u32,
    /// Whether the nose is known (from driving, or a shape that shows it); a fitted parked car
    /// may only know its axis.
    pub facing_known: bool,
    /// How well a 3D vehicle drawn at this pose matches the cameras' outlines (0–1), when fitted.
    pub fit: Option<f32>,
    /// What the describer (a vision-language model) says a detected vehicle really is:
    /// probabilities of car, truck, bin, other — None until it has looked.
    pub verdict: Option<[f32; 4]>,
}

/// The describer's answers, in `TrackedObject::verdict` order.
pub const VERDICT: [&str; 4] = ["car", "truck", "bin", "other"];

/// A parked vehicle's pose from render-and-compare (see `fit`).
#[derive(Debug, Clone, Copy)]
pub struct Fitted {
    pub x: f64,
    pub z: f64,
    pub heading: f64,
    /// Only the axis is known (heading or heading + 180°).
    pub axis_only: bool,
    pub score: f64,
    pub at: f64,
}

/// How often a parked vehicle is re-fitted, seconds.
const REFIT: f64 = 60.0;
/// How much a fit's score counts for less per minute of age when weighed against a re-fit.
const FIT_FADE: f64 = 0.01;
/// Overlap a fit needs before its pose is shown (drawn and real outlines mostly agree).
const FIT_TRUSTED: f64 = 0.35;

struct Track {
    id: u64,
    class: &'static str,
    x: f64,
    z: f64,
    vx: f64,
    vz: f64,
    heading: f64,
    heading_known: bool,
    score: f32,
    hits: u32,
    last_seen: f64,
    seen_by: Vec<(String, f64)>,
    /// The camera this object's position comes from (the one with the best view). Other cameras
    /// help decide who is who, but their placements are never averaged in: cameras disagree by
    /// up to a few metres, and an average sits where no camera saw it.
    owner: Option<String>,
    /// Where it was over the last few seconds (owner's placements): (time, x, z).
    trail: Vec<(f64, f64, f64)>,
    /// Latest sighting from each camera. Cameras disagree by up to a few metres along their lines
    /// of sight, so the track follows their weighted average instead of whichever spoke last.
    obs: Vec<Obs>,
    /// Parked/standing still: the position is held and only drifts slowly.
    still: bool,
    /// Where it was when it became still; leaving this by more than `start_distance` = moving.
    anchor: (f64, f64),
    /// Consecutive updates away from the anchor, and when it last moved fast.
    away: u32,
    away_since: f64,
    last_fast: f64,
    look: Option<Look>,
    /// The same, per camera.
    views: Vec<(String, Look)>,
    /// Render-and-compare pose while parked (shown instead of the box-based position), and
    /// when a fit was last attempted.
    fitted: Option<Fitted>,
    fit_tried: f64,
    /// The describer's verdict, and when it was asked.
    verdict: Option<[f32; 4]>,
    verdict_asked: f64,
}

use crate::detector::family;

/// Seconds of position history kept per object (for recognising two cameras' tracks of one
/// moving object by their shared motion).
const TRAIL: f64 = 4.0;

/// Do two tracks move together — the same path, offset by a near-constant amount (two cameras
/// that disagree about where someone is still agree on how they move)? Needs a couple of seconds
/// of overlapping history and real movement; returns the mean offset when they do.
fn co_moving(a: &Track, b: &Track) -> Option<f64> {
    let mut offs = Vec::new();
    for &(t, x, z) in &a.trail {
        if let Some(&(_, bx, bz)) = b.trail.iter().min_by(|p, q| (p.0 - t).abs().total_cmp(&(q.0 - t).abs())).filter(|p| (p.0 - t).abs() < 0.15) {
            offs.push((bx - x, bz - z, x, z));
        }
    }
    if offs.len() < 10 {
        return None;
    }
    let span = a.trail.last()?.0 - a.trail.first()?.0;
    let (fx, fz) = (offs.first()?.2, offs.first()?.3);
    let travelled = offs.iter().map(|o| (o.2 - fx).hypot(o.3 - fz)).fold(0.0, f64::max);
    if span < 2.0 || travelled < 1.2 {
        return None; // not moving enough for motion to prove anything
    }
    let n = offs.len() as f64;
    let (mx, mz) = (offs.iter().map(|o| o.0).sum::<f64>() / n, offs.iter().map(|o| o.1).sum::<f64>() / n);
    let spread = (offs.iter().map(|o| (o.0 - mx).powi(2) + (o.1 - mz).powi(2)).sum::<f64>() / n).sqrt();
    let mean = mx.hypot(mz);
    // Steady offset (well under how far they moved), and no bigger than cameras can disagree.
    (spread < 0.45 && spread < 0.3 * travelled && mean < 7.0).then_some(mean)
}

/// How far a detection may be from a track's predicted position and still be the same object.
/// Generous: camera poses still disagree by up to a metre or so between cameras.
fn gate(class: &str) -> f64 {
    match family(class) {
        0 => 1.8,
        1 => 4.0,
        _ => 2.5,
    }
}

/// Distance from where it stood that counts as really moving (beyond box jitter).
fn start_distance(class: &str) -> f64 {
    match family(class) {
        0 => 0.7,
        1 => 1.8,
        _ => 1.0,
    }
}

/// Seconds without a detection before a track is dropped. Parked cars are often missed for a few
/// frames (occlusion, a passing person): keep them rather than blinking.
fn max_age(t: &Track) -> f64 {
    match (t.still, family(t.class)) {
        (true, 1) => 6.0,
        (true, _) => 3.0,
        _ => 1.5,
    }
}

#[derive(Default)]
pub struct Tracker {
    tracks: Vec<Track>,
    next_id: u64,
    /// Ids that were merged into, or renamed to, another id.
    aliases: HashMap<u64, u64>,
    lost: Vec<Lost>,
    /// What fingerprints have taught, per object family (people and vehicles look different).
    stats: [AppearanceStats; 4],
}

impl Tracker {
    /// What the fingerprints have taught about one object family (0 people, 1 vehicles, …).
    pub fn stats(&self, family: u8) -> AppearanceStats {
        self.stats[family as usize % 4]
    }
}

impl Track {
    fn has_look(&self) -> bool {
        self.look.as_ref().is_some_and(|l| l.n >= LOOK_MIN)
    }
}

impl Tracker {
    /// Update with one camera's detections taken at time `now` (seconds). Returns the track id
    /// each detection went to (None: dropped as a duplicate box).
    pub fn update(&mut self, camera: &str, dets: &[GroundDetection], now: f64) -> Vec<Option<u64>> {
        let mut assigned: Vec<Option<u64>> = vec![None; dets.len()];
        // Candidate pairs with a normalised cost (≤ 1 = acceptable), cheapest first; each detection
        // and each track used at most once.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            let dt = (now - t.last_seen).clamp(0.0, 1.0);
            let (px, pz) = (t.x + t.vx * dt, t.z + t.vz * dt);
            for (di, d) in dets.iter().enumerate() {
                if family(d.class) != family(t.class) {
                    continue;
                }
                // Against the track's predicted position (plain distance).
                let mut cost = (d.x - px).hypot(d.z - pz) / gate(d.class);
                for o in t.obs.iter().filter(|o| now - o.t < 1.0) {
                    let apart = (d.x - o.x).hypot(d.z - o.z);
                    if apart > max_apart(d.class) {
                        continue;
                    }
                    cost = cost.min(if o.camera == camera {
                        // This camera's own last reading shares its offset: plain distance.
                        apart / gate(d.class)
                    } else {
                        // Another camera's: compatible within both lines-of-sight ellipses?
                        (mahalanobis2((d.x, d.z, d.from), (o.x, o.z, o.from)) / COMPATIBLE).sqrt()
                    });
                }
                if cost < 1.0 {
                    pairs.push((cost, ti, di));
                }
            }
        }
        // An image track can outlast its 3D track (someone hidden for a few seconds): bring the
        // 3D track back under the same id where the image says they reappeared.
        for d in dets {
            let Some(prev) = d.prev.map(|p| self.resolve(p)) else { continue };
            if self.tracks.iter().any(|t| t.id == prev) {
                continue;
            }
            if let Some(k) = self.lost.iter().position(|l| l.id == prev && family(l.class) == family(d.class)) {
                let l = self.lost.remove(k);
                let w = 1.0 / (1.0 + (d.range / 8.0).powi(2));
                self.tracks.push(Track {
                    id: l.id,
                    class: l.class,
                    x: d.x,
                    z: d.z,
                    vx: 0.0,
                    vz: 0.0,
                    heading: 0.0,
                    heading_known: false,
                    score: d.score,
                    hits: l.hits.max(3),
                    last_seen: now,
                    seen_by: vec![(camera.to_string(), now)],
                    owner: Some(camera.to_string()),
                    trail: vec![(now, d.x, d.z)],
                    obs: vec![Obs { camera: camera.to_string(), x: d.x, z: d.z, t: now, w, from: d.from }],
                    still: true,
                    anchor: (d.x, d.z),
                    away: 0,
                    away_since: now,
                    last_fast: now,
                    look: l.look,
                    views: Vec::new(),
                    fitted: None,
                    fit_tried: f64::MIN,
                verdict: None,
                verdict_asked: f64::MIN,
                });
            }
        }
        // Continuity in the image beats geometry: a box overlapping this camera's box from the
        // previous frame stays with that box's track (taken before any position-based match).
        for (di, d) in dets.iter().enumerate() {
            let Some(prev) = d.prev.map(|p| self.resolve(p)) else { continue };
            if let Some(ti) = self.tracks.iter().position(|t| t.id == prev && family(t.class) == family(d.class)) {
                pairs.push((-1.0, ti, di));
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut track_used = vec![false; self.tracks.len()];
        let mut det_used = vec![false; dets.len()];
        for (_, ti, di) in pairs {
            if track_used[ti] || det_used[di] {
                continue;
            }
            track_used[ti] = true;
            det_used[di] = true;
            let d = &dets[di];
            let t = &mut self.tracks[ti];
            assigned[di] = Some(t.id);
            let w = 1.0 / (1.0 + (d.range / 8.0).powi(2));
            t.obs.retain(|o| o.camera != camera && now - o.t < 1.0);
            t.obs.push(Obs { camera: camera.to_string(), x: d.x, z: d.z, t: now, w, from: d.from });
            // Ownership: keep the owner while it sees the object, unless another camera's view is
            // clearly better (nearer); if it stopped seeing it, the best current view takes over.
            let fresh = |o: &&Obs| now - o.t < 0.5;
            let owner_w = t.owner.as_ref().and_then(|c| t.obs.iter().filter(fresh).find(|o| &o.camera == c)).map(|o| o.w);
            let best = t.obs.iter().filter(fresh).max_by(|a, b| a.w.total_cmp(&b.w)).map(|o| (o.camera.clone(), o.w));
            let handover = match (owner_w, &best) {
                (None, Some((c, _))) => Some(c.clone()),
                (Some(ow), Some((c, bw))) if *bw > 1.5 * ow && t.owner.as_deref() != Some(c.as_str()) => Some(c.clone()),
                _ => None,
            };
            let took_over = handover.as_deref() == Some(camera) && t.owner.as_deref() != Some(camera);
            if let Some(c) = handover {
                t.owner = Some(c);
            }
            if t.owner.as_deref() != Some(camera) {
                // Another camera's view of it: it's still here (and who it is), but its position
                // comes from the owner.
                t.hits += 1;
                t.last_seen = now;
                t.seen_by.retain(|(c, _)| c != camera);
                t.seen_by.push((camera.to_string(), now));
                continue;
            }
            let (mx, mz) = (d.x, d.z);
            if took_over {
                // Just taken over from another camera: start from this camera's placement
                // rather than easing from the old one's (they may disagree by metres).
                t.x = mx;
                t.z = mz;
                t.anchor = (mx, mz);
                t.away = 0;
            }
            let dt = (now - t.last_seen).clamp(1e-3, 1.0);

            if t.still {
                // Hold position; average out box jitter (quickly while it settles, then slowly).
                let a = if t.hits < 10 { 0.3 } else { 0.06 };
                t.x += a * (mx - t.x);
                t.z += a * (mz - t.z);
                t.vx = 0.0;
                t.vz = 0.0;
                if (mx - t.anchor.0).hypot(mz - t.anchor.1) > start_distance(t.class) {
                    if t.away == 0 {
                        t.away_since = t.last_seen;
                    }
                    t.away += 1;
                    if t.away >= 4 {
                        // Start moving with the velocity it actually left with.
                        let el = (now - t.away_since).max(0.2);
                        t.vx = ((mx - t.anchor.0) / el).clamp(-20.0, 20.0) * 0.6;
                        t.vz = ((mz - t.anchor.1) / el).clamp(-20.0, 20.0) * 0.6;
                        t.still = false;
                        t.fitted = None;
                        t.last_fast = now;
                        t.x = mx;
                        t.z = mz;
                        if t.vx.hypot(t.vz) > 0.3 {
                            t.heading = t.vx.atan2(t.vz).to_degrees().rem_euclid(360.0);
                            t.heading_known = true;
                        }
                    }
                } else {
                    t.away = 0;
                }
            } else {
                // α-β filter: smooth position, learn velocity from the residual.
                let (px, pz) = (t.x + t.vx * dt, t.z + t.vz * dt);
                let (rx, rz) = (mx - px, mz - pz);
                const ALPHA: f64 = 0.45;
                const BETA: f64 = 0.12;
                t.x = px + ALPHA * rx;
                t.z = pz + ALPHA * rz;
                t.vx = (t.vx + BETA * rx / dt).clamp(-20.0, 20.0);
                t.vz = (t.vz + BETA * rz / dt).clamp(-20.0, 20.0);
                let speed = t.vx.hypot(t.vz);
                let fast = if family(t.class) == 0 { 0.4 } else { 1.0 };
                if speed > fast {
                    t.last_fast = now;
                    t.heading = t.vx.atan2(t.vz).to_degrees().rem_euclid(360.0);
                    t.heading_known = true;
                }
                // Slow for a while → it has stopped.
                if now - t.last_fast > 1.5 {
                    t.still = true;
                    t.anchor = (t.x, t.z);
                    t.away = 0;
                    t.vx = 0.0;
                    t.vz = 0.0;
                }
            }
            // Prefer the more specific vehicle label once it's confident.
            if d.score > t.score || d.class == t.class {
                t.class = d.class;
            }
            t.score = 0.7 * t.score + 0.3 * d.score;
            t.hits += 1;
            t.last_seen = now;
            t.trail.push((now, t.x, t.z));
            t.trail.retain(|p| now - p.0 <= TRAIL);
            t.seen_by.retain(|(c, _)| c != camera);
            t.seen_by.push((camera.to_string(), now));
        }
        for (di, d) in dets.iter().enumerate() {
            if det_used[di] {
                continue;
            }
            // Right next to a track this camera just updated → another view of the same object
            // (e.g. a car's front and side boxed separately), not a new one.
            let dup = self.tracks.iter().any(|t| {
                t.last_seen == now && family(t.class) == family(d.class) && (t.x - d.x).hypot(t.z - d.z) < gate(d.class) * 0.6
            });
            if dup {
                continue;
            }
            let w = 1.0 / (1.0 + (d.range / 8.0).powi(2));
            assigned[di] = Some(self.next_id);
            self.tracks.push(Track {
                id: self.next_id,
                class: d.class,
                x: d.x,
                z: d.z,
                vx: 0.0,
                vz: 0.0,
                heading: 0.0,
                heading_known: false,
                score: d.score,
                hits: 1,
                last_seen: now,
                seen_by: vec![(camera.to_string(), now)],
                owner: Some(camera.to_string()),
                trail: vec![(now, d.x, d.z)],
                obs: vec![Obs { camera: camera.to_string(), x: d.x, z: d.z, t: now, w, from: d.from }],
                // New things start still: a parked car stays put; a moving one leaves its anchor
                // within a few frames.
                still: true,
                anchor: (d.x, d.z),
                away: 0,
                away_since: now,
                last_fast: now,
                look: None,
                views: Vec::new(),
                fitted: None,
                fit_tried: f64::MIN,
                verdict: None,
                verdict_asked: f64::MIN,
            });
            self.next_id += 1;
        }
        self.merge_duplicates(now);
        let (keep, ended): (Vec<Track>, Vec<Track>) = std::mem::take(&mut self.tracks).into_iter().partition(|t| now - t.last_seen < max_age(t));
        self.tracks = keep;
        for t in ended {
            if t.hits >= 3 {
                self.lost.push(Lost { id: t.id, class: t.class, x: t.x, z: t.z, look: t.look.filter(|l| l.n >= LOOK_MIN), hits: t.hits, at: now });
            }
        }
        self.lost.retain(|l| now - l.at < REMEMBER);
        assigned.into_iter().map(|a| a.map(|id| self.resolve(id))).collect()
    }

    /// Parked vehicles due a render-and-compare fit (never fitted first, then the stalest):
    /// (id, class, x, z).
    pub fn needs_fit(&self, now: f64) -> Vec<(u64, &'static str, f64, f64)> {
        let mut due: Vec<(f64, u64, &'static str, f64, f64)> = self
            .tracks
            .iter()
            .filter(|t| t.still && t.hits >= 5 && family(t.class) == 1 && now - t.last_seen < 0.5)
            .filter(|t| now - t.fit_tried > REFIT)
            .map(|t| (t.fit_tried, t.id, t.class, t.x, t.z))
            .collect();
        due.sort_by(|a, b| a.0.total_cmp(&b.0));
        due.into_iter().map(|(_, id, c, x, z)| (id, c, x, z)).collect()
    }

    /// Vehicles the describer hasn't judged yet (asked at most once a minute each).
    pub fn needs_verdict(&self, now: f64) -> Vec<u64> {
        self.tracks
            .iter()
            .filter(|t| family(t.class) == 1 && t.hits >= 5 && t.verdict.is_none() && now - t.verdict_asked > 60.0 && now - t.last_seen < 0.5)
            .map(|t| t.id)
            .collect()
    }

    pub fn verdict_asked(&mut self, id: u64, now: f64) {
        let id = self.resolve(id);
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.verdict_asked = now;
        }
    }

    /// Whether track `id` is standing still (parked).
    pub fn is_still(&self, id: u64) -> bool {
        let id = self.resolve(id);
        self.tracks.iter().any(|t| t.id == id && t.still)
    }

    /// Record the describer's verdict on a vehicle track.
    pub fn set_verdict(&mut self, id: u64, v: [f32; 4]) {
        let id = self.resolve(id);
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.verdict = Some(v);
        }
    }

    /// Note that track `id` is being fitted now (so it isn't queued again until it's due).
    pub fn fit_started(&mut self, id: u64, now: f64) {
        let id = self.resolve(id);
        if let Some(t) = self.tracks.iter_mut().find(|t| t.id == id) {
            t.fit_tried = now;
        }
    }

    /// Record a fitted pose for track `id` (if it's still parked). A parked vehicle doesn't
    /// change, so a re-fit only replaces the pose when it explains the cameras about as well or
    /// better (a later frame may be blurrier, darker or partly hidden); the old fit's standing
    /// fades over minutes so a real change still gets through. An axis-only heading takes
    /// whichever end agrees with a heading it already knew.
    pub fn apply_fit(&mut self, id: u64, mut f: Fitted) {
        let id = self.resolve(id);
        let Some(t) = self.tracks.iter_mut().find(|t| t.id == id && t.still) else { return };
        if let Some(old) = t.fitted {
            let standing = old.score - FIT_FADE * (f.at - old.at) / 60.0;
            if f.score < standing - 0.03 {
                return;
            }
        }
        if f.axis_only {
            let known = t.fitted.filter(|p| !p.axis_only).map(|p| p.heading).or(t.heading_known.then_some(t.heading));
            if let Some(k) = known {
                let d = (f.heading - k).rem_euclid(360.0);
                if d > 90.0 && d < 270.0 {
                    f.heading = (f.heading + 180.0).rem_euclid(360.0);
                }
                f.axis_only = false;
            }
        }
        t.fitted = Some(f);
    }

    /// The current id of a track that may since have been merged or renamed.
    pub fn resolve(&self, mut id: u64) -> u64 {
        while let Some(&to) = self.aliases.get(&id) {
            id = to;
        }
        id
    }

    /// Fingerprints of several objects cut from one frame of `camera`: each joins its track, and
    /// every pair of the same kind teaches what "different" looks like (two objects in one frame
    /// are two objects — the detector already merged overlapping boxes of one).
    pub fn add_frame_looks(&mut self, camera: &str, looks: &[(u64, Vec<f32>)], now: f64) {
        let fam: Vec<Option<u8>> = looks
            .iter()
            .map(|(id, _)| {
                let id = self.resolve(*id);
                self.tracks.iter().find(|t| t.id == id).map(|t| family(t.class))
            })
            .collect();
        for i in 0..looks.len() {
            for j in i + 1..looks.len() {
                if let (Some(a), Some(b)) = (fam[i], fam[j]) {
                    if a == b && self.resolve(looks[i].0) != self.resolve(looks[j].0) {
                        let s = crate::appearance::similarity(&looks[i].1, &looks[j].1) as f64;
                        self.stats[a as usize % 4].learn_different(s);
                    }
                }
            }
        }
        for (id, f) in looks {
            self.add_look(*id, camera, f, now);
        }
    }

    /// Add a crop fingerprint (unit length) from `camera` to track `id` — computed off the
    /// detection thread, so it may arrive after the track was merged or renamed.
    pub fn add_look(&mut self, id: u64, camera: &str, fingerprint: &[f32], now: f64) {
        let id = self.resolve(id);
        let Some(ti) = self.tracks.iter().position(|t| t.id == id) else { return };
        let t = &mut self.tracks[ti];
        let fam = family(t.class) as usize % 4;
        // This object as another camera sees it: teaches what "same" looks like across views.
        for (_, v) in t.views.iter().filter(|(c, v)| c != camera && v.n >= LOOK_MIN) {
            self.stats[fam].learn_same(crate::appearance::similarity(&v.mean(), fingerprint) as f64);
        }
        match t.views.iter_mut().find(|(c, _)| c == camera) {
            Some((_, v)) => v.add(fingerprint),
            None => t.views.push((camera.to_string(), Look { sum: fingerprint.to_vec(), n: 1 })),
        }
        match &mut t.look {
            Some(look) => look.add(fingerprint),
            None => t.look = Some(Look { sum: fingerprint.to_vec(), n: 1 }),
        }
        if self.tracks[ti].look.as_ref().is_some_and(|l| l.n == LOOK_MIN) {
            self.recognise(ti, now);
        }
    }

    /// A track that just got a trusted look: is it an object that left earlier? Then it takes
    /// that object's identity back. Far from where it was last seen needs a strong likeness.
    fn recognise(&mut self, ti: usize, now: f64) {
        let t = &self.tracks[ti];
        let look = t.look.as_ref().unwrap();
        let best = self
            .lost
            .iter()
            .enumerate()
            .filter(|(_, l)| family(l.class) == family(t.class) && l.look.is_some())
            .map(|(k, l)| {
                let odds = self.stats[family(t.class) as usize % 4].odds(l.look.as_ref().unwrap().similarity(look));
                let near = (l.x - t.x).hypot(l.z - t.z) < max_apart(t.class);
                (k, odds, near)
            })
            .filter(|&(_, odds, near)| if near { odds >= 1.5 } else { odds >= 3.5 })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((k, _, _)) = best {
            let l = self.lost.remove(k);
            let t = &mut self.tracks[ti];
            self.aliases.insert(t.id, l.id);
            t.id = l.id;
            t.look.as_mut().unwrap().merge(l.look.as_ref().unwrap());
            t.hits = t.hits.max(3);
        }
        let _ = now;
    }

    /// Two still tracks of the same kind are one object that got split when they're seen by
    /// *different* cameras whose sightings are compatible along their lines of sight, or when
    /// they're nearly on top of each other. A camera that currently sees both proves they're two.
    fn merge_duplicates(&mut self, now: f64) {
        let mut i = 0;
        while i < self.tracks.len() {
            let mut j = i + 1;
            while j < self.tracks.len() {
                let (a, b) = (&self.tracks[i], &self.tracks[j]);
                // Parked/standing tracks merge on geometry (looks only adjust it); moving ones —
                // people walking past, seen by two cameras — need both looks to vouch for it.
                let fam = family(a.class) as usize % 4;
                let both_still = a.still && b.still;
                let same = family(a.class) == family(b.class) && (both_still || (a.has_look() && b.has_look()));
                let recent = |t: &Track| t.obs.iter().filter(|o| now - o.t < 1.0).map(|o| o.camera.clone()).collect::<BTreeSet<_>>();
                let (ca, cb) = (recent(a), recent(b));
                let seen_together = ca.intersection(&cb).next().is_some();
                // Appearance as evidence: likely-same looks widen how far apart the sightings may
                // be, likely-different looks narrow it. Unknown looks leave geometry alone.
                let looks = if a.has_look() && b.has_look() {
                    Some(a.look.as_ref().unwrap().similarity(b.look.as_ref().unwrap()))
                } else {
                    None
                };
                let odds = looks.map_or(1.0, |s| self.stats[fam].odds(s));
                let same = same && (both_still || odds >= 1.5);
                let reach = odds.sqrt(); // 0.5 … 2
                let on_top = (a.x - b.x).hypot(a.z - b.z) < gate(a.class) * 0.35 * reach.min(1.0) && odds >= 0.5;
                let compatible = !seen_together
                    && a.obs.iter().filter(|o| now - o.t < 1.0).any(|oa| {
                        b.obs.iter().filter(|o| now - o.t < 1.0 && o.camera != oa.camera).any(|ob| {
                            (oa.x - ob.x).hypot(oa.z - ob.z) < max_apart(a.class) * reach
                                && mahalanobis2((oa.x, oa.z, oa.from), (ob.x, ob.z, ob.from)) < COMPATIBLE * odds
                        })
                    });
                // Two cameras' tracks moving in lockstep are one object, even metres apart.
                let lockstep = family(a.class) == family(b.class)
                    && !seen_together
                    && ca.is_disjoint(&cb)
                    && !ca.is_empty()
                    && !cb.is_empty()
                    && co_moving(a, b).is_some()
                    && odds >= 0.5;
                if (same && (compatible || (on_top && !seen_together))) || lockstep {
                    let b = self.tracks.remove(j);
                    self.aliases.insert(b.id, self.tracks[i].id);
                    let a = &mut self.tracks[i];
                    match (&mut a.look, &b.look) {
                        (Some(la), Some(lb)) => la.merge(lb),
                        (None, Some(lb)) => a.look = Some(lb.clone()),
                        _ => {}
                    }
                    for (c, v) in b.views {
                        match a.views.iter_mut().find(|(x, _)| *x == c) {
                            Some((_, av)) => av.merge(&v),
                            None => a.views.push((c, v)),
                        }
                    }
                    for o in b.obs {
                        if !a.obs.iter().any(|x| x.camera == o.camera) {
                            a.obs.push(o);
                        }
                    }
                    for s in b.seen_by {
                        if !a.seen_by.iter().any(|x| x.0 == s.0) {
                            a.seen_by.push(s);
                        }
                    }
                    if b.trail.len() > a.trail.len() && a.owner.is_none() {
                        a.trail = b.trail.clone();
                    }
                    // The merged object is owned by whichever camera has the best current view.
                    if let Some(o) = a.obs.iter().filter(|o| now - o.t < 1.0).max_by(|p, q| p.w.total_cmp(&q.w)) {
                        a.owner = Some(o.camera.clone());
                        a.x = o.x;
                        a.z = o.z;
                        a.anchor = (a.x, a.z);
                    }
                    if a.fitted.is_none() {
                        a.fitted = b.fitted;
                    }
                    if a.verdict.is_none() {
                        a.verdict = b.verdict;
                    }
                    if b.heading_known && !a.heading_known {
                        a.heading = b.heading;
                        a.heading_known = true;
                    }
                    a.hits += b.hits;
                    a.last_seen = a.last_seen.max(b.last_seen);
                    continue;
                }
                j += 1;
            }
            i += 1;
        }
    }

    /// Confirmed tracks (seen at least three times), extrapolated to `now`.
    pub fn objects(&self, now: f64) -> Vec<TrackedObject> {
        self.tracks
            .iter()
            .filter(|t| t.hits >= 3)
            // Only what a camera is seeing now. Objects missed for a moment are still remembered
            // (to link them up again when they reappear), but drawing them leaves ghosts behind —
            // often of someone already shown again under their linked id. Parked things are
            // kept on screen through short gaps (someone walking in front of a parked car).
            .filter(|t| now - t.last_seen < if t.still && family(t.class) == 1 { 6.0 } else { 0.7 })
            // The describer says it's neither a vehicle nor a bin (a letterbox, a hedge…): don't
            // show it as anything.
            .filter(|t| t.verdict.is_none_or(|v| v[3] < v[0] + v[1] + v[2]))
            .map(|t| {
                let dt = (now - t.last_seen).clamp(0.0, 0.5);
                let cams: BTreeSet<String> =
                    t.seen_by.iter().filter(|(_, at)| now - at < 1.0).map(|(c, _)| c.clone()).collect();
                // A poor match (a "car" that isn't one, or heavily hidden) doesn't move it.
                let fit = t.fitted.filter(|f| t.still && f.score >= FIT_TRUSTED);
                // A "vehicle" the describer recognises as a wheelie bin is shown as one.
                let class = match t.verdict {
                    Some(v) if family(t.class) == 1 && v[2] > v[0] + v[1] => "bin",
                    _ => t.class,
                };
                TrackedObject {
                    id: t.id,
                    class,
                    x: fit.map_or(t.x + t.vx * dt, |f| f.x),
                    z: fit.map_or(t.z + t.vz * dt, |f| f.z),
                    heading: fit.map_or(t.heading, |f| f.heading),
                    heading_known: t.heading_known || fit.is_some(),
                    facing_known: fit.map_or(t.heading_known, |f| !f.axis_only),
                    fit: t.fitted.filter(|_| t.still).map(|f| f.score as f32),
                    verdict: t.verdict,
                    speed: t.vx.hypot(t.vz),
                    score: t.score,
                    cameras: cams.into_iter().collect(),
                    looks: t.look.as_ref().map_or(0, |l| l.n),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(cam: &str, class: &'static str, x: f64, z: f64) -> GroundDetection {
        GroundDetection { camera: cam.into(), class, score: 0.8, x, z, range: 5.0, from: (x, z - 5.0), prev: None }
    }

    #[test]
    fn walking_person_keeps_one_id_and_gets_a_heading() {
        let mut t = Tracker::default();
        for i in 0..20 {
            let x = i as f64 * 0.15; // 1.5 m/s along +X at 10 Hz
            t.update("a", &[det("a", "person", x, 0.0)], i as f64 * 0.1);
        }
        let objs = t.objects(1.9);
        assert_eq!(objs.len(), 1);
        assert!((objs[0].heading - 90.0).abs() < 10.0, "{}", objs[0].heading);
        assert!(objs[0].speed > 0.8);
    }

    #[test]
    fn two_cameras_seeing_one_person_make_one_track() {
        let mut t = Tracker::default();
        for i in 0..5 {
            let now = i as f64 * 0.1;
            t.update("a", &[det("a", "person", 2.0, 3.0)], now);
            t.update("b", &[det("b", "person", 2.6, 3.4)], now + 0.05);
        }
        let objs = t.objects(0.5);
        assert_eq!(objs.len(), 1);
        assert_eq!(objs[0].cameras, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn parked_car_seen_by_two_disagreeing_cameras_stays_put() {
        let mut t = Tracker::default();
        let mut seed = 3u64;
        let mut jitter = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.6
        };
        let mut first = None;
        for i in 0..100 {
            let now = i as f64 * 0.1;
            let (ja, jb) = (jitter(), jitter());
            t.update("a", &[det("a", "car", 2.0 + ja, 3.0 - ja)], now);
            t.update("b", &[det("b", "car", 2.9 + jb, 3.4 + jb)], now + 0.05);
            let objs = t.objects(now + 0.05);
            assert!(objs.len() <= 1, "split into {}", objs.len());
            if let (Some(o), true) = (objs.first(), i >= 15) {
                let p = *first.get_or_insert((o.x, o.z));
                assert!((o.x - p.0).hypot(o.z - p.1) < 0.5, "drifted to {:?}", (o.x, o.z));
                assert!(o.speed < 0.01);
            }
        }
    }

    #[test]
    fn people_and_cars_dont_merge_and_tracks_expire() {
        let mut t = Tracker::default();
        for i in 0..3 {
            t.update("a", &[det("a", "person", 0.0, 0.0), det("a", "car", 0.5, 0.0)], i as f64 * 0.1);
        }
        assert_eq!(t.objects(0.2).len(), 2);
        t.update("a", &[], 10.0);
        assert!(t.objects(10.0).is_empty());
    }

    fn det_from(cam: &str, class: &'static str, x: f64, z: f64, from: (f64, f64)) -> GroundDetection {
        let range = (x - from.0).hypot(z - from.1);
        GroundDetection { camera: cam.into(), class, score: 0.8, x, z, range, from, prev: None }
    }

    #[test]
    fn one_car_seen_by_two_cameras_far_apart_along_their_rays_is_one_car() {
        // The real Corolla case: the Front Duo puts it 2.8 m from where the Turret does.
        let (duo, tur) = ((-6.97, -5.97), (-0.1, -5.0));
        let mut t = Tracker::default();
        for k in 0..12 {
            let now = k as f64 * 0.1;
            t.update("duo", &[det_from("duo", "car", 0.7, -5.6, duo)], now);
            t.update("tur", &[det_from("tur", "car", 2.8, -7.4, tur)], now + 0.05);
        }
        let cars = t.objects(1.2);
        assert_eq!(cars.len(), 1, "{cars:?}");
    }

    fn fp(seed: u64) -> Vec<f32> {
        // A distinct unit vector per seed (plus a little noise per call site).
        let mut v: Vec<f32> = (0..16).map(|k| (((seed * 31 + k) * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    #[test]
    fn alike_looks_merge_from_further_apart_and_different_looks_dont() {
        // Two cameras disagree by 6 m, beyond what geometry alone accepts.
        let run = |look_b: u64| {
            let (ca, cb) = ((0.0, -10.0), (10.0, 0.0));
            let mut t = Tracker::default();
            for k in 0..12 {
                let now = k as f64 * 0.1;
                let a = t.update("a", &[det_from("a", "car", 0.0, 0.0, ca)], now)[0].unwrap();
                let b = t.update("b", &[det_from("b", "car", 0.0, 6.0, cb)], now + 0.05);
                if k < 3 && look_b != 0 {
                    t.add_look(a, "a", &fp(1), now);
                    if let Some(b) = b[0] {
                        t.add_look(b, "b", &fp(look_b), now);
                    }
                }
            }
            t.objects(1.2).len()
        };
        assert_eq!(run(0), 2, "geometry alone shouldn't merge them");
        assert_eq!(run(1), 1, "same look should merge");
        assert_eq!(run(2), 2, "different look should stay apart");
    }

    #[test]
    fn a_fitted_parked_car_shows_its_fit_until_it_drives_off() {
        let mut t = Tracker::default();
        let mut id = 0;
        for k in 0..10 {
            id = t.update("a", &[det("a", "car", 1.0, 1.0)], k as f64 * 0.1)[0].unwrap();
        }
        assert_eq!(t.needs_fit(0.9).len(), 1);
        t.fit_started(id, 0.9);
        t.apply_fit(id, Fitted { x: 1.6, z: 0.4, heading: 30.0, axis_only: true, score: 0.7, at: 0.9 });
        assert!(t.needs_fit(1.0).is_empty());
        // A worse re-fit (blurrier frame) doesn't replace it.
        t.apply_fit(id, Fitted { x: 2.5, z: 0.0, heading: 100.0, axis_only: true, score: 0.5, at: 0.95 });
        let o = &t.objects(1.0)[0];
        assert_eq!((o.x, o.z, o.heading, o.facing_known), (1.6, 0.4, 30.0, false));
        // Drives off along +x: the fit is dropped and motion takes over.
        for k in 0..30 {
            let now = 1.0 + k as f64 * 0.1;
            t.update("a", &[det("a", "car", 1.0 + k as f64 * 0.5, 1.0)], now);
        }
        let o = &t.objects(4.0)[0];
        assert!(o.fit.is_none() && o.x > 5.0, "{o:?}");
    }

    #[test]
    fn a_box_that_overlaps_last_frames_box_keeps_its_track_despite_jitter() {
        let mut t = Tracker::default();
        let first = t.update("a", &[det("a", "person", 0.0, 10.0)], 0.0)[0].unwrap();
        // Its ground position jumps 6 m along the line of sight (feet hidden), but in the image
        // it's the same box.
        let mut d = det("a", "person", 0.0, 16.0);
        d.prev = Some(first);
        assert_eq!(t.update("a", &[d], 0.12)[0], Some(first));
        // Without the image link the same jump starts a new track.
        assert_ne!(t.update("a", &[det("a", "person", 0.0, 22.0)], 0.24)[0], Some(first));
    }

    #[test]
    fn a_person_is_placed_by_the_camera_with_the_best_view_not_an_average() {
        // The near camera (a) and a far one (b) disagree by 3 m about someone standing still.
        let mut t = Tracker::default();
        for k in 0..20 {
            let now = k as f64 * 0.1;
            let ids = t.update("a", &[det_from("a", "person", 0.0, 4.0, (0.0, 0.0))], now);
            let mut d = det_from("b", "person", 0.0, 7.0, (0.0, 20.0));
            d.prev = ids[0]; // same person (e.g. matched by outfit)
            t.update("b", &[d], now + 0.05);
        }
        let o = t.objects(2.0);
        assert_eq!(o.len(), 1, "{o:?}");
        assert!((o[0].z - 4.0).abs() < 0.3, "should follow the near camera: {:?}", o[0]);
    }

    #[test]
    fn two_cameras_tracks_moving_in_lockstep_merge_even_metres_apart() {
        // Someone walks 3 m along x; camera b places them 4.5 m off throughout.
        let (ca, cb) = ((0.0, -10.0), (15.0, 5.0));
        let mut t = Tracker::default();
        for k in 0..30 {
            let now = k as f64 * 0.1;
            let x = k as f64 * 0.1;
            t.update("a", &[det_from("a", "person", x, 0.0, ca)], now);
            t.update("b", &[det_from("b", "person", x + 3.2, 3.2, cb)], now + 0.05);
        }
        assert_eq!(t.objects(3.0).len(), 1, "{:?}", t.objects(3.0));
        // Two people walking side by side, both seen by camera a, stay two.
        let mut t = Tracker::default();
        for k in 0..30 {
            let now = k as f64 * 0.1;
            let x = k as f64 * 0.1;
            t.update("a", &[det_from("a", "person", x, 0.0, ca), det_from("a", "person", x, 1.5, ca)], now);
        }
        assert_eq!(t.objects(3.0).len(), 2);
    }

    #[test]
    fn odds_rise_with_similarity() {
        let st = AppearanceStats { same: 0.62, same_sd: 0.12, different: 0.54, different_sd: 0.05, ..Default::default() };
        let mut prev = 0.0;
        for k in 0..=100 {
            let o = st.odds(k as f64 / 100.0);
            assert!(o >= prev - 1e-9, "odds fell at {k}: {o} < {prev}");
            prev = o;
        }
        assert!(st.odds(0.2) < 1.0 && st.odds(0.9) > 1.0);
    }

    #[test]
    fn people_walking_past_two_cameras_merge_only_when_their_outfits_match() {
        // Two cameras disagree by 3.5 m about someone walking along x.
        let run = |look_b: u64| {
            let (ca, cb) = ((0.0, -8.0), (12.0, 4.0));
            let mut t = Tracker::default();
            for k in 0..30 {
                let now = k as f64 * 0.1;
                let x = k as f64 * 0.15;
                let a = t.update("a", &[det_from("a", "person", x, 0.0, ca)], now);
                let b = t.update("b", &[det_from("b", "person", x, 3.5, cb)], now + 0.05);
                if k % 3 == 0 {
                    if let Some(a) = a[0] {
                        t.add_look(a, "a", &fp(5), now);
                    }
                    if let Some(b) = b[0] {
                        t.add_look(b, "b", &fp(look_b), now);
                    }
                }
            }
            t.objects(3.0).iter().filter(|o| o.class == "person").count()
        };
        assert_eq!(run(5), 1, "same outfit: one person");
        assert_eq!(run(6), 2, "different outfits: two people");
    }

    #[test]
    fn a_car_that_comes_back_gets_its_id_back() {
        let mut t = Tracker::default();
        let mut first = 0;
        for k in 0..10 {
            let id = t.update("a", &[det("a", "car", 1.0, 1.0)], k as f64 * 0.1)[0].unwrap();
            t.add_look(id, "a", &fp(7), k as f64 * 0.1);
            first = id;
        }
        t.update("a", &[], 30.0); // gone
        assert!(t.objects(30.0).is_empty());
        let mut back = 0;
        for k in 0..5 {
            let now = 60.0 + k as f64 * 0.1;
            let id = t.update("a", &[det("a", "car", 1.3, 0.8)], now)[0].unwrap();
            t.add_look(id, "a", &fp(7), now);
            back = t.resolve(id);
        }
        assert_eq!(back, first);
        assert_eq!(t.objects(60.4)[0].id, first);
    }

    #[test]
    fn two_cars_side_by_side_seen_by_one_camera_stay_two() {
        // Corolla and the neighbour's SUV, 2.6 m apart: the Duo sees both, the Turret one.
        let (duo, tur) = ((-6.97, -5.97), (-0.1, -5.0));
        let mut t = Tracker::default();
        for k in 0..12 {
            let now = k as f64 * 0.1;
            t.update("duo", &[det_from("duo", "car", 0.7, -5.6, duo), det_from("duo", "car", 3.3, -5.6, duo)], now);
            t.update("tur", &[det_from("tur", "car", 2.8, -7.4, tur)], now + 0.05);
        }
        let cars = t.objects(1.2);
        assert_eq!(cars.len(), 2, "{cars:?}");
    }
}
