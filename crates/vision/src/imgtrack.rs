//! Tracking within one camera's image, before anything is placed in 3D.
//!
//! In the image a person moves only a little between frames, and what they wear stays the same —
//! far steadier evidence than their ground position, which jumps whenever their feet are hidden
//! (behind a car, a letterbox, another child) or they're far away. So each camera links its boxes
//! over time first:
//!   - every box gets an outfit signature: the colours of the top and bottom half of the body
//!     (or object), cheap enough to compute for every box in every frame;
//!   - a box joins the track it overlaps where that track should be now, if the outfits agree;
//!   - a track that disappears is remembered for a few seconds and found again by its outfit near
//!     where it was (someone ducking behind something), instead of becoming someone new;
//!   - someone who left the picture altogether is remembered for a couple of minutes: a newcomer
//!     whose outfit clearly matches exactly one of them — and that one clearly better than anyone
//!     else — is them coming back.
//! The 3D tracker then keeps the identity each image track carries.
//!
//! For people, whether a box continues a track is decided by the learned association model
//! (`assoc`) when one has been trained: a calibrated "same person?" probability from overlap,
//! distance, outfit, size and ground position. Without a model — and for vehicles, and for gaps
//! longer than the model was trained on — the hand-set rule below decides. Either way, boxes and
//! tracks are then paired globally (Hungarian), not greedily.

use crate::assoc::{self, Sighting};
use std::sync::Arc;

/// Colour bins per half: 12 hues (for colourful pixels) + 4 brightness levels (greys, black,
/// white).
const BINS: usize = 16;
pub const OUTFIT: usize = 2 * BINS;

/// An RGB image (pixels, width, height).
pub type Rgb<'a> = (&'a [u8], usize, usize);

fn pixel(img: Rgb, u: f32, v: f32) -> [u8; 3] {
    let (px, w, h) = img;
    let (x, y) = (((u * w as f32) as usize).min(w - 1), ((v * h as f32) as usize).min(h - 1));
    let i = (y * w + x) * 3;
    [px[i], px[i + 1], px[i + 2]]
}

/// Outfit signature of the box `b` (normalised x0, y0, x1, y1) in an RGB frame: a colour
/// histogram of its upper half and of its lower half (each sums to 1). With the camera's
/// empty-scene `background`, pixels that look like the background there (the lawn, the path
/// behind a child) are left out, so the signature is the person's own colours.
pub fn outfit(frame: Rgb, background: Option<Rgb>, b: [f32; 4]) -> [f32; OUTFIT] {
    let mut out = [0f32; OUTFIT];
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    for (half, (y0, y1)) in [(0.12f32, 0.5f32), (0.5, 0.92)].into_iter().enumerate() {
        // Sample points; keep those that differ from the empty scene (unless that leaves too
        // few — strong shade, or no background — then use them all).
        let mut pts: Vec<[u8; 3]> = Vec::with_capacity(16 * 12);
        let mut fore: Vec<[u8; 3]> = Vec::with_capacity(16 * 12);
        for gy in 0..16 {
            for gx in 0..12 {
                let u = b[0] + bw * (0.1 + 0.8 * (gx as f32 + 0.5) / 12.0);
                let v = b[1] + bh * (y0 + (y1 - y0) * (gy as f32 + 0.5) / 16.0);
                let p = pixel(frame, u, v);
                pts.push(p);
                if let Some(bg) = background {
                    let q = pixel(bg, u, v);
                    let d: i32 = (0..3).map(|k| (p[k] as i32 - q[k] as i32).abs()).sum();
                    if d > 70 {
                        fore.push(p);
                    }
                }
            }
        }
        let use_pts = if fore.len() * 5 >= pts.len() { &fore } else { &pts };
        let mut total = 0f32;
        for p in use_pts {
            {
                let (r, g, bl) = (p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
                let mx = r.max(g).max(bl);
                let mn = r.min(g).min(bl);
                let sat = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
                let bin = if sat > 0.25 && mx > 0.15 {
                    let d = mx - mn;
                    let h = if mx == r {
                        ((g - bl) / d).rem_euclid(6.0)
                    } else if mx == g {
                        (bl - r) / d + 2.0
                    } else {
                        (r - g) / d + 4.0
                    };
                    ((h / 6.0 * 12.0) as usize).min(11)
                } else {
                    12 + ((mx * 4.0) as usize).min(3)
                };
                out[half * BINS + bin] += 1.0;
                total += 1.0;
            }
        }
        if total > 0.0 {
            for k in 0..BINS {
                out[half * BINS + k] /= total;
            }
        }
    }
    out
}

/// How alike two outfits are, 0–1 (Bhattacharyya coefficient, averaged over the two halves).
pub fn outfit_match(a: &[f32; OUTFIT], b: &[f32; OUTFIT]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p * q).sqrt()).sum::<f32>() / 2.0
}

#[derive(Debug, Clone)]
struct Track {
    /// This image track's own id (stable for its lifetime).
    uid: u64,
    /// The 3D tracker's id for this object, once it has one.
    gid: Option<u64>,
    family: u8,
    bbox: [f32; 4],
    /// Box change per second (while it's being seen).
    vel: [f32; 4],
    outfit: [f32; OUTFIT],
    /// The latest box's own outfit and ground position (what the association model compares).
    seen_outfit: [f32; OUTFIT],
    ground: Option<(f64, f64, f64)>,
    last: f64,
    /// The latest single height measurement and when (what the association model compares).
    seen_stature: Option<(f32, f64)>,
    /// Recent measurements of this person's height and of the ground height under them
    /// (metres), taken while their feet were seen.
    statures: Vec<(f32, f32)>,
}

/// One camera's image tracks.
#[derive(Default)]
pub struct ImageTracker {
    tracks: Vec<Track>,
    /// For the detections of the latest `assign`: which track each went to.
    pending: Vec<usize>,
    next_uid: u64,
    /// The learned "same person?" model, when one is trained.
    model: Option<Arc<assoc::Model>>,
}

/// One detection for `assign`: object family, box, outfit.
pub struct ImageBox {
    pub family: u8,
    pub bbox: [f32; 4],
    pub outfit: [f32; OUTFIT],
    /// Ground position (x, z, metres) and distance from the camera, when placed.
    pub ground: Option<(f64, f64, f64)>,
    /// The person's height as measured in this frame (metres), when their feet are seen.
    pub stature: Option<f32>,
}

/// The longest gap the association model judges, seconds (what it was trained on); longer gaps
/// fall back to the hand-set rule.
const MODEL_GAP: f64 = 3.0;

/// How long someone who left the picture is remembered for recognising them by outfit, seconds.
fn recall(family: u8) -> f64 {
    match family {
        0 => 120.0,
        _ => 0.0,
    }
}

/// Outfit match needed to recognise a returning person, and how much better than the next-best
/// candidate it must be.
const RECALL_MATCH: f32 = 0.85;
const RECALL_MARGIN: f32 = 0.06;

/// How long a vanished track is remembered, seconds.
fn memory(family: u8) -> f64 {
    match family {
        0 => 6.0,  // people duck behind things
        1 => 10.0, // vehicles get hidden by people walking past
        _ => 3.0,
    }
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

impl ImageTracker {
    /// A tracker that decides people's links with the learned association model.
    pub fn with_model(model: Option<Arc<assoc::Model>>) -> Self {
        Self { model, ..Self::default() }
    }

    /// Link this frame's boxes (taken at `now`, seconds) to the camera's tracks. `aspect` is the
    /// image's width / height. Returns, per box, the 3D id its track already carries (None for a
    /// new object or one not yet placed). Call `commit` with the 3D ids afterwards.
    pub fn assign(&mut self, boxes: &[ImageBox], aspect: f32, now: f64) -> Vec<Option<u64>> {
        self.tracks.retain(|t| now - t.last <= memory(t.family).max(recall(t.family)));
        // Candidate links with a cost; cheapest first, one-to-one.
        let mut cand: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, t) in self.tracks.iter().enumerate() {
            if now - t.last > memory(t.family) {
                continue; // only recognised by outfit, below
            }
            let age = (now - t.last) as f32;
            // Where it should be now (only extrapolate briefly).
            let dt = age.min(0.5);
            let pred = [0, 1, 2, 3].map(|k| t.bbox[k] + t.vel[k] * dt);
            let h = (t.bbox[3] - t.bbox[1]).max(1e-3);
            let (cx, cy) = ((pred[0] + pred[2]) / 2.0, (pred[1] + pred[3]) / 2.0);
            for (di, b) in boxes.iter().enumerate() {
                if b.family != t.family {
                    continue;
                }
                if let (Some(m), 0) = (&self.model, t.family) {
                    if age as f64 <= MODEL_GAP {
                        let stature = t.seen_stature.filter(|s| now - s.1 <= MODEL_GAP).map(|s| s.0);
                        let a = Sighting { t: t.last, cam: 0, bbox: t.bbox, outfit: &t.seen_outfit, ground: t.ground, stature };
                        let c = Sighting { t: now, cam: 0, bbox: b.bbox, outfit: &b.outfit, ground: b.ground, stature: b.stature };
                        let p = m.prob(&assoc::features(&a, &c, aspect));
                        if p > 0.5 {
                            cand.push((-(p.max(1e-9).ln()) as f32, ti, di));
                        }
                        continue;
                    }
                }
                let o = iou(&pred, &b.bbox);
                let app = outfit_match(&t.outfit, &b.outfit);
                let (bx, by) = ((b.bbox[0] + b.bbox[2]) / 2.0, (b.bbox[1] + b.bbox[3]) / 2.0);
                // Centre distance in box heights (x scaled to image units).
                let dn = ((bx - cx) * aspect).hypot(by - cy) / h;
                // Overlapping where it should be (outfit not contradicting), or — after being
                // hidden — near where it was with a clearly matching outfit.
                let ok = (o >= 0.2 && app >= 0.5) || (dn <= 1.0 + 0.5 * age && app >= 0.75);
                if ok {
                    cand.push(((1.0 - o) * 0.4 + (1.0 - app) * 0.6 + 0.05 * age, ti, di));
                }
            }
        }
        // Pair them up globally: the set of links with the lowest total cost, one-to-one.
        const NO_LINK: f64 = 1e3;
        let mut cost = vec![vec![NO_LINK; boxes.len()]; self.tracks.len()];
        for &(c, ti, di) in &cand {
            cost[ti][di] = c as f64;
        }
        let mut used_t = vec![false; self.tracks.len()];
        let mut pending = vec![usize::MAX; boxes.len()];
        for (ti, di) in assoc::assign(&cost).into_iter().enumerate() {
            if let Some(di) = di.filter(|&di| cost[ti][di] < NO_LINK) {
                used_t[ti] = true;
                pending[di] = ti;
            }
        }
        // Newcomers: someone who left earlier coming back? Only an unmistakable outfit match.
        for (di, b) in boxes.iter().enumerate() {
            if pending[di] != usize::MAX {
                continue;
            }
            let mut scored: Vec<(f32, usize)> = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(ti, t)| !used_t[*ti] && t.family == b.family && t.gid.is_some() && now - t.last > memory(t.family))
                .map(|(ti, t)| (outfit_match(&t.outfit, &b.outfit), ti))
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            if std::env::var("MINASTIR_DEBUG_TRACK").is_ok() && b.family == 0 {
                let near: Vec<String> = self
                    .tracks
                    .iter()
                    .filter(|t| t.family == 0 && t.gid.is_some())
                    .map(|t| format!("{}:{:.2}{}", t.gid.unwrap(), outfit_match(&t.outfit, &b.outfit), if now - t.last > memory(t.family) { "(gone)" } else { "" }))
                    .collect();
                eprintln!("newcomer at {now:.1}s box {:?}: outfit vs known people {}", b.bbox.map(|v| (v * 100.0).round() / 100.0), near.join(" "));
            }
            if let Some(&(best, ti)) = scored.first() {
                let next = scored.get(1).map_or(0.0, |s| s.0);
                if best >= RECALL_MATCH && best - next >= RECALL_MARGIN {
                    used_t[ti] = true;
                    pending[di] = ti;
                }
            }
        }
        let mut hints = vec![None; boxes.len()];
        for (di, b) in boxes.iter().enumerate() {
            if pending[di] == usize::MAX {
                self.next_uid += 1;
                self.tracks.push(Track { uid: self.next_uid, gid: None, family: b.family, bbox: b.bbox, vel: [0.0; 4], outfit: b.outfit, seen_outfit: b.outfit, ground: b.ground, last: now, seen_stature: b.stature.map(|h| (h, now)), statures: Vec::new() });
                pending[di] = self.tracks.len() - 1;
                continue;
            }
            let t = &mut self.tracks[pending[di]];
            let dt = (now - t.last) as f32;
            if dt > 1e-3 && dt < 0.6 {
                for k in 0..4 {
                    t.vel[k] = 0.6 * t.vel[k] + 0.4 * (b.bbox[k] - t.bbox[k]) / dt;
                }
            } else {
                t.vel = [0.0; 4];
            }
            t.bbox = b.bbox;
            t.seen_outfit = b.outfit;
            t.ground = b.ground;
            if let Some(h) = b.stature {
                t.seen_stature = Some((h, now));
            }
            for k in 0..OUTFIT {
                t.outfit[k] = 0.85 * t.outfit[k] + 0.15 * b.outfit[k];
            }
            t.last = now;
            hints[di] = t.gid;
        }
        self.pending = pending;
        hints
    }

    /// The image-track ids of the boxes of the latest `assign`.
    pub fn uids(&self) -> Vec<u64> {
        self.pending.iter().map(|&ti| self.tracks.get(ti).map_or(0, |t| t.uid)).collect()
    }

    /// Record a measurement of the height of the person of box `di` of the latest `assign`, and
    /// of the ground height under their feet.
    pub fn add_stature(&mut self, di: usize, h: f32, floor: f32) {
        if let Some(t) = self.pending.get(di).and_then(|&ti| self.tracks.get_mut(ti)) {
            t.statures.push((h, floor));
            if t.statures.len() > 40 {
                t.statures.remove(0);
            }
        }
    }

    /// The height of the person of box `di` of the latest `assign` and the ground height where
    /// they were last seen standing: medians of their measurements, once there are a few.
    pub fn stature(&self, di: usize) -> Option<(f32, f32)> {
        let t = self.tracks.get(*self.pending.get(di)?)?;
        if t.statures.len() < 3 {
            return None;
        }
        let med = |f: fn(&(f32, f32)) -> f32| {
            let mut v: Vec<f32> = t.statures.iter().map(f).collect();
            v.sort_by(f32::total_cmp);
            v[v.len() / 2]
        };
        let recent = &t.statures[t.statures.len().saturating_sub(5)..];
        let mut floor: Vec<f32> = recent.iter().map(|s| s.1).collect();
        floor.sort_by(f32::total_cmp);
        Some((med(|s| s.0), floor[floor.len() / 2]))
    }

    /// The 3D ids the boxes of the latest `assign` ended up with.
    pub fn commit(&mut self, gids: &[Option<u64>]) {
        for (di, g) in gids.iter().enumerate() {
            if let (Some(g), Some(&ti)) = (g, self.pending.get(di)) {
                if let Some(t) = self.tracks.get_mut(ti) {
                    t.gid = Some(*g);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(x: f32, colour: usize) -> ImageBox {
        let mut outfit = [0f32; OUTFIT];
        outfit[colour] = 1.0;
        outfit[BINS + colour] = 1.0;
        ImageBox { family: 0, bbox: [x, 0.4, x + 0.05, 0.6], outfit, ground: None, stature: None }
    }

    #[test]
    fn a_child_hidden_for_two_seconds_comes_back_as_themselves() {
        let mut it = ImageTracker::default();
        it.assign(&[bx(0.30, 2), bx(0.45, 9)], 16.0 / 9.0, 0.0);
        it.commit(&[Some(1), Some(2)]);
        // Pink (2) ducks behind the letterbox; black (9) stays.
        for k in 1..16 {
            let h = it.assign(&[bx(0.45, 9)], 16.0 / 9.0, k as f64 * 0.125);
            it.commit(&h);
            assert_eq!(h, vec![Some(2)]);
        }
        // Pink reappears a little to the side, two seconds later.
        let h = it.assign(&[bx(0.33, 2), bx(0.45, 9)], 16.0 / 9.0, 2.0);
        assert_eq!(h, vec![Some(1), Some(2)]);
    }

    #[test]
    fn someone_who_left_comes_back_as_themselves_by_outfit() {
        let mut it = ImageTracker::default();
        it.assign(&[bx(0.30, 2), bx(0.60, 9)], 16.0 / 9.0, 0.0);
        it.commit(&[Some(1), Some(2)]);
        // Both leave the picture; thirty seconds later pink (2) walks in somewhere else.
        let h = it.assign(&[bx(0.85, 2)], 16.0 / 9.0, 30.0);
        assert_eq!(h, vec![Some(1)]);
        // A stranger in a colour nobody wore is new.
        let h = it.assign(&[bx(0.85, 2), bx(0.10, 5)], 16.0 / 9.0, 30.2);
        assert_eq!(h, vec![Some(1), None]);
    }

    /// A stand-in for a trained model: "same" exactly when the outfits match.
    fn outfit_model() -> Arc<assoc::Model> {
        let mut w = [0.0; assoc::N];
        w[4] = 10.0; // outfit
        Arc::new(assoc::Model::single(assoc::Net { mean: [0.0; assoc::N], sd: [1.0; assoc::N], w1: vec![w], b1: vec![-7.5], w2: vec![5.0], b2: 0.0 }))
    }

    #[test]
    fn with_a_model_crossing_children_and_a_hidden_child_keep_their_ids() {
        let mut it = ImageTracker::with_model(Some(outfit_model()));
        it.assign(&[bx(0.30, 2), bx(0.40, 9)], 16.0 / 9.0, 0.0);
        it.commit(&[Some(1), Some(2)]);
        for k in 1..=8 {
            let s = k as f32 * 0.0125;
            let h = it.assign(&[bx(0.40 - s, 9), bx(0.30 + s, 2)], 16.0 / 9.0, k as f64 * 0.125);
            it.commit(&h);
            assert_eq!(h, vec![Some(2), Some(1)], "step {k}");
        }
        // Pink ducks out of sight for two seconds and comes back.
        let h = it.assign(&[bx(0.30, 9)], 16.0 / 9.0, 1.5);
        it.commit(&h);
        let h = it.assign(&[bx(0.42, 2), bx(0.30, 9)], 16.0 / 9.0, 3.0);
        assert_eq!(h, vec![Some(1), Some(2)]);
    }

    #[test]
    fn two_children_crossing_keep_their_own_ids() {
        let mut it = ImageTracker::default();
        it.assign(&[bx(0.30, 2), bx(0.40, 9)], 16.0 / 9.0, 0.0);
        it.commit(&[Some(1), Some(2)]);
        // They walk past each other.
        for k in 1..=8 {
            let s = k as f32 * 0.0125;
            let h = it.assign(&[bx(0.40 - s, 9), bx(0.30 + s, 2)], 16.0 / 9.0, k as f64 * 0.125);
            it.commit(&h);
            assert_eq!(h, vec![Some(2), Some(1)], "step {k}");
        }
    }
}
