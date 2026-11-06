//! The street in front of the house as clean parallel bands: road, kerb-side nature strips,
//! footpaths, front lawns.
//!
//! Per-camera ground maps are noisy at the cell level (a shaded footpath reads as road, borders
//! wobble), but a suburban street has strong structure: long straight strips, each one ground type,
//! all parallel. So we fuse every camera's map into one world grid, find the direction along which
//! the road cells line up most tightly, and then cut the across-street profile into bands with a
//! small dynamic programme (each band one type, a penalty per change). Shadows and stray labels
//! are outvoted along the band's whole length, and the result has exact, straight edges.

use crate::{groundmap::GroundMap, segment::Label};
use minastir_core::onvif::Pose;
use serde::Serialize;

const RES: f64 = 0.1;
/// A street strip must run on for at least this long (metres).
const MIN_RUN: f64 = 6.0;
/// Gaps (metres) a strip may have along its length — driveway crossings.
const MAX_GAP: f64 = 6.0;
/// Strips beside the road count as street if they start within this distance of the kerb…
const REACH: f64 = 6.0;
/// …and are cut off this far from it (the property line, roughly).
const VERGE: f64 = 8.0;
/// Bin size (metres) for driveway detection.
const DW_BIN: f64 = 0.5;
/// Ground types a band can be; index 0 = no ground data.
const TYPES: [Label; 4] = [Label::Road, Label::Sidewalk, Label::Grass, Label::Ground];

#[derive(Debug, Clone, Serialize)]
pub struct Band {
    /// Across-street offsets (metres) from the street's reference line.
    pub t0: f64,
    pub t1: f64,
    pub label: Label,
    /// Along-street extent with data in this band.
    pub s0: f64,
    pub s1: f64,
    /// Share of this band's evidence that agrees with its label.
    pub support: f64,
}

/// A paved strip crossing the street layout from the kerb into a lot.
#[derive(Debug, Clone, Serialize)]
pub struct Driveway {
    pub s0: f64,
    pub s1: f64,
    pub t0: f64,
    pub t1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Street {
    /// A point on the reference line (world x, z) and the along-street direction (degrees,
    /// same convention as yaw: 0 = +z, 90 = +x).
    pub origin: [f64; 2],
    pub heading: f64,
    pub bands: Vec<Band>,
    pub driveways: Vec<Driveway>,
}

/// A fused world grid of ground-type evidence.
struct Votes {
    x0: f64,
    z0: f64,
    n: usize,
    /// Per cell, weight per TYPES index.
    w: Vec<[f32; 4]>,
}

fn fuse(maps: &[(&GroundMap, Pose)]) -> Votes {
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for (g, p) in maps {
        lo = [lo[0].min(p.x - g.half), lo[1].min(p.z - g.half)];
        hi = [hi[0].max(p.x + g.half), hi[1].max(p.z + g.half)];
    }
    let n = (((hi[0] - lo[0]).max(hi[1] - lo[1])) / RES).ceil() as usize + 1;
    let mut w = vec![[0f32; 4]; n * n];
    for (g, p) in maps {
        let (s, c) = p.yaw.to_radians().sin_cos();
        let step = (RES / crate::groundmap::FINE).round().max(1.0) as usize;
        for r in (0..g.size).step_by(step) {
            for cc in (0..g.size).step_by(step) {
                let Some(k) = TYPES.iter().position(|&t| t as u8 == g.ground[r * g.size + cc]) else { continue };
                let lx = -g.half + (cc as f64 + 0.5) * crate::groundmap::FINE;
                let lz = -g.half + (r as f64 + 0.5) * crate::groundmap::FINE;
                let d = lx.hypot(lz);
                // Near ground is measured far better than distant ground.
                let wt = 1.0 / (1.0 + (d / 10.0).powi(2));
                let (wx, wz) = (p.x + lx * c + lz * s, p.z - lx * s + lz * c);
                let (i, j) = (((wx - lo[0]) / RES) as usize, ((wz - lo[1]) / RES) as usize);
                if i < n && j < n {
                    w[j * n + i][k] += wt as f32;
                }
            }
        }
    }
    Votes { x0: lo[0], z0: lo[1], n, w }
}

/// Fit the street from every camera's ground map (map, camera pose).
pub fn fit(maps: &[(&GroundMap, Pose)]) -> Option<Street> {
    if maps.is_empty() {
        return None;
    }
    let v = fuse(maps);
    let cells: Vec<(f64, f64, [f32; 4])> = (0..v.n * v.n)
        .filter(|&i| v.w[i].iter().any(|&x| x > 0.0))
        .map(|i| (v.x0 + ((i % v.n) as f64 + 0.5) * RES, v.z0 + ((i / v.n) as f64 + 0.5) * RES, v.w[i]))
        .collect();
    let road_total: f64 = cells.iter().map(|c| c.2[0] as f64).sum();
    if road_total < 50.0 {
        return None;
    }
    let centre = {
        let (mut sx, mut sz) = (0.0, 0.0);
        for c in &cells {
            sx += c.0 * c.2[0] as f64;
            sz += c.1 * c.2[0] as f64;
        }
        [sx / road_total, sz / road_total]
    };

    // 1. Direction: the road's across-street profile is sharpest (road cells pile into the fewest
    //    bins) when we look straight along the street.
    let profile_sharpness = |heading: f64| -> f64 {
        let (s, c) = heading.to_radians().sin_cos();
        let mut hist = std::collections::HashMap::<i64, f64>::new();
        for cell in &cells {
            let t = (cell.0 - centre[0]) * c - (cell.1 - centre[1]) * s; // across
            *hist.entry((t / 0.25).floor() as i64).or_default() += cell.2[0] as f64;
        }
        hist.values().map(|h| h * h).sum::<f64>()
    };
    let mut best = (0.0, f64::MIN);
    for k in 0..180 {
        let h = k as f64;
        let q = profile_sharpness(h);
        if q > best.1 {
            best = (h, q);
        }
    }
    let mut heading = best.0;
    for k in -10..=10 {
        let h = best.0 + k as f64 * 0.1;
        let q = profile_sharpness(h);
        if q > best.1 {
            best = (h, q);
            heading = h;
        }
    }

    // 2. Across-street profile: evidence per type per 10 cm of offset.
    let (s, c) = heading.to_radians().sin_cos();
    let along = |x: f64, z: f64| (x - centre[0]) * s + (z - centre[1]) * c;
    let across = |x: f64, z: f64| (x - centre[0]) * c - (z - centre[1]) * s;
    let ts: Vec<f64> = cells.iter().map(|cl| across(cl.0, cl.1)).collect();
    let tmin = ts.iter().cloned().fold(f64::MAX, f64::min);
    let tmax = ts.iter().cloned().fold(f64::MIN, f64::max);
    let nb = ((tmax - tmin) / RES).ceil() as usize + 1;
    let mut prof = vec![[0f64; 4]; nb];
    let mut srange = vec![(f64::MAX, f64::MIN); nb];
    for (cl, t) in cells.iter().zip(&ts) {
        let b = ((t - tmin) / RES) as usize;
        for k in 0..4 {
            prof[b][k] += cl.2[k] as f64;
        }
        let sa = along(cl.0, cl.1);
        srange[b] = (srange[b].0.min(sa), srange[b].1.max(sa));
    }

    // 3. Bands: states 0..4 = TYPES, 4 = no data. Cost per bin = evidence against the state,
    //    plus a fixed cost per change (so a band must be worth its edges).
    let total: Vec<f64> = prof.iter().map(|p| p.iter().sum()).collect();
    let mean_total = total.iter().sum::<f64>() / nb as f64;
    let cost = |b: usize, st: usize| -> f64 {
        if st == 4 {
            // "Nothing here" is cheap only where there is little evidence.
            total[b]
        } else {
            total[b] - prof[b][st] + 0.02 * mean_total
        }
    };
    let change = 6.0 * mean_total; // ≈ the evidence in 60 cm of a typical strip
    let mut dp = vec![[f64::MAX; 5]; nb];
    let mut from = vec![[0usize; 5]; nb];
    for st in 0..5 {
        dp[0][st] = cost(0, st);
    }
    for b in 1..nb {
        for st in 0..5 {
            let (mut bv, mut bs) = (f64::MAX, 0);
            for prev in 0..5 {
                let v = dp[b - 1][prev] + if prev == st { 0.0 } else { change };
                if v < bv {
                    (bv, bs) = (v, prev);
                }
            }
            dp[b][st] = bv + cost(b, st);
            from[b][st] = bs;
        }
    }
    let mut states = vec![0usize; nb];
    states[nb - 1] = (0..5).min_by(|&a, &b| dp[nb - 1][a].total_cmp(&dp[nb - 1][b])).unwrap();
    for b in (1..nb).rev() {
        states[b - 1] = from[b][states[b]];
    }

    let mut bands = Vec::new();
    let mut b0 = 0;
    for b in 1..=nb {
        if b == nb || states[b] != states[b0] {
            let st = states[b0];
            if st < 4 {
                let (t0, t1) = (tmin + b0 as f64 * RES, tmin + b as f64 * RES);
                // Along-street run: 1 m bins where this band's type holds its own. A real street
                // strip runs on for many metres; a driveway or patio crossing the profile doesn't.
                let mut agree_s = std::collections::BTreeMap::<i64, (f64, f64)>::new();
                for (cl, t) in cells.iter().zip(&ts) {
                    if *t >= t0 && *t < t1 {
                        let e = agree_s.entry(along(cl.0, cl.1).floor() as i64).or_default();
                        e.0 += cl.2[st] as f64;
                        e.1 += cl.2.iter().sum::<f32>() as f64;
                    }
                }
                // Merge good bins into runs, bridging gaps up to MAX_GAP (driveway crossings).
                let mut run: Option<(i64, i64)> = None;
                let mut cur: Option<(i64, i64)> = None;
                for (&k, &(a, all)) in &agree_s {
                    if !(all > 0.0 && a > 0.35 * all) {
                        continue;
                    }
                    cur = match cur {
                        Some((c0, c1)) if (k - c1) as f64 <= MAX_GAP + 1.0 => Some((c0, k)),
                        _ => Some((k, k)),
                    };
                    let c = cur.unwrap();
                    if run.is_none_or(|r| c.1 - c.0 > r.1 - r.0) {
                        run = Some(c);
                    }
                }
                if let Some((r0, r1)) = run {
                    let len = (r1 - r0 + 1) as f64;
                    if len >= MIN_RUN {
                        let agree: f64 = (b0..b).map(|i| prof[i][st]).sum();
                        let all: f64 = (b0..b).map(|i| total[i]).sum();
                        bands.push(Band { t0, t1, label: TYPES[st], s0: r0 as f64, s1: r1 as f64 + 1.0, support: agree / all.max(1e-9) });
                    }
                }
            }
            b0 = b;
        }
    }
    // The street reserve: the main road (longest road strip) plus the strips either side of it,
    // out to the property line. Anything starting further out belongs to the lots, not the street.
    let main = bands
        .iter()
        .enumerate()
        .filter(|(_, b)| b.label == Label::Road)
        .max_by(|a, b| (a.1.s1 - a.1.s0).total_cmp(&(b.1.s1 - b.1.s0)))?
        .0;
    if bands[main].s1 - bands[main].s0 < 15.0 {
        return None;
    }
    let (r0, r1) = (bands[main].t0, bands[main].t1);
    let bands: Vec<Band> = bands
        .into_iter()
        .filter(|b| (b.t0 >= r1 && b.t0 - r1 <= REACH) || (b.t1 <= r0 && r0 - b.t1 <= REACH) || (b.t0 == r0 && b.t1 == r1))
        .map(|mut b| {
            b.t0 = b.t0.max(r0 - VERGE);
            b.t1 = b.t1.min(r1 + VERGE);
            b
        })
        .collect();
    // Driveways: paved ground (road- or footpath-labelled; the model has no driveway class)
    // running across the strips beside the road, on either side.
    let footpaths: Vec<(f64, f64)> = bands.iter().filter(|b| b.label == Label::Sidewalk).map(|b| (b.t0, b.t1)).collect();
    let mut driveways = Vec::new();
    for side in [-1.0f64, 1.0] {
        let kerb = if side < 0.0 { r0 } else { r1 };
        let dist = |t: f64| (t - kerb) * side;
        let paved = |w: &[f32; 4]| (w[0] + w[1]) as f64;
        let all = |w: &[f32; 4]| w.iter().sum::<f32>() as f64;
        // Along the street: how paved the verge is (footpath strips excluded — paved anyway).
        let mut sb = std::collections::BTreeMap::<i64, (f64, f64)>::new();
        for (cl, &t) in cells.iter().zip(&ts) {
            let d = dist(t);
            if !(0.0..8.0).contains(&d) || footpaths.iter().any(|&(a, b)| t >= a && t < b) {
                continue;
            }
            let e = sb.entry((along(cl.0, cl.1) / DW_BIN).floor() as i64).or_default();
            e.0 += paved(&cl.2);
            e.1 += all(&cl.2);
        }
        let mut runs: Vec<(i64, i64)> = Vec::new();
        for (&k, &(p, a)) in &sb {
            if a <= 0.0 || p < 0.5 * a {
                continue;
            }
            match runs.last_mut() {
                Some(r) if k - r.1 <= 2 => r.1 = k,
                _ => runs.push((k, k)),
            }
        }
        for (k0, k1) in runs {
            let (s0, s1) = (k0 as f64 * DW_BIN, (k1 + 1) as f64 * DW_BIN);
            if !(2.4..=9.0).contains(&(s1 - s0)) {
                continue;
            }
            // Into the lot: as far as the ground across this stretch stays paved.
            let mut db = std::collections::BTreeMap::<i64, (f64, f64)>::new();
            for (cl, &t) in cells.iter().zip(&ts) {
                let (d, sa) = (dist(t), along(cl.0, cl.1));
                if d >= 0.0 && d < 14.0 && sa >= s0 && sa < s1 {
                    let e = db.entry((d / DW_BIN).floor() as i64).or_default();
                    e.0 += paved(&cl.2);
                    e.1 += all(&cl.2);
                }
            }
            let (mut reach, mut misses) = (0i64, 0);
            for k in 0..(14.0 / DW_BIN) as i64 {
                match db.get(&k) {
                    Some(&(p, a)) if a > 0.0 && p >= 0.45 * a => {
                        reach = k + 1;
                        misses = 0;
                    }
                    _ => {
                        misses += 1;
                        if misses > 2 {
                            break;
                        }
                    }
                }
            }
            let len = reach as f64 * DW_BIN;
            if len >= 2.0 {
                let far = kerb + side * len;
                driveways.push(Driveway { s0, s1, t0: kerb.min(far), t1: kerb.max(far) });
            }
        }
    }
    Some(Street { origin: centre, heading, bands, driveways })
}
