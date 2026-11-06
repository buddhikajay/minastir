//! Calibrate a set of cameras together.
//!
//! Each camera is first measured on its own (MoGe + SegFormer → lens, tilt, lean, height). Single
//! -image depth is reliable along the line of sight but reads wide, distorted security lenses as
//! too narrow (a ~87° UniFi Bullet comes out at ~64°), which squeezes everything sideways and
//! makes overlapping cameras disagree. So we then compare overlapping cameras through matched
//! points: for each camera, try widening its lens and keep the width at which its 3D points agree
//! best with its neighbours' (tightest fit, depth scale closest to 1). Finally the cameras are
//! placed relative to each other from those points.

use crate::{
    align::solve_similarity,
    calibrate::{assemble, CalibrationResult, LensRaw},
    geometry::CameraModel,
    matcher::Matcher,
    register::{collect_matches, edges_from, place, RawMatch, RegCam, Registered},
};
use anyhow::Result;
use minastir_core::onvif::Pose;
use std::{collections::HashMap, sync::Arc};

pub struct SurveyCam {
    pub id: String,
    pub label: String,
    pub rough: Pose,
    pub image: image::RgbImage,
    pub raw: Vec<LensRaw>,
}

pub struct Surveyed {
    pub id: String,
    pub result: CalibrationResult,
    pub placed: Registered,
}

/// Field-of-view widening factors tried (tan of the half-angle × k). MoGe under-reads wide
/// lenses; it doesn't over-read them, so nothing below 1.
const KS: [f64; 7] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6];
/// A camera's lens is only changed when that improves agreement by at least this much.
const MIN_GAIN: f64 = 0.04;

fn local_model(r: &CalibrationResult, _rough: Pose) -> CameraModel {
    let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: r.calibration.height, pitch: r.calibration.pitch };
    let mut cal = r.calibration.clone();
    cal.scale = 1.0;
    CameraModel::new(zero, &cal, Some(Arc::new(r.lens_maps())))
}

/// How badly two cameras disagree: RMS of the robust 3D fit plus a penalty for needing a depth
/// scale far from 1 (both depths are metric, so a big scale means a lens is wrong).
fn disagreement(ma: &CameraModel, mb: &CameraModel, ms: &[RawMatch]) -> f64 {
    let pairs: Vec<_> = ms
        .iter()
        .filter_map(|m| {
            let (pa, pb) = (ma.point_at(m.a.0, m.a.1)?, mb.point_at(m.b.0, m.b.1)?);
            (pa[0].hypot(pa[1]) < 35.0 && pb[0].hypot(pb[1]) < 35.0).then_some((pa, pb))
        })
        .collect();
    match solve_similarity(&pairs) {
        Some((s, inl, rms)) if inl.len() >= 12 => rms + 0.8 * s.s.ln().abs(),
        _ => 1.5,
    }
}

pub fn survey(cams: Vec<SurveyCam>, matcher: &mut Matcher, log: &mut dyn FnMut(String)) -> Result<Vec<Surveyed>> {
    let n = cams.len();
    // Each camera's geometry at every lens width (cheap: the models already ran).
    let mut fits: Vec<Vec<Option<(CalibrationResult, CameraModel)>>> = Vec::new();
    for c in &cams {
        let mut per_k = Vec::new();
        // A panorama's width is fixed by its geometry; only flat lenses get widened.
        let fixed = c.raw.iter().any(|r| r.cylindrical);
        for k in KS {
            if fixed && k != 1.0 {
                per_k.push(None);
                continue;
            }
            per_k.push(assemble(&c.raw, &vec![k; c.raw.len()]).ok().map(|r| {
                let m = local_model(&r, c.rough);
                (r, m)
            }));
        }
        anyhow::ensure!(per_k[0].is_some(), "{}: not enough visible ground to calibrate", c.label);
        fits.push(per_k);
    }

    // Matches don't depend on the lens width: compute once.
    let reg0: Vec<RegCam> = cams
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (r, _) = fits[i][0].as_ref().unwrap();
            RegCam { id: c.id.clone(), label: c.label.clone(), rough: c.rough, calibration: r.calibration.clone(), local: local_model(r, c.rough), image: c.image.clone() }
        })
        .collect();
    let all = if n >= 2 { collect_matches(&reg0, matcher, 0.2)? } else { Vec::new() };
    let linked: Vec<&((usize, usize), Vec<RawMatch>)> = all.iter().filter(|(_, ms)| ms.len() >= 25).collect();

    // Coordinate descent over per-camera lens width.
    let mut k = vec![0usize; n];
    let mut memo: HashMap<(usize, usize, usize, usize), f64> = HashMap::new();
    let mut pair_cost = |a: usize, b: usize, ka: usize, kb: usize, ms: &[RawMatch]| -> f64 {
        *memo.entry((a, b, ka, kb)).or_insert_with(|| match (&fits[a][ka], &fits[b][kb]) {
            (Some((_, ma)), Some((_, mb))) => disagreement(ma, mb, ms),
            _ => 9.0,
        })
    };
    for _sweep in 0..3 {
        let mut changed = false;
        for i in 0..n {
            let mine: Vec<_> = linked.iter().filter(|((a, b), _)| *a == i || *b == i).collect();
            if mine.is_empty() {
                continue;
            }
            let mut total = |ki: usize| -> f64 {
                mine.iter()
                    .map(|((a, b), ms)| {
                        let (ka, kb) = if *a == i { (ki, k[*b]) } else { (k[*a], ki) };
                        pair_cost(*a, *b, ka, kb, ms)
                    })
                    .sum::<f64>()
                    / mine.len() as f64
            };
            let current = total(k[i]);
            let (best, cost) = (0..KS.len()).map(|ki| (ki, total(ki))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            if best != k[i] && cost + MIN_GAIN < current {
                log(format!("{}: lens {:.0}° → {:.0}° (disagreement {:.2} → {:.2})", cams[i].label,
                    fits[i][k[i]].as_ref().unwrap().0.calibration.lenses[0].hfov, fits[i][best].as_ref().unwrap().0.calibration.lenses[0].hfov, current, cost));
                k[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Place cameras with the chosen lenses.
    let mut chosen: Vec<CalibrationResult> = Vec::new();
    let mut reg = Vec::new();
    for (i, c) in cams.into_iter().enumerate() {
        let (r, m) = fits[i][k[i]].take().unwrap();
        reg.push(RegCam { id: c.id, label: c.label, rough: c.rough, calibration: r.calibration.clone(), local: m, image: c.image });
        chosen.push(r);
    }
    let edges = edges_from(&reg, &all, log);
    let placed = place(&reg, &edges);
    Ok(chosen.into_iter().zip(placed).zip(reg).map(|((result, placed), c)| Surveyed { id: c.id, result, placed }).collect())
}
