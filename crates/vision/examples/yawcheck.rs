//! How well one camera's static scene (walls, fences, poles, trees, kerbs…) lines up with
//! another's, as the first camera's heading is varied — independent evidence for a heading
//! correction (no people involved). Diagnostic.
//! cargo run --release -p minastir-vision --example yawcheck -- <data dir> <camera to turn> <reference camera>
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{geometry::CameraModel, scenemap::LensMap, segment::Label};
use std::{collections::HashMap, path::Path, sync::Arc};

fn model(c: &CameraInfo, data: &Path, yaw_delta: f64) -> CameraModel {
    let cal = c.calibration.clone().unwrap();
    let maps: Vec<LensMap> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin"))).unwrap()).collect();
    let p = c.pose.unwrap();
    CameraModel::new(Pose { yaw: p.yaw + yaw_delta, ..p }, &cal, Some(Arc::new(maps)))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let (c, r) = (cams.iter().find(|c| c.id() == a[1]).unwrap(), cams.iter().find(|c| c.id() == a[2]).unwrap());
    let keep = |l: Label| matches!(l, Label::Building | Label::Fence | Label::Pole | Label::Tree | Label::Plant);
    // Reference points on a 0.25 m grid for nearest-neighbour lookups.
    let refp = model(r, data, 0.0).labelled_points(2, 25.0, keep);
    let mut grid: HashMap<(i64, i64), Vec<(f64, f64)>> = HashMap::new();
    for (x, z, _, _) in &refp {
        grid.entry(((x / 0.25).floor() as i64, (z / 0.25).floor() as i64)).or_default().push((*x, *z));
    }
    let nearest = |x: f64, z: f64| {
        let (gx, gz) = ((x / 0.25).floor() as i64, (z / 0.25).floor() as i64);
        let mut best = f64::MAX;
        for dx in -8..=8 {
            for dz in -8..=8 {
                for (px, pz) in grid.get(&(gx + dx, gz + dz)).into_iter().flatten() {
                    best = best.min((px - x).hypot(pz - z));
                }
            }
        }
        best.min(2.0)
    };
    // Only points both cameras could plausibly see: within 2.5 m of a reference point at the
    // best heading count as shared; score = share of the turned camera's points within 0.5 m.
    println!("reference: {} structure points from {}", refp.len(), r.name.clone().unwrap_or(r.id()));
    for k in (-2..=14).step_by(2) {
        let d = k as f64 * 2.5;
        let pts = model(c, data, d).labelled_points(2, 25.0, keep);
        let ds: Vec<f64> = pts.iter().map(|(x, z, _, _)| nearest(*x, *z)).collect();
        let close = ds.iter().filter(|v| **v < 0.5).count() as f64 / ds.len().max(1) as f64;
        let mut s = ds.clone();
        s.sort_by(f64::total_cmp);
        println!("heading {:+6.1}°: {:5.1}% of structure within 0.5 m of the reference's, median gap {:.2} m", d, close * 100.0, s[s.len() / 2]);
    }
    Ok(())
}
