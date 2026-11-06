//! Solve camera placements from matched points and report vehicle agreement before/after.
//! cargo run --release -p minastir-vision --example register -- <data dir> [out cameras.json]
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{
    calibrate::resize, detector::{Detector, INPUT}, geometry::CameraModel, matcher::Matcher, onnx::Device,
    register::{match_edges, place, RegCam}, scenemap::LensMap,
};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let mut cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut reg = Vec::new();
    let mut maps_of = std::collections::HashMap::new();
    for c in &cams {
        let (Some(pose), Some(cal)) = (c.pose, c.calibration.clone()) else { continue };
        let maps: Vec<LensMap> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect::<Result<_, _>>()?;
        let maps = Arc::new(maps);
        maps_of.insert(c.id(), maps.clone());
        let mut cal1 = cal.clone(); cal1.scale = 1.0;
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height, ..pose };
        reg.push(RegCam {
            id: c.id(), label: c.name.clone().unwrap_or(c.label()), rough: pose, local: CameraModel::new(zero, &cal1, Some(maps)),
            calibration: cal, image: image::open(data.join(c.id()).join("background.png"))?.to_rgb8(),
        });
    }
    let model = std::env::var("MATCHER").unwrap_or("aliked_lightglue_k2048".into());
    let min_score: f32 = std::env::var("MIN_SCORE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let mut matcher = Matcher::load(&data.join(format!("models/{model}.onnx")), Device::Cpu)?;
    let t = std::time::Instant::now();
    let edges = match_edges(&reg, &mut matcher, min_score, &mut |m| println!("  {m}"))?;
    let placed = place(&reg, &edges);
    println!("solved in {:?}", t.elapsed());
    for r in &placed {
        let c = reg.iter().find(|c| c.id == r.id).unwrap();
        println!("  {:<28} yaw {:>6.1}→{:>6.1}  pos ({:>5.1},{:>5.1})→({:>5.1},{:>5.1})  scale {:.3}  {}",
            c.label, c.rough.yaw, r.pose.yaw, c.rough.x, c.rough.z, r.pose.x, r.pose.z, r.scale,
            match &r.alignment { Some(al) if al.anchor => "[anchor]".into(), Some(al) => format!("via {} ({} pts, rms {:.2} m)", al.reference.as_deref().unwrap_or("?"), al.inliers, al.rms), None => "kept (no overlap)".into() });
    }

    // ---- evaluation: parked vehicles in the backgrounds, placed by each camera ----
    let mut det = Detector::load(&data.join("models/dfine_m_obj2coco.onnx"), true)?;
    let mut vehicles: Vec<(String, Vec<(f64, f64, f64, f64)>)> = Vec::new(); // id, (u, v, depth-push, score)
    for c in &reg {
        let (w, h) = (c.image.width() as usize, c.image.height() as usize);
        let tiles = if w as f64 / h as f64 >= 1.9 { 2 } else { 1 };
        let mut list = Vec::new();
        for t in 0..tiles {
            let x0 = w * t / tiles; let x1 = w * (t + 1) / tiles;
            let crop = image::imageops::crop_imm(&c.image, x0 as u32, 0, (x1 - x0) as u32, h as u32).to_image();
            let rgb = resize(crop.as_raw(), x1 - x0, h, INPUT, INPUT);
            for d in det.detect(&[&rgb], 0.5)?.remove(0) {
                if !matches!(d.class, "car" | "truck" | "bus") { continue; }
                let u = (t as f64 + ((d.bbox[0] + d.bbox[2]) / 2.0) as f64) / tiles as f64;
                list.push((u, d.bbox[3] as f64, if d.class == "car" { 2.0 } else { 2.8 }, d.score as f64));
            }
        }
        vehicles.push((c.id.clone(), list));
    }
    let report = |name: &str, poses: &dyn Fn(&str) -> (Pose, f64)| {
        let mut world: Vec<(String, Vec<(f64, f64)>)> = Vec::new();
        for (id, list) in &vehicles {
            let c = reg.iter().find(|c| &c.id == id).unwrap();
            let (pose, scale) = poses(id);
            let mut cal = c.calibration.clone(); cal.scale = scale;
            let m = CameraModel::new(pose, &cal, Some(maps_of[id].clone()));
            let pts = list.iter().filter_map(|&(u, v, push, _)| {
                let (x, z) = m.ground_point(u, v, 45.0)?;
                let (dx, dz) = (x - pose.x, z - pose.z); let l = dx.hypot(dz).max(1e-6);
                Some((x + dx / l * push, z + dz / l * push))
            }).collect();
            world.push((c.label.clone(), pts));
        }
        let mut gaps = Vec::new();
        for i in 0..world.len() { for j in i + 1..world.len() {
            // greedy one-to-one matching within 6 m
            let mut cand: Vec<(f64, usize, usize)> = Vec::new();
            for (p, a) in world[i].1.iter().enumerate() { for (q, b) in world[j].1.iter().enumerate() {
                let d = (a.0 - b.0).hypot(a.1 - b.1); if d < 6.0 { cand.push((d, p, q)); }
            }}
            cand.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut up, mut uq, mut ds) = (vec![false; world[i].1.len()], vec![false; world[j].1.len()], Vec::new());
            for (d, p, q) in cand { if !up[p] && !uq[q] { up[p] = true; uq[q] = true; ds.push(d); } }
            if !ds.is_empty() { println!("    {} ↔ {}: {}", world[i].0, world[j].0, ds.iter().map(|d| format!("{d:.1}")).collect::<Vec<_>>().join(", ")); gaps.extend(ds); }
        }}
        gaps.sort_by(f64::total_cmp);
        let mean = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;
        println!("  {name}: {} shared vehicles, median gap {:.2} m, mean {:.2} m", gaps.len(), gaps.get(gaps.len() / 2).copied().unwrap_or(f64::NAN), mean);
    };
    println!("vehicle agreement (same parked vehicle as placed by different cameras):");
    report("BEFORE (hand placement)", &|id| (reg.iter().find(|c| c.id == id).unwrap().rough, 1.0));
    report("AFTER  (matched)", &|id| { let r = placed.iter().find(|r| r.id == id).unwrap(); (r.pose, r.scale) });

    if let Some(out) = a.get(1) {
        for r in &placed {
            if let Some(c) = cams.iter_mut().find(|c| c.id() == r.id) {
                c.pose = Some(r.pose);
                if let Some(cal) = c.calibration.as_mut() { cal.scale = r.scale; cal.alignment = r.alignment.clone(); }
            }
        }
        std::fs::write(out, serde_json::to_vec_pretty(&cams)?)?;
    }
    Ok(())
}
