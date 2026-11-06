//! Solve each camera lens's field of view / distortion / pose from the 3D points of every other
//! camera it shares matches with. Diagnostic for lens calibration.
//! cargo run --release -p minastir-vision --example resect -- <data dir>
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{
    align::solve_similarity, geometry::CameraModel, matcher::Matcher, onnx::Device,
    register::{collect_matches, depth_pairs, invert, RegCam},
    resect::{resect, LensPose, Obs}, scenemap::LensMap,
};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut reg = Vec::new();
    for c in &cams {
        let (Some(pose), Some(cal)) = (c.pose, c.calibration.clone()) else { continue };
        let maps: Vec<LensMap> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect::<Result<_, _>>()?;
        let mut cal1 = cal.clone(); cal1.scale = 1.0;
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height, ..pose };
        reg.push(RegCam {
            id: c.id(), label: c.name.clone().unwrap_or(c.label()), rough: pose, local: CameraModel::new(zero, &cal1, Some(Arc::new(maps))),
            calibration: cal, image: image::open(data.join(c.id()).join("background.png"))?.to_rgb8(),
        });
    }
    let min_score: f32 = std::env::var("MIN_SCORE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.2);
    let mut matcher = Matcher::load(&data.join("models/aliked_lightglue_k2048.onnx"), Device::Cpu)?;
    let all = collect_matches(&reg, &mut matcher, min_score)?;
    for ((a, b), ms) in &all {
        let (a, b) = (*a, *b);
        let pairs = depth_pairs(&reg, a, b, ms);
        let Some((sim_ba, inl, _)) = solve_similarity(&pairs) else { println!("{} ↔ {}: no 3D fit ({} matches)", reg[a].label, reg[b].label, ms.len()); continue };
        for (t, r) in [(b, a), (a, b)] {
            let sim = if t == b { sim_ba } else { invert(&sim_ba) }; // target-local → reference-local
            let cal = &reg[t].calibration;
            for (li, lens) in cal.lenses.iter().enumerate() {
                let obs: Vec<Obs> = ms.iter().filter_map(|m| {
                    let ((tl, tp), (rp_l, rp)) = if t == a { ((m.la, m.a), (m.lb, m.b)) } else { ((m.lb, m.b), (m.la, m.a)) };
                    let _ = rp_l;
                    if tl != li { return None; }
                    let p = reg[r].local.point_at(rp.0, rp.1)?;
                    if p[0].hypot(p[1]) > 40.0 { return None; }
                    Some(Obs { u: (tp.0 - lens.u0) / (lens.u1 - lens.u0), v: tp.1, p: [p[0], p[2], p[1]] })
                }).collect();
                let c3 = sim.apply([0.0, 0.0, cal.height]);
                let w = reg[t].image.width() as f64 * (lens.u1 - lens.u0);
                let aspect = w / reg[t].image.height() as f64;
                let mut best: Option<minastir_vision::resect::Resection> = None;
                for sign in [1.0, -1.0] {
                    let init = LensPose { yaw: sim.theta + sign * lens.yaw_offset, pitch: lens.pitch, roll: lens.roll, c: [c3[0], c3[2], c3[1]], tan_h: (lens.hfov.to_radians() / 2.0).tan(), k1: 0.0 };
                    if let Some(rs) = resect(&obs, init, aspect) {
                        if best.as_ref().is_none_or(|b| rs.inliers.len() > b.inliers.len()) { best = Some(rs); }
                    }
                    if lens.yaw_offset == 0.0 { break; }
                }
                match best {
                    Some(res) => println!(
                        "{:>26} lens{li} vs {:<26} obs {:>3} inl {:>3} rms {:.4} | hfov {:>5.1} (moge {:>5.1}) k1 {:+.3} pitch {:>5.1} ({:>5.1}) roll {:>5.1} ({:>5.1}) h {:.2} ({:.2}) yaw {:>6.1} (init {:>6.1}) [sim inl {}]",
                        reg[t].label, reg[r].label, obs.len(), res.inliers.len(), res.rms, res.pose.hfov(), lens.hfov, res.pose.k1, res.pose.pitch, lens.pitch, res.pose.roll, lens.roll, res.pose.c[1], c3[2], res.pose.yaw, sim.theta, inl.len()),
                    None => println!("{:>26} lens{li} vs {:<26} obs {:>3} — no solution", reg[t].label, reg[r].label, obs.len()),
                }
            }
        }
    }
    Ok(())
}
