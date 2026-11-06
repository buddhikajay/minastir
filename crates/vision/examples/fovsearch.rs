//! How well do camera pairs agree as each camera's field of view is widened? Diagnostic.
//! cargo run --release -p minastir-vision --example fovsearch -- <data dir>
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{
    align::solve_similarity, calibrate::{assemble, Calibrator}, geometry::CameraModel, matcher::Matcher, moge::Moge,
    onnx::Device, register::{collect_matches, RegCam}, segment::Segmenter,
};
use std::{path::Path, sync::Arc};

const KS: [f64; 9] = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7];

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut cal = Calibrator {
        moge: Moge::load(&data.join("models/moge2_vitl_normal.onnx"), Device::Cpu)?,
        segmenter: Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?,
        num_tokens: 2500,
    };
    let mut reg = Vec::new();
    let mut models: Vec<Vec<CameraModel>> = Vec::new(); // per camera, per k
    for c in &cams {
        let (Some(pose), Some(_)) = (c.pose, c.calibration.clone()) else { continue };
        if c.name.as_deref() == Some("Reolink Duo Floodlight PoE") || c.id() == "192168187" { continue; }
        let img = image::open(data.join(c.id()).join("background.png"))?.to_rgb8();
        let t = std::time::Instant::now();
        let raw = cal.analyse(img.as_raw(), img.width() as usize, img.height() as usize)?;
        eprintln!("analysed {} in {:?}", c.id(), t.elapsed());
        let mut per_k = Vec::new();
        let mut cal0 = None;
        for k in KS {
            let r = assemble(&raw, &vec![k; raw.len()])?;
            let maps = Arc::new(r.lens_maps());
            let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: r.calibration.height, pitch: r.calibration.pitch };
            println!("{:<26} k {:.1}: hfov {:>5.1}  pitch {:>5.1}  roll {:>5.1}  height {:.2}", c.name.clone().unwrap_or(c.id()), k,
                r.calibration.lenses[0].hfov, r.calibration.pitch, r.calibration.roll, r.calibration.height);
            per_k.push(CameraModel::new(zero, &r.calibration, Some(maps)));
            if k == 1.0 { cal0 = Some(r.calibration.clone()); }
        }
        models.push(per_k);
        let cal0 = cal0.unwrap();
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal0.height, pitch: cal0.pitch };
        reg.push(RegCam { id: c.id(), label: c.name.clone().unwrap_or(c.label()), rough: pose, local: CameraModel::new(zero, &cal0, None), calibration: cal0, image: img });
    }
    let mut matcher = Matcher::load(&data.join("models/aliked_lightglue_k2048.onnx"), Device::Cpu)?;
    let all = collect_matches(&reg, &mut matcher, 0.2)?;
    for ((a, b), ms) in &all {
        println!("\n{} (rows k) ↔ {} (cols k): inliers/rms/scale   [{} matches]", reg[*a].label, reg[*b].label, ms.len());
        for (ia, _) in KS.iter().enumerate() {
            let mut line = format!("{:.1} |", KS[ia]);
            for (ib, _) in KS.iter().enumerate() {
                let (ma, mb) = (&models[*a][ia], &models[*b][ib]);
                let pairs: Vec<_> = ms.iter().filter_map(|m| {
                    let (pa, pb) = (ma.point_at(m.a.0, m.a.1)?, mb.point_at(m.b.0, m.b.1)?);
                    (pa[0].hypot(pa[1]) < 35.0 && pb[0].hypot(pb[1]) < 35.0).then_some((pa, pb))
                }).collect();
                match solve_similarity(&pairs) {
                    Some((s, inl, rms)) => line += &format!(" {:>2}/{:.2}/{:.2}", inl.len(), rms, s.s),
                    None => line += "      --      ",
                }
            }
            println!("{line}");
        }
    }
    Ok(())
}
