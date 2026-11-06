//! Ground positions of every kerb pixel (road beside footpath / grass) in a camera's scene map, as
//! "x z u v" lines — to measure the street through that camera. Diagnostic.
//! cargo run --release -p minastir-vision --example kerbpts -- <data dir> <camera id>
use minastir_core::onvif::CameraInfo;
use minastir_vision::{geometry::CameraModel, scenemap::LensMap, segment::Label};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let infos: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let c = infos.iter().find(|c| c.id() == a[1]).expect("camera");
    let cal = c.calibration.clone().expect("calibrated");
    let map = LensMap::load(&data.join(c.id()).join("scene_lens0.bin"))?;
    let (w, h) = (map.width, map.height);
    let labels = map.labels.clone();
    let model = CameraModel::new(c.pose.expect("placed"), &cal, Some(Arc::new(vec![map])));
    let beside = |l: Label| matches!(l, Label::Sidewalk | Label::Grass | Label::Ground);
    // VEHICLES=1: every vehicle pixel's 3D point instead (x z u v height).
    if std::env::var("VEHICLES").is_ok() {
        for y in 0..h {
            for x in 0..w {
                if labels[y * w + x] == Label::Vehicle {
                    let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                    if let Some(p) = model.point_at(u, v) {
                        println!("{:.3} {:.3} {:.4} {:.4} {:.3}", p[0], p[1], u, v, p[2]);
                    }
                }
            }
        }
        return Ok(());
    }
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if labels[y * w + x] != Label::Road {
                continue;
            }
            if [labels[(y - 1) * w + x], labels[(y + 1) * w + x], labels[y * w + x - 1], labels[y * w + x + 1]].iter().any(|&l| beside(l)) {
                let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                if let Some(p) = model.ground_point3(u, v, 60.0) {
                    println!("{:.3} {:.3} {:.4} {:.4} {:.3}", p[0], p[2], u, v, p[1]);
                }
            }
        }
    }
    Ok(())
}
