//! Ground points of normalised image points through a camera's lens and scene map, and the
//! direction from each point to the next. Diagnostic.
//! cargo run --release -p minastir-vision --example groundpts -- <data dir> <camera id> <u,v>…
use minastir_core::onvif::CameraInfo;
use minastir_vision::{geometry::CameraModel, scenemap::LensMap};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let infos: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let c = infos.iter().find(|c| c.id() == a[1]).expect("camera");
    let cal = c.calibration.clone().expect("calibrated");
    let maps: Option<Vec<LensMap>> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin"))).ok()).collect();
    let flat = std::env::var("FLAT").is_ok();
    let model = CameraModel::new(c.pose.expect("placed"), &cal, if flat { None } else { maps.map(Arc::new) });
    let pts: Vec<[f64; 3]> = a[2..]
        .iter()
        .filter_map(|s| {
            let uv: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
            let g = model.ground_point3(uv[0], uv[1], 80.0);
            println!("({:.4},{:.4}) → {:?}", uv[0], uv[1], g.map(|g| g.map(|x| (x * 100.0).round() / 100.0)));
            g
        })
        .collect();
    for w in pts.windows(2) {
        let (dx, dz) = (w[1][0] - w[0][0], w[1][2] - w[0][2]);
        println!("direction {:.1}° (axis {:.1}°), {:.2} m apart", dx.atan2(dz).to_degrees().rem_euclid(360.0), dx.atan2(dz).to_degrees().rem_euclid(180.0), dx.hypot(dz));
    }
    Ok(())
}
