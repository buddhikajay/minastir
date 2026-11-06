//! Where one camera puts image points on the ground: scene-map lookup vs. the flat-ground lens
//! model, and the 3D point the background map has there. Diagnostic.
//! cargo run --release -p minastir-vision --example groundat -- <data dir> <camera id> u,v …
use minastir_core::onvif::CameraInfo;
use minastir_vision::{geometry::CameraModel, scenemap::LensMap};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let c = cams.iter().find(|c| c.id() == a[1]).expect("camera");
    let cal = c.calibration.clone().unwrap();
    let maps: Vec<LensMap> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect::<anyhow::Result<_>>()?;
    let with_map = CameraModel::new(c.pose.unwrap(), &cal, Some(Arc::new(maps)));
    let flat = CameraModel::new(c.pose.unwrap(), &cal, None);
    println!("camera at ({:.1},{:.1}) height {:.2} yaw {:.0} pitch {:.1}", c.pose.unwrap().x, c.pose.unwrap().z, c.pose.unwrap().height, c.pose.unwrap().yaw, c.pose.unwrap().pitch);
    for p in &a[2..] {
        let v: Vec<f64> = p.split(',').map(|x| x.parse().unwrap()).collect();
        let f = |o: Option<(f64, f64)>| o.map_or("—".into(), |(x, z)| format!("({x:.1},{z:.1})"));
        println!(
            "{p}: map ground {}  flat ground {}  background 3D {:?}",
            f(with_map.ground_point(v[0], v[1], 60.0)),
            f(flat.ground_point(v[0], v[1], 60.0)),
            with_map.point_at(v[0], v[1]).map(|q| q.map(|x| (x * 10.0).round() / 10.0))
        );
    }
    Ok(())
}
