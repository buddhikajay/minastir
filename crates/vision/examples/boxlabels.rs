//! What the background labels show inside given boxes for one camera. Diagnostic.
//! cargo run --release -p minastir-vision --example boxlabels -- <data dir> <camera id> x0,y0,x1,y1 …
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
    let model = CameraModel::new(c.pose.unwrap(), &cal, Some(Arc::new(maps)));
    for b in &a[2..] {
        let v: Vec<f64> = b.split(',').map(|x| x.parse().unwrap()).collect();
        println!("{b}: (vehicle, fixed, bin) = {:?}", model.box_labels([v[0], v[1], v[2], v[3]]));
    }
    Ok(())
}
