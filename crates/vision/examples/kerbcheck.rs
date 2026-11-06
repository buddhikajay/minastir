//! Draws lines of the fitted street (kerbs, as across-street distances) into a camera's frame,
//! through its calibrated lens: if they land on the real kerbs, the camera's geometry holds at
//! that range. Diagnostic.
//! cargo run --release -p minastir-vision --example kerbcheck -- <data dir> <camera id> <frame.jpg> <out.png> <heading> <ox> <oz> <t>…
use image::Rgb;
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
    let model = CameraModel::new(c.pose.expect("placed"), &cal, maps.map(Arc::new));
    let mut img = image::open(&a[2])?.to_rgb8();
    let (w, h) = (img.width() as f64, img.height() as f64);
    let num = |i: usize| a[i].parse::<f64>().unwrap();
    let (hd, ox, oz) = (num(4).to_radians(), num(5), num(6));
    let n = (hd.cos(), -hd.sin());
    let colours = [Rgb([255, 0, 0]), Rgb([255, 200, 0]), Rgb([0, 255, 255]), Rgb([255, 0, 255])];
    let ts: Vec<f64> = a[7..].iter().map(|s| s.parse::<f64>().unwrap()).collect();
    // Each pixel's own ground point (through the camera's lens and scene map); pixels whose
    // across-street distance is within a few centimetres of a line are drawn.
    let mut marks = Vec::new();
    for py in (0..img.height()).step_by(2) {
        for px in (0..img.width()).step_by(2) {
            let Some(g) = model.ground_point3(px as f64 / w, py as f64 / h, 80.0) else { continue };
            let t = (g[0] - ox) * n.0 + (g[2] - oz) * n.1;
            if let Some(k) = ts.iter().position(|l| (t - l).abs() < 0.06) {
                marks.push((px, py, k));
            }
        }
    }
    for (px, py, k) in marks {
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            if px + dx < img.width() && py + dy < img.height() {
                img.put_pixel(px + dx, py + dy, colours[k % colours.len()]);
            }
        }
    }
    img.save(&a[3])?;
    Ok(())
}
