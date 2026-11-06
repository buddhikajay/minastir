//! Re-run labelling (multi-scale) on each calibrated camera's background and replace the labels
//! in its saved scene maps, keeping the 3D points and the calibration untouched.
//! cargo run --release -p minastir-vision --example relabel -- <data dir>
use minastir_core::onvif::CameraInfo;
use minastir_vision::{onnx::Device, scenemap::LensMap, segment::Segmenter};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut seg = Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?;
    for c in &cams {
        let Some(cal) = &c.calibration else { continue };
        let dir = data.join(c.id());
        let Ok(img) = image::open(dir.join("background.png")) else { continue };
        let img = img.to_rgb8();
        let (w, h) = (img.width() as f64, img.height());
        for (i, l) in cal.lenses.iter().enumerate() {
            let path = dir.join(format!("scene_lens{i}.bin"));
            let mut map = LensMap::load(&path)?;
            let (x0, x1) = ((l.u0 * w).round() as u32, (l.u1 * w).round() as u32);
            let crop = image::imageops::crop_imm(&img, x0, 0, x1 - x0, h).to_image();
            let t = std::time::Instant::now();
            let lm = seg.segment_multiscale(crop.as_raw(), crop.width() as usize, h as usize, 1280)?;
            let mut changed = 0;
            for y in 0..map.height {
                for x in 0..map.width {
                    let k = y * map.width + x;
                    let nl = lm.at((x as f64 + 0.5) / map.width as f64, (y as f64 + 0.5) / map.height as f64);
                    changed += (nl != map.labels[k]) as usize;
                    map.labels[k] = nl;
                }
            }
            map.save(&path)?;
            println!("{} lens {i}: {:.1}% of labels changed ({:?})", c.name.clone().unwrap_or(c.id()), 100.0 * changed as f64 / map.labels.len() as f64, t.elapsed());
        }
    }
    Ok(())
}
