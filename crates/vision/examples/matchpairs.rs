//! Match every lens of every calibrated camera against every other; draw the matches.
//! cargo run --release -p minastir-vision --example matchpairs -- <data dir> <out dir>
use minastir_core::onvif::CameraInfo;
use minastir_vision::{matcher::{Img, Matcher}, onnx::Device};
use std::{path::Path, time::Instant};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, out) = (Path::new(&a[0]), Path::new(&a[1]));
    std::fs::create_dir_all(out)?;
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut m = Matcher::load(&data.join("models/aliked_lightglue_k2048.onnx"), Device::Cpu)?;
    // (camera label, lens index, image)
    let mut lenses = Vec::new();
    for c in &cams {
        let Some(cal) = &c.calibration else { continue };
        let img = image::open(data.join(c.id()).join("background.png"))?.to_rgb8();
        for (i, l) in cal.lenses.iter().enumerate() {
            let x0 = (l.u0 * img.width() as f64) as u32;
            let x1 = (l.u1 * img.width() as f64) as u32;
            let crop = image::imageops::crop_imm(&img, x0, 0, x1 - x0, img.height()).to_image();
            lenses.push((c.name.clone().unwrap_or(c.label()), i, crop));
        }
    }
    for i in 0..lenses.len() {
        for j in i + 1..lenses.len() {
            if lenses[i].0 == lenses[j].0 { continue; }
            let (la, lb) = (&lenses[i], &lenses[j]);
            let t = Instant::now();
            let ms = m.match_pair(
                &Img { rgb: la.2.as_raw(), width: la.2.width() as usize, height: la.2.height() as usize },
                &Img { rgb: lb.2.as_raw(), width: lb.2.width() as usize, height: lb.2.height() as usize },
                0.5,
            )?;
            println!("{}#{} ↔ {}#{}: {} matches ({:?})", la.0, la.1, lb.0, lb.1, ms.len(), t.elapsed());
            if ms.len() < 15 { continue; }
            // side-by-side drawing at 800 px height each
            let h = 600u32;
            let ra = image::imageops::resize(&la.2, la.2.width() * h / la.2.height(), h, image::imageops::FilterType::Triangle);
            let rb = image::imageops::resize(&lb.2, lb.2.width() * h / lb.2.height(), h, image::imageops::FilterType::Triangle);
            let mut canvas = image::RgbImage::new(ra.width() + rb.width(), h);
            image::imageops::replace(&mut canvas, &ra, 0, 0);
            image::imageops::replace(&mut canvas, &rb, ra.width() as i64, 0);
            for (k, mm) in ms.iter().enumerate() {
                let p0 = (mm.a.0 * ra.width() as f64, mm.a.1 * h as f64);
                let p1 = (ra.width() as f64 + mm.b.0 * rb.width() as f64, mm.b.1 * h as f64);
                let col = [[255u8, 60, 60], [60, 255, 60], [60, 160, 255], [255, 220, 0], [255, 0, 255]][k % 5];
                let steps = (p1.0 - p0.0).abs().max((p1.1 - p0.1).abs()) as usize;
                for s in 0..=steps {
                    let t = s as f64 / steps.max(1) as f64;
                    let (x, y) = ((p0.0 + (p1.0 - p0.0) * t) as u32, (p0.1 + (p1.1 - p0.1) * t) as u32);
                    if x < canvas.width() && y < h { canvas.put_pixel(x, y, image::Rgb(col)); }
                }
            }
            canvas.save(out.join(format!("{}{}_{}{}.jpg", la.0.replace(' ', ""), la.1, lb.0.replace(' ', ""), lb.1).replace(['(', ')'], "")))?;
        }
    }
    Ok(())
}
