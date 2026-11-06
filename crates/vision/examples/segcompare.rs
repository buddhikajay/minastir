//! Whole-image vs multi-scale labelling, side by side. Diagnostic.
//! cargo run --release -p minastir-vision --example segcompare -- <data dir> <camera id> <out.png> [fine_short]
use minastir_vision::{calibrate::{label_color, resize}, onnx::Device, segment::{Segmenter, INPUT}};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let fine: usize = a.get(3).map_or(1280, |s| s.parse().unwrap());
    let img = image::open(data.join(&a[1]).join("background.png"))?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut seg = Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?;
    let n = if w as f64 / h as f64 > 2.2 { 2 } else { 1 };
    let ow = 1600u32;
    let oh = (ow as usize * h / w) as u32;
    let mut out = image::RgbImage::new(ow, oh * 2);
    for i in 0..n {
        let (x0, x1) = (w * i / n, w * (i + 1) / n);
        let crop = image::imageops::crop_imm(&img, x0 as u32, 0, (x1 - x0) as u32, h as u32).to_image();
        let cw = x1 - x0;
        let t = std::time::Instant::now();
        let coarse = seg.segment(&resize(crop.as_raw(), cw, h, INPUT, INPUT))?;
        let t1 = t.elapsed();
        let t = std::time::Instant::now();
        let fine_map = seg.segment_multiscale(crop.as_raw(), cw, h, fine)?;
        println!("lens {i}: whole {:?} ({}x{}), multiscale {:?} ({}x{})", t1, coarse.width, coarse.height, t.elapsed(), fine_map.width, fine_map.height);
        let (px0, px1) = (ow as usize * i / n, ow as usize * (i + 1) / n);
        for y in 0..oh {
            for x in px0..px1 {
                let u = (x - px0) as f64 / (px1 - px0) as f64;
                let v = y as f64 / oh as f64;
                let src = img.get_pixel((x0 as f64 + u * cw as f64) as u32, (v * h as f64) as u32).0;
                for (row, m) in [(0, &coarse), (1, &fine_map)] {
                    let c = label_color(m.at(u, v));
                    let px = [0, 1, 2].map(|k| (src[k] as f32 * 0.35 + c[k] as f32 * 0.65) as u8);
                    out.put_pixel(x as u32, y + row * oh, image::Rgb(px));
                }
            }
        }
    }
    out.save(&a[2])?;
    Ok(())
}
