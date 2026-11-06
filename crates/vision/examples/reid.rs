//! Do appearance fingerprints recognise the same vehicle across cameras? Diagnostic.
//! cargo run --release -p minastir-vision --example reid -- <models dir> <dir with {id}.jpg + boxes.json>
use minastir_vision::{appearance::{similarity, Crop, Embedder, Kind}, onnx::Device};
use std::{collections::BTreeMap, path::Path};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (models, dir) = (Path::new(&a[0]), Path::new(&a[1]));
    let boxes: BTreeMap<String, Vec<(String, f64, f64, f64, f64)>> = serde_json::from_slice(&std::fs::read(dir.join("boxes.json"))?)?;
    // Optional third argument forces a model: dino2 | dino3.
    let dev = if std::env::var("REID_GPU").is_ok() { Device::Gpu } else { Device::Cpu };
    let mut emb = match a.get(2).map(String::as_str) {
        Some("dino2") => Embedder::load_kind(&models.join(minastir_vision::appearance::MODEL_FILE), Kind::Dino, Device::Cpu)?,
        Some("dino3") => Embedder::load_kind(&models.join(minastir_vision::appearance::DINO3_MODEL_FILE), Kind::Dino3, dev)?,
        _ => Embedder::load_best(models, Device::Cpu)?,
    };
    println!("model: {:?}", emb.kind());
    let mut labels = Vec::new();
    let mut crops_img = Vec::new();
    for (id, bs) in &boxes {
        if bs.is_empty() { continue; }
        let img = image::open(dir.join(format!("{id}.jpg")))?.to_rgb8();
        let (w, h) = (img.width() as f64, img.height() as f64);
        for (k, (cls, x0, y0, x1, y1)) in bs.iter().enumerate() {
            let (px0, py0, px1, py1) = ((x0 * w).max(0.0), (y0 * h).max(0.0), (x1 * w).min(w), (y1 * h).min(h));
            if px1 - px0 < 24.0 || py1 - py0 < 24.0 { continue; }
            let c = image::imageops::crop_imm(&img, px0 as u32, py0 as u32, (px1 - px0) as u32, (py1 - py0) as u32).to_image();
            labels.push(format!("{}:{}:{}", &id[..4.min(id.len())], k, cls));
            crops_img.push(c);
        }
    }
    let crops: Vec<Crop> = crops_img.iter().map(|c| Crop { rgb: c.as_raw(), width: c.width() as usize, height: c.height() as usize }).collect();
    emb.embed(&crops[..1])?; // warm-up (GPU compile)
    let t = std::time::Instant::now();
    let f = emb.embed(&crops)?;
    println!("{} crops embedded in {:?}", f.len(), t.elapsed());
    // Best cross-camera match for each crop.
    for i in 0..f.len() {
        let cam = labels[i].split(':').next().unwrap();
        let mut best: Vec<(f32, usize)> = (0..f.len()).filter(|&j| labels[j].split(':').next().unwrap() != cam).map(|j| (similarity(&f[i], &f[j]), j)).collect();
        best.sort_by(|a, b| b.0.total_cmp(&a.0));
        let top: Vec<String> = best.iter().take(3).map(|(s, j)| format!("{} {:.2}", labels[*j], s)).collect();
        println!("{:<22} → {}", labels[i], top.join("  |  "));
    }
    // Contact sheet: 128-px tiles in label order.
    let n = crops_img.len() as u32;
    let cols = 8u32;
    let mut sheet = image::RgbImage::from_pixel(cols * 132, n.div_ceil(cols) * 132, image::Rgb([20, 20, 20]));
    for (k, c) in crops_img.iter().enumerate() {
        let s = 128.0 / c.width().max(c.height()) as f32;
        let r = image::imageops::resize(c, ((c.width() as f32 * s) as u32).max(1), ((c.height() as f32 * s) as u32).max(1), image::imageops::FilterType::Triangle);
        image::imageops::overlay(&mut sheet, &r, ((k as u32 % cols) * 132 + 2) as i64, ((k as u32 / cols) * 132 + 2) as i64);
    }
    sheet.save(dir.join("sheet.png"))?;
    for (k, l) in labels.iter().enumerate() { print!("{k}={l}  "); }
    println!();
    Ok(())
}
