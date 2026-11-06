//! Ask the scene describer about each detected object in a set of snapshots. Diagnostic.
//! cargo run --release -p minastir-vision --example describe -- <models dir> <dir with {id}.jpg + boxes.json>
use minastir_vision::{describe::Describer, onnx::Device};
use std::{collections::BTreeMap, path::Path};

const KINDS: &[&str] = &["car", "truck", "bin", "other"];
const KIND_Q: &str = "What is the main object in this picture? Answer with one word: car, truck, bin, or other.";
const FACING: &[&str] = &["left", "right", "toward", "away"];
const FACING_Q: &str = "Which way is the front of this vehicle facing in the picture? Answer with one word: left, right, toward (the camera), or away (from the camera).";

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (models, dir) = (Path::new(&a[0]), Path::new(&a[1]));
    let boxes: BTreeMap<String, Vec<(String, f64, f64, f64, f64)>> = serde_json::from_slice(&std::fs::read(dir.join("boxes.json"))?)?;
    let t = std::time::Instant::now();
    let mut d = Describer::load(models, Device::Cpu)?;
    println!("loaded in {:?}", t.elapsed());
    let mut k = 0;
    for (id, bs) in &boxes {
        if bs.is_empty() { continue; }
        let img = image::open(dir.join(format!("{id}.jpg")))?.to_rgb8();
        let (w, h) = (img.width() as f64, img.height() as f64);
        for (j, (cls, x0, y0, x1, y1)) in bs.iter().enumerate() {
            let (px0, py0, px1, py1) = ((x0 * w).max(0.0), (y0 * h).max(0.0), (x1 * w).min(w), (y1 * h).min(h));
            if px1 - px0 < 24.0 || py1 - py0 < 24.0 { continue; }
            let c = image::imageops::crop_imm(&img, px0 as u32, py0 as u32, (px1 - px0) as u32, (py1 - py0) as u32).to_image();
            let (cw, ch) = (c.width() as usize, c.height() as usize);
            let t = std::time::Instant::now();
            let kind = d.choose(c.as_raw(), cw, ch, KIND_Q, KINDS)?;
            let facing = d.choose(c.as_raw(), cw, ch, FACING_Q, FACING)?;
            let said = if k < 3 { d.answer(c.as_raw(), cw, ch, "Describe this object in one short sentence.", 30)? } else { String::new() };
            let fmt = |names: &[&str], p: &[f32]| names.iter().zip(p).map(|(n, p)| format!("{n} {:.2}", p)).collect::<Vec<_>>().join(" ");
            println!("{k:>2}={}:{j}:{cls:<6} [{}]  facing [{}]  {:?}  {said}", &id[..4], fmt(KINDS, &kind), fmt(FACING, &facing), t.elapsed());
            k += 1;
        }
    }
    Ok(())
}
