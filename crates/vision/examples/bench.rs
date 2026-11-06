//! cargo run --release -p minastir-vision --example bench -- <model.onnx> <image>...
use minastir_vision::detector::{Detector, INPUT};
use std::{path::Path, time::Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = Path::new(&args[0]);
    let frames: Vec<Vec<u8>> = args[1..]
        .iter()
        .map(|p| {
            let img = image::open(p).unwrap().resize_exact(INPUT as u32, INPUT as u32, image::imageops::FilterType::Triangle);
            img.to_rgb8().into_raw()
        })
        .collect();
    for gpu in std::env::var("MODES").unwrap_or("cpu,gpu".into()).split(",").map(|m| m == "gpu").collect::<Vec<_>>() {
        let t = Instant::now();
        let mut det = Detector::load(model, gpu)?;
        println!("== {} (load {:?})", if gpu { "CoreML" } else { "CPU" }, t.elapsed());
        if !gpu { println!("{}", det.describe()); }
        for (i, f) in frames.iter().enumerate() {
            let mut times = vec![];
            let mut last = vec![];
            for _ in 0..6 {
                let t = Instant::now();
                last = det.detect(&[f], 0.4)?.remove(0);
                times.push(t.elapsed().as_secs_f32() * 1000.0);
            }
            println!("  img {i}: first {:.0} ms, steady {:.1} ms -> {:?}", times[0], times[5],
                last.iter().map(|d| format!("{} {:.2} [{:.2},{:.2},{:.2},{:.2}]", d.class, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3])).collect::<Vec<_>>());
        }
    }
    Ok(())
}
