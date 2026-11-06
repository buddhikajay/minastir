//! Compare detector models on chosen frames of a video, decoded the way the live engine sees
//! them (one 640×640 tile, or two for panoramas). Diagnostic.
//! cargo run --release -p minastir-vision --example detcompare -- <video> <model.onnx>[,<model.onnx>…] <t1> <t2> …
use minastir_vision::detector::{Detector, INPUT};
use std::{path::Path, process::Command};

fn tile(video: &str, t: f64) -> anyhow::Result<Vec<u8>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-ss", &format!("{t}"), "-i", video, "-frames:v", "1", "-vf", &format!("scale={INPUT}:{INPUT}"), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        .output()?;
    anyhow::ensure!(out.stdout.len() == INPUT * INPUT * 3, "no frame at {t}");
    Ok(out.stdout)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let video = &a[0];
    let times: Vec<f64> = a[2..].iter().map(|s| s.parse().unwrap()).collect();
    let frames: Vec<Vec<u8>> = times.iter().map(|&t| tile(video, t)).collect::<anyhow::Result<_>>()?;
    for model in a[1].split(',') {
        let mut det = Detector::load(Path::new(model), true)?;
        det.detect(&[&frames[0]], 0.45)?; // warm-up
        let t0 = std::time::Instant::now();
        println!("== {}", Path::new(model).file_name().unwrap().to_string_lossy());
        for (t, f) in times.iter().zip(&frames) {
            let d = det.detect(&[f], 0.45)?.remove(0);
            let list: Vec<String> = d.iter().map(|d| format!("{} {:.2} ({:.2},{:.2})–({:.2},{:.2})", d.class, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3])).collect();
            println!("  t {t:>5.1}: {}", list.join("  |  "));
        }
        println!("  {:.0} ms per frame", t0.elapsed().as_secs_f64() * 1000.0 / times.len() as f64);
    }
    Ok(())
}
