//! Which "person" detections are really people? Compares each person box in a video frame with
//! the same region of the camera's empty-scene background. Diagnostic.
//! cargo run --release -p minastir-vision --example persons -- <models dir> <video> <background.png> <out dir> <t1> <t2> …
use image::{imageops, Rgb, RgbImage};
use minastir_vision::{
    detector::{Detector, INPUT},
    live::background_match,
};
use std::{path::Path, process::Command};

fn frame_at(video: &str, t: f64) -> anyhow::Result<RgbImage> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-ss", &format!("{t}"), "-i", video, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"])
        .output()?;
    Ok(image::load_from_memory(&out.stdout)?.to_rgb8())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (models, video, bg, out) = (Path::new(&a[0]), &a[1], image::open(&a[2])?.to_rgb8(), Path::new(&a[3]));
    let mut det = Detector::load(&models.join(minastir_vision::live::detector_model().0), true)?;
    let mut tiles = Vec::new();
    for ts in &a[4..] {
        let t: f64 = ts.parse()?;
        let frame = frame_at(video, t)?;
        let small = imageops::resize(&frame, INPUT as u32, INPUT as u32, imageops::FilterType::Triangle);
        let (w, h) = (frame.width() as f64, frame.height() as f64);
        for d in det.detect(&[small.as_raw()], 0.45)?.remove(0) {
            if d.class != "person" {
                continue;
            }
            let b = d.bbox.map(|v| v as f64);
            let m = background_match(&frame, &bg, b);
            println!("t {t:>5.1}  person {:.2}  box ({:.3},{:.3})–({:.3},{:.3})  matches empty scene {:.2}", d.score, b[0], b[1], b[2], b[3], m);
            let (x0, y0, x1, y1) = ((b[0] * w) as u32, (b[1] * h) as u32, (b[2] * w) as u32, (b[3] * h) as u32);
            let (cw, ch) = ((x1 - x0).max(1), (y1 - y0).max(1));
            let s = 160.0 / ch as f64;
            let fit = |img: &RgbImage| {
                let (bx, by) = ((b[0] * img.width() as f64) as u32, (b[1] * img.height() as f64) as u32);
                let (bw, bh) = (((b[2] - b[0]) * img.width() as f64) as u32, ((b[3] - b[1]) * img.height() as f64) as u32);
                let c = imageops::crop_imm(img, bx, by, bw.max(1), bh.max(1)).to_image();
                imageops::resize(&c, ((cw as f64 * s) as u32).max(1), 160, imageops::FilterType::Triangle)
            };
            let (f, g) = (fit(&frame), fit(&bg));
            let mut pair = RgbImage::from_pixel(f.width() * 2 + 4, 160, Rgb([20, 20, 20]));
            imageops::overlay(&mut pair, &f, 0, 0);
            imageops::overlay(&mut pair, &g, (f.width() + 4) as i64, 0);
            tiles.push(pair);
        }
    }
    let w: u32 = tiles.iter().map(|t| t.width() + 8).sum::<u32>().min(2400);
    let rows = {
        let (mut rows, mut x) = (1u32, 0u32);
        for t in &tiles {
            if x + t.width() > w {
                rows += 1;
                x = 0;
            }
            x += t.width() + 8;
        }
        rows
    };
    let mut sheet = RgbImage::from_pixel(w, rows * 168, Rgb([0, 0, 0]));
    let (mut x, mut y) = (0u32, 0u32);
    for t in &tiles {
        if x + t.width() > w {
            x = 0;
            y += 168;
        }
        imageops::overlay(&mut sheet, t, x as i64, y as i64);
        x += t.width() + 8;
    }
    sheet.save(out.join("persons.png"))?;
    Ok(())
}
