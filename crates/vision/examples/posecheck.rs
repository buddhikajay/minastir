//! Body keypoints on the people in video frames: draws them (green = confident, red = not), with
//! the feet the pose model finds (blue line) against the bottom of the detection box (yellow).
//! cargo run --release -p minastir-vision --example posecheck -- <models dir> <video> <out.png> <t1> <t2> …
use image::{imageops, Rgb, RgbImage};
use minastir_vision::{
    detector::{Detector, INPUT},
    onnx::Device,
    pose::Pose,
};
use std::{path::Path, process::Command, time::Instant};

fn frame_at(video: &str, t: f64) -> anyhow::Result<RgbImage> {
    let out = Command::new("ffmpeg").args(["-v", "error", "-ss", &format!("{t}"), "-i", video, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]).output()?;
    Ok(image::load_from_memory(&out.stdout)?.to_rgb8())
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let models = Path::new(&a[0]);
    let mut det = Detector::load(&models.join(minastir_vision::live::detector_model().0), true)?;
    let device = if std::env::var("POSE_CPU").is_ok() { Device::Cpu } else { Device::Gpu };
    let mut pose = Pose::load(models, device)?;
    println!("{}", pose.describe());
    let mut tiles = Vec::new();
    let (mut ms, mut n) = (0.0, 0);
    for ts in &a[3..] {
        let t: f64 = ts.parse()?;
        let frame = frame_at(&a[1], t)?;
        let (w, h) = (frame.width() as usize, frame.height() as usize);
        let small = imageops::resize(&frame, INPUT as u32, INPUT as u32, imageops::FilterType::Triangle);
        for d in det.detect(&[small.as_raw()], 0.45)?.remove(0).into_iter().filter(|d| d.class == "person") {
            let started = Instant::now();
            let k = pose.estimate((frame.as_raw(), w, h), d.bbox)?;
            if n > 0 {
                ms += started.elapsed().as_secs_f64() * 1000.0; // first call includes warm-up
            }
            n += 1;
            let b = d.bbox;
            let (x0, y0) = (((b[0] - 0.02).max(0.0) * w as f32) as u32, ((b[1] - 0.02).max(0.0) * h as f32) as u32);
            let (x1, y1) = (((b[2] + 0.02).min(1.0) * w as f32) as u32, ((b[3] + 0.04).min(1.0) * h as f32) as u32);
            let mut crop = imageops::crop_imm(&frame, x0, y0, x1 - x0, y1 - y0).to_image();
            let to = |u: f32, v: f32| ((u * w as f32) as i64 - x0 as i64, (v * h as f32) as i64 - y0 as i64);
            let dot = |img: &mut RgbImage, (x, y): (i64, i64), c: Rgb<u8>, r: i64| {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx >= 0 && yy >= 0 && (xx as u32) < img.width() && (yy as u32) < img.height() {
                            img.put_pixel(xx as u32, yy as u32, c);
                        }
                    }
                }
            };
            for p in k.0 {
                dot(&mut crop, to(p[0], p[1]), if p[2] >= 0.3 { Rgb([0, 255, 0]) } else { Rgb([255, 0, 0]) }, 3);
            }
            let line = |img: &mut RgbImage, v: f32, c: Rgb<u8>| {
                let y = (v * h as f32) as i64 - y0 as i64;
                for x in 0..img.width() {
                    for dy in -1..=1 {
                        let yy = y + dy;
                        if yy >= 0 && (yy as u32) < img.height() {
                            img.put_pixel(x, yy as u32, c);
                        }
                    }
                }
            };
            line(&mut crop, b[3], Rgb([255, 220, 0]));
            let feet = k.feet(0.3);
            if let Some((_, fv)) = feet {
                line(&mut crop, fv, Rgb([0, 120, 255]));
            }
            println!("t {t:>6.1} box h {:.3}  feet {}  ankles {:.2}/{:.2}", b[3] - b[1], feet.map_or("hidden".into(), |f| format!("v {:.3} (box bottom {:.3})", f.1, b[3])), k.0[15][2], k.0[16][2]);
            let s = 220.0 / crop.height() as f32;
            tiles.push(imageops::resize(&crop, ((crop.width() as f32 * s) as u32).max(1), 220, imageops::FilterType::Triangle));
        }
    }
    println!("pose: {:.1} ms per person ({} people)", ms / (n.max(2) - 1) as f64, n);
    let total: u32 = tiles.iter().map(|t| t.width() + 6).sum();
    let width = total.min(2000);
    let rows = (total + width - 1) / width + 1;
    let mut sheet = RgbImage::from_pixel(width, rows * 226, Rgb([0, 0, 0]));
    let (mut x, mut y) = (0u32, 0u32);
    for t in &tiles {
        if x + t.width() > width {
            x = 0;
            y += 226;
        }
        imageops::overlay(&mut sheet, t, x as i64, y as i64);
        x += t.width() + 6;
    }
    sheet.save(&a[2])?;
    Ok(())
}
