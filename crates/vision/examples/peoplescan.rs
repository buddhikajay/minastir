//! When are there people in a video? Decodes it at a low frame rate and runs the detector on
//! every frame; prints the stretches with people in them. For picking footage worth harvesting.
//! cargo run --release -p minastir-vision --example peoplescan -- <models dir> <video> [fps]
use minastir_vision::detector::{Detector, INPUT};
use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (models, video) = (Path::new(&a[0]), &a[1]);
    let fps: f64 = a.get(2).map_or(Ok(1.0), |s| s.parse())?;
    let mut det = Detector::load(&models.join(minastir_vision::live::detector_model().0), true)?;
    let side = INPUT as usize;
    let mut ff = Command::new("ffmpeg")
        .args(["-v", "error", "-i", video, "-vf", &format!("fps={fps},scale={}:{side}", 2 * side), "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        .stdout(Stdio::piped())
        .spawn()?;
    let mut out = ff.stdout.take().unwrap();
    // Two square tiles side by side, as the live engine sees a 16:9 frame.
    let mut buf = vec![0u8; 2 * side * side * 3];
    let (mut left, mut right) = (vec![0u8; side * side * 3], vec![0u8; side * side * 3]);
    let (mut k, mut run_start, mut last_seen, mut most) = (0usize, None::<f64>, 0.0f64, 0usize);
    let mut total = 0.0;
    let mut stretches = 0;
    while out.read_exact(&mut buf).is_ok() {
        let t = k as f64 / fps;
        k += 1;
        for y in 0..side {
            let row = &buf[y * 2 * side * 3..(y + 1) * 2 * side * 3];
            left[y * side * 3..(y + 1) * side * 3].copy_from_slice(&row[..side * 3]);
            right[y * side * 3..(y + 1) * side * 3].copy_from_slice(&row[side * 3..]);
        }
        let mut n = 0;
        for tile in [&left, &right] {
            n += det.detect(&[tile.as_slice()], 0.5)?.remove(0).iter().filter(|d| d.class == "person").count();
        }
        if n > 0 {
            if run_start.is_none() {
                run_start = Some(t);
                most = 0;
            }
            most = most.max(n);
            last_seen = t;
        } else if let Some(s) = run_start {
            if t - last_seen > 5.0 {
                let fmt = |x: f64| format!("{:02}:{:02}:{:02}", (x / 3600.0) as u32, ((x % 3600.0) / 60.0) as u32, (x % 60.0) as u32);
                println!("{} – {}  ({:.0} s, up to {most} people)", fmt(s), fmt(last_seen), last_seen - s + 1.0 / fps);
                total += last_seen - s + 1.0 / fps;
                stretches += 1;
                run_start = None;
            }
        }
    }
    if let Some(s) = run_start {
        total += last_seen - s + 1.0 / fps;
        stretches += 1;
        println!("{s:.0} s – end");
    }
    println!("{stretches} stretches with people, {total:.0} s in all, over {:.0} s of video", k as f64 / fps);
    let _ = ff.wait();
    Ok(())
}
