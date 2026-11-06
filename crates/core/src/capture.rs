//! Frame capture via ffmpeg (handles every RTSP/codec quirk cameras throw at us).
//! Besides raw frames, produces a temporal-median "background" image per camera, which
//! removes people/cars passing through — the static scene is what we reconstruct.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct CaptureOpts {
    pub frames: u32,
    /// Frames per second sampled from the stream.
    pub fps: f32,
}

pub struct CaptureResult {
    pub frames_dir: PathBuf,
    pub background: PathBuf,
}

pub fn capture(source: &str, out_dir: &Path, opts: &CaptureOpts) -> Result<CaptureResult> {
    let frames_dir = out_dir.join("frames");
    std::fs::create_dir_all(&frames_dir)?;
    for e in std::fs::read_dir(&frames_dir)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "jpg") {
            std::fs::remove_file(p)?;
        }
    }

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-y"]);
    if source.starts_with("rtsp") {
        // TCP avoids smeared frames from UDP packet loss; timeout is in microseconds.
        cmd.args(["-rtsp_transport", "tcp", "-timeout", "10000000"]);
    }
    cmd.args(["-i", source])
        .args(["-vf", &format!("fps={}", opts.fps)])
        .args(["-frames:v", &opts.frames.to_string(), "-q:v", "2"])
        .arg(frames_dir.join("frame_%03d.jpg"));
    run(cmd).context("capturing frames")?;

    let n = std::fs::read_dir(&frames_dir)?
        .filter(|e| e.as_ref().is_ok_and(|e| e.path().extension().is_some_and(|x| x == "jpg")))
        .count() as u32;
    if n == 0 {
        bail!("ffmpeg produced no frames from {}", redact(source));
    }

    // Temporal median over all captured frames. tmedian only emits frames whose window is full,
    // so with radius = (n-1)/2 its first output is the median of the whole capture.
    let radius = ((n - 1) / 2).clamp(1, 127);
    let background = out_dir.join("background.png");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-framerate", "1", "-i"])
        .arg(frames_dir.join("frame_%03d.jpg"));
    if n >= 3 {
        cmd.args(["-vf", &format!("tmedian=radius={radius}")]);
    }
    cmd.args(["-fps_mode", "passthrough", "-frames:v", "1", "-update", "1"]).arg(&background);
    run(cmd).context("computing median background")?;
    if !background.exists() {
        bail!("ffmpeg did not write {}", background.display());
    }

    Ok(CaptureResult { frames_dir, background })
}

/// Grab a single JPEG frame (scaled down to `max_width`) — used for live tiles in the UI.
pub fn snapshot(source: &str, max_width: u32) -> Result<Vec<u8>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error"]);
    if source.starts_with("rtsp") {
        cmd.args(["-rtsp_transport", "tcp", "-timeout", "5000000"]);
    }
    cmd.args(["-i", source, "-frames:v", "1", "-an"])
        .args(["-vf", &format!("scale='min({max_width},iw)':-2")])
        .args(["-q:v", "4", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]);
    let out = cmd.output().context("failed to run ffmpeg — is it installed and on PATH?")?;
    if !out.status.success() || out.stdout.is_empty() {
        bail!("snapshot failed: {}", redact(String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(out.stdout)
}

fn run(mut cmd: Command) -> Result<()> {
    let out = cmd.output().context("failed to run ffmpeg — is it installed and on PATH?")?;
    if !out.status.success() {
        bail!("ffmpeg failed: {}", redact(String::from_utf8_lossy(&out.stderr).trim()));
    }
    Ok(())
}

/// Hide passwords embedded in URLs before printing.
pub fn redact(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("://") {
        let (head, tail) = rest.split_at(i + 3);
        out.push_str(head);
        let end = tail.find(|c: char| c == '/' || c.is_whitespace()).unwrap_or(tail.len());
        match tail[..end].rfind('@') {
            Some(at) => {
                let user = tail[..at].split(':').next().unwrap_or("");
                out.push_str(&format!("{user}:***@{}", &tail[at + 1..end]));
            }
            None => out.push_str(&tail[..end]),
        }
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}
