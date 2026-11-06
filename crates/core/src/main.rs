use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use minastir_core::onvif::{self, CameraInfo};
use minastir_core::{capture, discovery};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "minastir", about = "Plug-and-play 3D reconstruction from ONVIF security cameras")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Creds {
    /// ONVIF/RTSP username
    #[arg(short, long, env = "MINASTIR_USER", default_value = "admin")]
    user: String,
    /// ONVIF/RTSP password
    #[arg(short, long, env = "MINASTIR_PASS", hide_env_values = true, default_value = "")]
    pass: String,
}

#[derive(Subcommand)]
enum Cmd {
    /// List ONVIF cameras on the local network (no credentials needed)
    Discover {
        #[arg(long, default_value_t = 4)]
        timeout_secs: u64,
    },
    /// Discover + query every camera (or only --host ones) and write a camera inventory
    Scan {
        #[command(flatten)]
        creds: Creds,
        /// Camera IP(s) to query directly, skipping multicast discovery
        #[arg(long = "host")]
        hosts: Vec<String>,
        /// ONVIF HTTP port when using --host
        #[arg(long, default_value_t = 80)]
        port: u16,
        #[arg(long, default_value = "data/cameras.json")]
        out: PathBuf,
    },
    /// Grab frames + a people-free background image from each camera in the inventory
    Capture {
        #[command(flatten)]
        creds: Creds,
        #[arg(long, default_value = "data/cameras.json")]
        inventory: PathBuf,
        /// Capture from an arbitrary source (RTSP URL or video file) instead of the inventory
        #[arg(long)]
        source: Option<String>,
        #[arg(long, default_value = "data")]
        out: PathBuf,
        #[arg(long, default_value_t = 15)]
        frames: u32,
        #[arg(long, default_value_t = 1.0)]
        fps: f32,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Discover { timeout_secs } => {
            let found = discovery::discover(Duration::from_secs(timeout_secs))?;
            for d in &found {
                println!(
                    "{:15}  {:<20} {:<20} {}",
                    d.ip,
                    d.scope_value("name").unwrap_or_default(),
                    d.scope_value("hardware").unwrap_or_default(),
                    d.device_url().unwrap_or("?")
                );
            }
            println!("{} ONVIF device(s) found", found.len());
            if found.is_empty() {
                eprintln!("hint: discovery is often disabled on cameras or blocked across VLANs; try `scan --host <ip>`");
            }
        }

        Cmd::Scan { creds, hosts, port, out } => {
            let targets: Vec<(String, String)> = if hosts.is_empty() {
                discovery::discover(Duration::from_secs(4))?
                    .into_iter()
                    .filter_map(|d| d.device_url().map(|u| (d.ip.clone(), u.to_string())))
                    .collect()
            } else {
                hosts
                    .iter()
                    .map(|h| {
                        let addr = if h.contains(':') { h.clone() } else { format!("{h}:{port}") };
                        let ip = h.split(':').next().unwrap_or(h).to_string();
                        (ip, format!("http://{addr}/onvif/device_service"))
                    })
                    .collect()
            };
            if targets.is_empty() {
                bail!("no cameras found; pass --host <ip> to query cameras directly");
            }

            let mut client = onvif::Client::new(&creds.user, &creds.pass)?;
            let mut cameras = Vec::new();
            for (ip, url) in targets {
                match client.probe(&ip, &url) {
                    Ok(cam) => {
                        print_camera(&cam);
                        cameras.push(cam);
                    }
                    Err(e) => eprintln!("✗ {ip}: {e:#}"),
                }
            }
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(&out, serde_json::to_string_pretty(&cameras)?)?;
            println!("\nwrote {} camera(s) to {}", cameras.len(), out.display());
        }

        Cmd::Capture { creds, inventory, source, out, frames, fps } => {
            let opts = capture::CaptureOpts { frames, fps };
            let jobs: Vec<(String, String)> = match source {
                Some(src) => vec![("manual".into(), src)],
                None => {
                    let text = std::fs::read_to_string(&inventory)
                        .with_context(|| format!("reading {} (run `scan` first)", inventory.display()))?;
                    let cams: Vec<CameraInfo> = serde_json::from_str(&text)?;
                    cams.iter()
                        .filter_map(|c| {
                            let uri = c.best_profile()?.stream_uri.as_deref()?;
                            Some((c.id(), onvif::with_credentials(uri, &creds.user, &creds.pass)))
                        })
                        .collect()
                }
            };
            for (id, src) in jobs {
                println!("capturing {id} from {} ({frames} frames @ {fps} fps)…", capture::redact(&src));
                match capture::capture(&src, &out.join(&id), &opts) {
                    Ok(r) => println!("  ✓ frames: {}  background: {}", r.frames_dir.display(), r.background.display()),
                    Err(e) => eprintln!("  ✗ {e:#}"),
                }
            }
        }
    }
    Ok(())
}

fn print_camera(c: &CameraInfo) {
    println!(
        "✓ {} — {} {} (fw {}, s/n {}){}",
        c.ip,
        c.manufacturer.as_deref().unwrap_or("?"),
        c.model.as_deref().unwrap_or("?"),
        c.firmware.as_deref().unwrap_or("?"),
        c.serial.as_deref().unwrap_or("?"),
        if c.ptz { "  [PTZ]" } else { "" }
    );
    for p in &c.profiles {
        println!(
            "    {:<16} {:>5}x{:<5} {:<6} {}",
            p.name,
            p.width.unwrap_or(0),
            p.height.unwrap_or(0),
            p.encoding.as_deref().unwrap_or(""),
            p.stream_uri.as_deref().unwrap_or("(no stream uri)")
        );
    }
}
