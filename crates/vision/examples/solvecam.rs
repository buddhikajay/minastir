//! Re-solve one camera's orientation and field of view from people another camera sees at the same
//! moments. Diagnostic (prints the result; changes nothing).
//! cargo run --release -p minastir-vision --example solvecam -- <data dir> <snapshots.jsonl> <camera to solve> <reference camera>
//! The JSONL is a log of live snapshots with per-camera detection boxes and ground sightings.
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::geometry::CameraModel;
use std::path::Path;

/// One moment: the solved camera's people (feet pixel u, v) and the reference camera's people
/// (ground x, z).
struct Moment {
    feet: Vec<(f64, f64)>,
    refs: Vec<(f64, f64)>,
}

#[derive(Clone, Copy, Debug)]
struct Params {
    pitch: f64,
    roll: f64,
    yaw: f64,
    /// Field-of-view factor on tan(half-angle).
    fov: f64,
    /// Shift of the mounting position, metres.
    dx: f64,
    dz: f64,
}

const CAP: f64 = 3.0;

fn model(base: &CameraInfo, p: Params) -> CameraModel {
    let mut cal = base.calibration.clone().unwrap();
    let b = base.pose.unwrap();
    let pose = Pose { pitch: p.pitch, yaw: p.yaw, x: b.x + p.dx, z: b.z + p.dz, ..b };
    cal.roll = p.roll;
    cal.pitch = p.pitch;
    for l in &mut cal.lenses {
        l.hfov = 2.0 * (p.fov * (l.hfov.to_radians() / 2.0).tan()).atan().to_degrees();
        l.vfov = 2.0 * (p.fov * (l.vfov.to_radians() / 2.0).tan()).atan().to_degrees();
        l.pitch = p.pitch;
        l.roll = p.roll;
    }
    CameraModel::new(pose, &cal, None)
}

/// Robust disagreement: each foot's distance to the nearest reference person at that moment,
/// capped (people only one camera sees cost the same whatever the parameters). Returns (cost,
/// feet matched within 1 m, median matched distance).
fn score(base: &CameraInfo, p: Params, moments: &[Moment]) -> (f64, usize, f64) {
    let m = model(base, p);
    let (mut cost, mut ds) = (0.0, Vec::new());
    for mo in moments {
        for &(u, v) in &mo.feet {
            let d = m
                .ground_point(u, v, 60.0)
                .map(|(x, z)| mo.refs.iter().map(|r| (r.0 - x).hypot(r.1 - z)).fold(f64::MAX, f64::min))
                .unwrap_or(CAP)
                .min(CAP);
            cost += d * d;
            if d < CAP {
                ds.push(d);
            }
        }
    }
    let within = ds.iter().filter(|d| **d < 1.0).count();
    ds.sort_by(f64::total_cmp);
    (cost, within, ds.get(ds.len() / 2).copied().unwrap_or(f64::NAN))
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, log, solve, reference) = (Path::new(&a[0]), &a[1], &a[2], &a[3]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let base = cams.iter().find(|c| &c.id() == solve).expect("camera to solve").clone();
    let mut moments = Vec::new();
    let mut last_feet: Vec<(f64, f64)> = Vec::new();
    for line in std::fs::read_to_string(log)?.lines() {
        let s: serde_json::Value = serde_json::from_str(line)?;
        let feet: Vec<(f64, f64)> = s["cams"][solve.as_str()]["dets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|d| d["class"] == "person")
            .filter_map(|d| {
                let b: Vec<f64> = d["bbox"].as_array()?.iter().filter_map(|x| x.as_f64()).collect();
                (b[3] < 0.98).then(|| ((b[0] + b[2]) / 2.0, b[3]))
            })
            .collect();
        let refs: Vec<(f64, f64)> = s["cams"][reference.as_str()]["ground"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|g| g[0] == "person")
            .filter_map(|g| Some((g[1].as_f64()?, g[2].as_f64()?)))
            .collect();
        // Snapshots repeat between frames: keep each distinct set of boxes once.
        if feet.is_empty() || refs.is_empty() || feet == last_feet {
            continue;
        }
        last_feet = feet.clone();
        moments.push(Moment { feet, refs });
    }
    let cal = base.calibration.as_ref().unwrap();
    let pose = base.pose.unwrap();
    let now = Params { pitch: pose.pitch, roll: cal.roll, yaw: pose.yaw, fov: 1.0, dx: 0.0, dz: 0.0 };
    let n_feet: usize = moments.iter().map(|m| m.feet.len()).sum();
    println!("{} moments, {} feet", moments.len(), n_feet);
    let report = |label: &str, p: Params| {
        let (c, within, med) = score(&base, p, &moments);
        println!("{label}: pitch {:.1}° roll {:.1}° yaw {:.1}° fov ×{:.2} shift ({:.1},{:.1})  cost {:.0}  feet within 1 m {} ({:.0}%)  median {:.2} m", p.pitch, p.roll, p.yaw, p.fov, p.dx, p.dz, c, within, 100.0 * within as f64 / n_feet as f64, med);
    };
    report("current ", now);

    // Coarse grid, then shrink around the best.
    let mut best = (f64::MAX, now);
    for pi in 0..=15 {
        for ri in 0..=11 {
            for fi in 0..=12 {
                for yi in 0..=16 {
                    let p = Params { pitch: 10.0 + 2.0 * pi as f64, roll: -20.0 + 2.0 * ri as f64, fov: 0.8 + 0.07 * fi as f64, yaw: now.yaw - 20.0 + 2.5 * yi as f64, dx: 0.0, dz: 0.0 };
                    let c = score(&base, p, &moments).0;
                    if c < best.0 {
                        best = (c, p);
                    }
                }
            }
        }
    }
    report("coarse  ", best.1);
    // Coordinate descent over all six, with shrinking steps (position shifts up to ±2 m).
    // FREE=angles solves only tilt, lean and heading (lens and mounting position kept).
    let angles_only = std::env::var("FREE").as_deref() == Ok("angles");
    if angles_only {
        best = (f64::MAX, now);
        for pi in 0..=30 {
            for ri in 0..=15 {
                for yi in 0..=24 {
                    let p = Params { pitch: 10.0 + pi as f64, roll: -20.0 + 2.0 * ri as f64, yaw: now.yaw - 30.0 + 2.5 * yi as f64, ..now };
                    let c = score(&base, p, &moments).0;
                    if c < best.0 {
                        best = (c, p);
                    }
                }
            }
        }
        report("coarse (angles)", best.1);
    }
    let mut step = if angles_only { [1.0, 1.0, 0.0, 1.5, 0.0, 0.0] } else { [1.0, 1.0, 0.04, 1.5, 0.5, 0.5] };
    for _ in 0..8 {
        loop {
            let mut improved = false;
            for k in 0..6 {
                for sgn in [-1.0, 1.0] {
                    let mut p = best.1;
                    let d = sgn * step[k];
                    match k {
                        0 => p.pitch += d,
                        1 => p.roll += d,
                        2 => p.fov += d,
                        3 => p.yaw += d,
                        4 => p.dx = (p.dx + d).clamp(-2.0, 2.0),
                        _ => p.dz = (p.dz + d).clamp(-2.0, 2.0),
                    }
                    let c = score(&base, p, &moments).0;
                    if c < best.0 - 1e-9 {
                        best = (c, p);
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        step.iter_mut().for_each(|s| *s /= 2.0);
    }
    report("solved  ", best.1);
    Ok(())
}
