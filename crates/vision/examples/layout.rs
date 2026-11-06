//! How well do the cameras agree about the ground?
//!
//! Each calibrated camera's scene map is a metric 3D point plus a semantic label per pixel. Placed
//! at the camera's stored pose, every camera that can see a patch of road should put that road at
//! the same height and call it the same thing. Where two cameras overlap, the gap between their
//! surfaces is the layout error, in metres — the number to drive lens and pose work.
//!
//! Read-only: loads what calibration already wrote and measures it. Nothing is re-solved.
//!
//! cargo run --release -p minastir-vision --example layout -- <data dir>

use minastir_core::onvif::CameraInfo;
use minastir_vision::{geometry::CameraModel, scenemap::LensMap, segment::Label};
use std::{collections::HashMap, path::Path, sync::Arc};

/// Top-down cell size, metres. A kerb is ~0.15 m high, so cells much coarser than this start
/// mixing road and footpath into one height.
const CELL: f64 = 0.25;
/// Ignore points further than this from their own camera: depth noise grows with range.
const MAX_RANGE: f64 = 40.0;
/// A pair needs at least this many shared cells before its numbers mean anything.
const MIN_SHARED: usize = 40;

const LABEL_NAME: [&str; 13] = [
    "other", "road", "sidewalk", "grass", "ground", "building", "tree", "plant", "fence", "vehicle", "sky", "water",
    "pole",
];

/// What one camera says about one top-down cell.
struct Cell {
    height: f64,
    label: Label,
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Collapse a camera's ground points into one height and one label per cell.
fn cells_of(
    model: &CameraModel,
    maps: &[LensMap],
    lenses: &[minastir_core::calib::LensCalibration],
    cam_x: f64,
    cam_z: f64,
) -> HashMap<(i32, i32), Cell> {
    let mut acc: HashMap<(i32, i32), (Vec<f64>, [usize; 13])> = HashMap::new();
    for (li, m) in maps.iter().enumerate() {
        let lens = &lenses[li];
        for y in 0..m.height {
            for x in 0..m.width {
                let label = m.labels[y * m.width + x];
                if !label.is_ground() {
                    continue;
                }
                let (lu, v) = ((x as f64 + 0.5) / m.width as f64, (y as f64 + 0.5) / m.height as f64);
                // point_at takes u across the whole frame; a panorama's lenses each cover a slice.
                let u = lens.u0 + lu * (lens.u1 - lens.u0);
                let Some(p) = model.point_at(u, v) else { continue };
                if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                    continue;
                }
                if (p[0] - cam_x).hypot(p[1] - cam_z) > MAX_RANGE {
                    continue;
                }
                let key = ((p[0] / CELL).floor() as i32, (p[1] / CELL).floor() as i32);
                let e = acc.entry(key).or_insert_with(|| (Vec::new(), [0; 13]));
                e.0.push(p[2]);
                e.1[label as usize] += 1;
            }
        }
    }
    acc.into_iter()
        .map(|(k, (mut hs, counts))| {
            let best = counts.iter().enumerate().max_by_key(|(_, n)| **n).map(|(i, _)| i).unwrap_or(0);
            (k, Cell { height: median(&mut hs), label: Label::from_u8(best as u8) })
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(args.first().map(|s| s.as_str()).unwrap_or("data"));
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;

    let mut named: Vec<(String, HashMap<(i32, i32), Cell>)> = Vec::new();
    for c in &cams {
        let (Some(pose), Some(cal)) = (c.pose, c.calibration.clone()) else { continue };
        let label = c.name.clone().unwrap_or_else(|| c.label());
        let maps: Vec<LensMap> = (0..cal.lenses.len())
            .map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin"))))
            .collect::<Result<_, _>>()?;
        let shared = Arc::new(maps);
        let model = CameraModel::new(pose, &cal, Some(shared.clone()));
        let cells = cells_of(&model, &shared, &cal.lenses, pose.x, pose.z);
        println!(
            "{:<28} {:>6} ground cells ({:.0} m² at {:.2} m), pose ({:+.2}, {:+.2}) yaw {:.1} hfov {:.1}",
            label,
            cells.len(),
            cells.len() as f64 * CELL * CELL,
            CELL,
            pose.x,
            pose.z,
            pose.yaw,
            cal.lenses[0].hfov,
        );
        named.push((label, cells));
    }

    println!("\nPairwise agreement where both cameras see the same ground cell:");
    println!("{:<26} {:<26} {:>7} {:>9} {:>9} {:>8} {:>7}", "camera", "vs", "shared", "med |Δh|", "rms Δh", "p90 |Δh|", "label");
    for i in 0..named.len() {
        for j in (i + 1)..named.len() {
            let (a, b) = (&named[i], &named[j]);
            let mut diffs: Vec<f64> = Vec::new();
            let mut same_label = 0usize;
            for (k, ca) in &a.1 {
                let Some(cb) = b.1.get(k) else { continue };
                diffs.push((ca.height - cb.height).abs());
                if ca.label == cb.label {
                    same_label += 1;
                }
            }
            if diffs.is_empty() {
                println!("{:<26} {:<26} {:>7}  (no overlap)", a.0, b.0, 0);
                continue;
            }
            let n = diffs.len();
            let rms = (diffs.iter().map(|d| d * d).sum::<f64>() / n as f64).sqrt();
            let mut sorted = diffs.clone();
            let med = median(&mut sorted);
            let p90 = sorted[(n as f64 * 0.9) as usize % n];
            let note = if n < MIN_SHARED { " (thin)" } else { "" };
            println!(
                "{:<26} {:<26} {:>7} {:>9.2} {:>9.2} {:>8.2} {:>6.0}%{}",
                a.0,
                b.0,
                n,
                med,
                rms,
                p90,
                100.0 * same_label as f64 / n as f64,
                note
            );
        }
    }

    // What the fused map would look like: cells claimed by more than one camera, and what they
    // disagree about when they disagree.
    let mut owners: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, (_, cells)) in named.iter().enumerate() {
        for k in cells.keys() {
            owners.entry(*k).or_default().push(i);
        }
    }
    let multi = owners.values().filter(|v| v.len() > 1).count();
    println!("\n{} ground cells total, {} seen by more than one camera ({:.0}%)", owners.len(), multi,
        100.0 * multi as f64 / owners.len().max(1) as f64);

    let mut conflicts: HashMap<(&str, &str), usize> = HashMap::new();
    for (k, who) in &owners {
        if who.len() < 2 {
            continue;
        }
        let mut ls: Vec<Label> = who.iter().map(|i| named[*i].1[k].label).collect();
        ls.sort_by_key(|l| *l as u8);
        ls.dedup();
        for w in ls.windows(2) {
            *conflicts.entry((LABEL_NAME[w[0] as usize], LABEL_NAME[w[1] as usize])).or_default() += 1;
        }
    }
    if !conflicts.is_empty() {
        let mut v: Vec<_> = conflicts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        println!("\nMost common label disagreements on shared cells:");
        for ((a, b), n) in v.into_iter().take(8) {
            println!("  {:>5} vs {:<10} {:>6} cells", a, b, n);
        }
    }
    Ok(())
}
