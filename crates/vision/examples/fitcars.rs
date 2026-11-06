//! Render-and-compare on real snapshots: fit a 3D car to every camera's outline of it. Diagnostic.
//! cargo run --release -p minastir-vision --example fitcars -- <data dir> <models dir> <snaps dir>
//! The snaps dir holds {camera id}.jpg (sharp) and boxes.json ({id: [[class, x0, y0, x1, y1], …]},
//! normalised boxes). Writes car{k}.png overlays there: blue = the real outline (SAM), red = the
//! fitted 3D car drawn through each camera's lens.
use image::{Rgb, RgbImage};
use minastir_core::onvif::CameraInfo;
use minastir_vision::{
    fit::{fit, shape_for, Pose2, View},
    geometry::CameraModel,
    roof::{BoxMask, Sam},
    scenemap::LensMap,
};
use std::{collections::BTreeMap, path::Path, sync::Arc};

struct Cam {
    id: String,
    model: CameraModel,
    img: RgbImage,
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, models, snaps) = (Path::new(&a[0]), Path::new(&a[1]), Path::new(&a[2]));
    let infos: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let boxes: BTreeMap<String, Vec<(String, f64, f64, f64, f64)>> = serde_json::from_slice(&std::fs::read(snaps.join("boxes.json"))?)?;
    let mut cams = Vec::new();
    for c in &infos {
        let (Some(cal), Some(pose)) = (c.calibration.clone(), c.pose) else { continue };
        let Ok(img) = image::open(snaps.join(format!("{}.jpg", c.id()))) else { continue };
        let maps: Option<Vec<LensMap>> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin"))).ok()).collect();
        cams.push(Cam { id: c.id(), model: CameraModel::new(pose, &cal, maps.map(Arc::new)), img: img.to_rgb8() });
    }

    if std::env::var("ROUNDTRIP").is_ok() {
        for c in &cams {
            for (u, v) in [(0.2, 0.8), (0.5, 0.7), (0.8, 0.9), (0.3, 0.6)] {
                let g = c.model.ground_point(u, v, 60.0);
                let back = g.and_then(|(x, z)| c.model.project(x, z, 0.0));
                println!("{} ({u},{v}) → ground {:?} → image {:?}", &c.id[..4], g, back);
            }
        }
        return Ok(());
    }
    // Each box's rough ground position (bottom centre, pushed back half a car along the ray).
    struct Det {
        cam: usize,
        class: String,
        b: [f64; 4],
        at: (f64, f64),
        /// Ground height under the box bottom, in the camera's frame (from its scene map).
        base: f64,
    }
    let mut dets = Vec::new();
    for (ci, c) in cams.iter().enumerate() {
        for (class, x0, y0, x1, y1) in boxes.get(&c.id).into_iter().flatten() {
            let u = (x0 + x1) / 2.0;
            let Some((x, z)) = c.model.ground_point(u, *y1, 45.0) else { continue };
            let base = c.model.point_at(u, *y1).map_or(0.0, |p| p[2]);
            let (dx, dz) = (x - c.model.pose.x, z - c.model.pose.z);
            let r = dx.hypot(dz).max(1e-6);
            dets.push(Det { cam: ci, class: class.clone(), b: [*x0, *y0, *x1, *y1], at: (x + dx / r, z + dz / r), base });
        }
    }
    // 1. Each sighting on its own: outline it, fit a car to that one view.
    let mut sam = Sam::load(models)?;
    let mut wheel_finder = if std::env::var("WHEELS").is_ok() { Some(minastir_vision::wheels::Wheels::load(models, minastir_vision::onnx::Device::Cpu)?) } else { None };
    let mut masks: Vec<Option<BoxMask>> = Vec::new();
    let mut solo: Vec<Option<minastir_vision::fit::Fit>> = Vec::new();
    for (di, d) in dets.iter().enumerate() {
        let c = &cams[d.cam];
        let (w, h) = (c.img.width() as f64, c.img.height() as f64);
        let bpx = [d.b[0] * w, d.b[1] * h, d.b[2] * w, d.b[3] * h];
        if bpx[2] - bpx[0] < 30.0 || bpx[3] - bpx[1] < 20.0 {
            masks.push(None);
            solo.push(None);
            continue;
        }
        let m = sam.box_mask(&c.img, d.cam as u64 * 1000 + di as u64, bpx)?;
        let v = View::new(&c.model, (w, h), &m).on_ground(d.base);
        let f = fit(&[v], &shape_for(&d.class), d.at);
        masks.push(Some(m));
        solo.push(Some(f));
    }
    // 2. Group sightings whose own fits land within 3 m (one per camera): the same car.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let at = |di: usize| solo[di].as_ref().map(|f| (f.pose.x, f.pose.z));
    for di in 0..dets.len() {
        let Some(p) = at(di) else { continue };
        let best = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.iter().all(|&o| dets[o].cam != dets[di].cam))
            .map(|(gi, g)| (gi, g.iter().filter_map(|&o| at(o)).map(|q| (q.0 - p.0).hypot(q.1 - p.1)).fold(f64::MAX, f64::min)))
            .filter(|(_, d)| *d < 3.0)
            .min_by(|x, y| x.1.total_cmp(&y.1));
        match best {
            Some((gi, _)) => groups[gi].push(di),
            None => groups.push(vec![di]),
        }
    }
    for (k, g) in groups.iter().enumerate() {
        let class = if g.iter().any(|&d| dets[d].class == "truck") { "truck" } else { "car" };
        let shape = shape_for(class);
        let masks: Vec<(usize, BoxMask)> = g.iter().filter_map(|&di| masks[di].clone().map(|m| (di, m))).collect();
        if masks.is_empty() {
            continue;
        }
        let views: Vec<View> = masks
            .iter()
            .map(|(di, m)| {
                let c = &cams[dets[*di].cam];
                View::new(&c.model, (c.img.width() as f64, c.img.height() as f64), m).on_ground(dets[*di].base)
            })
            .collect();
        let fits: Vec<(f64, f64)> = g.iter().filter_map(|&d| at(d)).collect();
        let start = (fits.iter().map(|p| p.0).sum::<f64>() / fits.len() as f64, fits.iter().map(|p| p.1).sum::<f64>() / fits.len() as f64);
        if std::env::var("WHY").is_ok() {
            for ((di, m), v) in masks.iter().zip(&views) {
                let d = &dets[*di];
                let p = Pose2 { x: start.0, z: start.1, heading: 0.0 };
                let o = v.outline(p, &shape);
                let corners: Vec<Option<(f64, f64)>> = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)].iter().map(|(dx, dz)| v.model.project(start.0 + dx, start.1 + dz, 0.0)).collect();
                println!("  car{k} {} box {:?} mask cover {:.2} win ({:.0},{:.0} {}x{}) outline {} corners {:?}", &cams[d.cam].id[..4], d.b.map(|x| (x * 100.0).round() / 100.0), v.coverage(), m.x0, m.y0, m.width, m.height, o.is_some(), corners);
            }
        }
        if std::env::var("PROFILE").is_ok() {
            for (vi, v) in views.iter().enumerate() {
                let name = &cams[dets[masks[vi].0].cam].id[..4];
                let prof = minastir_vision::fit::heading_profile(std::slice::from_ref(v), &shape, start, 6.0, 10.0);
                let row: Vec<String> = prof.iter().filter(|(h, _, _)| *h < 180.0).map(|(h, s, p)| format!("{h:.0}°:{s:.2}@({:.1},{:.1})", p.x, p.z)).collect();
                println!("  car{k} {name} profile: {}", row.join(" "));
            }
        }
        // With WHEELS=1: the wheels' axis steers the heading, as in the live engine.
        let mut hint = None;
        if let Some(w) = wheel_finder.as_mut() {
            let mut axes = Vec::new();
            for (di, _) in &masks {
                let d = &dets[*di];
                let c = &cams[d.cam];
                let bb = [d.b[0] as f32, d.b[1] as f32, d.b[2] as f32, d.b[3] as f32];
                let found = w.detect((c.img.as_raw(), c.img.width() as usize, c.img.height() as usize), bb)?;
                let ax = minastir_vision::wheels::axis(&c.model, &found, bb, shape.length, c.img.width() as f64 / c.img.height() as f64);
                println!("  car{k} {} wheels {:?} → axis {:?}", &c.id[..4], found.iter().map(|w| (w.score * 100.0).round() / 100.0).collect::<Vec<_>>(), ax);
                axes.extend(ax);
            }
            hint = minastir_vision::wheels::combine(&axes).map(|a| (a, 8.0));
        }
        let t = std::time::Instant::now();
        let f = minastir_vision::fit::fit_guided(&views, &shape, start, 4.0, hint);
        let names: Vec<String> = masks.iter().map(|(di, _)| cams[dets[*di].cam].id[..4].to_string()).collect();
        println!(
            "car{k:<2} {class:<5} views {:<16} heading {:>5.1}°{}  at ({:>6.1},{:>6.1})  moved {:.1} m  overlap {:.2} {:?}  turned 90°: {:.2}  {:?}",
            names.join("+"),
            f.pose.heading,
            if f.axis_only { " (axis)" } else { "" },
            f.pose.x,
            f.pose.z,
            (f.pose.x - start.0).hypot(f.pose.z - start.1),
            f.score,
            f.per_view.iter().map(|s| (s * 100.0).round() / 100.0).collect::<Vec<_>>(),
            f.across,
            t.elapsed()
        );
        if g.len() > 1 {
            for &di in g {
                if let Some(sf) = &solo[di] {
                    println!("      alone in {}: ({:.1},{:.1}) heading {:.0}° overlap {:.2}", &cams[dets[di].cam].id[..4], sf.pose.x, sf.pose.z, sf.pose.heading, sf.score);
                }
            }
        }
        save_overlay(&snaps.join(format!("car{k}.png")), &masks, &views, &cams, &dets.iter().map(|d| d.cam).collect::<Vec<_>>(), f.pose, &shape)?;
    }
    Ok(())
}

/// Each view's window side by side: the real outline tinted blue, the fitted car's outline red.
fn save_overlay(path: &Path, masks: &[(usize, BoxMask)], views: &[View], cams: &[Cam], cam_of: &[usize], pose: Pose2, shape: &minastir_vision::fit::Shape) -> anyhow::Result<()> {
    let tiles: Vec<RgbImage> = masks
        .iter()
        .zip(views)
        .map(|((di, m), v)| {
            let c = &cams[cam_of[*di]];
            let mut t = image::imageops::crop_imm(&c.img, m.x0 as u32, m.y0 as u32, m.width as u32, m.height as u32).to_image();
            for y in 0..t.height().min(m.height as u32) {
                for x in 0..t.width().min(m.width as u32) {
                    if m.cells[y as usize * m.width + x as usize] {
                        let p = t.get_pixel_mut(x, y);
                        p.0 = [p.0[0] / 2, p.0[1] / 2, ((p.0[2] as u16 + 255) / 2) as u8];
                    }
                }
            }
            if let Some(polys) = v.outline(pose, shape) {
                for poly in polys {
                    for i in 0..poly.len() {
                        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
                        let n = ((q[0] - p[0]).hypot(q[1] - p[1]) as usize).max(1);
                        for s in 0..=n {
                            let f = s as f64 / n as f64;
                            let (x, y) = (p[0] + (q[0] - p[0]) * f - m.x0, p[1] + (q[1] - p[1]) * f - m.y0);
                            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                                let (xi, yi) = (x as i64 + dx, y as i64 + dy);
                                if xi >= 0 && yi >= 0 && (xi as u32) < t.width() && (yi as u32) < t.height() {
                                    t.put_pixel(xi as u32, yi as u32, Rgb([255, 40, 40]));
                                }
                            }
                        }
                    }
                }
            }
            // Common height for the strip.
            let s = 360.0 / t.height() as f64;
            image::imageops::resize(&t, ((t.width() as f64 * s) as u32).max(1), 360, image::imageops::FilterType::Triangle)
        })
        .collect();
    let w: u32 = tiles.iter().map(|t| t.width() + 6).sum();
    let mut out = RgbImage::from_pixel(w, 360, Rgb([20, 20, 20]));
    let mut x = 0;
    for t in &tiles {
        image::imageops::overlay(&mut out, t, x as i64, 0);
        x += t.width() + 6;
    }
    out.save(path)?;
    Ok(())
}
