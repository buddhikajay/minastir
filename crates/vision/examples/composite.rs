//! All calibrated cameras' ground maps placed with their current poses, as one top-down picture
//! (plus one picture per camera, in world coordinates). Diagnostic.
//! cargo run --release -p minastir-vision --example composite -- <data dir> <out dir>
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{geometry::CameraModel, groundmap, scenemap::LensMap, segment::Label, street};
use std::{path::Path, sync::Arc};

const RES: f64 = 0.1; // m per pixel
const HALF: f64 = 30.0; // metres around the world origin

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, out) = (Path::new(&a[0]), Path::new(&a[1]));
    std::fs::create_dir_all(out)?;
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let n = (2.0 * HALF / RES) as u32;
    let bg = image::Rgb([250u8, 250, 250]);
    let mut all = image::RgbImage::from_pixel(n, n, bg);
    let col = |l: Label| -> Option<[u8; 3]> {
        Some(match l {
            Label::Road => [140, 145, 155],
            Label::Sidewalk => [225, 200, 150],
            Label::Grass => [140, 200, 125],
            Label::Ground => [190, 160, 120],
            _ => return None,
        })
    };
    // World (x, z) → pixel; north (+z) up, +x to the right as on the plan.
    let px = |x: f64, z: f64| -> Option<(u32, u32)> {
        let (c, r) = ((x + HALF) / RES, (HALF - z) / RES);
        (c >= 0.0 && r >= 0.0 && c < n as f64 && r < n as f64).then(|| (c as u32, r as u32))
    };
    let mut fitted: Vec<(groundmap::GroundMap, Pose)> = Vec::new();
    for c in &cams {
        let (Some(cal), Some(pose)) = (c.calibration.clone(), c.pose) else { continue };
        let maps: anyhow::Result<Vec<LensMap>> =
            (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect();
        let Ok(maps) = maps else { continue };
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height * cal.scale, pitch: pose.pitch };
        let g = groundmap::build(&CameraModel::new(zero, &cal, Some(Arc::new(maps))), 40.0);
        let mut one = image::RgbImage::from_pixel(n, n, bg);
        let (s, co) = pose.yaw.to_radians().sin_cos();
        for r in 0..g.size {
            for cc in 0..g.size {
                let Some(rgb) = col(Label::from_u8(g.ground[r * g.size + cc])) else { continue };
                let lx = -g.half + (cc as f64 + 0.5) * groundmap::FINE;
                let lz = -g.half + (r as f64 + 0.5) * groundmap::FINE;
                let (wx, wz) = (pose.x + lx * co + lz * s, pose.z - lx * s + lz * co);
                if let Some((x, y)) = px(wx, wz) {
                    one.put_pixel(x, y, image::Rgb(rgb));
                    let e = all.get_pixel(x, y).0;
                    let m = if e == bg.0 { rgb } else { [0, 1, 2].map(|k| ((e[k] as u16 + rgb[k] as u16) / 2) as u8) };
                    all.put_pixel(x, y, image::Rgb(m));
                }
            }
        }
        for img in [&mut one, &mut all] {
            if let Some((x, y)) = px(pose.x, pose.z) {
                for d in -6i32..=6 {
                    for (xx, yy) in [(x as i32 + d, y as i32), (x as i32, y as i32 + d)] {
                        if xx >= 0 && yy >= 0 && (xx as u32) < n && (yy as u32) < n {
                            img.put_pixel(xx as u32, yy as u32, image::Rgb([200, 0, 0]));
                        }
                    }
                }
            }
        }
        one.save(out.join(format!("{}.png", c.name.clone().unwrap_or(c.id()).replace([' ', '(', ')'], ""))))?;
        fitted.push((g, pose));
    }
    let refs: Vec<(&groundmap::GroundMap, Pose)> = fitted.iter().map(|(g, p)| (g, *p)).collect();
    if let Some(st) = street::fit(&refs) {
        println!("street heading {:.1}°, origin ({:.2}, {:.2})", st.heading, st.origin[0], st.origin[1]);
        let mut img = all.clone();
        let (sn, cs) = st.heading.to_radians().sin_cos();
        for b in &st.bands {
            println!("  {:>9?}  t {:>6.2} .. {:>6.2}  ({:.2} m wide)  s {:>6.1} .. {:>5.1}  support {:.0}%", b.label, b.t0, b.t1, b.t1 - b.t0, b.s0, b.s1, b.support * 100.0);
            let Some(rgb) = col(b.label) else { continue };
            let mut t = b.t0;
            while t < b.t1 {
                let mut sa = b.s0;
                while sa < b.s1 {
                    // along (sn, cs), across (cs, -sn)
                    let (x, z) = (st.origin[0] + sa * sn + t * cs, st.origin[1] + sa * cs - t * sn);
                    if let Some((px_, py_)) = px(x, z) {
                        img.put_pixel(px_, py_, image::Rgb(rgb));
                    }
                    sa += RES * 0.5;
                }
                t += RES * 0.5;
            }
        }
        for d in &st.driveways {
            println!("  driveway  t {:>6.2} .. {:>6.2}  s {:>6.1} .. {:>5.1}  ({:.1} m wide, {:.1} m long)", d.t0, d.t1, d.s0, d.s1, d.s1 - d.s0, d.t1 - d.t0);
            let mut t = d.t0;
            while t < d.t1 {
                let mut sa = d.s0;
                while sa < d.s1 {
                    let (x, z) = (st.origin[0] + sa * sn + t * cs, st.origin[1] + sa * cs - t * sn);
                    if let Some((px_, py_)) = px(x, z) {
                        img.put_pixel(px_, py_, image::Rgb([205, 195, 185]));
                    }
                    sa += RES * 0.5;
                }
                t += RES * 0.5;
            }
        }
        img.save(out.join("street.png"))?;
    }
    // 5 m grid.
    for k in (0..n).step_by((5.0 / RES) as usize) {
        for t in 0..n {
            for (x, y) in [(k, t), (t, k)] {
                let p = all.get_pixel_mut(x, y);
                p.0 = p.0.map(|v| v.saturating_sub(18));
            }
        }
    }
    all.save(out.join("all.png"))?;
    Ok(())
}
