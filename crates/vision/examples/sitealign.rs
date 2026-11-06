//! Auto-align the site map image against the cameras' map. Diagnostic.
//! cargo run --release -p minastir-vision --example sitealign -- <data dir> [out.png]
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{geometry::CameraModel, groundmap, scenemap::LensMap, sitealign, street};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut built = Vec::new();
    for c in &cams {
        let (Some(cal), Some(pose)) = (c.calibration.clone(), c.pose) else { continue };
        let maps: anyhow::Result<Vec<LensMap>> =
            (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect();
        let Ok(maps) = maps else { continue };
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height * cal.scale, pitch: pose.pitch };
        built.push((groundmap::build(&CameraModel::new(zero, &cal, Some(Arc::new(maps))), 40.0), pose));
    }
    let refs: Vec<_> = built.iter().map(|(g, p)| (g, *p)).collect();
    let st = street::fit(&refs);
    let tpl = sitealign::template(&refs, st.as_ref()).expect("template");
    let img = image::load_from_memory(&std::fs::read(data.join("site_map.bin"))?)?.to_rgb8();
    println!("photo {}x{}", img.width(), img.height());
    let t = std::time::Instant::now();
    let p = sitealign::align(&img, &tpl).expect("aligned");
    println!("{p:?} in {:?}", t.elapsed());
    if let Ok(site) = serde_json::from_slice::<serde_json::Value>(&std::fs::read(data.join("site.json"))?) {
        let m = &site["map"];
        if let (Some(x), Some(z), Some(s), Some(r)) = (m["x"].as_f64(), m["z"].as_f64(), m["scale"].as_f64(), m["rotation"].as_f64()) {
            println!("manual placement: scale {s:.4} rot {r:.1} → score {:.3}", sitealign::evaluate(&img, &tpl, x, z, s, r));
            println!("auto placement   → score {:.3}", sitealign::evaluate(&img, &tpl, p.x, p.z, p.scale, p.rotation));
        }
    }
    if let Some(out) = a.get(1) {
        // Overlay: photo resampled into world, template on top (green = grass, red = paved).
        let (half, res) = (45.0, 0.1);
        let n = (2.0 * half / res) as u32;
        let (sn, cs) = p.rotation.to_radians().sin_cos();
        let cx = refs.iter().map(|r| r.1.x).sum::<f64>() / refs.len() as f64;
        let cz = refs.iter().map(|r| r.1.z).sum::<f64>() / refs.len() as f64;
        let mut o = image::RgbImage::new(n, n);
        for y in 0..n {
            for x in 0..n {
                let (wx, wz) = (cx - half + x as f64 * res, cz - half + y as f64 * res);
                let (dx, dz) = ((wx - p.x) / p.scale, (wz - p.z) / p.scale);
                let (u, v) = (dx * cs + dz * sn, -dx * sn + dz * cs);
                if u >= 0.0 && v >= 0.0 && (u as u32) < img.width() && (v as u32) < img.height() {
                    o.put_pixel(x, y, *img.get_pixel(u as u32, v as u32));
                }
            }
        }
        // Street bands outline on top.
        if let Some(st) = &st {
            let (sn2, cs2) = st.heading.to_radians().sin_cos();
            for b in st.bands.iter().filter(|b| b.label == minastir_vision::segment::Label::Road) {
                for t in [b.t0, b.t1] {
                    let mut s = b.s0;
                    while s < b.s1 {
                        let (wx, wz) = (st.origin[0] + s * sn2 + t * cs2, st.origin[1] + s * cs2 - t * sn2);
                        let (px, py) = (((wx - cx + half) / res) as i64, ((wz - cz + half) / res) as i64);
                        if px >= 0 && py >= 0 && px < n as i64 && py < n as i64 {
                            o.put_pixel(px as u32, py as u32, image::Rgb([255, 40, 40]));
                        }
                        s += 0.05;
                    }
                }
            }
        }
        for (_, pose) in &refs {
            let (px, py) = (((pose.x - cx + half) / res) as i64, ((pose.z - cz + half) / res) as i64);
            for d in -8..=8 {
                for (xx, yy) in [(px + d, py), (px, py + d)] {
                    if xx >= 0 && yy >= 0 && xx < n as i64 && yy < n as i64 {
                        o.put_pixel(xx as u32, yy as u32, image::Rgb([0, 120, 255]));
                    }
                }
            }
        }
        o.save(out)?;
    }
    Ok(())
}
