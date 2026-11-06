//! Top-down view of every calibrated camera's scene map at its current pose.
//! cargo run --release -p minastir-vision --example bev -- <data dir> <out.png>
use minastir_core::onvif::CameraInfo;
use minastir_vision::{align::{align, AlignCam}, calibrate::label_color, geometry::CameraModel, scenemap::LensMap};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let mut cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    if a.get(2).map(String::as_str) == Some("--align") {
        let mut acs = Vec::new();
        for c in &cams {
            let (Some(pose), Some(cal)) = (c.pose, c.calibration.as_ref()) else { continue };
            let maps: Vec<LensMap> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect::<Result<_, _>>()?;
            let zero = minastir_core::onvif::Pose { x: 0.0, z: 0.0, yaw: 0.0, ..pose };
            let m = CameraModel::new(zero, cal, Some(Arc::new(maps)));
            let local = m.scene_points(2, 30.0).into_iter().map(|(x, z, _, l)| (x as f32, z as f32, l)).collect();
            acs.push(AlignCam { id: c.id(), rough: pose, local });
        }
        let t = std::time::Instant::now();
        let res = align(&acs);
        println!("aligned in {:?}", t.elapsed());
        for r in &res {
            let c = cams.iter_mut().find(|c| c.id() == r.id).unwrap();
            let label = c.name.clone().unwrap_or(c.label());
            let p = c.pose.as_mut().unwrap();
            println!("  {:<28} {} yaw {:>6.1}° → {:>6.1}°  pos ({:>5.1},{:>5.1}) → ({:>5.1},{:>5.1})  agreement {:?}",
                label, if r.anchor { "[anchor]" } else { "        " }, p.yaw, r.yaw, p.x, p.z, r.x, r.z, r.agreement.map(|a| (a * 100.0).round()));
            p.yaw = r.yaw; p.x = r.x; p.z = r.z;
        }
        if let Some(out) = a.get(3) { std::fs::write(out, serde_json::to_vec_pretty(&cams)?)?; }
    }
    let (res, half) = (0.1f64, 40.0f64); // 10 cm cells, 80 m square
    let n = (2.0 * half / res) as u32;
    let mut img = image::RgbImage::from_pixel(n, n, image::Rgb([245, 246, 247]));
    for c in &cams {
        let (Some(pose), Some(cal)) = (c.pose, c.calibration.as_ref()) else { continue };
        let maps: Vec<LensMap> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect::<Result<_, _>>()?;
        let m = CameraModel::new(pose, cal, Some(Arc::new(maps)));
        let pts = m.scene_points(2, 35.0);
        println!("{}: {} points", c.label(), pts.len());
        for (x, z, _y, l) in pts {
            let (px, py) = (((x + half) / res) as i64, ((z + half) / res) as i64);
            if px >= 0 && py >= 0 && (px as u32) < n && (py as u32) < n {
                let col = label_color(l);
                let p = img.get_pixel_mut(px as u32, py as u32);
                for k in 0..3 { p[k] = ((p[k] as u16 + 2 * col[k] as u16) / 3) as u8; }
            }
        }
        // camera marker
        let (cx, cy) = (((pose.x + half) / res) as i64, ((pose.z + half) / res) as i64);
        for dx in -6..=6 { for dy in -6..=6 { if dx * dx + dy * dy <= 36 {
            let (px, py) = (cx + dx, cy + dy);
            if px >= 0 && py >= 0 && (px as u32) < n && (py as u32) < n { img.put_pixel(px as u32, py as u32, image::Rgb([0, 111, 255])); }
        }}}
    }
    img.save(&a[1])?;
    Ok(())
}
