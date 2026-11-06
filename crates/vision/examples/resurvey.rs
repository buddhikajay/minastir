//! Re-calibrates a set of cameras together (lens with distortion, tilt, height, then lens width,
//! depth scale and placement from points matched between them), keeping the first camera where
//! it is, and writes a complete copy of the data directory's cameras + scene maps to <out dir>.
//! Compare with groundpts / kerbcheck pointed at <out dir>; copy over by hand to install.
//! cargo run --release -p minastir-vision --example resurvey -- <data dir> <out dir> <fixed camera id> <camera id>…
use minastir_core::onvif::CameraInfo;
use minastir_vision::{
    calibrate::{write_outputs, Calibrator},
    matcher::Matcher,
    moge::Moge,
    onnx::Device,
    segment::Segmenter,
    survey::{survey, SurveyCam},
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, out) = (Path::new(&a[0]), Path::new(&a[1]));
    let ids: Vec<String> = a[2..].to_vec();
    let mut infos: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut cal = Calibrator {
        moge: Moge::load(&data.join("models/moge2_vitl_normal.onnx"), Device::Cpu)?,
        segmenter: Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?,
        num_tokens: 2500,
    };
    // Survey anchors the best-connected camera, and on a tie the last one: list the fixed camera
    // last. (Re-anchored below in case it isn't.)
    let mut order = ids.clone();
    order.rotate_left(1);
    let mut cams = Vec::new();
    for id in &order {
        let info = infos.iter().find(|i| &i.id() == id).expect("camera");
        let image = image::open(data.join(id).join("background.png"))?.to_rgb8();
        let raw = cal.analyse(image.as_raw(), image.width() as usize, image.height() as usize)?;
        println!("{}: k1 {:?}", info.label(), raw.iter().map(|r| (r.k1 * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        cams.push(SurveyCam { id: id.clone(), label: info.label(), rough: info.pose.expect("placed"), image, raw });
    }
    let images: Vec<image::RgbImage> = cams.iter().map(|c| c.image.clone()).collect();
    let mut matcher = Matcher::load(&data.join("models").join(minastir_vision::matcher::MODEL_FILE), Device::Cpu)?;
    let results = survey(cams, &mut matcher, &mut |l| println!("  {l}"))?;

    // Keep the fixed camera exactly where it was: map its solved placement back onto its old one.
    let fixed = results.iter().find(|s| s.id == ids[0]).unwrap();
    let old = infos.iter().find(|i| i.id() == ids[0]).unwrap().pose.unwrap();
    let d = (old.yaw - fixed.placed.pose.yaw).to_radians();
    let k = 1.0 / fixed.placed.scale;
    let (sd, cd) = d.sin_cos();
    let to_old = |x: f64, z: f64| {
        let (rx, rz) = (x - fixed.placed.pose.x, z - fixed.placed.pose.z);
        (old.x + k * (rx * cd + rz * sd), old.z + k * (-rx * sd + rz * cd))
    };
    std::fs::create_dir_all(out)?;
    for (s, img) in results.iter().zip(&images) {
        std::fs::create_dir_all(out.join(&s.id))?;
        write_outputs(img, &s.result, &out.join(&s.id))?;
        let info = infos.iter_mut().find(|i| i.id() == s.id).unwrap();
        let mut c = s.result.calibration.clone();
        c.scale = s.placed.scale * k;
        c.alignment = s.placed.alignment.clone();
        let (x, z) = to_old(s.placed.pose.x, s.placed.pose.z);
        let before = info.pose.unwrap();
        let pose = minastir_core::onvif::Pose {
            x: (x * 100.0).round() / 100.0,
            z: (z * 100.0).round() / 100.0,
            yaw: ((s.placed.pose.yaw + d.to_degrees()).rem_euclid(360.0) * 10.0).round() / 10.0,
            pitch: (c.pitch * 10.0).round() / 10.0,
            height: (c.height * c.scale * 100.0).round() / 100.0,
        };
        let l = &c.lenses[0];
        println!(
            "{}: hfov {:.1}° k1 {:.3} widen {:.2} scale {:.3} | pose ({:.2}, {:.2}) yaw {:.1} pitch {:.1} h {:.2}  (was ({:.2}, {:.2}) yaw {:.1} pitch {:.1} h {:.2})",
            info.label(), l.hfov, l.k1, c.fov_correction, c.scale, pose.x, pose.z, pose.yaw, pose.pitch, pose.height,
            before.x, before.z, before.yaw, before.pitch, before.height
        );
        info.pose = Some(pose);
        info.calibration = Some(c);
    }
    std::fs::write(out.join("cameras.json"), serde_json::to_vec_pretty(&infos)?)?;
    for info in &infos {
        if !ids.contains(&info.id()) && data.join(info.id()).exists() {
            std::fs::create_dir_all(out.join(info.id()))?;
            for e in std::fs::read_dir(data.join(info.id()))? {
                let p = e?.path();
                if p.extension().is_some_and(|x| x == "bin") {
                    std::fs::copy(&p, out.join(info.id()).join(p.file_name().unwrap()))?;
                }
            }
        }
    }
    Ok(())
}
