//! Re-measures one camera's lens from its background (with barrel distortion), and writes a copy
//! of the data directory's camera list with that camera's new lens, tilt and height, plus its new
//! scene map, into <out dir> (pose position, heading and depth scale are kept). Compare with
//! groundpts / kerbcheck pointed at <out dir>, then copy over by hand to install.
//! cargo run --release -p minastir-vision --example relens -- <data dir> <camera id> <out dir>
use minastir_core::onvif::CameraInfo;
use minastir_vision::{
    calibrate::{assemble, write_outputs, Calibrator},
    distortion,
    moge::Moge,
    onnx::Device,
    segment::Segmenter,
};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, id, out) = (Path::new(&a[0]), a[1].as_str(), Path::new(&a[2]));
    let mut cal = Calibrator {
        moge: Moge::load(&data.join("models/moge2_vitl_normal.onnx"), Device::Cpu)?,
        segmenter: Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?,
        num_tokens: 2500,
    };
    let img = image::open(data.join(id).join("background.png"))?.to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = cal.analyse(img.as_raw(), w, h)?;
    for (i, r) in raw.iter().enumerate() {
        let fit = distortion::from_kerbs(&r.labels);
        println!("lens {i}: k1 {:.3} ({:?})", r.k1, fit);
    }
    let r = assemble(&raw, &vec![1.0; raw.len()])?;
    let c = &r.calibration;
    println!("pitch {:.1}° roll {:.1}° height {:.2} m", c.pitch, c.roll, c.height);
    for l in &c.lenses {
        println!("  hfov {:.1}° vfov {:.1}° k1 {:.3} (ground {} pts, rms {:.3} m)", l.hfov, l.vfov, l.k1, l.ground_inliers, l.ground_rms);
    }
    std::fs::create_dir_all(out.join(id))?;
    write_outputs(&img, &r, &out.join(id))?;
    let mut infos: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    for info in infos.iter_mut().filter(|i| i.id() == id) {
        let old = info.calibration.clone();
        let mut new = c.clone();
        new.scale = old.as_ref().map_or(1.0, |o| o.scale);
        new.alignment = old.as_ref().and_then(|o| o.alignment.clone());
        if let Some(p) = info.pose.as_mut() {
            p.pitch = (new.pitch * 10.0).round() / 10.0;
            p.height = (new.height * new.scale * 100.0).round() / 100.0;
        }
        info.calibration = Some(new);
    }
    std::fs::write(out.join("cameras.json"), serde_json::to_vec_pretty(&infos)?)?;
    // The other cameras' scene maps, so the copy is a complete data directory for the tools.
    for info in &infos {
        if info.id() != id {
            let (src, dst) = (data.join(info.id()), out.join(info.id()));
            std::fs::create_dir_all(&dst)?;
            for e in std::fs::read_dir(&src)? {
                let p = e?.path();
                if p.extension().is_some_and(|x| x == "bin") {
                    std::fs::copy(&p, dst.join(p.file_name().unwrap()))?;
                }
            }
        }
    }
    Ok(())
}
