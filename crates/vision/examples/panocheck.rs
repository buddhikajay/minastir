//! Calibrate one camera's background offline and draw its ground map. Diagnostic.
//! cargo run --release -p minastir-vision --example panocheck -- <data dir> <camera id> <out.png>
use minastir_core::onvif::Pose;
use minastir_vision::{calibrate::Calibrator, geometry::CameraModel, groundmap, moge::Moge, onnx::Device, segment::Segmenter};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let data = Path::new(&a[0]);
    let mut cal = Calibrator {
        moge: Moge::load(&data.join("models/moge2_vitl_normal.onnx"), Device::Cpu)?,
        segmenter: Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?,
        num_tokens: 2500,
    };
    let img = image::open(data.join(&a[1]).join("background.png"))?.to_rgb8();
    let t = std::time::Instant::now();
    let r = cal.calibrate(img.as_raw(), img.width() as usize, img.height() as usize)?;
    let c = &r.calibration;
    println!("calibrated in {:?}: height {:.2} pitch {:.1} roll {:.1}", t.elapsed(), c.height, c.pitch, c.roll);
    for l in &c.lenses {
        println!("  lens hfov {:.1} vfov {:.1} yaw_off {:.1} cyl {} ground {} slope {:.1}", l.hfov, l.vfov, l.yaw_offset, l.cylindrical, l.ground_inliers, l.ground_slope);
    }
    let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: c.height, pitch: c.pitch };
    let model = CameraModel::new(zero, c, Some(Arc::new(r.lens_maps())));
    let g = groundmap::build(&model, 40.0);
    g.image().save(&a[2])?;
    Ok(())
}
