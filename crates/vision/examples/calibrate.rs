//! cargo run --release -p minastir-vision --example calibrate -- <data dir> <out dir>
//! Calibrates every camera's background.png (CPU) and writes scene maps + overlays to <out dir>/<id>/.
use minastir_vision::{calibrate::{calibrate_file, Calibrator}, moge::Moge, onnx::Device, segment::Segmenter};
use std::{path::Path, time::Instant};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, out) = (Path::new(&a[0]), Path::new(&a[1]));
    let mut cal = Calibrator {
        moge: Moge::load(&data.join("models/moge2_vitl_normal.onnx"), Device::Cpu)?,
        segmenter: Segmenter::load(&data.join("models/segformer_b5_ade.onnx"), Device::Cpu)?,
        num_tokens: 2500,
    };
    for e in std::fs::read_dir(data)? {
        let dir = e?.path();
        let bg = dir.join("background.png");
        if !bg.exists() { continue; }
        let id = dir.file_name().unwrap().to_string_lossy().to_string();
        let od = out.join(&id);
        std::fs::create_dir_all(&od)?;
        let t = Instant::now();
        match calibrate_file(&mut cal, &bg, &od) {
            Ok(c) => {
                println!("== {id} in {:?}: height {:.2} m, body pitch {:.1}° roll {:.1}°", t.elapsed(), c.height, c.pitch, c.roll);
                for (i, l) in c.lenses.iter().enumerate() {
                    println!("   lens {i}: hfov {:.1}° vfov {:.1}° yaw_off {:.1}° (ground pts {}, walls {}, slope {:.1}°)",
                        l.hfov, l.vfov, l.yaw_offset, l.ground_inliers, l.wall_points, l.ground_slope);
                }
            }
            Err(e) => println!("== {id}: {e:#}"),
        }
    }
    Ok(())
}
