//! Refines the site photo's placement near where it is (see `sitealign::refine`) against the
//! cameras' current calibration, and prints old and new placement. Diagnostic; writes nothing.
//! cargo run --release -p minastir-vision --example siterefine -- <data dir>
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{geometry::CameraModel, groundmap, scenemap::LensMap, sitealign, street};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let data = Path::new(&std::env::args().nth(1).unwrap()).to_path_buf();
    let infos: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    let mut built = Vec::new();
    for c in &infos {
        let (Some(cal), Some(pose)) = (c.calibration.clone(), c.pose) else { continue };
        let maps: Option<Vec<LensMap>> = (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin"))).ok()).collect();
        let Some(maps) = maps else { continue };
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height * cal.scale, pitch: pose.pitch };
        let model = CameraModel::new(zero, &cal, Some(Arc::new(maps)));
        built.push((groundmap::build(&model, 40.0), pose));
    }
    let refs: Vec<_> = built.iter().map(|(g, p)| (g, *p)).collect();
    let st = street::fit(&refs);
    println!("street: {}", serde_json::to_string(&st)?);
    let tpl = sitealign::template(&refs, st.as_ref()).expect("template");
    let site: serde_json::Value = serde_json::from_slice(&std::fs::read(data.join("site.json"))?)?;
    let m = &site["map"];
    let f = |k: &str| m[k].as_f64().unwrap();
    let img = image::load_from_memory(&std::fs::read(data.join("site_map.bin"))?)?.to_rgb8();
    let n = refs.len() as f64;
    let about = (refs.iter().map(|r| r.1.x).sum::<f64>() / n, refs.iter().map(|r| r.1.z).sum::<f64>() / n);
    let start = (f("x"), f("z"), f("scale"), f("rotation"));
    let p = sitealign::refine(&img, &tpl, start, about);
    println!("old: x {:.3} z {:.3} scale {:.5} rot {:.2}", start.0, start.1, start.2, start.3);
    println!("new: x {:.3} z {:.3} scale {:.5} rot {:.2}  score {:.4} (×{:.2} the old placement's)", p.x, p.z, p.scale, p.rotation, p.score, p.confidence);
    Ok(())
}
