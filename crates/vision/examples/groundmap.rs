//! Top-down ground map per calibrated camera, as pictures. Diagnostic.
//! cargo run --release -p minastir-vision --example groundmap -- <data dir> <out dir>
use minastir_core::onvif::{CameraInfo, Pose};
use minastir_vision::{geometry::CameraModel, groundmap, scenemap::LensMap};
use std::{path::Path, sync::Arc};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (data, out) = (Path::new(&a[0]), Path::new(&a[1]));
    std::fs::create_dir_all(out)?;
    let cams: Vec<CameraInfo> = serde_json::from_slice(&std::fs::read(data.join("cameras.json"))?)?;
    for c in &cams {
        let Some(cal) = c.calibration.clone() else { continue };
        let maps: anyhow::Result<Vec<LensMap>> =
            (0..cal.lenses.len()).map(|i| LensMap::load(&data.join(c.id()).join(format!("scene_lens{i}.bin")))).collect();
        let Ok(maps) = maps else { continue };
        let zero = Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height * cal.scale, pitch: c.pose.map_or(cal.pitch, |p| p.pitch) };
        let model = CameraModel::new(zero, &cal, Some(Arc::new(maps)));
        let t = std::time::Instant::now();
        let g = groundmap::build(&model, 40.0);
        let known = g.ground.iter().filter(|&&l| l != 0).count() as f64 * groundmap::FINE * groundmap::FINE;
        println!(
            "{:<28} {:>6.0} m² ground, {:>4} m² filled under cars, {:>5} columns, {:>6} runs, {:?}",
            c.name.clone().unwrap_or(c.id()),
            known,
            (g.filled_under_vehicles as f64 * groundmap::CELL * groundmap::CELL) as i64,
            g.structures.len(),
            g.rle().len() / 2,
            t.elapsed()
        );
        println!("    bins: {:?}", g.shapes.bins.iter().map(|b| format!("({:.1},{:.1})", b.x, b.z)).collect::<Vec<_>>());
        println!("    {} walls ({} fences), {} trees ({} shrubs), {} poles",
            g.shapes.walls.len(), g.shapes.walls.iter().filter(|w| w.depth == 0.0).count(),
            g.shapes.trees.len(), g.shapes.trees.iter().filter(|t| t.shrub).count(), g.shapes.poles.len());
        for w in &g.shapes.walls {
            println!("      {:?} ({:.1},{:.1})-({:.1},{:.1}) top {:.1}", w.label, w.a[0], w.a[1], w.b[0], w.b[1], w.top);
        }
        g.image().save(out.join(format!("{}.png", c.id())))?;
    }
    Ok(())
}
