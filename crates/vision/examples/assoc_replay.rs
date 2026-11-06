//! Replays a harvest log's boxes through the image tracker, with the hand-set rule and with a
//! learned association model, and compares how many people each makes of the same footage.
//! cargo run --release -p minastir-vision --example assoc_replay -- <harvest.jsonl> <model.json> [from fraction]
//!
//! `from` (0–1, default 0) skips the start of the log: pass 0.7 to judge only the part the model
//! was not trained on.
use minastir_vision::{
    assoc::Model,
    imgtrack::{ImageBox, ImageTracker, OUTFIT},
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

struct Frame {
    t: f64,
    cam: String,
    boxes: Vec<ImageBox>,
}

fn load(path: &str) -> anyhow::Result<Vec<Frame>> {
    let mut frames = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        if v["look"].is_u64() {
            continue;
        }
        let boxes = v["boxes"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|b| !b["x"].is_null())
            .map(|b| {
                let f = |x: &serde_json::Value| x.as_f64().unwrap_or(0.0) as f32;
                let bx: Vec<f32> = b["box"].as_array().unwrap().iter().map(f).collect();
                let mut outfit = [0f32; OUTFIT];
                for (o, x) in outfit.iter_mut().zip(b["outfit"].as_array().unwrap()) {
                    *o = f(x);
                }
                ImageBox {
                    family: b["fam"].as_u64().unwrap_or(9) as u8,
                    bbox: [bx[0], bx[1], bx[2], bx[3]],
                    outfit,
                    ground: Some((b["x"].as_f64().unwrap(), b["z"].as_f64().unwrap(), b["range"].as_f64().unwrap_or(10.0))),
                    stature: b["h"].as_f64().map(|h| h as f32),
                }
            })
            .collect();
        frames.push(Frame { t: v["t"].as_f64().unwrap_or(0.0), cam: v["cam"].as_str().unwrap_or("").to_string(), boxes });
    }
    frames.sort_by(|a, b| a.t.total_cmp(&b.t));
    Ok(frames)
}

/// Per camera: people tracks seen for at least a second, and how many frames had a person box.
fn run(frames: &[Frame], model: Option<Arc<Model>>, from_t: f64) -> BTreeMap<String, (usize, usize, f64)> {
    let mut trackers: HashMap<String, ImageTracker> = HashMap::new();
    // (camera, uid) → (first, last) time seen, within the judged stretch.
    let mut spans: HashMap<(String, u64), (f64, f64)> = HashMap::new();
    let mut busiest: HashMap<String, usize> = HashMap::new();
    for f in frames {
        let it = trackers.entry(f.cam.clone()).or_insert_with(|| ImageTracker::with_model(model.clone()));
        let hints = it.assign(&f.boxes, 16.0 / 9.0, f.t);
        let uids = it.uids();
        // Stand-in for the 3D tracker: each image track is its own object.
        let gids: Vec<Option<u64>> = hints.iter().zip(&uids).map(|(h, u)| h.or(Some(*u))).collect();
        it.commit(&gids);
        if f.t < from_t {
            continue;
        }
        let people = f.boxes.iter().filter(|b| b.family == 0).count();
        let e = busiest.entry(f.cam.clone()).or_default();
        *e = (*e).max(people);
        for (b, g) in f.boxes.iter().zip(&gids) {
            if b.family == 0 {
                let s = spans.entry((f.cam.clone(), g.unwrap())).or_insert((f.t, f.t));
                s.1 = f.t;
            }
        }
    }
    let mut out: BTreeMap<String, (usize, usize, f64)> = BTreeMap::new();
    for ((cam, _), (a, b)) in &spans {
        if b - a >= 1.0 {
            let e = out.entry(cam.clone()).or_default();
            e.0 += 1;
            e.2 += b - a;
        }
    }
    for (cam, n) in busiest {
        out.entry(cam).or_default().1 = n;
    }
    out
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let frames = load(&a[0])?;
    let model = Arc::new(Model::load(std::path::Path::new(&a[1]))?);
    let from: f64 = a.get(2).map_or(Ok(0.0), |s| s.parse())?;
    let (t0, t1) = (frames.first().map_or(0.0, |f| f.t), frames.last().map_or(0.0, |f| f.t));
    let from_t = t0 + from * (t1 - t0);
    println!("{} frames, {:.0} s; judging from {:.0} s", frames.len(), t1 - t0, from_t - t0);
    let rule = run(&frames, None, from_t);
    let learned = run(&frames, Some(model), from_t);
    println!("{:<16} {:>14} {:>16} {:>14}", "camera", "most at once", "people: rule", "learned");
    for (cam, r) in &rule {
        let l = learned.get(cam).copied().unwrap_or_default();
        println!("{cam:<16} {:>14} {:>9} ({:>4.0} s avg) {:>6} ({:>4.0} s avg)", r.1, r.0, r.2 / r.0.max(1) as f64, l.0, l.2 / l.0.max(1) as f64);
    }
    Ok(())
}
