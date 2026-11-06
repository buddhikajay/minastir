//! The always-on live engine (runs in the server process; the web app only renders its output).
//!
//!   per camera:  ffmpeg (hardware decode) → 1280-px RGB frames at a few fps → "latest frame" slot
//!   one worker:  newest frame per camera → detector (Metal GPU, on a 640-px copy) → ground
//!                projection → tracker
//!   fingerprints: sharp crops of people and vehicles (whole body / whole vehicle) from the full frame, ~2×/s per camera → appearance model
//!                (CPU, its own thread) → back into the tracker as evidence of identity
//!   fitting:     parked vehicles, one at a time → outlines (Segment Anything) in every camera
//!                that sees them → 3D vehicle fitted to all outlines (`fit`) → pose into the tracker
//!   describing:  each new "vehicle", once → a vision-language model says car / truck / bin /
//!                other; non-vehicles stop being shown as vehicles (bins are shown as bins)
//!   publish:     a snapshot of tracked objects ~10×/s through a callback
//!
//! Frames are dropped, never queued: if inference falls behind, cameras just get fewer fps.

use crate::{
    appearance::{self, Crop, Embedder},
    detector::{Detection, Detector, INPUT},
    geometry,
    onnx::Device,
    describe::Describer,
    fit::{fit_guided, shape_for, View},
    wheels,
    imgtrack::{outfit, ImageBox, ImageTracker},
    roof::Sam,
    tracker::{AppearanceStats, Fitted, GroundDetection, TrackedObject, Tracker},
};
use anyhow::{bail, Context, Result};
use minastir_core::onvif::Pose;
use serde::Serialize;
use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub const MODEL_URL: &str = "https://huggingface.co/onnx-community/dfine_m_obj2coco-ONNX/resolve/main/onnx/model.onnx";
pub const MODEL_FILE: &str = "dfine_m_obj2coco.onnx";

/// The detector to run: D-FINE-X by default (on this site's footage it stops seeing trucks in a
/// letterbox and phantom cars that the M model reports, at ~36 ms per tile instead of ~22 on the
/// GPU); MINASTIR_DETECTOR=m picks the faster M. Returns (file, URL, size for the message).
pub fn detector_model() -> (&'static str, &'static str, &'static str) {
    match std::env::var("MINASTIR_DETECTOR").as_deref() {
        Ok("m") => (MODEL_FILE, MODEL_URL, "79 MB"),
        _ => ("dfine_x_obj2coco.onnx", "https://huggingface.co/onnx-community/dfine_x_obj2coco-ONNX/resolve/main/onnx/model.onnx", "251 MB"),
    }
}

/// Frames are decoded at this multiple of the detector's input: the detector gets a downscaled
/// copy, fingerprints get sharp crops.
const SHARP: usize = 2;
/// How often each camera's vehicles are fingerprinted, and how many at a time.
const LOOK_EVERY: Duration = Duration::from_millis(500);
const LOOKS_PER_FRAME: usize = 10;
/// Smallest crop worth fingerprinting (pixels of the sharp frame, each side).
const LOOK_MIN_PX: f64 = 40.0;

/// A camera the engine should watch.
#[derive(Clone)]
pub struct CamSpec {
    pub id: String,
    /// ffmpeg input (credentials included).
    pub source: String,
    /// Width / height of that stream.
    pub aspect: f64,
    pub pose: Pose,
    /// Measured lens/tilt/roll, and the background's per-pixel 3D map (when calibrated).
    pub calibration: Option<minastir_core::calib::CameraCalibration>,
    pub maps: Option<Arc<Vec<crate::scenemap::LensMap>>>,
    /// The camera's empty-scene picture (people-free background), when captured.
    pub background: Option<Arc<image::RgbImage>>,
}

impl CamSpec {
    fn model(&self) -> Option<geometry::CameraModel> {
        self.calibration.as_ref().map(|c| geometry::CameraModel::new(self.pose, c, self.maps.clone()))
    }
}

pub struct LiveConfig {
    pub model_dir: PathBuf,
    pub fps: f32,
    pub min_score: f32,
    pub use_gpu: bool,
    /// Where the latest analysed frame of each camera is kept (for showing the footage).
    pub frames: FrameStore,
}

/// The latest analysed frame of each camera, shared with whoever wants to show it.
#[derive(Clone, Default)]
pub struct FrameStore(Arc<Mutex<HashMap<String, (Arc<Frame>, f64)>>>);

impl FrameStore {
    fn put(&self, camera: &str, frame: Arc<Frame>, aspect: f64) {
        self.0.lock().unwrap().insert(camera.to_string(), (frame, aspect));
    }

    /// The camera's latest analysed frame as a JPEG `width` pixels wide, in its true shape.
    pub fn jpeg(&self, camera: &str, width: u32) -> Option<Vec<u8>> {
        let (frame, aspect) = self.0.lock().unwrap().get(camera).cloned()?;
        if frame.at.elapsed() > Duration::from_secs(5) {
            return None;
        }
        let (w, h) = ((INPUT * SHARP * frame.tiles) as u32, (INPUT * SHARP) as u32);
        let img = image::RgbImage::from_raw(w, h, frame.data.clone())?;
        let out_h = ((width as f64 / aspect).round() as u32).max(1);
        let small = image::imageops::resize(&img, width, out_h, image::imageops::FilterType::Triangle);
        let mut buf = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 75).encode_image(&small).ok()?;
        Some(buf)
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct CameraStatus {
    /// Frames analysed per second.
    pub fps: f32,
    pub error: Option<String>,
    /// Latest raw detections (normalised boxes) — for overlays on the video.
    pub detections: Vec<Detection>,
    /// Where this camera alone places each of them on the ground: (class, x, z). Comparing
    /// cameras here shows how well they agree (i.e. how good the calibration is).
    /// Each entry: (class, x, z, track id, foot point u, v in the image) — the track id and foot
    /// point let sightings of one object by two cameras be compared (calibration, handover).
    pub ground: Vec<(&'static str, f64, f64, Option<u64>, f32, f32)>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Snapshot {
    /// Unix time, seconds.
    pub time: f64,
    /// "starting", "downloading", "loading", "running", "error".
    pub state: &'static str,
    pub message: String,
    /// Running on the GPU via CoreML.
    pub accelerated: bool,
    /// Mean inference time per frame, ms.
    pub infer_ms: f32,
    pub fingerprints: FingerprintStatus,
    pub fitting: FitStatus,
    pub describing: FitStatus,
    pub objects: Vec<TrackedObject>,
    pub cameras: BTreeMap<String, CameraStatus>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FingerprintStatus {
    /// "off", "downloading", "loading", "running", "error".
    pub state: &'static str,
    pub message: String,
    pub model: String,
    /// Crops fingerprinted per second, and mean time per crop (ms).
    pub per_sec: f32,
    pub crop_ms: f32,
    /// What it has learned "same" and "different" look like, for vehicles and for people.
    pub stats: Option<AppearanceStats>,
    pub people: Option<AppearanceStats>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct FitStatus {
    /// "off", "loading", "running", "error".
    pub state: &'static str,
    pub message: String,
    /// Fits done, and the last one's time (ms).
    pub done: u64,
    pub last_ms: f32,
}

pub struct LiveHandle {
    stop: Arc<AtomicBool>,
}

impl Drop for LiveHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Start the engine. `specs` is polled for the current cameras and poses (cheap; called ~1×/s);
/// `publish` receives every snapshot.
pub fn spawn(
    config: LiveConfig,
    specs: impl Fn() -> Vec<CamSpec> + Send + 'static,
    publish: impl Fn(&Snapshot) + Send + 'static,
) -> LiveHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    thread::Builder::new()
        .name("live-engine".into())
        .spawn(move || {
            let mut snap = Snapshot { state: "starting", ..Default::default() };
            if let Err(e) = run(&config, &specs, &publish, &stop2, &mut snap) {
                snap.state = "error";
                snap.message = format!("{e:#}");
                snap.objects.clear();
                eprintln!("live engine stopped: {e:#}");
                publish(&snap);
            }
        })
        .expect("spawn live engine");
    LiveHandle { stop }
}

pub struct Frame {
    data: Vec<u8>,
    tiles: usize,
    at: Instant,
}

struct Reader {
    spec: CamSpec,
    model: Option<geometry::CameraModel>,
    slot: Arc<Mutex<Option<Frame>>>,
    error: Arc<Mutex<Option<String>>>,
    stop: Arc<AtomicBool>,
    analysed: Vec<Instant>,
    detections: Vec<Detection>,
    ground: Vec<(&'static str, f64, f64, Option<u64>, f32, f32)>,
    last_look: Option<Instant>,
    /// The latest sharp frame and which track each vehicle box in it went to.
    last: Option<(Arc<Frame>, Vec<(u64, [f32; 4], &'static str)>)>,
    /// Tracks within this camera's image (overlap + outfit).
    image: ImageTracker,
    /// Spots the describer judged not to be a vehicle (a letterbox…): vehicle boxes there are
    /// part of the scene, not vehicles.
    not_vehicles: Vec<[f32; 4]>,
}

impl Drop for Reader {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn run(
    config: &LiveConfig,
    specs: &dyn Fn() -> Vec<CamSpec>,
    publish: &dyn Fn(&Snapshot),
    stop: &AtomicBool,
    snap: &mut Snapshot,
) -> Result<()> {
    let (file, url, size) = detector_model();
    let model = config.model_dir.join(file);
    if !model.exists() {
        snap.state = "downloading";
        snap.message = format!("Downloading the detection model ({size}, once)");
        publish(snap);
        download(url, &model)?;
    }
    snap.state = "loading";
    snap.message = "Loading the detection model".into();
    publish(snap);
    let mut detector = Detector::load(&model, config.use_gpu).or_else(|e| {
        eprintln!("GPU detector failed ({e:#}); falling back to CPU");
        Detector::load(&model, false)
    })?;
    snap.accelerated = detector.accelerated;

    let epoch = Instant::now();
    let mut tracker = Tracker::default();
    // Body keypoints: whether a person's feet are in view, and from that their height — a cue
    // to who's who (children and adults) for the association model. MINASTIR_POSE:
    //   off      no pose model
    //   measure  (default) measure heights only
    //   place    also place people whose feet are hidden by their head and height
    //   ankles   place by the ankle keypoints instead of the box bottom
    // Placement stays opt-in: on the 5:47 pm replay it made positions less steady — these cameras
    // hang low, so a head's line of sight is shallow and a small height error moves a person far.
    let pose_mode = std::env::var("MINASTIR_POSE").unwrap_or_else(|_| "measure".into());
    let mut pose = if pose_mode == "off" {
        None
    } else {
        match crate::pose::Pose::load(&config.model_dir, if config.use_gpu { Device::Gpu } else { Device::Cpu }) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("pose model unavailable ({e:#}); placing people by the bottom of their box");
                None
            }
        }
    };
    let mut harvest = Harvest::open();
    let assoc = load_assoc(&config.model_dir);
    let looks = LookWorker::spawn(config.model_dir.clone());
    let fitter = FitWorker::spawn(config.model_dir.clone());
    let describer = DescribeWorker::spawn(config.model_dir.clone());
    // Which camera and box each describer question was about.
    let mut asked: HashMap<u64, (String, [f32; 4])> = HashMap::new();
    let mut readers: HashMap<String, Reader> = HashMap::new();
    let mut last_specs = Instant::now() - Duration::from_secs(10);
    let mut last_publish = Instant::now();
    let mut infer_ms = 0f32;

    snap.state = "running";
    snap.message.clear();
    while !stop.load(Ordering::Relaxed) {
        if last_specs.elapsed() > Duration::from_secs(1) {
            last_specs = Instant::now();
            let current = specs();
            readers.retain(|id, _| current.iter().any(|s| &s.id == id));
            for spec in current {
                match readers.get_mut(&spec.id) {
                    // Pose edits apply immediately; a new source restarts the reader.
                    Some(r) if r.spec.source == spec.source && r.spec.aspect == spec.aspect => {
                        r.model = spec.model();
                        r.spec = spec;
                    }
                    _ => {
                        let mut r = start_reader(spec, config.fps);
                        r.image = ImageTracker::with_model(assoc.clone());
                        readers.insert(r.spec.id.clone(), r);
                    }
                }
            }
        }

        let mut worked = false;
        for (id, r) in readers.iter_mut() {
            let Some(frame) = r.slot.lock().unwrap().take() else { continue };
            worked = true;
            let t = Instant::now();
            let mut dets = Vec::new();
            for tile in 0..frame.tiles {
                let rgb = tile_of(&frame.data, frame.tiles, tile);
                for mut d in detector.detect(&[&rgb], config.min_score)?.remove(0) {
                    let n = frame.tiles as f32;
                    d.bbox[0] = (tile as f32 + d.bbox[0]) / n;
                    d.bbox[2] = (tile as f32 + d.bbox[2]) / n;
                    dets.push(d);
                }
            }
            let ms = t.elapsed().as_secs_f32() * 1000.0 / frame.tiles as f32;
            infer_ms = if infer_ms == 0.0 { ms } else { 0.9 * infer_ms + 0.1 * ms };

            // Vehicle boxes on a spot the describer already judged not a vehicle: the scene itself.
            let overlap = |a: &[f32; 4], b: &[f32; 4]| {
                let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
                let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
                let i = w * h;
                let u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
                if u > 0.0 { i / u } else { 0.0 }
            };
            dets.retain(|d| crate::detector::family(d.class) != 1 || !r.not_vehicles.iter().any(|n| overlap(&d.bbox, n) >= 0.5));
            // A second, smaller vehicle box mostly inside another is part of that vehicle
            // (the detector sometimes boxes a car's half as well as the car), not a second one.
            let inside = |a: &Detection, b: &Detection| {
                let w = (a.bbox[2].min(b.bbox[2]) - a.bbox[0].max(b.bbox[0])).max(0.0);
                let h = (a.bbox[3].min(b.bbox[3]) - a.bbox[1].max(b.bbox[1])).max(0.0);
                let area = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
                area > 0.0 && w * h / area > 0.7
            };
            let area = |d: &Detection| (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]);
            let keep: Vec<bool> = dets
                .iter()
                .map(|a| {
                    !dets.iter().any(|b| {
                        // Vehicles only: people really do stand inside each other's boxes (a child
                        // in front of a parent).
                        !std::ptr::eq(a, b)
                            && crate::detector::family(a.class) == 1
                            && crate::detector::family(b.class) == 1
                            && area(b) > area(a)
                            && inside(a, b)
                    })
                })
                .collect();
            let mut k = 0;
            dets.retain(|_| {
                k += 1;
                keep[k - 1]
            });
            let (fw, fh) = (INPUT * SHARP * frame.tiles, INPUT * SHARP);
            // People's keypoints: their feet, when the pose model sees them.
            let keypoints: Vec<Option<crate::pose::Keypoints>> = dets
                .iter()
                .map(|d| if d.class == "person" { pose.as_mut().and_then(|p| p.estimate((&frame.data, fw, fh), d.bbox).ok()) } else { None })
                .collect();
            let feet: Vec<Option<(f32, f32)>> = keypoints.iter().map(|k| k.and_then(|k| k.feet(crate::pose::SEEN))).collect();
            // Seen feet: the box bottom is the sole, and steadier than the ankle keypoints (which
            // sit ~10 cm up and swap between feet with every step) — so the pose model only says
            // *whether* the feet are in view. MINASTIR_POSE=ankles places by the ankles instead.
            let ankle_feet = pose_mode == "ankles";
            let (gidx, ground): (Vec<usize>, Vec<GroundDetection>) = dets
                .iter()
                .enumerate()
                .filter_map(|(k, d)| ground_detection(id, d, &r.spec, r.model.as_ref(), if ankle_feet { feet[k] } else { None }).map(|g| (k, g)))
                .unzip();
            // Track within this camera's image first (overlap + outfit, with memory for anyone
            // briefly hidden): each box carries its image track's 3D id into the 3D tracker.
            let mut ground = ground;
            // Heights, measured while someone's feet are seen: (height, ground height under them).
            let heights: Vec<Option<(f32, f32)>> = gidx
                .iter()
                .map(|&k| {
                    let (d, m, f) = (&dets[k], r.model.as_ref()?, feet[k]?);
                    if d.class != "person" || d.bbox[1] < 0.01 {
                        return None;
                    }
                    let u = ((d.bbox[0] + d.bbox[2]) / 2.0) as f64;
                    let (fu, fv) = if ankle_feet { (f.0 as f64, f.1 as f64) } else { (u, d.bbox[3] as f64) };
                    let foot = m.ground_point3(fu, fv, 60.0)?;
                    Some((stature(m, u, d.bbox[1] as f64, foot)? as f32, foot[1] as f32))
                })
                .collect();
            let boxes: Vec<ImageBox> = gidx
                .iter()
                .zip(&ground)
                .zip(&heights)
                .map(|((&k, g), h)| {
                    let bg = r.spec.background.as_deref().map(|b| (b.as_raw().as_slice(), b.width() as usize, b.height() as usize));
                    ImageBox { family: crate::detector::family(dets[k].class), bbox: dets[k].bbox, outfit: outfit((&frame.data, fw, fh), bg, dets[k].bbox), ground: Some((g.x, g.z, g.range)), stature: h.map(|h| h.0) }
                })
                .collect();
            let now_s = (frame.at - epoch).as_secs_f64();
            let hints = r.image.assign(&boxes, r.spec.aspect as f32, now_s);
            for (g, h) in ground.iter_mut().zip(&hints) {
                g.prev = *h;
            }
            for (i, h) in heights.iter().enumerate() {
                if let Some((h, floor)) = h {
                    r.image.add_stature(i, *h, *floor);
                }
            }
            // Optionally, people whose feet are hidden behind something: placed by their head.
            if let (Some(m), true) = (r.model.as_ref(), pose_mode == "place" || pose_mode == "ankles") {
                for (i, &k) in gidx.iter().enumerate() {
                    let d = &dets[k];
                    if d.class != "person" || keypoints[k].is_none() || feet[k].is_some() || d.bbox[1] < 0.01 {
                        continue;
                    }
                    let u = ((d.bbox[0] + d.bbox[2]) / 2.0) as f64;
                    // Only if the box is cut short: its bottom and top make them clearly shorter
                    // than they are. Otherwise its bottom is at their feet after all (small and
                    // far away, the pose model is just unsure of them).
                    let cut = |known: f32| {
                        let shown = m.ground_point3(u, d.bbox[3] as f64, 60.0).and_then(|p| stature_any(m, u, d.bbox[1] as f64, p));
                        shown.is_none_or(|h| h < CUT_SHORT * known as f64)
                    };
                    let placed = r.image.stature(i).filter(|&(h, _)| cut(h)).and_then(|(h, floor)| head_placement(m, u, d.bbox[1] as f64, h as f64, floor as f64));
                    if let Some((x, z)) = placed {
                        ground[i].x = x;
                        ground[i].z = z;
                        ground[i].range = (x - r.spec.pose.x).hypot(z - r.spec.pose.z);
                    }
                }
            }
            let ids = tracker.update(id, &ground, now_s);
            r.image.commit(&ids);
            if let Some(h) = harvest.as_mut() {
                h.frame(id, now_s, (r.spec.pose.x, r.spec.pose.z), &boxes, &r.image.uids(), &ids, &ground, &gidx.iter().map(|&k| dets[k].score).collect::<Vec<_>>());
            }
            let frame = Arc::new(frame);
            config.frames.put(id, frame.clone(), r.spec.aspect);
            r.last = Some((
                frame.clone(),
                gidx.iter().zip(&ids).filter_map(|(&k, tid)| Some((tid.as_ref().copied()?, dets[k].bbox, dets[k].class))).collect(),
            ));
            if r.last_look.is_none_or(|t| t.elapsed() >= LOOK_EVERY) && looks.ready() {
                let picks: Vec<(u64, &Detection)> = gidx
                    .iter()
                    .zip(&ids)
                    .filter_map(|(&k, tid)| Some((tid.as_ref().copied()?, &dets[k])))
                    .filter(|(_, d)| crate::detector::family(d.class) <= 1) // people and vehicles
                    .collect();
                let job = crops_for(&frame, r.spec.aspect, &picks);
                if !job.is_empty() && looks.send(id.clone(), job) {
                    r.last_look = Some(Instant::now());
                }
            }
            r.ground = ground
                .iter()
                .zip(&gidx)
                .zip(&ids)
                .map(|((g, &k), tid)| {
                    let b = dets[k].bbox;
                    (g.class, (g.x * 100.0).round() / 100.0, (g.z * 100.0).round() / 100.0, *tid, (b[0] + b[2]) / 2.0, b[3])
                })
                .collect();
            r.detections = dets;
            r.analysed.push(Instant::now());
            r.analysed.retain(|t| t.elapsed() < Duration::from_secs(2));
        }

        // Fit the next parked vehicle that's due, with every camera that sees it right now.
        if fitter.idle() {
            let now = (Instant::now() - epoch).as_secs_f64();
            for (tid, class, x, z) in tracker.needs_fit(now) {
                let views: Vec<FitView> = readers
                    .values()
                    .filter_map(|r| {
                        let (frame, seen) = r.last.as_ref()?;
                        if frame.at.elapsed() > Duration::from_secs(1) || r.spec.calibration.is_none() {
                            return None;
                        }
                        let (_, bbox, _) = seen.iter().find(|(t, _, _)| tracker.resolve(*t) == tid)?;
                        Some(FitView { spec: r.spec.clone(), frame: frame.clone(), bbox: *bbox })
                    })
                    .collect();
                if !views.is_empty() {
                    tracker.fit_started(tid, now);
                    fitter.send(FitJob { track: tid, class, start: (x, z), views });
                    break;
                }
            }
        }
        // Ask the describer what the next unjudged "vehicle" really is (car, truck, bin, other).
        if describer.idle() {
            let now = (Instant::now() - epoch).as_secs_f64();
            'ask: for tid in tracker.needs_verdict(now) {
                for r in readers.values() {
                    let Some((frame, seen)) = r.last.as_ref() else { continue };
                    if frame.at.elapsed() > Duration::from_secs(1) {
                        continue;
                    }
                    if let Some((_, bbox, _)) = seen.iter().find(|(t, _, _)| tracker.resolve(*t) == tid) {
                        if let Some(crop) = crop_box(frame, r.spec.aspect, *bbox, 0.15) {
                            tracker.verdict_asked(tid, now);
                            asked.insert(tid, (r.spec.id.clone(), *bbox));
                            describer.send((tid, crop));
                            break 'ask;
                        }
                    }
                }
            }
        }
        for (tid, v) in describer.results() {
            tracker.set_verdict(tid, v);
            // Not a vehicle (and it's standing still where it was seen): remember the spot.
            if let Some((cam, bbox)) = asked.remove(&tid) {
                if v[2] + v[3] > v[0] + v[1] && tracker.is_still(tid) {
                    if let Some(r) = readers.get_mut(&cam) {
                        r.not_vehicles.push(bbox);
                    }
                }
            }
        }
        for (tid, mut f) in fitter.results() {
            f.at = (Instant::now() - epoch).as_secs_f64();
            tracker.apply_fit(tid, f);
        }

        for (cam, got) in looks.results() {
            let now = (Instant::now() - epoch).as_secs_f64();
            if let Some(h) = harvest.as_mut() {
                h.looks(&cam, now, &got);
            }
            tracker.add_frame_looks(&cam, &got, now);
        }

        if last_publish.elapsed() >= Duration::from_millis(100) {
            last_publish = Instant::now();
            snap.fingerprints = looks.status();
            snap.fitting = fitter.status();
            snap.describing = describer.status();
            snap.fingerprints.stats = Some(tracker.stats(1));
            snap.fingerprints.people = Some(tracker.stats(0));
            snap.time = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
            snap.infer_ms = infer_ms;
            snap.objects = tracker.objects((Instant::now() - epoch).as_secs_f64());
            snap.cameras = readers
                .iter()
                .map(|(id, r)| {
                    let stale = r.analysed.last().is_none_or(|t| t.elapsed() > Duration::from_secs(2));
                    (
                        id.clone(),
                        CameraStatus {
                            fps: r.analysed.len() as f32 / 2.0,
                            error: r.error.lock().unwrap().clone(),
                            detections: if stale { Vec::new() } else { r.detections.clone() },
                            ground: if stale { Vec::new() } else { r.ground.clone() },
                        },
                    )
                })
                .collect();
            publish(snap);
        }
        if !worked {
            thread::sleep(Duration::from_millis(4));
        }
    }
    Ok(())
}

/// Bottom-centre of the box is where the object touches the ground. For vehicles that's the near
/// side, so push the point back by about half a car along the viewing direction.
///
/// Vehicle boxes are also sanity-checked against the scene: a "car" that is narrower than any car
/// on the ground, or that sits where the people-free background shows a fixed object (wheelie
/// bins, letterboxes, hedges), is dropped. A car cut off by the bottom of the frame has no visible
/// ground contact, so it is placed from its roofline instead.
/// How much nearer the camera a person's visible front is than their middle, metres.
const PERSON_DEPTH: f64 = 0.15;

/// People's heights outside this range (metres) are mismeasurements (a box cut short, a person
/// bending down), not people.
const STATURE: (f64, f64) = (0.6, 2.2);

/// Height (metres) of a person whose feet are at world point `foot` (x, ground height, z) and
/// whose head is at image point (u, v_top): how high that pixel's line of sight is above their
/// feet where it passes over them.
fn stature(m: &geometry::CameraModel, u: f64, v_top: f64, foot: [f64; 3]) -> Option<f64> {
    stature_any(m, u, v_top, foot).filter(|h| (STATURE.0..=STATURE.1).contains(h))
}

/// `stature` without the sanity range: how tall a box makes someone look.
fn stature_any(m: &geometry::CameraModel, u: f64, v_top: f64, foot: [f64; 3]) -> Option<f64> {
    let d = m.sight(u, v_top)?;
    let reach = (foot[0] - m.pose.x).hypot(foot[2] - m.pose.z);
    let t = reach / d[0].hypot(d[2]).max(1e-9);
    Some(m.pose.height + t * d[1] - foot[1])
}

/// A box that makes someone look under this fraction of their height is cut short by something
/// in front of them.
const CUT_SHORT: f64 = 0.85;

/// Where a person of height `h`, standing on ground at height `floor`, whose head is at image
/// point (u, v_top) stands (their middle, like `ground_detection`).
fn head_placement(m: &geometry::CameraModel, u: f64, v_top: f64, h: f64, floor: f64) -> Option<(f64, f64)> {
    let d = m.sight(u, v_top)?;
    let drop = floor + h - m.pose.height;
    if d[1] > -1e-3 || drop >= 0.0 {
        return None; // head at or above the camera: can't tell how far
    }
    let t = drop / d[1];
    let (dx, dz) = (d[0] * t, d[2] * t);
    let len = dx.hypot(dz);
    if !(0.3..=60.0).contains(&len) {
        return None;
    }
    Some((m.pose.x + dx + dx / len * PERSON_DEPTH, m.pose.z + dz + dz / len * PERSON_DEPTH))
}

/// Place a detection on the ground. `feet`: for a person, where the pose model sees their feet
/// (normalised image point) — the bottom of the box is only the bottom of what's visible.
fn ground_detection(camera: &str, d: &Detection, spec: &CamSpec, model: Option<&geometry::CameraModel>, feet: Option<(f32, f32)>) -> Option<GroundDetection> {
    let ground = |u: f64, v: f64| match model {
        Some(m) => m.ground_point(u, v, 60.0),
        None => geometry::ground_point(&spec.pose, spec.aspect, u, v, 60.0),
    };
    let b = [d.bbox[0] as f64, d.bbox[1] as f64, d.bbox[2] as f64, d.bbox[3] as f64];
    let u = (b[0] + b[2]) / 2.0;
    let vehicle = crate::detector::family(d.class) == 1;
    // MINASTIR_DEBUG_VEHICLES=1: say why a vehicle box doesn't make it onto the ground.
    let why = |reason: &str| {
        if vehicle && std::env::var("MINASTIR_DEBUG_VEHICLES").is_ok() {
            eprintln!("vehicle {camera} {} {:.2} box {:.3},{:.3},{:.3},{:.3}: {reason}", d.class, d.score, b[0], b[1], b[2], b[3]);
        }
    };
    let cut_bottom = b[3] > 0.97;
    let cut_side = b[0] < 0.01 || b[2] > 0.99;
    if vehicle {
        // Touching the side of the frame and reaching into its bottom quarter: right beside the
        // wall the camera hangs on, at an extreme angle. No car stands there; bins and boxes do,
        // squashed by the lens edge into car-like shapes.
        if cut_side && b[3] > 0.75 {
            why("touches the frame side, low down");
            return None;
        }
        // No veto from what the empty-scene picture showed behind the box: cars park in front of
        // garages, hedges and carports, exactly where the background is "wall" or "plant". The
        // describer judges each vehicle track from its own crop (car, truck, bin or neither).
        if !cut_side {
            if let (Some(l), Some(r)) = (ground(b[0], b[3]), ground(b[2], b[3])) {
                if (l.0 - r.0).hypot(l.1 - r.1) < 1.4 {
                    why(&format!("only {:.1} m wide on the ground", (l.0 - r.0).hypot(l.1 - r.1)));
                    return None;
                }
            }
        }
    }
    let depth = match d.class {
        "car" => 2.0,
        "truck" | "bus" => 2.8,
        "bicycle" | "motorcycle" => 0.6,
        _ => PERSON_DEPTH,
    };
    let (u, foot_v) = match feet {
        Some((fu, fv)) if d.class == "person" => (fu as f64, fv as f64),
        _ => (u, b[3]),
    };
    let roof = match d.class {
        "truck" | "bus" => 2.4,
        _ => 1.45,
    };
    let (mut x, mut z, push) = match (vehicle && cut_bottom, model) {
        // Top of the box ≈ the far edge of the roof; the car's centre is about a metre nearer.
        (true, Some(m)) => match m.point_at_height(u, b[1], roof, 60.0) {
            Some((x, z)) => (x, z, -1.0),
            None => {
                let (x, z) = ground(u, b[3])?;
                (x, z, depth)
            }
        },
        _ => {
            let Some((x, z)) = ground(u, foot_v) else {
                why("no ground point under it");
                return None;
            };
            // A parked car's lower half is often hidden by a nearer car, a hedge or a palm: the
            // box then ends above the wheels and its bottom lands far beyond the car — where the
            // box, as wide as it is, would make the car impossibly long. The box's width isn't
            // affected by what hides the top or bottom; so when the bottom makes the car too
            // wide to be one and the roof (the box top) gives a believable width, use the roof.
            // (Not the other way round: a box top cut by the frame or a tree lies too.)
            let span = |p: Option<(f64, f64)>, q: Option<(f64, f64)>| p.zip(q).map(|(p, q)| (p.0 - q.0).hypot(p.1 - q.1));
            let longest = match d.class {
                "truck" | "bus" => 7.5,
                _ => 6.0,
            };
            let roof_at = |u: f64| model.and_then(|m| m.point_at_height(u, b[1], roof, 60.0));
            let too_wide = vehicle && !cut_side && span(ground(b[0], b[3]), ground(b[2], b[3])).is_none_or(|w| w > longest);
            match (too_wide, roof_at(u), span(roof_at(b[0]), roof_at(b[2]))) {
                (true, Some((rx, rz)), Some(w)) if w <= longest && b[1] > 0.01 => {
                    why(&format!("bottom hidden ({:.1} m wide from it): placed from the roof", span(ground(b[0], b[3]), ground(b[2], b[3])).unwrap_or(f64::NAN)));
                    (rx, rz, -1.0)
                }
                _ => (x, z, depth),
            }
        }
    };
    let (dx, dz) = (x - spec.pose.x, z - spec.pose.z);
    let len = dx.hypot(dz).max(1e-6);
    x += dx / len * push;
    z += dz / len * push;
    let range = (x - spec.pose.x).hypot(z - spec.pose.z);
    if range > 45.0 {
        why(&format!("{range:.0} m away"));
        return None;
    }
    why(&format!("placed at ({x:.1}, {z:.1}), {range:.0} m"));
    Some(GroundDetection { camera: camera.to_string(), class: d.class, score: d.score, x, z, range, from: (spec.pose.x, spec.pose.z), prev: None })
}

/// How much the region of box `b` (normalised x0, y0, x1, y1) looks like the same region of the
/// camera's empty-scene background: zero-mean normalised correlation of their brightness, −1…1
/// (insensitive to overall brightness and contrast, so the time of day matters less). Near 1 =
/// what's there is part of the scene itself (a plant, a letterbox), not something in front of it.
pub fn background_match(frame: &image::RgbImage, background: &image::RgbImage, b: [f64; 4]) -> f64 {
    background_match_raw((frame.as_raw(), frame.width() as usize, frame.height() as usize), (background.as_raw(), background.width() as usize, background.height() as usize), b)
}

/// `background_match` on raw RGB buffers (pixels, width, height).
pub fn background_match_raw(a: (&[u8], usize, usize), bg: (&[u8], usize, usize), b: [f64; 4]) -> f64 {
    const GW: usize = 24;
    const GH: usize = 48;
    let grey = |img: (&[u8], usize, usize), u: f64, v: f64| {
        let (px, w, h) = img;
        let (x, y) = (((u * w as f64) as usize).min(w - 1), ((v * h as f64) as usize).min(h - 1));
        let i = (y * w + x) * 3;
        0.299 * px[i] as f64 + 0.587 * px[i + 1] as f64 + 0.114 * px[i + 2] as f64
    };
    let (mut xa, mut xb) = (Vec::with_capacity(GW * GH), Vec::with_capacity(GW * GH));
    for gy in 0..GH {
        for gx in 0..GW {
            let u = b[0] + (b[2] - b[0]) * (gx as f64 + 0.5) / GW as f64;
            let v = b[1] + (b[3] - b[1]) * (gy as f64 + 0.5) / GH as f64;
            xa.push(grey(a, u, v));
            xb.push(grey(bg, u, v));
        }
    }
    let n = xa.len() as f64;
    let (ma, mb) = (xa.iter().sum::<f64>() / n, xb.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (p, q) in xa.iter().zip(&xb) {
        let (da, db) = (p - ma, q - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa < 1e-6 || sbb < 1e-6 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

/// Width frames are decoded at for a stream of this shape: pick a stream at least this wide, or
/// the "sharp" frame is only an enlargement.
pub fn decode_width(aspect: f64) -> u32 {
    (INPUT * SHARP * tiles_for(aspect)) as u32
}

/// Panoramas are decoded 2:1 and split into two square tiles so the detector sees undistorted
/// shapes; everything else is one 640×640 tile.
fn tiles_for(aspect: f64) -> usize {
    if aspect >= 1.9 {
        2
    } else {
        1
    }
}

/// One detector tile (INPUT×INPUT RGB) from a sharp frame: the tile's square, averaged down by
/// `SHARP`.
fn tile_of(data: &[u8], tiles: usize, tile: usize) -> Vec<u8> {
    let stride = INPUT * SHARP * tiles * 3;
    let x0 = tile * INPUT * SHARP;
    let mut out = Vec::with_capacity(INPUT * INPUT * 3);
    let n = (SHARP * SHARP) as u32;
    for y in 0..INPUT {
        for x in 0..INPUT {
            let mut acc = [0u32; 3];
            for dy in 0..SHARP {
                let row = (y * SHARP + dy) * stride;
                for dx in 0..SHARP {
                    let i = row + (x0 + x * SHARP + dx) * 3;
                    acc[0] += data[i] as u32;
                    acc[1] += data[i + 1] as u32;
                    acc[2] += data[i + 2] as u32;
                }
            }
            out.extend(acc.iter().map(|a| ((a + n / 2) / n) as u8));
        }
    }
    out
}

/// A fingerprint job: track ids and their RGB crops (width, height, pixels).
type LookJob = Vec<(u64, usize, usize, Vec<u8>)>;
/// Fingerprints for a job: the camera, and (track id, fingerprint) pairs.
type LookResult = (String, Vec<(u64, Vec<f32>)>);

/// Crops of the given detections from the sharp frame, largest first, with the frame's squash
/// undone (frames are decoded to square tiles whatever the stream's shape).
fn crops_for(frame: &Frame, aspect: f64, picks: &[(u64, &Detection)]) -> LookJob {
    let (fw, fh) = ((INPUT * SHARP * frame.tiles) as f64, (INPUT * SHARP) as f64);
    let stretch = aspect / frame.tiles as f64; // true width / decoded width, per pixel
    // (track, pixel box, is a person)
    let mut boxes: Vec<(u64, [f64; 4], bool)> = picks
        .iter()
        .map(|(id, d)| {
            let b = d.bbox.map(|v| v as f64);
            let pad = 0.04 * (b[2] - b[0]).max(b[3] - b[1]);
            let person = crate::detector::family(d.class) == 0;
            (*id, [((b[0] - pad) * fw).max(0.0), ((b[1] - pad) * fh).max(0.0), ((b[2] + pad) * fw).min(fw), ((b[3] + pad) * fh).min(fh)], person)
        })
        // People are tall and narrow: judge them by height.
        .filter(|(_, b, person)| if *person { b[3] - b[1] >= LOOK_MIN_PX * 1.2 && b[2] - b[0] >= LOOK_MIN_PX * 0.35 } else { b[2] - b[0] >= LOOK_MIN_PX && b[3] - b[1] >= LOOK_MIN_PX })
        .collect();
    // People first (they move, and need their looks most), then the largest.
    let area = |b: &[f64; 4]| (b[2] - b[0]) * (b[3] - b[1]);
    boxes.sort_by(|a, b| b.2.cmp(&a.2).then(area(&b.1).total_cmp(&area(&a.1))));
    boxes.truncate(LOOKS_PER_FRAME);
    let stride = fw as usize * 3;
    boxes
        .into_iter()
        .map(|(id, b, _)| {
            let (x0, y0, w, h) = (b[0] as usize, b[1] as usize, (b[2] - b[0]) as usize, (b[3] - b[1]) as usize);
            let mut px = Vec::with_capacity(w * h * 3);
            for y in y0..y0 + h {
                px.extend_from_slice(&frame.data[y * stride + x0 * 3..y * stride + (x0 + w) * 3]);
            }
            let tw = ((w as f64 * stretch).round() as usize).max(1);
            if tw != w {
                let img = image::RgbImage::from_raw(w as u32, h as u32, px).expect("crop size");
                let img = image::imageops::resize(&img, tw as u32, h as u32, image::imageops::FilterType::Triangle);
                (id, tw, h, img.into_raw())
            } else {
                (id, w, h, px)
            }
        })
        .collect()
}

/// A box (normalised) cut from a sharp frame with `pad` (fraction of its size) around it, in its
/// true shape: (width, height, RGB). None if it's tiny.
fn crop_box(frame: &Frame, aspect: f64, b: [f32; 4], pad: f64) -> Option<(usize, usize, Vec<u8>)> {
    let (fw, fh) = ((INPUT * SHARP * frame.tiles) as f64, (INPUT * SHARP) as f64);
    let b = b.map(|v| v as f64);
    let (pw, ph) = ((b[2] - b[0]) * pad, (b[3] - b[1]) * pad);
    let (x0, y0) = (((b[0] - pw) * fw).max(0.0) as usize, ((b[1] - ph) * fh).max(0.0) as usize);
    let (x1, y1) = (((b[2] + pw) * fw).min(fw) as usize, ((b[3] + ph) * fh).min(fh) as usize);
    let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
    if w < 16 || h < 16 {
        return None;
    }
    let stride = fw as usize * 3;
    let mut px = Vec::with_capacity(w * h * 3);
    for y in y0..y1 {
        px.extend_from_slice(&frame.data[y * stride + x0 * 3..y * stride + x1 * 3]);
    }
    let tw = ((w as f64 * aspect / frame.tiles as f64).round() as usize).max(1);
    let img = image::RgbImage::from_raw(w as u32, h as u32, px)?;
    let img = image::imageops::resize(&img, tw as u32, h as u32, image::imageops::FilterType::Triangle);
    Some((tw, h, img.into_raw()))
}

const VERDICT_QUESTION: &str = "What is the main object in this picture? Answer with one word: car, truck, bin, or other.";

/// The describer (vision-language model) on its own thread: one crop at a time → probabilities
/// of car / truck / bin / other.
struct DescribeWorker {
    jobs: mpsc::SyncSender<(u64, (usize, usize, Vec<u8>))>,
    results: mpsc::Receiver<(u64, [f32; 4])>,
    status: Arc<Mutex<FitStatus>>,
    busy: Arc<AtomicBool>,
}

impl DescribeWorker {
    fn spawn(models: PathBuf) -> Self {
        let (jobs, job_rx) = mpsc::sync_channel::<(u64, (usize, usize, Vec<u8>))>(1);
        let (res_tx, results) = mpsc::channel();
        let status = Arc::new(Mutex::new(FitStatus { state: "loading", ..Default::default() }));
        let busy = Arc::new(AtomicBool::new(true));
        let (st, bz) = (status.clone(), busy.clone());
        thread::Builder::new()
            .name("describer".into())
            .spawn(move || {
                let mut d = match Describer::load(&models, Device::Cpu) {
                    Ok(d) => d,
                    Err(e) => {
                        let mut s = st.lock().unwrap();
                        s.state = "off";
                        s.message = format!("describer not available: {e:#}");
                        return;
                    }
                };
                st.lock().unwrap().state = "running";
                bz.store(false, Ordering::Relaxed);
                for (tid, (w, h, px)) in job_rx {
                    let t = Instant::now();
                    match d.choose(&px, w, h, VERDICT_QUESTION, &crate::tracker::VERDICT) {
                        Ok(p) if p.len() == 4 => {
                            if res_tx.send((tid, [p[0], p[1], p[2], p[3]])).is_err() {
                                return;
                            }
                        }
                        Ok(_) => {}
                        Err(e) => st.lock().unwrap().message = format!("{e:#}"),
                    }
                    {
                        let mut s = st.lock().unwrap();
                        s.done += 1;
                        s.last_ms = t.elapsed().as_secs_f32() * 1000.0;
                    }
                    bz.store(false, Ordering::Relaxed);
                }
            })
            .expect("spawn describer");
        Self { jobs, results, status, busy }
    }

    fn idle(&self) -> bool {
        !self.busy.load(Ordering::Relaxed)
    }

    fn send(&self, job: (u64, (usize, usize, Vec<u8>))) {
        self.busy.store(true, Ordering::Relaxed);
        if self.jobs.try_send(job).is_err() {
            self.busy.store(false, Ordering::Relaxed);
        }
    }

    fn results(&self) -> Vec<(u64, [f32; 4])> {
        self.results.try_iter().collect()
    }

    fn status(&self) -> FitStatus {
        self.status.lock().unwrap().clone()
    }
}

/// Training-data log for learning association (MINASTIR_HARVEST=<file.jsonl>): one line per
/// analysed camera frame — every box with its outfit signature, image-track and 3D-track ids and
/// ground placement — plus appearance fingerprints (at most one per track every 2 s, to keep the
/// file manageable). Raw observations only: nothing here is edited by later decisions.
/// The learned "same person?" model for the image trackers, if one has been trained
/// (`assoc_train`). MINASTIR_ASSOC=off uses the hand-set rule instead, for comparison.
fn load_assoc(models: &std::path::Path) -> Option<Arc<crate::assoc::Model>> {
    if std::env::var("MINASTIR_ASSOC").is_ok_and(|v| v == "off") {
        eprintln!("association: hand-set rule (MINASTIR_ASSOC=off)");
        return None;
    }
    let path = models.join(crate::assoc::PEOPLE_MODEL_FILE);
    if !path.exists() {
        return None;
    }
    match crate::assoc::Model::load(&path) {
        Ok(m) => {
            eprintln!("association: learned model {}", path.display());
            Some(Arc::new(m))
        }
        Err(e) => {
            eprintln!("association: can't use {} ({e:#}); hand-set rule instead", path.display());
            None
        }
    }
}

struct Harvest {
    out: std::io::BufWriter<std::fs::File>,
    last_look: HashMap<u64, f64>,
}

impl Harvest {
    fn open() -> Option<Self> {
        let path = std::env::var("MINASTIR_HARVEST").ok()?;
        match std::fs::File::create(&path) {
            Ok(f) => Some(Self { out: std::io::BufWriter::new(f), last_look: HashMap::new() }),
            Err(e) => {
                eprintln!("harvest: can't write {path}: {e}");
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn frame(&mut self, cam: &str, t: f64, from: (f64, f64), boxes: &[ImageBox], uids: &[u64], gids: &[Option<u64>], ground: &[GroundDetection], scores: &[f32]) {
        use std::io::Write;
        let r3 = |v: f32| (v * 1000.0).round() / 1000.0;
        let items: Vec<serde_json::Value> = boxes
            .iter()
            .enumerate()
            .map(|(k, b)| {
                serde_json::json!({
                    "fam": b.family,
                    "box": b.bbox.map(|v| (v * 10000.0).round() / 10000.0),
                    "score": r3(scores.get(k).copied().unwrap_or(0.0)),
                    "outfit": b.outfit.iter().map(|v| r3(*v)).collect::<Vec<_>>(),
                    "uid": uids.get(k).copied().unwrap_or(0),
                    "gid": gids.get(k).copied().flatten(),
                    "x": ground.get(k).map(|g| (g.x * 100.0).round() / 100.0),
                    "z": ground.get(k).map(|g| (g.z * 100.0).round() / 100.0),
                    "range": ground.get(k).map(|g| (g.range * 100.0).round() / 100.0),
                    "h": b.stature.map(|h| (h * 100.0).round() / 100.0),
                })
            })
            .collect();
        let line = serde_json::json!({ "t": (t * 1000.0).round() / 1000.0, "cam": cam, "from": [from.0, from.1], "boxes": items });
        let _ = writeln!(self.out, "{line}");
        let _ = self.out.flush();
    }

    fn looks(&mut self, cam: &str, t: f64, got: &[(u64, Vec<f32>)]) {
        use std::io::Write;
        for (gid, f) in got {
            if self.last_look.get(gid).is_some_and(|&at| t - at < 2.0) {
                continue;
            }
            self.last_look.insert(*gid, t);
            let fp: Vec<f32> = f.iter().map(|v| (v * 1000.0).round() / 1000.0).collect();
            let line = serde_json::json!({ "t": (t * 1000.0).round() / 1000.0, "cam": cam, "look": gid, "fp": fp });
            let _ = writeln!(self.out, "{line}");
        }
        let _ = self.out.flush();
    }
}

/// One camera's view of a vehicle to fit: its sharp frame and the vehicle's box (normalised).
struct FitView {
    spec: CamSpec,
    frame: Arc<Frame>,
    bbox: [f32; 4],
}

struct FitJob {
    track: u64,
    class: &'static str,
    start: (f64, f64),
    views: Vec<FitView>,
}

/// Render-and-compare on its own thread: outlines each view with Segment Anything, fits a 3D
/// vehicle to all of them, returns the pose. One job at a time.
struct FitWorker {
    jobs: mpsc::SyncSender<FitJob>,
    results: mpsc::Receiver<(u64, Fitted)>,
    status: Arc<Mutex<FitStatus>>,
    busy: Arc<AtomicBool>,
}

impl FitWorker {
    fn spawn(models: PathBuf) -> Self {
        let (jobs, job_rx) = mpsc::sync_channel::<FitJob>(1);
        let (res_tx, results) = mpsc::channel();
        let status = Arc::new(Mutex::new(FitStatus { state: "loading", ..Default::default() }));
        let busy = Arc::new(AtomicBool::new(true));
        let (st, bz) = (status.clone(), busy.clone());
        thread::Builder::new()
            .name("fitting".into())
            .spawn(move || {
                let mut sam = match Sam::load(&models) {
                    Ok(s) => s,
                    Err(e) => {
                        let mut s = st.lock().unwrap();
                        s.state = "error";
                        s.message = format!("Segment Anything not available: {e:#}");
                        return;
                    }
                };
                // Wheels say which way a parked vehicle points; without the model, the outlines
                // alone decide (much less reliably at range).
                let mut finder = if std::env::var("MINASTIR_WHEELS").as_deref() == Ok("off") {
                    None
                } else {
                    if !wheels::Wheels::present(&models) {
                        st.lock().unwrap().message = "Downloading the wheel detector (720 MB, once)".into();
                    }
                    match wheels::fetch(&models).and_then(|_| wheels::Wheels::load(&models, Device::Cpu)) {
                        Ok(w) => {
                            st.lock().unwrap().message.clear();
                            Some(w)
                        }
                        Err(e) => {
                            st.lock().unwrap().message = format!("wheel detector not available: {e:#}");
                            None
                        }
                    }
                };
                st.lock().unwrap().state = "running";
                bz.store(false, Ordering::Relaxed);
                let mut key = 0u64;
                for job in job_rx {
                    let t = Instant::now();
                    match fit_job(&mut sam, finder.as_mut(), &job, &mut key) {
                        Ok(Some(f)) => {
                            if res_tx.send((job.track, f)).is_err() {
                                return;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => st.lock().unwrap().message = format!("{e:#}"),
                    }
                    {
                        let mut s = st.lock().unwrap();
                        s.done += 1;
                        s.last_ms = t.elapsed().as_secs_f32() * 1000.0;
                    }
                    bz.store(false, Ordering::Relaxed);
                }
            })
            .expect("spawn fitting worker");
        Self { jobs, results, status, busy }
    }

    fn idle(&self) -> bool {
        !self.busy.load(Ordering::Relaxed)
    }

    fn send(&self, job: FitJob) {
        self.busy.store(true, Ordering::Relaxed);
        if self.jobs.try_send(job).is_err() {
            self.busy.store(false, Ordering::Relaxed);
        }
    }

    fn results(&self) -> Vec<(u64, Fitted)> {
        self.results.try_iter().collect()
    }

    fn status(&self) -> FitStatus {
        self.status.lock().unwrap().clone()
    }
}

/// How far the outlines may turn a vehicle from its wheels' axis, degrees (the wheels' own
/// error: contact points are box bottoms, a few pixels each at 20 m).
const WHEEL_AXIS_TOL: f64 = 8.0;

/// Outline the vehicle in each view (and find its wheels), and fit one 3D vehicle to all of it.
fn fit_job(sam: &mut Sam, mut finder: Option<&mut wheels::Wheels>, job: &FitJob, key: &mut u64) -> Result<Option<Fitted>> {
    let shape = shape_for(job.class);
    let mut parts = Vec::new(); // (camera model, frame size, mask, ground height)
    let mut axes = Vec::new();
    for v in &job.views {
        let Some(model) = v.spec.model() else { continue };
        let (fw, fh) = ((INPUT * SHARP * v.frame.tiles) as f64, (INPUT * SHARP) as f64);
        let b = v.bbox.map(|x| x as f64);
        let bpx = [b[0] * fw, b[1] * fh, b[2] * fw, b[3] * fh];
        if bpx[2] - bpx[0] < 24.0 || bpx[3] - bpx[1] < 16.0 {
            continue;
        }
        // A window around the box (Segment Anything sees only this).
        let (cx, cy, hw, hh) = ((bpx[0] + bpx[2]) / 2.0, (bpx[1] + bpx[3]) / 2.0, (bpx[2] - bpx[0]), (bpx[3] - bpx[1]));
        let (x0, y0) = ((cx - hw).max(0.0).floor(), (cy - hh).max(0.0).floor());
        let (x1, y1) = ((cx + hw).min(fw).ceil(), (cy + hh).min(fh).ceil());
        let (ww, wh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let stride = fw as usize * 3;
        let mut px = Vec::with_capacity(ww * wh * 3);
        for y in y0 as usize..y0 as usize + wh {
            px.extend_from_slice(&v.frame.data[y * stride + x0 as usize * 3..y * stride + (x0 as usize + ww) * 3]);
        }
        let crop = image::RgbImage::from_raw(ww as u32, wh as u32, px).context("crop")?;
        *key += 1;
        let mut mask = sam.box_mask(&crop, *key, [bpx[0] - x0, bpx[1] - y0, bpx[2] - x0, bpx[3] - y0])?;
        mask.x0 += x0;
        mask.y0 += y0;
        let base = model.point_at((b[0] + b[2]) / 2.0, b[3]).map_or(0.0, |p| p[2]);
        if let Some(w) = finder.as_deref_mut() {
            let found = w.detect((&v.frame.data, fw as usize, fh as usize), v.bbox)?;
            if let Some(a) = wheels::axis(&model, &found, v.bbox, shape.length, fw / fh) {
                axes.push(a);
            }
        }
        parts.push((model, (fw, fh), mask, base));
    }
    if parts.is_empty() {
        return Ok(None);
    }
    let views: Vec<View> = parts.iter().map(|(m, size, mask, base)| View::new(m, *size, mask).on_ground(*base)).collect();
    // The wheels' axis, when any view shows two wheels on one side, steers the heading.
    let hint = wheels::combine(&axes).map(|a| (a, WHEEL_AXIS_TOL));
    let f = fit_guided(&views, &shape, job.start, 3.0, hint);
    Ok(Some(Fitted { x: f.pose.x, z: f.pose.z, heading: f.pose.heading, axis_only: f.axis_only, score: f.score, at: 0.0 }))
}

/// The appearance model on its own thread: takes crop jobs when idle (never queues), returns
/// fingerprints.
struct LookWorker {
    jobs: mpsc::SyncSender<(String, LookJob)>,
    results: mpsc::Receiver<LookResult>,
    status: Arc<Mutex<FingerprintStatus>>,
    ready: Arc<AtomicBool>,
    done: Mutex<Vec<Instant>>,
}

impl LookWorker {
    fn spawn(models: PathBuf) -> Self {
        let (jobs, job_rx) = mpsc::sync_channel::<(String, LookJob)>(1);
        let (res_tx, results) = mpsc::channel();
        let status = Arc::new(Mutex::new(FingerprintStatus { state: "loading", ..Default::default() }));
        let ready = Arc::new(AtomicBool::new(false));
        let (st, rd) = (status.clone(), ready.clone());
        thread::Builder::new()
            .name("fingerprints".into())
            .spawn(move || {
                let set = |state: &'static str, msg: String| {
                    let mut s = st.lock().unwrap();
                    s.state = state;
                    s.message = msg;
                };
                let model = models.join(appearance::DINO3_MODEL_FILE);
                if !model.exists() {
                    set("downloading", "Downloading the appearance model (343 MB, once)".into());
                    let dir = model.parent().unwrap().to_path_buf();
                    let got = std::fs::create_dir_all(&dir).map_err(anyhow::Error::from).and_then(|_| {
                        download(&format!("{}model.onnx_data", appearance::DINO3_MODEL_URL), &dir.join("model.onnx_data"))?;
                        download(&format!("{}model.onnx", appearance::DINO3_MODEL_URL), &model)
                    });
                    if let Err(e) = got {
                        set("error", format!("download failed: {e:#}"));
                        return;
                    }
                }
                set("loading", "Loading the appearance model".into());
                let mut emb = match Embedder::load_best(&models, Device::Cpu) {
                    Ok(e) => e,
                    Err(e) => {
                        set("error", format!("{e:#}"));
                        return;
                    }
                };
                st.lock().unwrap().model = format!("{:?}", emb.kind());
                set("running", String::new());
                rd.store(true, Ordering::Relaxed);
                let mut ms = 0f32;
                for (cam, job) in job_rx {
                    let t = Instant::now();
                    let crops: Vec<Crop> = job.iter().map(|(_, w, h, px)| Crop { rgb: px, width: *w, height: *h }).collect();
                    match emb.embed(&crops) {
                        Ok(f) => {
                            let per = t.elapsed().as_secs_f32() * 1000.0 / job.len().max(1) as f32;
                            ms = if ms == 0.0 { per } else { 0.9 * ms + 0.1 * per };
                            st.lock().unwrap().crop_ms = ms;
                            if res_tx.send((cam, job.iter().map(|j| j.0).zip(f).collect())).is_err() {
                                return;
                            }
                        }
                        Err(e) => set("error", format!("{e:#}")),
                    }
                }
            })
            .expect("spawn fingerprint worker");
        Self { jobs, results, status, ready, done: Mutex::new(Vec::new()) }
    }

    fn ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Hand over a job if the worker is free.
    fn send(&self, camera: String, job: LookJob) -> bool {
        self.jobs.try_send((camera, job)).is_ok()
    }

    fn results(&self) -> Vec<LookResult> {
        let got: Vec<LookResult> = self.results.try_iter().collect();
        let mut done = self.done.lock().unwrap();
        done.extend(got.iter().flat_map(|g| g.1.iter()).map(|_| Instant::now()));
        done.retain(|t| t.elapsed() < Duration::from_secs(5));
        got
    }

    fn status(&self) -> FingerprintStatus {
        let mut s = self.status.lock().unwrap().clone();
        s.per_sec = self.done.lock().unwrap().len() as f32 / 5.0;
        s
    }
}

fn start_reader(spec: CamSpec, fps: f32) -> Reader {
    let slot = Arc::new(Mutex::new(None));
    let error = Arc::new(Mutex::new(None));
    let stop = Arc::new(AtomicBool::new(false));
    let (slot2, error2, stop2, spec2) = (slot.clone(), error.clone(), stop.clone(), spec.clone());
    thread::Builder::new()
        .name(format!("live-{}", spec.id))
        .spawn(move || {
            let mut backoff = Duration::from_secs(2);
            while !stop2.load(Ordering::Relaxed) {
                let started = Instant::now();
                let res = read_stream(&spec2, fps, &slot2, &stop2);
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                *error2.lock().unwrap() = Some(match res {
                    Ok(()) => "stream ended".into(),
                    Err(e) => minastir_core::capture::redact(&format!("{e:#}")),
                });
                if started.elapsed() > Duration::from_secs(30) {
                    backoff = Duration::from_secs(2);
                }
                // Sleep in small steps so a stop request is honoured quickly.
                let until = Instant::now() + backoff;
                while Instant::now() < until && !stop2.load(Ordering::Relaxed) {
                    thread::sleep(Duration::from_millis(200));
                }
                backoff = (backoff * 2).min(Duration::from_secs(30));
            }
        })
        .expect("spawn reader");
    Reader { model: spec.model(), spec, slot, error, stop, analysed: Vec::new(), detections: Vec::new(), ground: Vec::new(), last_look: None, last: None, image: ImageTracker::default(), not_vehicles: Vec::new() }
}

struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn read_stream(spec: &CamSpec, fps: f32, slot: &Mutex<Option<Frame>>, stop: &AtomicBool) -> Result<()> {
    let tiles = tiles_for(spec.aspect);
    let (w, h) = (INPUT * SHARP * tiles, INPUT * SHARP);
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay"]);
    if cfg!(target_os = "macos") {
        cmd.args(["-hwaccel", "videotoolbox"]); // decode on the media engine, not the CPU
    }
    if spec.source.starts_with("rtsp") {
        cmd.args(["-rtsp_transport", "tcp", "-timeout", "10000000"]);
    } else {
        cmd.args(["-re", "-stream_loop", "-1"]); // files: play in real time, loop (demos/tests)
    }
    cmd.args(["-i", &spec.source, "-an"])
        .args(["-vf", &format!("fps={fps},scale={w}:{h}")])
        .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = KillOnDrop(cmd.spawn().context("failed to run ffmpeg")?);
    let mut out = child.0.stdout.take().context("no ffmpeg stdout")?;
    let frame_len = w * h * 3;
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut buf = vec![0u8; frame_len];
        if let Err(e) = out.read_exact(&mut buf) {
            let mut err = String::new();
            if let Some(mut s) = child.0.stderr.take() {
                let _ = s.read_to_string(&mut err);
            }
            let err = err.trim();
            if err.is_empty() {
                bail!("stream closed ({e})");
            }
            bail!("{}", err.lines().last().unwrap_or(err));
        }
        *slot.lock().unwrap() = Some(Frame { data: buf, tiles, at: Instant::now() });
    }
}

pub fn download(url: &str, dest: &Path) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = dest.with_extension("part");
    let mut resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?
        .get(url)
        .send()?
        .error_for_status()
        .context("downloading the detection model")?;
    let mut file = std::fs::File::create(&tmp)?;
    resp.copy_to(&mut file)?;
    std::fs::rename(&tmp, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiles_split_and_average_down() {
        // Sharp frame, 2 tiles: tile 1 starts at x = INPUT*SHARP; each output pixel is the mean
        // of a SHARP×SHARP block.
        let (w, h) = (INPUT * SHARP * 2, INPUT * SHARP);
        let mut data = vec![0u8; w * h * 3];
        let at = |x: usize, y: usize| (y * w + x) * 3;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            data[at(INPUT * SHARP + dx, dy)] = 8; // tile 1, output pixel (0, 0)
        }
        data[at(INPUT * SHARP + 2 * SHARP, 5 * SHARP)] = 40; // tile 1, output (2, 5): one of four
        let t1 = tile_of(&data, 2, 1);
        assert_eq!(t1.len(), INPUT * INPUT * 3);
        assert_eq!(t1[0], 8);
        assert_eq!(t1[(5 * INPUT + 2) * 3], 10);
        assert_eq!(tile_of(&data, 2, 0)[0], 0);
    }
}
