//! Minastir server: JSON API over the core crate, plus static hosting of the web UI.
//!
//! Env:
//!   MINASTIR_ADDR     bind address        (default 127.0.0.1:8080 — no auth yet, keep it local)
//!   MINASTIR_DATA     data directory      (default ./data)
//!   MINASTIR_WEB_DIR  built UI directory  (default ./apps/web/dist)
//!   MINASTIR_REPLAY   replay to start with (a folder under <data>/replay/, or a path to one, of
//!                     <camera id>.mp4 files played instead of the cameras); switchable at
//!                     runtime through /api/source, which remembers the choice

use anyhow::{anyhow, Result};
use axum::{
    body::Bytes,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use minastir_core::{
    capture::{self, CaptureOpts},
    discovery, naming,
    onvif::{self, CameraInfo, SourceKind},
    stream,
};
use minastir_vision::{
    calibrate::{write_outputs, Calibrator},
    live::{self, CamSpec, LiveConfig},
    moge::Moge,
    onnx::Device,
    scenemap::LensMap,
    segment::Segmenter,
    survey::{survey, SurveyCam},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    net::SocketAddr,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{Mutex, RwLock};
use tower_http::services::{ServeDir, ServeFile};

const SNAPSHOT_TTL: Duration = Duration::from_secs(2);

#[derive(Clone, Default, Serialize, Deserialize)]
struct Creds {
    user: String,
    pass: String,
}

#[derive(Clone, Default, Serialize)]
struct Job {
    running: bool,
    message: String,
    done: usize,
    total: usize,
}

type SnapSlot = Arc<Mutex<Option<(Instant, Bytes)>>>;

/// The site plan: an optional map image (e.g. a satellite screenshot the user provides) placed
/// under the plan, and house outlines traced on it. Stored in `site.json`; the image itself in
/// `site_map.bin` (both in the data directory, never in the repository).
#[derive(Clone, Default, Serialize, Deserialize)]
struct Site {
    #[serde(default)]
    map: Option<MapImage>,
    #[serde(default)]
    houses: Vec<House>,
}

/// Placement of the map image: image pixel (u, v) → world (x, z) =
/// (x, z) + scale · R(rotation) · (u, v), with u along +x and v along +z at rotation 0.
#[derive(Clone, Serialize, Deserialize)]
struct MapImage {
    width: u32,
    height: u32,
    content_type: String,
    x: f64,
    z: f64,
    /// Metres per image pixel.
    scale: f64,
    /// Degrees, positive turns +u from +x toward +z.
    rotation: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct House {
    /// Outline corners, world (x, z).
    points: Vec<[f64; 2]>,
    /// Wall height, metres.
    height: f64,
}

struct App {
    data: PathBuf,
    cameras: RwLock<Vec<CameraInfo>>,
    /// Per-camera credentials (cameras from different vendors rarely share logins).
    creds: RwLock<HashMap<String, Creds>>,
    snaps: std::sync::Mutex<HashMap<String, SnapSlot>>,
    job: RwLock<Job>,
    /// Latest live-engine snapshot, pre-serialised (every WebSocket client gets the same bytes).
    live: tokio::sync::watch::Sender<Arc<str>>,
    /// Loaded per-lens scene maps, keyed by camera id, tagged with the calibration time.
    scene_maps: std::sync::Mutex<HashMap<String, (f64, Arc<Vec<LensMap>>)>>,
    site: RwLock<Site>,
    /// Segment Anything, loaded on first use (for outlining roofs in the site photo).
    sam: std::sync::Mutex<Option<minastir_vision::roof::Sam>>,
    /// Latest fitted street (from /world), used to face parked cars along the road / driveway.
    street: std::sync::Mutex<Option<minastir_vision::street::Street>>,
    /// Latest analysed frame of each camera (the footage the live engine is looking at).
    frames: live::FrameStore,
    /// Decoded empty-scene pictures, tagged with their file's modification time.
    backgrounds: std::sync::Mutex<HashMap<String, (std::time::SystemTime, Arc<image::RgbImage>)>>,
    /// Decoded site photo, tagged with its file's modification time.
    site_img: std::sync::Mutex<Option<(std::time::SystemTime, Arc<image::RgbImage>)>>,
    /// Test video the live engine plays instead of the cameras (a replay folder), or None for
    /// the cameras themselves.
    replay: std::sync::RwLock<Option<PathBuf>>,
}

/// What the live engine watches, as saved in <data>/source.json.
#[derive(Serialize, Deserialize, Default)]
struct SourceChoice {
    /// A folder name under <data>/replay, or None for the live cameras.
    replay: Option<String>,
}

type AppState = State<Arc<App>>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

type ApiResult<T> = Result<T, ApiError>;

#[tokio::main]
async fn main() -> Result<()> {
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let data = PathBuf::from(env("MINASTIR_DATA", "data"));
    let web = PathBuf::from(env("MINASTIR_WEB_DIR", "apps/web/dist"));
    let addr: SocketAddr = env("MINASTIR_ADDR", "127.0.0.1:8080").parse()?;
    std::fs::create_dir_all(&data)?;

    let site: Site = read_json(&data.join("site.json")).unwrap_or_default();
    // MINASTIR_REPLAY wins over the remembered choice.
    let replay = match std::env::var("MINASTIR_REPLAY") {
        Ok(p) if !p.is_empty() => {
            let path = PathBuf::from(&p);
            Some(if path.is_dir() { path } else { data.join("replay").join(p) })
        }
        _ => read_json::<SourceChoice>(&data.join("source.json")).and_then(|c| c.replay).map(|n| data.join("replay").join(n)),
    }
    .filter(|p| p.is_dir());
    let app = Arc::new(App {
        cameras: RwLock::new(read_json(&data.join("cameras.json")).unwrap_or_default()),
        creds: RwLock::new(read_json(&data.join("credentials.json")).unwrap_or_default()),
        data,
        snaps: Default::default(),
        job: Default::default(),
        scene_maps: Default::default(),
        site: RwLock::new(site),
        sam: Default::default(),
        street: Default::default(),
        site_img: Default::default(),
        frames: Default::default(),
        backgrounds: Default::default(),
        replay: std::sync::RwLock::new(replay),
        live: tokio::sync::watch::channel(Arc::from("{\"state\":\"off\",\"objects\":[],\"cameras\":{}}")).0,
    });

    // Live detection/tracking runs here in the server, always — the UI only renders it.
    let _live = (env("MINASTIR_LIVE", "1") != "0").then(|| start_live(app.clone()));
    // Fit the street once at startup, so parked cars face the right way before anyone opens the UI.
    {
        let app = app.clone();
        tokio::spawn(async move {
            let _ = world(State(app)).await;
        });
    }

    let api = Router::new()
        .route("/status", get(status))
        .route("/cameras", get(list_cameras))
        .route("/discover", get(discover))
        .route("/adopt", post(adopt))
        .route("/streams", post(add_stream))
        .route("/cameras/{id}", axum::routing::delete(remove_camera).patch(update_camera))
        .route("/cameras/{id}/refresh", post(refresh_camera))
        .route("/capture", post(start_capture))
        .route("/live", get(live_ws))
        .route("/source", get(get_source).put(put_source))
        .route("/live/{id}/frame.jpg", get(live_frame))
        .route("/calibrate", post(start_calibrate))
        .route("/cameras/{id}/calibration.jpg", get(calibration_image))
        .route("/cameras/{id}/scene", get(scene_points))
        .route("/cameras/{id}/groundmap", get(ground_map))
        .route("/world", get(world))
        .route("/site", get(get_site).put(put_site))
        .route("/site/map/align", post(auto_align_site_map))
        .route("/site/roof", post(find_roof))
        .route(
            "/site/map",
            get(get_site_map)
                .post(upload_site_map)
                .delete(delete_site_map)
                .layer(axum::extract::DefaultBodyLimit::max(40 * 1024 * 1024)),
        )
        .route("/cameras/{id}/snapshot", get(snapshot))
        .route("/cameras/{id}/background", get(background))
        .fallback(|| async { ApiError(StatusCode::NOT_FOUND, "no such endpoint".into()) })
        .with_state(app);

    let spa = ServeDir::new(&web).fallback(ServeFile::new(web.join("index.html")));
    let router = Router::new().nest("/api", api).fallback_service(spa);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("minastir listening on http://{addr}");
    if !web.join("index.html").exists() {
        println!("  (UI not built at {} — run `npm run build`, or use `npm run dev`)", web.display());
    }
    axum::serve(listener, router).await?;
    Ok(())
}

// ---------- handlers ----------

#[derive(Serialize)]
struct CameraView<'a> {
    id: String,
    label: String,
    model_label: String,
    detected_name: Option<String>,
    has_background: bool,
    #[serde(flatten)]
    info: &'a CameraInfo,
}

fn views<'a>(app: &App, cams: &'a [CameraInfo]) -> Vec<CameraView<'a>> {
    cams.iter()
        .map(|c| CameraView {
            id: c.id(),
            label: c.label(),
            model_label: c.model_label(),
            detected_name: c.detected_name(),
            has_background: background_path(&app.data, &c.id()).exists(),
            info: c,
        })
        .collect()
}

async fn status(State(app): AppState) -> Json<Value> {
    let cams = app.cameras.read().await;
    let captured = cams.iter().filter(|c| background_path(&app.data, &c.id()).exists()).count();
    Json(json!({ "cameras": cams.len(), "captured": captured, "job": *app.job.read().await }))
}

async fn list_cameras(State(app): AppState) -> Json<Value> {
    let cams = app.cameras.read().await;
    Json(json!(views(&app, &cams)))
}

#[derive(Serialize)]
struct DiscoveredView {
    endpoint: String,
    ip: String,
    device_url: String,
    name: Option<String>,
    hardware: Option<String>,
    /// Non-default name from discovery scopes or reverse DNS, if any.
    display_name: Option<String>,
    /// Id of the inventory camera at this address, if it's already adopted.
    adopted_id: Option<String>,
}

/// Multicast WS-Discovery — no credentials needed.
async fn discover(State(app): AppState) -> ApiResult<Json<Vec<DiscoveredView>>> {
    let found = tokio::task::spawn_blocking(|| -> Result<_> {
        let found = discovery::discover(Duration::from_secs(3))?;
        // Reverse DNS in parallel — a missing PTR record can take a while to time out.
        let dns: Vec<Option<String>> = std::thread::scope(|s| {
            let handles: Vec<_> = found.iter().map(|d| s.spawn(|| naming::reverse_dns(&d.ip))).collect();
            handles.into_iter().map(|h| h.join().ok().flatten()).collect()
        });
        Ok(found.into_iter().zip(dns).collect::<Vec<_>>())
    })
    .await
    .map_err(|e| anyhow!(e))??;
    let cams = app.cameras.read().await;
    Ok(Json(
        found
            .into_iter()
            .filter_map(|(d, dns)| {
                let url = d.device_url()?.to_string();
                let (scope_name, hardware) = (d.scope_value("name"), d.scope_value("hardware"));
                let hints: Vec<&str> = hardware.as_deref().into_iter().collect();
                let display_name = naming::pick(scope_name.as_deref().into_iter().chain(dns.as_deref()), &hints);
                Some(DiscoveredView {
                    display_name,
                    adopted_id: cams.iter().find(|c| c.is_device(&url)).map(|c| c.id()),
                    name: scope_name,
                    hardware,
                    endpoint: d.endpoint,
                    ip: d.ip,
                    device_url: url,
                })
            })
            .collect(),
    ))
}

#[derive(Deserialize)]
struct AdoptReq {
    /// Device service URL from discovery…
    device_url: Option<String>,
    /// …or a manually entered "ip" / "ip:port".
    host: Option<String>,
    user: String,
    pass: String,
}

/// Log in to a camera, read its profiles, and add it (or refresh it) in the inventory.
async fn adopt(State(app): AppState, Json(req): Json<AdoptReq>) -> ApiResult<Json<Value>> {
    let url = match (req.device_url.as_deref(), req.host.as_deref().map(str::trim)) {
        (Some(u), _) if !u.is_empty() => u.to_string(),
        (_, Some(h)) if !h.is_empty() => format!("http://{h}/onvif/device_service"),
        _ => return Err(ApiError(StatusCode::BAD_REQUEST, "device_url or host is required".into())),
    };
    let ip = onvif::host_port(&url)
        .map(|(h, _)| h)
        .ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, format!("invalid address {url}")))?;

    let (user, pass) = (req.user.clone(), req.pass.clone());
    let cam = tokio::task::spawn_blocking(move || -> Result<CameraInfo> {
        onvif::Client::new(&user, &pass)?.probe(&ip, &url)
    })
    .await
    .map_err(|e| anyhow!(e))?
    .map_err(|e| {
        let msg = format!("{e:#}");
        if msg.contains("authentication rejected") {
            ApiError(StatusCode::UNAUTHORIZED, "Wrong username or password".into())
        } else {
            ApiError(StatusCode::BAD_GATEWAY, msg)
        }
    })?;

    let cam = save_camera(&app, cam, Some(Creds { user: req.user, pass: req.pass })).await?;
    Ok(Json(json!(views(&app, std::slice::from_ref(&cam))[0])))
}

#[derive(Deserialize)]
struct StreamReq {
    url: String,
    #[serde(default)]
    name: Option<String>,
    /// Override / supply credentials not embedded in the URL.
    #[serde(default)]
    user: Option<String>,
    #[serde(default)]
    pass: Option<String>,
}

/// Add a camera by its RTSP/RTSPS URL (no ONVIF) — e.g. UniFi Protect's RTSPS output.
async fn add_stream(State(app): AppState, Json(req): Json<StreamReq>) -> ApiResult<Json<Value>> {
    let parsed = stream::parse(&req.url).map_err(|e| ApiError(StatusCode::BAD_REQUEST, format!("{e:#}")))?;
    let user = req.user.filter(|u| !u.is_empty()).unwrap_or(parsed.user);
    let pass = req.pass.filter(|_| !user.is_empty()).unwrap_or(parsed.pass);
    let (url, u, p) = (parsed.url.clone(), user.clone(), pass.clone());
    let info = tokio::task::spawn_blocking(move || stream::probe(&onvif::with_credentials(&url, &u, &p)))
        .await
        .map_err(|e| anyhow!(e))?
        .map_err(|e| {
            let msg = format!("{e:#}");
            if msg.contains("authentication rejected") {
                ApiError(StatusCode::UNAUTHORIZED, "The stream needs a username and password".into())
            } else {
                ApiError(StatusCode::BAD_GATEWAY, msg)
            }
        })?;
    let creds = (!user.is_empty()).then_some(Creds { user, pass });
    let mut cam = save_camera(&app, stream::camera(&parsed.url, info), creds).await?;
    // A name typed when adding wins over one kept from an earlier add of the same URL.
    if let Some(n) = req.name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty()) {
        let mut cams = app.cameras.write().await;
        if let Some(c) = cams.iter_mut().find(|c| c.id() == cam.id()) {
            c.name = Some(n);
            cam = c.clone();
        }
        persist(&app, &cams, &*app.creds.read().await)?;
    }
    Ok(Json(json!(views(&app, std::slice::from_ref(&cam))[0])))
}

/// Insert/replace a probed camera, keeping the user's name for it. Returns the stored camera.
async fn save_camera(app: &App, mut cam: CameraInfo, new_creds: Option<Creds>) -> Result<CameraInfo> {
    let id = cam.id();
    let mut cams = app.cameras.write().await;
    let mut creds = app.creds.write().await;
    let same = |c: &CameraInfo| c.id() == id || (cam.kind == SourceKind::Onvif && c.is_device(&cam.device_url));
    if let Some(prev) = cams.iter().find(|c| same(c)) {
        cam.name = prev.name.clone();
        cam.pose = prev.pose;
        cam.calibration = prev.calibration.clone();
    }
    if let Some(c) = new_creds {
        creds.insert(id.clone(), c);
    }
    cams.retain(|c| !same(c));
    cams.push(cam.clone());
    app.snaps.lock().unwrap().remove(&id);
    persist(app, &cams, &creds)?;
    Ok(cam)
}

/// Re-read an adopted camera with its saved credentials (picks up new names, streams, firmware).
async fn refresh_camera(State(app): AppState, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let (ip, url, kind) = {
        let cams = app.cameras.read().await;
        let c = cams
            .iter()
            .find(|c| c.id() == id)
            .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown camera {id}")))?;
        (c.ip.clone(), c.device_url.clone(), c.kind)
    };
    let creds = app.creds.read().await.get(&id).cloned().unwrap_or_default();
    let cam = tokio::task::spawn_blocking(move || match kind {
        SourceKind::Onvif => onvif::Client::new(&creds.user, &creds.pass)?.probe(&ip, &url),
        SourceKind::Stream => {
            let info = stream::probe(&onvif::with_credentials(&url, &creds.user, &creds.pass))?;
            Ok(stream::camera(&url, info))
        }
    })
        .await
        .map_err(|e| anyhow!(e))?
        .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}")))?;
    let cam = save_camera(&app, cam, None).await?;
    Ok(Json(json!(views(&app, std::slice::from_ref(&cam))[0])))
}

/// Partial update. Only keys present in the body change:
///   "name": string | null   — empty/null clears it (falls back to the detected name)
///   "pose": Pose | null     — null un-places the camera
async fn update_camera(
    State(app): AppState,
    Path(id): Path<String>,
    Json(body): Json<serde_json::Map<String, Value>>,
) -> ApiResult<Json<Value>> {
    let bad = |e: serde_json::Error| ApiError(StatusCode::BAD_REQUEST, e.to_string());
    let mut cams = app.cameras.write().await;
    let cam = cams
        .iter_mut()
        .find(|c| c.id() == id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown camera {id}")))?;
    if let Some(v) = body.get("name") {
        let name: Option<String> = serde_json::from_value(v.clone()).map_err(bad)?;
        cam.name = name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty());
    }
    if let Some(v) = body.get("pose") {
        cam.pose = serde_json::from_value(v.clone()).map_err(bad)?;
    }
    let updated = cam.clone();
    persist(&app, &cams, &*app.creds.read().await)?;
    Ok(Json(json!(views(&app, std::slice::from_ref(&updated))[0])))
}

async fn remove_camera(State(app): AppState, Path(id): Path<String>) -> ApiResult<StatusCode> {
    let mut cams = app.cameras.write().await;
    let mut creds = app.creds.write().await;
    let before = cams.len();
    cams.retain(|c| c.id() != id);
    if cams.len() == before {
        return Err(ApiError(StatusCode::NOT_FOUND, format!("unknown camera {id}")));
    }
    creds.remove(&id);
    app.snaps.lock().unwrap().remove(&id);
    persist(&app, &cams, &creds)?;
    Ok(StatusCode::NO_CONTENT)
}

fn persist(app: &App, cams: &[CameraInfo], creds: &HashMap<String, Creds>) -> Result<()> {
    write_file(&app.data.join("cameras.json"), &serde_json::to_vec_pretty(cams)?, false)?;
    write_file(&app.data.join("credentials.json"), &serde_json::to_vec_pretty(creds)?, true)
}

async fn camera_source(app: &App, id: &str) -> ApiResult<String> {
    let cams = app.cameras.read().await;
    let cam = cams
        .iter()
        .find(|c| c.id() == id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown camera {id}")))?;
    let uri = cam
        .stream_uri()
        .ok_or_else(|| ApiError(StatusCode::CONFLICT, "camera has no RTSP stream".into()))?;
    let c = app.creds.read().await.get(id).cloned().unwrap_or_default();
    Ok(onvif::with_credentials(uri, &c.user, &c.pass))
}

#[derive(Deserialize)]
struct SnapshotQuery {
    /// Wider than the default 960 px (e.g. full resolution for analysis); not cached.
    width: Option<u32>,
}

async fn snapshot(State(app): AppState, Path(id): Path<String>, Query(q): Query<SnapshotQuery>) -> ApiResult<Response> {
    let src = camera_source(&app, &id).await?;
    if let Some(w) = q.width.filter(|&w| w != 960) {
        let bytes = tokio::task::spawn_blocking(move || capture::snapshot(&src, w.clamp(160, 8192)))
            .await
            .map_err(|e| anyhow!(e))?
            .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}")))?;
        return Ok(jpeg(Bytes::from(bytes)));
    }
    let slot = app.snaps.lock().unwrap().entry(id).or_default().clone();
    // Per-camera lock: concurrent viewers share one ffmpeg grab instead of stampeding the camera.
    let mut cached = slot.lock().await;
    if let Some((at, bytes)) = cached.as_ref() {
        if at.elapsed() < SNAPSHOT_TTL {
            return Ok(jpeg(bytes.clone()));
        }
    }
    let bytes = tokio::task::spawn_blocking(move || capture::snapshot(&src, 960))
        .await
        .map_err(|e| anyhow!(e))?
        .map_err(|e| ApiError(StatusCode::BAD_GATEWAY, format!("{e:#}")))?;
    let bytes = Bytes::from(bytes);
    *cached = Some((Instant::now(), bytes.clone()));
    Ok(jpeg(bytes))
}

async fn background(State(app): AppState, Path(id): Path<String>) -> ApiResult<Response> {
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(ApiError(StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let bytes = tokio::fs::read(background_path(&app.data, &id))
        .await
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "not captured yet".into()))?;
    Ok(([(header::CONTENT_TYPE, "image/png"), (header::CACHE_CONTROL, "no-cache")], bytes).into_response())
}

async fn start_capture(State(app): AppState) -> ApiResult<Json<Job>> {
    let mut job = app.job.write().await;
    if job.running {
        return Err(ApiError(StatusCode::CONFLICT, "capture already running".into()));
    }
    let cams = app.cameras.read().await.clone();
    if cams.is_empty() {
        return Err(ApiError(StatusCode::CONFLICT, "no cameras added yet".into()));
    }
    *job = Job { running: true, message: "Starting capture".into(), done: 0, total: cams.len() };

    let app2 = app.clone();
    tokio::spawn(async move {
        let mut failures = Vec::new();
        for (i, cam) in cams.iter().enumerate() {
            app2.job.write().await.message = format!("Capturing {}", cam.label());
            let id = cam.id();
            let result = match camera_source(&app2, &id).await {
                Ok(src) => {
                    let out = app2.data.join(&id);
                    tokio::task::spawn_blocking(move || {
                        capture::capture(&src, &out, &CaptureOpts { frames: 15, fps: 1.0 })
                    })
                    .await
                    .map_err(|e| anyhow!(e))
                    .and_then(|r| r)
                    .map(|_| ())
                }
                Err(e) => Err(anyhow!(e.1)),
            };
            if let Err(e) = result {
                failures.push(format!("{}: {e:#}", cam.label()));
            }
            app2.job.write().await.done = i + 1;
        }
        let mut job = app2.job.write().await;
        job.running = false;
        job.message = if failures.is_empty() {
            "Capture complete".into()
        } else {
            format!("{} of {} failed — {}", failures.len(), job.total, failures.join("; "))
        };
    });

    Ok(Json(job.clone()))
}

// ---------- live ----------

fn start_live(app: Arc<App>) -> live::LiveHandle {
    let fps: f32 = std::env::var("MINASTIR_LIVE_FPS").ok().and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let config = LiveConfig {
        model_dir: app.data.join("models"),
        fps,
        min_score: 0.45,
        use_gpu: std::env::var("MINASTIR_GPU").as_deref() != Ok("0"),
        frames: app.frames.clone(),
    };
    let specs_app = app.clone();
    let site_app = app.clone();
    let tx = app.live.clone();
    live::spawn(
        config,
        move || live_specs(&specs_app),
        move |snap| {
            // A vehicle can't be inside a house: drop sightings that land in a traced house
            // (usually the extreme edge of a wide lens, where placement is least reliable).
            let houses = site_app.site.blocking_read().houses.clone();
            let street = site_app.street.lock().unwrap().clone();
            let json = if houses.is_empty() && street.is_none() {
                serde_json::to_string(snap)
            } else {
                let mut s = snap.clone();
                s.objects.retain(|o| {
                    let vehicle = matches!(o.class, "car" | "truck" | "bus");
                    let inside = vehicle && houses.iter().any(|h| inside_house(&h.points, [o.x, o.z], 0.8));
                    if inside && std::env::var("MINASTIR_DEBUG_VEHICLES").is_ok() {
                        eprintln!("vehicle track {} at ({:.1}, {:.1}): inside a traced house, not shown", o.id, o.x, o.z);
                    }
                    !inside
                });
                serde_json::to_string(&s)
            };
            if let Ok(json) = json {
                tx.send_replace(Arc::from(json));
            }
        },
    )
}

/// Cameras the live engine should watch: every placed camera, on its smallest stream that's at
/// least as wide as the engine decodes to (sharp crops for fingerprints and fitting; smaller
/// streams would only be enlarged) — except panoramas, see below.
fn live_specs(app: &App) -> Vec<CamSpec> {
    let cams = app.cameras.blocking_read();
    let creds = app.creds.blocking_read();
    // Replay: plays <dir>/<camera id>.mp4 (looped, in real time) instead of each camera's
    // stream; cameras without a file are left out.
    let replay = app.replay.read().unwrap().clone();
    cams.iter()
        .filter_map(|c| {
            let pose = c.pose?;
            let best = c.best_profile()?;
            if let Some(dir) = &replay {
                let file = dir.join(format!("{}.mp4", c.id()));
                if !file.exists() {
                    return None;
                }
                let aspect = match (best.width, best.height) {
                    (Some(w), Some(h)) if h > 0 => w as f64 / h as f64,
                    _ => 16.0 / 9.0,
                };
                let maps = c.calibration.as_ref().and_then(|cal| scene_maps(app, &c.id(), cal));
                return Some(CamSpec { id: c.id(), source: file.display().to_string(), aspect, pose, calibration: c.calibration.clone(), maps, background: background_image(app, &c.id()) });
            }
            let best_aspect = match (best.width, best.height) {
                (Some(w), Some(h)) if h > 0 => w as f64 / h as f64,
                _ => 16.0 / 9.0,
            };
            // Sharp crops want a stream as wide as the engine decodes to, but the Reolink Duo's
            // main stream (4608 px) drops sessions unpredictably; its sub-stream is reliable.
            let min_w = if best_aspect >= 1.9 { 1280 } else { live::decode_width(best_aspect) };
            let profile = c
                .profiles
                .iter()
                .filter(|p| p.stream_uri.is_some() && p.width.unwrap_or(0) >= min_w)
                .min_by_key(|p| p.width.unwrap_or(0))
                .unwrap_or(best);
            let aspect = match (profile.width, profile.height) {
                (Some(w), Some(h)) if h > 0 => w as f64 / h as f64,
                _ => best_aspect,
            };
            let cr = creds.get(&c.id()).cloned().unwrap_or_default();
            let maps = c.calibration.as_ref().and_then(|cal| scene_maps(app, &c.id(), cal));
            Some(CamSpec {
                id: c.id(),
                source: onvif::with_credentials(profile.stream_uri.as_deref()?, &cr.user, &cr.pass),
                aspect,
                pose,
                calibration: c.calibration.clone(),
                maps,
                background: background_image(app, &c.id()),
            })
        })
        .collect()
}

/// A camera's empty-scene picture, decoded once and reloaded only when the file changes.
fn background_image(app: &App, id: &str) -> Option<Arc<image::RgbImage>> {
    let path = background_path(&app.data, id);
    let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
    let mut cache = app.backgrounds.lock().unwrap();
    if let Some((at, img)) = cache.get(id) {
        if *at == modified {
            return Some(img.clone());
        }
    }
    let img = Arc::new(image::open(&path).ok()?.to_rgb8());
    cache.insert(id.to_string(), (modified, img.clone()));
    Some(img)
}

/// Scene maps for a calibrated camera, loaded from disk once per calibration.
fn scene_maps(app: &App, id: &str, cal: &minastir_core::calib::CameraCalibration) -> Option<Arc<Vec<LensMap>>> {
    let mut cache = app.scene_maps.lock().unwrap();
    if let Some((at, maps)) = cache.get(id) {
        if *at == cal.computed_at {
            return Some(maps.clone());
        }
    }
    let maps: Vec<LensMap> = (0..cal.lenses.len())
        .map(|i| LensMap::load(&app.data.join(id).join(format!("scene_lens{i}.bin"))))
        .collect::<Result<_>>()
        .map_err(|e| eprintln!("scene map for {id}: {e:#}"))
        .ok()?;
    let maps = Arc::new(maps);
    cache.insert(id.to_string(), (cal.computed_at, maps.clone()));
    Some(maps)
}

// ---------- calibration ----------

const MOGE_URL: &str = "https://huggingface.co/Ruicheng/moge-2-vitl-normal-onnx/resolve/main/model.onnx";
const SEGFORMER_URL: &str = "https://huggingface.co/Xenova/segformer-b5-finetuned-ade-640-640/resolve/main/onnx/model.onnx";

/// Measure every camera's lens, tilt, lean and height from its own (people-free) image.
/// Captures a background first where there isn't one.
async fn start_calibrate(State(app): AppState) -> ApiResult<Json<Job>> {
    let mut job = app.job.write().await;
    if job.running {
        return Err(ApiError(StatusCode::CONFLICT, format!("busy: {}", job.message)));
    }
    let cams = app.cameras.read().await.clone();
    if cams.is_empty() {
        return Err(ApiError(StatusCode::CONFLICT, "no cameras added yet".into()));
    }
    *job = Job { running: true, message: "Starting calibration".into(), done: 0, total: cams.len() };
    let app2 = app.clone();
    tokio::task::spawn_blocking(move || {
        let failures = run_calibration(&app2, &cams);
        let mut job = app2.job.blocking_write();
        job.running = false;
        job.message = match failures {
            // Keep the alignment summary ("Calibration complete · aligned …") if there is one.
            Ok(f) if f.is_empty() && job.message.starts_with("Calibration complete") => job.message.clone(),
            Ok(f) if f.is_empty() => "Calibration complete".into(),
            Ok(f) => format!("{} of {} failed — {}", f.len(), job.total, f.join("; ")),
            Err(e) => format!("Calibration failed: {e:#}"),
        };
    });
    Ok(Json(job.clone()))
}

fn run_calibration(app: &Arc<App>, cams: &[CameraInfo]) -> Result<Vec<String>> {
    let set = |msg: String| app.job.blocking_write().message = msg;
    let models = app.data.join("models");
    let (moge, seg) = (models.join("moge2_vitl_normal.onnx"), models.join("segformer_b5_ade.onnx"));
    for (path, url, what) in [(&moge, MOGE_URL, "geometry model (1.3 GB)"), (&seg, SEGFORMER_URL, "segmentation model (340 MB)")] {
        if !path.exists() {
            set(format!("Downloading the {what}, once"));
            live::download(url, path)?;
        }
    }
    set("Loading calibration models".into());
    // CPU: MoGe's variable input shapes don't compile for CoreML, and this runs rarely.
    let mut cal = Calibrator { moge: Moge::load(&moge, Device::Cpu)?, segmenter: Segmenter::load(&seg, Device::Cpu)?, num_tokens: 2500 };

    let rt = tokio::runtime::Handle::current();
    let mut failures = Vec::new();
    let mut survey_cams = Vec::new();
    for (i, cam) in cams.iter().enumerate() {
        let id = cam.id();
        let dir = app.data.join(&id);
        let bg = background_path(&app.data, &id);
        let result = (|| -> Result<_> {
            if !bg.exists() {
                set(format!("Capturing {}", cam.label()));
                let src = rt.block_on(camera_source(app, &id)).map_err(|e| anyhow!(e.1))?;
                capture::capture(&src, &dir, &CaptureOpts { frames: 15, fps: 1.0 })?;
            }
            set(format!("Measuring {}", cam.label()));
            let img = image::open(&bg)?.to_rgb8();
            let raw = cal.analyse(img.as_raw(), img.width() as usize, img.height() as usize)?;
            // Fails early if the lens can't be fitted at all (e.g. no ground in view).
            minastir_vision::calibrate::assemble(&raw, &vec![1.0; raw.len()])?;
            Ok((img, raw))
        })();
        match result {
            Ok((image, raw)) => survey_cams.push(SurveyCam {
                id: id.clone(),
                label: cam.label(),
                rough: cam.pose.unwrap_or(minastir_core::onvif::Pose { x: 0.0, z: 0.0, height: 3.0, yaw: 0.0, pitch: 20.0 }),
                image,
                raw,
            }),
            Err(e) => failures.push(format!("{}: {e:#}", cam.label())),
        }
        app.job.blocking_write().done = i + 1;
    }
    if survey_cams.is_empty() {
        return Ok(failures);
    }

    // Compare overlapping cameras: correct lens widths, then place them relative to each other.
    set("Matching cameras against each other".into());
    let model = app.data.join("models").join(minastir_vision::matcher::MODEL_FILE);
    if !model.exists() {
        live::download(minastir_vision::matcher::MODEL_URL, &model)?;
    }
    let mut matcher = minastir_vision::matcher::Matcher::load(&model, Device::Cpu)?;
    let images: std::collections::HashMap<String, image::RgbImage> = survey_cams.iter().map(|c| (c.id.clone(), c.image.clone())).collect();
    let results = survey(survey_cams, &mut matcher, &mut |line| println!("calibrate: {line}"))?;

    let mut moved = Vec::new();
    let mut widened = Vec::new();
    {
        let mut inv = app.cameras.blocking_write();
        for s in &results {
            let Some(stored) = inv.iter_mut().find(|x| x.id() == s.id) else { continue };
            if let Err(e) = write_outputs(&images[&s.id], &s.result, &app.data.join(&s.id)) {
                failures.push(format!("{}: {e:#}", stored.label()));
                continue;
            }
            let mut c = s.result.calibration.clone();
            c.scale = s.placed.scale;
            c.alignment = s.placed.alignment.clone();
            if c.fov_correction > 1.01 {
                widened.push(format!("{} lens {:.0}°", stored.label(), c.lenses[0].hfov));
            }
            let mut pose = match &s.placed.alignment {
                Some(al) if !al.anchor => {
                    moved.push(format!("{} ({} points, ±{:.2} m)", stored.label(), al.inliers, al.rms));
                    s.placed.pose
                }
                _ => stored.pose.unwrap_or(s.placed.pose),
            };
            pose.pitch = (c.pitch * 10.0).round() / 10.0;
            pose.height = (c.height * c.scale * 100.0).round() / 100.0;
            stored.pose = Some(pose);
            stored.calibration = Some(c);
        }
        persist(app, &inv, &app.creds.blocking_read())?;
    }
    let mut summary = "Calibration complete".to_string();
    if !widened.is_empty() {
        summary += &format!(" · corrected {}", widened.join(", "));
    }
    summary += &if moved.is_empty() {
        " · no camera pair overlapped strongly enough to align".to_string()
    } else {
        format!(" · aligned {}", moved.join(", "))
    };
    set(summary);
    Ok(failures)
}

/// What a calibrated camera sees, as labelled 3D points in its own levelled frame (camera at the
/// origin, heading 0, depth scale applied): flat [x, z, height, label, …] with label codes from
/// `segment::Label`. The UI places them with the camera's current pose, so they follow edits.
async fn scene_points(State(app): AppState, Path(id): Path<String>) -> ApiResult<Response> {
    let cam = app
        .cameras
        .read()
        .await
        .iter()
        .find(|c| c.id() == id)
        .cloned()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown camera {id}")))?;
    let cal = cam.calibration.clone().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "not calibrated yet".into()))?;
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || -> Option<String> {
        let maps = scene_maps(&app2, &cam.id(), &cal)?;
        let pose = cam.pose.unwrap_or(minastir_core::onvif::Pose { x: 0.0, z: 0.0, height: cal.height, yaw: 0.0, pitch: cal.pitch });
        let zero = minastir_core::onvif::Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height * cal.scale, ..pose };
        let model = minastir_vision::geometry::CameraModel::new(zero, &cal, Some(maps));
        let pts = model.scene_points(3, 40.0);
        let mut out = String::with_capacity(pts.len() * 20 + 16);
        out.push_str("{\"points\":[");
        for (i, (x, z, y, l)) in pts.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("{:.2},{:.2},{:.2},{}", x, z, y, *l as u8));
        }
        out.push_str("]}");
        Some(out)
    })
    .await
    .map_err(|e| anyhow!(e))?
    .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "scene map missing — recalibrate".into()))?;
    Ok(([(header::CONTENT_TYPE, "application/json"), (header::CACHE_CONTROL, "no-cache")], body).into_response())
}

/// Clean top-down map of what a calibrated camera sees, in its own levelled frame (camera at the
/// origin, heading 0): ground types run-length encoded on a fine grid with cars removed, plus
/// walls, trees and poles as simple shapes. The UI places it with the camera's current pose.
async fn ground_map(State(app): AppState, Path(id): Path<String>) -> ApiResult<Response> {
    let cam = app
        .cameras
        .read()
        .await
        .iter()
        .find(|c| c.id() == id)
        .cloned()
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, format!("unknown camera {id}")))?;
    let cal = cam.calibration.clone().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "not calibrated yet".into()))?;
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || -> Option<String> {
        use minastir_vision::groundmap;
        let g = build_ground_map(&app2, &cam, &cal)?;
        Some(
            serde_json::json!({
                "half": g.half,
                "cell": groundmap::FINE,
                "column": groundmap::COLUMN,
                "size": g.size,
                "rle": g.rle(),
                "shapes": g.shapes,
            })
            .to_string(),
        )
    })
    .await
    .map_err(|e| anyhow!(e))?
    .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "scene map missing — recalibrate".into()))?;
    Ok(([(header::CONTENT_TYPE, "application/json"), (header::CACHE_CONTROL, "no-cache")], body).into_response())
}

/// A calibrated camera's ground map in its own levelled frame (see `ground_map`).
fn build_ground_map(
    app: &App,
    cam: &minastir_core::onvif::CameraInfo,
    cal: &minastir_core::calib::CameraCalibration,
) -> Option<minastir_vision::groundmap::GroundMap> {
    let maps = scene_maps(app, &cam.id(), cal)?;
    let pitch = cam.pose.map_or(cal.pitch, |p| p.pitch);
    let zero = minastir_core::onvif::Pose { x: 0.0, z: 0.0, yaw: 0.0, height: cal.height * cal.scale, pitch };
    let model = minastir_vision::geometry::CameraModel::new(zero, cal, Some(maps));
    Some(minastir_vision::groundmap::build(&model, 40.0))
}

/// The scene built from every placed, calibrated camera at its current pose: the street as
/// clean bands (road, nature strips, footpaths, driveways; `null` without enough road) and the
/// cameras' walls, trees and poles in world coordinates, merged where views overlap.
async fn world(State(app): AppState) -> ApiResult<Response> {
    let cams = app.cameras.read().await.clone();
    let houses = app.site.read().await.houses.clone();
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || -> String {
        let built: Vec<_> = cams
            .iter()
            .filter_map(|c| {
                let (cal, pose) = (c.calibration.as_ref()?, c.pose?);
                Some((build_ground_map(&app2, c, cal)?, pose))
            })
            .collect();
        let refs: Vec<_> = built.iter().map(|(g, p)| (g, *p)).collect();
        let shapes: Vec<_> = built.iter().map(|(g, p)| (&g.shapes, *p)).collect();
        let mut merged = minastir_vision::shapes::merge_world(&shapes);
        // Traced houses replace whatever the cameras guessed there.
        let near = |p: [f32; 2], m: f64| houses.iter().any(|h| near_house(&h.points, [p[0] as f64, p[1] as f64], m));
        merged.walls.retain(|w| !near([(w.a[0] + w.b[0]) / 2.0, (w.a[1] + w.b[1]) / 2.0], 1.2));
        merged.trees.retain(|t| !near([t.x, t.z], 0.0));
        let street = minastir_vision::street::fit(&refs);
        *app2.street.lock().unwrap() = street.clone();
        serde_json::json!({
            "street": street,
            "shapes": merged,
        })
        .to_string()
    })
    .await
    .map_err(|e| anyhow!(e))?;
    Ok(([(header::CONTENT_TYPE, "application/json"), (header::CACHE_CONTROL, "no-cache")], body).into_response())
}

async fn get_site(State(app): AppState) -> Json<Site> {
    Json(app.site.read().await.clone())
}

/// Replace the map placement and house outlines. The image's own metadata (size, type) stays as
/// uploaded.
async fn put_site(State(app): AppState, Json(mut site): Json<Site>) -> ApiResult<Json<Site>> {
    let mut cur = app.site.write().await;
    site.map = match (site.map, &cur.map) {
        (Some(mut m), Some(old)) => {
            m.width = old.width;
            m.height = old.height;
            m.content_type = old.content_type.clone();
            if !(m.scale.is_finite() && m.scale > 0.0) {
                return Err(ApiError(StatusCode::BAD_REQUEST, "bad map scale".into()));
            }
            Some(m)
        }
        _ => None,
    };
    site.houses.retain(|h| h.points.len() >= 3 && h.height > 0.0);
    // Moving the photo without touching the outlines (dragging it into place, or undoing an
    // automatic placement): the outlines were traced on it, so they move with it.
    let same_houses = site.houses.len() == cur.houses.len() && site.houses.iter().zip(&cur.houses).all(|(a, b)| a.points == b.points);
    if let (true, Some(old), Some(new)) = (same_houses, &cur.map, &site.map) {
        follow_map(&mut site.houses, old, new);
    }
    write_file(&app.data.join("site.json"), &serde_json::to_vec_pretty(&site).map_err(|e| anyhow!(e))?, false)?;
    *cur = site.clone();
    Ok(Json(site))
}

/// Moves outlines traced on the map photo along with it, from placement `old` to `new`: each
/// corner keeps its spot on the photo.
fn follow_map(houses: &mut [House], old: &MapImage, new: &MapImage) {
    let (so, co) = old.rotation.to_radians().sin_cos();
    let (sn, cn) = new.rotation.to_radians().sin_cos();
    for p in houses.iter_mut().flat_map(|h| h.points.iter_mut()) {
        // World → photo pixel under the old placement, then back under the new one.
        let (dx, dz) = ((p[0] - old.x) / old.scale, (p[1] - old.z) / old.scale);
        let (u, v) = (dx * co + dz * so, -dx * so + dz * co);
        *p = [new.x + new.scale * (u * cn - v * sn), new.z + new.scale * (u * sn + v * cn)];
    }
}

/// Upload a map image (raw PNG/JPEG body). Placed initially centred on the cameras, ~80 m wide.
async fn upload_site_map(State(app): AppState, body: Bytes) -> ApiResult<Json<Site>> {
    let fmt = image::guess_format(&body).map_err(|_| ApiError(StatusCode::BAD_REQUEST, "not an image".into()))?;
    let content_type = match fmt {
        image::ImageFormat::Png => "image/png",
        image::ImageFormat::Jpeg => "image/jpeg",
        _ => return Err(ApiError(StatusCode::BAD_REQUEST, "use a PNG or JPEG image".into())),
    };
    let (w, h) = image::load_from_memory(&body).map_err(|e| ApiError(StatusCode::BAD_REQUEST, format!("{e}")))?.into_rgb8().dimensions();
    let placed: Vec<_> = app.cameras.read().await.iter().filter_map(|c| c.pose).collect();
    let (cx, cz) = if placed.is_empty() {
        (0.0, 0.0)
    } else {
        (placed.iter().map(|p| p.x).sum::<f64>() / placed.len() as f64, placed.iter().map(|p| p.z).sum::<f64>() / placed.len() as f64)
    };
    let scale = 80.0 / w as f64;
    write_file(&app.data.join("site_map.bin"), &body, false)?;
    let mut site = app.site.write().await;
    site.map = Some(MapImage {
        width: w,
        height: h,
        content_type: content_type.into(),
        x: cx - scale * w as f64 / 2.0,
        z: cz - scale * h as f64 / 2.0,
        scale,
        rotation: 0.0,
    });
    write_file(&app.data.join("site.json"), &serde_json::to_vec_pretty(&*site).map_err(|e| anyhow!(e))?, false)?;
    Ok(Json(site.clone()))
}

/// Place the map image automatically by matching it against the cameras' map (lawn vs paving
/// pattern, every rotation and a wide range of scales). Saves the result; returns the site plus
/// the previous placement (for undo) and a confidence figure.
async fn auto_align_site_map(State(app): AppState) -> ApiResult<Json<Value>> {
    let previous = app.site.read().await.map.clone().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "no map image".into()))?;
    let cams = app.cameras.read().await.clone();
    let app2 = app.clone();
    let placed = tokio::task::spawn_blocking(move || -> Result<minastir_vision::sitealign::Placement, String> {
        let built: Vec<_> = cams
            .iter()
            .filter_map(|c| {
                let (cal, pose) = (c.calibration.as_ref()?, c.pose?);
                Some((build_ground_map(&app2, c, cal)?, pose))
            })
            .collect();
        let refs: Vec<_> = built.iter().map(|(g, p)| (g, *p)).collect();
        let street = minastir_vision::street::fit(&refs);
        let tpl = minastir_vision::sitealign::template(&refs, street.as_ref()).ok_or("the cameras don't see enough ground yet — calibrate first")?;
        let (_, img) = site_image(&app2).ok_or("can't read the map image")?;
        minastir_vision::sitealign::align(&img, &tpl).ok_or_else(|| "couldn't find a match — is the photo of the same place?".to_string())
    })
    .await
    .map_err(|e| anyhow!(e))?
    .map_err(|e| ApiError(StatusCode::UNPROCESSABLE_ENTITY, e))?;
    let mut site = app.site.write().await;
    let map = MapImage { x: placed.x, z: placed.z, scale: placed.scale, rotation: placed.rotation, ..previous.clone() };
    follow_map(&mut site.houses, &previous, &map);
    site.map = Some(map);
    write_file(&app.data.join("site.json"), &serde_json::to_vec_pretty(&*site).map_err(|e| anyhow!(e))?, false)?;
    Ok(Json(json!({ "site": *site, "previous": previous, "confidence": placed.confidence, "score": placed.score })))
}

#[derive(Deserialize)]
struct RoofPoint {
    x: f64,
    z: f64,
    #[serde(default = "yes")]
    include: bool,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct RoofRequest {
    /// First point: the click on the roof; more points refine (include / exclude).
    points: Vec<RoofPoint>,
}

/// The site photo (labels painted out), decoded once and kept until the file changes.
fn site_image(app: &App) -> Option<(u64, Arc<image::RgbImage>)> {
    let path = app.data.join("site_map.bin");
    let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
    let mut slot = app.site_img.lock().unwrap();
    if let Some((t, img)) = slot.as_ref() {
        if *t == mtime {
            return Some((mtime.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0), img.clone()));
        }
    }
    // Map labels (street / place names, house numbers) are painted out once: they look like hard
    // edges and split roofs.
    let img = Arc::new(minastir_vision::roof::remove_labels(&image::load_from_memory(&std::fs::read(&path).ok()?).ok()?.to_rgb8()));
    *slot = Some((mtime, img.clone()));
    Some((mtime.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0), img))
}

/// Outline the roof at a click (world x, z) in the placed photo, refined by further include /
/// exclude points. Uses Segment Anything when its model is installed, else colour region growing.
/// Returns the outline in world coordinates; nothing is saved — the UI adds it as a house.
async fn find_roof(State(app): AppState, Json(req): Json<RoofRequest>) -> ApiResult<Json<Value>> {
    let m = app.site.read().await.map.clone().ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "no map image".into()))?;
    if req.points.is_empty() {
        return Err(ApiError(StatusCode::BAD_REQUEST, "click on a roof".into()));
    }
    let app2 = app.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<Option<(Vec<[f64; 2]>, &'static str)>, String> {
        let (photo, img) = site_image(&app2).ok_or("can't read the map image")?;
        let (sn, cs) = m.rotation.to_radians().sin_cos();
        let to_img = |x: f64, z: f64| {
            let (dx, dz) = ((x - m.x) / m.scale, (z - m.z) / m.scale);
            (dx * cs + dz * sn, -dx * sn + dz * cs)
        };
        let prompts: Vec<_> = req
            .points
            .iter()
            .map(|p| {
                let (u, v) = to_img(p.x, p.z);
                minastir_vision::roof::Prompt { u, v, include: p.include }
            })
            .collect();
        let models = app2.data.join("models");
        let mut sam = app2.sam.lock().unwrap();
        if sam.is_none() && models.join(minastir_vision::roof::SAM_ENCODER).exists() {
            *sam = minastir_vision::roof::Sam::load(&models).map_err(|e| eprintln!("segment anything: {e:#}")).ok();
        }
        let (poly, how) = match sam.as_mut() {
            Some(s) => (s.outline_prompts(&img, photo, &prompts, m.scale).map_err(|e| format!("{e:#}"))?, "sam"),
            None => (minastir_vision::roof::outline(&img, (prompts[0].u, prompts[0].v), m.scale), "colour"),
        };
        Ok(poly.map(|p| (p.into_iter().map(|[u, v]| [m.x + m.scale * (u * cs - v * sn), m.z + m.scale * (u * sn + v * cs)]).collect(), how)))
    })
    .await
    .map_err(|e| anyhow!(e))?
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let (points, how) = result.ok_or_else(|| ApiError(StatusCode::UNPROCESSABLE_ENTITY, "couldn't find a roof there".into()))?;
    Ok(Json(json!({ "points": points, "method": how })))
}

async fn get_site_map(State(app): AppState) -> ApiResult<Response> {
    let ct = app.site.read().await.map.as_ref().map(|m| m.content_type.clone()).ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "no map image".into()))?;
    let bytes = tokio::fs::read(app.data.join("site_map.bin")).await.map_err(|_| ApiError(StatusCode::NOT_FOUND, "no map image".into()))?;
    Ok(([(header::CONTENT_TYPE, ct.as_str()), (header::CACHE_CONTROL, "no-cache")], bytes).into_response())
}

async fn delete_site_map(State(app): AppState) -> ApiResult<Json<Site>> {
    let mut site = app.site.write().await;
    site.map = None;
    let _ = std::fs::remove_file(app.data.join("site_map.bin"));
    write_file(&app.data.join("site.json"), &serde_json::to_vec_pretty(&*site).map_err(|e| anyhow!(e))?, false)?;
    Ok(Json(site.clone()))
}

/// Whether world point p is inside polygon `poly` by at least `inset` metres (not near its edge).
fn inside_house(poly: &[[f64; 2]], p: [f64; 2], inset: f64) -> bool {
    near_house(poly, p, 0.0) && !near_house_edge(poly, p, inset)
}

fn near_house_edge(poly: &[[f64; 2]], p: [f64; 2], margin: f64) -> bool {
    let n = poly.len();
    (0..n).any(|i| {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let d = [b[0] - a[0], b[1] - a[1]];
        let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]).max(1e-9)).clamp(0.0, 1.0);
        (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1]) < margin
    })
}

/// Whether world point p is inside polygon `poly` or within `margin` metres of its outline.
fn near_house(poly: &[[f64; 2]], p: [f64; 2], margin: f64) -> bool {
    let n = poly.len();
    let mut inside = false;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
            inside = !inside;
        }
        let d = [b[0] - a[0], b[1] - a[1]];
        let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]).max(1e-9)).clamp(0.0, 1.0);
        if (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1]) < margin {
            return true;
        }
    }
    inside
}

async fn calibration_image(State(app): AppState, Path(id): Path<String>) -> ApiResult<Response> {
    if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(ApiError(StatusCode::BAD_REQUEST, "bad id".into()));
    }
    let bytes = tokio::fs::read(app.data.join(&id).join("calibration.jpg"))
        .await
        .map_err(|_| ApiError(StatusCode::NOT_FOUND, "not calibrated yet".into()))?;
    Ok(([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-cache")], bytes).into_response())
}

/// Streams live snapshots (tracked objects in world coordinates) to the browser.
#[derive(Deserialize)]
struct FrameQuery {
    width: Option<u32>,
}

/// The frame the live engine most recently analysed for a camera (replay footage in replay mode).
async fn live_frame(State(app): AppState, Path(id): Path<String>, Query(q): Query<FrameQuery>) -> ApiResult<Response> {
    let width = q.width.unwrap_or(480).clamp(120, 1920);
    let frames = app.frames.clone();
    let bytes = tokio::task::spawn_blocking(move || frames.jpeg(&id, width))
        .await
        .map_err(|e| anyhow!(e))?
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "no recent frame".into()))?;
    Ok(([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")], bytes).into_response())
}

/// Test videos: folders under <data>/replay holding <camera id>.mp4 files.
fn replays(data: &FsPath) -> Vec<Value> {
    let mut out: Vec<(String, Value)> = std::fs::read_dir(data.join("replay"))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            let mut cameras: Vec<String> = std::fs::read_dir(e.path())
                .ok()?
                .flatten()
                .filter_map(|f| f.file_name().to_str()?.strip_suffix(".mp4").map(str::to_string))
                .collect();
            if cameras.is_empty() {
                return None;
            }
            cameras.sort();
            // A readable name, if replay.json gives one.
            let title = read_json::<Value>(&e.path().join("replay.json")).and_then(|v| v["title"].as_str().map(str::to_string));
            Some((name.clone(), json!({ "name": name, "cameras": cameras, "title": title })))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.into_iter().map(|(_, v)| v).collect()
}

async fn get_source(State(app): AppState) -> Json<Value> {
    let current = app.replay.read().unwrap().as_ref().and_then(|p| p.file_name()?.to_str().map(str::to_string));
    Json(json!({ "replay": current, "replays": replays(&app.data) }))
}

/// Switches the live engine between the cameras ({"replay": null}) and a test video
/// ({"replay": "<folder>"}); it changes over within a second, and the choice is remembered.
async fn put_source(State(app): AppState, Json(choice): Json<SourceChoice>) -> ApiResult<Json<Value>> {
    let path = match &choice.replay {
        None => None,
        Some(n) => {
            let ok = !n.is_empty() && n.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) && !n.starts_with('.');
            let p = app.data.join("replay").join(n);
            if !ok || !p.is_dir() {
                return Err(ApiError(StatusCode::NOT_FOUND, format!("no test video called {n:?}")));
            }
            Some(p)
        }
    };
    *app.replay.write().unwrap() = path;
    write_file(&app.data.join("source.json"), serde_json::to_string_pretty(&choice).map_err(|e| anyhow!(e))?.as_bytes(), false)?;
    Ok(get_source(State(app)).await)
}

async fn live_ws(State(app): AppState, ws: WebSocketUpgrade) -> Response {
    let mut rx = app.live.subscribe();
    ws.on_upgrade(move |mut socket: WebSocket| async move {
        loop {
            let msg = rx.borrow_and_update().clone();
            if socket.send(Message::Text(msg.as_ref().into())).await.is_err() {
                break;
            }
            tokio::select! {
                changed = rx.changed() => if changed.is_err() { break },
                incoming = socket.recv() => match incoming {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                },
            }
        }
    })
}

// ---------- helpers ----------

fn jpeg(bytes: Bytes) -> Response {
    ([(header::CONTENT_TYPE, "image/jpeg"), (header::CACHE_CONTROL, "no-store")], bytes).into_response()
}

fn background_path(data: &FsPath, id: &str) -> PathBuf {
    data.join(id).join("background.png")
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &FsPath) -> Option<T> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

fn write_file(path: &FsPath, contents: &[u8], private: bool) -> Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    if private {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    #[cfg(not(unix))]
    let _ = private;
    opts.open(path)?.write_all(contents)?;
    Ok(())
}
