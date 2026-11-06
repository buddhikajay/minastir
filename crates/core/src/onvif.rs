//! Minimal ONVIF SOAP client: device info, capabilities, media profiles, stream/snapshot URIs.
//! Auth is WS-Security UsernameToken (PasswordDigest), timestamped with the camera's own clock
//! so cameras whose time drifts from ours still accept requests.

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use chrono::{DateTime, Duration as ChronoDuration, NaiveDate, Utc};
use rand::RngCore;
use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub token: String,
    pub name: String,
    pub encoding: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub stream_uri: Option<String>,
    pub snapshot_uri: Option<String>,
}

/// Where a camera sits in the scene, in metres. Set by hand in the layout editor; later used as
/// the starting estimate for reconstruction, which refines it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Pose {
    pub x: f64,
    pub z: f64,
    /// Mounting height above the floor.
    pub height: f64,
    /// Heading in degrees; 0 looks along +Z, increasing counter-clockwise seen from above.
    pub yaw: f64,
    /// Downward tilt in degrees.
    pub pitch: f64,
}

/// How a camera was added.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// Adopted over ONVIF; `device_url` is its device service.
    #[default]
    Onvif,
    /// Added by an RTSP/RTSPS URL; `device_url` is the stream URL (see `stream`).
    Stream,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    #[serde(default)]
    pub kind: SourceKind,
    pub ip: String,
    pub device_url: String,
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub firmware: Option<String>,
    pub serial: Option<String>,
    pub ptz: bool,
    pub profiles: Vec<Profile>,
    /// Name set by the user in Minastir — always wins.
    #[serde(default)]
    pub name: Option<String>,
    /// Placement in the scene (None = not placed yet).
    #[serde(default)]
    pub pose: Option<Pose>,
    /// Lens / tilt / height measured from the camera's own image (see `calib`).
    #[serde(default)]
    pub calibration: Option<crate::calib::CameraCalibration>,
    /// Name candidates reported by the camera / network (see `naming`).
    /// Device name from the vendor's own API (e.g. the name set in the Reolink app).
    #[serde(default)]
    pub vendor_name: Option<String>,
    #[serde(default)]
    pub hostname: Option<String>,
    #[serde(default)]
    pub osd_texts: Vec<String>,
    #[serde(default)]
    pub dns_name: Option<String>,
}

impl CameraInfo {
    /// Stable identifier for folders/filenames.
    pub fn id(&self) -> String {
        if self.kind == SourceKind::Stream {
            return crate::stream::stream_id(&self.device_url);
        }
        match &self.serial {
            Some(s) if !s.is_empty() => s.chars().filter(|c| c.is_ascii_alphanumeric()).collect(),
            _ => self.ip.replace('.', "_"),
        }
    }

    /// Display name: the user's name, else a name the camera/network reports that isn't a
    /// factory default, else the model.
    pub fn label(&self) -> String {
        if let Some(n) = self.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
            return n.to_string();
        }
        self.detected_name().unwrap_or_else(|| self.model_label())
    }

    /// Best non-generic name found on the camera or network, if any.
    pub fn detected_name(&self) -> Option<String> {
        let hints: Vec<&str> = [self.manufacturer.as_deref(), self.model.as_deref()].into_iter().flatten().collect();
        let candidates = self
            .vendor_name
            .as_deref()
            .into_iter()
            .chain(self.osd_texts.iter().map(String::as_str))
            .chain(self.hostname.as_deref())
            .chain(self.dns_name.as_deref());
        crate::naming::pick(candidates, &hints)
    }

    /// "Manufacturer Model", without repeating the brand.
    pub fn model_label(&self) -> String {
        match (&self.manufacturer, &self.model) {
            // Many vendors already prefix the model with the brand ("Reolink Duo Floodlight PoE").
            (Some(m), Some(md)) if md.to_lowercase().starts_with(&m.to_lowercase()) => md.clone(),
            (Some(m), Some(md)) => format!("{m} {md}"),
            (_, Some(md)) => md.clone(),
            _ => self.ip.clone(),
        }
    }

    /// RTSP URI of the best profile.
    pub fn stream_uri(&self) -> Option<&str> {
        self.best_profile()?.stream_uri.as_deref()
    }

    /// Highest-resolution profile that has an RTSP URI — what we want for reconstruction.
    pub fn best_profile(&self) -> Option<&Profile> {
        self.profiles
            .iter()
            .filter(|p| p.stream_uri.is_some())
            .max_by_key(|p| p.width.unwrap_or(0) * p.height.unwrap_or(0))
    }
}

pub struct Client {
    http: reqwest::blocking::Client,
    user: String,
    pass: String,
    clock_offset: ChronoDuration,
}

impl Client {
    pub fn new(user: &str, pass: &str) -> Result<Self> {
        let http = reqwest::blocking::Client::builder().timeout(Duration::from_secs(8)).build()?;
        Ok(Self { http, user: user.into(), pass: pass.into(), clock_offset: ChronoDuration::zero() })
    }

    fn envelope(&self, body: &str, auth: bool) -> String {
        let header = if auth { self.security_header() } else { String::new() };
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema"><s:Header>{header}</s:Header><s:Body>{body}</s:Body></s:Envelope>"#
        )
    }

    fn security_header(&self) -> String {
        let mut nonce = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut nonce);
        let created = (Utc::now() + self.clock_offset).format("%Y-%m-%dT%H:%M:%S%.3fZ").to_string();
        let mut h = Sha1::new();
        h.update(nonce);
        h.update(created.as_bytes());
        h.update(self.pass.as_bytes());
        let digest = B64.encode(h.finalize());
        format!(
            r#"<Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"><UsernameToken><Username>{}</Username><Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</Password><Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{}</Nonce><Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">{created}</Created></UsernameToken></Security>"#,
            xml_escape(&self.user),
            B64.encode(nonce),
        )
    }

    fn call(&self, url: &str, body: &str, auth: bool) -> Result<String> {
        let resp = self
            .http
            .post(url)
            .header("Content-Type", "application/soap+xml; charset=utf-8")
            .body(self.envelope(body, auth))
            .send()
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            let reason = Document::parse(&text)
                .ok()
                .and_then(|d| text_of(d.root(), "Text").or_else(|| text_of(d.root(), "Value")))
                .unwrap_or_else(|| text.chars().take(200).collect());
            // Cameras signal bad credentials as HTTP 401, or HTTP 400/500 with a ter:NotAuthorized fault.
            if status.as_u16() == 401 || text.contains("NotAuthorized") {
                bail!("authentication rejected by {url} (check username/password)");
            }
            bail!("{url} returned {status}: {reason}");
        }
        Ok(text)
    }

    /// Align our WS-Security timestamps to the camera clock (GetSystemDateAndTime needs no auth).
    fn sync_clock(&mut self, device_url: &str) {
        let Ok(xml) = self.call(device_url, "<tds:GetSystemDateAndTime/>", false) else { return };
        let Ok(doc) = Document::parse(&xml) else { return };
        let Some(utc) = find(doc.root(), "UTCDateTime") else { return };
        let num = |n: &str| text_of(utc, n).and_then(|t| t.parse::<u32>().ok());
        let (Some(y), Some(mo), Some(d), Some(h), Some(mi), Some(s)) =
            (num("Year"), num("Month"), num("Day"), num("Hour"), num("Minute"), num("Second"))
        else {
            return;
        };
        if let Some(cam) = NaiveDate::from_ymd_opt(y as i32, mo, d).and_then(|dt| dt.and_hms_opt(h, mi, s)) {
            let cam: DateTime<Utc> = DateTime::from_naive_utc_and_offset(cam, Utc);
            self.clock_offset = cam - Utc::now();
        }
    }

    pub fn probe(&mut self, ip: &str, device_url: &str) -> Result<CameraInfo> {
        self.sync_clock(device_url);

        let info = self.call(device_url, "<tds:GetDeviceInformation/>", true)?;
        let info = Document::parse(&info)?;
        let r = info.root();

        let caps = self.call(device_url, "<tds:GetCapabilities><tds:Category>All</tds:Category></tds:GetCapabilities>", true)?;
        let caps = Document::parse(&caps)?;
        let media_url = find(caps.root(), "Media")
            .and_then(|m| text_of(m, "XAddr"))
            .ok_or_else(|| anyhow!("camera reports no Media service"))?;
        let ptz = find(caps.root(), "PTZ").and_then(|m| text_of(m, "XAddr")).is_some();
        // Cameras behind NAT/port-forwarding often report internal addresses; trust the IP we reached.
        let media_url = rehost(&media_url, device_url);

        let profiles_xml = self.call(&media_url, "<trt:GetProfiles/>", true)?;
        let profiles_doc = Document::parse(&profiles_xml)?;
        let mut profiles = Vec::new();
        for p in profiles_doc.descendants().filter(|n| n.tag_name().name() == "Profiles") {
            let Some(token) = p.attribute("token") else { continue };
            let vec = find(p, "VideoEncoderConfiguration");
            let num = |n: &str| vec.and_then(|v| text_of(v, n)).and_then(|t| t.parse().ok());
            let mut prof = Profile {
                token: token.into(),
                name: child_text(p, "Name").unwrap_or_else(|| token.into()),
                encoding: vec.and_then(|v| text_of(v, "Encoding")),
                width: num("Width"),
                height: num("Height"),
                stream_uri: None,
                snapshot_uri: None,
            };
            let t = xml_escape(token);
            let stream_req = format!(
                "<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream><tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup><trt:ProfileToken>{t}</trt:ProfileToken></trt:GetStreamUri>"
            );
            prof.stream_uri = self
                .call(&media_url, &stream_req, true)
                .ok()
                .and_then(|x| Document::parse(&x).ok().and_then(|d| text_of(d.root(), "Uri")))
                .map(|u| rehost(&u, device_url));
            let snap_req = format!("<trt:GetSnapshotUri><trt:ProfileToken>{t}</trt:ProfileToken></trt:GetSnapshotUri>");
            prof.snapshot_uri = self
                .call(&media_url, &snap_req, true)
                .ok()
                .and_then(|x| Document::parse(&x).ok().and_then(|d| text_of(d.root(), "Uri")))
                .map(|u| rehost(&u, device_url));
            profiles.push(prof);
        }

        // Name sources (all optional — plenty of cameras don't implement them).
        let hostname = self
            .call(device_url, "<tds:GetHostname/>", true)
            .ok()
            .and_then(|x| Document::parse(&x).ok().and_then(|d| text_of(d.root(), "Name")));
        let osd_texts = self
            .call(&media_url, "<trt:GetOSDs/>", true)
            .ok()
            .and_then(|x| {
                Document::parse(&x).ok().map(|d| {
                    d.descendants()
                        .filter(|n| n.tag_name().name() == "PlainText")
                        .filter_map(|n| n.text().map(|t| t.trim().to_string()))
                        .filter(|t| !t.is_empty())
                        .collect()
                })
            })
            .unwrap_or_default();

        let manufacturer = text_of(r, "Manufacturer");
        let vendor_name = match manufacturer.as_deref().map(str::to_lowercase) {
            Some(m) if m.contains("reolink") => {
                crate::vendor::reolink_device_name(&http_host(device_url), &self.user, &self.pass)
            }
            _ => None,
        };

        Ok(CameraInfo {
            kind: SourceKind::Onvif,
            name: None,
            pose: None,
            calibration: None,
            vendor_name,
            hostname,
            osd_texts,
            dns_name: crate::naming::reverse_dns(ip),
            ip: ip.into(),
            device_url: device_url.into(),
            manufacturer,
            model: text_of(r, "Model"),
            firmware: text_of(r, "FirmwareVersion"),
            serial: text_of(r, "SerialNumber"),
            ptz,
            profiles,
        })
    }
}

fn find<'a, 'i>(node: Node<'a, 'i>, local: &str) -> Option<Node<'a, 'i>> {
    node.descendants().find(|n| n.is_element() && n.tag_name().name() == local)
}

fn text_of(node: Node, local: &str) -> Option<String> {
    find(node, local).and_then(|n| n.text()).map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

fn child_text(node: Node, local: &str) -> Option<String> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == local)
        .and_then(|n| n.text())
        .map(|t| t.trim().to_string())
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&apos;")
}

/// Split "scheme://host[:port]/rest" → (scheme, host, port, rest).
fn split_url(u: &str) -> Option<(&str, &str, Option<&str>, &str)> {
    let (scheme, rest) = u.split_once("://")?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let authority = authority.rsplit('@').next().unwrap_or(authority);
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.contains(']') || h.ends_with(']') => (h, Some(p)),
        _ => (authority, None),
    };
    Some((scheme, host, port, path))
}

/// (host, port) of a URL, with the scheme's default port filled in. Used to match the same device
/// across discovery results and the saved inventory.
pub fn host_port(u: &str) -> Option<(String, u16)> {
    let (scheme, host, port, _) = split_url(u)?;
    let port = match port {
        Some(p) => p.parse().ok()?,
        None => match scheme {
            "https" => 443,
            "rtsp" => 554,
            "rtsps" => 322,
            _ => 80,
        },
    };
    Some((host.trim_matches(|c| c == '[' || c == ']').to_string(), port))
}

/// Host for the camera's web/API port (port 80) — ONVIF often runs on a different port.
fn http_host(device_url: &str) -> String {
    host_port(device_url).map(|(h, _)| h).unwrap_or_default()
}

/// Same device service, regardless of path or default-port spelling.
pub fn same_device(a: &str, b: &str) -> bool {
    matches!((host_port(a), host_port(b)), (Some(x), Some(y)) if x == y)
}

impl CameraInfo {
    /// Same physical ONVIF device as `device_url`. Stream cameras never match by address — one
    /// UniFi console serves every camera's stream from the same host and port.
    pub fn is_device(&self, device_url: &str) -> bool {
        self.kind == SourceKind::Onvif && same_device(&self.device_url, device_url)
    }
}

/// Replace the host in `uri` with the host from `reached` (keeping uri's scheme, port, path).
fn rehost(uri: &str, reached: &str) -> String {
    match (split_url(uri), split_url(reached)) {
        (Some((scheme, _, port, path)), Some((_, host, _, _))) => match port {
            Some(p) => format!("{scheme}://{host}:{p}{path}"),
            None => format!("{scheme}://{host}{path}"),
        },
        _ => uri.to_string(),
    }
}

/// Embed credentials into an RTSP URI (percent-encoding reserved characters).
pub fn with_credentials(uri: &str, user: &str, pass: &str) -> String {
    let enc = |s: &str| -> String {
        s.bytes()
            .map(|b| match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (b as char).to_string(),
                _ => format!("%{b:02X}"),
            })
            .collect()
    };
    match uri.split_once("://") {
        Some((scheme, rest)) if !user.is_empty() => {
            let rest = rest.split_once('@').map(|(_, r)| r).unwrap_or(rest);
            format!("{scheme}://{}:{}@{rest}", enc(user), enc(pass))
        }
        _ => uri.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rehost_keeps_port_and_path() {
        assert_eq!(
            rehost("rtsp://10.0.0.5:554/Streaming/Channels/101?transportmode=unicast", "http://192.168.1.20/onvif/device_service"),
            "rtsp://192.168.1.20:554/Streaming/Channels/101?transportmode=unicast"
        );
        assert_eq!(rehost("http://10.0.0.5/onvif/Media", "http://192.168.1.20:8080/x"), "http://192.168.1.20/onvif/Media");
    }

    #[test]
    fn device_matching_ignores_default_port_and_path() {
        assert!(same_device("http://10.0.0.5/onvif/device_service", "http://10.0.0.5:80/onvif/device_service"));
        assert!(!same_device("http://10.0.0.5:8000/x", "http://10.0.0.5/x"));
        assert_eq!(host_port("rtsp://u:p@cam.local/s"), Some(("cam.local".into(), 554)));
    }

    #[test]
    fn credentials_are_encoded() {
        assert_eq!(with_credentials("rtsp://cam:554/s", "admin", "p@ss:1"), "rtsp://admin:p%40ss%3A1@cam:554/s");
        assert_eq!(with_credentials("rtsp://old:x@cam/s", "a", "b"), "rtsp://a:b@cam/s");
    }
}
