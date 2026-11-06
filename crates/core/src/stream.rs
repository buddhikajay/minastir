//! Cameras added by a plain RTSP / RTSPS URL instead of ONVIF — e.g. UniFi Protect's
//! "Secure RTSPS Output" (rtsps://<console>:7441/<token>), NVR channel URLs, or cameras whose
//! ONVIF is missing or broken. We get no device info, so the stream itself is probed for its
//! codec and resolution.

use crate::onvif::{self, CameraInfo, Profile, SourceKind};
use anyhow::{bail, Context, Result};
use std::process::Command;

/// A stream URL split into the part we store and its embedded credentials (if any).
#[derive(Debug, PartialEq)]
pub struct StreamUrl {
    pub url: String,
    pub user: String,
    pub pass: String,
}

/// Validate and clean a pasted URL.
/// - only rtsp:// and rtsps:// are accepted
/// - `user:pass@` is pulled out so the password is stored with the other credentials (0600)
/// - UniFi's `?enableSrtp` is dropped: it asks for SRTP, which ffmpeg doesn't speak; the stream
///   is still encrypted by TLS on the RTSPS connection
pub fn parse(input: &str) -> Result<StreamUrl> {
    let input = input.trim();
    let Some((scheme, rest)) = input.split_once("://") else { bail!("Enter a full rtsp:// or rtsps:// URL") };
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "rtsp" && scheme != "rtsps" {
        bail!("Only rtsp:// and rtsps:// streams are supported");
    }
    let (authority, path) = rest.find('/').map(|i| rest.split_at(i)).unwrap_or((rest, ""));
    let (userinfo, host) = match authority.rsplit_once('@') {
        Some((u, h)) => (Some(u), h),
        None => (None, authority),
    };
    if host.is_empty() {
        bail!("The URL has no host");
    }
    let (user, pass) = match userinfo {
        Some(u) => {
            let (a, b) = u.split_once(':').unwrap_or((u, ""));
            (percent_decode(a), percent_decode(b))
        }
        None => (String::new(), String::new()),
    };
    let path = match path.split_once('?') {
        Some((p, q)) => {
            let kept: Vec<&str> = q.split('&').filter(|kv| !kv.is_empty() && !kv.eq_ignore_ascii_case("enableSrtp")).collect();
            if kept.is_empty() { p.to_string() } else { format!("{p}?{}", kept.join("&")) }
        }
        None => path.to_string(),
    };
    Ok(StreamUrl { url: format!("{scheme}://{host}{path}"), user, pass })
}

/// True for UniFi Protect's RTSP(S) restream ports.
pub fn is_unifi(url: &str) -> bool {
    matches!(onvif::host_port(url), Some((_, 7441 | 7447)))
}

#[derive(Debug, Default)]
pub struct StreamInfo {
    pub codec: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

/// Open the stream with ffprobe and read the first video stream's codec and size.
/// `source` includes credentials when needed.
pub fn probe(source: &str) -> Result<StreamInfo> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-rtsp_transport", "tcp", "-timeout", "10000000"])
        .args(["-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "json"])
        .arg(source)
        .output()
        .context("failed to run ffprobe — is ffmpeg installed and on PATH?")?;
    let err = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        let e = err.to_lowercase();
        if e.contains("401") || e.contains("unauthorized") {
            bail!("authentication rejected");
        }
        if e.contains("404") || e.contains("not found") || e.contains("end of file") || e.contains("invalid data") {
            bail!("The stream was refused — check the URL (UniFi links change if the stream is disabled and re-enabled)");
        }
        if e.contains("timed out") || e.contains("connection refused") || e.contains("no route") {
            bail!("Can't reach {}", crate::capture::redact(source));
        }
        bail!("Couldn't open the stream: {}", crate::capture::redact(err.trim()));
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).context("unexpected ffprobe output")?;
    let s = v.pointer("/streams/0").context("the URL has no video stream")?;
    let num = |k: &str| s.get(k).and_then(|x| x.as_u64()).map(|x| x as u32);
    Ok(StreamInfo {
        codec: s.get("codec_name").and_then(|c| c.as_str()).map(|c| c.to_uppercase()),
        width: num("width"),
        height: num("height"),
    })
}

/// Build an inventory entry for a probed stream.
pub fn camera(url: &str, info: StreamInfo) -> CameraInfo {
    let ip = onvif::host_port(url).map(|(h, _)| h).unwrap_or_default();
    let unifi = is_unifi(url);
    CameraInfo {
        kind: SourceKind::Stream,
        ip,
        device_url: url.to_string(),
        manufacturer: unifi.then(|| "Ubiquiti".to_string()),
        model: Some(if unifi { "UniFi Protect stream" } else { "RTSP stream" }.to_string()),
        firmware: None,
        serial: None,
        ptz: false,
        profiles: vec![Profile {
            token: "stream".into(),
            name: "Stream".into(),
            encoding: info.codec,
            width: info.width,
            height: info.height,
            stream_uri: Some(url.to_string()),
            snapshot_uri: None,
        }],
        name: None,
        pose: None,
        calibration: None,
        vendor_name: None,
        hostname: None,
        osd_texts: Vec::new(),
        dns_name: None,
    }
}

/// Stable short id for a stream URL (FNV-1a, so it never changes between builds). Several
/// streams can share one host — every UniFi camera is served from the console's IP.
pub fn stream_id(url: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in url.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("s{:012x}", h & 0xffff_ffff_ffff)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            let hex = std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unifi_link_is_cleaned() {
        let s = parse(" rtsps://192.168.1.1:7441/AbCdEfGhIjKlMnOp?enableSrtp ").unwrap();
        assert_eq!(s, StreamUrl { url: "rtsps://192.168.1.1:7441/AbCdEfGhIjKlMnOp".into(), user: "".into(), pass: "".into() });
        assert!(is_unifi(&s.url));
    }

    #[test]
    fn credentials_are_extracted() {
        let s = parse("rtsp://admin:p%40ss@10.0.0.5:554/h264?channel=1&enableSrtp").unwrap();
        assert_eq!(s.url, "rtsp://10.0.0.5:554/h264?channel=1");
        assert_eq!((s.user.as_str(), s.pass.as_str()), ("admin", "p@ss"));
        assert!(!is_unifi(&s.url));
    }

    #[test]
    fn rejects_other_schemes() {
        assert!(parse("http://10.0.0.5/stream").is_err());
        assert!(parse("10.0.0.5").is_err());
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let a = stream_id("rtsps://192.168.1.1:7441/aaa");
        assert_eq!(a, stream_id("rtsps://192.168.1.1:7441/aaa"));
        assert_ne!(a, stream_id("rtsps://192.168.1.1:7441/bbb"));
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric()));
    }
}
