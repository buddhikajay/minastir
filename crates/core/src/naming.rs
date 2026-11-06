//! Finding a human name for a camera.
//!
//! Cameras expose names in several places — OSD overlay text, ONVIF hostname, the DHCP hostname
//! the router publishes in reverse DNS, WS-Discovery scopes — but most of them hold factory
//! defaults ("IPC-BO", "C100_DD2DBA", "Reolink"). We collect candidates and keep the first one a
//! person actually chose.

use std::net::IpAddr;

/// Brand / device-class words that never identify a specific camera on their own.
const GENERIC: &[&str] = &[
    "ipc", "ipcam", "ipcamera", "camera", "cam", "webcam", "nvr", "dvr", "onvif", "localhost", "device",
    "reolink", "tapo", "tplink", "hikvision", "hikvisiondigital", "dahua", "amcrest", "axis", "uniview", "foscam",
    "annke", "lorex", "wyze", "eufy", "imou", "ezviz", "ubiquiti", "unifi",
];

fn norm(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

/// True if `name` is a factory default rather than something a person picked.
/// `hints` are the device's manufacturer / model / hardware strings.
pub fn is_generic(name: &str, hints: &[&str]) -> bool {
    let n = name.trim();
    let ln = norm(n);
    if ln.len() < 2 || GENERIC.contains(&ln.as_str()) {
        return true;
    }
    let lower = n.to_lowercase();
    // Reolink's "IPC-BO", Hikvision's "IPCamera 01"…
    if lower.starts_with("ipc-") || lower.starts_with("ipc_") || lower.starts_with("ipcamera") {
        return true;
    }
    // Model + MAC/serial suffix: "C100_DD2DBA", "Camera-3F9A21".
    if let Some((_, tail)) = n.rsplit_once(['_', '-']) {
        if tail.len() >= 6 && tail.chars().all(|c| c.is_ascii_hexdigit()) {
            return true;
        }
    }
    for h in hints {
        let hn = norm(h);
        if hn.len() < 2 {
            continue;
        }
        // Exactly the model/brand, or the model plus only brand words ("Tapo C100").
        if ln == hn || hn.contains(&ln) {
            return true;
        }
        if let Some(rest) = ln.strip_suffix(&hn).or_else(|| ln.strip_prefix(&hn)) {
            if rest.is_empty() || GENERIC.contains(&rest) || hints.iter().any(|o| norm(o) == rest) {
                return true;
            }
        }
    }
    false
}

/// First non-generic candidate, trimmed.
pub fn pick<'a>(candidates: impl IntoIterator<Item = &'a str>, hints: &[&str]) -> Option<String> {
    candidates
        .into_iter()
        .map(str::trim)
        .find(|c| !c.is_empty() && !is_generic(c, hints))
        .map(String::from)
}

/// Hostname the network knows this IP by (usually the camera's DHCP hostname, published by the
/// router), reduced to its first label: "Front.localdomain." → "Front".
pub fn reverse_dns(ip: &str) -> Option<String> {
    let addr: IpAddr = ip.parse().ok()?;
    let host = dns_lookup::lookup_addr(&addr).ok()?;
    let first = host.trim_end_matches('.').split('.').next()?.to_string();
    // The resolver echoes the IP back when there's no PTR record.
    if first.is_empty() || first.parse::<u8>().is_ok() || host == ip {
        return None;
    }
    // macOS' resolver lowercases cached names ("front"); restore a readable capital.
    Some(if first.chars().any(|c| c.is_uppercase()) { first } else { capitalize(&first) })
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_defaults_are_generic() {
        let reolink = ["REOLINK", "Reolink Duo Floodlight PoE", "ReolinkDuoFloodlightPoE"];
        for n in ["IPC-BO", "Reolink", "Reolink Duo Floodlight PoE", "REOLINK"] {
            assert!(is_generic(n, &reolink), "{n}");
        }
        let tapo = ["tp-link", "C100"];
        for n in ["C100_DD2DBA", "Tapo C100", "C100", "Tapo"] {
            assert!(is_generic(n, &tapo), "{n}");
        }
    }

    #[test]
    fn chosen_names_are_kept() {
        let hints = ["REOLINK", "Reolink Duo Floodlight PoE"];
        for n in ["Front", "Back Yard", "Garage-2024", "Driveway C100"] {
            assert!(!is_generic(n, &hints), "{n}");
        }
        assert_eq!(pick(["IPC-BO", " Back Yard "], &hints).as_deref(), Some("Back Yard"));
    }
}
