//! ONVIF WS-Discovery: multicast a Probe for NetworkVideoTransmitters and collect replies.

use anyhow::Result;
use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

const MULTICAST: SocketAddrV4 = SocketAddrV4::new(Ipv4Addr::new(239, 255, 255, 250), 3702);

#[derive(Debug, Clone)]
pub struct Discovered {
    /// WS-Addressing endpoint reference (usually urn:uuid:...) — unique per device, even when
    /// several devices sit behind one IP (NVR channels, port-forwarded cameras).
    pub endpoint: String,
    /// Host of the device service URL (falls back to the responding address).
    pub ip: String,
    /// Device service endpoints (a camera may advertise several, e.g. IPv4 + IPv6).
    pub xaddrs: Vec<String>,
    pub scopes: Vec<String>,
}

impl Discovered {
    /// Prefer an IPv4 endpoint (cameras often also advertise IPv6 / link-local ones).
    pub fn device_url(&self) -> Option<&str> {
        self.xaddrs
            .iter()
            .find(|x| !x.contains('['))
            .or_else(|| self.xaddrs.first())
            .map(String::as_str)
    }

    /// Human hints from onvif://www.onvif.org/name/... and /hardware/... scopes.
    pub fn scope_value(&self, key: &str) -> Option<String> {
        let needle = format!("/{key}/");
        self.scopes.iter().find_map(|s| {
            s.find(&needle).map(|i| s[i + needle.len()..].replace("%20", " "))
        })
    }
}

fn probe_message() -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl">
<e:Header><w:MessageID>uuid:{}</w:MessageID><w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To><w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header>
<e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>"#,
        uuid::Uuid::new_v4()
    )
}

pub fn discover(timeout: Duration) -> Result<Vec<Discovered>> {
    let sock = UdpSocket::bind("0.0.0.0:0")?;
    sock.set_multicast_ttl_v4(2)?;
    sock.set_read_timeout(Some(Duration::from_millis(300)))?;

    let msg = probe_message();
    // UDP is lossy; send the probe a few times.
    for _ in 0..3 {
        sock.send_to(msg.as_bytes(), MULTICAST)?;
    }

    let mut found: BTreeMap<String, Discovered> = BTreeMap::new();
    let mut buf = vec![0u8; 65535];
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let (n, addr) = match sock.recv_from(&mut buf) {
            Ok(r) => r,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
            Err(e) => return Err(e.into()),
        };
        let text = String::from_utf8_lossy(&buf[..n]);
        let Ok(doc) = roxmltree::Document::parse(&text) else { continue };
        let words = |name: &str| -> Vec<String> {
            doc.descendants()
                .find(|n| n.tag_name().name() == name)
                .and_then(|n| n.text())
                .map(|t| t.split_whitespace().map(String::from).collect())
                .unwrap_or_default()
        };
        let endpoint = doc
            .descendants()
            .find(|n| n.tag_name().name() == "EndpointReference")
            .and_then(|n| n.descendants().find(|c| c.tag_name().name() == "Address"))
            .and_then(|n| n.text())
            .map(|t| t.trim().to_string())
            .unwrap_or_default();
        let mut d = Discovered { endpoint, ip: addr.ip().to_string(), xaddrs: words("XAddrs"), scopes: words("Scopes") };
        if let Some((host, _)) = d.device_url().and_then(crate::onvif::host_port) {
            d.ip = host;
        }
        if d.endpoint.is_empty() {
            d.endpoint = d.device_url().unwrap_or(&d.ip).to_string();
        }
        found.entry(d.endpoint.clone()).or_insert(d);
    }
    Ok(found.into_values().collect())
}
