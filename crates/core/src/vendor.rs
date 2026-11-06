//! Vendor-specific extras that ONVIF doesn't cover — currently the device name people set in the
//! vendor's app, which is the most reliable "what is this camera called" source.

use serde_json::{json, Value};
use std::time::Duration;

/// Device name from Reolink's local HTTP API (`GetDevName`), using the same account as ONVIF.
/// Uses a short-lived token session and logs out afterwards — Reolink cameras allow only a
/// handful of concurrent sessions.
pub fn reolink_device_name(host: &str, user: &str, pass: &str) -> Option<String> {
    let http = reqwest::blocking::Client::builder().timeout(Duration::from_secs(5)).build().ok()?;
    let base = format!("http://{host}/api.cgi");
    let post = |query: &str, body: Value| -> Option<Value> {
        let resp = http.post(format!("{base}?{query}")).json(&body).send().ok()?;
        let v: Value = resp.json().ok()?;
        let first = v.get(0)?.clone();
        (first.get("code").and_then(Value::as_i64) == Some(0)).then_some(first)
    };

    let login = post(
        "cmd=Login",
        json!([{ "cmd": "Login", "param": { "User": { "Version": "0", "userName": user, "password": pass } } }]),
    )?;
    let token = login.pointer("/value/Token/name")?.as_str()?.to_string();

    let name = post(
        &format!("cmd=GetDevName&token={token}"),
        json!([{ "cmd": "GetDevName", "action": 0, "param": { "channel": 0 } }]),
    )
    .and_then(|v| v.pointer("/value/DevName/name").and_then(Value::as_str).map(str::to_string));

    let _ = post(&format!("cmd=Logout&token={token}"), json!([{ "cmd": "Logout", "param": {} }]));
    name.map(|n| n.trim().to_string()).filter(|n| !n.is_empty())
}
