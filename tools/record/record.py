#!/usr/bin/env python3
"""Record every camera's main stream to disk (no re-encoding), in 10-minute files.

  python3 tools/record/record.py --data data --out data/recordings --hours 12 \
      --earliest 05:30 --latest 07:00

Waits until `--earliest`, then starts once most cameras have left night mode (their picture has
colour again; infrared night mode is grey) or at `--latest`, whichever comes first. `--now`
starts immediately. Each camera is recorded by its own ffmpeg, restarted if the stream drops; a
camera whose main stream keeps failing falls back to its sub-stream for ten minutes.

Camera list and credentials come from the Minastir data dir (cameras.json, credentials.json).
Credentials are never written to the log.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse

SEGMENT_S = 600


def log(msg):
    print(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}", flush=True)


def redact(s):
    return re.sub(r"(rtsps?://)[^@/\s]*@", r"\1***@", s)


def with_creds(uri, user, pw):
    if not user:
        return uri
    scheme, rest = uri.split("://", 1)
    return f"{scheme}://{urllib.parse.quote(user, safe='')}:{urllib.parse.quote(pw, safe='')}@{rest}"


def cameras(data):
    cams = json.load(open(os.path.join(data, "cameras.json")))
    creds = json.load(open(os.path.join(data, "credentials.json")))
    out = []
    for c in cams:
        profiles = [p for p in c.get("profiles", []) if p.get("stream_uri")]
        if not profiles:
            continue
        cid = cam_id(c)
        cr = creds.get(cid, {})
        by_width = sorted(profiles, key=lambda p: p.get("width") or 0)
        main, sub = by_width[-1], by_width[0]
        out.append({
            "id": cid,
            "name": c.get("name") or cid,
            "main": with_creds(main["stream_uri"], cr.get("user", ""), cr.get("pass", "")),
            "sub": with_creds(sub["stream_uri"], cr.get("user", ""), cr.get("pass", "")),
        })
    return out


def cam_id(c):
    """The server's camera id (CameraInfo::id): FNV-1a of the URL for plain streams, else the
    serial's alphanumerics, else the IP with dots as underscores."""
    if c.get("kind") == "stream":
        h = 0xCBF29CE484222325
        for byte in c["device_url"].encode():
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"s{h & 0xFFFFFFFFFFFF:012x}"
    serial = c.get("serial") or ""
    if serial:
        return "".join(ch for ch in serial if ch.isascii() and ch.isalnum())
    return c["ip"].replace(".", "_")


def ffmpeg_input(url):
    args = ["-rtsp_transport", "tcp", "-timeout", "10000000"] if url.startswith("rtsp") else []
    return args + ["-i", url]


def colourfulness(url):
    """Mean saturation (0–1) of one small frame, or None if the camera didn't answer."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *ffmpeg_input(url), "-frames:v", "1",
           "-vf", "scale=96:54", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return None
    if len(raw) < 96 * 54 * 3:
        return None
    total = 0.0
    for i in range(0, len(raw), 3):
        r, g, b = raw[i], raw[i + 1], raw[i + 2]
        mx, mn = max(r, g, b), min(r, g, b)
        total += (mx - mn) / mx if mx > 0 else 0.0
    return total / (len(raw) / 3)


def wait_for_daylight(cams, earliest, latest):
    now = dt.datetime.now()
    start = now.replace(hour=earliest[0], minute=earliest[1], second=0, microsecond=0)
    deadline = now.replace(hour=latest[0], minute=latest[1], second=0, microsecond=0)
    if deadline <= now:  # times are for tomorrow morning
        start += dt.timedelta(days=1)
        deadline += dt.timedelta(days=1)
    if start > now:
        log(f"waiting until {start:%a %H:%M} to check for daylight")
        time.sleep((start - now).total_seconds())
    while True:
        sats = {c["name"]: colourfulness(c["sub"]) for c in cams}
        answered = [s for s in sats.values() if s is not None]
        colour = [s for s in answered if s > 0.06]
        log("colour check: " + ", ".join(f"{n} {'—' if s is None else f'{s:.3f}'}" for n, s in sats.items()))
        if answered and len(colour) * 2 > len(answered):
            log("most cameras are in colour: starting")
            return
        if dt.datetime.now() >= deadline:
            log("latest start time reached: starting")
            return
        time.sleep(120)


def record(cam, out, until):
    folder = os.path.join(out, cam["id"])
    os.makedirs(folder, exist_ok=True)
    failures = []  # times main-stream runs ended within a minute of starting
    sub_until = 0.0
    while time.time() < until:
        use_sub = time.time() < sub_until
        url = cam["sub"] if use_sub else cam["main"]
        remaining = int(until - time.time())
        if remaining <= 5:
            break
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *ffmpeg_input(url),
               "-map", "0", "-c", "copy", "-t", str(remaining),
               "-f", "segment", "-segment_time", str(SEGMENT_S), "-reset_timestamps", "1", "-strftime", "1",
               os.path.join(folder, ("sub-" if use_sub else "") + "%Y%m%d-%H%M%S.mp4")]
        started = time.time()
        log(f"{cam['name']}: recording {'sub' if use_sub else 'main'} stream")
        p = subprocess.run(cmd, capture_output=True, text=True)
        ran = time.time() - started
        err = redact((p.stderr or "").strip().splitlines()[-1] if p.stderr and p.stderr.strip() else f"exit {p.returncode}")
        if time.time() >= until - 5:
            break
        log(f"{cam['name']}: stream ended after {ran:.0f} s ({err}); reconnecting")
        if not use_sub and ran < 60:
            failures = [t for t in failures if time.time() - t < 300] + [time.time()]
            if len(failures) >= 3:
                log(f"{cam['name']}: main stream keeps failing; sub-stream for 10 minutes")
                sub_until = time.time() + 600
                failures = []
        time.sleep(5)
    log(f"{cam['name']}: done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/recordings")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--earliest", default="05:30")
    ap.add_argument("--latest", default="07:00")
    ap.add_argument("--now", action="store_true")
    a = ap.parse_args()
    cams = cameras(a.data)
    if not cams:
        sys.exit("no cameras with streams in cameras.json")
    log("cameras: " + ", ".join(f"{c['name']} ({c['id']})" for c in cams))
    if not a.now:
        hm = lambda s: tuple(int(x) for x in s.split(":"))
        wait_for_daylight(cams, hm(a.earliest), hm(a.latest))
    begin = dt.datetime.now()
    out = os.path.join(a.out, f"{begin:%Y%m%d-%H%M}")
    until = time.time() + a.hours * 3600
    log(f"recording {a.hours:g} h into {out}, until {dt.datetime.fromtimestamp(until):%a %H:%M}")
    threads = [threading.Thread(target=record, args=(c, out, until), daemon=True) for c in cams]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("all cameras done")


if __name__ == "__main__":
    main()
