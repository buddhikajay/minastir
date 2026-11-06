#!/usr/bin/env python3
"""Turn a recording (tools/record/record.py output) into a synchronised replay for the live engine.

  python3 tools/record/recording_replay.py data/recordings/<start> [--cameras id,id] [--name NAME]

Each camera's 10-minute segments (<camera id>/<YYYYmmdd-HHMMSS>.mp4, main stream; sub-stream
fall-backs are skipped) are joined without re-encoding, then all cameras are trimmed to the moment
the last one started, so frame times line up. Writes data/replay/<name>/<camera id>.mp4 and a
replay.json; play with MINASTIR_REPLAY=data/replay/<name>.
"""
import argparse
import datetime as dt
import json
import os
import subprocess


def start_of(path):
    return dt.datetime.strptime(os.path.basename(path)[:15], "%Y%m%d-%H%M%S")


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return float(out.strip() or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--cameras", help="comma-separated camera ids (default: all)")
    ap.add_argument("--name")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    rec = a.recording.rstrip("/")
    cams = sorted(d for d in os.listdir(rec) if os.path.isdir(os.path.join(rec, d)))
    if a.cameras:
        cams = [c for c in cams if c in a.cameras.split(",")]
    segs = {}
    for c in cams:
        files = sorted(os.path.join(rec, c, f) for f in os.listdir(os.path.join(rec, c)) if f.endswith(".mp4") and not f.startswith("sub-"))
        files = [f for f in files if duration(f) > 1.0]
        if files:
            segs[c] = files
    if not segs:
        raise SystemExit("no segments found")
    common = max(start_of(f[0]) for f in segs.values())
    name = a.name or os.path.basename(rec)
    out = os.path.join(a.data, "replay", name)
    os.makedirs(out, exist_ok=True)
    info = {"recording": rec, "start": common.isoformat(), "cameras": {}}
    for c, files in segs.items():
        lst = os.path.join(out, f"{c}.txt")
        with open(lst, "w") as f:
            for p in files:
                f.write(f"file '{os.path.abspath(p)}'\n")
        skip = (common - start_of(files[0])).total_seconds()
        dest = os.path.join(out, f"{c}.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-ss", f"{skip:.2f}", "-map", "0:v:0", "-c", "copy", dest], check=True)
        os.remove(lst)
        info["cameras"][c] = {"segments": len(files), "trimmed_s": skip, "duration_s": duration(dest)}
        print(f"{c}: {len(files)} segments, trimmed {skip:.0f} s → {info['cameras'][c]['duration_s'] / 60:.1f} min")
    json.dump(info, open(os.path.join(out, "replay.json"), "w"), indent=2)
    print(f"replay ready: MINASTIR_REPLAY={out}")


if __name__ == "__main__":
    main()
