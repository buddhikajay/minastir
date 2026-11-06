#!/usr/bin/env python3
"""Turn clips exported from UniFi Protect into a synchronised replay for the live engine.

  python3 tools/record/clips.py --src ~/Downloads --data data \
      --camera "G5 Bullet=seb30ed761e56" --camera "G5 Turret Ultra=sbccf4ccf9469"

Export names look like "G5 Bullet 11-5-2026, 5.48.20pm GMT+11 - 11-5-2026, 5.49.47pm GMT+11.mp4"
(start and end to the second). The script:
  1. copies each camera's clips into data/clips/<camera id>/<start>.mp4, dropping duplicates and
     clips wholly inside another;
  2. finds the stretches every camera covers at once (each stretch inside one clip per camera);
  3. cuts those stretches out of every camera's clips and joins them, in time order, into
     data/replay/<name>/<camera id>.mp4 — equal length per camera, frame-for-frame in step —
     plus a replay.json listing the stretches.
Run the server with MINASTIR_REPLAY=data/replay/<name> to play them as if live.
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess

NAME = re.compile(
    r"^(?P<cam>.+?) (?P<d0>\d+-\d+-\d{4}), (?P<t0>\d+\.\d+\.\d+[ap]m) GMT(?P<tz>[+-]\d+) - "
    r"(?P<d1>\d+-\d+-\d{4}), (?P<t1>\d+\.\d+\.\d+[ap]m) GMT[+-]\d+(?: \(\d+\))?\.mp4$"
)
MIN_STRETCH = 8.0  # seconds


def parse_time(d, t):
    return dt.datetime.strptime(f"{d} {t}", "%m-%d-%Y %I.%M.%S%p")


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--camera", action="append", required=True, help='"<export name>=<camera id>"')
    ap.add_argument("--name", help="replay name (default: date and first start time)")
    a = ap.parse_args()
    names = dict(c.split("=", 1) for c in a.camera)

    # 1. Collect clips per camera, without duplicates or clips inside others.
    clips = {cid: [] for cid in names.values()}
    for f in sorted(os.listdir(os.path.expanduser(a.src))):
        m = NAME.match(f)
        if not m or m["cam"] not in names:
            continue
        start, end = parse_time(m["d0"], m["t0"]), parse_time(m["d1"], m["t1"])
        path = os.path.join(os.path.expanduser(a.src), f)
        clips[names[m["cam"]]].append({"start": start, "end": end, "path": path, "len": duration(path)})
    for cid, cs in clips.items():
        cs.sort(key=lambda c: (c["start"], -(c["end"] - c["start"]).total_seconds()))
        kept = []
        for c in cs:
            if any(k["start"] <= c["start"] and c["end"] <= k["end"] for k in kept):
                continue
            kept.append(c)
        clips[cid] = kept
        folder = os.path.join(a.data, "clips", cid)
        os.makedirs(folder, exist_ok=True)
        for c in kept:
            dest = os.path.join(folder, f"{c['start']:%Y%m%d-%H%M%S}.mp4")
            if not os.path.exists(dest):
                shutil.copy2(c["path"], dest)
            c["file"] = dest
        print(f"{cid}: {len(kept)} clips, " + ", ".join(f"{c['start']:%H:%M:%S}–{c['end']:%H:%M:%S}" for c in kept))

    # 2. Stretches every camera covers, each within a single clip per camera. Greedy in time:
    # from the current moment take, per camera, the clip covering it that runs longest.
    cams = list(clips)
    t = min(c["start"] for cs in clips.values() for c in cs)
    t_end = max(c["end"] for cs in clips.values() for c in cs)
    stretches = []
    while t < t_end:
        chosen = {}
        for cid in cams:
            cover = [c for c in clips[cid] if c["start"] <= t < c["end"]]
            if cover:
                chosen[cid] = max(cover, key=lambda c: c["end"])
        if len(chosen) == len(cams):
            end = min(c["end"] for c in chosen.values())
            if (end - t).total_seconds() >= MIN_STRETCH:
                stretches.append((t, end, chosen))
            t = end
        else:
            nxt = [c["start"] for cs in clips.values() for c in cs if c["start"] > t]
            if not nxt:
                break
            t = min(nxt)
    if not stretches:
        print("no stretch is covered by every camera")
        return
    total = sum((e - s).total_seconds() for s, e, _ in stretches)
    print(f"{len(stretches)} shared stretches, {total:.0f} s: " + ", ".join(f"{s:%H:%M:%S}–{e:%H:%M:%S}" for s, e, _ in stretches))

    # 3. Cut and join per camera (re-encoded so cuts land on exact frames).
    name = a.name or f"{stretches[0][0]:%Y%m%d-%H%M%S}"
    out = os.path.join(a.data, "replay", name)
    os.makedirs(out, exist_ok=True)
    for cid in cams:
        inputs, parts = [], []
        for k, (s, e, chosen) in enumerate(stretches):
            c = chosen[cid]
            off = (s - c["start"]).total_seconds()
            inputs += ["-ss", f"{off:.3f}", "-t", f"{(e - s).total_seconds():.3f}", "-i", c["file"]]
            # Exactly the stretch's length (a clip may end a little before its name says: hold its
            # last frame) so every camera stays in step across stretches.
            length = (e - s).total_seconds()
            parts.append(f"[{k}:v:0]setpts=PTS-STARTPTS,fps=15,tpad=stop_mode=clone:stop_duration=3,trim=duration={length:.3f},setpts=PTS-STARTPTS[v{k}]")
        graph = ";".join(parts) + ";" + "".join(f"[v{k}]" for k in range(len(stretches))) + f"concat=n={len(stretches)}:v=1:a=0[out]"
        dest = os.path.join(out, f"{cid}.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex", graph,
                        "-map", "[out]", "-c:v", "h264_videotoolbox", "-b:v", "16M", dest], check=True)
        print(f"  {dest}: {duration(dest):.1f} s")
    json.dump({"cameras": cams, "stretches": [{"start": s.isoformat(), "end": e.isoformat()} for s, e, _ in stretches]},
              open(os.path.join(out, "replay.json"), "w"), indent=2)
    print(f"replay ready: MINASTIR_REPLAY={out}")


if __name__ == "__main__":
    main()
